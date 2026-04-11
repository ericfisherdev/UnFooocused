"""Anisotropic filtering for diffusion sharpness control.

Implements adaptive anisotropic bilateral blur used to apply sharpness
during the diffusion sampling process. The filter smooths noise predictions
while preserving edges, controlled by the sharpness parameter.

No domain logic — pure signal processing infrastructure.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import pad

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """Unpack a 2D kernel size to (ky, kx) tuple."""
    if isinstance(kernel_size, int):
        return kernel_size, kernel_size
    assert len(kernel_size) == 2, "2D Kernel size should have a length of 2."
    return int(kernel_size[0]), int(kernel_size[1])


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """Compute zero-padding for a given kernel size."""
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _gaussian(window_size: int, sigma: Tensor) -> Tensor:
    """Compute 1D Gaussian kernel for a batch of sigma values."""
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Compute 2D Gaussian kernel from scalar sigma."""
    sigma_tensor = torch.tensor([[sigma, sigma]], device=device, dtype=dtype)
    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma_tensor[:, 0, None], sigma_tensor[:, 1, None]
    kernel_y = _gaussian(ksize_y, sigma_y)[..., None]
    kernel_x = _gaussian(ksize_x, sigma_x)[..., None]
    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def _bilateral_blur(
    input_tensor: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: float,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    """Core bilateral blur implementation.

    Args:
        input_tensor: Input tensor (B, C, H, W).
        guidance: Optional guidance tensor for joint bilateral blur.
        kernel_size: Blur kernel size.
        sigma_color: Color sigma for bilateral weighting.
        sigma_space: Spatial sigma for Gaussian weighting.
        border_type: Padding mode.
        color_distance_type: Distance metric ('l1' or 'l2').

    Returns:
        Blurred tensor with same shape as input.
    """
    if isinstance(sigma_color, Tensor):
        sigma_color = sigma_color.to(device=input_tensor.device, dtype=input_tensor.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input_tensor, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)

    if guidance is None:
        guidance = input_tensor
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only accepts l1 or l2")

    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()

    space_kernel = _get_gaussian_kernel2d(
        kernel_size, sigma_space, device=input_tensor.device, dtype=input_tensor.dtype
    )
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    return (unfolded_input * kernel).sum(-1) / (kernel.sum(-1) + 1e-8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bilateral_blur(
    input_tensor: Tensor,
    kernel_size: tuple[int, int] | int = (13, 13),
    sigma_color: float | Tensor = 3.0,
    sigma_space: float = 3.0,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    """Apply bilateral blur to input tensor.

    Args:
        input_tensor: Input tensor (B, C, H, W).
        kernel_size: Blur kernel size.
        sigma_color: Color sigma for bilateral weighting.
        sigma_space: Spatial sigma for Gaussian weighting.
        border_type: Padding mode.
        color_distance_type: Distance metric ('l1' or 'l2').

    Returns:
        Blurred tensor with same shape as input.
    """
    return _bilateral_blur(input_tensor, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def adaptive_anisotropic_filter(x: Tensor, g: Tensor | None = None) -> Tensor:
    """Apply adaptive anisotropic bilateral filter for sharpness control.

    Normalizes the guidance signal by its standard deviation, then applies
    a bilateral blur. Used during diffusion sampling to control noise
    structure based on the sharpness parameter.

    Args:
        x: Input tensor (B, C, H, W) — typically the positive eps.
        g: Optional guidance tensor. Defaults to x if None.

    Returns:
        Filtered tensor with same shape as x.
    """
    if g is None:
        g = x
    s, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)
    s = s + 1e-5
    guidance = (g - m) / s
    return _bilateral_blur(
        x,
        guidance,
        kernel_size=(13, 13),
        sigma_color=3.0,
        sigma_space=3.0,
        border_type="reflect",
        color_distance_type="l1",
    )
