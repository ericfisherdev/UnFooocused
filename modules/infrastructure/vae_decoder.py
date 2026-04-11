"""LdmVAEDecoder and LdmLatentPreviewer — concrete VAE infrastructure adapters.

Bridges the domain VAEDecoder and LatentPreviewer protocols to ldm_patched's
VAEDecode/VAEDecodeTiled nodes and the VAEApprox lightweight preview model.

All ldm_patched and torch dependencies are injected via constructor
callables or objects, keeping this module testable with fakes.

Domain errors raised:
    RuntimeError — VAE model unavailable or GPU memory exhausted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from modules.domain.protocols import LatentTensor, StableDiffusionModel
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default tiled decoding threshold (pixels on longest side)
# ---------------------------------------------------------------------------

_DEFAULT_TILED_THRESHOLD: int = 2048
_DEFAULT_TILE_SIZE: int = 512
_LATENT_SCALE_FACTOR: int = 8


# ---------------------------------------------------------------------------
# Type aliases for injected decode operations
# ---------------------------------------------------------------------------

VAEDecodeOp = Any
"""Object with .decode(samples=, vae=) -> (image_batch_tensor,)."""

VAEDecodeTiledOp = Any
"""Object with .decode(samples=, vae=, tile_size=) -> (image_batch_tensor,)."""

RearrangeFn = Callable[..., Any]
"""Callable that rearranges tensor from (B, C, H, W) to (B, H, W, C) and extracts first sample."""


# ---------------------------------------------------------------------------
# LdmVAEDecoder — concrete VAEDecoder adapter
# ---------------------------------------------------------------------------


class LdmVAEDecoder:
    """Concrete VAEDecoder adapter backed by ldm_patched VAEDecode nodes.

    Automatically selects tiled decoding for large images to avoid OOM.

    Dependencies injected via constructor enable unit testing with fakes.

    Args:
        vae_decode_op: Object wrapping ldm_patched's VAEDecode node.
        vae_decode_tiled_op: Object wrapping ldm_patched's VAEDecodeTiled node.
        tiled_threshold: Pixel threshold above which tiled decoding is used.
            Images where either dimension exceeds this use tiled decoding.
    """

    __slots__ = ("_tiled_threshold", "_vae_decode_op", "_vae_decode_tiled_op")

    def __init__(
        self,
        *,
        vae_decode_op: VAEDecodeOp,
        vae_decode_tiled_op: VAEDecodeTiledOp,
        tiled_threshold: int = _DEFAULT_TILED_THRESHOLD,
    ) -> None:
        self._vae_decode_op = vae_decode_op
        self._vae_decode_tiled_op = vae_decode_tiled_op
        self._tiled_threshold = tiled_threshold

    def __repr__(self) -> str:
        return f"LdmVAEDecoder(tiled_threshold={self._tiled_threshold})"

    def decode(self, vae: StableDiffusionModel, latent: LatentTensor) -> list[NDArray[Any]]:
        """Decode a latent tensor into pixel-space images.

        Automatically uses tiled decoding when the implied pixel dimensions
        exceed the tiled threshold. Converts the VAE output from float [0, 1]
        tensors to uint8 [0, 255] numpy arrays.

        Args:
            vae: The VAE model (or full SD model bundle containing VAE).
            latent: Latent tensor dict with 'samples' key.

        Returns:
            List of numpy arrays, each (H, W, 3) uint8.

        Raises:
            RuntimeError: If the VAE is unavailable or GPU memory is exhausted.
        """
        use_tiled = _should_use_tiled(latent, self._tiled_threshold)

        if use_tiled:
            logger.debug("Using tiled VAE decode (threshold=%d)", self._tiled_threshold)
            image_batch = self._vae_decode_tiled_op.decode(
                samples=latent,
                vae=vae,
                tile_size=_DEFAULT_TILE_SIZE,
            )[0]
        else:
            image_batch = self._vae_decode_op.decode(
                samples=latent,
                vae=vae,
            )[0]

        return _pytorch_to_numpy(image_batch)


# ---------------------------------------------------------------------------
# LdmLatentPreviewer — concrete LatentPreviewer adapter
# ---------------------------------------------------------------------------


class LdmLatentPreviewer:
    """Concrete LatentPreviewer adapter backed by a VAEApprox model.

    Produces fast, low-quality preview images from partial latent tensors
    during sampling. These previews are streamed to the client via WebSocket.

    Dependencies injected via constructor enable unit testing with fakes.

    Args:
        vae_approx_model: A loaded VAEApprox model (or fake).
        rearrange_fn: Callable that rearranges (B, C, H, W) tensor to
            (H, W, C) numpy-compatible format for the first batch element.
    """

    __slots__ = ("_rearrange_fn", "_vae_approx_model")

    def __init__(
        self,
        *,
        vae_approx_model: Any,
        rearrange_fn: RearrangeFn,
    ) -> None:
        self._vae_approx_model = vae_approx_model
        self._rearrange_fn = rearrange_fn

    def __repr__(self) -> str:
        return "LdmLatentPreviewer()"

    def preview(self, latent: LatentTensor) -> NDArray[Any]:
        """Generate a preview image from a partial latent tensor.

        Runs the latent through VAEApprox, rescales from [-1, 1] to [0, 255],
        rearranges from (B, C, H, W) to (H, W, C), and clips to uint8.

        Args:
            latent: Raw latent tensor (B, 4, H, W) from an in-progress step.

        Returns:
            Numpy array (H, W, 3) uint8 preview image.

        Raises:
            RuntimeError: If the preview model is not loaded.
        """
        x_sample = latent.to(self._vae_approx_model.current_type)
        x_sample = self._vae_approx_model(x_sample) * 127.5 + 127.5
        x_sample = self._rearrange_fn(x_sample)
        return np.clip(x_sample.cpu().numpy(), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Factory for building a real LdmLatentPreviewer from weights on disk
# ---------------------------------------------------------------------------


def build_latent_previewer(*, vae_approx_path: str) -> LdmLatentPreviewer:
    """Build a production LdmLatentPreviewer from VAEApprox weights.

    Loads the VAEApprox model from disk, configures dtype/device,
    and returns a ready-to-use previewer.

    Args:
        vae_approx_path: Path to the xlvaeapp.pth weights file.

    Returns:
        Configured LdmLatentPreviewer instance.

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    import einops
    import ldm_patched.modules.model_management
    import torch
    from modules.core_vae_approx import VAEApprox

    sd = torch.load(vae_approx_path, map_location="cpu", weights_only=True)
    model = VAEApprox()
    model.load_state_dict(sd)
    del sd
    model.eval()

    if ldm_patched.modules.model_management.should_use_fp16():
        model.half()
        model.current_type = torch.float16
    else:
        model.float()
        model.current_type = torch.float32

    model.to(ldm_patched.modules.model_management.get_torch_device())

    def rearrange_fn(tensor: Any) -> Any:
        """Rearrange (B, C, H, W) -> first sample (H, W, C)."""
        return einops.rearrange(tensor, "b c h w -> b h w c")[0]

    return LdmLatentPreviewer(
        vae_approx_model=model,
        rearrange_fn=rearrange_fn,
    )


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _should_use_tiled(latent: LatentTensor, threshold: int) -> bool:
    """Determine whether tiled decoding should be used based on latent size.

    The latent is 1/8 the pixel resolution. If either implied pixel dimension
    exceeds the threshold, tiled decoding is preferred to avoid OOM.

    Args:
        latent: Latent tensor dict with 'samples' key.
        threshold: Pixel dimension threshold.

    Returns:
        True if tiled decoding should be used.
    """
    samples = latent.get("samples", latent) if isinstance(latent, dict) else latent
    if hasattr(samples, "shape") and len(samples.shape) >= 4:
        _, _, h, w = samples.shape
        pixel_h = h * _LATENT_SCALE_FACTOR
        pixel_w = w * _LATENT_SCALE_FACTOR
        return pixel_h > threshold or pixel_w > threshold
    return False


def _pytorch_to_numpy(image_batch: Any) -> list[NDArray[Any]]:
    """Convert a batch of torch image tensors to uint8 numpy arrays.

    Mirrors FwdFooocus's core.pytorch_to_numpy: clips to [0, 255] and
    converts to uint8. Handles both real torch tensors and fakes that
    support .cpu().numpy().

    Args:
        image_batch: Tensor-like (B, H, W, 3) in float [0, 1] range.

    Returns:
        List of (H, W, 3) uint8 numpy arrays.
    """
    arr = image_batch.cpu().numpy()
    return [np.clip(255.0 * sample, 0, 255).astype(np.uint8) for sample in arr]
