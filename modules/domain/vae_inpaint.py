"""Domain layer for VAE inpaint encoding (UNF-71).

Pure torch tensor math used to prepare an image + mask for VAE encoding:

- `blend_masked_to_gray` replaces masked pixels with neutral 0.5 gray so the
  VAE does not leak original content into the inpainted region.
- `downsample_mask_to_latent` shrinks a pixel-space mask to the latent grid
  (8x downsample) via bilinear interpolate + max-pool so any masked pixel
  inside a cell marks that cell as masked.
- `VaeInpaintEncoding` bundles the three latent tensors produced by the
  encoder service: inpaint latent, latent mask, fill latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class VaeInpaintEncoding:
    """Latent tensors produced by VAE inpaint encoding."""

    latent_inpaint: torch.Tensor
    latent_mask: torch.Tensor
    latent_fill: torch.Tensor


def blend_masked_to_gray(pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked pixel regions with 0.5 gray.

    `pixels` is shaped (B, H, W, C) (channels-last as consumed by the VAE);
    `mask` is shaped (B, H, W) with values in [0, 1]. Returns a new tensor —
    input is not mutated.
    """
    weight = mask.round().unsqueeze(-1)
    return pixels * (1.0 - weight) + 0.5 * weight


def downsample_mask_to_latent(mask: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """Downsample a pixel-space mask to latent spatial dimensions.

    `mask` is (B, H, W). Returns (B, 1, latent_h, latent_w). Any pixel inside
    an 8x8 block that is masked propagates to the latent cell via max-pool.
    """
    channel_mask = mask.unsqueeze(1)
    pixel_h, pixel_w = latent_h * 8, latent_w * 8
    resized = torch.nn.functional.interpolate(
        channel_mask, size=(pixel_h, pixel_w), mode="bilinear", align_corners=False
    ).round()
    pooled = torch.nn.functional.max_pool2d(resized, kernel_size=(8, 8))
    return pooled.round()
