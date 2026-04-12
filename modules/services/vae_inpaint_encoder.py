"""Service layer for VAE inpaint encoding (UNF-71).

Orchestrates the domain `vae_inpaint` helpers around an injected `VaeEncoder`
to produce the three latent tensors needed by the inpainting pipeline:
`latent_inpaint`, `latent_mask`, and `latent_fill`.
"""

from __future__ import annotations

from typing import Protocol

import torch  # noqa: TC002 — runtime-referenced by Protocol signature and function args
from modules.domain.vae_inpaint import (
    VaeInpaintEncoding,
    blend_masked_to_gray,
    downsample_mask_to_latent,
)


class VaeEncoder(Protocol):
    """Structural interface for a VAE encoder.

    Accepts a pixel tensor of shape (B, H, W, C) and returns a latent tensor
    of shape (B, C_latent, H/8, W/8).
    """

    def encode(self, pixels: torch.Tensor) -> torch.Tensor: ...


def encode_vae_inpaint(
    encoder: VaeEncoder,
    pixels: torch.Tensor,
    mask: torch.Tensor,
    fill_pixels: torch.Tensor,
) -> VaeInpaintEncoding:
    """Encode image + mask + fill into latent space for inpainting.

    Steps:
      1. Blend masked pixel regions to 0.5 gray.
      2. Encode the blended pixels via the VAE.
      3. Downsample the mask to the latent spatial grid.
      4. Encode the fill pixels separately (used for mixing masked latents).
    """
    blended = blend_masked_to_gray(pixels, mask)
    latent_inpaint = encoder.encode(blended)
    latent_h, latent_w = latent_inpaint.shape[-2], latent_inpaint.shape[-1]
    latent_mask = downsample_mask_to_latent(mask, latent_h=latent_h, latent_w=latent_w)
    latent_fill = encoder.encode(fill_pixels)
    return VaeInpaintEncoding(
        latent_inpaint=latent_inpaint,
        latent_mask=latent_mask.to(device=latent_inpaint.device, dtype=latent_inpaint.dtype),
        latent_fill=latent_fill,
    )
