"""Unit tests for UNF-71 VAE inpaint encoding.

RED-phase tests encoding acceptance criteria:
- Masked pixels are blended to 0.5 gray before encoding.
- Unmasked pixels pass through unchanged.
- Mask is downsampled to latent spatial dimensions via max-pool
  (any masked pixel in an 8x8 block yields a masked latent cell).
- VAE encoder is called for both the masked inpaint image and the fill image.
- Encoding result holds latent_inpaint, latent_mask, latent_fill with
  matching spatial shapes.
"""

from __future__ import annotations

import pytest
import torch
from modules.domain.vae_inpaint import (
    VaeInpaintEncoding,
    blend_masked_to_gray,
    downsample_mask_to_latent,
)
from modules.services.vae_inpaint_encoder import encode_vae_inpaint


class _FakeVaeEncoder:
    """Fake VAE encoder that downsamples HxW to (H/8)x(W/8) with 4 latent channels.

    Output tensor values equal the batch-averaged mean of each 8x8 block so
    tests can assert the encoder actually received the blended input.
    """

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append(pixels.clone())
        b, h, w, _ = pixels.shape
        latent_h, latent_w = h // 8, w // 8
        avg = pixels.mean(dim=-1, keepdim=False)
        pooled = torch.nn.functional.avg_pool2d(avg.unsqueeze(1), kernel_size=8, stride=8)
        return pooled.expand(b, 4, latent_h, latent_w).contiguous()


def _solid_pixels(b: int, h: int, w: int, value: float) -> torch.Tensor:
    return torch.full((b, h, w, 3), value, dtype=torch.float32)


def _zero_mask(b: int, h: int, w: int) -> torch.Tensor:
    return torch.zeros((b, h, w), dtype=torch.float32)


class TestBlendMaskedToGray:
    def test_fully_masked_pixels_become_half(self) -> None:
        pixels = _solid_pixels(1, 8, 8, value=1.0)
        mask = torch.ones((1, 8, 8), dtype=torch.float32)
        blended = blend_masked_to_gray(pixels, mask)
        assert torch.allclose(blended, torch.full_like(pixels, 0.5))

    def test_unmasked_pixels_pass_through(self) -> None:
        pixels = _solid_pixels(1, 8, 8, value=0.3)
        mask = _zero_mask(1, 8, 8)
        blended = blend_masked_to_gray(pixels, mask)
        assert torch.allclose(blended, pixels)

    def test_partial_mask_blends_only_masked_region(self) -> None:
        pixels = _solid_pixels(1, 8, 8, value=0.9)
        mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        mask[:, :4, :] = 1.0
        blended = blend_masked_to_gray(pixels, mask)
        assert torch.allclose(blended[:, :4, :, :], torch.full((1, 4, 8, 3), 0.5))
        assert torch.allclose(blended[:, 4:, :, :], torch.full((1, 4, 8, 3), 0.9))

    def test_does_not_mutate_input(self) -> None:
        pixels = _solid_pixels(1, 8, 8, value=0.9)
        mask = torch.ones((1, 8, 8), dtype=torch.float32)
        pixels_copy = pixels.clone()
        blend_masked_to_gray(pixels, mask)
        assert torch.equal(pixels, pixels_copy)


class TestDownsampleMaskToLatent:
    def test_output_shape_matches_target(self) -> None:
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        latent_mask = downsample_mask_to_latent(mask, latent_h=8, latent_w=8)
        assert latent_mask.shape == (1, 1, 8, 8)

    def test_any_masked_pixel_propagates_to_latent_cell(self) -> None:
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        mask[0, 0, 0] = 1.0
        latent_mask = downsample_mask_to_latent(mask, latent_h=8, latent_w=8)
        assert latent_mask[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert latent_mask[0, 0, 0, 1:].sum().item() == pytest.approx(0.0)

    def test_fully_masked_region_maps_to_all_ones(self) -> None:
        mask = torch.ones((1, 64, 64), dtype=torch.float32)
        latent_mask = downsample_mask_to_latent(mask, latent_h=8, latent_w=8)
        assert torch.allclose(latent_mask, torch.ones_like(latent_mask))

    def test_fully_unmasked_region_maps_to_all_zeros(self) -> None:
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        latent_mask = downsample_mask_to_latent(mask, latent_h=8, latent_w=8)
        assert torch.allclose(latent_mask, torch.zeros_like(latent_mask))


class TestVaeInpaintEncoding:
    def test_is_frozen_value_object(self) -> None:
        encoding = VaeInpaintEncoding(
            latent_inpaint=torch.zeros(1, 4, 8, 8),
            latent_mask=torch.zeros(1, 1, 8, 8),
            latent_fill=torch.zeros(1, 4, 8, 8),
        )
        with pytest.raises((AttributeError, TypeError)):
            encoding.latent_inpaint = torch.zeros(1, 4, 8, 8)  # type: ignore[misc]


class TestEncodeVaeInpaint:
    def test_calls_encoder_with_blended_inpaint_pixels(self) -> None:
        pixels = _solid_pixels(1, 64, 64, value=1.0)
        mask = torch.ones((1, 64, 64), dtype=torch.float32)
        fill_pixels = _solid_pixels(1, 64, 64, value=0.25)
        encoder = _FakeVaeEncoder()
        encode_vae_inpaint(encoder, pixels, mask, fill_pixels)
        assert len(encoder.calls) == 2
        assert torch.allclose(encoder.calls[0], torch.full_like(pixels, 0.5))

    def test_calls_encoder_with_fill_pixels_second(self) -> None:
        pixels = _solid_pixels(1, 64, 64, value=1.0)
        mask = _zero_mask(1, 64, 64)
        fill_pixels = _solid_pixels(1, 64, 64, value=0.25)
        encoder = _FakeVaeEncoder()
        encode_vae_inpaint(encoder, pixels, mask, fill_pixels)
        assert torch.allclose(encoder.calls[1], fill_pixels)

    def test_result_latent_shapes_match(self) -> None:
        pixels = _solid_pixels(1, 64, 64, value=0.7)
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        mask[0, :32, :32] = 1.0
        fill_pixels = _solid_pixels(1, 64, 64, value=0.5)
        encoder = _FakeVaeEncoder()
        result = encode_vae_inpaint(encoder, pixels, mask, fill_pixels)
        assert result.latent_inpaint.shape == (1, 4, 8, 8)
        assert result.latent_fill.shape == (1, 4, 8, 8)
        assert result.latent_mask.shape[-2:] == (8, 8)

    def test_latent_mask_reflects_downsampled_input_mask(self) -> None:
        pixels = _solid_pixels(1, 64, 64, value=0.7)
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        mask[0, :32, :32] = 1.0
        fill_pixels = _solid_pixels(1, 64, 64, value=0.5)
        encoder = _FakeVaeEncoder()
        result = encode_vae_inpaint(encoder, pixels, mask, fill_pixels)
        assert torch.allclose(result.latent_mask[0, 0, :4, :4], torch.ones(4, 4))
        assert torch.allclose(result.latent_mask[0, 0, 4:, :], torch.zeros(4, 8))
        assert torch.allclose(result.latent_mask[0, 0, :, 4:], torch.zeros(8, 4))

    def test_does_not_mutate_input_tensors(self) -> None:
        pixels = _solid_pixels(1, 64, 64, value=0.7)
        mask = torch.ones((1, 64, 64), dtype=torch.float32)
        fill_pixels = _solid_pixels(1, 64, 64, value=0.5)
        pixels_copy = pixels.clone()
        mask_copy = mask.clone()
        fill_copy = fill_pixels.clone()
        encoder = _FakeVaeEncoder()
        encode_vae_inpaint(encoder, pixels, mask, fill_pixels)
        assert torch.equal(pixels, pixels_copy)
        assert torch.equal(mask, mask_copy)
        assert torch.equal(fill_pixels, fill_copy)
