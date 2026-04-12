"""Unit tests for UNF-70 outpaint padding logic.

RED-phase tests encoding acceptance criteria:
- Image padded correctly per selected directions.
- Mask extended with white for new padded areas.
- Multiple directions can be combined.
- Strength and respective field overridden to 1.0 when outpainting.
- Original image content preserved in non-padded area.
"""

from __future__ import annotations

import numpy as np
import pytest
from modules.domain.outpaint import (
    OUTPAINT_PADDING_RATIO,
    OutpaintDirection,
    OutpaintOverrides,
    apply_outpaint_padding,
)


def _solid_image(h: int, w: int, value: int = 120) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _solid_mask(h: int, w: int, value: int = 0) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


class TestOutpaintDirectionEnum:
    def test_has_four_cardinal_directions(self) -> None:
        names = {d.name for d in OutpaintDirection}
        assert names == {"TOP", "BOTTOM", "LEFT", "RIGHT"}


class TestOutpaintOverrides:
    def test_overrides_are_one_when_outpainting(self) -> None:
        overrides = OutpaintOverrides.forced()
        assert overrides.inpaint_strength == pytest.approx(1.0)
        assert overrides.inpaint_respective_field == pytest.approx(1.0)


class TestApplyOutpaintPaddingNoDirections:
    def test_empty_directions_returns_inputs_unchanged(self) -> None:
        image = _solid_image(64, 80)
        mask = _solid_mask(64, 80)
        result = apply_outpaint_padding(image, mask, directions=())
        assert np.array_equal(result.image, image)
        assert np.array_equal(result.mask, mask)

    def test_empty_directions_has_no_forced_overrides(self) -> None:
        image = _solid_image(64, 80)
        mask = _solid_mask(64, 80)
        result = apply_outpaint_padding(image, mask, directions=())
        assert result.overrides is None


class TestApplyOutpaintPaddingTop:
    def test_top_padding_adds_30_percent_to_height(self) -> None:
        image = _solid_image(100, 50)
        mask = _solid_mask(100, 50)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.TOP,))
        expected_pad = int(100 * OUTPAINT_PADDING_RATIO)
        assert result.image.shape[0] == 100 + expected_pad
        assert result.image.shape[1] == 50

    def test_top_padding_extends_mask_with_white(self) -> None:
        image = _solid_image(100, 50)
        mask = _solid_mask(100, 50, value=0)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.TOP,))
        pad = int(100 * OUTPAINT_PADDING_RATIO)
        assert np.all(result.mask[:pad] == 255)
        assert np.all(result.mask[pad:] == 0)

    def test_top_padding_uses_edge_replication(self) -> None:
        image = np.zeros((20, 10, 3), dtype=np.uint8)
        image[0, :, :] = 77
        mask = _solid_mask(20, 10)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.TOP,))
        pad = int(20 * OUTPAINT_PADDING_RATIO)
        assert np.all(result.image[:pad, :, 0] == 77)


class TestApplyOutpaintPaddingBottom:
    def test_bottom_padding_extends_mask_with_white(self) -> None:
        image = _solid_image(100, 50)
        mask = _solid_mask(100, 50)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.BOTTOM,))
        pad = int(100 * OUTPAINT_PADDING_RATIO)
        assert np.all(result.mask[-pad:] == 255)
        assert np.all(result.mask[:-pad] == 0)

    def test_bottom_padding_uses_edge_replication(self) -> None:
        image = np.zeros((20, 10, 3), dtype=np.uint8)
        image[-1, :, :] = 99
        mask = _solid_mask(20, 10)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.BOTTOM,))
        pad = int(20 * OUTPAINT_PADDING_RATIO)
        assert np.all(result.image[-pad:, :, 0] == 99)


class TestApplyOutpaintPaddingLeft:
    def test_left_padding_adds_30_percent_to_width(self) -> None:
        image = _solid_image(50, 100)
        mask = _solid_mask(50, 100)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.LEFT,))
        expected_pad = int(100 * OUTPAINT_PADDING_RATIO)
        assert result.image.shape[1] == 100 + expected_pad
        assert result.image.shape[0] == 50

    def test_left_padding_extends_mask_with_white(self) -> None:
        image = _solid_image(50, 100)
        mask = _solid_mask(50, 100)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.LEFT,))
        pad = int(100 * OUTPAINT_PADDING_RATIO)
        assert np.all(result.mask[:, :pad] == 255)
        assert np.all(result.mask[:, pad:] == 0)


class TestApplyOutpaintPaddingRight:
    def test_right_padding_extends_mask_with_white(self) -> None:
        image = _solid_image(50, 100)
        mask = _solid_mask(50, 100)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.RIGHT,))
        pad = int(100 * OUTPAINT_PADDING_RATIO)
        assert np.all(result.mask[:, -pad:] == 255)
        assert np.all(result.mask[:, :-pad] == 0)


class TestApplyOutpaintPaddingCombinations:
    def test_all_four_directions_expand_both_axes(self) -> None:
        image = _solid_image(100, 100)
        mask = _solid_mask(100, 100)
        result = apply_outpaint_padding(
            image,
            mask,
            directions=(
                OutpaintDirection.TOP,
                OutpaintDirection.BOTTOM,
                OutpaintDirection.LEFT,
                OutpaintDirection.RIGHT,
            ),
        )
        pad_h = int(100 * OUTPAINT_PADDING_RATIO)
        pad_w = int(100 * OUTPAINT_PADDING_RATIO)
        assert result.image.shape == (100 + 2 * pad_h, 100 + 2 * pad_w, 3)

    def test_vertical_and_horizontal_padding_are_independent(self) -> None:
        image = _solid_image(50, 50)
        mask = _solid_mask(50, 50)
        result = apply_outpaint_padding(
            image,
            mask,
            directions=(OutpaintDirection.TOP, OutpaintDirection.LEFT),
        )
        pad_top = int(50 * OUTPAINT_PADDING_RATIO)
        pad_left = int(50 * OUTPAINT_PADDING_RATIO)
        assert result.image.shape == (50 + pad_top, 50 + pad_left, 3)

    def test_original_image_region_preserved(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
        mask = _solid_mask(40, 40)
        result = apply_outpaint_padding(
            image,
            mask,
            directions=(OutpaintDirection.TOP, OutpaintDirection.RIGHT),
        )
        pad_top = int(40 * OUTPAINT_PADDING_RATIO)
        assert np.array_equal(result.image[pad_top : pad_top + 40, :40], image)


class TestApplyOutpaintPaddingOverrides:
    def test_any_direction_forces_unit_overrides(self) -> None:
        image = _solid_image(32, 32)
        mask = _solid_mask(32, 32)
        result = apply_outpaint_padding(image, mask, directions=(OutpaintDirection.LEFT,))
        assert result.overrides is not None
        assert result.overrides.inpaint_strength == pytest.approx(1.0)
        assert result.overrides.inpaint_respective_field == pytest.approx(1.0)


class TestApplyOutpaintPaddingInputImmutability:
    def test_does_not_mutate_input_arrays(self) -> None:
        image = _solid_image(32, 32, value=10)
        mask = _solid_mask(32, 32, value=0)
        image_copy = image.copy()
        mask_copy = mask.copy()
        apply_outpaint_padding(image, mask, directions=(OutpaintDirection.BOTTOM,))
        assert np.array_equal(image, image_copy)
        assert np.array_equal(mask, mask_copy)
