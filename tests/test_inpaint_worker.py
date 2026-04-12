"""Unit tests for UNF-69 InpaintWorker domain logic.

RED-phase tests encoding acceptance criteria:
- InpaintWorker correctly computes interested area from mask.
- Fill computation produces smooth inpainting boundaries.
- Post-processing blends generated image with original seamlessly.
- Unit tests for mask bounding box computation and color correction.
"""

from __future__ import annotations

import numpy as np
import pytest
from modules.domain.inpaint import (
    InterestedArea,
    binarize_mask,
    color_correct,
    compute_initial_bounds,
    expand_bounds_to_min_ratio,
    fooocus_fill,
)
from modules.services.inpaint_worker import InpaintWorker


def _mask_with_rect(shape: tuple[int, int], a: int, b: int, c: int, d: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[a:b, c:d] = 255
    return mask


class TestInterestedArea:
    def test_rejects_non_positive_area(self) -> None:
        with pytest.raises(ValueError):
            InterestedArea(top=10, bottom=10, left=0, right=5)

    def test_height_and_width(self) -> None:
        area = InterestedArea(top=2, bottom=10, left=3, right=13)
        assert area.height == 8
        assert area.width == 10


class TestComputeInitialBounds:
    def test_bounds_cover_mask_region(self) -> None:
        mask = _mask_with_rect((100, 100), 30, 50, 40, 70)
        area = compute_initial_bounds(mask > 0)
        assert area.top <= 30
        assert area.bottom >= 50
        assert area.left <= 40
        assert area.right >= 70

    def test_bounds_are_clamped_to_image(self) -> None:
        mask = _mask_with_rect((50, 50), 0, 10, 0, 10)
        area = compute_initial_bounds(mask > 0)
        assert 0 <= area.top < area.bottom <= 50
        assert 0 <= area.left < area.right <= 50

    def test_symmetric_padding_around_centroid(self) -> None:
        mask = _mask_with_rect((200, 200), 80, 120, 80, 120)
        area = compute_initial_bounds(mask > 0)
        vertical_margin_top = 100 - area.top
        vertical_margin_bottom = area.bottom - 100
        assert abs(vertical_margin_top - vertical_margin_bottom) <= 1


class TestExpandBoundsToMinRatio:
    def test_k_one_returns_full_image(self) -> None:
        mask = _mask_with_rect((64, 80), 30, 40, 30, 40)
        area = compute_initial_bounds(mask > 0)
        expanded = expand_bounds_to_min_ratio(area, mask.shape, k=1.0)
        assert expanded.top == 0
        assert expanded.bottom == 64
        assert expanded.left == 0
        assert expanded.right == 80

    def test_k_zero_leaves_bounds_unchanged(self) -> None:
        mask = _mask_with_rect((100, 100), 40, 60, 40, 60)
        area = compute_initial_bounds(mask > 0)
        expanded = expand_bounds_to_min_ratio(area, mask.shape, k=0.0)
        assert expanded == area

    def test_expansion_reaches_min_ratio(self) -> None:
        mask = _mask_with_rect((200, 200), 90, 110, 90, 110)
        area = compute_initial_bounds(mask > 0)
        expanded = expand_bounds_to_min_ratio(area, mask.shape, k=0.5)
        assert expanded.height >= 100
        assert expanded.width >= 100


class TestBinarizeMask:
    def test_threshold_at_127(self) -> None:
        mask = np.array([[0, 50, 127, 128, 200, 255]], dtype=np.uint8)
        out = binarize_mask(mask, threshold=127)
        assert out.tolist() == [[0, 0, 0, 255, 255, 255]]

    def test_dtype_preserved_as_uint8(self) -> None:
        mask = np.full((4, 4), 200, dtype=np.uint8)
        out = binarize_mask(mask, threshold=127)
        assert out.dtype == np.uint8


class TestColorCorrect:
    def test_mask_255_returns_foreground(self) -> None:
        fg = np.full((4, 4, 3), 200, dtype=np.uint8)
        bg = np.full((4, 4, 3), 40, dtype=np.uint8)
        mask = np.full((4, 4), 255, dtype=np.uint8)
        out = color_correct(fg, bg, mask)
        assert np.array_equal(out, fg)

    def test_mask_zero_returns_background(self) -> None:
        fg = np.full((4, 4, 3), 200, dtype=np.uint8)
        bg = np.full((4, 4, 3), 40, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        out = color_correct(fg, bg, mask)
        assert np.array_equal(out, bg)

    def test_half_mask_blends_linearly(self) -> None:
        fg = np.full((2, 2, 3), 200, dtype=np.uint8)
        bg = np.full((2, 2, 3), 40, dtype=np.uint8)
        mask = np.full((2, 2), 128, dtype=np.uint8)
        out = color_correct(fg, bg, mask)
        expected = round(200 * (128 / 255) + 40 * (1 - 128 / 255))
        assert out[0, 0, 0] == pytest.approx(expected, abs=1)

    def test_output_clipped_to_uint8(self) -> None:
        fg = np.full((2, 2, 3), 255, dtype=np.uint8)
        bg = np.full((2, 2, 3), 255, dtype=np.uint8)
        mask = np.full((2, 2), 200, dtype=np.uint8)
        out = color_correct(fg, bg, mask)
        assert out.dtype == np.uint8
        assert out.max() <= 255


class TestFooocusFill:
    def test_unmasked_pixels_are_preserved(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        mask = _mask_with_rect((32, 32), 10, 20, 10, 20)

        def identity_blur(x: np.ndarray, k: int) -> np.ndarray:
            return x.copy()

        filled = fooocus_fill(image, mask, blur_fn=identity_blur)
        unmasked = mask < 127
        assert np.array_equal(filled[unmasked], image[unmasked])

    def test_masked_pixels_are_replaced_by_blur(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = 50
        mask = _mask_with_rect((16, 16), 4, 12, 4, 12)

        def constant_blur(x: np.ndarray, k: int) -> np.ndarray:
            return np.full_like(x, 200)

        filled = fooocus_fill(image, mask, blur_fn=constant_blur)
        masked = mask >= 127
        assert np.all(filled[masked] == 200)


class _FakeImageOps:
    """Test double for image operations (resample / upscale / shape ceil)."""

    def __init__(self) -> None:
        self.resample_calls: list[tuple[int, int]] = []

    def resample(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        self.resample_calls.append((width, height))
        result = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)
        result[...] = image.mean(axis=(0, 1)).astype(image.dtype)
        return result

    def upscale(self, image: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    def image_shape_ceil(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        return max(h, w)

    def set_image_shape_ceil(self, image: np.ndarray, ceil: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = ceil / max(h, w)
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))
        return self.resample(image, new_w, new_h)


class TestInpaintWorkerConstruction:
    def test_interested_area_matches_mask_bbox(self) -> None:
        image = np.full((256, 256, 3), 80, dtype=np.uint8)
        mask = _mask_with_rect((256, 256), 100, 140, 110, 150)
        worker = InpaintWorker(image=image, mask=mask, image_ops=_FakeImageOps(), use_fill=False, k=0.25)
        area = worker.interested_area
        assert area.top <= 100
        assert area.bottom >= 140
        assert area.left <= 110
        assert area.right >= 150

    def test_interested_image_is_resized_to_ceil_1024(self) -> None:
        image = np.full((256, 256, 3), 80, dtype=np.uint8)
        mask = _mask_with_rect((256, 256), 100, 140, 110, 150)
        worker = InpaintWorker(image=image, mask=mask, image_ops=_FakeImageOps(), use_fill=False, k=0.25)
        h, w = worker.interested_image.shape[:2]
        assert max(h, w) == 1024

    def test_fill_enabled_triggers_fooocus_fill(self) -> None:
        image = np.full((256, 256, 3), 80, dtype=np.uint8)
        mask = _mask_with_rect((256, 256), 100, 140, 110, 150)
        worker = InpaintWorker(image=image, mask=mask, image_ops=_FakeImageOps(), use_fill=True, k=0.25)
        assert worker.interested_fill.shape == worker.interested_image.shape


class TestInpaintWorkerPostProcess:
    def test_post_process_pastes_generated_pixels_inside_mask(self) -> None:
        image = np.full((128, 128, 3), 30, dtype=np.uint8)
        mask = _mask_with_rect((128, 128), 40, 80, 40, 80)
        worker = InpaintWorker(image=image, mask=mask, image_ops=_FakeImageOps(), use_fill=False, k=0.0)
        generated = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = worker.post_process(generated)
        masked_pixels = result[mask == 255]
        assert np.all(masked_pixels == 200), "masked region should be fully replaced by generated content"

    def test_post_process_preserves_outside_interested_area(self) -> None:
        image = np.full((128, 128, 3), 30, dtype=np.uint8)
        mask = _mask_with_rect((128, 128), 40, 80, 40, 80)
        worker = InpaintWorker(image=image, mask=mask, image_ops=_FakeImageOps(), use_fill=False, k=0.0)
        generated = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = worker.post_process(generated)
        assert result[0, 0, 0] == 30
        assert result[127, 127, 0] == 30
