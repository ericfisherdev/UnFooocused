"""RED-phase tests for UNF-74 mask generation models.

Encodes acceptance criteria:
- All 8 mask model options dispatch correctly (u2net, u2netp, u2net_human_seg,
  u2net_cloth_seg, silueta, isnet-general-use, isnet-anime, sam).
- SAM path accepts point and box prompts with variant selection (vit_b/l/h).
- MaskGenerator dispatches rembg vs SAM paths via injected Protocols (DIP).
- combine_masks merges user-drawn mask with generated mask (union/intersect/sub).
- Value objects are frozen.
"""

from __future__ import annotations

import numpy as np
import pytest
from modules.domain.mask_generation import (
    MaskCombineMode,
    MaskGenerationRequest,
    MaskModel,
    SamPrompt,
    SamVariant,
    combine_masks,
)
from modules.services.mask_generator import (
    MaskGenerator,
    RembgSegmenter,  # noqa: F401 — re-exported protocol for test reference
    SamSegmenter,  # noqa: F401
)


class _FakeRembg:
    """Records (image, model) calls; returns a deterministic fake mask."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, str]] = []

    def segment(self, image: np.ndarray, model: str) -> np.ndarray:
        self.calls.append((image, model))
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 2, : w // 2] = True
        return mask


class _FakeSam:
    """Records prompts; returns a deterministic mask per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, SamPrompt]] = []

    def segment(self, image: np.ndarray, prompt: SamPrompt) -> np.ndarray:
        self.calls.append((image, prompt))
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return mask


class TestMaskModel:
    def test_has_eight_options(self) -> None:
        names = {m.value for m in MaskModel}
        assert names == {
            "u2net",
            "u2netp",
            "u2net_human_seg",
            "u2net_cloth_seg",
            "silueta",
            "isnet-general-use",
            "isnet-anime",
            "sam",
        }

    def test_sam_is_distinct_member(self) -> None:
        assert MaskModel.SAM.value == "sam"


class TestSamVariant:
    def test_has_three_variants(self) -> None:
        assert {v.value for v in SamVariant} == {"vit_b", "vit_l", "vit_h"}


class TestSamPrompt:
    def test_is_frozen(self) -> None:
        prompt = SamPrompt(
            points=((10, 20),),
            point_labels=(1,),
            boxes=(),
            variant=SamVariant.VIT_B,
        )
        with pytest.raises((AttributeError, TypeError)):
            prompt.variant = SamVariant.VIT_L  # type: ignore[misc]

    def test_point_labels_must_match_points(self) -> None:
        with pytest.raises(ValueError, match="point_labels"):
            SamPrompt(
                points=((10, 20), (30, 40)),
                point_labels=(1,),
                boxes=(),
                variant=SamVariant.VIT_B,
            )

    def test_requires_at_least_one_prompt(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            SamPrompt(points=(), point_labels=(), boxes=(), variant=SamVariant.VIT_B)

    def test_box_only_prompt_is_valid(self) -> None:
        prompt = SamPrompt(
            points=(),
            point_labels=(),
            boxes=((10, 20, 100, 200),),
            variant=SamVariant.VIT_H,
        )
        assert prompt.boxes == ((10, 20, 100, 200),)


class TestMaskGenerationRequest:
    def test_is_frozen(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        req = MaskGenerationRequest(image=img, model=MaskModel.U2NET, sam_prompt=None)
        with pytest.raises((AttributeError, TypeError)):
            req.model = MaskModel.SILUETA  # type: ignore[misc]

    def test_sam_model_requires_prompt(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="sam_prompt"):
            MaskGenerationRequest(image=img, model=MaskModel.SAM, sam_prompt=None)

    def test_non_sam_model_ignores_prompt(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        req = MaskGenerationRequest(image=img, model=MaskModel.U2NET, sam_prompt=None)
        assert req.sam_prompt is None


class TestMaskGenerator:
    @pytest.mark.parametrize(
        "model",
        [
            MaskModel.U2NET,
            MaskModel.U2NETP,
            MaskModel.U2NET_HUMAN_SEG,
            MaskModel.U2NET_CLOTH_SEG,
            MaskModel.SILUETA,
            MaskModel.ISNET_GENERAL_USE,
            MaskModel.ISNET_ANIME,
        ],
    )
    def test_rembg_models_dispatch_to_rembg_segmenter(self, model: MaskModel) -> None:
        rembg = _FakeRembg()
        sam = _FakeSam()
        generator = MaskGenerator(rembg=rembg, sam=sam)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        request = MaskGenerationRequest(image=image, model=model, sam_prompt=None)

        mask = generator.generate(request)

        assert len(rembg.calls) == 1
        assert rembg.calls[0][1] == model.value
        assert sam.calls == []
        assert mask.shape == (16, 16)
        assert mask.dtype == bool

    def test_sam_model_dispatches_to_sam_segmenter(self) -> None:
        rembg = _FakeRembg()
        sam = _FakeSam()
        generator = MaskGenerator(rembg=rembg, sam=sam)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        prompt = SamPrompt(
            points=((8, 8),),
            point_labels=(1,),
            boxes=(),
            variant=SamVariant.VIT_B,
        )
        request = MaskGenerationRequest(image=image, model=MaskModel.SAM, sam_prompt=prompt)

        mask = generator.generate(request)

        assert len(sam.calls) == 1
        assert sam.calls[0][1] is prompt
        assert rembg.calls == []
        assert mask.dtype == bool


class TestCombineMasks:
    def test_union_mode_logical_or(self) -> None:
        a = np.array([[True, False], [False, True]])
        b = np.array([[False, True], [False, True]])
        result = combine_masks(a, b, MaskCombineMode.UNION)
        expected = np.array([[True, True], [False, True]])
        assert np.array_equal(result, expected)

    def test_intersect_mode_logical_and(self) -> None:
        a = np.array([[True, True], [False, True]])
        b = np.array([[True, False], [False, True]])
        result = combine_masks(a, b, MaskCombineMode.INTERSECT)
        expected = np.array([[True, False], [False, True]])
        assert np.array_equal(result, expected)

    def test_subtract_mode_removes_overlay_from_base(self) -> None:
        base = np.array([[True, True], [True, True]])
        overlay = np.array([[False, True], [False, True]])
        result = combine_masks(base, overlay, MaskCombineMode.SUBTRACT)
        expected = np.array([[True, False], [True, False]])
        assert np.array_equal(result, expected)

    def test_shape_mismatch_raises(self) -> None:
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 5), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            combine_masks(a, b, MaskCombineMode.UNION)

    def test_output_is_boolean(self) -> None:
        a = np.zeros((3, 3), dtype=bool)
        b = np.ones((3, 3), dtype=bool)
        result = combine_masks(a, b, MaskCombineMode.UNION)
        assert result.dtype == bool
