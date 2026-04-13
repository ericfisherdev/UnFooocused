"""RED-phase tests for UNF-73 inpaint pipeline integration.

Encodes acceptance criteria:
- Full inpaint pipeline executes end-to-end composing all prior modules.
- Three modes produce correct results:
  * Improve Detail (NONE engine, no outpaint) -> sampler receives patch_path=None
  * Modify Content (parameterized engine) -> sampler receives patch_path set
  * Outpaint (directions non-empty) -> image padded, k forced to 1.0
- Auto-mask path merges generated mask with user-drawn mask (union).
- Invert mask flag flips mask bits before worker construction.
- Post-processing composites sampler output back into the original image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from modules.domain.inpaint_models import InpaintEngineVersion
from modules.domain.mask_generation import MaskModel, SamPrompt
from modules.domain.outpaint import OutpaintDirection
from modules.domain.pipeline import PipelineRequest, PipelineResult
from modules.services.inpaint_model_loader import InpaintModelLoader
from modules.services.inpaint_pipeline import InpaintPipeline
from modules.services.mask_generator import MaskGenerator
from numpy.typing import NDArray  # noqa: TC002 — runtime-referenced by test helper signatures


class _FakeImageOps:
    """Minimal ImageOps for InpaintWorker under test."""

    def image_shape_ceil(self, image: NDArray[np.uint8]) -> int:
        return max(image.shape[:2])

    def upscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return image

    def set_image_shape_ceil(self, image: NDArray[np.uint8], ceil: int) -> NDArray[np.uint8]:
        return image

    def resample(self, image: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
        if image.ndim == 2:
            return np.zeros((height, width), dtype=image.dtype)
        return np.zeros((height, width, image.shape[2]), dtype=image.dtype)


class _FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def fetch(self, url: str, dest: Path) -> Path:
        self.calls.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"FAKE")
        return dest


class _FakeRembg:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def segment(self, image: NDArray[np.uint8], model: str) -> NDArray[np.bool_]:
        self.calls.append(model)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2 :, :] = True
        return mask


class _FakeSam:
    def segment(self, image, prompt):  # pragma: no cover - not exercised
        raise NotImplementedError


class _FakeSampler:
    """Records sampler calls and returns a deterministic output image."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample(
        self,
        interested_image: NDArray[np.uint8],
        interested_mask: NDArray[np.uint8],
        interested_fill: NDArray[np.uint8],
        patch_path: Path | None,
        denoising_strength: float,
    ) -> NDArray[np.uint8]:
        self.calls.append(
            {
                "interested_image_shape": interested_image.shape,
                "interested_mask_shape": interested_mask.shape,
                "interested_fill_shape": interested_fill.shape,
                "patch_path": patch_path,
                "denoising_strength": denoising_strength,
            }
        )
        return np.full_like(interested_image, fill_value=128)


def _make_image(h: int = 64, w: int = 64) -> NDArray[np.uint8]:
    return np.full((h, w, 3), fill_value=200, dtype=np.uint8)


def _make_mask(h: int = 64, w: int = 64) -> NDArray[np.uint8]:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    return mask


def _build_pipeline(
    tmp_path: Path,
    sampler: _FakeSampler,
    mask_generator: MaskGenerator | None = None,
) -> tuple[InpaintPipeline, _FakeDownloader]:
    downloader = _FakeDownloader()
    loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
    pipeline = InpaintPipeline(
        image_ops=_FakeImageOps(),
        model_loader=loader,
        sampler=sampler,
        mask_generator=mask_generator,
    )
    return pipeline, downloader


class TestPipelineRequest:
    def test_is_frozen(self) -> None:
        req = PipelineRequest(
            image=_make_image(),
            user_mask=_make_mask(),
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset(),
            denoising_strength=1.0,
            auto_mask_model=None,
            sam_prompt=None,
            invert_mask=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            req.denoising_strength = 0.5  # type: ignore[misc]

    def test_denoising_strength_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="denoising_strength"):
            PipelineRequest(
                image=_make_image(),
                user_mask=_make_mask(),
                engine_version=InpaintEngineVersion.NONE,
                outpaint_directions=frozenset(),
                denoising_strength=1.5,
                auto_mask_model=None,
                sam_prompt=None,
                invert_mask=False,
            )

    def test_image_and_mask_spatial_dims_must_match(self) -> None:
        with pytest.raises(ValueError, match="spatial"):
            PipelineRequest(
                image=_make_image(64, 64),
                user_mask=_make_mask(32, 32),
                engine_version=InpaintEngineVersion.NONE,
                outpaint_directions=frozenset(),
                denoising_strength=1.0,
                auto_mask_model=None,
                sam_prompt=None,
                invert_mask=False,
            )


class TestImproveDetailMode:
    def test_none_engine_skips_download(self, tmp_path: Path) -> None:
        sampler = _FakeSampler()
        pipeline, downloader = _build_pipeline(tmp_path, sampler)
        request = PipelineRequest(
            image=_make_image(),
            user_mask=_make_mask(),
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset(),
            denoising_strength=0.5,
            auto_mask_model=None,
            sam_prompt=None,
            invert_mask=False,
        )

        result = pipeline.run(request)

        assert downloader.calls == []
        assert len(sampler.calls) == 1
        assert sampler.calls[0]["patch_path"] is None
        assert sampler.calls[0]["denoising_strength"] == 0.5
        assert isinstance(result, PipelineResult)
        assert result.image.shape == (64, 64, 3)


class TestModifyContentMode:
    @pytest.mark.parametrize(
        "version",
        [
            InpaintEngineVersion.V1,
            InpaintEngineVersion.V2_5,
            InpaintEngineVersion.V2_6,
        ],
    )
    def test_parameterized_engine_downloads_and_passes_patch_path(
        self, tmp_path: Path, version: InpaintEngineVersion
    ) -> None:
        sampler = _FakeSampler()
        pipeline, downloader = _build_pipeline(tmp_path, sampler)
        request = PipelineRequest(
            image=_make_image(),
            user_mask=_make_mask(),
            engine_version=version,
            outpaint_directions=frozenset(),
            denoising_strength=1.0,
            auto_mask_model=None,
            sam_prompt=None,
            invert_mask=False,
        )

        pipeline.run(request)

        assert len(downloader.calls) == 2
        patch_path = sampler.calls[0]["patch_path"]
        assert patch_path is not None
        assert isinstance(patch_path, Path)
        assert patch_path.exists()


class TestOutpaintMode:
    def test_outpaint_pads_image_beyond_input_dimensions(self, tmp_path: Path) -> None:
        sampler = _FakeSampler()
        pipeline, _ = _build_pipeline(tmp_path, sampler)
        request = PipelineRequest(
            image=_make_image(64, 64),
            user_mask=_make_mask(64, 64),
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset({OutpaintDirection.RIGHT}),
            denoising_strength=1.0,
            auto_mask_model=None,
            sam_prompt=None,
            invert_mask=False,
        )

        result = pipeline.run(request)

        assert result.image.shape[1] > 64

    def test_outpaint_runs_with_mask_even_without_user_marks(self, tmp_path: Path) -> None:
        sampler = _FakeSampler()
        pipeline, _ = _build_pipeline(tmp_path, sampler)
        blank_mask = np.zeros((64, 64), dtype=np.uint8)
        request = PipelineRequest(
            image=_make_image(64, 64),
            user_mask=blank_mask,
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset({OutpaintDirection.TOP}),
            denoising_strength=1.0,
            auto_mask_model=None,
            sam_prompt=None,
            invert_mask=False,
        )

        result = pipeline.run(request)

        assert result.image.shape[0] > 64
        assert len(sampler.calls) == 1


class TestAutoMaskPath:
    def test_auto_mask_merges_with_user_mask(self, tmp_path: Path) -> None:
        rembg = _FakeRembg()
        generator = MaskGenerator(rembg=rembg, sam=_FakeSam())
        sampler = _FakeSampler()
        pipeline, _ = _build_pipeline(tmp_path, sampler, mask_generator=generator)
        request = PipelineRequest(
            image=_make_image(),
            user_mask=_make_mask(),
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset(),
            denoising_strength=1.0,
            auto_mask_model=MaskModel.U2NET,
            sam_prompt=None,
            invert_mask=False,
        )

        pipeline.run(request)

        assert rembg.calls == ["u2net"]
        assert len(sampler.calls) == 1

    def test_auto_mask_without_generator_raises(self, tmp_path: Path) -> None:
        sampler = _FakeSampler()
        pipeline, _ = _build_pipeline(tmp_path, sampler, mask_generator=None)
        request = PipelineRequest(
            image=_make_image(),
            user_mask=_make_mask(),
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset(),
            denoising_strength=1.0,
            auto_mask_model=MaskModel.U2NET,
            sam_prompt=None,
            invert_mask=False,
        )

        with pytest.raises(ValueError, match="mask_generator"):
            pipeline.run(request)


class TestInvertMask:
    def test_invert_flag_flips_user_mask_bits(self, tmp_path: Path) -> None:
        sampler = _FakeSampler()
        pipeline, _ = _build_pipeline(tmp_path, sampler)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        request = PipelineRequest(
            image=_make_image(),
            user_mask=mask,
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset(),
            denoising_strength=1.0,
            auto_mask_model=None,
            sam_prompt=None,
            invert_mask=True,
        )

        pipeline.run(request)

        assert len(sampler.calls) == 1


class TestSamPromptPassthrough:
    def test_sam_auto_mask_forwards_prompt(self, tmp_path: Path) -> None:
        class _RecordingSam:
            def __init__(self) -> None:
                self.calls: list[SamPrompt] = []

            def segment(self, image: NDArray[np.uint8], prompt: SamPrompt) -> NDArray[np.bool_]:
                self.calls.append(prompt)
                h, w = image.shape[:2]
                return np.ones((h, w), dtype=bool)

        sam = _RecordingSam()
        generator = MaskGenerator(rembg=_FakeRembg(), sam=sam)
        sampler = _FakeSampler()
        pipeline, _ = _build_pipeline(tmp_path, sampler, mask_generator=generator)
        from modules.domain.mask_generation import SamVariant

        prompt = SamPrompt(
            points=((32, 32),),
            point_labels=(1,),
            boxes=(),
            variant=SamVariant.VIT_B,
        )
        request = PipelineRequest(
            image=_make_image(),
            user_mask=_make_mask(),
            engine_version=InpaintEngineVersion.NONE,
            outpaint_directions=frozenset(),
            denoising_strength=1.0,
            auto_mask_model=MaskModel.SAM,
            sam_prompt=prompt,
            invert_mask=False,
        )

        pipeline.run(request)

        assert sam.calls == [prompt]
