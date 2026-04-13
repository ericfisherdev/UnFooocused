"""Service layer orchestrator for end-to-end inpaint generation (UNF-73).

Composes mask generation, outpaint padding, model download, interested-area
preparation, diffusion sampling, and post-process compositing into a single
`run()` call. All heavy dependencies (sampler, image ops, segmenters) are
injected via Protocols so the pipeline is fully testable with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from modules.domain.mask_generation import MaskGenerationRequest
from modules.domain.outpaint import apply_outpaint_padding
from modules.domain.pipeline import PipelineRequest, PipelineResult
from modules.services.inpaint_worker import DEFAULT_RESPECTIVE_FIELD, ImageOps, InpaintWorker

if TYPE_CHECKING:
    from pathlib import Path

    from modules.services.inpaint_model_loader import InpaintModelLoader
    from modules.services.mask_generator import MaskGenerator
    from numpy.typing import NDArray


class Sampler(Protocol):
    """Structural contract for the diffusion sampler — hides torch from the domain."""

    def sample(
        self,
        interested_image: NDArray[np.uint8],
        interested_mask: NDArray[np.uint8],
        interested_fill: NDArray[np.uint8],
        patch_path: Path | None,
        denoising_strength: float,
    ) -> NDArray[np.uint8]: ...


@dataclass(frozen=True, slots=True)
class InpaintPipeline:
    """Orchestrates the full inpaint generation pipeline."""

    image_ops: ImageOps
    model_loader: InpaintModelLoader
    sampler: Sampler
    mask_generator: MaskGenerator | None = None

    def run(self, request: PipelineRequest) -> PipelineResult:
        mask = self._resolve_mask(request)
        if request.invert_mask:
            mask = 255 - mask

        image, mask, effective_k = self._apply_outpaint(request, mask)
        paths = self.model_loader.ensure_models(request.engine_version)

        worker = InpaintWorker(
            image=image,
            mask=mask,
            image_ops=self.image_ops,
            use_fill=True,
            k=effective_k,
        )
        generated = self.sampler.sample(
            interested_image=worker.interested_image,
            interested_mask=worker.interested_mask,
            interested_fill=worker.interested_fill,
            patch_path=paths.patch_path,
            denoising_strength=request.denoising_strength,
        )
        return PipelineResult(image=worker.post_process(generated))

    def _resolve_mask(self, request: PipelineRequest) -> NDArray[np.uint8]:
        if request.auto_mask_model is None:
            return request.user_mask
        if self.mask_generator is None:
            raise ValueError("mask_generator required when auto_mask_model is set")
        gen_request = MaskGenerationRequest(
            image=request.image,
            model=request.auto_mask_model,
            sam_prompt=request.sam_prompt,
        )
        auto_mask_bool = self.mask_generator.generate(gen_request)
        auto_mask_uint8 = (auto_mask_bool.astype(np.uint8)) * 255
        return np.maximum(request.user_mask, auto_mask_uint8)

    def _apply_outpaint(
        self,
        request: PipelineRequest,
        mask: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], float]:
        if not request.outpaint_directions:
            return request.image, mask, DEFAULT_RESPECTIVE_FIELD
        padded = apply_outpaint_padding(
            request.image,
            mask,
            tuple(request.outpaint_directions),
        )
        return padded.image, padded.mask, 1.0


__all__ = ["InpaintPipeline", "Sampler"]
