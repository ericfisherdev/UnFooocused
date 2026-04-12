"""Service layer for mask generation (UNF-74).

Dispatches MaskGenerationRequest to the appropriate segmenter Protocol.
rembg handles the seven U2Net/ISNet options; SAM handles point/box prompts.
Both segmenters are injected so tests can substitute fakes without loading
real networks or touching disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from modules.domain.mask_generation import (
    MaskGenerationRequest,
    MaskModel,
    SamPrompt,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class RembgSegmenter(Protocol):
    """Structural contract for rembg-backed salient-object segmentation."""

    def segment(self, image: NDArray[np.uint8], model: str) -> NDArray[np.bool_]: ...


class SamSegmenter(Protocol):
    """Structural contract for Segment Anything prompt-driven segmentation."""

    def segment(self, image: NDArray[np.uint8], prompt: SamPrompt) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class MaskGenerator:
    """Dispatches a mask generation request to rembg or SAM."""

    rembg: RembgSegmenter
    sam: SamSegmenter

    def generate(self, request: MaskGenerationRequest) -> NDArray[np.bool_]:
        if request.model is MaskModel.SAM:
            assert request.sam_prompt is not None
            return self.sam.segment(request.image, request.sam_prompt)
        return self.rembg.segment(request.image, request.model.value)
