"""Domain layer for inpaint pipeline request/result aggregates (UNF-73).

Value objects describing a single end-to-end inpaint generation request and
its result. No framework imports — orchestration lives in the service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002 — runtime-referenced by dataclass field annotations

if TYPE_CHECKING:
    from modules.domain.inpaint_models import InpaintEngineVersion
    from modules.domain.mask_generation import MaskModel, SamPrompt
    from modules.domain.outpaint import OutpaintDirection
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Immutable end-to-end inpaint pipeline request.

    Invariants enforced in __post_init__:
    - denoising_strength in [0.0, 1.0]
    - image and user_mask spatial dimensions match

    Note: `image` and `user_mask` are numpy arrays whose element contents are
    mutable even though the dataclass is frozen. Callers must treat the arrays
    as read-only after construction to preserve aggregate invariants.
    """

    image: NDArray[np.uint8]
    user_mask: NDArray[np.uint8]
    engine_version: InpaintEngineVersion
    outpaint_directions: frozenset[OutpaintDirection]
    denoising_strength: float
    auto_mask_model: MaskModel | None
    sam_prompt: SamPrompt | None
    invert_mask: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.denoising_strength <= 1.0:
            raise ValueError(
                f"denoising_strength must be in [0.0, 1.0], got {self.denoising_strength}",
            )
        if self.image.shape[:2] != self.user_mask.shape:
            raise ValueError(
                "image and user_mask spatial dimensions must match: "
                f"image={self.image.shape[:2]}, user_mask={self.user_mask.shape}",
            )


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Immutable end-to-end inpaint pipeline result."""

    image: NDArray[np.uint8]
