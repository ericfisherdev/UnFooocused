"""Domain layer for automatic mask generation (UNF-74).

Value objects + enums + pure functions for the mask generation bounded context.
No framework imports — rembg, segment_anything, and torch live in the service
layer behind Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, assert_never

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MaskModel(Enum):
    """Automatic mask generation model options.

    Seven rembg-backed options cover U2Net and ISNet families; SAM is dispatched
    separately because it requires point/box prompts rather than salient-object
    segmentation.
    """

    U2NET = "u2net"
    U2NETP = "u2netp"
    U2NET_HUMAN_SEG = "u2net_human_seg"
    U2NET_CLOTH_SEG = "u2net_cloth_seg"
    SILUETA = "silueta"
    ISNET_GENERAL_USE = "isnet-general-use"
    ISNET_ANIME = "isnet-anime"
    SAM = "sam"


class SamVariant(Enum):
    """Segment Anything backbone variant."""

    VIT_B = "vit_b"
    VIT_L = "vit_l"
    VIT_H = "vit_h"


class MaskCombineMode(Enum):
    """Boolean combination mode for merging generated + user-drawn masks."""

    UNION = "union"
    INTERSECT = "intersect"
    SUBTRACT = "subtract"


@dataclass(frozen=True, slots=True)
class SamPrompt:
    """Point/box prompts for SAM segmentation.

    Invariants:
    - `point_labels` length must equal `points` length (1=foreground, 0=background).
    - At least one of points or boxes must be non-empty.
    """

    points: tuple[tuple[int, int], ...]
    point_labels: tuple[int, ...]
    boxes: tuple[tuple[int, int, int, int], ...]
    variant: SamVariant

    def __post_init__(self) -> None:
        if len(self.point_labels) != len(self.points):
            raise ValueError(
                "point_labels length must match points length",
            )
        if not self.points and not self.boxes:
            raise ValueError("SamPrompt requires at least one point or box")


@dataclass(frozen=True, slots=True)
class MaskGenerationRequest:
    """Aggregate request for a single mask generation call."""

    image: NDArray[np.uint8]
    model: MaskModel
    sam_prompt: SamPrompt | None

    def __post_init__(self) -> None:
        if self.model is MaskModel.SAM and self.sam_prompt is None:
            raise ValueError("sam_prompt is required when model is SAM")


def combine_masks(
    base: NDArray[np.bool_],
    overlay: NDArray[np.bool_],
    mode: MaskCombineMode,
) -> NDArray[np.bool_]:
    """Combine two boolean masks under the given mode.

    UNION = logical OR, INTERSECT = logical AND, SUBTRACT = base AND NOT overlay.
    """
    if base.shape != overlay.shape:
        raise ValueError(
            f"mask shape mismatch: base={base.shape} overlay={overlay.shape}",
        )
    if mode is MaskCombineMode.UNION:
        return np.logical_or(base, overlay)
    if mode is MaskCombineMode.INTERSECT:
        return np.logical_and(base, overlay)
    if mode is MaskCombineMode.SUBTRACT:
        return np.logical_and(base, np.logical_not(overlay))
    assert_never(mode)
