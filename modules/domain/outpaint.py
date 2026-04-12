"""Domain layer for outpaint padding (UNF-70).

Pure numpy padding of an image and its inpaint mask along one or more cardinal
directions. Each selected direction extends its axis by a fixed ratio of the
image's current dimension on that axis (30%), replicates edge pixels for the
image, and fills new mask regions with white (255) so the diffusion pipeline
treats them as the inpainting target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 — runtime-referenced by dataclass field and function signatures

OUTPAINT_PADDING_RATIO: float = 0.3
"""Fraction of the current axis dimension added as padding per direction."""


class OutpaintDirection(Enum):
    """Cardinal directions that can be extended during outpainting."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class OutpaintOverrides:
    """Forced-one overrides applied when any outpaint direction is selected.

    Outpainting always needs a full-strength inpaint pass that considers the
    entire image as the receptive field, so the generation pipeline must set
    `inpaint_strength` and `inpaint_respective_field` to 1.0 for the run.
    """

    inpaint_strength: float
    inpaint_respective_field: float

    @classmethod
    def forced(cls) -> OutpaintOverrides:
        return cls(inpaint_strength=1.0, inpaint_respective_field=1.0)


@dataclass(frozen=True, slots=True)
class OutpaintPaddingResult:
    """Padded image + mask, plus optional forced overrides for the pipeline.

    `overrides` is None when no directions were selected (caller keeps the
    user-supplied strength and respective field), and an `OutpaintOverrides`
    instance whenever at least one direction triggered padding.
    """

    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    overrides: OutpaintOverrides | None


def _pad_axis(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    axis: int,
    before: int,
    after: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    image_pad_width: list[tuple[int, int]] = [(0, 0)] * image.ndim
    mask_pad_width: list[tuple[int, int]] = [(0, 0)] * mask.ndim
    image_pad_width[axis] = (before, after)
    mask_pad_width[axis] = (before, after)
    padded_image = np.pad(image, image_pad_width, mode="edge")
    padded_mask = np.pad(mask, mask_pad_width, mode="constant", constant_values=255)
    return padded_image, padded_mask


def apply_outpaint_padding(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    directions: tuple[OutpaintDirection, ...],
) -> OutpaintPaddingResult:
    """Extend `image` and `mask` along each selected outpaint direction.

    Padding amounts are derived from the *original* height for top/bottom and
    the *original* width for left/right — opposite-axis padding does not
    influence the perpendicular pad size. This mirrors the upstream outpaint
    reference where both vertical directions use the pre-pad height and both
    horizontal directions use the pre-pad width.
    """
    if not directions:
        return OutpaintPaddingResult(image=image, mask=mask, overrides=None)

    original_height, original_width = image.shape[:2]
    vertical_pad = int(original_height * OUTPAINT_PADDING_RATIO)
    horizontal_pad = int(original_width * OUTPAINT_PADDING_RATIO)
    padded_image = image
    padded_mask = mask

    if OutpaintDirection.TOP in directions:
        padded_image, padded_mask = _pad_axis(padded_image, padded_mask, axis=0, before=vertical_pad, after=0)
    if OutpaintDirection.BOTTOM in directions:
        padded_image, padded_mask = _pad_axis(padded_image, padded_mask, axis=0, before=0, after=vertical_pad)
    if OutpaintDirection.LEFT in directions:
        padded_image, padded_mask = _pad_axis(padded_image, padded_mask, axis=1, before=horizontal_pad, after=0)
    if OutpaintDirection.RIGHT in directions:
        padded_image, padded_mask = _pad_axis(padded_image, padded_mask, axis=1, before=0, after=horizontal_pad)

    return OutpaintPaddingResult(
        image=np.ascontiguousarray(padded_image),
        mask=np.ascontiguousarray(padded_mask),
        overrides=OutpaintOverrides.forced(),
    )
