"""Domain layer for inpainting mask and image processing (UNF-69).

Pure numpy value objects and functions capturing the inpainting domain language:
interested-area bounding boxes, mask binarization, color correction, and the
fooocus fill smoothing algorithm. Infrastructure (torch, PIL, cv2) is kept out
of this module — heavy image operations are supplied via Protocols defined in
`modules.services.inpaint_worker`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BlurFn = Callable[[NDArray[np.uint8], int], NDArray[np.uint8]]
"""Box-blur dependency used by `fooocus_fill` — injected for testability."""

INITIAL_BOUNDS_PADDING: float = 1.15
"""Symmetric expansion factor applied to the raw mask bbox half-extent."""

FOOOCUS_FILL_SCHEDULE: tuple[tuple[int, int], ...] = (
    (512, 2),
    (256, 2),
    (128, 4),
    (64, 4),
    (33, 8),
    (15, 8),
    (5, 16),
    (3, 16),
)
"""Multi-scale box-blur schedule ported from the original InpaintWorker implementation."""


@dataclass(frozen=True, slots=True)
class InterestedArea:
    """Axis-aligned bounding box covering the inpaint-relevant region.

    Coordinates are half-open in (top, bottom, left, right) form, matching
    numpy slice semantics: image[top:bottom, left:right].
    """

    top: int
    bottom: int
    left: int
    right: int

    def __post_init__(self) -> None:
        if self.bottom <= self.top or self.right <= self.left:
            raise ValueError(
                f"InterestedArea must have positive height and width: "
                f"top={self.top} bottom={self.bottom} left={self.left} right={self.right}"
            )

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.top, self.bottom, self.left, self.right)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def compute_initial_bounds(mask: NDArray[np.bool_]) -> InterestedArea:
    """Compute a symmetric square-ish bounding box centred on the mask region.

    The returned box expands the raw bbox of truthy mask pixels by ~15% and is
    clamped to the mask shape. Mirrors `compute_initial_abcd` in the upstream inpainting reference.
    """
    if mask.ndim != 2:
        raise ValueError(f"compute_initial_bounds requires a 2D mask, got shape {mask.shape}")
    indices = np.where(mask)
    if indices[0].size == 0:
        raise ValueError("compute_initial_bounds requires a non-empty mask")

    h, w = mask.shape[:2]
    top_raw = int(indices[0].min())
    bottom_raw = int(indices[0].max())
    left_raw = int(indices[1].min())
    right_raw = int(indices[1].max())

    raw_h = bottom_raw - top_raw + 1
    raw_w = right_raw - left_raw + 1
    target_side = int(np.ceil(max(raw_h, raw_w) * INITIAL_BOUNDS_PADDING))

    pad_h = max(0, target_side - raw_h)
    pad_w = max(0, target_side - raw_w)

    top = _clamp(top_raw - pad_h // 2, 0, h)
    bottom = _clamp(bottom_raw + (pad_h - pad_h // 2) + 1, 0, h)
    left = _clamp(left_raw - pad_w // 2, 0, w)
    right = _clamp(right_raw + (pad_w - pad_w // 2) + 1, 0, w)
    return InterestedArea(top=top, bottom=bottom, left=left, right=right)


def expand_bounds_to_min_ratio(area: InterestedArea, shape: tuple[int, ...], k: float) -> InterestedArea:
    """Grow `area` until each side covers at least `k` of the image dimension.

    k==1.0 returns the full image; k==0.0 returns `area` unchanged. Expansion
    alternates between the smaller axis to keep the box approximately square.
    Ported from the upstream `solve_abcd` inpainting helper.
    """
    if not 0.0 <= k <= 1.0:
        raise ValueError(f"k must be in [0.0, 1.0], got {k}")

    h, w = shape[:2]
    if k == 1.0:
        return InterestedArea(top=0, bottom=h, left=0, right=w)
    if k == 0.0:
        return area

    top, bottom, left, right = area.top, area.bottom, area.left, area.right
    min_h = h * k
    min_w = w * k
    while (bottom - top) < min_h or (right - left) < min_w:
        height_now = bottom - top
        width_now = right - left
        add_vertical = height_now < width_now
        add_horizontal = not add_vertical
        if height_now == h:
            add_horizontal = True
            add_vertical = False
        if width_now == w:
            add_vertical = True
            add_horizontal = False
        if add_vertical:
            top -= 1
            bottom += 1
        if add_horizontal:
            left -= 1
            right += 1
        top = _clamp(top, 0, h)
        bottom = _clamp(bottom, 0, h)
        left = _clamp(left, 0, w)
        right = _clamp(right, 0, w)
        if top == 0 and bottom == h and left == 0 and right == w:
            break
    return InterestedArea(top=top, bottom=bottom, left=left, right=right)


def binarize_mask(mask: NDArray[np.uint8], threshold: int = 127) -> NDArray[np.uint8]:
    """Threshold `mask` to {0, 255}. Values > threshold become 255, else 0."""
    result = np.zeros_like(mask, dtype=np.uint8)
    result[mask > threshold] = 255
    return result


def color_correct(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Alpha-blend foreground over background using mask as a soft matte.

    mask is expected to be uint8 in [0, 255]. Output is clipped and cast to
    uint8. Mirrors `InpaintWorker.color_correction` from the upstream inpainting reference.
    """
    weight = (mask.astype(np.float32) / 255.0)[..., None]
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    blended = fg * weight + bg * (1.0 - weight)
    return np.clip(blended, 0, 255).astype(np.uint8)


def fooocus_fill(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    blur_fn: BlurFn,
    schedule: tuple[tuple[int, int], ...] = FOOOCUS_FILL_SCHEDULE,
) -> NDArray[np.uint8]:
    """Multi-scale smoothing of masked regions with unmasked pixels pinned.

    For each (radius, repeats) step in `schedule`, apply `blur_fn(image, radius)`
    then restore the original unmasked pixels. Produces smooth colour bleed
    into the masked region, seeding the inpaint diffusion pass. Mirrors the
    upstream `fooocus_fill` algorithm.
    """
    if image.ndim < 2 or mask.ndim != 2:
        raise ValueError("fooocus_fill expects image with >=2 dims and a 2D mask")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"fooocus_fill image/mask spatial mismatch: image={image.shape[:2]}, mask={mask.shape}")
    current = image.copy()
    unmasked_coords = np.where(mask < 127)
    store = image[unmasked_coords]
    for radius, repeats in schedule:
        for _ in range(repeats):
            current = blur_fn(current, radius)
            current[unmasked_coords] = store
    return current
