"""InpaintWorker service — orchestrates mask preparation and post-processing.

The worker holds the original image and mask, precomputes the interested area
at canonical 1024-ceil resolution, and (optionally) seeds the masked region
with a smooth `fooocus_fill`. After diffusion generates a replacement image,
`post_process` composites it back into the original with color correction.

Heavy image operations (resample / upscale / shape ceil) are injected via an
`ImageOps` Protocol to keep this service testable without torch, cv2, or PIL.
The `InpaintHead` torch patching step lives in an infrastructure module and is
delegated here by accepting a caller-supplied patcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from modules.domain.inpaint import (
    FOOOCUS_FILL_SCHEDULE,
    InterestedArea,
    binarize_mask,
    color_correct,
    compute_initial_bounds,
    expand_bounds_to_min_ratio,
    fooocus_fill,
)
from numpy.typing import NDArray  # noqa: TC002 — runtime-referenced by Protocol signatures

# Note: "respective" preserves the upstream inpainting reference spelling of "receptive field".
DEFAULT_RESPECTIVE_FIELD: float = 0.618
"""Default `k` for interested-area expansion (upstream inpainting reference default)."""

CANONICAL_CEIL: int = 1024
"""Interested-image target max(H, W) before diffusion."""


class ImageOps(Protocol):
    """Image resize/upscale contract — hides PIL / ESRGAN / cv2 from the domain."""

    def resample(self, image: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]: ...

    def upscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]: ...

    def image_shape_ceil(self, image: NDArray[np.uint8]) -> int: ...

    def set_image_shape_ceil(self, image: NDArray[np.uint8], ceil: int) -> NDArray[np.uint8]: ...


def _default_blur(image: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    """PIL-backed box blur for production fooocus_fill use.

    Imported lazily so unit tests that exercise pure domain paths don't require
    PIL on the import path. Tests pass their own BlurFn via the constructor.
    """
    from PIL import Image, ImageFilter

    return np.array(Image.fromarray(image).filter(ImageFilter.BoxBlur(radius)))


@dataclass(slots=True)
class InpaintWorker:
    """Mask + image preparation and post-process compositing for SDXL inpainting.

    Construction performs:
      1. Compute symmetric bounding box around mask, expanded by `k`.
      2. Crop image & mask to the interested area.
      3. Upscale tiny crops via injected ImageOps.upscale.
      4. Resize the crop to 1024-ceil so SDXL can process it.
      5. Binarize the resized mask at threshold 127.
      6. If `use_fill` — seed the masked region with multi-scale fooocus_fill.
    """

    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    image_ops: ImageOps
    use_fill: bool = True
    k: float = DEFAULT_RESPECTIVE_FIELD
    interested_area: InterestedArea = field(init=False)
    interested_mask: NDArray[np.uint8] = field(init=False)
    interested_image: NDArray[np.uint8] = field(init=False)
    interested_fill: NDArray[np.uint8] = field(init=False)

    def __post_init__(self) -> None:
        initial = compute_initial_bounds(self.mask > 0)
        area = expand_bounds_to_min_ratio(initial, self.mask.shape, k=self.k)
        self.interested_area = area

        top, bottom, left, right = area.as_tuple()
        cropped_image = self.image[top:bottom, left:right]
        cropped_mask = self.mask[top:bottom, left:right]

        if self.image_ops.image_shape_ceil(cropped_image) < CANONICAL_CEIL:
            cropped_image = self.image_ops.upscale(cropped_image)
        cropped_image = self.image_ops.set_image_shape_ceil(cropped_image, CANONICAL_CEIL)
        h, w = cropped_image.shape[:2]

        resized_mask = self.image_ops.resample(cropped_mask, w, h)
        self.interested_image = cropped_image
        self.interested_mask = binarize_mask(resized_mask, threshold=127)

        if self.use_fill:
            self.interested_fill = fooocus_fill(
                cropped_image,
                self.interested_mask,
                blur_fn=_default_blur,
                schedule=FOOOCUS_FILL_SCHEDULE,
            )
        else:
            self.interested_fill = cropped_image.copy()

    def post_process(self, generated: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Resample the diffusion output back to the interested area and blend."""
        top, bottom, left, right = self.interested_area.as_tuple()
        target_h = bottom - top
        target_w = right - left
        patch = self.image_ops.resample(generated, target_w, target_h)
        result = self.image.copy()
        result[top:bottom, left:right] = patch
        return color_correct(foreground=result, background=self.image, mask=self.mask)
