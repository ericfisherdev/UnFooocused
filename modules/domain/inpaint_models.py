"""Domain layer for inpaint model specs + InpaintHead (UNF-72).

Version-aware catalog of the inpaint head + patch model assets used by the
parameterized inpainting pipeline, plus the `InpaintHead` torch module that
projects `[latent_mask | latent_inpaint]` into a 320-channel feature tensor
injected into UNet input block 0.

Selecting `InpaintEngineVersion.NONE` means "Improve Detail" mode: no head,
no patch, no UNet injection. Upstream reference: HuggingFace
`lllyasviel/fooocus_inpaint`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

HEAD_URL: str = "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
HEAD_FILENAME: str = "fooocus_inpaint_head.pth"

_PATCH_BASE_URL: str = "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/"


class InpaintEngineVersion(Enum):
    """Parameterized inpainting engine version.

    `NONE` disables parameterized inpainting (Improve Detail mode).
    """

    NONE = "none"
    V1 = "v1"
    V2_5 = "v2.5"
    V2_6 = "v2.6"


@dataclass(frozen=True, slots=True)
class InpaintModelSpec:
    """One downloadable inpaint asset: remote URL + local filename."""

    filename: str
    url: str


HEAD_SPEC: InpaintModelSpec = InpaintModelSpec(filename=HEAD_FILENAME, url=HEAD_URL)

PATCH_SPECS: Mapping[InpaintEngineVersion, InpaintModelSpec] = MappingProxyType(
    {
        InpaintEngineVersion.V1: InpaintModelSpec(
            filename="inpaint.fooocus.patch",
            url=_PATCH_BASE_URL + "inpaint.fooocus.patch",
        ),
        InpaintEngineVersion.V2_5: InpaintModelSpec(
            filename="inpaint_v25.fooocus.patch",
            url=_PATCH_BASE_URL + "inpaint_v25.fooocus.patch",
        ),
        InpaintEngineVersion.V2_6: InpaintModelSpec(
            filename="inpaint_v26.fooocus.patch",
            url=_PATCH_BASE_URL + "inpaint_v26.fooocus.patch",
        ),
    }
)


def resolve_download_plan(
    version: InpaintEngineVersion,
) -> tuple[InpaintModelSpec, ...]:
    """Return the assets required for a given engine version.

    NONE returns an empty tuple — the caller should skip all parameterized
    inpainting logic. Every other version returns `(HEAD_SPEC, patch_spec)`.
    """
    if version is InpaintEngineVersion.NONE:
        return ()
    return (HEAD_SPEC, PATCH_SPECS[version])


class InpaintHead(torch.nn.Module):
    """Single-conv projection from `[latent_mask|latent_inpaint]` to 320 features.

    The learned parameter is a (320, 5, 3, 3) weight tensor applied via a
    replicate-padded 3x3 convolution so the output preserves spatial dims.
    Loaded from `fooocus_inpaint_head.pth` at runtime.
    """

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)
