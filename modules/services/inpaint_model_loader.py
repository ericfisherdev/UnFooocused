"""Service layer for inpaint model downloading (UNF-72).

Orchestrates on-demand download + caching of the inpaint head and
version-specific patch model to a local `models/inpaint/` directory, using
an injected `ModelDownloader` so tests can substitute a fake without
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — runtime-referenced by dataclass field and Protocol signature
from typing import Protocol

from modules.domain.inpaint_models import (
    HEAD_FILENAME,
    PATCH_SPECS,
    InpaintEngineVersion,
    InpaintModelSpec,
    resolve_download_plan,
)


class ModelDownloader(Protocol):
    """Structural interface for fetching a remote asset to a local path."""

    def fetch(self, url: str, dest: Path) -> Path: ...


@dataclass(frozen=True, slots=True)
class InpaintModelPaths:
    """Resolved local paths for inpaint head + patch after ensure_models().

    Both fields are None when `InpaintEngineVersion.NONE` was requested
    (Improve Detail mode — parameterized inpainting disabled).
    """

    head_path: Path | None
    patch_path: Path | None


@dataclass(frozen=True, slots=True)
class InpaintModelLoader:
    """Ensures inpaint models are present on disk, downloading only missing files."""

    models_dir: Path
    downloader: ModelDownloader

    def ensure_models(self, version: InpaintEngineVersion) -> InpaintModelPaths:
        plan = resolve_download_plan(version)
        if not plan:
            return InpaintModelPaths(head_path=None, patch_path=None)

        for spec in plan:
            self._fetch_if_missing(spec)

        head_path = self.models_dir / HEAD_FILENAME
        patch_spec = PATCH_SPECS[version]
        patch_path = self.models_dir / patch_spec.filename
        return InpaintModelPaths(head_path=head_path, patch_path=patch_path)

    def _fetch_if_missing(self, spec: InpaintModelSpec) -> Path:
        dest = self.models_dir / spec.filename
        if dest.exists():
            return dest
        return self.downloader.fetch(spec.url, dest)
