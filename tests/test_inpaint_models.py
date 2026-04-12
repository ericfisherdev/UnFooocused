"""Unit tests for UNF-72 inpaint model downloading + patching.

RED-phase tests encoding acceptance criteria:
- Version-aware download plan resolves head + version-specific patch.
- `InpaintEngineVersion.NONE` skips all parameterized inpainting.
- Downloader is called only for missing files (caching).
- InpaintHead forward preserves spatial dimensions and produces 320 channels.
- Returned model paths resolve under the configured models directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path
from modules.domain.inpaint_models import (
    HEAD_FILENAME,
    PATCH_SPECS,
    InpaintEngineVersion,
    InpaintHead,
    InpaintModelSpec,
    resolve_download_plan,
)
from modules.services.inpaint_model_loader import (
    InpaintModelLoader,
    InpaintModelPaths,
)


class _FakeDownloader:
    """Records fetch calls and writes a sentinel file at the destination."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def fetch(self, url: str, dest: Path) -> Path:
        self.calls.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"FAKE_MODEL_DATA")
        return dest


class TestInpaintEngineVersion:
    def test_has_four_versions_including_none(self) -> None:
        names = {v.name for v in InpaintEngineVersion}
        assert names == {"NONE", "V1", "V2_5", "V2_6"}


class TestInpaintModelSpec:
    def test_is_frozen_value_object(self) -> None:
        spec = InpaintModelSpec(filename="x.pth", url="https://example/x.pth")
        with pytest.raises((AttributeError, TypeError)):
            spec.filename = "y.pth"  # type: ignore[misc]


class TestResolveDownloadPlan:
    def test_none_returns_empty_plan(self) -> None:
        assert resolve_download_plan(InpaintEngineVersion.NONE) == ()

    def test_v1_returns_head_and_v1_patch(self) -> None:
        plan = resolve_download_plan(InpaintEngineVersion.V1)
        filenames = [spec.filename for spec in plan]
        assert HEAD_FILENAME in filenames
        assert PATCH_SPECS[InpaintEngineVersion.V1].filename in filenames

    def test_v2_5_returns_head_and_v2_5_patch(self) -> None:
        plan = resolve_download_plan(InpaintEngineVersion.V2_5)
        filenames = [spec.filename for spec in plan]
        assert HEAD_FILENAME in filenames
        assert PATCH_SPECS[InpaintEngineVersion.V2_5].filename in filenames

    def test_v2_6_returns_head_and_v2_6_patch(self) -> None:
        plan = resolve_download_plan(InpaintEngineVersion.V2_6)
        filenames = [spec.filename for spec in plan]
        assert HEAD_FILENAME in filenames
        assert PATCH_SPECS[InpaintEngineVersion.V2_6].filename in filenames

    def test_patch_specs_are_version_unique(self) -> None:
        filenames = {
            v: PATCH_SPECS[v].filename
            for v in (
                InpaintEngineVersion.V1,
                InpaintEngineVersion.V2_5,
                InpaintEngineVersion.V2_6,
            )
        }
        assert len(set(filenames.values())) == 3


class TestInpaintHead:
    def test_parameter_shape_is_320x5x3x3(self) -> None:
        head = InpaintHead()
        assert head.head.shape == (320, 5, 3, 3)

    def test_forward_preserves_spatial_dimensions(self) -> None:
        head = InpaintHead()
        x = torch.zeros(1, 5, 8, 8)
        y = head(x)
        assert y.shape[-2:] == (8, 8)

    def test_forward_produces_320_output_channels(self) -> None:
        head = InpaintHead()
        x = torch.zeros(2, 5, 16, 16)
        y = head(x)
        assert y.shape[0] == 2
        assert y.shape[1] == 320


class TestInpaintModelLoader:
    def test_none_version_skips_download(self, tmp_path: Path) -> None:
        downloader = _FakeDownloader()
        loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
        paths = loader.ensure_models(InpaintEngineVersion.NONE)
        assert paths.head_path is None
        assert paths.patch_path is None
        assert downloader.calls == []

    def test_v1_downloads_head_and_patch(self, tmp_path: Path) -> None:
        downloader = _FakeDownloader()
        loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
        paths = loader.ensure_models(InpaintEngineVersion.V1)
        assert paths.head_path == tmp_path / HEAD_FILENAME
        assert paths.patch_path == tmp_path / PATCH_SPECS[InpaintEngineVersion.V1].filename
        assert len(downloader.calls) == 2

    def test_v2_5_downloads_v2_5_patch(self, tmp_path: Path) -> None:
        downloader = _FakeDownloader()
        loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
        paths = loader.ensure_models(InpaintEngineVersion.V2_5)
        assert paths.patch_path == tmp_path / PATCH_SPECS[InpaintEngineVersion.V2_5].filename

    def test_v2_6_downloads_v2_6_patch(self, tmp_path: Path) -> None:
        downloader = _FakeDownloader()
        loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
        paths = loader.ensure_models(InpaintEngineVersion.V2_6)
        assert paths.patch_path == tmp_path / PATCH_SPECS[InpaintEngineVersion.V2_6].filename

    def test_cached_files_are_not_redownloaded(self, tmp_path: Path) -> None:
        (tmp_path / HEAD_FILENAME).write_bytes(b"cached_head")
        (tmp_path / PATCH_SPECS[InpaintEngineVersion.V1].filename).write_bytes(b"cached_patch")
        downloader = _FakeDownloader()
        loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
        paths = loader.ensure_models(InpaintEngineVersion.V1)
        assert paths.head_path == tmp_path / HEAD_FILENAME
        assert paths.patch_path == tmp_path / PATCH_SPECS[InpaintEngineVersion.V1].filename
        assert downloader.calls == []

    def test_partially_cached_downloads_only_missing(self, tmp_path: Path) -> None:
        (tmp_path / HEAD_FILENAME).write_bytes(b"cached_head")
        downloader = _FakeDownloader()
        loader = InpaintModelLoader(models_dir=tmp_path, downloader=downloader)
        loader.ensure_models(InpaintEngineVersion.V2_6)
        assert len(downloader.calls) == 1
        assert downloader.calls[0][1].name == PATCH_SPECS[InpaintEngineVersion.V2_6].filename


class TestInpaintModelPaths:
    def test_is_frozen_value_object(self, tmp_path: Path) -> None:
        paths = InpaintModelPaths(head_path=None, patch_path=None)
        with pytest.raises((AttributeError, TypeError)):
            paths.head_path = tmp_path / "x"  # type: ignore[misc]
