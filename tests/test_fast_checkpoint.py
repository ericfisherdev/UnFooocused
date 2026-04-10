"""RED tests for UNF-25: fast checkpoint caching.

Tests are organized by acceptance criterion:
  AC1: config.txt accepts path_fast_checkpoints key
  AC2: Cache hit — fast storage checked first when configured
  AC3: Cache miss — atomic copy from source to fast storage
  AC4: Cache hit — load directly from fast storage with no copy
  AC5: Feature disabled — behaves as before when not configured
  AC6: Copy failure — logged and falls back to original path
  AC7: Default steps hardcoded to 30
  AC8: Existing tests continue to pass (verified by running full suite)

Following outside-in TDD (Percival): unit tests for the fast_checkpoint
domain module, then integration with config and async_worker.
"""

from __future__ import annotations

import json
import os

import pytest

# ---------------------------------------------------------------------------
# AC1: config.txt accepts path_fast_checkpoints key
# ---------------------------------------------------------------------------


class TestConfigAcceptsPathFastCheckpoints:
    """AC1: path_fast_checkpoints is a recognized config key (string path)."""

    def test_defaults_contain_path_fast_checkpoints(self):
        from modules.config import _DEFAULTS

        assert "path_fast_checkpoints" in _DEFAULTS

    def test_default_value_is_empty_string(self):
        from modules.config import _DEFAULTS

        assert _DEFAULTS["path_fast_checkpoints"] == ""

    def test_load_config_returns_path_fast_checkpoints(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert "path_fast_checkpoints" in result

    def test_config_txt_can_set_path_fast_checkpoints(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"path_fast_checkpoints": "/fast/nvme"}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["path_fast_checkpoints"] == "/fast/nvme"

    def test_app_config_has_path_fast_checkpoints_field(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert hasattr(cfg, "path_fast_checkpoints")

    def test_app_config_path_fast_checkpoints_is_string(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg.path_fast_checkpoints, str)

    def test_app_config_path_fast_checkpoints_defaults_to_empty(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert cfg.path_fast_checkpoints == ""


# ---------------------------------------------------------------------------
# AC2 & AC4: Cache hit — fast storage checked first, loaded directly
# ---------------------------------------------------------------------------


class TestFastCheckpointCacheHit:
    """AC2/AC4: When fast storage has the file, use it directly with no copy."""

    @pytest.fixture
    def slow_dir(self, tmp_path):
        d = tmp_path / "slow"
        d.mkdir()
        return d

    @pytest.fixture
    def fast_dir(self, tmp_path):
        d = tmp_path / "fast"
        d.mkdir()
        return d

    @pytest.fixture
    def checkpoint_on_slow(self, slow_dir):
        ckpt = slow_dir / "model.safetensors"
        ckpt.write_bytes(b"\x00" * 1024)
        return ckpt

    def test_returns_fast_path_when_cached(self, slow_dir, fast_dir, checkpoint_on_slow):
        # Pre-populate fast storage
        fast_file = fast_dir / "model.safetensors"
        fast_file.write_bytes(b"\x01" * 512)

        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path=str(fast_dir))
        assert result == str(fast_file)

    def test_does_not_overwrite_cached_file(self, slow_dir, fast_dir, checkpoint_on_slow):
        # Pre-populate fast storage with different content
        fast_file = fast_dir / "model.safetensors"
        fast_file.write_bytes(b"\x01" * 512)

        from modules.fast_checkpoint import resolve_checkpoint_path

        resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path=str(fast_dir))
        # Verify content was NOT overwritten
        assert fast_file.stat().st_size == 512


# ---------------------------------------------------------------------------
# AC3: Cache miss — atomic copy from source to fast storage
# ---------------------------------------------------------------------------


class TestFastCheckpointCacheMiss:
    """AC3: On cache miss, copy atomically from source to fast storage."""

    @pytest.fixture
    def slow_dir(self, tmp_path):
        d = tmp_path / "slow"
        d.mkdir()
        return d

    @pytest.fixture
    def fast_dir(self, tmp_path):
        d = tmp_path / "fast"
        d.mkdir()
        return d

    @pytest.fixture
    def checkpoint_on_slow(self, slow_dir):
        ckpt = slow_dir / "model.safetensors"
        ckpt.write_bytes(b"\x00" * 1024)
        return ckpt

    def test_copies_to_fast_on_first_use(self, slow_dir, fast_dir, checkpoint_on_slow):
        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path=str(fast_dir))
        expected_fast = str(fast_dir / "model.safetensors")
        assert result == expected_fast
        assert os.path.isfile(expected_fast)

    def test_copied_file_matches_original_size(self, slow_dir, fast_dir, checkpoint_on_slow):
        from modules.fast_checkpoint import resolve_checkpoint_path

        resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path=str(fast_dir))
        fast_file = fast_dir / "model.safetensors"
        assert fast_file.stat().st_size == 1024

    def test_no_tmp_file_remains_after_copy(self, slow_dir, fast_dir, checkpoint_on_slow):
        from modules.fast_checkpoint import resolve_checkpoint_path

        resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path=str(fast_dir))
        tmp_file = fast_dir / "model.safetensors.tmp"
        assert not tmp_file.exists()

    def test_subdirectory_checkpoint_preserves_structure(self, slow_dir, fast_dir):
        subdir = slow_dir / "sdxl"
        subdir.mkdir()
        (subdir / "model.safetensors").write_bytes(b"\x00" * 256)

        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("sdxl/model.safetensors", [str(slow_dir)], fast_path=str(fast_dir))
        expected = str(fast_dir / "sdxl" / "model.safetensors")
        assert result == expected
        assert os.path.isfile(expected)


# ---------------------------------------------------------------------------
# AC5: Feature disabled — normal behavior when not configured
# ---------------------------------------------------------------------------


class TestFastCheckpointDisabled:
    """AC5: When path_fast_checkpoints is empty or None, load from original paths."""

    @pytest.fixture
    def slow_dir(self, tmp_path):
        d = tmp_path / "slow"
        d.mkdir()
        return d

    @pytest.fixture
    def checkpoint_on_slow(self, slow_dir):
        ckpt = slow_dir / "model.safetensors"
        ckpt.write_bytes(b"\x00" * 1024)
        return ckpt

    def test_returns_original_path_when_fast_path_is_none(self, slow_dir, checkpoint_on_slow):
        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path=None)
        assert result == str(checkpoint_on_slow)

    def test_returns_original_path_when_fast_path_is_empty(self, slow_dir, checkpoint_on_slow):
        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("model.safetensors", [str(slow_dir)], fast_path="")
        assert result == str(checkpoint_on_slow)

    def test_returns_constructed_path_when_file_not_found(self, slow_dir):
        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("nonexistent.safetensors", [str(slow_dir)], fast_path=None)
        expected = os.path.abspath(os.path.realpath(os.path.join(str(slow_dir), "nonexistent.safetensors")))
        assert result == expected


# ---------------------------------------------------------------------------
# AC6: Copy failure — falls back to original path
# ---------------------------------------------------------------------------


class TestFastCheckpointCopyFailure:
    """AC6: Copy failures are logged and fall back to loading from original path."""

    @pytest.fixture
    def slow_dir(self, tmp_path):
        d = tmp_path / "slow"
        d.mkdir()
        return d

    @pytest.fixture
    def checkpoint_on_slow(self, slow_dir):
        ckpt = slow_dir / "model.safetensors"
        ckpt.write_bytes(b"\x00" * 1024)
        return ckpt

    def test_falls_back_on_unwritable_fast_path(self, slow_dir, checkpoint_on_slow):
        from modules.fast_checkpoint import resolve_checkpoint_path

        # Use a path that cannot be written to
        result = resolve_checkpoint_path(
            "model.safetensors",
            [str(slow_dir)],
            fast_path="/proc/fake_nonexistent_dir",
        )
        assert result == str(checkpoint_on_slow)

    def test_falls_back_when_checkpoint_not_found_anywhere(self, slow_dir, tmp_path):
        from modules.fast_checkpoint import resolve_checkpoint_path

        fast_dir = str(tmp_path / "fast_nonexistent")
        result = resolve_checkpoint_path(
            "nonexistent.safetensors",
            [str(slow_dir)],
            fast_path=fast_dir,
        )
        # Should return constructed path from first checkpoint folder
        expected = os.path.abspath(os.path.realpath(os.path.join(str(slow_dir), "nonexistent.safetensors")))
        assert result == expected


# ---------------------------------------------------------------------------
# AC7: Default steps hardcoded to 30
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_config_singleton():
    """Reset the config singleton before and after each test."""
    from modules.config import reset_config

    reset_config()
    yield
    reset_config()


class TestDefaultStepsHardcodedTo30:
    """AC7: Default generation steps are 30 regardless of performance preset logic."""

    def test_defaults_dict_has_steps_30(self):
        from modules.config import _DEFAULTS

        assert _DEFAULTS["default_steps"] == 30

    def test_app_config_default_steps_is_30_without_config_file(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert cfg.default_steps == 30

    def test_app_config_default_steps_is_30_with_default_config(self):
        from modules.config import get_config

        cfg = get_config()
        assert cfg.default_steps == 30

    def test_config_txt_can_still_override_steps(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_steps": 50}))
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=config_file)
        cfg = AppConfig.from_dict(raw)
        assert cfg.default_steps == 50


# ---------------------------------------------------------------------------
# AC3 (path traversal safety)
# ---------------------------------------------------------------------------


class TestFastCheckpointPathTraversal:
    """Ensure path traversal attacks are blocked."""

    @pytest.fixture
    def slow_dir(self, tmp_path):
        d = tmp_path / "slow"
        d.mkdir()
        return d

    @pytest.fixture
    def fast_dir(self, tmp_path):
        d = tmp_path / "fast"
        d.mkdir()
        return d

    def test_rejects_absolute_checkpoint_name(self, slow_dir, fast_dir):
        from modules.fast_checkpoint import resolve_checkpoint_path

        # Absolute path should be treated as unsafe for fast cache
        result = resolve_checkpoint_path("/etc/passwd", [str(slow_dir)], fast_path=str(fast_dir))
        # Should fall back, not write to fast cache, and stay within slow_dir
        assert str(fast_dir) not in result
        resolved = os.path.realpath(result)
        assert resolved.startswith(str(slow_dir))

    def test_rejects_parent_directory_traversal(self, slow_dir, fast_dir):
        from modules.fast_checkpoint import resolve_checkpoint_path

        result = resolve_checkpoint_path("../../../etc/passwd", [str(slow_dir)], fast_path=str(fast_dir))
        assert str(fast_dir) not in result
        resolved = os.path.realpath(result)
        assert resolved.startswith(str(slow_dir))
