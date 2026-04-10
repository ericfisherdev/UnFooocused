"""Unit tests for model/LoRA file discovery — inner-loop TDD.

Cycles:
1. _discover_files() finds .safetensors files in given paths
2. _discover_files() returns empty list for non-existent paths
3. update_model_filenames() / update_lora_filenames() refresh the lists
"""


class TestDiscoverFiles:
    """Cycle 1: _discover_files() finds .safetensors in directories."""

    def test_finds_safetensors_files(self, tmp_path):
        (tmp_path / "model_a.safetensors").write_bytes(b"\x00")
        (tmp_path / "model_b.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert "model_a.safetensors" in result
        assert "model_b.safetensors" in result

    def test_excludes_non_safetensors(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        (tmp_path / "readme.txt").write_bytes(b"\x00")
        (tmp_path / "config.json").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert "model.safetensors" in result
        assert "readme.txt" not in result
        assert "config.json" not in result

    def test_returns_sorted_filenames(self, tmp_path):
        (tmp_path / "zebra.safetensors").write_bytes(b"\x00")
        (tmp_path / "alpha.safetensors").write_bytes(b"\x00")
        (tmp_path / "mid.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert result == ["alpha.safetensors", "mid.safetensors", "zebra.safetensors"]

    def test_scans_multiple_directories(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "from_a.safetensors").write_bytes(b"\x00")
        (dir_b / "from_b.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(dir_a), str(dir_b)])
        assert "from_a.safetensors" in result
        assert "from_b.safetensors" in result

    def test_deduplicates_across_directories(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "shared.safetensors").write_bytes(b"\x00")
        (dir_b / "shared.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(dir_a), str(dir_b)])
        assert result.count("shared.safetensors") == 1

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "UPPER.SAFETENSORS").write_bytes(b"\x00")
        (tmp_path / "mixed.SafeTensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert len(result) == 2


class TestDiscoverFilesEmptyPaths:
    """Cycle 2: _discover_files() handles missing paths gracefully."""

    def test_empty_list_for_nonexistent_dir(self):
        from modules.config import _discover_files

        result = _discover_files(["/nonexistent/path/that/does/not/exist"])
        assert result == []

    def test_empty_list_for_empty_input(self):
        from modules.config import _discover_files

        result = _discover_files([])
        assert result == []

    def test_empty_list_for_dir_with_no_models(self, tmp_path):
        (tmp_path / "readme.txt").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert result == []

    def test_skips_unreadable_directory(self, tmp_path):
        """PermissionError on listdir should be logged and skipped."""
        readable_dir = tmp_path / "readable"
        unreadable_dir = tmp_path / "unreadable"
        readable_dir.mkdir()
        unreadable_dir.mkdir()
        (readable_dir / "good.safetensors").write_bytes(b"\x00")
        (unreadable_dir / "hidden.safetensors").write_bytes(b"\x00")
        unreadable_dir.chmod(0o000)

        from modules.config import _discover_files

        try:
            result = _discover_files([str(unreadable_dir), str(readable_dir)])
            # Should not crash — skips unreadable, still finds readable
            assert "good.safetensors" in result
        finally:
            unreadable_dir.chmod(0o755)


class TestRefreshFilenames:
    """Cycle 3: refresh functions return re-scanned file lists."""

    def test_refresh_model_filenames_rescans(self, tmp_path):
        from modules.config import refresh_model_filenames

        # Initially empty
        assert refresh_model_filenames([str(tmp_path)]) == []

        # Add a file
        (tmp_path / "new_model.safetensors").write_bytes(b"\x00")
        assert "new_model.safetensors" in refresh_model_filenames([str(tmp_path)])

    def test_refresh_lora_filenames_rescans(self, tmp_path):
        from modules.config import refresh_lora_filenames

        assert refresh_lora_filenames([str(tmp_path)]) == []

        (tmp_path / "lora_v1.safetensors").write_bytes(b"\x00")
        assert "lora_v1.safetensors" in refresh_lora_filenames([str(tmp_path)])
