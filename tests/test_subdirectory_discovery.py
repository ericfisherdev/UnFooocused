"""RED tests for subdirectory file discovery — UNF-26.

Acceptance Criteria:
1. Checkpoint dropdown shows all .safetensors files from paths_checkpoints
   (including subdirectories).
2. default_model from config is pre-selected in checkpoint dropdown on page load.
3. Selecting a different checkpoint updates the generation pipeline.
4. Output format buttons visually indicate which format is selected on load
   (webp by default).
5. LoRA mini-picker shows available .safetensors files from paths_loras.
6. Adding a LoRA from the picker adds it to active slots with correct name and
   trigger words.
7. All existing tests continue to pass.

This file covers the Python-testable backend concerns: criteria 1, 2, 3, 5.
Frontend-only criteria (4, 6) are covered by manual verification against the
template/JS changes and noted in the PR checklist.
"""

import dataclasses

import pytest

# ---------------------------------------------------------------------------
# AC-1: Checkpoint dropdown shows ALL .safetensors files including subdirs
# ---------------------------------------------------------------------------


class TestDiscoverFilesRecursive:
    """_discover_files must walk subdirectories and return relative paths."""

    def test_finds_files_in_subdirectories(self, tmp_path):
        """Files in subdirs like pony/model.safetensors must be discovered."""
        sub = tmp_path / "pony"
        sub.mkdir()
        (sub / "pony_xl.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert "pony/pony_xl.safetensors" in result

    def test_finds_files_in_deeply_nested_subdirectories(self, tmp_path):
        """Files multiple levels deep must also be discovered."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep_model.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert "a/b/c/deep_model.safetensors" in result

    def test_mixes_root_and_subdirectory_files(self, tmp_path):
        """Both root-level and subdirectory files appear in results."""
        (tmp_path / "root_model.safetensors").write_bytes(b"\x00")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "sub_model.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        assert "root_model.safetensors" in result
        assert "sub/sub_model.safetensors" in result

    def test_deduplicates_same_relative_path_across_roots(self, tmp_path):
        """Same relative path from different roots is deduplicated."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "shared.safetensors").write_bytes(b"\x00")
        (dir_b / "shared.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(dir_a), str(dir_b)])
        assert result.count("shared.safetensors") == 1

    def test_returns_forward_slash_paths_on_all_platforms(self, tmp_path):
        """Relative paths must use forward slashes, not backslashes."""
        sub = tmp_path / "category"
        sub.mkdir()
        (sub / "model.safetensors").write_bytes(b"\x00")

        from modules.config import _discover_files

        result = _discover_files([str(tmp_path)])
        for path in result:
            assert "\\" not in path, f"Backslash found in path: {path}"


# ---------------------------------------------------------------------------
# AC-1 + AC-5: API endpoints reflect recursive discovery
# ---------------------------------------------------------------------------


class TestApiModelsWithSubdirectories:
    """GET /api/models must return checkpoint and LoRA files from subdirs."""

    @pytest.fixture
    def nested_models_dir(self, tmp_path):
        """Create a directory tree with models at various depths."""
        ckpt_root = tmp_path / "checkpoints"
        ckpt_root.mkdir()
        (ckpt_root / "base_model.safetensors").write_bytes(b"\x00")
        pony = ckpt_root / "pony"
        pony.mkdir()
        (pony / "pony_v6.safetensors").write_bytes(b"\x00")

        lora_root = tmp_path / "loras"
        lora_root.mkdir()
        (lora_root / "detail.safetensors").write_bytes(b"\x00")
        chars = lora_root / "characters"
        chars.mkdir()
        (chars / "char_lora.safetensors").write_bytes(b"\x00")

        return {"checkpoints": str(ckpt_root), "loras": str(lora_root)}

    @pytest.fixture
    def client_with_nested(self, nested_models_dir):
        from modules.config import (
            get_config,
            refresh_lora_filenames,
            refresh_model_filenames,
            reset_config,
            set_config,
        )

        base = get_config()
        custom = dataclasses.replace(
            base,
            paths_checkpoints=(nested_models_dir["checkpoints"],),
            paths_loras=(nested_models_dir["loras"],),
            model_filenames=tuple(refresh_model_filenames([nested_models_dir["checkpoints"]])),
            lora_filenames=tuple(refresh_lora_filenames([nested_models_dir["loras"]])),
        )
        set_config(custom)

        from fastapi.testclient import TestClient
        from ui.app import app

        try:
            yield TestClient(app)
        finally:
            reset_config()

    def test_checkpoints_include_subdirectory_models(self, client_with_nested):
        data = client_with_nested.get("/api/models").json()
        assert "pony/pony_v6.safetensors" in data["checkpoints"]

    def test_checkpoints_include_root_models(self, client_with_nested):
        data = client_with_nested.get("/api/models").json()
        assert "base_model.safetensors" in data["checkpoints"]

    def test_loras_include_subdirectory_files(self, client_with_nested):
        data = client_with_nested.get("/api/models").json()
        assert "characters/char_lora.safetensors" in data["loras"]

    def test_loras_include_root_files(self, client_with_nested):
        data = client_with_nested.get("/api/models").json()
        assert "detail.safetensors" in data["loras"]


# ---------------------------------------------------------------------------
# AC-2: default_model from config is in the checkpoint list
# ---------------------------------------------------------------------------


class TestDefaultModelInCheckpointList:
    """The default_model value from /api/config should appear in /api/models checkpoints."""

    @pytest.fixture
    def client_with_default_model(self, tmp_path):
        """Set up a config where default_model matches an actual discovered file."""
        ckpt_root = tmp_path / "checkpoints"
        ckpt_root.mkdir()
        (ckpt_root / "juggernautXL_v8Rundiffusion.safetensors").write_bytes(b"\x00")
        (ckpt_root / "other_model.safetensors").write_bytes(b"\x00")

        from modules.config import (
            get_config,
            refresh_model_filenames,
            reset_config,
            set_config,
        )

        base = get_config()
        custom = dataclasses.replace(
            base,
            default_base_model_name="juggernautXL_v8Rundiffusion.safetensors",
            paths_checkpoints=(str(ckpt_root),),
            model_filenames=tuple(refresh_model_filenames([str(ckpt_root)])),
        )
        set_config(custom)

        from fastapi.testclient import TestClient
        from ui.app import app

        try:
            yield TestClient(app)
        finally:
            reset_config()

    def test_default_model_appears_in_model_list(self, client_with_default_model):
        config_data = client_with_default_model.get("/api/config").json()
        models_data = client_with_default_model.get("/api/models").json()
        assert config_data["default_model"] in models_data["checkpoints"]


# ---------------------------------------------------------------------------
# AC-3: Selecting a checkpoint updates generation pipeline
# ---------------------------------------------------------------------------


class TestCheckpointSelectionUpdatesGeneration:
    """POST /api/generate should use the base_model_name from the request body."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from ui.app import app

        return TestClient(app)

    def test_generate_uses_submitted_base_model(self, client):
        """The args list built by _build_generate_args must include the
        submitted base_model_name, not just the config default."""
        from ui.app import _build_generate_args

        custom_model = "custom_checkpoint.safetensors"
        args = _build_generate_args({"base_model_name": custom_model})
        assert custom_model in args, "The submitted base_model_name should appear in the args list"

    def test_generate_falls_back_to_config_default(self):
        """When no base_model_name is submitted, the config default is used."""
        from modules.config import get_config
        from ui.app import _build_generate_args

        cfg = get_config()
        args = _build_generate_args({})
        assert cfg.default_base_model_name in args


# ---------------------------------------------------------------------------
# AC-4: Output format default_output_format exposed via /api/config
# ---------------------------------------------------------------------------


class TestOutputFormatConfig:
    """The /api/config endpoint must expose default_output_format so the
    frontend can initialize the format radio buttons correctly."""

    @pytest.fixture
    def client_with_webp_default(self, tmp_path):
        from modules.config import get_config, reset_config, set_config

        base = get_config()
        custom = dataclasses.replace(base, default_output_format="webp")
        set_config(custom)

        from fastapi.testclient import TestClient
        from ui.app import app

        try:
            yield TestClient(app)
        finally:
            reset_config()

    def test_config_returns_default_output_format(self, client_with_webp_default):
        data = client_with_webp_default.get("/api/config").json()
        assert data["default_output_format"] == "webp"

    def test_default_output_format_field_exists(self):
        from fastapi.testclient import TestClient
        from ui.app import app

        client = TestClient(app)
        data = client.get("/api/config").json()
        assert "default_output_format" in data
        assert isinstance(data["default_output_format"], str)
        assert data["default_output_format"] != ""
