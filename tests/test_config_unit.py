"""Unit tests for modules.config — inner-loop TDD.

Each test drives one behavior. Tests are ordered by TDD cycle:
1. load_config() returns defaults when no config.txt
2. load_config() reads a valid config.txt
3. load_config() uses defaults for missing keys
4. load_config() raises ValueError on malformed JSON
5. load_config() resolves config.txt relative to project root
6. load_config() validates critical override types
"""

import json
import os

import pytest


class TestLoadConfigDefaults:
    """Cycle 1: load_config() returns all required keys with defaults."""

    def test_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from modules.config import load_config

        result = load_config()
        assert isinstance(result, dict)

    def test_contains_default_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from modules.config import load_config

        result = load_config()
        assert "default_model" in result
        assert isinstance(result["default_model"], str)

    def test_contains_paths_checkpoints(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from modules.config import load_config

        result = load_config()
        assert "paths_checkpoints" in result
        assert isinstance(result["paths_checkpoints"], list)

    def test_contains_paths_loras(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from modules.config import load_config

        result = load_config()
        assert "paths_loras" in result
        assert isinstance(result["paths_loras"], list)

    def test_contains_path_outputs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from modules.config import load_config

        result = load_config()
        assert "path_outputs" in result
        assert isinstance(result["path_outputs"], str)

    def test_contains_all_generation_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from modules.config import load_config

        result = load_config()
        expected_keys = [
            "default_model",
            "default_refiner",
            "default_refiner_switch",
            "default_performance",
            "default_aspect_ratio",
            "available_aspect_ratios",
            "default_image_number",
            "default_max_image_number",
            "default_output_format",
            "default_prompt",
            "default_prompt_negative",
            "default_styles",
            "default_cfg_scale",
            "default_sample_sharpness",
            "default_sampler",
            "default_scheduler",
            "default_loras",
            "default_loras_min_weight",
            "default_loras_max_weight",
            "default_max_lora_number",
        ]
        missing = [k for k in expected_keys if k not in result]
        assert not missing, f"Missing keys: {missing}"


class TestLoadConfigFromFile:
    """Cycle 2: load_config() reads and parses a valid config.txt."""

    def test_reads_custom_model(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "my_custom_model.safetensors",
                }
            )
        )
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_model"] == "my_custom_model.safetensors"

    def test_reads_custom_cfg_scale(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_cfg_scale": 12.0,
                }
            )
        )
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_cfg_scale"] == pytest.approx(12.0)

    def test_reads_custom_paths(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "paths_checkpoints": ["/custom/models"],
                    "paths_loras": ["/custom/loras"],
                }
            )
        )
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["paths_checkpoints"] == ["/custom/models"]
        assert result["paths_loras"] == ["/custom/loras"]


class TestLoadConfigPartial:
    """Cycle 3: load_config() uses defaults for missing keys in partial file."""

    def test_missing_keys_get_defaults(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "partial.safetensors",
                }
            )
        )
        from modules.config import load_config

        result = load_config(config_path=config_file)
        # Explicitly set
        assert result["default_model"] == "partial.safetensors"
        # Should still have defaults for everything else
        assert isinstance(result["default_cfg_scale"], int | float)
        assert isinstance(result["default_sampler"], str)
        assert result["default_sampler"] != ""
        assert isinstance(result["available_aspect_ratios"], list)
        assert len(result["available_aspect_ratios"]) >= 3


class TestLoadConfigMalformed:
    """Cycle 4: load_config() raises ValueError on malformed JSON."""

    def test_raises_on_invalid_json(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text("{ not valid json !!!")
        from modules.config import load_config

        with pytest.raises(ValueError, match=r"config\.txt"):
            load_config(config_path=config_file)

    def test_raises_on_non_dict_json(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps([1, 2, 3]))
        from modules.config import load_config

        with pytest.raises(ValueError, match=r"config\.txt"):
            load_config(config_path=config_file)


class TestLoadConfigStableBasePath:
    """Cycle 5: load_config() resolves config.txt from project root, not CWD."""

    def test_finds_config_when_cwd_differs(self, tmp_path, monkeypatch):
        """Config should load from project root even when CWD is elsewhere."""
        # Change CWD to a temp dir that has no config.txt
        monkeypatch.chdir(tmp_path)

        from modules.config import _DEFAULT_CONFIG_PATH

        # The default path should be absolute, not relative
        assert os.path.isabs(str(_DEFAULT_CONFIG_PATH))

    def test_default_config_path_is_project_root(self):
        from modules.config import _DEFAULT_CONFIG_PATH

        # Should resolve to <project_root>/config.txt
        assert str(_DEFAULT_CONFIG_PATH).endswith("config.txt")
        # Should be next to the modules/ directory (one level up)
        config_dir = os.path.dirname(str(_DEFAULT_CONFIG_PATH))
        assert os.path.isdir(os.path.join(config_dir, "modules"))


class TestLoadConfigValidation:
    """Cycle 6: load_config() validates critical override types."""

    def test_raises_on_null_paths_checkpoints(self, tmp_path):
        (tmp_path / "config.txt").write_text(
            json.dumps(
                {
                    "paths_checkpoints": None,
                }
            )
        )
        from modules.config import load_config

        with pytest.raises(ValueError, match="paths_checkpoints"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_null_paths_loras(self, tmp_path):
        (tmp_path / "config.txt").write_text(
            json.dumps(
                {
                    "paths_loras": None,
                }
            )
        )
        from modules.config import load_config

        with pytest.raises(ValueError, match="paths_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_inverted_lora_weight_range(self, tmp_path):
        (tmp_path / "config.txt").write_text(
            json.dumps(
                {
                    "default_loras_min_weight": 5.0,
                    "default_loras_max_weight": -5.0,
                }
            )
        )
        from modules.config import load_config

        with pytest.raises(ValueError, match="loras_min_weight"):
            load_config(config_path=tmp_path / "config.txt")
