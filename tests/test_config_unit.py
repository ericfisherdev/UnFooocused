"""Unit tests for modules.config — inner-loop TDD.

Each test drives one behavior. Tests are ordered by TDD cycle:
1. load_config() returns defaults when no config.txt
2. load_config() reads a valid config.txt
3. load_config() uses defaults for missing keys
4. load_config() raises ValueError on malformed JSON
"""

import json

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
            "default_model", "default_refiner", "default_refiner_switch",
            "default_performance", "default_aspect_ratio",
            "available_aspect_ratios", "default_image_number",
            "default_max_image_number", "default_output_format",
            "default_prompt", "default_prompt_negative", "default_styles",
            "default_cfg_scale", "default_sample_sharpness",
            "default_sampler", "default_scheduler",
            "default_loras", "default_loras_min_weight",
            "default_loras_max_weight", "default_max_lora_number",
        ]
        missing = [k for k in expected_keys if k not in result]
        assert not missing, f"Missing keys: {missing}"


class TestLoadConfigFromFile:
    """Cycle 2: load_config() reads and parses a valid config.txt."""

    def test_reads_custom_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text(json.dumps({
            "default_model": "my_custom_model.safetensors",
        }))
        from modules.config import load_config

        result = load_config()
        assert result["default_model"] == "my_custom_model.safetensors"

    def test_reads_custom_cfg_scale(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text(json.dumps({
            "default_cfg_scale": 12.0,
        }))
        from modules.config import load_config

        result = load_config()
        assert result["default_cfg_scale"] == 12.0

    def test_reads_custom_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text(json.dumps({
            "paths_checkpoints": ["/custom/models"],
            "paths_loras": ["/custom/loras"],
        }))
        from modules.config import load_config

        result = load_config()
        assert result["paths_checkpoints"] == ["/custom/models"]
        assert result["paths_loras"] == ["/custom/loras"]


class TestLoadConfigPartial:
    """Cycle 3: load_config() uses defaults for missing keys in partial file."""

    def test_missing_keys_get_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text(json.dumps({
            "default_model": "partial.safetensors",
        }))
        from modules.config import load_config

        result = load_config()
        # Explicitly set
        assert result["default_model"] == "partial.safetensors"
        # Should still have defaults for everything else
        assert isinstance(result["default_cfg_scale"], (int, float))
        assert isinstance(result["default_sampler"], str)
        assert result["default_sampler"] != ""
        assert isinstance(result["available_aspect_ratios"], list)
        assert len(result["available_aspect_ratios"]) >= 3


class TestLoadConfigMalformed:
    """Cycle 4: load_config() raises ValueError on malformed JSON."""

    def test_raises_on_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text("{ not valid json !!!")
        from modules.config import load_config

        with pytest.raises(ValueError, match="config.txt"):
            load_config()

    def test_raises_on_non_dict_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text(json.dumps([1, 2, 3]))
        from modules.config import load_config

        with pytest.raises(ValueError, match="config.txt"):
            load_config()
