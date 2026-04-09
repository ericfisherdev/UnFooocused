"""Functional tests for config file loading behavior.

Outer-loop TDD (Percival): verifies config loads from file and falls
back to defaults when absent.
"""

import json

import pytest


class TestConfigFromFile:
    """Config system loads values from config.txt when present."""

    def test_loads_custom_value_from_config_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({
            "default_model": "custom_model.safetensors",
            "default_cfg_scale": 9.5,
        }))

        monkeypatch.chdir(tmp_path)

        # Force re-import to pick up the new config.txt
        import importlib
        import modules.config as config

        importlib.reload(config)

        assert config.default_base_model_name == "custom_model.safetensors"
        assert config.default_cfg_scale == 9.5

    def test_partial_config_uses_defaults_for_missing_keys(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({
            "default_model": "partial_model.safetensors",
        }))

        monkeypatch.chdir(tmp_path)

        import importlib
        import modules.config as config

        importlib.reload(config)

        # Explicitly set field should be custom
        assert config.default_base_model_name == "partial_model.safetensors"
        # Unset fields should have sensible defaults
        assert isinstance(config.default_cfg_scale, (int, float))
        assert isinstance(config.default_sampler, str)
        assert config.default_sampler != ""


class TestConfigDefaults:
    """Config system provides sensible defaults when no config.txt exists."""

    def test_defaults_when_no_config_file(self, tmp_path, monkeypatch):
        # Ensure no config.txt exists
        monkeypatch.chdir(tmp_path)

        import importlib
        import modules.config as config

        importlib.reload(config)

        assert isinstance(config.default_base_model_name, str)
        assert isinstance(config.default_performance, str)
        assert isinstance(config.default_aspect_ratio, str)
        assert isinstance(config.available_aspect_ratios, list)
        assert len(config.available_aspect_ratios) >= 3
        assert isinstance(config.default_cfg_scale, (int, float))
        assert isinstance(config.default_sample_sharpness, (int, float))
        assert isinstance(config.default_sampler, str)
        assert isinstance(config.default_scheduler, str)

    def test_model_filenames_is_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        import importlib
        import modules.config as config

        importlib.reload(config)

        assert isinstance(config.model_filenames, list)
        assert isinstance(config.lora_filenames, list)
