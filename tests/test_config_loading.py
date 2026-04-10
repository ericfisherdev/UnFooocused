"""Functional tests for config file loading behavior.

Outer-loop TDD (Percival): verifies config loads from file and falls
back to defaults when absent.
"""

import json

import pytest
from modules.config import load_config


class TestConfigFromFile:
    """Config system loads values from config.txt when present."""

    def test_loads_custom_value_from_config_file(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "custom_model.safetensors",
                    "default_cfg_scale": 9.5,
                }
            )
        )

        result = load_config(config_path=config_file)
        assert result["default_model"] == "custom_model.safetensors"
        assert result["default_cfg_scale"] == pytest.approx(9.5)

    def test_partial_config_uses_defaults_for_missing_keys(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "partial_model.safetensors",
                }
            )
        )

        result = load_config(config_path=config_file)
        # Explicitly set field should be custom
        assert result["default_model"] == "partial_model.safetensors"
        # Unset fields should have sensible defaults
        assert isinstance(result["default_cfg_scale"], int | float)
        assert isinstance(result["default_sampler"], str)
        assert result["default_sampler"] != ""


class TestConfigDefaults:
    """Config system provides sensible defaults when no config.txt exists."""

    def test_defaults_when_no_config_file(self, tmp_path):
        # Point at a nonexistent config file
        result = load_config(config_path=tmp_path / "config.txt")

        assert isinstance(result["default_model"], str)
        assert isinstance(result["default_performance"], str)
        assert isinstance(result["default_aspect_ratio"], str)
        assert isinstance(result["available_aspect_ratios"], list)
        assert len(result["available_aspect_ratios"]) >= 3
        assert isinstance(result["default_cfg_scale"], int | float)
        assert isinstance(result["default_sample_sharpness"], int | float)
        assert isinstance(result["default_sampler"], str)
        assert isinstance(result["default_scheduler"], str)

    def test_model_filenames_is_list(self):
        from modules.config import get_config

        cfg = get_config()
        assert isinstance(cfg.model_filenames, list)
        assert isinstance(cfg.lora_filenames, list)
