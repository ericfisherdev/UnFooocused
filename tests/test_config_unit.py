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


class TestLoraConfig:
    """UNF-54: LoRA config option validation."""

    def test_default_loras_accepts_valid_entries(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_loras": [
                        [True, "adapter.safetensors", 0.8],
                        [False, "None", 1.0],
                    ]
                }
            )
        )
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_loras"][0] == [True, "adapter.safetensors", 0.8]

    def test_raises_when_default_loras_not_list(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": "nope"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_lora_entry_wrong_arity(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": [[True, "a.safetensors"]]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_lora_entry_bad_enabled_type(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": [["yes", "a.safetensors", 1.0]]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_lora_entry_bad_filename_type(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": [[True, 42, 1.0]]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_lora_entry_bad_weight_type(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": [[True, "a.safetensors", "heavy"]]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_lora_entry_not_list(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": ["flat"]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_min_weight_below_bound(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras_min_weight": -11.0}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras_min_weight"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_weight_above_bound(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras_max_weight": 11.0}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras_max_weight"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_min_weight_not_numeric(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras_min_weight": "low"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras_min_weight"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_weight_not_numeric(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras_max_weight": "high"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras_max_weight"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_lora_number_not_int(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_max_lora_number": 4.5}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_lora_number"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_lora_number_zero(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_max_lora_number": 0}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_lora_number"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_lora_number_negative(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_max_lora_number": -3}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_lora_number"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_min_weight_is_bool(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras_min_weight": True}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras_min_weight"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_weight_is_bool(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras_max_weight": False}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_loras_max_weight"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_max_lora_number_is_bool(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_max_lora_number": True}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_lora_number"):
            load_config(config_path=tmp_path / "config.txt")

    def test_validate_lora_entry_accepts_tuple(self):
        from modules.config import _validate_lora_entry

        _validate_lora_entry(0, (True, "adapter.safetensors", 0.8))

    def test_raises_when_loras_exceeds_max_lora_number(self, tmp_path):
        (tmp_path / "config.txt").write_text(
            json.dumps(
                {
                    "default_max_lora_number": 2,
                    "default_loras": [
                        [True, "a.safetensors", 1.0],
                        [True, "b.safetensors", 1.0],
                        [True, "c.safetensors", 1.0],
                    ],
                }
            )
        )
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_lora_number"):
            load_config(config_path=tmp_path / "config.txt")


class TestModelRefinerConfig:
    """UNF-52: Model and refiner config options."""

    def test_default_base_model_key_present(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert "default_base_model" in result

    def test_default_base_model_defaults_to_none(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert result["default_base_model"] is None

    def test_default_base_model_reads_from_file(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_base_model": "sdxl"}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_base_model"] == "sdxl"

    def test_previous_default_models_present(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert "previous_default_models" in result
        assert isinstance(result["previous_default_models"], list)

    def test_previous_default_models_defaults_empty(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert result["previous_default_models"] == []

    def test_defaults_not_mutated_across_calls(self, tmp_path):
        """load_config() must return fresh mutable containers — no shared state."""
        from modules.config import load_config

        first = load_config(config_path=tmp_path / "config.txt")
        first["previous_default_models"].append("leaked.safetensors")
        second = load_config(config_path=tmp_path / "config.txt")
        assert second["previous_default_models"] == []

    def test_previous_default_models_reads_from_file(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"previous_default_models": ["a.safetensors", "b.safetensors"]}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["previous_default_models"] == ["a.safetensors", "b.safetensors"]

    def test_raises_when_default_model_not_string(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_model": 42}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_model"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_default_refiner_not_string(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_refiner": 42}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_refiner"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_refiner_switch_below_zero(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_refiner_switch": -0.1}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_refiner_switch"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_refiner_switch_above_one(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_refiner_switch": 1.5}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_refiner_switch"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_refiner_switch_not_numeric(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_refiner_switch": "half"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_refiner_switch"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_refiner_switch_is_bool(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_refiner_switch": True}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_refiner_switch"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_previous_default_models_not_list(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"previous_default_models": "not-a-list"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="previous_default_models"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_default_base_model_wrong_type(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_base_model": 42}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_base_model"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_previous_default_models_has_non_string(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"previous_default_models": ["ok.safetensors", 42]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="previous_default_models"):
            load_config(config_path=tmp_path / "config.txt")


class TestAppConfigModelRefinerFields:
    """UNF-52: new fields exposed on AppConfig for generation pipeline wiring."""

    def test_app_config_has_default_base_model(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert cfg.default_base_model is None

    def test_app_config_has_previous_default_models(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert cfg.previous_default_models == ()

    def test_app_config_reads_custom_base_model_and_history(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_base_model": "sdxl",
                    "previous_default_models": ["old1.safetensors", "old2.safetensors"],
                }
            )
        )
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=config_file)
        cfg = AppConfig.from_dict(raw)
        assert cfg.default_base_model == "sdxl"
        assert cfg.previous_default_models == ("old1.safetensors", "old2.safetensors")
