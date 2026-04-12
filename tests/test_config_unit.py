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
from modules.config import _NEW_PATH_KEYS
from modules.flags import Performance


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

    def test_raises_when_lora_entry_weight_is_nan(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": [[True, "a.safetensors", float("nan")]]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="finite"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_when_lora_entry_weight_is_infinity(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_loras": [[True, "a.safetensors", float("inf")]]}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="finite"):
            load_config(config_path=tmp_path / "config.txt")

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


class TestSamplingConfigDefaults:
    """UNF-53: Sampling/generation config keys exist with correct defaults."""

    def test_default_cfg_tsnr_present(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["default_cfg_tsnr"] == pytest.approx(7.0)

    def test_default_clip_skip_present(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["default_clip_skip"] == 2

    def test_default_cfg_scale_default_is_seven(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["default_cfg_scale"] == pytest.approx(7.0)

    def test_default_sampler_default_is_dpmpp_2m_sde_gpu(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["default_sampler"] == "dpmpp_2m_sde_gpu"

    def test_default_scheduler_default_is_karras(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["default_scheduler"] == "karras"

    def test_default_sample_sharpness_default_is_two(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["default_sample_sharpness"] == pytest.approx(2.0)


class TestSamplingConfigOverrides:
    """UNF-53: Sampling config values can be overridden via config.txt."""

    def test_custom_cfg_tsnr(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_cfg_tsnr": 5.5}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_cfg_tsnr"] == pytest.approx(5.5)

    def test_custom_clip_skip(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_clip_skip": 4}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_clip_skip"] == 4

    def test_custom_sampler_valid(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_sampler": "euler"}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_sampler"] == "euler"

    def test_custom_scheduler_valid(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_scheduler": "exponential"}))
        from modules.config import load_config

        result = load_config(config_path=config_file)
        assert result["default_scheduler"] == "exponential"


class TestSamplingConfigValidation:
    """UNF-53: Invalid sampling config values raise ValueError with helpful messages."""

    def test_raises_on_unknown_sampler(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_sampler": "bogus_sampler"}))
        from modules.config import load_config

        with pytest.raises(ValueError) as excinfo:
            load_config(config_path=tmp_path / "config.txt")
        msg = str(excinfo.value)
        assert "default_sampler" in msg
        assert "bogus_sampler" in msg
        # Error lists valid options so users can fix it
        assert "euler" in msg
        assert "dpmpp_2m_sde_gpu" in msg

    def test_raises_on_unknown_scheduler(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_scheduler": "nonsense"}))
        from modules.config import load_config

        with pytest.raises(ValueError) as excinfo:
            load_config(config_path=tmp_path / "config.txt")
        msg = str(excinfo.value)
        assert "default_scheduler" in msg
        assert "nonsense" in msg
        assert "karras" in msg
        assert "normal" in msg

    def test_raises_on_clip_skip_below_min(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_clip_skip": 0}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_clip_skip"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_clip_skip_above_max(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_clip_skip": 13}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_clip_skip"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_clip_skip_non_int(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_clip_skip": "two"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_clip_skip"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_cfg_tsnr_non_numeric(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_cfg_tsnr": "high"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_cfg_tsnr"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_cfg_scale_non_positive(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_cfg_scale": 0}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_cfg_scale"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_cfg_scale_non_numeric(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_cfg_scale": "seven"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_cfg_scale"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_negative_sharpness(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_sample_sharpness": -0.1}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_sample_sharpness"):
            load_config(config_path=tmp_path / "config.txt")

    def test_raises_on_sharpness_non_numeric(self, tmp_path):
        (tmp_path / "config.txt").write_text(json.dumps({"default_sample_sharpness": "sharp"}))
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_sample_sharpness"):
            load_config(config_path=tmp_path / "config.txt")

    @pytest.mark.parametrize(
        "key",
        ["default_cfg_tsnr", "default_cfg_scale", "default_sample_sharpness"],
    )
    def test_rejects_non_finite_numeric_fields(self, tmp_path, key):
        # JSON spec has no NaN/Infinity, so write the raw literal that
        # Python's json.loads accepts by default.
        (tmp_path / "config.txt").write_text("{" + f'"{key}": NaN' + "}")
        from modules.config import load_config

        with pytest.raises(ValueError, match=key):
            load_config(config_path=tmp_path / "config.txt")

    @pytest.mark.parametrize(
        "key",
        ["default_cfg_tsnr", "default_cfg_scale", "default_sample_sharpness"],
    )
    def test_rejects_infinity_numeric_fields(self, tmp_path, key):
        (tmp_path / "config.txt").write_text("{" + f'"{key}": Infinity' + "}")
        from modules.config import load_config

        with pytest.raises(ValueError, match=key):
            load_config(config_path=tmp_path / "config.txt")

    @pytest.mark.parametrize(
        "key",
        ["default_cfg_tsnr", "default_cfg_scale", "default_sample_sharpness"],
    )
    def test_rejects_huge_integer_numeric_fields(self, tmp_path, key):
        huge = "1" + "0" * 400
        (tmp_path / "config.txt").write_text("{" + f'"{key}": {huge}' + "}")
        from modules.config import load_config

        with pytest.raises(ValueError, match=key):
            load_config(config_path=tmp_path / "config.txt")


class TestImageGenDefaultsConfig:
    """UNF-55: image generation defaults (prompt, styles, aspect ratio, image number, output format)."""

    def _write(self, tmp_path, payload):
        (tmp_path / "config.txt").write_text(json.dumps(payload))
        return tmp_path / "config.txt"

    def test_default_prompt_accepts_string(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_prompt": "a cat"}))
        assert result["default_prompt"] == "a cat"

    def test_raises_when_default_prompt_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_prompt"):
            load_config(config_path=self._write(tmp_path, {"default_prompt": 42}))

    def test_raises_when_default_prompt_negative_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_prompt_negative"):
            load_config(config_path=self._write(tmp_path, {"default_prompt_negative": ["bad"]}))

    def test_default_styles_accepts_list_of_strings(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_styles": ["Style A", "Style B"]}))
        assert result["default_styles"] == ["Style A", "Style B"]

    def test_raises_when_default_styles_not_list(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_styles"):
            load_config(config_path=self._write(tmp_path, {"default_styles": "nope"}))

    def test_raises_when_default_styles_has_non_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_styles"):
            load_config(config_path=self._write(tmp_path, {"default_styles": ["A", 1]}))

    def test_default_aspect_ratio_accepts_star_format(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_aspect_ratio": "1024*1024"}))
        assert result["default_aspect_ratio"] == "1024*1024"

    def test_default_aspect_ratio_accepts_x_format(self, tmp_path):
        from modules.config import load_config

        result = load_config(
            config_path=self._write(
                tmp_path,
                {"default_aspect_ratio": "1024x1024", "available_aspect_ratios": ["1024x1024"]},
            )
        )
        assert result["default_aspect_ratio"] == "1024x1024"

    def test_raises_when_default_aspect_ratio_bad_format(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_aspect_ratio"):
            load_config(config_path=self._write(tmp_path, {"default_aspect_ratio": "square"}))

    def test_raises_when_default_aspect_ratio_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_aspect_ratio"):
            load_config(config_path=self._write(tmp_path, {"default_aspect_ratio": 1024}))

    def test_raises_when_default_aspect_ratio_non_positive(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_aspect_ratio"):
            load_config(config_path=self._write(tmp_path, {"default_aspect_ratio": "0*1024"}))

    def test_available_aspect_ratios_accepts_list(self, tmp_path):
        from modules.config import load_config

        result = load_config(
            config_path=self._write(
                tmp_path,
                {"available_aspect_ratios": ["512*512", "1024*1024"], "default_aspect_ratio": "512*512"},
            )
        )
        assert result["available_aspect_ratios"] == ["512*512", "1024*1024"]

    def test_raises_when_available_aspect_ratios_not_list(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="available_aspect_ratios"):
            load_config(config_path=self._write(tmp_path, {"available_aspect_ratios": "512*512"}))

    def test_raises_when_available_aspect_ratios_has_bad_entry(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="available_aspect_ratios"):
            load_config(
                config_path=self._write(
                    tmp_path,
                    {"available_aspect_ratios": ["512*512", "bad"], "default_aspect_ratio": "512*512"},
                )
            )

    def test_raises_when_default_aspect_ratio_not_in_available(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_aspect_ratio"):
            load_config(
                config_path=self._write(
                    tmp_path,
                    {"available_aspect_ratios": ["512*512"], "default_aspect_ratio": "1024*1024"},
                )
            )

    def test_default_image_number_accepts_in_range(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_image_number": 4}))
        assert result["default_image_number"] == 4

    def test_raises_when_default_image_number_not_int(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_image_number"):
            load_config(config_path=self._write(tmp_path, {"default_image_number": 2.5}))

    def test_raises_when_default_image_number_is_bool(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_image_number"):
            load_config(config_path=self._write(tmp_path, {"default_image_number": True}))

    def test_raises_when_default_image_number_zero(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_image_number"):
            load_config(config_path=self._write(tmp_path, {"default_image_number": 0}))

    def test_clamps_default_image_number_to_max(self, tmp_path, caplog):
        import logging

        from modules.config import load_config

        with caplog.at_level(logging.WARNING, logger="modules.config"):
            result = load_config(
                config_path=self._write(
                    tmp_path,
                    {"default_image_number": 100, "default_max_image_number": 32},
                )
            )

        assert result["default_image_number"] == 32
        assert any("clamping" in rec.message for rec in caplog.records)

    def test_raises_when_default_max_image_number_not_positive(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_image_number"):
            load_config(config_path=self._write(tmp_path, {"default_max_image_number": 0}))

    def test_raises_when_default_max_image_number_is_bool(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_max_image_number"):
            load_config(config_path=self._write(tmp_path, {"default_max_image_number": True}))

    def test_default_output_format_accepts_png(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_output_format": "png"}))
        assert result["default_output_format"] == "png"

    def test_default_output_format_accepts_jpeg(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_output_format": "jpeg"}))
        assert result["default_output_format"] == "jpeg"

    def test_default_output_format_accepts_webp(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_output_format": "webp"}))
        assert result["default_output_format"] == "webp"

    def test_raises_when_default_output_format_invalid(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_output_format"):
            load_config(config_path=self._write(tmp_path, {"default_output_format": "gif"}))

    def test_raises_when_default_output_format_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_output_format"):
            load_config(config_path=self._write(tmp_path, {"default_output_format": 123}))


class TestVaePerformanceConfig:
    """UNF-56: VAE and performance config options."""

    def _write(self, tmp_path, payload):
        (tmp_path / "config.txt").write_text(json.dumps(payload))
        return tmp_path / "config.txt"

    def test_default_vae_key_present_with_default(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert "default_vae" in result
        assert isinstance(result["default_vae"], str)

    def test_default_vae_reads_from_file(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_vae": "custom_vae.safetensors"}))
        assert result["default_vae"] == "custom_vae.safetensors"

    def test_raises_when_default_vae_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_vae"):
            load_config(config_path=self._write(tmp_path, {"default_vae": 42}))

    @pytest.mark.parametrize("value", list(Performance.values()))
    def test_default_performance_accepts_enum_values(self, tmp_path, value):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_performance": value}))
        assert result["default_performance"] == value

    def test_raises_when_default_performance_invalid(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_performance"):
            load_config(config_path=self._write(tmp_path, {"default_performance": "Ludicrous"}))

    def test_raises_when_default_performance_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_performance"):
            load_config(config_path=self._write(tmp_path, {"default_performance": 99}))

    def test_overwrite_step_defaults_to_minus_one(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert result["default_overwrite_step"] == -1

    def test_overwrite_switch_defaults_to_minus_one(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert result["default_overwrite_switch"] == -1

    def test_overwrite_upscale_defaults_to_minus_one(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert result["default_overwrite_upscale"] == -1

    def test_overwrite_step_accepts_positive_int(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_overwrite_step": 50}))
        assert result["default_overwrite_step"] == 50

    def test_overwrite_step_accepts_minus_one(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_overwrite_step": -1}))
        assert result["default_overwrite_step"] == -1

    def test_raises_when_overwrite_step_below_minus_one(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_step"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_step": -5}))

    def test_raises_when_overwrite_step_not_int(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_step"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_step": 5.5}))

    def test_raises_when_overwrite_step_is_bool(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_step"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_step": True}))

    def test_overwrite_switch_accepts_positive_int(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_overwrite_switch": 20}))
        assert result["default_overwrite_switch"] == 20

    def test_raises_when_overwrite_switch_below_minus_one(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_switch"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_switch": -10}))

    def test_raises_when_overwrite_switch_not_int(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_switch"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_switch": "auto"}))

    def test_overwrite_upscale_accepts_positive_float(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_overwrite_upscale": 0.75}))
        assert result["default_overwrite_upscale"] == pytest.approx(0.75)

    def test_overwrite_upscale_accepts_minus_one(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_overwrite_upscale": -1}))
        assert result["default_overwrite_upscale"] == -1

    def test_raises_when_overwrite_upscale_not_numeric(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_upscale"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_upscale": "strong"}))

    def test_raises_when_overwrite_upscale_is_bool(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_upscale"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_upscale": True}))

    def test_raises_when_overwrite_upscale_below_minus_one(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_overwrite_upscale"):
            load_config(config_path=self._write(tmp_path, {"default_overwrite_upscale": -2.0}))

    def test_raises_when_overwrite_upscale_nan(self, tmp_path):
        from modules.config import load_config

        (tmp_path / "config.txt").write_text(json.dumps({"default_overwrite_upscale": float("nan")}))
        with pytest.raises(ValueError, match="default_overwrite_upscale"):
            load_config(config_path=tmp_path / "config.txt")


class TestUiMetadataAdvancedConfig:
    """UNF-57: UI/metadata/advanced config defaults + validation."""

    def _write(self, tmp_path, payload):
        (tmp_path / "config.txt").write_text(json.dumps(payload))
        return tmp_path / "config.txt"

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("default_advanced_checkbox", False),
            ("default_developer_debug_mode_checkbox", False),
            ("default_black_out_nsfw", False),
            ("default_save_metadata_to_images", False),
            ("default_save_only_final_enhanced_image", False),
            ("default_describe_apply_prompts_checkbox", True),
            ("default_metadata_scheme", "fooocus"),
            ("metadata_created_by", ""),
            ("default_describe_content_type", ["Photograph"]),
        ],
    )
    def test_default_value(self, tmp_path, key, expected):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result[key] == expected

    @pytest.mark.parametrize(
        "key",
        [
            "default_advanced_checkbox",
            "default_developer_debug_mode_checkbox",
            "default_black_out_nsfw",
            "default_save_metadata_to_images",
            "default_save_only_final_enhanced_image",
            "default_describe_apply_prompts_checkbox",
        ],
    )
    def test_bool_override_accepted(self, tmp_path, key):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {key: True}))
        assert result[key] is True

    @pytest.mark.parametrize(
        "key",
        [
            "default_advanced_checkbox",
            "default_developer_debug_mode_checkbox",
            "default_black_out_nsfw",
            "default_save_metadata_to_images",
            "default_save_only_final_enhanced_image",
            "default_describe_apply_prompts_checkbox",
        ],
    )
    def test_non_bool_rejected(self, tmp_path, key):
        from modules.config import load_config

        with pytest.raises(ValueError, match=key):
            load_config(config_path=self._write(tmp_path, {key: "yes"}))

    @pytest.mark.parametrize("scheme", ["fooocus", "a111", "comfy"])
    def test_metadata_scheme_valid(self, tmp_path, scheme):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_metadata_scheme": scheme}))
        assert result["default_metadata_scheme"] == scheme

    def test_metadata_scheme_invalid(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_metadata_scheme"):
            load_config(config_path=self._write(tmp_path, {"default_metadata_scheme": "exif"}))

    def test_metadata_scheme_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_metadata_scheme"):
            load_config(config_path=self._write(tmp_path, {"default_metadata_scheme": 7}))

    def test_metadata_created_by_accepts_string(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"metadata_created_by": "Alice"}))
        assert result["metadata_created_by"] == "Alice"

    def test_metadata_created_by_not_string(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="metadata_created_by"):
            load_config(config_path=self._write(tmp_path, {"metadata_created_by": 42}))

    @pytest.mark.parametrize(
        "entries",
        [["Photograph"], ["Art/Anime"], ["Photograph", "Art/Anime"]],
    )
    def test_describe_content_type_valid(self, tmp_path, entries):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"default_describe_content_type": entries}))
        assert result["default_describe_content_type"] == entries

    def test_describe_content_type_not_list(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_describe_content_type"):
            load_config(config_path=self._write(tmp_path, {"default_describe_content_type": "Photograph"}))

    def test_describe_content_type_unknown_member(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_describe_content_type"):
            load_config(config_path=self._write(tmp_path, {"default_describe_content_type": ["Sketch"]}))

    def test_describe_content_type_non_string_member(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="default_describe_content_type"):
            load_config(config_path=self._write(tmp_path, {"default_describe_content_type": [1]}))


class TestBaseModelPresetConfig:
    """UNF-59: base_model_preset config (warn+fallback semantics)."""

    def _write(self, tmp_path, payload):
        (tmp_path / "config.txt").write_text(json.dumps(payload))
        return tmp_path / "config.txt"

    def test_default_value(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "missing.txt")
        assert result["base_model_preset"] == "SDXL"

    @pytest.mark.parametrize("preset", ["SDXL", "Pony", "Illustrious"])
    def test_valid_preset_accepted(self, tmp_path, preset):
        from modules.config import load_config

        result = load_config(config_path=self._write(tmp_path, {"base_model_preset": preset}))
        assert result["base_model_preset"] == preset

    @pytest.mark.parametrize(
        "bad_preset",
        ["SD15", 42, None, ["SDXL"]],
        ids=["unknown_string", "int", "none", "list"],
    )
    def test_invalid_preset_warns_and_falls_back(self, tmp_path, caplog, bad_preset):
        import logging

        from modules.config import load_config

        with caplog.at_level(logging.WARNING, logger="modules.config"):
            result = load_config(config_path=self._write(tmp_path, {"base_model_preset": bad_preset}))
        assert result["base_model_preset"] == "SDXL"
        assert any("base_model_preset" in rec.message for rec in caplog.records)


class TestDirectoryPathConfig:
    """UNF-58: Directory path config keys."""

    def _write(self, tmp_path, payload):
        path = tmp_path / "config.txt"
        path.write_text(json.dumps(payload))
        return path

    @pytest.mark.parametrize("key", _NEW_PATH_KEYS)
    def test_default_path_key_present(self, tmp_path, key):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert key in result
        assert isinstance(result[key], str)

    def test_default_temp_path_cleanup_on_launch_is_bool(self, tmp_path):
        from modules.config import load_config

        result = load_config(config_path=tmp_path / "config.txt")
        assert "temp_path_cleanup_on_launch" in result
        assert isinstance(result["temp_path_cleanup_on_launch"], bool)
        assert result["temp_path_cleanup_on_launch"] is True

    @pytest.mark.parametrize("key", _NEW_PATH_KEYS)
    def test_path_key_accepts_string(self, tmp_path, key):
        from modules.config import load_config

        custom = str(tmp_path / "custom")
        result = load_config(config_path=self._write(tmp_path, {key: custom}))
        assert result[key] == custom

    @pytest.mark.parametrize("key", _NEW_PATH_KEYS)
    def test_path_key_rejects_non_string(self, tmp_path, key):
        from modules.config import load_config

        with pytest.raises(ValueError, match=key):
            load_config(config_path=self._write(tmp_path, {key: 42}))

    def test_temp_path_cleanup_rejects_non_bool(self, tmp_path):
        from modules.config import load_config

        with pytest.raises(ValueError, match="temp_path_cleanup_on_launch"):
            load_config(config_path=self._write(tmp_path, {"temp_path_cleanup_on_launch": "yes"}))

    def test_warns_when_path_does_not_exist(self, tmp_path, caplog):
        import logging

        from modules.config import load_config

        with caplog.at_level(logging.WARNING, logger="modules.config"):
            load_config(config_path=self._write(tmp_path, {"path_vae": "/nonexistent/vae/dir"}))
        assert any("path_vae" in rec.message for rec in caplog.records)

    def test_does_not_warn_when_path_exists(self, tmp_path, caplog):
        import logging

        from modules.config import load_config

        real_dir = tmp_path / "real_vae"
        real_dir.mkdir()
        with caplog.at_level(logging.WARNING, logger="modules.config"):
            load_config(config_path=self._write(tmp_path, {"path_vae": str(real_dir)}))
        assert not any("path_vae=" in rec.message and "does not exist" in rec.message for rec in caplog.records)

    def test_env_var_override_applies(self, tmp_path, monkeypatch):
        from modules.config import load_config

        monkeypatch.setenv("path_vae", "/env/override/vae")
        result = load_config(config_path=tmp_path / "config.txt")
        assert result["path_vae"] == "/env/override/vae"

    def test_env_var_override_applies_to_temp_path(self, tmp_path, monkeypatch):
        from modules.config import load_config

        monkeypatch.setenv("temp_path", "/env/temp")
        result = load_config(config_path=tmp_path / "config.txt")
        assert result["temp_path"] == "/env/temp"

    def test_env_var_override_beats_file(self, tmp_path, monkeypatch):
        from modules.config import load_config

        monkeypatch.setenv("path_vae", "/env/wins")
        result = load_config(config_path=self._write(tmp_path, {"path_vae": "/file/loses"}))
        assert result["path_vae"] == "/env/wins"

    def test_cleanup_temp_path_removes_files(self, tmp_path):
        from modules.config import cleanup_temp_path

        temp = tmp_path / "temp"
        temp.mkdir()
        (temp / "stale.txt").write_text("stale")
        sub = temp / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")

        cleanup_temp_path(str(temp))

        assert temp.exists()
        assert list(temp.iterdir()) == []

    def test_cleanup_temp_path_missing_dir_is_noop(self, tmp_path):
        from modules.config import cleanup_temp_path

        cleanup_temp_path(str(tmp_path / "does_not_exist"))

    def test_cleanup_temp_path_unlinks_child_symlink_preserving_target(self, tmp_path):
        from modules.config import cleanup_temp_path

        temp = tmp_path / "temp"
        temp.mkdir()
        target = tmp_path / "target_dir"
        target.mkdir()
        (target / "secret.txt").write_text("keep me")

        symlink = temp / "link_to_target"
        symlink.symlink_to(target)

        cleanup_temp_path(str(temp))

        assert not symlink.exists()
        assert target.is_dir()
        assert (target / "secret.txt").read_text() == "keep me"

    def test_cleanup_temp_path_refuses_symlinked_root(self, tmp_path, caplog):
        import logging

        from modules.config import cleanup_temp_path

        real = tmp_path / "real"
        real.mkdir()
        (real / "keep.txt").write_text("keep")
        link = tmp_path / "temp_link"
        link.symlink_to(real)

        with caplog.at_level(logging.WARNING, logger="modules.config"):
            cleanup_temp_path(str(link))

        assert (real / "keep.txt").exists()
        assert any("symlink" in rec.message for rec in caplog.records)

    def test_load_config_cleans_temp_when_enabled(self, tmp_path):
        from modules.config import load_config

        temp = tmp_path / "temp"
        temp.mkdir()
        (temp / "old.txt").write_text("old")
        load_config(
            config_path=self._write(
                tmp_path,
                {"temp_path": str(temp), "temp_path_cleanup_on_launch": True},
            )
        )
        assert list(temp.iterdir()) == []

    def test_load_config_skips_cleanup_when_disabled(self, tmp_path):
        from modules.config import load_config

        temp = tmp_path / "temp"
        temp.mkdir()
        (temp / "keep.txt").write_text("keep")
        load_config(
            config_path=self._write(
                tmp_path,
                {"temp_path": str(temp), "temp_path_cleanup_on_launch": False},
            )
        )
        assert (temp / "keep.txt").exists()
