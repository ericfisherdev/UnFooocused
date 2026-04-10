"""Unit tests for AppConfig dataclass and get_config() accessor."""

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_config_singleton():
    """Reset the config singleton before and after each test."""
    from modules.config import reset_config

    reset_config()
    yield
    reset_config()


class TestAppConfigConstruction:
    """AppConfig can be constructed from a config dict."""

    def test_from_dict_returns_app_config(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg, AppConfig)

    def test_has_typed_field_default_base_model_name(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg.default_base_model_name, str)

    def test_has_typed_field_default_cfg_scale(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg.default_cfg_scale, float)

    def test_has_typed_field_paths_checkpoints(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg.paths_checkpoints, tuple)

    def test_has_typed_field_model_filenames(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg.model_filenames, tuple)

    def test_has_typed_field_lora_filenames(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        assert isinstance(cfg.lora_filenames, tuple)

    def test_custom_values_from_dict(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "custom_model.safetensors",
                    "default_cfg_scale": 9.5,
                }
            )
        )
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=config_file)
        cfg = AppConfig.from_dict(raw)
        assert cfg.default_base_model_name == "custom_model.safetensors"
        assert cfg.default_cfg_scale == pytest.approx(9.5)

    def test_all_generation_fields_present(self, tmp_path):
        """AppConfig exposes all config fields as typed attributes."""
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)

        # Verify every field that app.py accesses exists on AppConfig
        expected_attrs = [
            "default_base_model_name",
            "default_refiner_model_name",
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
            "default_controlnet_image_count",
            "default_enhance_tabs",
            "paths_checkpoints",
            "paths_loras",
            "path_embeddings",
            "path_outputs",
            "model_filenames",
            "lora_filenames",
        ]
        missing = [a for a in expected_attrs if not hasattr(cfg, a)]
        assert not missing, f"AppConfig missing attributes: {missing}"


class TestAppConfigFrozen:
    """AppConfig is frozen — fields cannot be reassigned after construction."""

    def test_cannot_reassign_field(self, tmp_path):
        from modules.config import AppConfig, load_config

        raw = load_config(config_path=tmp_path / "config.txt")
        cfg = AppConfig.from_dict(raw)
        with pytest.raises(AttributeError):
            cfg.default_base_model_name = "hacked"  # type: ignore[misc]


class TestAppConfigDirectConstruction:
    """Tests can construct AppConfig directly without loading from file."""

    def test_construct_with_explicit_values(self):
        from modules.config import AppConfig

        cfg = AppConfig(
            default_base_model_name="test_model.safetensors",
            default_refiner_model_name="None",
            default_refiner_switch=0.5,
            default_performance="Speed",
            default_aspect_ratio="1024*1024",
            available_aspect_ratios=("1024*1024",),
            default_image_number=1,
            default_max_image_number=32,
            default_output_format="png",
            default_prompt="",
            default_prompt_negative="",
            default_styles=(),
            default_cfg_scale=4.0,
            default_sample_sharpness=2.0,
            default_sampler="dpmpp_2m_sde_gpu",
            default_scheduler="karras",
            default_loras=(),
            default_loras_min_weight=-2.0,
            default_loras_max_weight=2.0,
            default_max_lora_number=5,
            default_steps=30,
            default_controlnet_image_count=4,
            default_enhance_tabs=3,
            paths_checkpoints=("./models/checkpoints",),
            paths_loras=("./models/loras",),
            path_embeddings="./models/embeddings",
            path_outputs="./outputs",
            path_fast_checkpoints="",
            model_filenames=(),
            lora_filenames=(),
        )
        assert cfg.default_base_model_name == "test_model.safetensors"
        assert cfg.default_cfg_scale == pytest.approx(4.0)


class TestGetConfig:
    """get_config() provides lazy singleton access to AppConfig."""

    def test_returns_app_config_instance(self):
        from modules.config import AppConfig, get_config

        cfg = get_config()
        assert isinstance(cfg, AppConfig)

    def test_returns_same_instance_on_repeated_calls(self):
        from modules.config import get_config

        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_allows_reinitialization(self):
        """reset_config() clears the singleton so next get_config() reinitializes."""
        from modules.config import get_config, reset_config

        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # After reset, should be a new instance (not the same object)
        assert cfg1 is not cfg2

    def test_set_config_overrides_singleton(self):
        """set_config() allows tests to inject a custom AppConfig via dataclasses.replace."""
        import dataclasses

        from modules.config import get_config, reset_config, set_config

        try:
            base = get_config()
            custom = dataclasses.replace(base, default_base_model_name="injected_model.safetensors")
            set_config(custom)
            assert get_config().default_base_model_name == "injected_model.safetensors"
        finally:
            reset_config()


class TestRefreshFilenames:
    """Refreshing model/lora filenames returns new lists without mutating globals."""

    def test_refresh_model_filenames_returns_list(self, tmp_path):
        from modules.config import refresh_model_filenames

        result = refresh_model_filenames([str(tmp_path)])
        assert isinstance(result, list)

    def test_refresh_model_filenames_finds_new_files(self, tmp_path):
        from modules.config import refresh_model_filenames

        (tmp_path / "new_model.safetensors").write_bytes(b"\x00")
        result = refresh_model_filenames([str(tmp_path)])
        assert "new_model.safetensors" in result

    def test_refresh_lora_filenames_returns_list(self, tmp_path):
        from modules.config import refresh_lora_filenames

        result = refresh_lora_filenames([str(tmp_path)])
        assert isinstance(result, list)

    def test_refresh_lora_filenames_finds_new_files(self, tmp_path):
        from modules.config import refresh_lora_filenames

        (tmp_path / "new_lora.safetensors").write_bytes(b"\x00")
        result = refresh_lora_filenames([str(tmp_path)])
        assert "new_lora.safetensors" in result
