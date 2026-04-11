"""RED tests for UNF-43: Performance preset modes (LCM, Lightning, Hyper-SD).

Acceptance criteria encoded as tests:
AC-1: EXTREME_SPEED (LCM) applies: sampler=lcm, scheduler=lcm, steps=8, cfg=1.0, LoRA=sdxl_lcm_lora
AC-2: LIGHTNING applies: sampler=euler, scheduler=sgm_uniform, steps=4, cfg=1.0, LoRA=sdxl_lightning_4step_lora
AC-3: HYPER_SD applies: sampler=dpmpp_sde_gpu, scheduler=karras, steps=4, cfg=1.0, LoRA=sdxl_hyper_sd_4step_lora
AC-4: All speed presets disable the refiner model
AC-5: All speed presets set sharpness=0, adaptive_cfg=1.0, ADM scalers=1.0/1.0/0.0
AC-6: Scheduler patching works (lcm/tcd -> ModelSamplingDiscrete)
AC-7: Missing LoRA files produce a clear error message
AC-8: Tests verify each preset's parameter overrides (this file)
"""

from __future__ import annotations

import pytest
from modules.flags import Performance

# ---------------------------------------------------------------------------
# AC-1: EXTREME_SPEED (LCM) preset overrides
# ---------------------------------------------------------------------------


class TestExtremeSpeedPreset:
    """AC-1: Performance.EXTREME_SPEED applies LCM-specific overrides."""

    def test_sampler_is_lcm(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.sampler_name == "lcm"

    def test_scheduler_is_lcm(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.scheduler_name == "lcm"

    def test_steps_is_8(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.steps == 8

    def test_cfg_scale_is_1(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.cfg_scale == pytest.approx(1.0)

    def test_lora_filename(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.lora_filename == "sdxl_lcm_lora.safetensors"

    def test_lora_weight_is_1(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.lora_weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC-2: LIGHTNING preset overrides
# ---------------------------------------------------------------------------


class TestLightningPreset:
    """AC-2: Performance.LIGHTNING applies Lightning-specific overrides."""

    def test_sampler_is_euler(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.sampler_name == "euler"

    def test_scheduler_is_sgm_uniform(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.scheduler_name == "sgm_uniform"

    def test_steps_is_4(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.steps == 4

    def test_cfg_scale_is_1(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.cfg_scale == pytest.approx(1.0)

    def test_lora_filename(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.lora_filename == "sdxl_lightning_4step_lora.safetensors"

    def test_lora_weight_is_1(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.lora_weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC-3: HYPER_SD preset overrides
# ---------------------------------------------------------------------------


class TestHyperSDPreset:
    """AC-3: Performance.HYPER_SD applies Hyper-SD-specific overrides."""

    def test_sampler_is_dpmpp_sde_gpu(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.sampler_name == "dpmpp_sde_gpu"

    def test_scheduler_is_karras(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.scheduler_name == "karras"

    def test_steps_is_4(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.steps == 4

    def test_cfg_scale_is_1(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.cfg_scale == pytest.approx(1.0)

    def test_lora_filename(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.lora_filename == "sdxl_hyper_sd_4step_lora.safetensors"

    def test_lora_weight_is_0_8(self) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.lora_weight == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# AC-4: All speed presets disable the refiner
# ---------------------------------------------------------------------------


class TestAllSpeedPresetsDisableRefiner:
    """AC-4: All speed presets set refiner_disabled=True."""

    @pytest.mark.parametrize(
        "preset",
        [Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD],
        ids=["extreme_speed", "lightning", "hyper_sd"],
    )
    def test_refiner_disabled(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(preset)
        assert overrides.refiner_disabled is True


# ---------------------------------------------------------------------------
# AC-5: All speed presets set sharpness=0, adaptive_cfg=1.0, ADM scalers
# ---------------------------------------------------------------------------


class TestAllSpeedPresetsGlobalOverrides:
    """AC-5: sharpness=0, adaptive_cfg=1.0, ADM scalers=1.0/1.0/0.0."""

    @pytest.mark.parametrize(
        "preset",
        [Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD],
        ids=["extreme_speed", "lightning", "hyper_sd"],
    )
    def test_sharpness_is_zero(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(preset)
        assert overrides.sharpness == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "preset",
        [Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD],
        ids=["extreme_speed", "lightning", "hyper_sd"],
    )
    def test_adaptive_cfg_is_1(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(preset)
        assert overrides.adaptive_cfg == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "preset",
        [Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD],
        ids=["extreme_speed", "lightning", "hyper_sd"],
    )
    def test_adm_scaler_positive_is_1(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(preset)
        assert overrides.adm_scaler_positive == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "preset",
        [Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD],
        ids=["extreme_speed", "lightning", "hyper_sd"],
    )
    def test_adm_scaler_negative_is_1(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(preset)
        assert overrides.adm_scaler_negative == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "preset",
        [Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD],
        ids=["extreme_speed", "lightning", "hyper_sd"],
    )
    def test_adm_scaler_end_is_0(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        overrides = get_preset_overrides(preset)
        assert overrides.adm_scaler_end == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AC-6: Scheduler patching classification
# ---------------------------------------------------------------------------


class TestSchedulerPatchType:
    """AC-6: Preset overrides expose the correct scheduler patch type."""

    def test_lcm_scheduler_needs_discrete_patch(self) -> None:
        from modules.domain.performance_presets import SchedulerPatchType, get_preset_overrides

        overrides = get_preset_overrides(Performance.EXTREME_SPEED)
        assert overrides.scheduler_patch_type is SchedulerPatchType.MODEL_SAMPLING_DISCRETE

    def test_lightning_scheduler_needs_discrete_patch(self) -> None:
        from modules.domain.performance_presets import SchedulerPatchType, get_preset_overrides

        overrides = get_preset_overrides(Performance.LIGHTNING)
        assert overrides.scheduler_patch_type is SchedulerPatchType.MODEL_SAMPLING_DISCRETE

    def test_hyper_sd_scheduler_needs_continuous_edm_patch(self) -> None:
        from modules.domain.performance_presets import SchedulerPatchType, get_preset_overrides

        overrides = get_preset_overrides(Performance.HYPER_SD)
        assert overrides.scheduler_patch_type is SchedulerPatchType.CONTINUOUS_EDM


# ---------------------------------------------------------------------------
# AC-7: Missing LoRA produces clear error
# ---------------------------------------------------------------------------


class TestMissingLoRAError:
    """AC-7: validate_preset_lora raises ModelNotFoundError for missing LoRA."""

    def test_missing_lora_raises_model_not_found(self, tmp_path) -> None:
        from modules.domain.exceptions import ModelNotFoundError
        from modules.domain.performance_presets import validate_preset_lora

        with pytest.raises(ModelNotFoundError, match="sdxl_lcm_lora"):
            validate_preset_lora(
                Performance.EXTREME_SPEED,
                lora_directory=str(tmp_path),
            )

    def test_existing_lora_passes_validation(self, tmp_path) -> None:
        from modules.domain.performance_presets import validate_preset_lora

        lora_file = tmp_path / "sdxl_lcm_lora.safetensors"
        lora_file.touch()
        # Should not raise
        validate_preset_lora(
            Performance.EXTREME_SPEED,
            lora_directory=str(tmp_path),
        )

    def test_non_speed_preset_passes_without_lora(self, tmp_path) -> None:
        from modules.domain.performance_presets import validate_preset_lora

        # QUALITY has no LoRA — validation should pass
        validate_preset_lora(
            Performance.QUALITY,
            lora_directory=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Non-speed presets should return None
# ---------------------------------------------------------------------------


class TestNonSpeedPresetsReturnNone:
    """QUALITY and SPEED presets have no overrides."""

    @pytest.mark.parametrize(
        "preset",
        [Performance.QUALITY, Performance.SPEED],
        ids=["quality", "speed"],
    )
    def test_no_overrides_for_non_speed_presets(self, preset: Performance) -> None:
        from modules.domain.performance_presets import get_preset_overrides

        assert get_preset_overrides(preset) is None


# ---------------------------------------------------------------------------
# Apply preset to PipelineConfig
# ---------------------------------------------------------------------------


class TestApplyPresetToPipelineConfig:
    """Integration: apply_performance_preset produces correct PipelineConfig overrides."""

    def _make_base_config(self):
        """Build a minimal PipelineConfig for testing preset application."""
        from modules.domain.protocols import LoRAConfig
        from modules.services.diffusion_pipeline import PipelineConfig

        return PipelineConfig(
            checkpoint_path="/models/base.safetensors",
            loras=[LoRAConfig(filename="user_lora.safetensors", weight=0.7)],
            positive_prompt="a cat",
            negative_prompt="bad",
            sampler_name="dpmpp_2m_sde_gpu",
            scheduler="normal",
            steps=30,
            cfg_scale=7.0,
            seed=42,
            denoise=1.0,
            image_number=1,
            clip_skip=1,
            width=1024,
            height=1024,
            disable_seed_increment=False,
            freeu_enabled=False,
            freeu_b1=1.3,
            freeu_b2=1.4,
            freeu_s1=0.9,
            freeu_s2=0.2,
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="joint",
            refiner_switch=0.5,
        )

    def test_extreme_speed_overrides_sampler_and_scheduler(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.EXTREME_SPEED)
        assert result.sampler_name == "lcm"
        assert result.scheduler == "lcm"

    def test_extreme_speed_overrides_steps_and_cfg(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.EXTREME_SPEED)
        assert result.steps == 8
        assert result.cfg_scale == pytest.approx(1.0)

    def test_extreme_speed_disables_refiner(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.EXTREME_SPEED)
        assert result.has_refiner is False

    def test_extreme_speed_adds_lora_to_list(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.EXTREME_SPEED)
        lora_filenames = [lora.filename for lora in result.loras]
        assert "sdxl_lcm_lora.safetensors" in lora_filenames
        # User LoRA should also be preserved
        assert "user_lora.safetensors" in lora_filenames

    def test_lightning_overrides(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.LIGHTNING)
        assert result.sampler_name == "euler"
        assert result.scheduler == "sgm_uniform"
        assert result.steps == 4
        assert result.cfg_scale == pytest.approx(1.0)

    def test_hyper_sd_overrides(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.HYPER_SD)
        assert result.sampler_name == "dpmpp_sde_gpu"
        assert result.scheduler == "karras"
        assert result.steps == 4
        assert result.cfg_scale == pytest.approx(1.0)

    def test_quality_preset_returns_unchanged_config(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.QUALITY)
        assert result is config  # No overrides, same object returned

    def test_speed_preset_returns_unchanged_config(self) -> None:
        from modules.services.performance_preset_service import apply_performance_preset

        config = self._make_base_config()
        result = apply_performance_preset(config, Performance.SPEED)
        assert result is config
