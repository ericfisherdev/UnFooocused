"""Unit tests for PatchSettings system — anisotropic filtering, ADM guidance, adaptive CFG.

Tests the patch system that applies sharpness, ADM guidance scaling, and
adaptive CFG to the diffusion process. All tests run without GPU using fakes.

Acceptance Criteria covered:
  AC1: PatchSettings correctly applies sharpness via anisotropic filtering
  AC2: ADM guidance scalers modify SDXL's positive/negative conditioning strength
  AC3: adaptive_cfg applies TSNR-based CFG adjustment during sampling
  AC4: controlnet_softness parameter is stored for future ControlNet use
  AC5: PatchSettings are scoped per-generation (not leaked between tasks)
  AC6: patch_all() initializes all necessary hooks at application startup
  AC7: anisotropic filtering module is ported and functional
  AC8: Tests verify that different sharpness values produce measurably different outputs
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# AC1: PatchSettings correctly applies sharpness via anisotropic filtering
# ---------------------------------------------------------------------------


class TestSharpnessAnisotropicFiltering:
    """AC1: Sharpness controls anisotropic filtering applied during sampling."""

    def test_sharpness_zero_disables_filtering(self) -> None:
        """When sharpness=0, anisotropic filter alpha is 0 — output equals input."""
        from modules.infrastructure.patch_system import compute_sharpness_alpha

        alpha = compute_sharpness_alpha(sharpness=0.0, diffusion_progress=0.5)
        assert alpha == pytest.approx(0.0)

    def test_sharpness_positive_produces_nonzero_alpha(self) -> None:
        """When sharpness > 0 and progress > 0, alpha is positive."""
        from modules.infrastructure.patch_system import compute_sharpness_alpha

        alpha = compute_sharpness_alpha(sharpness=2.0, diffusion_progress=0.5)
        assert alpha > 0.0

    def test_sharpness_alpha_scales_with_sharpness_value(self) -> None:
        """Higher sharpness produces higher alpha."""
        from modules.infrastructure.patch_system import compute_sharpness_alpha

        alpha_low = compute_sharpness_alpha(sharpness=1.0, diffusion_progress=0.5)
        alpha_high = compute_sharpness_alpha(sharpness=5.0, diffusion_progress=0.5)
        assert alpha_high > alpha_low

    def test_sharpness_alpha_scales_with_diffusion_progress(self) -> None:
        """Higher diffusion progress produces higher alpha."""
        from modules.infrastructure.patch_system import compute_sharpness_alpha

        alpha_early = compute_sharpness_alpha(sharpness=2.0, diffusion_progress=0.1)
        alpha_late = compute_sharpness_alpha(sharpness=2.0, diffusion_progress=0.9)
        assert alpha_late > alpha_early

    def test_sharpness_alpha_formula_matches_reference(self) -> None:
        """Alpha = 0.001 * sharpness * diffusion_progress (reference formula)."""
        from modules.infrastructure.patch_system import compute_sharpness_alpha

        alpha = compute_sharpness_alpha(sharpness=2.0, diffusion_progress=0.5)
        expected = 0.001 * 2.0 * 0.5
        assert alpha == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AC2: ADM guidance scalers modify SDXL's positive/negative conditioning
# ---------------------------------------------------------------------------


class TestADMGuidanceScaling:
    """AC2: ADM scalers modify positive/negative conditioning dimensions."""

    def test_positive_adm_scales_width_and_height(self) -> None:
        """Positive prompt type scales width/height by positive_adm_scale."""
        from modules.infrastructure.patch_system import scale_adm_dimensions

        width, height = scale_adm_dimensions(
            width=1024,
            height=1024,
            prompt_type="positive",
            positive_adm_scale=1.5,
            negative_adm_scale=0.8,
        )
        assert width == 1536
        assert height == 1536

    def test_negative_adm_scales_width_and_height(self) -> None:
        """Negative prompt type scales width/height by negative_adm_scale."""
        from modules.infrastructure.patch_system import scale_adm_dimensions

        width, height = scale_adm_dimensions(
            width=1024,
            height=1024,
            prompt_type="negative",
            positive_adm_scale=1.5,
            negative_adm_scale=0.8,
        )
        assert width == 819
        assert height == 819

    def test_neutral_prompt_type_no_scaling(self) -> None:
        """Unknown prompt type leaves dimensions unchanged."""
        from modules.infrastructure.patch_system import scale_adm_dimensions

        width, height = scale_adm_dimensions(
            width=1024,
            height=1024,
            prompt_type="",
            positive_adm_scale=1.5,
            negative_adm_scale=0.8,
        )
        assert width == 1024
        assert height == 1024

    def test_adm_scale_one_is_identity(self) -> None:
        """Scale factor of 1.0 leaves dimensions unchanged."""
        from modules.infrastructure.patch_system import scale_adm_dimensions

        width, height = scale_adm_dimensions(
            width=768,
            height=512,
            prompt_type="positive",
            positive_adm_scale=1.0,
            negative_adm_scale=1.0,
        )
        assert width == 768
        assert height == 512


# ---------------------------------------------------------------------------
# AC3: adaptive_cfg applies TSNR-based CFG adjustment during sampling
# ---------------------------------------------------------------------------


class TestAdaptiveCFG:
    """AC3: Adaptive CFG modifies sampling behavior based on progress."""

    def test_cfg_below_adaptive_threshold_returns_real_eps(self) -> None:
        """When cfg_scale <= adaptive_cfg, no mimicking — pure real eps."""
        from modules.infrastructure.patch_system import compute_adaptive_cfg

        uncond = 1.0
        cond = 3.0
        cfg_scale = 5.0
        adaptive_cfg = 7.0
        progress = 0.5

        result = compute_adaptive_cfg(
            uncond=uncond,
            cond=cond,
            cfg_scale=cfg_scale,
            adaptive_cfg=adaptive_cfg,
            progress=progress,
        )
        # real_eps = uncond + cfg_scale * (cond - uncond) = 1 + 5*(3-1) = 11
        expected = 1.0 + 5.0 * (3.0 - 1.0)
        assert result == pytest.approx(expected)

    def test_cfg_above_adaptive_threshold_blends_with_mimic(self) -> None:
        """When cfg_scale > adaptive_cfg, result is blend of real and mimicked."""
        from modules.infrastructure.patch_system import compute_adaptive_cfg

        uncond = 1.0
        cond = 3.0
        cfg_scale = 10.0
        adaptive_cfg = 7.0
        progress = 0.5

        result = compute_adaptive_cfg(
            uncond=uncond,
            cond=cond,
            cfg_scale=cfg_scale,
            adaptive_cfg=adaptive_cfg,
            progress=progress,
        )
        real_eps = uncond + cfg_scale * (cond - uncond)
        mimic_eps = uncond + adaptive_cfg * (cond - uncond)
        expected = real_eps * progress + mimic_eps * (1.0 - progress)
        assert result == pytest.approx(expected)

    def test_adaptive_cfg_at_progress_zero_returns_mimicked(self) -> None:
        """At progress=0, when blending, result equals mimicked eps entirely."""
        from modules.infrastructure.patch_system import compute_adaptive_cfg

        uncond = 0.0
        cond = 2.0
        cfg_scale = 12.0
        adaptive_cfg = 7.0
        progress = 0.0

        result = compute_adaptive_cfg(
            uncond=uncond,
            cond=cond,
            cfg_scale=cfg_scale,
            adaptive_cfg=adaptive_cfg,
            progress=progress,
        )
        mimic_eps = uncond + adaptive_cfg * (cond - uncond)
        assert result == pytest.approx(mimic_eps)

    def test_adaptive_cfg_at_progress_one_returns_real(self) -> None:
        """At progress=1, when blending, result equals real eps entirely."""
        from modules.infrastructure.patch_system import compute_adaptive_cfg

        uncond = 0.0
        cond = 2.0
        cfg_scale = 12.0
        adaptive_cfg = 7.0
        progress = 1.0

        result = compute_adaptive_cfg(
            uncond=uncond,
            cond=cond,
            cfg_scale=cfg_scale,
            adaptive_cfg=adaptive_cfg,
            progress=progress,
        )
        real_eps = uncond + cfg_scale * (cond - uncond)
        assert result == pytest.approx(real_eps)


# ---------------------------------------------------------------------------
# AC4: controlnet_softness parameter is stored for future ControlNet use
# ---------------------------------------------------------------------------


class TestControlNetSoftness:
    """AC4: controlnet_softness stored in PatchSettings and runtime state."""

    def test_patch_settings_stores_controlnet_softness(self) -> None:
        from modules.infrastructure.sampler import PatchSettings

        settings = PatchSettings(controlnet_softness=0.5)
        assert settings.controlnet_softness == pytest.approx(0.5)

    def test_runtime_state_stores_controlnet_softness(self) -> None:
        from modules.infrastructure.patch_system import PatchSettingsState

        state = PatchSettingsState(controlnet_softness=0.4)
        assert state.controlnet_softness == pytest.approx(0.4)

    def test_default_controlnet_softness(self) -> None:
        from modules.infrastructure.patch_system import PatchSettingsState

        state = PatchSettingsState()
        assert state.controlnet_softness == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# AC5: PatchSettings are scoped per-generation (not leaked between tasks)
# ---------------------------------------------------------------------------


class TestPerGenerationScoping:
    """AC5: PatchSettings scoped per-generation, no leakage between tasks."""

    def test_registry_set_and_get_isolate_by_pid(self) -> None:
        """Different PIDs get independent PatchSettingsState."""
        from modules.infrastructure.patch_system import PatchSettingsRegistry

        registry = PatchSettingsRegistry()
        registry.set(pid=100, sharpness=3.0, adaptive_cfg=9.0)
        registry.set(pid=200, sharpness=1.0, adaptive_cfg=5.0)

        state_100 = registry.get(100)
        state_200 = registry.get(200)

        assert state_100.sharpness == pytest.approx(3.0)
        assert state_200.sharpness == pytest.approx(1.0)

    def test_registry_clear_removes_state(self) -> None:
        """Clearing a PID removes its state, preventing leakage."""
        from modules.infrastructure.patch_system import PatchSettingsRegistry

        registry = PatchSettingsRegistry()
        registry.set(pid=100, sharpness=3.0)

        registry.clear(100)

        # After clear, getting should return fresh defaults
        state = registry.get(100)
        assert state.sharpness == pytest.approx(2.0)  # default

    def test_registry_get_returns_defaults_for_unknown_pid(self) -> None:
        """Unknown PID returns default PatchSettingsState."""
        from modules.infrastructure.patch_system import PatchSettingsRegistry

        registry = PatchSettingsRegistry()
        state = registry.get(999)
        assert state.sharpness == pytest.approx(2.0)
        assert state.adaptive_cfg == pytest.approx(7.0)

    def test_context_manager_sets_and_clears(self) -> None:
        """Context manager scopes settings to a block, clears on exit."""
        from modules.infrastructure.patch_system import PatchSettingsRegistry

        registry = PatchSettingsRegistry()
        pid = 42

        with registry.scoped(pid=pid, sharpness=5.0, adaptive_cfg=12.0):
            state = registry.get(pid)
            assert state.sharpness == pytest.approx(5.0)

        # After exit, state should be cleared (defaults)
        state_after = registry.get(pid)
        assert state_after.sharpness == pytest.approx(2.0)

    def test_context_manager_clears_on_exception(self) -> None:
        """State cleaned up even if exception occurs inside context."""
        from modules.infrastructure.patch_system import PatchSettingsRegistry

        registry = PatchSettingsRegistry()
        pid = 42

        with pytest.raises(ValueError, match="boom"), registry.scoped(pid=pid, sharpness=5.0):
            raise ValueError("boom")

        state = registry.get(pid)
        assert state.sharpness == pytest.approx(2.0)  # cleaned up


# ---------------------------------------------------------------------------
# AC6: patch_all() initializes all necessary hooks at application startup
# ---------------------------------------------------------------------------


class TestPatchAll:
    """AC6: patch_all() installs monkey-patches on ldm_patched internals."""

    def test_patch_all_replaces_sampling_function(self) -> None:
        """patch_all() replaces ldm_patched.modules.samplers.sampling_function."""
        # Save originals
        import ldm_patched.modules.samplers as samplers_mod
        from modules.infrastructure import patch_system

        original = getattr(samplers_mod, "sampling_function", None)

        try:
            patch_system.patch_all()
            assert samplers_mod.sampling_function is not original or original is None
        finally:
            # Restore if needed
            if original is not None:
                samplers_mod.sampling_function = original

    def test_patch_all_replaces_sdxl_encode_adm(self) -> None:
        """patch_all() replaces SDXL.encode_adm."""
        import ldm_patched.modules.model_base as model_base_mod
        from modules.infrastructure import patch_system

        original = getattr(model_base_mod.SDXL, "encode_adm", None)

        try:
            patch_system.patch_all()
            assert model_base_mod.SDXL.encode_adm is not original or original is None
        finally:
            if original is not None:
                model_base_mod.SDXL.encode_adm = original

    def test_patch_all_is_idempotent(self) -> None:
        """Calling patch_all() twice does not break things."""
        import ldm_patched.modules.samplers as samplers_mod
        from modules.infrastructure import patch_system

        original = getattr(samplers_mod, "sampling_function", None)

        try:
            patch_system.patch_all()
            first = samplers_mod.sampling_function
            patch_system.patch_all()
            second = samplers_mod.sampling_function
            assert first is second  # same function, not double-wrapped
        finally:
            if original is not None:
                samplers_mod.sampling_function = original


# ---------------------------------------------------------------------------
# AC7: anisotropic filtering module is ported and functional
# ---------------------------------------------------------------------------


class TestAnisotropicFiltering:
    """AC7: Anisotropic filtering module exists and is functional."""

    def test_module_importable(self) -> None:
        from modules.infrastructure.anisotropic import adaptive_anisotropic_filter

        assert adaptive_anisotropic_filter is not None

    def test_bilateral_blur_importable(self) -> None:
        from modules.infrastructure.anisotropic import bilateral_blur

        assert bilateral_blur is not None

    def test_adaptive_filter_returns_same_shape(self) -> None:
        """Output tensor has same shape as input tensor."""
        import torch
        from modules.infrastructure.anisotropic import adaptive_anisotropic_filter

        x = torch.randn(1, 4, 64, 64)
        result = adaptive_anisotropic_filter(x)
        assert result.shape == x.shape

    def test_adaptive_filter_with_guidance(self) -> None:
        """Filter works with separate guidance tensor."""
        import torch
        from modules.infrastructure.anisotropic import adaptive_anisotropic_filter

        x = torch.randn(1, 4, 64, 64)
        g = torch.randn(1, 4, 64, 64)
        result = adaptive_anisotropic_filter(x, g=g)
        assert result.shape == x.shape

    def test_bilateral_blur_preserves_shape(self) -> None:
        """bilateral_blur returns tensor with same shape as input."""
        import torch
        from modules.infrastructure.anisotropic import bilateral_blur

        x = torch.randn(1, 4, 32, 32)
        result = bilateral_blur(x)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# AC8: Different sharpness values produce measurably different outputs
# ---------------------------------------------------------------------------


class TestSharpnessDifferentOutputs:
    """AC8: Different sharpness values produce measurably different outputs."""

    def test_zero_sharpness_returns_unfiltered(self) -> None:
        """Sharpness=0 means alpha=0, so filtered eps has no effect."""
        from modules.infrastructure.patch_system import apply_sharpness_blending

        positive_eps = 10.0
        filtered_eps = 5.0
        progress = 0.5

        result = apply_sharpness_blending(
            positive_eps=positive_eps,
            filtered_eps=filtered_eps,
            sharpness=0.0,
            diffusion_progress=progress,
        )
        assert result == pytest.approx(positive_eps)

    def test_high_sharpness_shifts_toward_filtered(self) -> None:
        """Higher sharpness blends more toward the filtered signal."""
        from modules.infrastructure.patch_system import apply_sharpness_blending

        positive_eps = 10.0
        filtered_eps = 5.0
        progress = 0.5

        result_low = apply_sharpness_blending(
            positive_eps=positive_eps,
            filtered_eps=filtered_eps,
            sharpness=1.0,
            diffusion_progress=progress,
        )
        result_high = apply_sharpness_blending(
            positive_eps=positive_eps,
            filtered_eps=filtered_eps,
            sharpness=10.0,
            diffusion_progress=progress,
        )
        # Higher sharpness should push result closer to filtered_eps
        assert abs(result_high - filtered_eps) < abs(result_low - filtered_eps)

    def test_different_sharpness_values_produce_different_results(self) -> None:
        """Two different non-zero sharpness values produce distinct outputs."""
        from modules.infrastructure.patch_system import apply_sharpness_blending

        positive_eps = 10.0
        filtered_eps = 5.0
        progress = 0.5

        result_a = apply_sharpness_blending(
            positive_eps=positive_eps,
            filtered_eps=filtered_eps,
            sharpness=2.0,
            diffusion_progress=progress,
        )
        result_b = apply_sharpness_blending(
            positive_eps=positive_eps,
            filtered_eps=filtered_eps,
            sharpness=5.0,
            diffusion_progress=progress,
        )
        assert result_a != pytest.approx(result_b)


# ---------------------------------------------------------------------------
# PatchSettingsState mutable runtime state tests
# ---------------------------------------------------------------------------


class TestPatchSettingsState:
    """PatchSettingsState holds mutable per-process runtime state."""

    def test_default_values(self) -> None:
        from modules.infrastructure.patch_system import PatchSettingsState

        state = PatchSettingsState()
        assert state.sharpness == pytest.approx(2.0)
        assert state.adm_scaler_end == pytest.approx(0.3)
        assert state.positive_adm_scale == pytest.approx(1.5)
        assert state.negative_adm_scale == pytest.approx(0.8)
        assert state.controlnet_softness == pytest.approx(0.25)
        assert state.adaptive_cfg == pytest.approx(7.0)
        assert state.global_diffusion_progress == pytest.approx(0.0)
        assert state.eps_record is None

    def test_global_diffusion_progress_is_mutable(self) -> None:
        from modules.infrastructure.patch_system import PatchSettingsState

        state = PatchSettingsState()
        state.global_diffusion_progress = 0.75
        assert state.global_diffusion_progress == pytest.approx(0.75)

    def test_eps_record_is_mutable(self) -> None:
        from modules.infrastructure.patch_system import PatchSettingsState

        state = PatchSettingsState()
        state.eps_record = "some_tensor"
        assert state.eps_record == "some_tensor"

    def test_from_patch_settings(self) -> None:
        """PatchSettingsState can be constructed from a frozen PatchSettings."""
        from modules.infrastructure.patch_system import PatchSettingsState
        from modules.infrastructure.sampler import PatchSettings

        settings = PatchSettings(sharpness=3.0, adaptive_cfg=9.0)
        state = PatchSettingsState.from_patch_settings(settings)
        assert state.sharpness == pytest.approx(3.0)
        assert state.adaptive_cfg == pytest.approx(9.0)
        assert state.global_diffusion_progress == pytest.approx(0.0)
