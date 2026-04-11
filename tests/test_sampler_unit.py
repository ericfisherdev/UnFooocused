"""Unit tests for LdmSampler — no GPU, no real models.

Tests use fakes for ldm_patched's ksampler, sigma calculation, and
BrownianTreeNoiseSampler so we can verify sampling orchestration,
callback bridging, PatchSettings application, and protocol conformance
without loading any model weights.

Acceptance Criteria covered:
  AC1: modules/infrastructure/sampler.py exists implementing Sampler protocol
  AC2: All samplers in flags.SAMPLER_NAMES work
  AC3: All schedulers in flags.SCHEDULER_NAMES work
  AC4: cfg_scale, seed, and steps parameters are respected
  AC5: Step-level callbacks fire with correct step/total_steps values
  AC6: PatchSettings (sharpness, ADM scalers, controlnet_softness, adaptive_cfg) applied
  AC7: BrownianTreeNoiseSampler initialized with correct seed and sigma range
  AC9: Unit tests pass without GPU using fakes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from modules.flags import SAMPLER_NAMES, SCHEDULER_NAMES

# ---------------------------------------------------------------------------
# Fakes — stand-ins for ldm_patched infrastructure
# ---------------------------------------------------------------------------


@dataclass
class FakeSigmas:
    """Fake sigma tensor mimicking torch.Tensor behavior needed by sampler."""

    values: list[float] = field(default_factory=lambda: [14.6, 10.0, 5.0, 1.0, 0.0])

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return FakeSigmas(values=self.values[key])
        return self.values[key]

    def __gt__(self, other: float) -> list[bool]:
        return [v > other for v in self.values]

    def min(self) -> float:
        positive = [v for v in self.values if v > 0]
        return min(positive) if positive else 0.0

    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    def __len__(self) -> int:
        return len(self.values)


class FakeModel:
    """Minimal fake for a loaded SDXL model bundle."""

    def __init__(self) -> None:
        self.model = MagicMock()
        self.unet_with_lora = self


class RecordingKSampler:
    """Records all calls to ksampler and returns a fake latent.

    Allows tests to verify that the correct parameters were passed
    through from LdmSampler.sample() to core.ksampler().
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.return_latent: dict[str, Any] = {"samples": "fake_denoised_tensor"}

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        # Simulate step callbacks if callback_function provided
        callback_fn = kwargs.get("callback_function")
        steps = kwargs.get("steps", 30)
        if callback_fn is not None:
            for step in range(steps):
                callback_fn(step, None, None, steps, None)
        return self.return_latent


class RecordingSigmaCalculator:
    """Records sigma calculation calls and returns fake sigmas."""

    def __init__(self, sigmas: FakeSigmas | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._sigmas = sigmas or FakeSigmas()

    def __call__(self, *, sampler: str, model: Any, scheduler: str, steps: int, denoise: float) -> FakeSigmas:
        self.calls.append(
            {"sampler": sampler, "model": model, "scheduler": scheduler, "steps": steps, "denoise": denoise}
        )
        return self._sigmas


class RecordingBrownianTreeInit:
    """Records BrownianTreeNoiseSampler.global_init calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, latent_tensor: Any, sigma_min: float, sigma_max: float, *, seed: int, cpu: bool) -> None:
        self.calls.append(
            {"latent_tensor": latent_tensor, "sigma_min": sigma_min, "sigma_max": sigma_max, "seed": seed, "cpu": cpu}
        )


class RecordingPatchSettingsApplier:
    """Records PatchSettings application calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sampler_config(
    *,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = 12345,
    denoise: float = 1.0,
    width: int = 1024,
    height: int = 1024,
) -> Any:
    """Build a SamplerConfig from domain protocols."""
    from modules.domain.protocols import SamplerConfig

    return SamplerConfig(
        sampler_name=sampler_name,
        scheduler=scheduler,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        denoise=denoise,
        width=width,
        height=height,
    )


def _make_patch_settings(
    *,
    sharpness: float = 2.0,
    adm_scaler_end: float = 0.3,
    positive_adm_scale: float = 1.5,
    negative_adm_scale: float = 0.8,
    controlnet_softness: float = 0.25,
    adaptive_cfg: float = 7.0,
) -> Any:
    """Build a PatchSettings value object."""
    from modules.infrastructure.sampler import PatchSettings

    return PatchSettings(
        sharpness=sharpness,
        adm_scaler_end=adm_scaler_end,
        positive_adm_scale=positive_adm_scale,
        negative_adm_scale=negative_adm_scale,
        controlnet_softness=controlnet_softness,
        adaptive_cfg=adaptive_cfg,
    )


def _make_ldm_sampler(
    *,
    ksampler_fn: Any = None,
    sigma_calculator: Any = None,
    brownian_tree_init: Any = None,
    patch_settings_applier: Any = None,
    generate_empty_latent_fn: Any = None,
) -> Any:
    """Build an LdmSampler with injected fakes."""
    from modules.infrastructure.sampler import LdmSampler

    return LdmSampler(
        ksampler_fn=ksampler_fn or RecordingKSampler(),
        sigma_calculator=sigma_calculator or RecordingSigmaCalculator(),
        brownian_tree_init=brownian_tree_init or RecordingBrownianTreeInit(),
        patch_settings_applier=patch_settings_applier or RecordingPatchSettingsApplier(),
        generate_empty_latent_fn=generate_empty_latent_fn or (lambda w, h: {"samples": f"empty_{w}x{h}"}),
    )


# ---------------------------------------------------------------------------
# AC1: modules/infrastructure/sampler.py exists implementing Sampler protocol
# ---------------------------------------------------------------------------


class TestLdmSamplerExistsAndSatisfiesProtocol:
    """AC1: sampler.py exists and LdmSampler satisfies the Sampler protocol."""

    def test_module_importable(self) -> None:
        from modules.infrastructure.sampler import LdmSampler

        assert LdmSampler is not None

    def test_satisfies_sampler_protocol(self) -> None:
        from modules.domain.protocols import Sampler

        sampler = _make_ldm_sampler()
        assert isinstance(sampler, Sampler)

    def test_patch_settings_importable(self) -> None:
        from modules.infrastructure.sampler import PatchSettings

        assert PatchSettings is not None


# ---------------------------------------------------------------------------
# AC2: All samplers in flags.SAMPLER_NAMES work
# ---------------------------------------------------------------------------


class TestAllSamplersWork:
    """AC2: Every sampler name in SAMPLER_NAMES is accepted."""

    @pytest.mark.parametrize("sampler_name", SAMPLER_NAMES)
    def test_sampler_name_passed_to_ksampler(self, sampler_name: str) -> None:
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(sampler_name=sampler_name)
        model = FakeModel()

        sampler.sample(
            model=model,
            positive="pos_cond",
            negative="neg_cond",
            latent=None,
            config=config,
            callback=None,
        )

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["sampler_name"] == sampler_name


# ---------------------------------------------------------------------------
# AC3: All schedulers in flags.SCHEDULER_NAMES work
# ---------------------------------------------------------------------------


class TestAllSchedulersWork:
    """AC3: Every scheduler name in SCHEDULER_NAMES is accepted."""

    @pytest.mark.parametrize("scheduler_name", SCHEDULER_NAMES)
    def test_scheduler_name_passed_to_ksampler(self, scheduler_name: str) -> None:
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(scheduler=scheduler_name)
        model = FakeModel()

        sampler.sample(
            model=model,
            positive="pos_cond",
            negative="neg_cond",
            latent=None,
            config=config,
            callback=None,
        )

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["scheduler"] == scheduler_name


# ---------------------------------------------------------------------------
# AC4: cfg_scale, seed, and steps parameters are respected
# ---------------------------------------------------------------------------


class TestParametersRespected:
    """AC4: cfg_scale, seed, steps forwarded correctly to ksampler."""

    def test_cfg_scale_forwarded(self) -> None:
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(cfg_scale=12.5)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert recorder.calls[0]["cfg"] == pytest.approx(12.5)

    def test_seed_forwarded(self) -> None:
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(seed=99999)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert recorder.calls[0]["seed"] == 99999

    def test_steps_forwarded(self) -> None:
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(steps=50)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert recorder.calls[0]["steps"] == 50

    def test_denoise_forwarded(self) -> None:
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(denoise=0.7)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert recorder.calls[0]["denoise"] == pytest.approx(0.7)

    def test_empty_latent_generated_when_none(self) -> None:
        """When latent=None, LdmSampler generates empty latent from width/height."""
        recorder = RecordingKSampler()
        latent_gen_calls: list[tuple[int, int]] = []

        def fake_gen(w: int, h: int) -> dict[str, str]:
            latent_gen_calls.append((w, h))
            return {"samples": f"empty_{w}x{h}"}

        sampler = _make_ldm_sampler(ksampler_fn=recorder, generate_empty_latent_fn=fake_gen)
        config = _make_sampler_config(width=768, height=1024)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert latent_gen_calls == [(768, 1024)]

    def test_provided_latent_used_directly(self) -> None:
        """When latent is provided, it should be passed through without generating empty."""
        recorder = RecordingKSampler()
        latent_gen_calls: list[tuple[int, int]] = []

        def fake_gen(w: int, h: int) -> dict[str, str]:
            latent_gen_calls.append((w, h))
            return {"samples": f"empty_{w}x{h}"}

        sampler = _make_ldm_sampler(ksampler_fn=recorder, generate_empty_latent_fn=fake_gen)
        config = _make_sampler_config()
        model = FakeModel()
        provided_latent = {"samples": "user_provided"}

        sampler.sample(model=model, positive="p", negative="n", latent=provided_latent, config=config, callback=None)

        assert latent_gen_calls == []  # should not generate
        assert recorder.calls[0]["latent"] == provided_latent


# ---------------------------------------------------------------------------
# AC5: Step-level callbacks fire with correct step/total_steps values
# ---------------------------------------------------------------------------


class TestStepCallbacks:
    """AC5: Callbacks fire with correct step/total_steps values."""

    def test_callback_called_per_step(self) -> None:
        steps = 10
        callback_records: list[tuple[int, int, Any]] = []

        def domain_callback(step: int, total: int, preview_image: Any) -> None:
            callback_records.append((step, total, preview_image))

        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config(steps=steps)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=domain_callback)

        # RecordingKSampler calls callback_function with (step, x0, x, total_steps, preview)
        # LdmSampler bridges this to domain callback with (step+1, total, preview)
        assert len(callback_records) == steps
        for i, (step, total, _preview) in enumerate(callback_records):
            assert step == i + 1  # 1-indexed
            assert total == steps

    def test_none_callback_does_not_crash(self) -> None:
        sampler = _make_ldm_sampler()
        config = _make_sampler_config(steps=5)
        model = FakeModel()

        # Should not raise
        result = sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)
        assert result is not None


# ---------------------------------------------------------------------------
# AC6: PatchSettings applied before sampling
# ---------------------------------------------------------------------------


class TestPatchSettingsApplied:
    """AC6: PatchSettings (sharpness, ADM scalers, controlnet_softness, adaptive_cfg) applied."""

    def test_patch_settings_applied_before_ksampler(self) -> None:
        patch_applier = RecordingPatchSettingsApplier()
        recorder = RecordingKSampler()
        sampler = _make_ldm_sampler(ksampler_fn=recorder, patch_settings_applier=patch_applier)
        config = _make_sampler_config()
        model = FakeModel()
        settings = _make_patch_settings(sharpness=3.0, adaptive_cfg=9.0)

        sampler.sample(
            model=model,
            positive="p",
            negative="n",
            latent=None,
            config=config,
            callback=None,
            patch_settings=settings,
        )

        assert len(patch_applier.calls) == 1
        assert patch_applier.calls[0]["sharpness"] == pytest.approx(3.0)
        assert patch_applier.calls[0]["adaptive_cfg"] == pytest.approx(9.0)

    def test_default_patch_settings_when_none_provided(self) -> None:
        """When no PatchSettings given, defaults are applied."""
        patch_applier = RecordingPatchSettingsApplier()
        sampler = _make_ldm_sampler(patch_settings_applier=patch_applier)
        config = _make_sampler_config()
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert len(patch_applier.calls) == 1
        # Default sharpness is 2.0
        assert patch_applier.calls[0]["sharpness"] == pytest.approx(2.0)

    def test_patch_settings_all_fields_forwarded(self) -> None:
        patch_applier = RecordingPatchSettingsApplier()
        sampler = _make_ldm_sampler(patch_settings_applier=patch_applier)
        config = _make_sampler_config()
        model = FakeModel()
        settings = _make_patch_settings(
            sharpness=1.5,
            adm_scaler_end=0.4,
            positive_adm_scale=2.0,
            negative_adm_scale=0.5,
            controlnet_softness=0.3,
            adaptive_cfg=8.0,
        )

        sampler.sample(
            model=model, positive="p", negative="n", latent=None, config=config, callback=None, patch_settings=settings
        )

        call = patch_applier.calls[0]
        assert call["sharpness"] == pytest.approx(1.5)
        assert call["adm_scaler_end"] == pytest.approx(0.4)
        assert call["positive_adm_scale"] == pytest.approx(2.0)
        assert call["negative_adm_scale"] == pytest.approx(0.5)
        assert call["controlnet_softness"] == pytest.approx(0.3)
        assert call["adaptive_cfg"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# AC7: BrownianTreeNoiseSampler initialized with correct seed and sigma range
# ---------------------------------------------------------------------------


class TestBrownianTreeInitialization:
    """AC7: BrownianTreeNoiseSampler initialized with correct seed and sigma range."""

    def test_brownian_tree_initialized_with_seed(self) -> None:
        bt_init = RecordingBrownianTreeInit()
        sampler = _make_ldm_sampler(brownian_tree_init=bt_init)
        config = _make_sampler_config(seed=42)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert len(bt_init.calls) == 1
        assert bt_init.calls[0]["seed"] == 42

    def test_brownian_tree_initialized_with_sigma_range(self) -> None:
        sigmas = FakeSigmas(values=[14.6, 10.0, 5.0, 1.0, 0.0])
        bt_init = RecordingBrownianTreeInit()
        sigma_calc = RecordingSigmaCalculator(sigmas=sigmas)
        sampler = _make_ldm_sampler(brownian_tree_init=bt_init, sigma_calculator=sigma_calc)
        config = _make_sampler_config()
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert len(bt_init.calls) == 1
        # sigma_min = min of positive sigmas = 1.0
        assert bt_init.calls[0]["sigma_min"] == pytest.approx(1.0)
        # sigma_max = max of all sigmas = 14.6
        assert bt_init.calls[0]["sigma_max"] == pytest.approx(14.6)

    def test_sigma_calculator_called_with_correct_params(self) -> None:
        sigma_calc = RecordingSigmaCalculator()
        sampler = _make_ldm_sampler(sigma_calculator=sigma_calc)
        config = _make_sampler_config(sampler_name="dpmpp_2m_sde_gpu", scheduler="karras", steps=30, denoise=0.8)
        model = FakeModel()

        sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert len(sigma_calc.calls) == 1
        call = sigma_calc.calls[0]
        assert call["sampler"] == "dpmpp_2m_sde_gpu"
        assert call["scheduler"] == "karras"
        assert call["steps"] == 30
        assert call["denoise"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# AC9: Return value is a latent tensor dict
# ---------------------------------------------------------------------------


class TestReturnValue:
    """AC9: sample() returns the denoised latent tensor from ksampler."""

    def test_returns_ksampler_output(self) -> None:
        recorder = RecordingKSampler()
        recorder.return_latent = {"samples": "my_denoised_result"}
        sampler = _make_ldm_sampler(ksampler_fn=recorder)
        config = _make_sampler_config()
        model = FakeModel()

        result = sampler.sample(model=model, positive="p", negative="n", latent=None, config=config, callback=None)

        assert result == {"samples": "my_denoised_result"}


# ---------------------------------------------------------------------------
# PatchSettings value object tests
# ---------------------------------------------------------------------------


class TestPatchSettingsValueObject:
    """PatchSettings is a frozen dataclass with sensible defaults."""

    def test_default_values(self) -> None:
        settings = _make_patch_settings()
        assert settings.sharpness == pytest.approx(2.0)
        assert settings.adm_scaler_end == pytest.approx(0.3)
        assert settings.positive_adm_scale == pytest.approx(1.5)
        assert settings.negative_adm_scale == pytest.approx(0.8)
        assert settings.controlnet_softness == pytest.approx(0.25)
        assert settings.adaptive_cfg == pytest.approx(7.0)

    def test_frozen(self) -> None:
        from modules.infrastructure.sampler import PatchSettings

        settings = PatchSettings()
        with pytest.raises(AttributeError):
            settings.sharpness = 5.0  # type: ignore[misc]

    def test_custom_values(self) -> None:
        settings = _make_patch_settings(sharpness=5.0, adaptive_cfg=12.0)
        assert settings.sharpness == pytest.approx(5.0)
        assert settings.adaptive_cfg == pytest.approx(12.0)
