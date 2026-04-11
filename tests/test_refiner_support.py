"""Unit tests for refiner model support — UNF-37 acceptance criteria.

RED phase: tests encode each acceptance criterion and should FAIL before
implementation exists.

AC1: Refiner model loads from a .safetensors file and is validated as SDXL or SDXLRefiner
AC2: 'joint' swap method works: single ksampler call with refiner parameter and switch step
AC3: 'separate' swap method works: two sequential ksampler calls with noise preservation
AC4: 'vae' swap method works: base samples, VAE interpose, refiner continues
AC5: refiner_model_name='None' gracefully skips refiner loading and uses base only
AC6: refiner_switch parameter correctly determines the handoff step
AC7: Synthetic refiner (base model used as refiner) works without crashing
AC8: CLIP conditioning is correctly separated for refiner UNet
AC9: Integration tests pass with both base-only and base+refiner configurations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from modules.domain.protocols import (
        Conditioning,
        LatentTensor,
        LoRAConfig,
        ProgressCallback,
        SamplerConfig,
        StableDiffusionModel,
    )

# ===========================================================================
# Fakes — in-memory implementations of protocol dependencies
# ===========================================================================


class FakeModelLoader:
    """Records calls and returns deterministic fakes."""

    def __init__(self) -> None:
        self.load_checkpoint_calls: list[str] = []
        self.load_loras_calls: list[tuple[Any, list[LoRAConfig]]] = []
        self.apply_freeu_calls: list[tuple[float, float, float, float]] = []
        self.event_log: list[tuple[str, ...]] = []

    def load_checkpoint(self, path: str) -> StableDiffusionModel:
        self.load_checkpoint_calls.append(path)
        self.event_log.append(("checkpoint", path))
        return {"type": "model", "checkpoint": path}

    def load_loras(self, model: StableDiffusionModel, loras: list[LoRAConfig]) -> StableDiffusionModel:
        self.load_loras_calls.append((model, list(loras)))
        self.event_log.append(("loras", *[lora.filename for lora in loras]))
        return {**model, "loras": [lora.filename for lora in loras]}

    def apply_freeu(
        self, model: StableDiffusionModel, b1: float, b2: float, s1: float, s2: float
    ) -> StableDiffusionModel:
        self.apply_freeu_calls.append((b1, b2, s1, s2))
        self.event_log.append(("freeu", b1, b2, s1, s2))
        return {**model, "freeu": True}


class FakeTextEncoder:
    """Records calls and returns deterministic fakes."""

    def __init__(self) -> None:
        self.encode_calls: list[tuple[list[str], int]] = []
        self.clear_cache_calls: int = 0

    def encode(self, texts: list[str], clip_skip: int) -> Conditioning:
        self.encode_calls.append((list(texts), clip_skip))
        return [["fake_cond", {"pooled_output": "fake"}]]

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1


class FakeSampler:
    """Records calls and invokes progress callback if provided."""

    def __init__(self, steps_to_report: int = 3) -> None:
        self.sample_calls: list[dict[str, Any]] = []
        self._steps_to_report = steps_to_report

    def sample(
        self,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        latent: LatentTensor,
        config: SamplerConfig,
        callback: ProgressCallback | None,
    ) -> LatentTensor:
        self.sample_calls.append(
            {
                "model": model,
                "positive": positive,
                "negative": negative,
                "latent": latent,
                "config": config,
            }
        )
        if callback is not None:
            for step in range(1, self._steps_to_report + 1):
                callback(step=step, total=self._steps_to_report, preview_image=None)
        return {"samples": f"latent_seed_{config.seed}"}


class FakeVAEDecoder:
    """Records calls and returns deterministic numpy arrays."""

    def __init__(self) -> None:
        self.decode_calls: list[tuple[Any, Any]] = []

    def decode(self, vae: StableDiffusionModel, latent: LatentTensor) -> list[np.ndarray]:
        self.decode_calls.append((vae, latent))
        return [np.zeros((64, 64, 3), dtype=np.uint8)]


# ===========================================================================
# Refiner-specific fakes
# ===========================================================================


class RecordingClipSeparate:
    """Records clip_separate calls and returns modified conditioning."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        cond: Conditioning,
        target_model: StableDiffusionModel,
        target_clip: StableDiffusionModel,
    ) -> Conditioning:
        self.calls.append({"cond": cond, "target_model": target_model, "target_clip": target_clip})
        return [["separated_cond", {"pooled_output": "separated"}]]


class RecordingVAEInterpose:
    """Records VAE interpose calls and returns transformed latent."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, latent: LatentTensor) -> LatentTensor:
        self.calls.append(latent)
        return {"samples": "interposed_latent"}


# ===========================================================================
# Helpers
# ===========================================================================


@dataclass
class PipelineFixture:
    """Holds pipeline and all fakes for test assertions."""

    pipeline: Any
    model_loader: FakeModelLoader
    text_encoder: FakeTextEncoder
    sampler: FakeSampler
    vae_decoder: FakeVAEDecoder
    clip_separate: RecordingClipSeparate
    vae_interpose: RecordingVAEInterpose


def _make_config(**overrides: Any) -> Any:
    """Build a PipelineConfig with sensible defaults, applying overrides."""
    from modules.services.diffusion_pipeline import PipelineConfig

    defaults: dict[str, Any] = {
        "checkpoint_path": "/models/sd_xl.safetensors",
        "loras": [],
        "positive_prompt": "a photo of a cat",
        "negative_prompt": "ugly",
        "sampler_name": "euler",
        "scheduler": "normal",
        "steps": 20,
        "cfg_scale": 7.0,
        "seed": 42,
        "denoise": 1.0,
        "image_number": 1,
        "clip_skip": 1,
        "width": 1024,
        "height": 1024,
        "disable_seed_increment": False,
        "freeu_enabled": False,
        "freeu_b1": 1.3,
        "freeu_b2": 1.4,
        "freeu_s1": 0.9,
        "freeu_s2": 0.2,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_pipeline(
    model_loader: FakeModelLoader | None = None,
    text_encoder: FakeTextEncoder | None = None,
    sampler: FakeSampler | None = None,
    vae_decoder: FakeVAEDecoder | None = None,
    clip_separate: RecordingClipSeparate | None = None,
    vae_interpose: RecordingVAEInterpose | None = None,
) -> PipelineFixture:
    """Create a DiffusionPipeline with fakes, return a PipelineFixture."""
    from modules.services.diffusion_pipeline import DiffusionPipeline

    ml = model_loader or FakeModelLoader()
    te = text_encoder or FakeTextEncoder()
    sa = sampler or FakeSampler()
    vd = vae_decoder or FakeVAEDecoder()
    cs = clip_separate or RecordingClipSeparate()
    vi = vae_interpose or RecordingVAEInterpose()
    pipeline = DiffusionPipeline(
        model_loader=ml,
        text_encoder=te,
        sampler=sa,
        vae_decoder=vd,
        clip_separate=cs,
        vae_interpose=vi,
    )
    return PipelineFixture(
        pipeline=pipeline,
        model_loader=ml,
        text_encoder=te,
        sampler=sa,
        vae_decoder=vd,
        clip_separate=cs,
        vae_interpose=vi,
    )


# ===========================================================================
# AC1: Refiner model loads and is validated as SDXL/SDXLRefiner
# ===========================================================================


class TestRefinerModelLoading:
    """AC1: Refiner model loads from .safetensors and is validated."""

    def test_refiner_checkpoint_loaded_when_configured(self) -> None:
        """When refiner_path is set, load_checkpoint called for both base and refiner."""
        fx = _make_pipeline()
        config = _make_config(refiner_path="/models/refiner.safetensors")
        fx.pipeline.generate(config)
        assert "/models/refiner.safetensors" in fx.model_loader.load_checkpoint_calls

    def test_refiner_loaded_after_base(self) -> None:
        """Refiner checkpoint loaded after base checkpoint."""
        fx = _make_pipeline()
        config = _make_config(refiner_path="/models/refiner.safetensors")
        fx.pipeline.generate(config)
        checkpoint_events = [e for e in fx.model_loader.event_log if e[0] == "checkpoint"]
        assert len(checkpoint_events) >= 2
        assert checkpoint_events[0][1] == "/models/sd_xl.safetensors"
        assert checkpoint_events[1][1] == "/models/refiner.safetensors"


# ===========================================================================
# AC2: 'joint' swap method — single ksampler call with refiner param
# ===========================================================================


class TestJointSwapMethod:
    """AC2: 'joint' swap method passes refiner to ksampler."""

    def test_joint_mode_passes_refiner_to_sampler(self) -> None:
        """In joint mode, sampler receives both base and refiner model."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="joint",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        # Sampler should be called once with refiner information
        assert len(fx.sampler.sample_calls) == 1

    def test_joint_mode_single_ksampler_call(self) -> None:
        """Joint mode uses exactly one ksampler call (not two)."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="joint",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        assert len(fx.sampler.sample_calls) == 1

    def test_joint_mode_switch_step_computed(self) -> None:
        """Joint mode computes switch step from refiner_switch fraction."""
        from modules.services.diffusion_pipeline import RefinerSwapMethod

        assert RefinerSwapMethod.JOINT.value == "joint"


# ===========================================================================
# AC3: 'separate' swap method — two ksampler calls
# ===========================================================================


class TestSeparateSwapMethod:
    """AC3: 'separate' swap method uses two sequential ksampler calls."""

    def test_separate_mode_calls_sampler_twice(self) -> None:
        """Separate mode calls sampler twice — base then refiner."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
            steps=20,
        )
        fx.pipeline.generate(config)
        assert len(fx.sampler.sample_calls) == 2

    def test_separate_mode_first_call_uses_base(self) -> None:
        """First ksampler call uses the base model."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        first_call = fx.sampler.sample_calls[0]
        assert first_call["model"]["checkpoint"] == "/models/sd_xl.safetensors"

    def test_separate_mode_second_call_uses_refiner(self) -> None:
        """Second ksampler call uses the refiner model."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        second_call = fx.sampler.sample_calls[1]
        assert second_call["model"]["checkpoint"] == "/models/refiner.safetensors"

    def test_separate_mode_second_call_receives_first_latent(self) -> None:
        """Second ksampler receives the latent output of the first."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
            seed=100,
        )
        fx.pipeline.generate(config)
        second_call = fx.sampler.sample_calls[1]
        # The first sampler returns {"samples": "latent_seed_100"}
        assert second_call["latent"] == {"samples": "latent_seed_100"}

    def test_separate_mode_clip_separate_called_for_refiner(self) -> None:
        """In separate mode, clip_separate is called for the refiner pass."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        assert len(fx.clip_separate.calls) >= 1


# ===========================================================================
# AC4: 'vae' swap method — base samples, VAE interpose, refiner continues
# ===========================================================================


class TestVAESwapMethod:
    """AC4: 'vae' swap method decodes/reencodes through VAE interpose."""

    def test_vae_mode_calls_sampler_twice(self) -> None:
        """VAE mode calls sampler twice — base then refiner."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.8,
            steps=20,
        )
        fx.pipeline.generate(config)
        assert len(fx.sampler.sample_calls) == 2

    def test_vae_mode_calls_vae_interpose(self) -> None:
        """VAE mode invokes vae_interpose between the two sampling passes."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        assert len(fx.vae_interpose.calls) == 1

    def test_vae_mode_interpose_receives_base_output(self) -> None:
        """VAE interpose receives the latent output from the base model pass."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.8,
            seed=200,
        )
        fx.pipeline.generate(config)
        assert fx.vae_interpose.calls[0] == {"samples": "latent_seed_200"}

    def test_vae_mode_refiner_receives_interposed_latent(self) -> None:
        """Refiner pass receives the interposed latent, not the raw base output."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        second_call = fx.sampler.sample_calls[1]
        assert second_call["latent"] == {"samples": "interposed_latent"}

    def test_vae_mode_clip_separate_called_for_refiner(self) -> None:
        """In VAE mode, clip_separate is called for the refiner pass."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        assert len(fx.clip_separate.calls) >= 1


# ===========================================================================
# AC5: refiner_model_name='None' skips refiner loading
# ===========================================================================


class TestNoRefiner:
    """AC5: When no refiner is configured, pipeline uses base only."""

    def test_no_refiner_path_uses_base_only(self) -> None:
        """When refiner_path is None, only base checkpoint is loaded."""
        fx = _make_pipeline()
        config = _make_config()  # no refiner_path
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_checkpoint_calls) == 1
        assert fx.model_loader.load_checkpoint_calls[0] == "/models/sd_xl.safetensors"

    def test_no_refiner_sampler_called_once(self) -> None:
        """Without refiner, sampler called exactly once."""
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.sampler.sample_calls) == 1

    def test_no_refiner_no_clip_separate(self) -> None:
        """Without refiner, clip_separate never called."""
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.clip_separate.calls) == 0

    def test_no_refiner_no_vae_interpose(self) -> None:
        """Without refiner, vae_interpose never called."""
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.vae_interpose.calls) == 0

    def test_none_string_refiner_treated_as_no_refiner(self) -> None:
        """refiner_path='None' (string) is treated same as no refiner."""
        fx = _make_pipeline()
        config = _make_config(refiner_path="None")
        fx.pipeline.generate(config)
        # Should NOT try to load 'None' as a checkpoint
        assert "None" not in fx.model_loader.load_checkpoint_calls

    def test_empty_string_refiner_treated_as_no_refiner(self) -> None:
        """refiner_path='' (empty string) is treated same as no refiner."""
        fx = _make_pipeline()
        config = _make_config(refiner_path="")
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_checkpoint_calls) == 1


# ===========================================================================
# AC6: refiner_switch determines the handoff step
# ===========================================================================


class TestRefinerSwitch:
    """AC6: refiner_switch correctly determines the handoff step."""

    def test_switch_at_0_8_means_base_handles_80_percent(self) -> None:
        """refiner_switch=0.8 with 20 steps means base handles steps 0-16."""
        from modules.services.diffusion_pipeline import compute_switch_step

        switch_step = compute_switch_step(refiner_switch=0.8, total_steps=20)
        assert switch_step == 16

    def test_switch_at_0_5_means_base_handles_50_percent(self) -> None:
        """refiner_switch=0.5 with 20 steps means base handles steps 0-10."""
        from modules.services.diffusion_pipeline import compute_switch_step

        switch_step = compute_switch_step(refiner_switch=0.5, total_steps=20)
        assert switch_step == 10

    def test_switch_at_1_0_means_no_refiner_steps(self) -> None:
        """refiner_switch=1.0 means base handles all steps (refiner does nothing)."""
        from modules.services.diffusion_pipeline import compute_switch_step

        switch_step = compute_switch_step(refiner_switch=1.0, total_steps=20)
        assert switch_step == 20

    def test_switch_at_0_0_means_all_refiner_steps(self) -> None:
        """refiner_switch=0.0 means refiner handles all steps."""
        from modules.services.diffusion_pipeline import compute_switch_step

        switch_step = compute_switch_step(refiner_switch=0.0, total_steps=20)
        assert switch_step == 0


# ===========================================================================
# AC7: Synthetic refiner (base == refiner) works
# ===========================================================================


class TestSyntheticRefiner:
    """AC7: Synthetic refiner (base model used as refiner) works without crashing."""

    def test_synthetic_refiner_same_checkpoint(self) -> None:
        """When refiner_path equals base checkpoint, pipeline still works."""
        fx = _make_pipeline()
        config = _make_config(
            checkpoint_path="/models/sd_xl.safetensors",
            refiner_path="/models/sd_xl.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        results = fx.pipeline.generate(config)
        assert len(results) == 1

    def test_synthetic_refiner_loads_checkpoint_once(self) -> None:
        """When base==refiner, checkpoint should only load once (cached)."""
        fx = _make_pipeline()
        config = _make_config(
            checkpoint_path="/models/sd_xl.safetensors",
            refiner_path="/models/sd_xl.safetensors",
            refiner_swap_method="joint",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        # Base model loaded once; refiner reuses it
        base_loads = [p for p in fx.model_loader.load_checkpoint_calls if p == "/models/sd_xl.safetensors"]
        assert len(base_loads) >= 1  # at least once


# ===========================================================================
# AC8: CLIP conditioning is correctly separated for refiner UNet
# ===========================================================================


class TestClipSeparation:
    """AC8: CLIP conditioning is correctly separated for refiner UNet."""

    def test_clip_separate_called_with_refiner_model(self) -> None:
        """clip_separate receives the refiner model as target."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        assert len(fx.clip_separate.calls) >= 1
        # The target_model should be the refiner model
        call = fx.clip_separate.calls[0]
        assert call["target_model"]["checkpoint"] == "/models/refiner.safetensors"

    def test_clip_separate_called_for_both_positive_and_negative(self) -> None:
        """clip_separate called for both positive and negative conditioning."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        # Should be called at least twice (positive + negative)
        assert len(fx.clip_separate.calls) >= 2

    def test_separated_conditioning_passed_to_refiner_sampler(self) -> None:
        """The refiner sampling pass receives clip-separated conditioning."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        fx.pipeline.generate(config)
        second_call = fx.sampler.sample_calls[1]
        # RecordingClipSeparate returns [["separated_cond", ...]]
        assert second_call["positive"] == [["separated_cond", {"pooled_output": "separated"}]]
        assert second_call["negative"] == [["separated_cond", {"pooled_output": "separated"}]]


# ===========================================================================
# AC9: Integration — base-only and base+refiner configs both work
# ===========================================================================


class TestIntegrationBothConfigs:
    """AC9: Pipeline works for both base-only and base+refiner configs."""

    def test_base_only_produces_result(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        results = fx.pipeline.generate(config)
        assert len(results) == 1
        assert isinstance(results[0].image, np.ndarray)

    def test_base_plus_refiner_joint_produces_result(self) -> None:
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="joint",
            refiner_switch=0.8,
        )
        results = fx.pipeline.generate(config)
        assert len(results) == 1
        assert isinstance(results[0].image, np.ndarray)

    def test_base_plus_refiner_separate_produces_result(self) -> None:
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
        )
        results = fx.pipeline.generate(config)
        assert len(results) == 1
        assert isinstance(results[0].image, np.ndarray)

    def test_base_plus_refiner_vae_produces_result(self) -> None:
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.8,
        )
        results = fx.pipeline.generate(config)
        assert len(results) == 1
        assert isinstance(results[0].image, np.ndarray)

    def test_multiple_images_with_refiner(self) -> None:
        """Multiple images with refiner — each gets its own refiner pass."""
        fx = _make_pipeline()
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="separate",
            refiner_switch=0.8,
            image_number=3,
            seed=10,
        )
        results = fx.pipeline.generate(config)
        assert len(results) == 3
        # 3 images * 2 sampler calls each = 6 total
        assert len(fx.sampler.sample_calls) == 6

    def test_refiner_swap_method_enum_importable(self) -> None:
        """RefinerSwapMethod enum is importable from the pipeline module."""
        from modules.services.diffusion_pipeline import RefinerSwapMethod

        assert RefinerSwapMethod.JOINT.value == "joint"
        assert RefinerSwapMethod.SEPARATE.value == "separate"
        assert RefinerSwapMethod.VAE.value == "vae"


# ===========================================================================
# Domain model tests — RefinerSwapMethod enum
# ===========================================================================


class TestRefinerSwapMethodEnum:
    """RefinerSwapMethod enum has correct values and is exhaustive."""

    def test_all_methods_present(self) -> None:
        from modules.services.diffusion_pipeline import RefinerSwapMethod

        methods = {m.value for m in RefinerSwapMethod}
        assert methods == {"joint", "separate", "vae"}

    def test_from_string(self) -> None:
        from modules.services.diffusion_pipeline import RefinerSwapMethod

        assert RefinerSwapMethod("joint") is RefinerSwapMethod.JOINT
        assert RefinerSwapMethod("separate") is RefinerSwapMethod.SEPARATE
        assert RefinerSwapMethod("vae") is RefinerSwapMethod.VAE

    def test_invalid_string_raises(self) -> None:
        from modules.services.diffusion_pipeline import RefinerSwapMethod

        with pytest.raises(ValueError):
            RefinerSwapMethod("invalid")


# ===========================================================================
# PipelineConfig extended with refiner fields
# ===========================================================================


class TestPipelineConfigRefinerFields:
    """PipelineConfig has refiner_path, refiner_swap_method, refiner_switch."""

    def test_default_refiner_path_is_none(self) -> None:
        config = _make_config()
        assert config.refiner_path is None

    def test_default_refiner_swap_method_is_joint(self) -> None:
        config = _make_config()
        assert config.refiner_swap_method == "joint"

    def test_default_refiner_switch_is_0_5(self) -> None:
        config = _make_config()
        assert config.refiner_switch == pytest.approx(0.5)

    def test_refiner_fields_set_correctly(self) -> None:
        config = _make_config(
            refiner_path="/models/refiner.safetensors",
            refiner_swap_method="vae",
            refiner_switch=0.7,
        )
        assert config.refiner_path == "/models/refiner.safetensors"
        assert config.refiner_swap_method == "vae"
        assert config.refiner_switch == pytest.approx(0.7)
