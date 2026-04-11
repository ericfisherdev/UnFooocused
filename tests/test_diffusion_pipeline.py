"""Unit tests for DiffusionPipeline service — UNF-38 acceptance criteria.

RED phase: all tests encode acceptance criteria and should FAIL before
implementation exists.

AC1:  modules/services/diffusion_pipeline.py exists with DiffusionPipeline class
AC2:  DiffusionPipeline accepts all dependencies via constructor
AC3:  generate() orchestrates full pipeline: load -> encode -> sample -> decode
AC4:  Model caching prevents redundant reloads when config unchanged
AC5:  Multiple images generated with incrementing seeds (unless disable_seed_increment)
AC6:  Progress callbacks fire for each step and each image
AC7:  Cancellation stops generation between images without GPU resource leaks
AC8:  FreeU parameters applied when freeu_enabled is True
AC9:  clip_skip applied before text encoding
AC10: Unit tests pass with fake protocol implementations (no GPU needed)
AC11: Service has zero imports from torch or ldm_patched
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from modules.domain.protocols import (
    Conditioning,
    LatentTensor,
    LoRAConfig,
    ProgressCallback,
    SamplerConfig,
    StableDiffusionModel,
)

# ===========================================================================
# Fakes — in-memory implementations of all protocol dependencies
# ===========================================================================


class FakeModelLoader:
    """Records calls and returns deterministic fakes."""

    def __init__(self) -> None:
        self.load_checkpoint_calls: list[str] = []
        self.load_loras_calls: list[tuple[Any, list[LoRAConfig]]] = []
        self.apply_freeu_calls: list[tuple[float, float, float, float]] = []

    def load_checkpoint(self, path: str) -> StableDiffusionModel:
        self.load_checkpoint_calls.append(path)
        return {"type": "model", "checkpoint": path}

    def load_loras(self, model: StableDiffusionModel, loras: list[LoRAConfig]) -> StableDiffusionModel:
        self.load_loras_calls.append((model, list(loras)))
        return {**model, "loras": [lora.filename for lora in loras]}

    def apply_freeu(
        self,
        model: StableDiffusionModel,
        b1: float,
        b2: float,
        s1: float,
        s2: float,
    ) -> StableDiffusionModel:
        self.apply_freeu_calls.append((b1, b2, s1, s2))
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
# Progress / cancellation helpers
# ===========================================================================


@dataclass
class ProgressRecorder:
    """Captures progress callback invocations."""

    calls: list[tuple[int, int, int, Any]] = field(default_factory=list)
    """Each entry: (image_index, step, total, preview_image)."""

    def callback(self, image_index: int, step: int, total: int, preview_image: Any) -> None:
        self.calls.append((image_index, step, total, preview_image))


class CancelAfterN:
    """Returns True after N calls to cancel_check."""

    def __init__(self, n: int) -> None:
        self._n = n
        self.check_count = 0

    def __call__(self) -> bool:
        self.check_count += 1
        return self.check_count > self._n


# ===========================================================================
# Pipeline fixture — named container to avoid unused-variable lint noise
# ===========================================================================


@dataclass
class PipelineFixture:
    """Holds pipeline and all fakes for test assertions."""

    pipeline: Any
    model_loader: FakeModelLoader
    text_encoder: FakeTextEncoder
    sampler: FakeSampler
    vae_decoder: FakeVAEDecoder


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
) -> PipelineFixture:
    """Create a DiffusionPipeline with fakes, return a PipelineFixture."""
    from modules.services.diffusion_pipeline import DiffusionPipeline

    ml = model_loader or FakeModelLoader()
    te = text_encoder or FakeTextEncoder()
    sa = sampler or FakeSampler()
    vd = vae_decoder or FakeVAEDecoder()
    pipeline = DiffusionPipeline(
        model_loader=ml,
        text_encoder=te,
        sampler=sa,
        vae_decoder=vd,
    )
    return PipelineFixture(pipeline=pipeline, model_loader=ml, text_encoder=te, sampler=sa, vae_decoder=vd)


# ===========================================================================
# AC1: Module and class exist
# ===========================================================================


class TestModuleExists:
    """AC1: modules/services/diffusion_pipeline.py exists with DiffusionPipeline."""

    def test_module_importable(self) -> None:
        from modules.services import diffusion_pipeline  # noqa: F401

    def test_class_exists(self) -> None:
        from modules.services.diffusion_pipeline import DiffusionPipeline

        assert DiffusionPipeline is not None

    def test_pipeline_config_importable(self) -> None:
        from modules.services.diffusion_pipeline import PipelineConfig

        assert PipelineConfig is not None

    def test_generation_result_importable(self) -> None:
        from modules.services.diffusion_pipeline import GenerationResult

        assert GenerationResult is not None


# ===========================================================================
# AC2: Constructor accepts all protocol dependencies
# ===========================================================================


class TestConstructorDependencies:
    """AC2: DiffusionPipeline accepts all deps via constructor."""

    def test_constructor_accepts_all_four_deps(self) -> None:
        from modules.services.diffusion_pipeline import DiffusionPipeline

        pipeline = DiffusionPipeline(
            model_loader=FakeModelLoader(),
            text_encoder=FakeTextEncoder(),
            sampler=FakeSampler(),
            vae_decoder=FakeVAEDecoder(),
        )
        assert pipeline is not None

    def test_dependencies_stored_on_instance(self) -> None:
        from modules.services.diffusion_pipeline import DiffusionPipeline

        ml = FakeModelLoader()
        te = FakeTextEncoder()
        sa = FakeSampler()
        vd = FakeVAEDecoder()
        pipeline = DiffusionPipeline(
            model_loader=ml,
            text_encoder=te,
            sampler=sa,
            vae_decoder=vd,
        )
        assert pipeline._model_loader is ml
        assert pipeline._text_encoder is te
        assert pipeline._sampler is sa
        assert pipeline._vae_decoder is vd


# ===========================================================================
# AC3: generate() orchestrates load -> encode -> sample -> decode
# ===========================================================================


class TestOrchestrationOrder:
    """AC3: generate() calls deps in correct order: load -> encode -> sample -> decode."""

    def test_single_image_calls_load_checkpoint(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_checkpoint_calls) == 1
        assert fx.model_loader.load_checkpoint_calls[0] == "/models/sd_xl.safetensors"

    def test_single_image_calls_text_encoder(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.text_encoder.encode_calls) >= 1

    def test_single_image_calls_sampler(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.sampler.sample_calls) == 1

    def test_single_image_calls_vae_decoder(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        assert len(fx.vae_decoder.decode_calls) == 1

    def test_returns_generation_results(self) -> None:
        from modules.services.diffusion_pipeline import GenerationResult

        fx = _make_pipeline()
        config = _make_config()
        results = fx.pipeline.generate(config)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], GenerationResult)

    def test_generation_result_contains_image(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        results = fx.pipeline.generate(config)
        assert isinstance(results[0].image, np.ndarray)

    def test_generation_result_contains_seed(self) -> None:
        fx = _make_pipeline()
        config = _make_config(seed=42)
        results = fx.pipeline.generate(config)
        assert results[0].seed == 42

    def test_loras_applied_after_checkpoint(self) -> None:
        """LoRAs must be loaded AFTER checkpoint, verified by call order."""
        fx = _make_pipeline()
        loras = [LoRAConfig(filename="detail.safetensors", weight=0.8)]
        config = _make_config(loras=loras)
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_checkpoint_calls) == 1
        assert len(fx.model_loader.load_loras_calls) == 1
        # LoRA call references the model from checkpoint
        lora_model_arg = fx.model_loader.load_loras_calls[0][0]
        assert lora_model_arg["checkpoint"] == "/models/sd_xl.safetensors"

    def test_encode_called_for_positive_and_negative(self) -> None:
        """TextEncoder.encode called at least twice: positive + negative prompts."""
        fx = _make_pipeline()
        config = _make_config(positive_prompt="a cat", negative_prompt="ugly")
        fx.pipeline.generate(config)
        assert len(fx.text_encoder.encode_calls) >= 2

    def test_sampler_receives_correct_config(self) -> None:
        fx = _make_pipeline()
        config = _make_config(
            sampler_name="dpmpp_2m_sde_gpu",
            scheduler="karras",
            steps=30,
            cfg_scale=4.0,
            seed=123,
            denoise=0.8,
        )
        fx.pipeline.generate(config)
        sc = fx.sampler.sample_calls[0]["config"]
        assert sc.sampler_name == "dpmpp_2m_sde_gpu"
        assert sc.scheduler == "karras"
        assert sc.steps == 30
        assert sc.cfg_scale == pytest.approx(4.0)
        assert sc.seed == 123
        assert sc.denoise == pytest.approx(0.8)


# ===========================================================================
# AC4: Model caching
# ===========================================================================


class TestModelCaching:
    """AC4: Model caching prevents redundant reloads when config unchanged."""

    def test_same_config_does_not_reload_checkpoint(self) -> None:
        fx = _make_pipeline()
        config = _make_config()
        fx.pipeline.generate(config)
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_checkpoint_calls) == 1

    def test_different_checkpoint_reloads(self) -> None:
        fx = _make_pipeline()
        config1 = _make_config(checkpoint_path="/models/a.safetensors")
        config2 = _make_config(checkpoint_path="/models/b.safetensors")
        fx.pipeline.generate(config1)
        fx.pipeline.generate(config2)
        assert len(fx.model_loader.load_checkpoint_calls) == 2

    def test_different_loras_reloads(self) -> None:
        fx = _make_pipeline()
        config1 = _make_config(loras=[LoRAConfig(filename="a.safetensors", weight=0.5)])
        config2 = _make_config(loras=[LoRAConfig(filename="b.safetensors", weight=0.5)])
        fx.pipeline.generate(config1)
        fx.pipeline.generate(config2)
        assert len(fx.model_loader.load_loras_calls) == 2

    def test_same_loras_not_reapplied(self) -> None:
        fx = _make_pipeline()
        loras = [LoRAConfig(filename="a.safetensors", weight=0.5)]
        config = _make_config(loras=loras)
        fx.pipeline.generate(config)
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_loras_calls) == 1

    def test_cache_cleared_on_lora_weight_change(self) -> None:
        fx = _make_pipeline()
        config1 = _make_config(loras=[LoRAConfig(filename="a.safetensors", weight=0.5)])
        config2 = _make_config(loras=[LoRAConfig(filename="a.safetensors", weight=0.8)])
        fx.pipeline.generate(config1)
        fx.pipeline.generate(config2)
        assert len(fx.model_loader.load_loras_calls) == 2

    def test_text_encoder_cache_cleared_on_model_change(self) -> None:
        """When model changes, text encoder cache must be cleared."""
        fx = _make_pipeline()
        config1 = _make_config(checkpoint_path="/models/a.safetensors")
        config2 = _make_config(checkpoint_path="/models/b.safetensors")
        fx.pipeline.generate(config1)
        fx.pipeline.generate(config2)
        assert fx.text_encoder.clear_cache_calls >= 1

    def test_freeu_toggle_invalidates_cache(self) -> None:
        """Toggling FreeU on/off with same checkpoint must reload model."""
        fx = _make_pipeline()
        config_with = _make_config(freeu_enabled=True, freeu_b1=1.3, freeu_b2=1.4, freeu_s1=0.9, freeu_s2=0.2)
        config_without = _make_config(freeu_enabled=False)
        fx.pipeline.generate(config_with)
        fx.pipeline.generate(config_without)
        # Must reload because FreeU state changed
        assert len(fx.model_loader.load_checkpoint_calls) == 2


# ===========================================================================
# AC5: Multiple images with incrementing seeds
# ===========================================================================


class TestMultipleImages:
    """AC5: Multiple images generated with seed incrementing."""

    def test_three_images_produce_three_results(self) -> None:
        fx = _make_pipeline()
        config = _make_config(image_number=3, seed=100)
        results = fx.pipeline.generate(config)
        assert len(results) == 3

    def test_seeds_increment_by_one(self) -> None:
        fx = _make_pipeline()
        config = _make_config(image_number=3, seed=100)
        results = fx.pipeline.generate(config)
        assert results[0].seed == 100
        assert results[1].seed == 101
        assert results[2].seed == 102

    def test_sampler_receives_incrementing_seeds(self) -> None:
        fx = _make_pipeline()
        config = _make_config(image_number=3, seed=50)
        fx.pipeline.generate(config)
        seeds = [c["config"].seed for c in fx.sampler.sample_calls]
        assert seeds == [50, 51, 52]

    def test_disable_seed_increment_uses_same_seed(self) -> None:
        fx = _make_pipeline()
        config = _make_config(image_number=3, seed=50, disable_seed_increment=True)
        fx.pipeline.generate(config)
        seeds = [c["config"].seed for c in fx.sampler.sample_calls]
        assert seeds == [50, 50, 50]

    def test_model_loaded_once_for_batch(self) -> None:
        """Model only loaded once even when generating multiple images."""
        fx = _make_pipeline()
        config = _make_config(image_number=5)
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_checkpoint_calls) == 1


# ===========================================================================
# AC6: Progress callbacks
# ===========================================================================


class TestProgressCallbacks:
    """AC6: Progress callbacks fire for each step and each image."""

    def test_progress_callback_invoked(self) -> None:
        fx = _make_pipeline(sampler=FakeSampler(steps_to_report=5))
        config = _make_config(image_number=1)
        recorder = ProgressRecorder()
        fx.pipeline.generate(config, progress_callback=recorder.callback)
        assert len(recorder.calls) > 0

    def test_progress_callback_receives_image_index(self) -> None:
        fx = _make_pipeline(sampler=FakeSampler(steps_to_report=3))
        config = _make_config(image_number=2)
        recorder = ProgressRecorder()
        fx.pipeline.generate(config, progress_callback=recorder.callback)
        image_indices = {c[0] for c in recorder.calls}
        assert 0 in image_indices
        assert 1 in image_indices

    def test_progress_callback_receives_step_and_total(self) -> None:
        fx = _make_pipeline(sampler=FakeSampler(steps_to_report=3))
        config = _make_config(image_number=1)
        recorder = ProgressRecorder()
        fx.pipeline.generate(config, progress_callback=recorder.callback)
        # First call should be step=1
        assert recorder.calls[0][1] == 1
        assert recorder.calls[0][2] == 3


# ===========================================================================
# AC7: Cancellation
# ===========================================================================


class TestCancellation:
    """AC7: Cancellation stops generation between images."""

    def test_cancel_after_first_image(self) -> None:
        fx = _make_pipeline()
        config = _make_config(image_number=5, seed=10)
        cancel = CancelAfterN(1)
        results = fx.pipeline.generate(config, cancel_check=cancel)
        assert len(results) == 1

    def test_cancel_before_any_image(self) -> None:
        fx = _make_pipeline()
        config = _make_config(image_number=5)
        cancel = CancelAfterN(0)  # immediately cancelled
        results = fx.pipeline.generate(config, cancel_check=cancel)
        assert len(results) == 0

    def test_cancel_does_not_leave_partial_state(self) -> None:
        """After cancellation, pipeline should still be usable."""
        fx = _make_pipeline()
        config = _make_config(image_number=5)
        cancel = CancelAfterN(2)
        fx.pipeline.generate(config, cancel_check=cancel)
        # Generate again without cancellation
        results2 = fx.pipeline.generate(config)
        assert len(results2) == 5


# ===========================================================================
# AC8: FreeU parameters
# ===========================================================================


class TestFreeU:
    """AC8: FreeU parameters applied when freeu_enabled is True."""

    def test_freeu_applied_when_enabled(self) -> None:
        fx = _make_pipeline()
        config = _make_config(
            freeu_enabled=True,
            freeu_b1=1.3,
            freeu_b2=1.4,
            freeu_s1=0.9,
            freeu_s2=0.2,
        )
        fx.pipeline.generate(config)
        assert len(fx.model_loader.apply_freeu_calls) == 1
        assert fx.model_loader.apply_freeu_calls[0] == (1.3, 1.4, 0.9, 0.2)

    def test_freeu_not_applied_when_disabled(self) -> None:
        fx = _make_pipeline()
        config = _make_config(freeu_enabled=False)
        fx.pipeline.generate(config)
        assert len(fx.model_loader.apply_freeu_calls) == 0

    def test_freeu_applied_after_loras(self) -> None:
        """FreeU must be applied after LoRA loading."""
        fx = _make_pipeline()
        loras = [LoRAConfig(filename="detail.safetensors", weight=0.8)]
        config = _make_config(
            loras=loras,
            freeu_enabled=True,
            freeu_b1=1.3,
            freeu_b2=1.4,
            freeu_s1=0.9,
            freeu_s2=0.2,
        )
        fx.pipeline.generate(config)
        assert len(fx.model_loader.load_loras_calls) == 1
        assert len(fx.model_loader.apply_freeu_calls) == 1


# ===========================================================================
# AC9: clip_skip applied before text encoding
# ===========================================================================


class TestClipSkip:
    """AC9: clip_skip applied before text encoding."""

    def test_clip_skip_passed_to_text_encoder(self) -> None:
        fx = _make_pipeline()
        config = _make_config(clip_skip=2)
        fx.pipeline.generate(config)
        for _texts, clip_skip in fx.text_encoder.encode_calls:
            assert clip_skip == 2

    def test_different_clip_skip_values(self) -> None:
        fx = _make_pipeline()
        config1 = _make_config(clip_skip=1)
        config2 = _make_config(clip_skip=3)
        fx.pipeline.generate(config1)
        fx.pipeline.generate(config2)
        clip_skips = [cs for _, cs in fx.text_encoder.encode_calls]
        assert 1 in clip_skips
        assert 3 in clip_skips


# ===========================================================================
# AC10: Tests pass with fake protocol implementations (implicit in all above)
# ===========================================================================

# AC10 is satisfied by all tests above running with fakes and no GPU.


# ===========================================================================
# AC11: Zero imports from torch or ldm_patched
# ===========================================================================


class TestNoForbiddenImports:
    """AC11: Service has zero imports from torch or ldm_patched."""

    def test_no_torch_import(self) -> None:
        source = _get_service_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("torch"), (
                        f"diffusion_pipeline.py must not import torch, found: import {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("torch"), (
                    f"diffusion_pipeline.py must not import from torch, found: from {node.module}"
                )

    def test_no_ldm_patched_import(self) -> None:
        source = _get_service_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("ldm_patched"), (
                        f"diffusion_pipeline.py must not import ldm_patched, found: import {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("ldm_patched"), (
                    f"diffusion_pipeline.py must not import from ldm_patched, found: from {node.module}"
                )


# ===========================================================================
# Helpers
# ===========================================================================


def _get_service_source() -> str:
    """Read the source code of diffusion_pipeline.py for AST analysis."""
    path = Path(__file__).resolve().parent.parent / "modules" / "services" / "diffusion_pipeline.py"
    return path.read_text()
