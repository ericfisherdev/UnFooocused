"""Tests for Worker <-> DiffusionPipeline integration — UNF-40 acceptance criteria.

RED phase: all tests encode acceptance criteria and should FAIL before
implementation exists.

AC1:  Worker.process_task() uses DiffusionPipeline for real image generation
AC2:  PipelineConfig.from_task() correctly extracts all parameters from AsyncTask
AC3:  Step-level progress with latent preview images is streamed via task.yields
AC4:  Output images are saved via modules.output with correct metadata
AC5:  Cancellation via task.last_stop='stop' interrupts the pipeline between steps
AC6:  OOM errors are caught and reported as error yields (not server crashes)
AC7:  _generate_stub_image() still works when STUB_MODE=true (for CI/testing without GPU)
AC8:  Functional test: POST /api/generate -> WebSocket streams progress -> finish with valid image path
AC9:  The saved image is a real SDXL-generated image, not a solid color
AC10: All existing async_worker tests continue to pass (or are updated for the new behavior)
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

try:
    import PIL  # noqa: F401

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_requires_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")


# ---------------------------------------------------------------------------
# Module-wide fixture: keep heartbeat fresh so browser-disconnect logic
# doesn't skip images during long test suite runs.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _keep_heartbeat_alive():
    """Ensure is_browser_connected() returns True during all tests in this module."""
    from modules.heartbeat import update_heartbeat

    update_heartbeat()
    yield
    update_heartbeat()


# ===========================================================================
# Fakes — same protocol-satisfying fakes as test_diffusion_pipeline.py
# ===========================================================================


class FakeModelLoader:
    """Records calls and returns deterministic fakes."""

    def __init__(self) -> None:
        self.load_checkpoint_calls: list[str] = []
        self.load_loras_calls: list[tuple[Any, list]] = []
        self.apply_freeu_calls: list[tuple[float, float, float, float]] = []

    def load_checkpoint(self, path: str) -> Any:
        self.load_checkpoint_calls.append(path)
        return {"type": "model", "checkpoint": path}

    def load_loras(self, model: Any, loras: list) -> Any:
        self.load_loras_calls.append((model, list(loras)))
        return {**model, "loras": [lora.filename for lora in loras]}

    def apply_freeu(self, model: Any, b1: float, b2: float, s1: float, s2: float) -> Any:
        self.apply_freeu_calls.append((b1, b2, s1, s2))
        return {**model, "freeu": True}


class FakeTextEncoder:
    """Records calls and returns deterministic fakes."""

    def __init__(self) -> None:
        self.encode_calls: list[tuple[list[str], int]] = []
        self.clear_cache_calls: int = 0

    def encode(self, texts: list[str], clip_skip: int) -> Any:
        self.encode_calls.append((list(texts), clip_skip))
        return [["fake_cond", {"pooled_output": "fake"}]]

    def set_clip(self, clip: Any) -> None:
        pass

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1


class FakeSampler:
    """Records calls and invokes progress callback with preview images."""

    def __init__(self, steps_to_report: int = 3) -> None:
        self.sample_calls: list[dict[str, Any]] = []
        self._steps_to_report = steps_to_report

    def sample(self, model: Any, positive: Any, negative: Any, latent: Any, config: Any, callback: Any) -> Any:
        self.sample_calls.append({"config": config})
        if callback is not None:
            for step in range(1, self._steps_to_report + 1):
                # Provide a small preview image as numpy array
                preview = np.zeros((64, 64, 3), dtype=np.uint8)
                callback(step=step, total=self._steps_to_report, preview_image=preview)
        return {"samples": f"latent_seed_{config.seed}"}


class FakeVAEDecoder:
    """Returns deterministic numpy arrays — NOT solid color."""

    def __init__(self, width: int = 1024, height: int = 1024) -> None:
        self._width = width
        self._height = height

    def decode(self, vae: Any, latent: Any) -> list[np.ndarray]:
        # Generate a gradient image (not solid color) to prove real pipeline ran
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        for y in range(self._height):
            img[y, :, 0] = int(y / self._height * 255)  # red gradient
            img[y, :, 1] = 128  # constant green
            img[y, :, 2] = int((self._height - y) / self._height * 255)  # inverse blue
        return [img]


class OOMSampler:
    """Simulates CUDA OOM error."""

    def sample(self, model: Any, positive: Any, negative: Any, latent: Any, config: Any, callback: Any) -> Any:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class CrashingVAEDecoder:
    """Simulates the real UNF-91 bug: VAE decode crashes with AttributeError.

    This is NOT a RuntimeError, so it is not caught by the OOM-specific
    handling in Worker._generate_with_pipeline — it must be caught at the
    process_task boundary instead.
    """

    def decode(self, vae: Any, latent: Any) -> list[np.ndarray]:
        raise AttributeError("'VAE' object has no attribute 'load_device'")


class CrashesOnSecondCallVAEDecoder:
    """Succeeds for the first image in a batch, then crashes on the second.

    Used to verify that already-generated images survive an unexpected
    exception partway through a multi-image batch (UNF-91 CodeRabbit finding).
    """

    def __init__(self) -> None:
        self._calls = 0
        self._fake = FakeVAEDecoder()

    def decode(self, vae: Any, latent: Any) -> list[np.ndarray]:
        self._calls += 1
        if self._calls >= 2:
            raise AttributeError("'VAE' object has no attribute 'load_device'")
        return self._fake.decode(vae, latent)


# ===========================================================================
# Helpers — reuse from test_async_worker
# ===========================================================================


def _minimal_args_list() -> list:
    """Build a minimal args list matching _build_generate_args() output order."""
    from modules.config import get_config

    cfg = get_config()

    lora_args: list = []
    for _ in range(cfg.default_max_lora_number):
        lora_args.extend([False, "None", 1.0])

    cn_args: list = []
    for _ in range(cfg.default_controlnet_image_count):
        cn_args.extend([None, 0.5, 1.0, "Disabled"])

    enhance_args: list = []
    for _ in range(cfg.default_enhance_tabs):
        enhance_args.extend(
            [
                False,
                "",
                "",
                "",
                "u2net",
                "full",
                "sam_vit_b_01ec64",
                0.25,
                0.3,
                0,
                False,
                "None",
                1.0,
                0.618,
                0,
                False,
            ]
        )

    return [
        False,  # generate_image_grid
        "a beautiful sunset",  # prompt
        "bad quality",  # negative_prompt
        "1024*1024",  # aspect_ratios_selection
        2,  # image_number
        "png",  # output_format
        42,  # seed
        False,  # read_wildcards_in_order
        2.0,  # sharpness
        4.0,  # cfg_scale
        "juggernautXL_v8Rundiffusion.safetensors",  # base_model_name
        "None",  # refiner_model_name
        0.5,  # refiner_switch
        *lora_args,
        False,  # input_image_checkbox
        "uov",  # current_tab
        "Disabled",  # uov_method
        None,  # uov_input_image
        [],  # outpaint_selections
        None,  # inpaint_input_image
        "",  # inpaint_additional_prompt
        None,  # inpaint_mask_image_upload
        False,  # disable_preview
        False,  # disable_intermediate_results
        False,  # disable_seed_increment
        False,  # black_out_nsfw
        1.5,  # adm_scaler_positive
        0.8,  # adm_scaler_negative
        0.3,  # adm_scaler_end
        7.0,  # adaptive_cfg
        2,  # clip_skip
        "dpmpp_2m_sde_gpu",  # sampler_name
        "karras",  # scheduler_name
        "Default (model)",  # vae_name
        -1,  # overwrite_step
        -1,  # overwrite_switch
        -1,  # overwrite_width
        -1,  # overwrite_height
        -1.0,  # overwrite_vary_strength
        -1.0,  # overwrite_upscale_strength
        False,  # mixing_image_prompt_and_vary_upscale
        False,  # mixing_image_prompt_and_inpaint
        False,  # debugging_cn_preprocessor
        False,  # skipping_cn_preprocessor
        64,  # canny_low_threshold
        128,  # canny_high_threshold
        "joint",  # refiner_swap_method
        0.25,  # controlnet_softness
        True,  # freeu_enabled
        1.01,  # freeu_b1
        1.02,  # freeu_b2
        0.99,  # freeu_s1
        0.95,  # freeu_s2
        False,  # debugging_inpaint_preprocessor
        False,  # inpaint_disable_initial_latent
        "None",  # inpaint_engine
        1.0,  # inpaint_strength
        0.618,  # inpaint_respective_field
        False,  # inpaint_advanced_masking_checkbox
        False,  # invert_mask_checkbox
        0,  # inpaint_erode_or_dilate
        False,  # save_final_enhanced_image_only
        True,  # save_metadata_to_images
        "fooocus",  # metadata_scheme
        *cn_args,
        False,  # debugging_dino
        0,  # dino_erode_or_dilate
        False,  # debugging_enhance_masks_checkbox
        None,  # enhance_input_image
        False,  # enhance_checkbox
        "Disabled",  # enhance_uov_method
        "Before First Enhancement",  # enhance_uov_processing_order
        "original",  # enhance_uov_prompt_type
        *enhance_args,
    ]


def _make_worker_with_fakes(
    tmp_path: Any,
    sampler: Any | None = None,
    vae_decoder: Any | None = None,
    model_loader: Any | None = None,
) -> tuple:
    """Create a Worker injected with fake DiffusionPipeline dependencies.

    Returns (worker, model_loader, text_encoder, sampler, vae_decoder).
    """
    from modules.async_worker import Worker

    ml = model_loader or FakeModelLoader()
    te = FakeTextEncoder()
    sa = sampler or FakeSampler()
    vd = vae_decoder or FakeVAEDecoder()

    from modules.services.diffusion_pipeline import DiffusionPipeline

    pipeline = DiffusionPipeline(
        model_loader=ml,
        text_encoder=te,
        sampler=sa,
        vae_decoder=vd,
    )

    worker = Worker(output_dir=str(tmp_path), pipeline=pipeline)
    return worker, ml, te, sa, vd


# ===========================================================================
# AC1: Worker.process_task() uses DiffusionPipeline for real image generation
# ===========================================================================


@_requires_pil
class TestWorkerUsesPipeline:
    """AC1: Worker delegates to DiffusionPipeline instead of _generate_stub_image."""

    def test_worker_accepts_pipeline_in_constructor(self, tmp_path):
        """Worker constructor accepts a pipeline kwarg."""
        worker, *_ = _make_worker_with_fakes(tmp_path)
        assert worker is not None

    def test_worker_calls_pipeline_generate(self, tmp_path):
        """Worker calls pipeline.generate() during process_task."""
        from modules.async_worker import AsyncTask

        worker, _ml, _te, sa, _vd = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        # Sampler should have been called (proves pipeline ran)
        assert len(sa.sample_calls) > 0

    def test_worker_does_not_call_stub_when_pipeline_provided(self, tmp_path):
        """When a pipeline is injected, _generate_stub_image is NOT called."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(_minimal_args_list())

        with patch("modules.async_worker._generate_stub_image") as mock_stub:
            worker.process_task(task)
            mock_stub.assert_not_called()

    def test_worker_produces_output_files_via_pipeline(self, tmp_path):
        """Output files are created from pipeline-generated images."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        paths = finish_events[0][1]
        assert len(paths) == task.image_number
        for p in paths:
            assert os.path.isfile(p)


# ===========================================================================
# AC2: PipelineConfig.from_task() correctly extracts all parameters
# ===========================================================================


class TestPipelineConfigFromTask:
    """AC2: PipelineConfig.from_task() maps AsyncTask fields to PipelineConfig."""

    def test_from_task_exists(self):
        """PipelineConfig has a from_task class/static method."""
        from modules.services.diffusion_pipeline import PipelineConfig

        assert hasattr(PipelineConfig, "from_task")

    def test_from_task_extracts_prompt(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.positive_prompt == "a beautiful sunset"

    def test_from_task_extracts_negative_prompt(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.negative_prompt == "bad quality"

    def test_from_task_extracts_dimensions(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.width == 1024
        assert config.height == 1024

    def test_from_task_extracts_seed(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.seed == 42

    def test_from_task_extracts_sampler(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.sampler_name == "dpmpp_2m_sde_gpu"

    def test_from_task_extracts_scheduler(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.scheduler == "karras"

    def test_from_task_extracts_cfg_scale(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.cfg_scale == pytest.approx(4.0)

    def test_from_task_extracts_steps(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.steps == 30  # default effective_steps

    def test_from_task_extracts_clip_skip(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.clip_skip == 2

    def test_from_task_extracts_freeu_params(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.freeu_enabled is True
        assert config.freeu_b1 == pytest.approx(1.01)
        assert config.freeu_b2 == pytest.approx(1.02)
        assert config.freeu_s1 == pytest.approx(0.99)
        assert config.freeu_s2 == pytest.approx(0.95)

    def test_from_task_extracts_checkpoint_path(self):
        """from_task uses base_model_name as checkpoint_path."""
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.checkpoint_path == "juggernautXL_v8Rundiffusion.safetensors"

    def test_from_task_sets_image_number_to_one(self):
        """from_task sets image_number=1 because Worker iterates externally."""
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        # Worker generates one image at a time, calling pipeline per image
        assert config.image_number == 1

    def test_from_task_extracts_disable_seed_increment(self):
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        assert config.disable_seed_increment is False

    def test_from_task_extracts_loras(self):
        """from_task converts task.loras to list of LoRAConfig."""
        from modules.async_worker import AsyncTask
        from modules.domain.protocols import LoRAConfig
        from modules.services.diffusion_pipeline import PipelineConfig

        args = _minimal_args_list()
        # Enable the first LoRA
        args[13] = True
        args[14] = "my_lora.safetensors"
        args[15] = 0.8
        task = AsyncTask(args)
        config = PipelineConfig.from_task(task)
        assert len(config.loras) == 1
        assert config.loras[0] == LoRAConfig(filename="my_lora.safetensors", weight=0.8)

    def test_with_seed_preserves_all_other_fields(self):
        """with_seed() returns a copy with only seed and disable_seed_increment changed."""
        from modules.async_worker import AsyncTask
        from modules.services.diffusion_pipeline import PipelineConfig

        task = AsyncTask(_minimal_args_list())
        config = PipelineConfig.from_task(task)
        seeded = config.with_seed(999)
        assert seeded.seed == 999
        assert seeded.disable_seed_increment is True
        assert seeded.checkpoint_path == config.checkpoint_path
        assert seeded.positive_prompt == config.positive_prompt
        assert seeded.steps == config.steps
        assert seeded.width == config.width
        assert seeded.height == config.height


# ===========================================================================
# AC3: Step-level progress with latent preview images via task.yields
# ===========================================================================


@_requires_pil
class TestProgressStreaming:
    """AC3: Pipeline progress callbacks are bridged to task.yields as preview events."""

    def test_preview_events_have_preview_images(self, tmp_path):
        """Preview events include non-None preview images when pipeline provides them."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, sampler=FakeSampler(steps_to_report=5))
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) > 0

        # At least some preview events should have non-None preview images
        previews_with_images = [p for p in preview_events if p[1][2] is not None]
        assert len(previews_with_images) > 0, "No preview events had preview images"

    def test_preview_percentage_increases(self, tmp_path):
        """Preview percentage increases over steps."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, sampler=FakeSampler(steps_to_report=5))
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        percentages = [p[1][0] for p in preview_events]
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i - 1]

    def test_preview_text_includes_step_info(self, tmp_path):
        """Preview text includes step and image information."""
        from modules.async_worker import AsyncTask

        args = _minimal_args_list()
        args[4] = 1  # image_number=1 for simpler assertion
        worker, *_ = _make_worker_with_fakes(tmp_path, sampler=FakeSampler(steps_to_report=3))
        task = AsyncTask(args)
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) > 0
        _flag, (_percentage, text, _img) = preview_events[0]
        assert "Step" in text or "step" in text.lower()


# ===========================================================================
# AC4: Output images saved via modules.output with correct metadata
# ===========================================================================


@_requires_pil
class TestOutputSaving:
    """AC4: Output images saved with correct metadata."""

    def test_saved_image_has_correct_dimensions(self, tmp_path):
        """Output image matches requested resolution."""
        from modules.async_worker import AsyncTask
        from PIL import Image

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        for path in task.results:
            img = Image.open(path)
            assert img.size == (1024, 1024)

    def test_saved_image_has_metadata_when_enabled(self, tmp_path):
        """PNG metadata includes prompt when save_metadata_to_images is True."""
        from modules.async_worker import AsyncTask
        from PIL import Image

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(_minimal_args_list())
        assert task.save_metadata_to_images is True
        worker.process_task(task)

        for path in task.results:
            img = Image.open(path)
            assert "parameters" in img.info
            assert "a beautiful sunset" in img.info["parameters"]


# ===========================================================================
# AC5: Cancellation via task.last_stop='stop' interrupts pipeline
# ===========================================================================


@_requires_pil
class TestCancellationWithPipeline:
    """AC5: Cancellation interrupts pipeline-driven generation."""

    def test_cancel_stops_producing_images(self, tmp_path):
        """Setting last_stop='stop' mid-generation produces fewer images."""
        from modules.async_worker import AsyncTask

        args = _minimal_args_list()
        args[4] = 10  # image_number

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(args)

        original_generate = worker._generate_single_image
        call_count = 0

        def cancelling_generate(t, idx, steps):
            nonlocal call_count
            result = original_generate(t, idx, steps)
            call_count += 1
            if call_count >= 3:
                t.last_stop = "stop"
            return result

        worker._generate_single_image = cancelling_generate
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        assert len(finish_events[0][1]) == 3

    def test_cancel_during_step_stops_without_crash(self, tmp_path):
        """Pipeline cancel_check bridges to task.last_stop correctly."""
        from modules.async_worker import AsyncTask

        args = _minimal_args_list()
        args[4] = 5  # image_number

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(args)
        # Set cancel before processing starts
        task.last_stop = "stop"
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        # Should produce 0 images since cancelled immediately
        assert len(finish_events[0][1]) == 0


# ===========================================================================
# AC6: OOM errors caught and reported as error yields
# ===========================================================================


@_requires_pil
class TestOOMHandling:
    """AC6: OOM errors are caught and reported, not crashes."""

    def test_oom_yields_error_event(self, tmp_path):
        """CUDA OOM produces an error yield instead of crashing."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, sampler=OOMSampler())
        task = AsyncTask(_minimal_args_list())
        # Should not raise
        worker.process_task(task)

        error_events = [y for y in task.yields if y[0] == "error"]
        assert len(error_events) >= 1
        # Error message should mention memory
        error_msg = str(error_events[0][1])
        assert "memory" in error_msg.lower() or "oom" in error_msg.lower() or "cuda" in error_msg.lower()

    def test_oom_still_yields_finish(self, tmp_path):
        """Even after OOM, a finish event is yielded."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, sampler=OOMSampler())
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1

    def test_oom_does_not_crash_server(self, tmp_path):
        """Worker remains usable after OOM — can process another task."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, sampler=OOMSampler())
        task1 = AsyncTask(_minimal_args_list())
        worker.process_task(task1)

        # Replace sampler with working one for second task
        worker._pipeline._sampler = FakeSampler()
        task2 = AsyncTask(_minimal_args_list())
        worker.process_task(task2)

        # Second task should succeed
        finish_events = [y for y in task2.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        assert len(finish_events[0][1]) == 2  # image_number=2


# ===========================================================================
# UNF-91: unexpected (non-RuntimeError) exceptions must yield a terminal
# error instead of propagating and silently hanging the generation UI
# ===========================================================================


@_requires_pil
class TestUnexpectedExceptionHandling:
    """UNF-91: any exception during task processing yields a terminal error."""

    def test_unexpected_exception_yields_error_event(self, tmp_path):
        """An AttributeError (e.g. the real VAE decode bug) produces an error yield."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, vae_decoder=CrashingVAEDecoder())
        task = AsyncTask(_minimal_args_list())

        worker.process_task(task)  # must not raise

        error_events = [y for y in task.yields if y[0] == "error"]
        assert len(error_events) == 1
        assert "load_device" in str(error_events[0][1])

    def test_unexpected_exception_does_not_propagate(self, tmp_path):
        """process_task() must catch unexpected exceptions, not let them propagate."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, vae_decoder=CrashingVAEDecoder())
        task = AsyncTask(_minimal_args_list())

        worker.process_task(task)

    def test_unexpected_exception_resets_task_processing_state(self, tmp_path):
        """task.processing must be reset to False even after an unexpected crash."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, vae_decoder=CrashingVAEDecoder())
        task = AsyncTask(_minimal_args_list())

        worker.process_task(task)

        assert task.processing is False

    def test_worker_remains_usable_after_unexpected_exception(self, tmp_path):
        """Worker can still process subsequent tasks after an unexpected crash."""
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, vae_decoder=CrashingVAEDecoder())
        task1 = AsyncTask(_minimal_args_list())
        worker.process_task(task1)

        worker._pipeline._vae_decoder = FakeVAEDecoder()
        task2 = AsyncTask(_minimal_args_list())
        worker.process_task(task2)

        finish_events = [y for y in task2.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        assert len(finish_events[0][1]) == 2  # image_number=2

    def test_partial_results_preserved_when_later_image_crashes(self, tmp_path):
        """Images already generated before an unexpected crash must not be lost.

        task.image_number=2: image 0 succeeds and is saved to disk, then image 1
        raises. task.results must still reflect image 0's output even though the
        exception skips the normal end-of-loop "finish" yield entirely — only
        one terminal yield ("error") is emitted, so ws_generation never has to
        choose between two competing terminal messages.
        """
        from modules.async_worker import AsyncTask

        worker, *_ = _make_worker_with_fakes(tmp_path, vae_decoder=CrashesOnSecondCallVAEDecoder())
        task = AsyncTask(_minimal_args_list())

        worker.process_task(task)

        assert len(task.results) == 1

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 0

        error_events = [y for y in task.yields if y[0] == "error"]
        assert len(error_events) == 1


# ===========================================================================
# AC7: STUB_MODE=true still works for CI/testing without GPU
# ===========================================================================


@_requires_pil
class TestStubMode:
    """AC7: STUB_MODE=true falls back to stub generation."""

    def test_stub_mode_uses_stub_image(self, tmp_path):
        """When STUB_MODE=true, Worker uses _generate_stub_image."""
        from modules.async_worker import AsyncTask, Worker

        with patch.dict(os.environ, {"STUB_MODE": "true"}):
            worker = Worker(output_dir=str(tmp_path))
            task = AsyncTask(_minimal_args_list())
            worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        paths = finish_events[0][1]
        assert len(paths) == 2
        for p in paths:
            assert os.path.isfile(p)

    def test_stub_mode_produces_solid_color_images(self, tmp_path):
        """Stub mode images are solid color (not gradient from pipeline)."""
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        with patch.dict(os.environ, {"STUB_MODE": "true"}):
            worker = Worker(output_dir=str(tmp_path))
            task = AsyncTask(_minimal_args_list())
            worker.process_task(task)

        img = Image.open(task.results[0])
        # Solid color = top-left pixel matches a sampling of other pixels
        top_left = img.getpixel((0, 0))
        center = img.getpixel((img.width // 2, img.height // 2))
        bottom_right = img.getpixel((img.width - 1, img.height - 1))
        assert top_left == center == bottom_right

    def test_no_stub_mode_by_default(self, tmp_path):
        """Without STUB_MODE, Worker requires pipeline (or creates one)."""
        from modules.async_worker import Worker

        with patch.dict(os.environ, {}, clear=False):
            # Remove STUB_MODE if set
            os.environ.pop("STUB_MODE", None)
            worker = Worker(output_dir=str(tmp_path))
            # Worker should have a pipeline attribute (or accept one)
            assert hasattr(worker, "_pipeline") or hasattr(worker, "pipeline")


# ===========================================================================
# AC9: Saved image is NOT a solid color (real pipeline output)
# ===========================================================================


@_requires_pil
class TestNotSolidColor:
    """AC9: Pipeline-generated images are NOT solid color."""

    def test_pipeline_image_is_not_solid_color(self, tmp_path):
        """Images from pipeline have varied pixel values (not solid color)."""
        from modules.async_worker import AsyncTask
        from PIL import Image

        worker, *_ = _make_worker_with_fakes(tmp_path)
        task = AsyncTask(_minimal_args_list())
        worker.process_task(task)

        img = Image.open(task.results[0])
        # Pipeline output should have varied pixels (gradient from fake VAE)
        top_left = img.getpixel((0, 0))
        bottom_right = img.getpixel((img.width - 1, img.height - 1))
        assert top_left != bottom_right, "Image is solid color — pipeline did not generate real content"


# ===========================================================================
# AC10: Existing tests still pass — verified by running full suite
# ===========================================================================

# AC10 is verified by running the full test suite, which includes
# test_async_worker.py. Those tests must continue to pass. This is
# not a separate test class — it's a CI gate.
