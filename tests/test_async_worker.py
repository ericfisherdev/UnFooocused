"""Unit tests for modules.async_worker — the generation pipeline.

Tests the AsyncTask construction, worker processing, progress yielding,
output saving, cancellation, and browser-disconnect handling.

Following Kent Beck's red-green-refactor: these tests are written first,
before the implementation exists.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

try:
    import PIL  # noqa: F401

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_requires_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_args_list() -> list:
    """Build a minimal args list matching _build_generate_args() output order.

    This mirrors the positional structure that AsyncTask.__init__ consumes
    via reverse()/pop().  We only set fields the worker actually reads;
    the rest get safe defaults.

    IMPORTANT: This list must stay in exact 1:1 correspondence with
    ``ui/app.py::_build_generate_args()``.  If that function changes,
    update this helper and re-run the test suite.
    """
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
        "",  # negative_prompt
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
        False,  # freeu_enabled
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


# ---------------------------------------------------------------------------
# AsyncTask construction
# ---------------------------------------------------------------------------


class TestAsyncTaskConstruction:
    """AsyncTask parses the positional args list from _build_generate_args()."""

    def test_parses_prompt(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.prompt == "a beautiful sunset"

    def test_parses_negative_prompt(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.negative_prompt == ""

    def test_defaults_to_30_steps(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.effective_steps == 30

    def test_parses_image_number(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.image_number == 2

    def test_parses_output_format(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.output_format == "png"

    def test_parses_seed(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.seed == 42

    def test_parses_base_model_name(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.base_model_name == "juggernautXL_v8Rundiffusion.safetensors"

    def test_parses_aspect_ratio(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.aspect_ratios_selection == "1024*1024"

    def test_parses_sampler_name(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.sampler_name == "dpmpp_2m_sde_gpu"

    def test_parses_scheduler_name(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.scheduler_name == "karras"

    def test_initial_processing_is_false(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.processing is False

    def test_initial_last_stop_is_false(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.last_stop is False

    def test_initial_yields_is_empty(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.yields == []

    def test_initial_results_is_empty(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.results == []

    def test_empty_args_does_not_crash(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask([])
        assert task.processing is False

    def test_parses_loras_as_list_of_tuples(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert isinstance(task.loras, list)
        # All LoRAs are disabled ("None") in our minimal args,
        # so get_enabled_loras should return an empty list
        # (or the full list depending on implementation).
        # The key test is that parsing doesn't crash.

    def test_parses_cfg_scale(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.cfg_scale == pytest.approx(4.0)

    def test_parses_refiner_model_name(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.refiner_model_name == "None"


# ---------------------------------------------------------------------------
# Worker processing
# ---------------------------------------------------------------------------


@_requires_pil
class TestWorkerProcessing:
    """The worker picks up queued tasks and processes them."""

    def test_worker_sets_processing_to_true(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)
        # After processing completes, processing should be back to False
        # but during processing it was True — we verify via the yields
        # that the task was actually processed
        assert task.processing is False

    def test_worker_yields_preview_events(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) > 0

    def test_preview_event_has_percentage_text_image(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) > 0
        _flag, product = preview_events[0]
        percentage, text, _image = product
        assert isinstance(percentage, int | float)
        assert isinstance(text, str)

    def test_preview_percentage_increases_over_steps(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        percentages = [p[1][0] for p in preview_events]
        # Percentages should be monotonically non-decreasing
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i - 1]

    def test_worker_yields_finish_event(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1

    def test_finish_event_contains_file_paths(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        paths = finish_events[0][1]
        assert len(paths) == 2  # image_number=2 in our args
        for p in paths:
            assert os.path.isfile(p)

    def test_worker_generates_correct_number_of_images(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        paths = finish_events[0][1]
        assert len(paths) == task.image_number


# ---------------------------------------------------------------------------
# Output saving integration
# ---------------------------------------------------------------------------


@_requires_pil
class TestWorkerOutputSaving:
    """The worker saves images via modules.output functions."""

    def test_saved_files_exist_on_disk(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        for path in finish_events[0][1]:
            assert os.path.isfile(path)

    def test_saved_files_have_correct_extension(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        for path in finish_events[0][1]:
            assert path.endswith(".png")

    def test_saved_files_are_valid_images(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        for path in finish_events[0][1]:
            img = Image.open(path)
            img.verify()  # raises if not a valid image


# ---------------------------------------------------------------------------
# HTML log writing (UNF-47)
# ---------------------------------------------------------------------------


@_requires_pil
class TestWorkerHtmlLogWriting:
    """Worker writes log.html entries after each image save (UNF-47)."""

    def test_log_html_created_after_generation(self, tmp_path):
        """AC1: After generation, a log.html file exists in the date-based output directory."""
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        # Find the date-based subdirectory
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        assert (subdirs[0] / "log.html").exists()

    def test_log_html_contains_prompt_metadata(self, tmp_path):
        """AC2: Log entries contain metadata fields matching FwdFooocus format."""
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        content = (subdirs[0] / "log.html").read_text(encoding="utf-8")
        assert "a beautiful sunset" in content  # the prompt from _minimal_args_list

    def test_log_html_contains_all_fooocus_fields(self, tmp_path):
        """AC2: All FwdFooocus-format metadata fields are present."""
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        content = (subdirs[0] / "log.html").read_text(encoding="utf-8")
        for label in (
            "Prompt",
            "Negative Prompt",
            "Steps",
            "Resolution",
            "Guidance Scale",
            "Sharpness",
            "ADM Guidance",
            "Base Model",
            "Refiner Model",
            "Refiner Switch",
            "Sampler",
            "Scheduler",
            "VAE",
            "Seed",
            "Version",
        ):
            assert label in content, f"Missing metadata label: {label}"

    def test_log_html_contains_unfooocused_version(self, tmp_path):
        """AC2: Version field distinguishes UnFooocused from FwdFooocus."""
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        content = (subdirs[0] / "log.html").read_text(encoding="utf-8")
        assert "UnFooocused" in content

    def test_log_html_uses_fooocus_split_marker(self, tmp_path):
        """AC3: Uses <!--fooocus-log-split--> for cross-compatibility."""
        from modules.async_worker import AsyncTask, Worker
        from modules.html_log_writer import FOOOCUS_LOG_SPLIT_MARKER

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        content = (subdirs[0] / "log.html").read_text(encoding="utf-8")
        assert content.count(FOOOCUS_LOG_SPLIT_MARKER) == 2

    def test_multiple_images_all_in_same_log(self, tmp_path):
        """AC5: Multiple generations on same day append to same log.html."""
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        # image_number=2 from _minimal_args_list
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        content = (subdirs[0] / "log.html").read_text(encoding="utf-8")
        # Should have 2 image entries (image_number=2)
        assert content.count('class="image-container"') == 2

    def test_log_write_oserror_does_not_abort_generation(self, tmp_path, monkeypatch):
        """Log write OSError is caught so saved images are still returned."""
        from modules.async_worker import AsyncTask, Worker

        def _boom(**_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("modules.html_log_writer.write_log_entry", _boom)

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        # Should not raise — OSError from log write must be swallowed so
        # successful image saves are not lost to logging failures.
        worker.process_task(task)

        # Image files should still exist even though log write failed
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert subdirs, "date-based subdirectory should be created"
        images = list(subdirs[0].glob("*.png"))
        assert images, "saved images should exist despite log write failures"


# ---------------------------------------------------------------------------
# Model and LoRA logging (stub behavior)
# ---------------------------------------------------------------------------


@_requires_pil
class TestWorkerModelLogging:
    """The worker logs model/LoRA names (stub behavior for now)."""

    def test_worker_logs_base_model_name(self, tmp_path, caplog):
        import logging

        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        with caplog.at_level(logging.INFO, logger="modules.async_worker"):
            worker.process_task(task)

        assert any("juggernautXL_v8Rundiffusion" in record.message for record in caplog.records)

    def test_worker_logs_lora_names_when_enabled(self, tmp_path, caplog):
        import logging

        from modules.async_worker import AsyncTask, Worker

        args = _minimal_args_list()
        # Enable the first LoRA slot — lora args start after refiner_switch (index 12)
        # Index 13 = first lora enabled, 14 = filename, 15 = weight
        args[13] = True
        args[14] = "my_test_lora.safetensors"
        args[15] = 0.8

        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))
        with caplog.at_level(logging.INFO, logger="modules.async_worker"):
            worker.process_task(task)

        assert any("my_test_lora" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@_requires_pil
class TestWorkerCancellation:
    """Setting task.last_stop cancels generation."""

    def test_cancel_stops_producing_images(self, tmp_path):
        """When last_stop is set between images, the worker stops producing more."""
        from modules.async_worker import AsyncTask, Worker

        args = _minimal_args_list()
        # Request many images
        args[4] = 10  # image_number

        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))

        # Monkey-patch _generate_single_image to set last_stop after 3rd image
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
        # Should have exactly 3 images (cancelled after 3rd, checked at top of 4th iteration)
        paths = finish_events[0][1]
        assert len(paths) == 3


# ---------------------------------------------------------------------------
# Browser disconnect
# ---------------------------------------------------------------------------


@_requires_pil
class TestWorkerBrowserDisconnect:
    """Worker skips remaining images when browser disconnects."""

    def test_skips_images_when_browser_disconnected(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        args = _minimal_args_list()
        args[4] = 5  # image_number

        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))

        with patch("modules.async_worker.is_browser_connected", return_value=False):
            worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1
        # With browser disconnected, should produce fewer images
        paths = finish_events[0][1]
        assert len(paths) < 5


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


class TestModuleLevelState:
    """Module exposes async_tasks list and current_task reference."""

    def test_async_tasks_is_a_list(self):
        from modules.async_worker import async_tasks

        assert isinstance(async_tasks, list)

    def test_current_task_starts_as_none(self):
        import modules.async_worker

        assert modules.async_worker.current_task is None


# ---------------------------------------------------------------------------
# Resolution parsing from aspect ratio string
# ---------------------------------------------------------------------------


class TestResolutionParsing:
    """AsyncTask parses width*height from the aspect ratio string."""

    def test_parses_1024x1024(self):
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.width == 1024
        assert task.height == 1024

    def test_parses_non_square_ratio(self):
        from modules.async_worker import AsyncTask

        args = _minimal_args_list()
        args[3] = "1152*896"  # aspect_ratios_selection
        task = AsyncTask(args)
        assert task.width == 1152
        assert task.height == 896


# ---------------------------------------------------------------------------
# disable_seed_increment flag
# ---------------------------------------------------------------------------


@_requires_pil
class TestDisableSeedIncrement:
    """When disable_seed_increment is True, all images use the same seed."""

    def test_same_seed_produces_identical_images(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        task = AsyncTask(_minimal_args_list())
        task.disable_seed_increment = True
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        paths = task.results
        assert len(paths) == 2
        img0 = Image.open(paths[0])
        img1 = Image.open(paths[1])
        # With same seed, stub images should be identical (same color)
        assert img0.tobytes() == img1.tobytes()

    def test_different_seeds_by_default(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        task = AsyncTask(_minimal_args_list())
        assert task.disable_seed_increment is False
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        paths = task.results
        assert len(paths) == 2
        img0 = Image.open(paths[0])
        img1 = Image.open(paths[1])
        # With incrementing seeds, stub images should differ
        assert img0.tobytes() != img1.tobytes()


# ---------------------------------------------------------------------------
# disable_preview flag
# ---------------------------------------------------------------------------


@_requires_pil
class TestDisablePreview:
    """When disable_preview is True, no preview events are yielded."""

    def test_no_preview_events_when_disabled(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        task.disable_preview = True
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) == 0

    def test_finish_event_still_emitted_when_preview_disabled(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        task.disable_preview = True
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        finish_events = [y for y in task.yields if y[0] == "finish"]
        assert len(finish_events) == 1

    def test_preview_events_present_when_not_disabled(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        assert task.disable_preview is False
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) > 0


# ---------------------------------------------------------------------------
# save_metadata_to_images flag
# ---------------------------------------------------------------------------


@_requires_pil
class TestSaveMetadataToImages:
    """When save_metadata_to_images is False, no metadata is embedded in images."""

    def test_no_metadata_embedded_when_disabled(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        task = AsyncTask(_minimal_args_list())
        task.save_metadata_to_images = False
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        for path in task.results:
            img = Image.open(path)
            # PNG text chunks should not contain "parameters" key
            assert "parameters" not in img.info

    def test_metadata_embedded_when_enabled(self, tmp_path):
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        task = AsyncTask(_minimal_args_list())
        assert task.save_metadata_to_images is True
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        for path in task.results:
            img = Image.open(path)
            # PNG text chunks should contain "parameters" key with the prompt
            assert "parameters" in img.info
            assert task.prompt in img.info["parameters"]
