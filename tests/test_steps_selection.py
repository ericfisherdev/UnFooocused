"""RED tests for UNF-7: generation steps selection.

Tests are organized by acceptance criterion:
  AC1: User can see and adjust steps in the UI (config endpoint exposes default_steps)
  AC2: Steps value defaults to a sensible value from config
  AC3: Steps value is sent to /api/generate and passed to the generation backend
  AC4: Steps value is persisted in session state per model family
  AC5: Steps value is displayed in gallery metadata for generated images

Following outside-in TDD (Percival): start with API-level functional tests,
then drill into unit tests for the worker and config.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

try:
    import PIL  # noqa: F401

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_requires_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")


@pytest.fixture
def client():
    from ui.app import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper: build a minimal args list (same as test_async_worker)
# ---------------------------------------------------------------------------


def _minimal_args_list(*, overwrite_step: int = -1) -> list:
    """Build a minimal args list with a configurable overwrite_step value."""
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
        ["Fooocus V2"],  # style_selections
        "Speed",  # performance_selection
        "1024*1024",  # aspect_ratios_selection
        1,  # image_number (1 for faster tests)
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
        overwrite_step,  # overwrite_step
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


# ===========================================================================
# AC1: /api/config exposes default_steps
# ===========================================================================


class TestConfigExposesDefaultSteps:
    """AC1: GET /api/config must include default_steps so the UI can populate the control."""

    def test_config_response_contains_default_steps(self, client: TestClient):
        data = client.get("/api/config").json()
        assert "default_steps" in data, "/api/config must include 'default_steps'"

    def test_default_steps_is_an_integer(self, client: TestClient):
        data = client.get("/api/config").json()
        assert isinstance(data["default_steps"], int), "default_steps must be an integer"

    def test_default_steps_is_within_valid_range(self, client: TestClient):
        data = client.get("/api/config").json()
        assert 1 <= data["default_steps"] <= 150, "default_steps must be between 1 and 150"


# ===========================================================================
# AC2: AppConfig has default_steps with a sensible default
# ===========================================================================


class TestAppConfigDefaultSteps:
    """AC2: AppConfig must have a default_steps field defaulting to the performance mode's steps."""

    def test_app_config_has_default_steps_field(self):
        from modules.config import get_config

        cfg = get_config()
        assert hasattr(cfg, "default_steps"), "AppConfig must have a 'default_steps' field"

    def test_default_steps_equals_performance_steps(self):
        """When no override is in config.txt, default_steps should match the default performance mode."""
        from modules.config import get_config
        from modules.flags import Performance

        cfg = get_config()
        expected = Performance(cfg.default_performance).steps()
        assert cfg.default_steps == expected

    def test_config_txt_can_override_default_steps(self, tmp_path):
        """A user-provided default_steps in config.txt should override the computed default."""
        import json

        from modules.config import AppConfig, load_config

        config_file = tmp_path / "config.txt"
        config_file.write_text(json.dumps({"default_steps": 50}))
        raw = load_config(config_file)
        cfg = AppConfig.from_dict(raw)
        assert cfg.default_steps == 50


# ===========================================================================
# AC3: POST /api/generate passes steps to the backend as overwrite_step
# ===========================================================================


class TestGeneratePassesSteps:
    """AC3: When the user sends steps in the request body, _build_generate_args maps it to overwrite_step."""

    def test_build_generate_args_maps_steps_to_overwrite_step(self):
        """When 'steps' is provided in the request body, overwrite_step should be set to that value."""
        from ui.app import _build_generate_args

        body = {"prompt": "test", "steps": 50}
        args = _build_generate_args(body)

        from modules.async_worker import AsyncTask

        task = AsyncTask(args)
        assert task.overwrite_step == 50

    def test_build_generate_args_defaults_overwrite_step_to_negative_one(self):
        """When no 'steps' is provided, overwrite_step should remain -1 (no override)."""
        from ui.app import _build_generate_args

        body = {"prompt": "test"}
        args = _build_generate_args(body)

        from modules.async_worker import AsyncTask

        task = AsyncTask(args)
        assert task.overwrite_step == -1

    def test_generate_endpoint_passes_steps_through(self, client: TestClient):
        """POST /api/generate with steps=50 should queue a task with overwrite_step=50."""
        from modules.async_worker import async_tasks

        # Clear any leftover tasks
        async_tasks.clear()

        response = client.post("/api/generate", json={"prompt": "test", "steps": 50})
        assert response.status_code == 200

        # The task should have been queued
        assert len(async_tasks) >= 1
        task = async_tasks[len(async_tasks) - 1]
        assert task.overwrite_step == 50

        # Cleanup
        async_tasks.clear()


# ===========================================================================
# AC3 (continued): Worker respects overwrite_step
# ===========================================================================


@_requires_pil
class TestWorkerRespectsOverwriteStep:
    """AC3: When overwrite_step > 0, the worker uses it instead of performance-derived steps."""

    def test_worker_uses_overwrite_step_when_positive(self, tmp_path):
        """With overwrite_step=50, the worker should run 50 steps, not the performance default."""
        from modules.async_worker import AsyncTask, Worker

        args = _minimal_args_list(overwrite_step=50)
        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        # Count preview events — each step yields one preview.
        # With 1 image and 50 steps, expect 50 preview events.
        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) == 50

    def test_worker_uses_performance_steps_when_overwrite_is_negative(self, tmp_path):
        """With overwrite_step=-1, the worker should use the performance preset's steps (Speed=30)."""
        from modules.async_worker import AsyncTask, Worker

        args = _minimal_args_list(overwrite_step=-1)
        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        # Speed performance = 30 steps, 1 image => 30 preview events
        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) == 30

    def test_worker_uses_overwrite_step_of_10(self, tmp_path):
        """Explicit overwrite_step=10 should produce exactly 10 step previews."""
        from modules.async_worker import AsyncTask, Worker

        args = _minimal_args_list(overwrite_step=10)
        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) == 10


# ===========================================================================
# AC4: Steps value is persisted in session state
# ===========================================================================


class TestStepsInSessionState:
    """AC4: The steps value must round-trip through session state persistence."""

    @pytest.fixture(autouse=True)
    def _isolated_database(self, tmp_path, monkeypatch):
        import modules.session_state as ssm

        db_path = str(tmp_path / "test_session_states.db")
        monkeypatch.setattr(ssm, "_db_path", db_path)
        monkeypatch.setattr(ssm, "_connection", None)
        yield
        conn = ssm._connection
        if conn is not None:
            conn.close()

    def test_steps_round_trips_through_session_state(self):
        from modules.session_state import load_state, save_state

        save_state("sdxl", {"prompt": "test", "steps": 50})
        loaded = load_state("sdxl")
        assert loaded is not None
        assert loaded["steps"] == 50

    def test_steps_persisted_per_model_family(self):
        from modules.session_state import load_state, save_state

        save_state("sdxl", {"steps": 30})
        save_state("pony", {"steps": 50})
        assert load_state("sdxl")["steps"] == 30
        assert load_state("pony")["steps"] == 50


# ===========================================================================
# AC5: Steps value appears in gallery metadata
# ===========================================================================


@_requires_pil
class TestStepsInOutputMetadata:
    """AC5: The steps count must be embedded in the output image metadata."""

    def test_metadata_includes_steps_when_enabled(self, tmp_path):
        """When save_metadata_to_images is True, the PNG metadata should contain the steps count."""
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        args = _minimal_args_list(overwrite_step=50)
        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        assert len(task.results) == 1
        img = Image.open(task.results[0])
        params = img.info.get("parameters", "")
        assert "50" in params, f"Steps count '50' not found in metadata: {params!r}"

    def test_metadata_includes_steps_label(self, tmp_path):
        """The metadata should identify the steps value with a recognizable label."""
        from modules.async_worker import AsyncTask, Worker
        from PIL import Image

        args = _minimal_args_list(overwrite_step=25)
        task = AsyncTask(args)
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        assert len(task.results) == 1
        img = Image.open(task.results[0])
        params = img.info.get("parameters", "")
        # The metadata should contain "Steps: 25" or similar
        assert "steps" in params.lower(), f"Steps label not found in metadata: {params!r}"
