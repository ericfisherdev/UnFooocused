"""RED tests for UNF-27: Remove performance presets and style selector from UI.

Acceptance criteria:
  AC1: No performance preset buttons (Speed/Quality/Extreme Speed) appear in the UI.
  AC2: Generation always uses 30 steps by default (unless overridden by advanced settings).
  AC3: No style selector dropdown appears in the UI.
  AC4: The generation request no longer includes a performance mode or selected styles.
  AC5: The app starts and generates images successfully without performance presets or styles.
  AC6: All existing tests continue to pass (with updates for removed functionality).

Following outside-in TDD (Percival): functional tests at the API/template level first,
then unit tests for the domain logic changes.
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
# Helper
# ---------------------------------------------------------------------------


def _minimal_args_list() -> list:
    """Build a minimal args list WITHOUT performance_selection or style_selections.

    After UNF-27, AsyncTask should no longer expect these fields in the args
    list. This helper builds the post-removal args layout.
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
        # style_selections REMOVED (AC4)
        # performance_selection REMOVED (AC4)
        "1024*1024",  # aspect_ratios_selection
        1,  # image_number
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


# ===========================================================================
# AC1: No performance preset buttons in the UI
# ===========================================================================


class TestNoPerformancePresetsInUI:
    """AC1: The compose template must not contain performance preset buttons."""

    def test_compose_html_has_no_performance_radio_group(self):
        """The compose.html partial must not contain a performance radio group."""
        from pathlib import Path

        compose_path = Path(__file__).resolve().parents[1] / "ui" / "templates" / "partials" / "compose.html"
        content = compose_path.read_text(encoding="utf-8")
        assert "performance" not in content.lower() or "performance" not in content, (
            "compose.html still contains performance-related content"
        )

    def test_compose_html_has_no_speed_quality_buttons(self):
        """No Speed/Quality/Extreme Speed button labels should exist."""
        from pathlib import Path

        compose_path = Path(__file__).resolve().parents[1] / "ui" / "templates" / "partials" / "compose.html"
        content = compose_path.read_text(encoding="utf-8")
        for label in ["Speed", "Quality", "Extreme Speed"]:
            assert label not in content, f"compose.html still contains performance preset label: {label!r}"

    def test_compose_html_has_no_set_performance_handler(self):
        """No setPerformance click handler should exist in the template."""
        from pathlib import Path

        compose_path = Path(__file__).resolve().parents[1] / "ui" / "templates" / "partials" / "compose.html"
        content = compose_path.read_text(encoding="utf-8")
        assert "setPerformance" not in content, "compose.html still references setPerformance handler"


# ===========================================================================
# AC2: Generation always uses 30 steps by default
# ===========================================================================


class TestDefaultStepsIs30:
    """AC2: Without performance presets, the default step count is always 30."""

    def test_async_task_defaults_to_30_steps(self):
        """AsyncTask with no overwrite_step should use 30 steps."""
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert task.effective_steps == 30

    def test_async_task_no_performance_selection_attribute(self):
        """AsyncTask should no longer have a performance_selection attribute."""
        from modules.async_worker import AsyncTask

        task = AsyncTask(_minimal_args_list())
        assert not hasattr(task, "performance_selection"), (
            "AsyncTask still has performance_selection — it should be removed"
        )

    def test_overwrite_step_still_works(self):
        """When overwrite_step is positive, it takes precedence over the default."""
        from modules.async_worker import AsyncTask

        args = _minimal_args_list()
        # overwrite_step position: find it in the args list
        # After removing style_selections and performance_selection,
        # overwrite_step is at a different index. We set it directly on the task.
        task = AsyncTask(args)
        task.overwrite_step = 50
        assert task.effective_steps == 50

    @_requires_pil
    def test_worker_uses_30_steps_by_default(self, tmp_path):
        """Worker generates exactly 30 step previews for 1 image with default settings."""
        from modules.async_worker import AsyncTask, Worker

        task = AsyncTask(_minimal_args_list())
        worker = Worker(output_dir=str(tmp_path))
        worker.process_task(task)

        preview_events = [y for y in task.yields if y[0] == "preview"]
        assert len(preview_events) == 30


# ===========================================================================
# AC3: No style selector dropdown in the UI
# ===========================================================================


class TestNoStyleSelectorInUI:
    """AC3: The compose template must not contain a style selector."""

    def test_compose_html_has_no_styles_dropdown(self):
        """The styles dropdown div should be removed from compose.html."""
        from pathlib import Path

        compose_path = Path(__file__).resolve().parents[1] / "ui" / "templates" / "partials" / "compose.html"
        content = compose_path.read_text(encoding="utf-8")
        assert "styles-dropdown" not in content, "compose.html still contains styles-dropdown"

    def test_compose_html_has_no_styles_search_input(self):
        """The style search input should be removed."""
        from pathlib import Path

        compose_path = Path(__file__).resolve().parents[1] / "ui" / "templates" / "partials" / "compose.html"
        content = compose_path.read_text(encoding="utf-8")
        assert "styleSearch" not in content, "compose.html still contains styleSearch"

    def test_compose_html_has_no_toggle_style_handler(self):
        """No toggleStyle click handler should exist in the template."""
        from pathlib import Path

        compose_path = Path(__file__).resolve().parents[1] / "ui" / "templates" / "partials" / "compose.html"
        content = compose_path.read_text(encoding="utf-8")
        assert "toggleStyle" not in content, "compose.html still references toggleStyle handler"


# ===========================================================================
# AC4: Generation request no longer includes performance or styles
# ===========================================================================


class TestGenerateRequestNoPerformanceOrStyles:
    """AC4: _build_generate_args should not emit performance_selection or style_selections."""

    def test_build_generate_args_omits_performance_selection(self):
        """The args list from _build_generate_args should not include a Performance value."""
        from ui.app import _build_generate_args

        body = {"prompt": "test"}
        args = _build_generate_args(body)

        from modules.async_worker import AsyncTask

        task = AsyncTask(args)
        assert not hasattr(task, "performance_selection"), (
            "AsyncTask still has performance_selection from _build_generate_args"
        )

    def test_build_generate_args_omits_style_selections(self):
        """The args list from _build_generate_args should not include style_selections."""
        from ui.app import _build_generate_args

        body = {"prompt": "test"}
        args = _build_generate_args(body)

        from modules.async_worker import AsyncTask

        task = AsyncTask(args)
        assert not hasattr(task, "style_selections"), "AsyncTask still has style_selections from _build_generate_args"

    def test_api_config_omits_default_performance(self, client: TestClient):
        """GET /api/config should no longer include default_performance."""
        data = client.get("/api/config").json()
        assert "default_performance" not in data, "/api/config still includes default_performance"

    def test_api_config_omits_default_styles(self, client: TestClient):
        """GET /api/config should no longer include default_styles."""
        data = client.get("/api/config").json()
        assert "default_styles" not in data, "/api/config still includes default_styles"


# ===========================================================================
# AC4 (continued): quick-settings.js cleanup
# ===========================================================================


class TestQuickSettingsJsCleanup:
    """AC4: quick-settings.js should not have performance or style state."""

    def test_quick_settings_has_no_performance_property(self):
        """The quickSettings component should not declare a 'performance' property."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "quick-settings.js"
        content = js_path.read_text(encoding="utf-8")
        # Check for performance as a declared property (not just any mention)
        assert "performance:" not in content and "setPerformance" not in content, (
            "quick-settings.js still has performance-related code"
        )

    def test_quick_settings_has_no_style_properties(self):
        """The quickSettings component should not declare style-related properties."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "quick-settings.js"
        content = js_path.read_text(encoding="utf-8")
        for prop in [
            "selectedStyles",
            "stylesOpen",
            "styleSearch",
            "filteredStyles",
            "toggleStyle",
            "isStyleSelected",
            "selectedStyleCount",
        ]:
            assert prop not in content, f"quick-settings.js still contains style property: {prop!r}"


# ===========================================================================
# AC4 (continued): stores.js cleanup
# ===========================================================================


class TestStoresJsCleanup:
    """AC4: Alpine stores should not have performance or selectedStyles."""

    def test_generation_store_has_no_performance(self):
        """$store.generation should not have a 'performance' property."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "stores.js"
        content = js_path.read_text(encoding="utf-8")
        # Look for performance in the generation store section
        # This is a bit broad but captures the intent
        assert "performance:" not in content and "defaultPerformance" not in content, (
            "stores.js still has performance-related properties"
        )

    def test_generation_store_has_no_selected_styles(self):
        """$store.generation should not have a 'selectedStyles' property."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "stores.js"
        content = js_path.read_text(encoding="utf-8")
        assert "selectedStyles" not in content, "stores.js still has selectedStyles property"

    def test_config_store_has_no_default_styles(self):
        """$store.config should not have 'defaultStyles'."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "stores.js"
        content = js_path.read_text(encoding="utf-8")
        assert "defaultStyles" not in content, "stores.js still has defaultStyles"

    def test_config_store_has_no_default_performance(self):
        """$store.config should not have 'defaultPerformance'."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "stores.js"
        content = js_path.read_text(encoding="utf-8")
        assert "defaultPerformance" not in content, "stores.js still has defaultPerformance"

    def test_data_store_no_longer_fetches_styles(self):
        """$store.data should not fetch from /api/styles."""
        from pathlib import Path

        js_path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "stores.js"
        content = js_path.read_text(encoding="utf-8")
        assert "/api/styles" not in content, "stores.js still fetches /api/styles"


# ===========================================================================
# AC5: App starts and generates images successfully
# ===========================================================================


@_requires_pil
class TestAppWorksWithoutPresetsOrStyles:
    """AC5: Full generation works after removal of presets and styles."""

    @pytest.fixture
    def _output_dir(self, tmp_path, monkeypatch):
        import modules.config as config_module

        original_cfg = config_module.get_config()
        field_values = {f.name: getattr(original_cfg, f.name) for f in original_cfg.__dataclass_fields__.values()}
        field_values["path_outputs"] = str(tmp_path)
        patched_cfg = config_module.AppConfig(**field_values)
        config_module.set_config(patched_cfg)
        yield tmp_path
        config_module.set_config(original_cfg)

    @pytest.fixture
    def _clean_task_queue(self):
        import modules.async_worker as worker_module

        worker_module.async_tasks.clear()
        worker_module.current_task = None
        yield
        worker_module.async_tasks.clear()
        worker_module.current_task = None

    def test_generate_without_performance_or_styles(self, client: TestClient, _output_dir, _clean_task_queue):
        """POST /api/generate with no performance or styles produces images."""
        from modules.async_worker import Worker, async_tasks

        body = {
            "prompt": "test without presets or styles",
            "image_number": 1,
            "steps": 30,
            "seed": 42,
        }
        response = client.post("/api/generate", json=body)
        assert response.status_code == 200
        assert response.json()["queued"] is True

        worker = Worker(output_dir=str(_output_dir))
        while async_tasks:
            task = async_tasks.pop(0)
            worker.process_task(task)

        png_files = list(_output_dir.rglob("*.png"))
        assert len(png_files) == 1

    def test_generate_request_body_without_performance_field(self, client: TestClient, _output_dir, _clean_task_queue):
        """A generation request that omits performance and style_selections should succeed."""
        body = {"prompt": "no performance field"}
        response = client.post("/api/generate", json=body)
        assert response.status_code == 200
