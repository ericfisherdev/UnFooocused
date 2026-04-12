"""
UnFooocused - FastAPI Application

Serves the Alpine.js/HTMX/GSAP frontend via Jinja2 templates.
Shares the same backend modules (async_worker, config, lora_metadata).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from modules.async_worker import AsyncTask

import modules.config as config
import modules.lora_metadata as lora_metadata
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.heartbeat import update_heartbeat

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

app = FastAPI(title="UnFooocused", docs_url=None, redoc_url=None)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

_worker_stop = threading.Event()
_worker_ready = threading.Event()
_worker_error: BaseException | None = None
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _worker_loop() -> None:
    """Background thread that processes tasks from the async_tasks queue."""
    global _worker_error
    from modules.async_worker import Worker, async_tasks
    from modules.infrastructure.pipeline_factory import build_pipeline

    try:
        cfg = config.get_config()
        pipeline = build_pipeline()
        worker = Worker(output_dir=cfg.path_outputs, pipeline=pipeline)
    except Exception as exc:
        _worker_error = exc
        _worker_ready.set()
        return

    _worker_ready.set()

    while not _worker_stop.is_set():
        task = None
        with _worker_lock:
            if async_tasks:
                task = async_tasks.pop(0)
        if task is not None:
            try:
                worker.process_task(task)
            except Exception:
                logger.exception("Worker failed processing task")
        else:
            time.sleep(0.1)


@app.on_event("startup")
def _start_worker() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_stop.clear()
    _worker_ready.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="generation-worker")
    _worker_thread.start()

    # Wait for pipeline construction to complete so bootstrap failures
    # surface before the app reports ready to accept requests.
    _worker_ready.wait(timeout=120)
    if _worker_error is not None:
        raise RuntimeError(f"Worker bootstrap failed: {_worker_error}") from _worker_error
    logger.info("Generation worker thread started")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "base.html")


# ---------------------------------------------------------------------------
# LoRA Library APIs (migrated from Gradio routes in webui.py)
# ---------------------------------------------------------------------------


@app.post("/api/lora-library-rescan")
async def lora_library_rescan() -> dict:
    """Trigger a rescan of the LoRA library."""
    scanner = lora_metadata.get_scanner()
    if scanner.is_scanning:
        return {"success": False, "error": "Scan already in progress"}
    scanner.start_scan(blocking=False)
    return {"success": True}


@app.get("/api/lora-library-scan-status")
async def lora_library_scan_status() -> dict:
    """Get the current scan status."""
    scanner = lora_metadata.get_scanner()
    stats = scanner.scan_stats
    return {
        "is_scanning": stats["is_scanning"],
        "scan_complete": stats["scan_complete"],
        "files_scanned": stats["files_scanned"],
        "files_failed": stats["files_failed"],
        "total_indexed": stats["total_indexed"],
        "elapsed_time": stats["elapsed_time"],
    }


@app.get("/api/lora-library-data")
async def lora_library_data() -> list:
    """Get all LoRA metadata for the library/picker."""
    return lora_metadata.get_all_library_data()


@app.get("/api/lora-trigger-words")
async def lora_trigger_words(filename: Annotated[str, Query(description="LoRA filename or relative path")]) -> dict:
    """Get trigger words for a specific LoRA."""
    trigger_words = lora_metadata.get_trigger_words_for_filename(filename)
    return {"filename": filename, "trigger_words": trigger_words}


# ---------------------------------------------------------------------------
# Heartbeat (migrated from Gradio route in webui.py)
# ---------------------------------------------------------------------------


@app.post("/api/heartbeat")
async def heartbeat_ping() -> dict:
    """Receive a heartbeat ping from the browser client."""
    update_heartbeat()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Config & Model Data APIs
# ---------------------------------------------------------------------------


@app.get("/api/config")
async def get_app_config() -> dict:
    """Return UI-relevant config values."""
    cfg = config.get_config()
    return {
        "default_model": cfg.default_base_model_name,
        "default_refiner": cfg.default_refiner_model_name,
        "default_refiner_switch": cfg.default_refiner_switch,
        "default_aspect_ratio": cfg.default_aspect_ratio,
        "available_aspect_ratios": cfg.available_aspect_ratios,
        "default_image_number": cfg.default_image_number,
        "max_image_number": cfg.default_max_image_number,
        "default_output_format": cfg.default_output_format,
        "default_prompt": cfg.default_prompt,
        "default_prompt_negative": cfg.default_prompt_negative,
        "default_cfg_scale": cfg.default_cfg_scale,
        "default_sample_sharpness": cfg.default_sample_sharpness,
        "default_sampler": cfg.default_sampler,
        "default_scheduler": cfg.default_scheduler,
        "default_loras": cfg.default_loras,
        "default_loras_min_weight": cfg.default_loras_min_weight,
        "default_loras_max_weight": cfg.default_loras_max_weight,
        "default_max_lora_number": cfg.default_max_lora_number,
        "default_steps": cfg.default_steps,
    }


@app.get("/api/models")
async def get_models() -> dict:
    """Return available checkpoints, refiners, and VAEs."""
    cfg = config.get_config()
    return {
        "checkpoints": cfg.model_filenames,
        "loras": cfg.lora_filenames,
    }


@app.get("/api/styles")
async def get_styles() -> dict:
    """Return available style names."""
    from modules.sdxl_styles import legal_style_names

    return {"styles": legal_style_names}


@app.get("/api/samplers")
async def get_samplers() -> dict:
    """Return available sampler and scheduler names."""
    from modules.flags import sampler_list, scheduler_list

    return {"samplers": sampler_list, "schedulers": scheduler_list}


# ---------------------------------------------------------------------------
# Generation — POST /api/generate
# ---------------------------------------------------------------------------


def _build_generate_args(body: dict) -> list:
    """
    Build the positional args list that AsyncTask.__init__ expects.

    The args list is consumed via reverse() + pop() so we build it
    in the same order that webui.py's generate_clicked() does.
    Params not exposed by the new UI yet get sensible defaults.
    """
    from modules.flags import disabled

    cfg = config.get_config()

    loras_input = body.get("loras", [])
    # Pad to default_max_lora_number slots: (enabled, filename, weight)
    lora_args = []
    for i in range(cfg.default_max_lora_number):
        if i < len(loras_input):
            entry = loras_input[i]
            lora_args.extend([True, entry.get("filename", "None"), float(entry.get("weight", 1.0))])
        else:
            lora_args.extend([False, "None", 1.0])

    # ControlNet image slots (all empty for now)
    cn_args = []
    for _ in range(cfg.default_controlnet_image_count):
        cn_args.extend([None, 0.5, 1.0, disabled])  # img, stop, weight, type

    # Enhance tabs (all disabled for now)
    enhance_args = []
    for _ in range(cfg.default_enhance_tabs):
        enhance_args.extend(
            [
                False,  # enhance_enabled
                "",  # enhance_mask_dino_prompt_text
                "",  # enhance_prompt
                "",  # enhance_negative_prompt
                "u2net",  # enhance_mask_model
                "full",  # enhance_mask_cloth_category
                "sam_vit_b_01ec64",  # enhance_mask_sam_model
                0.25,  # enhance_mask_text_threshold
                0.3,  # enhance_mask_box_threshold
                0,  # enhance_mask_sam_max_detections
                False,  # enhance_inpaint_disable_initial_latent
                "None",  # enhance_inpaint_engine
                1.0,  # enhance_inpaint_strength
                0.618,  # enhance_inpaint_respective_field
                0,  # enhance_inpaint_erode_or_dilate
                False,  # enhance_mask_invert
            ]
        )

    args = [
        body.get("generate_image_grid", False),
        body.get("prompt", ""),
        body.get("negative_prompt", ""),
        body.get("aspect_ratios_selection", cfg.default_aspect_ratio),
        max(1, min(int(body.get("image_number", cfg.default_image_number)), cfg.default_max_image_number)),
        body.get("output_format", "png"),
        int(body.get("seed", -1)),
        body.get("read_wildcards_in_order", False),
        float(body.get("sharpness", cfg.default_sample_sharpness)),
        float(body.get("cfg_scale", cfg.default_cfg_scale)),
        body.get("base_model_name", cfg.default_base_model_name),
        body.get("refiner_model_name", cfg.default_refiner_model_name),
        float(body.get("refiner_switch", cfg.default_refiner_switch)),
        *lora_args,
        body.get("input_image_checkbox", False),
        body.get("current_tab", "uov"),
        body.get("uov_method", disabled),
        None,  # uov_input_image
        [],  # outpaint_selections
        None,  # inpaint_input_image (dict with image+mask)
        "",  # inpaint_additional_prompt
        None,  # inpaint_mask_image_upload
        # Developer/debug settings
        body.get("disable_preview", False),
        body.get("disable_intermediate_results", False),
        body.get("disable_seed_increment", False),
        body.get("black_out_nsfw", False),
        float(body.get("adm_scaler_positive", 1.5)),
        float(body.get("adm_scaler_negative", 0.8)),
        float(body.get("adm_scaler_end", 0.3)),
        float(body.get("adaptive_cfg", 7.0)),
        int(body.get("clip_skip", 2)),
        body.get("sampler_name", cfg.default_sampler),
        body.get("scheduler_name", cfg.default_scheduler),
        body.get("vae_name", "Default (model)"),
        int(body.get("steps", body.get("overwrite_step", cfg.default_steps))),
        int(body.get("overwrite_switch", -1)),
        int(body.get("overwrite_width", -1)),
        int(body.get("overwrite_height", -1)),
        float(body.get("overwrite_vary_strength", -1)),
        float(body.get("overwrite_upscale_strength", -1)),
        body.get("mixing_image_prompt_and_vary_upscale", False),
        body.get("mixing_image_prompt_and_inpaint", False),
        body.get("debugging_cn_preprocessor", False),
        body.get("skipping_cn_preprocessor", False),
        int(body.get("canny_low_threshold", 64)),
        int(body.get("canny_high_threshold", 128)),
        body.get("refiner_swap_method", "joint"),
        float(body.get("controlnet_softness", 0.25)),
        body.get("freeu_enabled", False),
        float(body.get("freeu_b1", 1.01)),
        float(body.get("freeu_b2", 1.02)),
        float(body.get("freeu_s1", 0.99)),
        float(body.get("freeu_s2", 0.95)),
        body.get("debugging_inpaint_preprocessor", False),
        body.get("inpaint_disable_initial_latent", False),
        body.get("inpaint_engine", "None"),
        float(body.get("inpaint_strength", 1.0)),
        float(body.get("inpaint_respective_field", 0.618)),
        body.get("inpaint_advanced_masking_checkbox", False),
        body.get("invert_mask_checkbox", False),
        int(body.get("inpaint_erode_or_dilate", 0)),
        body.get("save_final_enhanced_image_only", False),
        body.get("save_metadata_to_images", True),
        body.get("metadata_scheme", "fooocus"),
        *cn_args,
        # DINO / enhance
        body.get("debugging_dino", False),
        int(body.get("dino_erode_or_dilate", 0)),
        body.get("debugging_enhance_masks_checkbox", False),
        None,  # enhance_input_image
        body.get("enhance_checkbox", False),
        body.get("enhance_uov_method", disabled),
        body.get("enhance_uov_processing_order", "Before First Enhancement"),
        body.get("enhance_uov_prompt_type", "original"),
        *enhance_args,
    ]
    return args


@app.post("/api/generate")
async def generate(request: Request) -> dict:
    """Submit a generation job to the async task queue."""
    from modules.async_worker import AsyncTask, async_tasks

    body = await request.json()
    args = _build_generate_args(body)
    task = AsyncTask(args)
    async_tasks.append(task)
    return {"queued": True, "task_id": id(task)}


@app.post("/api/generate/stop")
async def generate_stop() -> dict:
    """Stop the current generation.

    The worker pops the active task from async_tasks before processing,
    so we must also check the module-level current_task reference.

    Sets ``task.last_stop = "stop"`` which the worker checks between
    images and steps. No ldm_patched dependency — UnFooocused is standalone.
    """
    import modules.async_worker as async_worker
    from modules.async_worker import async_tasks

    # Check the currently running task first (already popped from queue)
    if async_worker.current_task is not None and async_worker.current_task.processing:
        async_worker.current_task.last_stop = "stop"
        return {"stopped": True}

    # Fall back to queued tasks that may have started processing
    for task in list(async_tasks):
        if task.processing:
            task.last_stop = "stop"
            return {"stopped": True}

    # Clear queued tasks so the next job doesn't start immediately
    async_tasks.clear()

    return {"stopped": False}


# ---------------------------------------------------------------------------
# WebSocket — generation progress streaming
# ---------------------------------------------------------------------------


def _encode_preview_image(img) -> str | None:
    """Encode a preview image (numpy array or PIL Image) to base64 JPEG."""
    if img is None:
        return None
    try:
        import io

        import numpy as np
        from PIL import Image

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not isinstance(img, Image.Image):
            return None

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.debug("Failed to encode preview image", exc_info=True)
        return None


def _build_yield_message(flag: str, product) -> dict | None:
    """Build a JSON-serialisable message dict from a task yield entry."""
    if flag == "preview":
        percentage, text, img = product
        return {
            "type": "preview",
            "percentage": percentage,
            "text": text,
            "image": _encode_preview_image(img),
        }
    if flag == "results":
        return {
            "type": "results",
            "images": [str(p) if not isinstance(p, str) else p for p in product],
        }
    if flag == "finish":
        return {
            "type": "finish",
            "images": [str(p) if not isinstance(p, str) else p for p in product],
        }
    return None


def _effective_port(parts, default_port: int) -> int | None:
    """Extract port from URL parts, returning None on malformed input."""
    try:
        return parts.port or default_port
    except ValueError:
        return None


def _reject_mismatched_origin(websocket: WebSocket) -> bool:
    """Return True if the WebSocket origin doesn't match the host header."""
    origin = websocket.headers.get("origin")
    if not origin:
        return False

    origin_parts = urlsplit(origin)
    host_header = websocket.headers.get("host", "")
    host_parts = urlsplit(f"//{host_header}")

    origin_host = (origin_parts.hostname or "").lower()
    request_host = (host_parts.hostname or "").lower()
    origin_port = _effective_port(origin_parts, 443 if origin_parts.scheme == "https" else 80)
    request_port = _effective_port(host_parts, 443 if websocket.url.scheme == "wss" else 80)
    if origin_port is None or request_port is None:
        return True

    return (origin_host, origin_port) != (request_host, request_port)


def _find_processing_task(async_tasks: list[AsyncTask], current_task: AsyncTask | None) -> AsyncTask | None:
    """Return the first processing task from the queue or current_task."""
    for task in list(async_tasks):
        if task.processing:
            return task
    # Fallback: the worker pops the task from async_tasks
    # before processing, so check current_task as well.
    if current_task is not None and current_task.processing:
        return current_task
    return None


async def _drain_remaining_yields(
    task: AsyncTask, yield_index: int, send_fn: Callable[[dict], Awaitable[None]]
) -> None:
    """Forward any un-sent yields from a task that just finished."""
    for flag, product in task.yields[yield_index:]:
        msg = _build_yield_message(flag, product)
        if msg is not None:
            await send_fn(msg)


@app.websocket("/ws/generation")
async def ws_generation(websocket: WebSocket) -> None:
    """
    Stream generation progress to the client.

    Polls the active task's yields list and forwards them as JSON messages.
    Message types: preview, results, finish, heartbeat.
    """
    if _reject_mismatched_origin(websocket):
        logger.warning(
            "Rejected WebSocket from mismatched origin: %s (host: %s)",
            websocket.headers.get("origin"),
            websocket.headers.get("host", ""),
        )
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()

    async def _send_and_heartbeat(message: dict) -> None:
        await websocket.send_json(message)
        update_heartbeat()

    import modules.async_worker as async_worker
    from modules.async_worker import async_tasks

    try:
        yield_index = 0
        active_task = None
        idle_count = 0

        while True:
            if active_task is None or not active_task.processing:
                if active_task is not None:
                    await _drain_remaining_yields(active_task, yield_index, _send_and_heartbeat)
                active_task = _find_processing_task(async_tasks, async_worker.current_task)
                yield_index = 0

            # Refresh heartbeat so the backend knows a client is connected
            # during silent gaps (model loading, long sampling steps).
            update_heartbeat()

            current_yields = active_task.yields if active_task is not None else []
            if yield_index < len(current_yields):
                idle_count = 0
                flag, product = current_yields[yield_index]
                yield_index += 1

                msg = _build_yield_message(flag, product)
                if msg is not None:
                    await websocket.send_json(msg)

                if flag == "finish":
                    active_task = None
                    yield_index = 0
            else:
                idle_count += 1
                if idle_count >= 50:  # 50 * 100ms = 5s
                    await websocket.send_json({"type": "heartbeat"})
                    idle_count = 0

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Generated image file serving
# ---------------------------------------------------------------------------

if os.path.isdir(config.get_config().path_outputs):
    app.mount(
        "/outputs",
        StaticFiles(directory=config.get_config().path_outputs),
        name="outputs",
    )
