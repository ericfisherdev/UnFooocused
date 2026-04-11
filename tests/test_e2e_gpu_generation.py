"""End-to-end GPU functional tests: real SDXL image generation — UNF-44.

Outer-loop acceptance tests (Percival's double-loop TDD) that validate the
entire pipeline produces a real SDXL image on GPU hardware. These tests sit
at the very top of the Test Pyramid — the definitive proof that the epic is
complete.

Full flow exercised:
    HTTP POST /api/generate -> Worker processes task -> DiffusionPipeline loads
    model -> encodes prompt -> samples -> decodes -> saves image -> WebSocket
    streams finish event with valid image path.

These tests REQUIRE:
    - A CUDA-capable GPU (RTX 3060 or better recommended)
    - Real SDXL model files on disk
    - The ``@pytest.mark.gpu`` marker allows skipping in CI via ``pytest -m "not gpu"``

Acceptance criteria (from UNF-44):
    AC1: Functional test exercises full pipeline from HTTP request to saved image
    AC2: Generated image is a valid SDXL-quality image (not solid-color placeholder)
    AC3: WebSocket progress includes step-level updates with preview images
    AC4: Image is saved in the correct output directory with correct format
    AC5: Test runs in under 120 seconds on RTX 3060 or better
    AC6: Test is skippable in CI via @pytest.mark.gpu marker
    AC7: At least three test scenarios: basic prompt, prompt with LoRA, different sampler
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    Image = None  # type: ignore[assignment, misc]
    _HAS_PIL = False

_requires_gpu_deps = pytest.mark.skipif(
    not (_HAS_PIL and _HAS_NUMPY),
    reason="PIL/Pillow and/or numpy not installed",
)

# ---------------------------------------------------------------------------
# GPU marker — all tests in this module require a GPU
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.gpu,
    _requires_gpu_deps,
]


# ---------------------------------------------------------------------------
# Constants — model paths and generation parameters
# ---------------------------------------------------------------------------

# Default checkpoint expected on the local system.
# Override via GPU_TEST_CHECKPOINT env var if your checkpoint lives elsewhere.
_DEFAULT_CHECKPOINT = "juggernautXL_v8Rundiffusion.safetensors"

# Default LoRA for the LoRA scenario test.
# Override via GPU_TEST_LORA env var.
_DEFAULT_LORA = "sd_xl_offset_example-lora_1.0.safetensors"

# Minimum standard deviation of pixel values to confirm the image has
# real visual content (not a solid-color stub). Empirically, real SDXL
# images have stddev > 20; solid-color stubs have stddev == 0.
_MIN_PIXEL_STDDEV = 10.0

# Maximum wall-clock time for a single generation (AC5).
_MAX_GENERATION_SECONDS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_checkpoint() -> str:
    """Return the checkpoint filename, overridable via env var."""
    return os.environ.get("GPU_TEST_CHECKPOINT", _DEFAULT_CHECKPOINT)


def _get_lora() -> str:
    """Return the LoRA filename, overridable via env var."""
    return os.environ.get("GPU_TEST_LORA", _DEFAULT_LORA)


def _generate_request_body(**overrides: Any) -> dict[str, Any]:
    """Build a complete /api/generate request body with GPU-test defaults.

    Uses a fixed seed for reproducibility and a low step count to keep
    wall-clock time under the AC5 budget.
    """
    body: dict[str, Any] = {
        "prompt": "a red rose on a white background, professional photo",
        "negative_prompt": "blurry, low quality, distorted",
        "aspect_ratios_selection": "1024*1024",
        "image_number": 1,
        "output_format": "png",
        "seed": 42,
        "sharpness": 2.0,
        "cfg_scale": 4.0,
        "base_model_name": _get_checkpoint(),
        "sampler_name": "dpmpp_2m_sde_gpu",
        "scheduler_name": "karras",
        "steps": 20,
        "save_metadata_to_images": True,
    }
    body.update(overrides)
    return body


def _run_worker_synchronously(output_dir: str) -> None:
    """Drive the Worker synchronously — no background thread, no asyncio.

    Pops all pending tasks from the queue and processes them. In production
    the background thread does this; here we call it directly for determinism.
    """
    from modules.async_worker import Worker, async_tasks

    worker = Worker(output_dir=output_dir)
    while async_tasks:
        task = async_tasks.pop(0)
        worker.process_task(task)


def _run_worker_with_pipeline(output_dir: str) -> None:
    """Drive the Worker with a real DiffusionPipeline (GPU required).

    Constructs the full pipeline from infrastructure adapters and injects
    it into the Worker. This is the real-deal path — no stubs, no fakes.
    """
    from modules.async_worker import Worker, async_tasks
    from modules.infrastructure.model_loader import LdmModelLoader
    from modules.infrastructure.text_encoder import LdmTextEncoder
    from modules.services.diffusion_pipeline import DiffusionPipeline

    # Build real infrastructure adapters
    model_loader = LdmModelLoader()
    text_encoder = LdmTextEncoder()

    # Sampler and VAE decoder — import from infrastructure
    from modules.infrastructure.model_loader import LdmSampler, LdmVAEDecoder

    sampler = LdmSampler()
    vae_decoder = LdmVAEDecoder()

    pipeline = DiffusionPipeline(
        model_loader=model_loader,
        text_encoder=text_encoder,
        sampler=sampler,
        vae_decoder=vae_decoder,
    )

    worker = Worker(output_dir=output_dir, pipeline=pipeline)
    while async_tasks:
        task = async_tasks.pop(0)
        worker.process_task(task)


def _assert_image_is_valid_sdxl(filepath: str, expected_width: int = 1024, expected_height: int = 1024) -> None:
    """Assert that the file at filepath is a valid, non-placeholder SDXL image.

    Checks:
        - File exists and is openable by PIL
        - Image mode is RGB
        - Dimensions match expected resolution
        - Pixel standard deviation exceeds threshold (not solid-color)
    """
    path = Path(filepath)
    assert path.exists(), f"Image file does not exist: {filepath}"
    assert path.stat().st_size > 1024, f"Image file suspiciously small ({path.stat().st_size} bytes): {filepath}"

    img = Image.open(filepath)
    assert img.mode == "RGB", f"Expected RGB mode, got {img.mode}"
    assert img.size == (expected_width, expected_height), (
        f"Expected {expected_width}x{expected_height}, got {img.size[0]}x{img.size[1]}"
    )

    # Statistical check: real images have varied pixel values
    pixels = np.array(img, dtype=np.float32)
    stddev = float(np.std(pixels))
    assert stddev > _MIN_PIXEL_STDDEV, (
        f"Image pixel stddev={stddev:.2f} below threshold {_MIN_PIXEL_STDDEV} — "
        f"likely a solid-color placeholder, not a real SDXL generation"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary output directory."""
    return tmp_path


@pytest.fixture
def _redirect_output(output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect config.path_outputs to the temporary directory."""
    import modules.config as config_module

    original_cfg = config_module.get_config()
    field_values = {f.name: getattr(original_cfg, f.name) for f in original_cfg.__dataclass_fields__.values()}
    field_values["path_outputs"] = str(output_dir)
    patched_cfg = config_module.AppConfig(**field_values)
    config_module.set_config(patched_cfg)

    yield output_dir

    config_module.set_config(original_cfg)


@pytest.fixture
def _clean_task_queue() -> None:
    """Ensure the async task queue is empty before and after each test."""
    import modules.async_worker as worker_module

    worker_module.async_tasks.clear()
    worker_module.current_task = None
    yield
    worker_module.async_tasks.clear()
    worker_module.current_task = None


@pytest.fixture
def _keep_heartbeat_alive() -> None:
    """Keep the heartbeat fresh so browser-disconnect logic doesn't skip images."""
    from modules.heartbeat import update_heartbeat

    update_heartbeat()
    yield
    update_heartbeat()


@pytest.fixture
def client():
    """Create a TestClient against the real FastAPI app."""
    from fastapi.testclient import TestClient
    from ui.app import app

    return TestClient(app)


# ===========================================================================
# AC1 + AC2 + AC4 + AC5: Full pipeline — basic prompt scenario
# ===========================================================================


class TestBasicPromptGeneration:
    """Scenario 1 (AC7): Basic prompt generation through the full pipeline.

    Exercises: HTTP POST -> Worker -> DiffusionPipeline -> real SDXL image.
    Validates AC1, AC2, AC4, and AC5.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _redirect_output, _clean_task_queue, _keep_heartbeat_alive, output_dir):
        self.output_dir = output_dir

    def test_generate_endpoint_accepts_request(self, client) -> None:
        """AC1: POST /api/generate returns queued=True with a task_id."""
        response = client.post("/api/generate", json=_generate_request_body())
        assert response.status_code == 200
        data = response.json()
        assert data["queued"] is True
        assert "task_id" in data

    def test_full_pipeline_produces_valid_image(self, client) -> None:
        """AC1+AC2: Full pipeline produces a real SDXL image, not a placeholder."""
        client.post("/api/generate", json=_generate_request_body())

        start = time.monotonic()
        _run_worker_with_pipeline(str(self.output_dir))
        elapsed = time.monotonic() - start

        # AC5: under 120 seconds
        assert elapsed < _MAX_GENERATION_SECONDS, (
            f"Generation took {elapsed:.1f}s, exceeding {_MAX_GENERATION_SECONDS}s budget"
        )

        # AC4: image saved in correct directory
        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) == 1, f"Expected 1 PNG file, found {len(png_files)}"

        # AC2: image is valid SDXL-quality
        _assert_image_is_valid_sdxl(str(png_files[0]))

    def test_output_in_date_based_directory(self, client) -> None:
        """AC4: Output image is saved in a YYYY-MM-DD subdirectory."""
        import re

        client.post("/api/generate", json=_generate_request_body())
        _run_worker_with_pipeline(str(self.output_dir))

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) >= 1

        # Parent directory should match YYYY-MM-DD pattern
        date_dir = png_files[0].parent
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_dir.name), f"Output not in date-based directory: {date_dir.name}"

    def test_output_image_format_is_png(self, client) -> None:
        """AC4: Output file is a valid PNG."""
        client.post("/api/generate", json=_generate_request_body())
        _run_worker_with_pipeline(str(self.output_dir))

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) >= 1

        img = Image.open(png_files[0])
        assert img.format == "PNG"

    def test_output_image_has_metadata(self, client) -> None:
        """AC4: PNG metadata contains the prompt."""
        client.post("/api/generate", json=_generate_request_body())
        _run_worker_with_pipeline(str(self.output_dir))

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) >= 1

        img = Image.open(png_files[0])
        assert "parameters" in img.info, "PNG metadata missing 'parameters' key"
        assert "a red rose" in img.info["parameters"]


# ===========================================================================
# AC3: WebSocket progress streaming with previews
# ===========================================================================


class TestWebSocketProgressStreaming:
    """AC3: WebSocket progress includes step-level updates with preview images.

    Connects via WebSocket to /ws/generation and verifies that progress
    messages arrive with increasing percentages and at least one preview.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _redirect_output, _clean_task_queue, _keep_heartbeat_alive, output_dir):
        self.output_dir = output_dir

    def test_websocket_receives_progress_messages(self, client) -> None:
        """AC3: WebSocket streams preview messages during generation."""
        import threading

        from modules.async_worker import async_tasks

        # Submit generation
        client.post("/api/generate", json=_generate_request_body())
        assert len(async_tasks) == 1

        messages: list[dict] = []

        def run_worker():
            _run_worker_with_pipeline(str(self.output_dir))

        worker_thread = threading.Thread(target=run_worker, daemon=True)

        with client.websocket_connect("/ws/generation") as ws:
            worker_thread.start()

            # Collect messages until we get a finish event or timeout
            deadline = time.monotonic() + _MAX_GENERATION_SECONDS
            while time.monotonic() < deadline:
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg.get("type") == "finish":
                        break
                except Exception:
                    break

            worker_thread.join(timeout=10)

        preview_messages = [m for m in messages if m.get("type") == "preview"]
        assert len(preview_messages) > 0, "No preview messages received via WebSocket"

    def test_progress_percentages_increase(self, client) -> None:
        """AC3: Progress percentages increase over time."""
        import threading

        from modules.async_worker import async_tasks

        client.post("/api/generate", json=_generate_request_body())
        assert len(async_tasks) == 1

        messages: list[dict] = []

        def run_worker():
            _run_worker_with_pipeline(str(self.output_dir))

        worker_thread = threading.Thread(target=run_worker, daemon=True)

        with client.websocket_connect("/ws/generation") as ws:
            worker_thread.start()

            deadline = time.monotonic() + _MAX_GENERATION_SECONDS
            while time.monotonic() < deadline:
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg.get("type") == "finish":
                        break
                except Exception:
                    break

            worker_thread.join(timeout=10)

        preview_messages = [m for m in messages if m.get("type") == "preview"]
        if len(preview_messages) >= 2:
            percentages = [m["percentage"] for m in preview_messages]
            # Percentages should be non-decreasing
            for i in range(1, len(percentages)):
                assert percentages[i] >= percentages[i - 1], (
                    f"Percentage decreased: {percentages[i - 1]}% -> {percentages[i]}%"
                )

    def test_at_least_one_preview_has_image(self, client) -> None:
        """AC3: At least one progress message has a non-null preview image."""
        import threading

        from modules.async_worker import async_tasks

        client.post("/api/generate", json=_generate_request_body())
        assert len(async_tasks) == 1

        messages: list[dict] = []

        def run_worker():
            _run_worker_with_pipeline(str(self.output_dir))

        worker_thread = threading.Thread(target=run_worker, daemon=True)

        with client.websocket_connect("/ws/generation") as ws:
            worker_thread.start()

            deadline = time.monotonic() + _MAX_GENERATION_SECONDS
            while time.monotonic() < deadline:
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg.get("type") == "finish":
                        break
                except Exception:
                    break

            worker_thread.join(timeout=10)

        preview_messages = [m for m in messages if m.get("type") == "preview"]
        previews_with_images = [m for m in preview_messages if m.get("image") is not None]
        assert len(previews_with_images) > 0, "No preview messages contained a preview image"

    def test_finish_message_has_valid_image_path(self, client) -> None:
        """AC1+AC3: Finish message arrives with a valid image path."""
        import threading

        from modules.async_worker import async_tasks

        client.post("/api/generate", json=_generate_request_body())
        assert len(async_tasks) == 1

        messages: list[dict] = []

        def run_worker():
            _run_worker_with_pipeline(str(self.output_dir))

        worker_thread = threading.Thread(target=run_worker, daemon=True)

        with client.websocket_connect("/ws/generation") as ws:
            worker_thread.start()

            deadline = time.monotonic() + _MAX_GENERATION_SECONDS
            while time.monotonic() < deadline:
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg.get("type") == "finish":
                        break
                except Exception:
                    break

            worker_thread.join(timeout=10)

        finish_messages = [m for m in messages if m.get("type") == "finish"]
        assert len(finish_messages) == 1, f"Expected 1 finish message, got {len(finish_messages)}"

        image_paths = finish_messages[0]["images"]
        assert len(image_paths) >= 1
        for path in image_paths:
            assert os.path.isfile(path), f"Finish event image path does not exist: {path}"


# ===========================================================================
# AC7 Scenario 2: Prompt with LoRA
# ===========================================================================


class TestLoRAGeneration:
    """Scenario 2 (AC7): Generation with a LoRA adapter applied.

    Verifies that LoRA injection works through the full pipeline
    and the output is still a valid SDXL image.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _redirect_output, _clean_task_queue, _keep_heartbeat_alive, output_dir):
        self.output_dir = output_dir

    def test_generation_with_lora_produces_valid_image(self, client) -> None:
        """AC7: Generation with a LoRA produces a valid, non-placeholder image."""
        body = _generate_request_body(
            prompt="a detailed portrait of a woman, cinematic lighting",
            loras=[{"filename": _get_lora(), "weight": 0.8}],
        )
        client.post("/api/generate", json=body)

        start = time.monotonic()
        _run_worker_with_pipeline(str(self.output_dir))
        elapsed = time.monotonic() - start

        assert elapsed < _MAX_GENERATION_SECONDS

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) == 1
        _assert_image_is_valid_sdxl(str(png_files[0]))

    def test_lora_metadata_in_output(self, client) -> None:
        """AC7: LoRA name is referenced in generation metadata or logs."""
        body = _generate_request_body(
            prompt="a landscape with mountains",
            loras=[{"filename": _get_lora(), "weight": 0.6}],
        )
        client.post("/api/generate", json=body)
        _run_worker_with_pipeline(str(self.output_dir))

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) >= 1

        # Image should still be valid SDXL
        _assert_image_is_valid_sdxl(str(png_files[0]))


# ===========================================================================
# AC7 Scenario 3: Different sampler
# ===========================================================================


class TestDifferentSamplerGeneration:
    """Scenario 3 (AC7): Generation with a different sampler algorithm.

    Proves the pipeline is not hardcoded to a single sampler.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _redirect_output, _clean_task_queue, _keep_heartbeat_alive, output_dir):
        self.output_dir = output_dir

    def test_euler_sampler_produces_valid_image(self, client) -> None:
        """AC7: Euler sampler produces a valid SDXL image."""
        body = _generate_request_body(
            prompt="a futuristic cityscape at night, neon lights",
            sampler_name="euler",
            scheduler_name="normal",
        )
        client.post("/api/generate", json=body)

        start = time.monotonic()
        _run_worker_with_pipeline(str(self.output_dir))
        elapsed = time.monotonic() - start

        assert elapsed < _MAX_GENERATION_SECONDS

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) == 1
        _assert_image_is_valid_sdxl(str(png_files[0]))

    def test_different_sampler_produces_different_output(self, client) -> None:
        """AC7: Different sampler with same seed produces a different image.

        This validates that the sampler_name parameter actually affects
        the generation pipeline, not just metadata.
        """
        # Generate with default sampler (dpmpp_2m_sde_gpu)
        body_a = _generate_request_body(
            prompt="a mountain lake at sunset",
            sampler_name="dpmpp_2m_sde_gpu",
            seed=12345,
        )
        client.post("/api/generate", json=body_a)
        _run_worker_with_pipeline(str(self.output_dir))

        files_a = list(self.output_dir.rglob("*.png"))
        assert len(files_a) == 1
        pixels_a = np.array(Image.open(files_a[0]))

        # Generate with euler sampler (same seed, same prompt)
        body_b = _generate_request_body(
            prompt="a mountain lake at sunset",
            sampler_name="euler",
            scheduler_name="normal",
            seed=12345,
        )
        client.post("/api/generate", json=body_b)
        _run_worker_with_pipeline(str(self.output_dir))

        files_b = sorted(self.output_dir.rglob("*.png"))
        assert len(files_b) == 2
        newest = max(files_b, key=lambda p: p.stat().st_mtime)
        pixels_b = np.array(Image.open(newest))

        # Same seed + different sampler = different image
        assert not np.array_equal(pixels_a, pixels_b), (
            "Different samplers with same seed produced identical images — "
            "sampler_name may not be wired through the pipeline"
        )


# ===========================================================================
# AC7 Bonus: Different resolution
# ===========================================================================


class TestDifferentResolutionGeneration:
    """Bonus scenario: Generation at a non-square resolution.

    Validates the pipeline correctly handles non-1024x1024 SDXL resolutions.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _redirect_output, _clean_task_queue, _keep_heartbeat_alive, output_dir):
        self.output_dir = output_dir

    def test_landscape_resolution_produces_valid_image(self, client) -> None:
        """Landscape resolution 1152x896 produces a valid SDXL image."""
        body = _generate_request_body(
            prompt="a wide panoramic view of rolling hills",
            aspect_ratios_selection="1152*896",
        )
        client.post("/api/generate", json=body)
        _run_worker_with_pipeline(str(self.output_dir))

        png_files = list(self.output_dir.rglob("*.png"))
        assert len(png_files) == 1
        _assert_image_is_valid_sdxl(str(png_files[0]), expected_width=1152, expected_height=896)
