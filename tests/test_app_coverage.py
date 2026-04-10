"""Tests for untested ui/app.py paths: WebSocket handler, origin validation,
preview encoding, yield message building, and generation stop flow.

Acceptance criteria (UNF-23):
  AC1: ui/app.py line coverage reaches 80% or higher.
  AC2: WebSocket origin validation has tests for accept, reject, and malformed cases.
  AC3: /ws/generation endpoint has at least one integration test verifying
       progress message delivery during generation.
  AC4: All existing tests continue to pass.
"""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.websockets import WebSocketDisconnect

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None

_requires_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")

# ---------------------------------------------------------------------------
# Unit tests: _build_yield_message
# ---------------------------------------------------------------------------


class TestBuildYieldMessage:
    """Unit tests for _build_yield_message covering all message types."""

    def test_preview_flag_returns_preview_message(self) -> None:
        from ui.app import _build_yield_message

        product = (42, "Step 3/10", None)
        result = _build_yield_message("preview", product)

        assert result is not None
        assert result["type"] == "preview"
        assert result["percentage"] == 42
        assert result["text"] == "Step 3/10"
        assert result["image"] is None

    def test_results_flag_returns_results_message(self) -> None:
        from ui.app import _build_yield_message

        product = ["/path/to/img1.png", "/path/to/img2.png"]
        result = _build_yield_message("results", product)

        assert result is not None
        assert result["type"] == "results"
        assert result["images"] == ["/path/to/img1.png", "/path/to/img2.png"]

    def test_finish_flag_returns_finish_message(self) -> None:
        from ui.app import _build_yield_message

        product = ["/path/to/final.png"]
        result = _build_yield_message("finish", product)

        assert result is not None
        assert result["type"] == "finish"
        assert result["images"] == ["/path/to/final.png"]

    def test_unknown_flag_returns_none(self) -> None:
        from ui.app import _build_yield_message

        result = _build_yield_message("unknown_flag", ("some", "data"))
        assert result is None

    def test_results_stringifies_path_objects(self) -> None:
        """Non-string path objects (e.g. pathlib.Path) should be stringified."""
        from pathlib import Path

        from ui.app import _build_yield_message

        product = [Path("/output/image.png")]
        result = _build_yield_message("results", product)

        assert result is not None
        assert result["images"] == ["/output/image.png"]
        assert isinstance(result["images"][0], str)

    def test_finish_stringifies_path_objects(self) -> None:
        from pathlib import Path

        from ui.app import _build_yield_message

        product = [Path("/output/final.png")]
        result = _build_yield_message("finish", product)

        assert result is not None
        assert isinstance(result["images"][0], str)


# ---------------------------------------------------------------------------
# Unit tests: _encode_preview_image
# ---------------------------------------------------------------------------


class TestEncodePreviewImage:
    """Unit tests for _encode_preview_image with various input types."""

    def test_none_input_returns_none(self) -> None:
        from ui.app import _encode_preview_image

        assert _encode_preview_image(None) is None

    @_requires_numpy
    def test_numpy_array_returns_base64_string(self) -> None:
        import numpy as np
        from ui.app import _encode_preview_image

        # Create a small 10x10 RGB array
        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red image

        result = _encode_preview_image(img_array)
        assert result is not None
        assert isinstance(result, str)

        # Verify it's valid base64 by decoding
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    @_requires_numpy
    def test_pil_image_returns_base64_string(self) -> None:
        """PIL Image encoding requires numpy to be importable.

        The function imports numpy before PIL inside a single try block,
        so when numpy is unavailable the PIL path also fails silently.
        """
        from PIL import Image
        from ui.app import _encode_preview_image

        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        result = _encode_preview_image(img)

        assert result is not None
        assert isinstance(result, str)

        # Verify the base64 decodes to a JPEG
        import base64
        import io

        decoded = base64.b64decode(result)
        reopened = Image.open(io.BytesIO(decoded))
        assert reopened.format == "JPEG"

    def test_pil_image_without_numpy_returns_none(self) -> None:
        """When numpy is not importable, PIL images return None due to import structure."""
        if _HAS_NUMPY:
            pytest.skip("numpy is available; this test is for environments without numpy")

        from PIL import Image
        from ui.app import _encode_preview_image

        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        # The function imports numpy at the top of its try block,
        # so ModuleNotFoundError is caught and None is returned.
        result = _encode_preview_image(img)
        assert result is None

    def test_non_image_input_returns_none(self) -> None:
        from ui.app import _encode_preview_image

        result = _encode_preview_image("not an image")
        assert result is None

    def test_integer_input_returns_none(self) -> None:
        from ui.app import _encode_preview_image

        result = _encode_preview_image(42)
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: _effective_port
# ---------------------------------------------------------------------------


class TestEffectivePort:
    """Unit tests for _effective_port helper."""

    def test_explicit_port_is_returned(self) -> None:
        from urllib.parse import urlsplit

        from ui.app import _effective_port

        parts = urlsplit("http://localhost:8888")
        assert _effective_port(parts, 80) == 8888

    def test_no_port_returns_default(self) -> None:
        from urllib.parse import urlsplit

        from ui.app import _effective_port

        parts = urlsplit("http://localhost")
        assert _effective_port(parts, 80) == 80

    def test_malformed_port_returns_none(self) -> None:
        from urllib.parse import urlsplit

        from ui.app import _effective_port

        # urlsplit with a malformed port raises ValueError on .port access
        parts = urlsplit("//localhost:notaport")
        assert _effective_port(parts, 80) is None


# ---------------------------------------------------------------------------
# Unit tests: _reject_mismatched_origin
# ---------------------------------------------------------------------------


class TestRejectMismatchedOrigin:
    """Unit tests for _reject_mismatched_origin (AC2)."""

    @staticmethod
    def _make_websocket(origin: str | None, host: str, scheme: str = "ws") -> MagicMock:
        """Build a mock WebSocket with the given headers and URL scheme."""
        ws = MagicMock()
        headers: dict[str, str] = {"host": host}
        if origin is not None:
            headers["origin"] = origin
        ws.headers = headers
        ws.url.scheme = scheme
        return ws

    def test_matching_origin_is_accepted(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket("http://localhost:7865", "localhost:7865")
        assert _reject_mismatched_origin(ws) is False

    def test_mismatched_origin_is_rejected(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket("http://evil.com:7865", "localhost:7865")
        assert _reject_mismatched_origin(ws) is True

    def test_no_origin_header_is_accepted(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket(None, "localhost:7865")
        assert _reject_mismatched_origin(ws) is False

    def test_mismatched_port_is_rejected(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket("http://localhost:9999", "localhost:7865")
        assert _reject_mismatched_origin(ws) is True

    def test_malformed_origin_port_is_rejected(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket("http://localhost:abc", "localhost:7865")
        assert _reject_mismatched_origin(ws) is True

    def test_malformed_host_port_is_rejected(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket("http://localhost:7865", "localhost:xyz")
        assert _reject_mismatched_origin(ws) is True

    def test_https_origin_with_default_port_matches(self) -> None:
        from ui.app import _reject_mismatched_origin

        ws = self._make_websocket("https://example.com", "example.com", scheme="wss")
        assert _reject_mismatched_origin(ws) is False


# ---------------------------------------------------------------------------
# Unit tests: _find_processing_task
# ---------------------------------------------------------------------------


class TestFindProcessingTask:
    """Unit tests for _find_processing_task helper."""

    @staticmethod
    def _make_task(processing: bool = False) -> MagicMock:
        task = MagicMock()
        task.processing = processing
        return task

    def test_finds_processing_task_in_queue(self) -> None:
        from ui.app import _find_processing_task

        idle = self._make_task(processing=False)
        active = self._make_task(processing=True)
        result = _find_processing_task([idle, active], None)
        assert result is active

    def test_falls_back_to_current_task(self) -> None:
        from ui.app import _find_processing_task

        current = self._make_task(processing=True)
        result = _find_processing_task([], current)
        assert result is current

    def test_returns_none_when_nothing_processing(self) -> None:
        from ui.app import _find_processing_task

        idle = self._make_task(processing=False)
        result = _find_processing_task([idle], None)
        assert result is None

    def test_returns_none_for_empty_queue_and_no_current(self) -> None:
        from ui.app import _find_processing_task

        result = _find_processing_task([], None)
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: _drain_remaining_yields
# ---------------------------------------------------------------------------


class TestDrainRemainingYields:
    """Unit tests for _drain_remaining_yields async helper."""

    def test_sends_remaining_yields_from_index(self) -> None:
        import asyncio

        from ui.app import _drain_remaining_yields

        task = MagicMock()
        task.yields = [
            ("preview", (10, "Step 1", None)),
            ("preview", (20, "Step 2", None)),
            ("finish", ["/path/to/img.png"]),
        ]

        sent: list[dict[str, Any]] = []

        async def fake_send(msg: dict) -> None:
            sent.append(msg)

        asyncio.run(_drain_remaining_yields(task, 1, fake_send))

        assert len(sent) == 2
        assert sent[0]["type"] == "preview"
        assert sent[1]["type"] == "finish"

    def test_skips_unknown_flags(self) -> None:
        import asyncio

        from ui.app import _drain_remaining_yields

        task = MagicMock()
        task.yields = [("unknown", "whatever")]

        sent: list[dict[str, Any]] = []

        async def fake_send(msg: dict) -> None:
            sent.append(msg)

        asyncio.run(_drain_remaining_yields(task, 0, fake_send))
        assert len(sent) == 0


# ---------------------------------------------------------------------------
# Integration tests: POST /api/generate/stop
# ---------------------------------------------------------------------------


class TestGenerateStopIntegration:
    """Integration tests for POST /api/generate/stop (AC1 coverage)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from ui.app import app

        return TestClient(app)

    @pytest.fixture(autouse=True)
    def _clean_task_queue(self):
        import modules.async_worker as worker_module

        worker_module.async_tasks.clear()
        worker_module.current_task = None
        yield
        worker_module.async_tasks.clear()
        worker_module.current_task = None

    def test_stop_with_nothing_running_returns_stopped_false(self, client) -> None:
        response = client.post("/api/generate/stop")
        assert response.status_code == 200
        assert response.json()["stopped"] is False

    def test_stop_with_current_task_processing(self, client) -> None:
        import modules.async_worker as worker_module

        task = MagicMock()
        task.processing = True
        task.last_stop = False
        worker_module.current_task = task

        response = client.post("/api/generate/stop")
        assert response.status_code == 200
        assert response.json()["stopped"] is True
        assert task.last_stop == "stop"

    def test_stop_with_queued_processing_task(self, client) -> None:
        import modules.async_worker as worker_module

        task = MagicMock()
        task.processing = True
        task.last_stop = False
        worker_module.async_tasks.append(task)

        response = client.post("/api/generate/stop")
        assert response.status_code == 200
        assert response.json()["stopped"] is True
        assert task.last_stop == "stop"

    def test_stop_clears_queued_non_processing_tasks(self, client) -> None:
        import modules.async_worker as worker_module

        task = MagicMock()
        task.processing = False
        worker_module.async_tasks.append(task)

        response = client.post("/api/generate/stop")
        assert response.status_code == 200
        assert response.json()["stopped"] is False
        # Queue should be cleared
        assert len(worker_module.async_tasks) == 0


# ---------------------------------------------------------------------------
# Integration test: WebSocket /ws/generation progress delivery (AC3)
# ---------------------------------------------------------------------------


class TestWsGenerationIntegration:
    """Integration tests for /ws/generation WebSocket endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from ui.app import app

        return TestClient(app)

    @pytest.fixture(autouse=True)
    def _clean_task_queue(self):
        import modules.async_worker as worker_module

        worker_module.async_tasks.clear()
        worker_module.current_task = None
        yield
        worker_module.async_tasks.clear()
        worker_module.current_task = None

    def test_websocket_receives_heartbeat_when_idle(self, client, monkeypatch) -> None:
        """When no task is running, the WebSocket sends heartbeat messages."""
        import asyncio
        import threading

        import ui.app as app_module

        _original_sleep = asyncio.sleep

        async def _fast_sleep(delay: float) -> None:
            # Speed up the 0.1s poll interval to near-instant
            await _original_sleep(0.001)

        monkeypatch.setattr(app_module.asyncio, "sleep", _fast_sleep)

        received: list[dict] = []

        def ws_reader():
            with client.websocket_connect("/ws/generation") as ws:
                msg = ws.receive_json()
                received.append(msg)

        # Run in a thread with a timeout
        thread = threading.Thread(target=ws_reader, daemon=True)
        thread.start()
        thread.join(timeout=2.0)

        assert len(received) >= 1
        assert received[0]["type"] == "heartbeat"

    def test_websocket_receives_progress_during_generation(self, client) -> None:
        """AC3: WebSocket delivers progress messages during a generation task.

        Rather than fighting threading/timing issues with a real Worker,
        we simulate a processing task by directly populating task.yields
        and setting task.processing. The WebSocket handler polls task.yields
        every 100ms, so we just need the task to be visible when it polls.
        """
        import threading
        import time

        import modules.async_worker as worker_module
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        body = {"prompt": "ws test", "image_number": 1, "steps": 5, "seed": 42}
        args = _build_generate_args(body)
        task = AsyncTask(args)
        task.processing = True
        # Make the task visible via current_task (worker pops from queue)
        worker_module.current_task = task

        received: list[dict] = []
        ws_ready = threading.Event()

        def ws_reader():
            with client.websocket_connect("/ws/generation") as ws:
                ws_ready.set()
                while True:
                    try:
                        msg = ws.receive_json()
                        received.append(msg)
                        if msg.get("type") == "finish":
                            break
                    except Exception:
                        break

        ws_thread = threading.Thread(target=ws_reader, daemon=True)
        ws_thread.start()
        ws_ready.wait(timeout=3.0)

        # Simulate yields that the worker would produce, giving the
        # WebSocket poller time to see them.
        time.sleep(0.2)  # Let the WS handler find the task
        task.yields.append(("preview", (50, "Image 1/1, Step 3/5", None)))
        time.sleep(0.2)
        task.yields.append(("preview", (100, "Image 1/1, Step 5/5", None)))
        time.sleep(0.2)
        task.processing = False
        worker_module.current_task = None
        task.yields.append(("finish", ["test_output.png"]))

        ws_thread.join(timeout=5.0)

        message_types = [m["type"] for m in received]
        assert "preview" in message_types, f"No preview messages received. Got: {message_types}"
        assert "finish" in message_types, f"No finish message received. Got: {message_types}"

    def test_websocket_rejects_mismatched_origin(self, client) -> None:
        """AC2: WebSocket rejects connections with mismatched origin."""
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                "/ws/generation",
                headers={"origin": "http://evil.com:9999"},
            ),
        ):
            pass  # Should not reach here -- connection is closed by origin check
