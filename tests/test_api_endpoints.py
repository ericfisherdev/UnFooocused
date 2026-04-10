"""Functional tests for API endpoints not yet covered by other test files.

Outer-loop TDD (Percival): these are integration tests using FastAPI TestClient
against the real app instance. They define the contract between frontend and backend.

Endpoints tested here:
  - POST /api/heartbeat
  - GET  /api/lora-library-scan-status
  - POST /api/lora-library-rescan
  - GET  /api/lora-library-data
  - GET  /api/lora-trigger-words
  - GET  /api/styles
  - GET  /api/samplers
  - POST /api/generate
  - POST /api/generate/stop

Endpoints already covered elsewhere (DO NOT duplicate):
  - GET /api/config   (test_api_config.py)
  - GET /api/models   (test_api_models.py)
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient against the real FastAPI app."""
    from ui.app import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/heartbeat
# ---------------------------------------------------------------------------


class TestPostHeartbeat:
    """POST /api/heartbeat records a browser heartbeat and returns ok."""

    def test_returns_200(self, client: TestClient):
        response = client.post("/api/heartbeat")
        assert response.status_code == 200

    def test_response_contains_ok_true(self, client: TestClient):
        data = client.post("/api/heartbeat").json()
        assert data == {"ok": True}

    def test_updates_heartbeat_timestamp(self, client: TestClient):
        """Heartbeat should update the internal timestamp so is_browser_connected returns True."""
        from modules.heartbeat import is_browser_connected

        client.post("/api/heartbeat")
        assert is_browser_connected(timeout_seconds=5.0) is True


# ---------------------------------------------------------------------------
# GET /api/lora-library-scan-status
# ---------------------------------------------------------------------------


SCAN_STATUS_FIELDS = {
    "is_scanning": bool,
    "scan_complete": bool,
    "files_scanned": int,
    "files_failed": int,
    "total_indexed": int,
    "elapsed_time": (int, float),
}


class TestGetLoraScanStatus:
    """GET /api/lora-library-scan-status returns scanner statistics."""

    def test_returns_200(self, client: TestClient):
        response = client.get("/api/lora-library-scan-status")
        assert response.status_code == 200

    def test_contains_all_required_fields(self, client: TestClient):
        data = client.get("/api/lora-library-scan-status").json()
        missing = set(SCAN_STATUS_FIELDS) - set(data)
        assert not missing, f"Missing fields: {missing}"

    @pytest.mark.parametrize(
        "field,expected_type",
        list(SCAN_STATUS_FIELDS.items()),
        ids=list(SCAN_STATUS_FIELDS.keys()),
    )
    def test_field_has_correct_type(self, client: TestClient, field: str, expected_type):
        data = client.get("/api/lora-library-scan-status").json()
        assert isinstance(data[field], expected_type), f"{field}: expected {expected_type}, got {type(data[field])}"


# ---------------------------------------------------------------------------
# POST /api/lora-library-rescan
# ---------------------------------------------------------------------------


class TestPostLoraRescan:
    """POST /api/lora-library-rescan triggers a LoRA library rescan."""

    def test_returns_200(self, client: TestClient):
        response = client.post("/api/lora-library-rescan")
        assert response.status_code == 200

    def test_returns_success_true_when_not_scanning(self, client: TestClient):
        """When no scan is in progress, starting one should succeed."""
        import modules.lora_metadata as lora_metadata

        scanner = lora_metadata.get_scanner()
        # Ensure no scan is running by waiting if one is active
        if scanner.is_scanning:
            scanner.stop_scan()

        try:
            data = client.post("/api/lora-library-rescan").json()
            assert data == {"success": True}
        finally:
            # Clean up: stop background scan even if assertion fails
            if scanner.is_scanning:
                scanner.stop_scan()

    def test_returns_success_false_when_already_scanning(self, client: TestClient):
        """When a scan is already in progress, a second request should fail."""
        import modules.lora_metadata as lora_metadata

        scanner = lora_metadata.get_scanner()
        # Simulate in-progress scan while preserving prior state.
        with scanner._lock:
            previous_is_scanning = scanner._is_scanning
            scanner._is_scanning = True
        try:
            data = client.post("/api/lora-library-rescan").json()
            assert data == {"success": False, "error": "Scan already in progress"}
        finally:
            with scanner._lock:
                scanner._is_scanning = previous_is_scanning


# ---------------------------------------------------------------------------
# GET /api/lora-library-data
# ---------------------------------------------------------------------------


class TestGetLoraLibraryData:
    """GET /api/lora-library-data returns LoRA metadata for the library."""

    def test_returns_200(self, client: TestClient):
        response = client.get("/api/lora-library-data")
        assert response.status_code == 200

    def test_returns_a_list(self, client: TestClient):
        data = client.get("/api/lora-library-data").json()
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# GET /api/lora-trigger-words
# ---------------------------------------------------------------------------


class TestGetLoraTriggerWords:
    """GET /api/lora-trigger-words returns trigger words for a LoRA filename."""

    def test_returns_200(self, client: TestClient):
        response = client.get("/api/lora-trigger-words", params={"filename": "nonexistent.safetensors"})
        assert response.status_code == 200

    def test_response_contains_filename_and_trigger_words(self, client: TestClient):
        data = client.get("/api/lora-trigger-words", params={"filename": "test.safetensors"}).json()
        assert "filename" in data
        assert "trigger_words" in data

    def test_echoes_back_requested_filename(self, client: TestClient):
        data = client.get("/api/lora-trigger-words", params={"filename": "my_lora.safetensors"}).json()
        assert data["filename"] == "my_lora.safetensors"

    def test_trigger_words_is_a_list(self, client: TestClient):
        data = client.get("/api/lora-trigger-words", params={"filename": "test.safetensors"}).json()
        assert isinstance(data["trigger_words"], list)

    def test_returns_422_without_filename_param(self, client: TestClient):
        """The filename query parameter is required."""
        response = client.get("/api/lora-trigger-words")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/styles
# ---------------------------------------------------------------------------


class TestGetStyles:
    """GET /api/styles returns available style names."""

    def test_returns_200_with_styles_list(self, client: TestClient):
        response = client.get("/api/styles")
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert isinstance(data["styles"], list)

    def test_styles_list_is_nonempty(self, client: TestClient):
        data = client.get("/api/styles").json()
        assert len(data["styles"]) > 0

    def test_styles_contains_default_styles(self, client: TestClient):
        """The default styles from config must be present in the API response."""
        data = client.get("/api/styles").json()
        style_names = data["styles"]
        for expected in ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]:
            assert expected in style_names, f"Default style {expected!r} missing from /api/styles"


# ---------------------------------------------------------------------------
# xfail: GET /api/samplers (blocks on UNF-4)
# ---------------------------------------------------------------------------


class TestGetSamplers:
    """GET /api/samplers returns sampler and scheduler lists from local flags module."""

    def test_returns_200(self, client: TestClient):
        response = client.get("/api/samplers")
        assert response.status_code == 200

    def test_response_contains_samplers_and_schedulers(self, client: TestClient):
        data = client.get("/api/samplers").json()
        assert "samplers" in data
        assert "schedulers" in data

    def test_samplers_is_a_nonempty_list(self, client: TestClient):
        data = client.get("/api/samplers").json()
        assert isinstance(data["samplers"], list)
        assert len(data["samplers"]) > 0

    def test_schedulers_is_a_nonempty_list(self, client: TestClient):
        data = client.get("/api/samplers").json()
        assert isinstance(data["schedulers"], list)
        assert len(data["schedulers"]) > 0

    def test_samplers_contains_default_sampler(self, client: TestClient):
        """The default sampler (dpmpp_2m_sde_gpu) must be in the list."""
        data = client.get("/api/samplers").json()
        assert "dpmpp_2m_sde_gpu" in data["samplers"]

    def test_schedulers_contains_default_scheduler(self, client: TestClient):
        """The default scheduler (karras) must be in the list."""
        data = client.get("/api/samplers").json()
        assert "karras" in data["schedulers"]


# ---------------------------------------------------------------------------
# xfail: POST /api/generate (blocks on UNF-5)
# ---------------------------------------------------------------------------


class TestPostGenerate:
    """POST /api/generate queues an image generation task."""

    def test_returns_200_with_queued_response(self, client: TestClient):
        response = client.post("/api/generate", json={"prompt": "a test prompt"})
        assert response.status_code == 200
        data = response.json()
        assert data["queued"] is True
        assert "task_id" in data


# ---------------------------------------------------------------------------
# POST /api/generate/stop
# ---------------------------------------------------------------------------


class TestPostGenerateStop:
    """POST /api/generate/stop stops the current generation."""

    def test_returns_200_with_stopped_response(self, client: TestClient):
        response = client.post("/api/generate/stop")
        assert response.status_code == 200
        data = response.json()
        assert "stopped" in data
        assert isinstance(data["stopped"], bool)
