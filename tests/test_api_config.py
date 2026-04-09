"""Functional tests for GET /api/config.

Outer-loop TDD (Percival): these tests define WHAT to build.
They must all FAIL (RED) before any production code is written.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient against the real FastAPI app.

    Imports are deferred so the config module must be importable
    from UnFooocused's own modules/ — not FwdFooocus.
    """
    from ui.app import app

    return TestClient(app)


REQUIRED_FIELDS = {
    "default_model": str,
    "default_refiner": str,
    "default_refiner_switch": (int, float),
    "default_performance": str,
    "default_aspect_ratio": str,
    "available_aspect_ratios": list,
    "default_image_number": int,
    "max_image_number": int,
    "default_output_format": str,
    "default_prompt": str,
    "default_prompt_negative": str,
    "default_styles": list,
    "default_cfg_scale": (int, float),
    "default_sample_sharpness": (int, float),
    "default_sampler": str,
    "default_scheduler": str,
    "default_loras": list,
    "default_loras_min_weight": (int, float),
    "default_loras_max_weight": (int, float),
    "default_max_lora_number": int,
}


class TestGetConfig:
    """GET /api/config returns all required fields with correct types."""

    def test_returns_200(self, client: TestClient):
        response = client.get("/api/config")
        assert response.status_code == 200

    def test_contains_all_required_fields(self, client: TestClient):
        data = client.get("/api/config").json()
        missing = set(REQUIRED_FIELDS) - set(data)
        assert not missing, f"Missing fields: {missing}"

    @pytest.mark.parametrize(
        "field,expected_type",
        list(REQUIRED_FIELDS.items()),
        ids=list(REQUIRED_FIELDS.keys()),
    )
    def test_field_has_correct_type(
        self, client: TestClient, field: str, expected_type
    ):
        data = client.get("/api/config").json()
        assert field in data, f"Field {field!r} missing from response"
        assert isinstance(data[field], expected_type), (
            f"{field}: expected {expected_type}, got {type(data[field])}"
        )

    def test_available_aspect_ratios_not_empty(self, client: TestClient):
        data = client.get("/api/config").json()
        assert len(data["available_aspect_ratios"]) >= 3

    def test_default_aspect_ratio_is_in_available(self, client: TestClient):
        data = client.get("/api/config").json()
        assert data["default_aspect_ratio"] in data["available_aspect_ratios"]

    def test_default_sampler_is_nonempty_string(self, client: TestClient):
        data = client.get("/api/config").json()
        assert data["default_sampler"] != ""

    def test_default_scheduler_is_nonempty_string(self, client: TestClient):
        data = client.get("/api/config").json()
        assert data["default_scheduler"] != ""

    def test_lora_weight_range_is_valid(self, client: TestClient):
        data = client.get("/api/config").json()
        assert data["default_loras_min_weight"] < data["default_loras_max_weight"]

    def test_max_image_number_is_positive(self, client: TestClient):
        data = client.get("/api/config").json()
        assert data["max_image_number"] >= 1
