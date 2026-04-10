"""Functional tests for GET /api/models.

Outer-loop TDD (Percival): verifies model discovery works end-to-end.
"""

import dataclasses

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def models_dir(tmp_path):
    """Create a temp directory with fake .safetensors files."""
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "test_model_v1.safetensors").write_bytes(b"\x00")
    (checkpoints_dir / "test_model_v2.safetensors").write_bytes(b"\x00")
    (checkpoints_dir / "not_a_model.txt").write_bytes(b"\x00")

    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "detail_enhancer.safetensors").write_bytes(b"\x00")
    (loras_dir / "readme.md").write_bytes(b"\x00")

    return {"checkpoints": str(checkpoints_dir), "loras": str(loras_dir)}


@pytest.fixture
def client_with_models(models_dir):
    """Create a TestClient with config pointing at temp model dirs."""
    from modules.config import (
        get_config,
        refresh_lora_filenames,
        refresh_model_filenames,
        reset_config,
        set_config,
    )

    base = get_config()
    custom = dataclasses.replace(
        base,
        paths_checkpoints=(models_dir["checkpoints"],),
        paths_loras=(models_dir["loras"],),
        model_filenames=tuple(refresh_model_filenames([models_dir["checkpoints"]])),
        lora_filenames=tuple(refresh_lora_filenames([models_dir["loras"]])),
    )
    set_config(custom)

    try:
        from ui.app import app

        yield TestClient(app)
    finally:
        reset_config()


class TestGetModels:
    """GET /api/models returns discovered checkpoint and LoRA files."""

    def test_returns_200(self, client_with_models: TestClient):
        response = client_with_models.get("/api/models")
        assert response.status_code == 200

    def test_response_has_checkpoints_and_loras_keys(self, client_with_models: TestClient):
        data = client_with_models.get("/api/models").json()
        assert "checkpoints" in data
        assert "loras" in data

    def test_discovers_safetensors_checkpoints(self, client_with_models: TestClient):
        data = client_with_models.get("/api/models").json()
        filenames = data["checkpoints"]
        assert "test_model_v1.safetensors" in filenames
        assert "test_model_v2.safetensors" in filenames

    def test_excludes_non_safetensors_from_checkpoints(self, client_with_models: TestClient):
        data = client_with_models.get("/api/models").json()
        filenames = data["checkpoints"]
        assert "not_a_model.txt" not in filenames

    def test_discovers_safetensors_loras(self, client_with_models: TestClient):
        data = client_with_models.get("/api/models").json()
        filenames = data["loras"]
        assert "detail_enhancer.safetensors" in filenames

    def test_excludes_non_safetensors_from_loras(self, client_with_models: TestClient):
        data = client_with_models.get("/api/models").json()
        filenames = data["loras"]
        assert "readme.md" not in filenames


class TestGetModelsEmptyPaths:
    """GET /api/models handles missing/empty directories gracefully."""

    def test_empty_list_for_nonexistent_path(self):
        from modules.config import get_config, reset_config, set_config

        base = get_config()
        custom = dataclasses.replace(
            base,
            paths_checkpoints=("/nonexistent/path",),
            paths_loras=("/nonexistent/loras",),
            model_filenames=(),
            lora_filenames=(),
        )
        set_config(custom)

        from ui.app import app

        try:
            client = TestClient(app)
            data = client.get("/api/models").json()
            assert data["checkpoints"] == []
            assert data["loras"] == []
        finally:
            reset_config()
