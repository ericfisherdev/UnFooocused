"""Functional tests for GET /api/models.

Outer-loop TDD (Percival): verifies model discovery works end-to-end.
"""

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
def client_with_models(models_dir, monkeypatch):
    """Create a TestClient with config pointing at temp model dirs."""
    import modules.config as config
    from modules.config import AppConfig, get_config, reset_config, set_config

    # Build an AppConfig with temp model dirs and discovered files
    base = get_config()
    custom = AppConfig(
        default_base_model_name=base.default_base_model_name,
        default_refiner_model_name=base.default_refiner_model_name,
        default_refiner_switch=base.default_refiner_switch,
        default_performance=base.default_performance,
        default_aspect_ratio=base.default_aspect_ratio,
        available_aspect_ratios=base.available_aspect_ratios,
        default_image_number=base.default_image_number,
        default_max_image_number=base.default_max_image_number,
        default_output_format=base.default_output_format,
        default_prompt=base.default_prompt,
        default_prompt_negative=base.default_prompt_negative,
        default_styles=base.default_styles,
        default_cfg_scale=base.default_cfg_scale,
        default_sample_sharpness=base.default_sample_sharpness,
        default_sampler=base.default_sampler,
        default_scheduler=base.default_scheduler,
        default_loras=base.default_loras,
        default_loras_min_weight=base.default_loras_min_weight,
        default_loras_max_weight=base.default_loras_max_weight,
        default_max_lora_number=base.default_max_lora_number,
        default_controlnet_image_count=base.default_controlnet_image_count,
        default_enhance_tabs=base.default_enhance_tabs,
        paths_checkpoints=[models_dir["checkpoints"]],
        paths_loras=[models_dir["loras"]],
        path_embeddings=base.path_embeddings,
        path_outputs=base.path_outputs,
        model_filenames=config.refresh_model_filenames([models_dir["checkpoints"]]),
        lora_filenames=config.refresh_lora_filenames([models_dir["loras"]]),
    )
    set_config(custom)

    # Also patch module-level globals for backward compatibility
    monkeypatch.setattr(config, "paths_checkpoints", [models_dir["checkpoints"]])
    monkeypatch.setattr(config, "paths_loras", [models_dir["loras"]])
    config.update_model_filenames()
    config.update_lora_filenames()

    from ui.app import app

    yield TestClient(app)
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

    def test_empty_list_for_nonexistent_path(self, monkeypatch):
        import modules.config as config
        from modules.config import AppConfig, get_config, reset_config, set_config

        base = get_config()
        custom = AppConfig(
            default_base_model_name=base.default_base_model_name,
            default_refiner_model_name=base.default_refiner_model_name,
            default_refiner_switch=base.default_refiner_switch,
            default_performance=base.default_performance,
            default_aspect_ratio=base.default_aspect_ratio,
            available_aspect_ratios=base.available_aspect_ratios,
            default_image_number=base.default_image_number,
            default_max_image_number=base.default_max_image_number,
            default_output_format=base.default_output_format,
            default_prompt=base.default_prompt,
            default_prompt_negative=base.default_prompt_negative,
            default_styles=base.default_styles,
            default_cfg_scale=base.default_cfg_scale,
            default_sample_sharpness=base.default_sample_sharpness,
            default_sampler=base.default_sampler,
            default_scheduler=base.default_scheduler,
            default_loras=base.default_loras,
            default_loras_min_weight=base.default_loras_min_weight,
            default_loras_max_weight=base.default_loras_max_weight,
            default_max_lora_number=base.default_max_lora_number,
            default_controlnet_image_count=base.default_controlnet_image_count,
            default_enhance_tabs=base.default_enhance_tabs,
            paths_checkpoints=["/nonexistent/path"],
            paths_loras=["/nonexistent/loras"],
            path_embeddings=base.path_embeddings,
            path_outputs=base.path_outputs,
            model_filenames=[],
            lora_filenames=[],
        )
        set_config(custom)

        # Also patch module-level globals for backward compatibility
        monkeypatch.setattr(config, "paths_checkpoints", ["/nonexistent/path"])
        monkeypatch.setattr(config, "paths_loras", ["/nonexistent/loras"])
        config.update_model_filenames()
        config.update_lora_filenames()

        from ui.app import app

        try:
            client = TestClient(app)
            data = client.get("/api/models").json()
            assert data["checkpoints"] == []
            assert data["loras"] == []
        finally:
            reset_config()
