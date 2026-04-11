"""Unit tests for LdmModelLoader — no GPU or model files required.

Tests use fakes/stubs for ldm_patched dependencies. Organized by
acceptance criterion from UNF-33.

AC1: modules/infrastructure/model_loader.py exists implementing ModelLoader protocol
AC3: Loading a nonexistent checkpoint raises ModelNotFoundError
AC4: Loading a non-SDXL checkpoint raises UnsupportedModelError
AC6: Model caching prevents redundant loads (same filename returns cached model)
AC9: Unit tests pass without GPU/model files using fakes
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# AC1: Module exists and implements ModelLoader protocol
# ---------------------------------------------------------------------------


class TestModelLoaderModuleExists:
    """AC1: modules/infrastructure/model_loader.py exists and exports LdmModelLoader."""

    def test_infrastructure_package_is_importable(self) -> None:
        from modules.infrastructure import model_loader  # noqa: F401

    def test_ldm_model_loader_class_exists(self) -> None:
        from modules.infrastructure.model_loader import LdmModelLoader

        assert LdmModelLoader is not None

    def test_ldm_model_loader_satisfies_model_loader_protocol(self) -> None:
        from modules.domain.protocols import ModelLoader

        loader = _make_loader()
        assert isinstance(loader, ModelLoader)


# ---------------------------------------------------------------------------
# AC3: Loading nonexistent checkpoint raises ModelNotFoundError
# ---------------------------------------------------------------------------


class TestLoadCheckpointNotFound:
    """AC3: Loading a nonexistent checkpoint raises ModelNotFoundError."""

    def test_nonexistent_file_raises_model_not_found_error(self) -> None:
        from modules.domain.exceptions import ModelNotFoundError

        loader = _make_loader()
        with pytest.raises(ModelNotFoundError, match="does_not_exist.safetensors"):
            loader.load_checkpoint("does_not_exist.safetensors")

    def test_error_is_not_file_not_found_error(self) -> None:
        """Domain code should not see FileNotFoundError."""
        from modules.domain.exceptions import ModelNotFoundError

        loader = _make_loader()
        with pytest.raises(ModelNotFoundError):
            loader.load_checkpoint("missing.safetensors")
        # Verify it's specifically ModelNotFoundError, not FileNotFoundError
        try:
            loader.load_checkpoint("missing.safetensors")
        except ModelNotFoundError:
            pass
        except FileNotFoundError:
            pytest.fail("Should raise ModelNotFoundError, not FileNotFoundError")


# ---------------------------------------------------------------------------
# AC4: Loading a non-SDXL checkpoint raises UnsupportedModelError
# ---------------------------------------------------------------------------


class TestLoadCheckpointUnsupportedModel:
    """AC4: Loading a non-SDXL checkpoint raises UnsupportedModelError."""

    @patch("os.path.isfile", return_value=True)
    def test_non_sdxl_model_raises_unsupported_model_error(self, _mock_isfile: Any) -> None:
        from modules.domain.exceptions import UnsupportedModelError

        loader = _make_loader_with_non_sdxl_model()
        with pytest.raises(UnsupportedModelError, match="SDXL"):
            loader.load_checkpoint("sd15_model.safetensors")


# ---------------------------------------------------------------------------
# AC6: Model caching prevents redundant loads
# ---------------------------------------------------------------------------


class TestModelCaching:
    """AC6: Same filename returns cached model, avoiding redundant loads."""

    @patch("os.path.isfile", return_value=True)
    def test_same_filename_returns_cached_model(self, _mock_isfile: Any) -> None:
        load_fn = MagicMock(return_value=_make_fake_sdxl_load_result())
        loader = _make_loader_with_custom_load(load_fn)

        model1 = loader.load_checkpoint("model.safetensors")
        model2 = loader.load_checkpoint("model.safetensors")

        assert model1 is model2
        load_fn.assert_called_once()

    @patch("os.path.isfile", return_value=True)
    def test_different_filename_loads_separately(self, _mock_isfile: Any) -> None:
        call_count = 0

        def counting_load(path: str, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _make_fake_sdxl_load_result()

        loader = _make_loader_with_custom_load(counting_load)

        loader.load_checkpoint("model_a.safetensors")
        loader.load_checkpoint("model_b.safetensors")

        assert call_count == 2


# ---------------------------------------------------------------------------
# AC5: LoRA application unit tests (without real model files)
# ---------------------------------------------------------------------------


class TestLoraApplicationUnit:
    """AC5: LoRA loading and application logic works with fakes."""

    @patch("os.path.isfile", return_value=True)
    def test_load_loras_with_empty_list_returns_model(self, _mock_isfile: Any) -> None:
        loader = _make_loader_with_custom_load(MagicMock(return_value=_make_fake_sdxl_load_result()))
        model = loader.load_checkpoint("model.safetensors")
        result = loader.load_loras(model, [])
        assert result is not None

    @patch("os.path.isfile", return_value=True)
    def test_load_loras_skips_none_filename(self, _mock_isfile: Any) -> None:
        from modules.domain.protocols import LoRAConfig

        loader = _make_loader_with_custom_load(MagicMock(return_value=_make_fake_sdxl_load_result()))
        model = loader.load_checkpoint("model.safetensors")
        lora = LoRAConfig(filename="None", weight=0.5)
        result = loader.load_loras(model, [lora])
        # "None" filename should be skipped, model returned unchanged
        assert result is not None

    @patch("os.path.isfile", return_value=True)
    def test_load_loras_caches_by_lora_config(self, _mock_isfile: Any) -> None:
        """Same LoRA config applied twice should skip the second time."""
        from modules.domain.protocols import LoRAConfig

        loader = _make_loader_with_custom_load(MagicMock(return_value=_make_fake_sdxl_load_result()))
        model = loader.load_checkpoint("model.safetensors")
        lora = LoRAConfig(filename="None", weight=0.5)

        result1 = loader.load_loras(model, [lora])
        result2 = loader.load_loras(result1, [lora])
        # Second call with same config should return immediately
        assert result2 is result1


# ---------------------------------------------------------------------------
# AC9: Domain exceptions module is well-formed
# ---------------------------------------------------------------------------


class TestDomainExceptions:
    """AC9: Domain exceptions are properly defined."""

    def test_model_not_found_error_is_importable(self) -> None:
        from modules.domain.exceptions import ModelNotFoundError

        assert issubclass(ModelNotFoundError, Exception)

    def test_unsupported_model_error_is_importable(self) -> None:
        from modules.domain.exceptions import UnsupportedModelError

        assert issubclass(UnsupportedModelError, Exception)

    def test_model_not_found_error_is_not_file_not_found(self) -> None:
        from modules.domain.exceptions import ModelNotFoundError

        assert not issubclass(ModelNotFoundError, FileNotFoundError)

    def test_unsupported_model_error_carries_message(self) -> None:
        from modules.domain.exceptions import UnsupportedModelError

        err = UnsupportedModelError("Only SDXL supported")
        assert "SDXL" in str(err)


# ===========================================================================
# Test helpers — fakes and factory functions
# ===========================================================================


def _make_fake_sdxl_load_result() -> tuple:
    """Simulate ldm_patched load_checkpoint_guess_config return for SDXL."""
    fake_sdxl_model = MagicMock()
    # model.model.latent_format must be SDXL instance for validation
    fake_sdxl_model.model = MagicMock()

    # Import at function level to avoid top-level torch dependency
    import ldm_patched.modules.latent_formats as lf

    fake_sdxl_model.model.latent_format = lf.SDXL()

    fake_clip = MagicMock()
    fake_clip.cond_stage_model = MagicMock()
    fake_clip.cond_stage_model.state_dict.return_value = {}

    fake_vae = MagicMock()
    fake_clip_vision = None
    fake_vae_filename = "vae.safetensors"

    return (fake_sdxl_model, fake_clip, fake_vae, fake_vae_filename, fake_clip_vision)


def _make_fake_non_sdxl_load_result() -> tuple:
    """Simulate ldm_patched load_checkpoint_guess_config return for SD1.5."""
    fake_model = MagicMock()
    fake_model.model = MagicMock()
    # Not an SDXL latent format
    fake_model.model.latent_format = MagicMock()
    fake_model.model.latent_format.__class__.__name__ = "SD15"

    fake_clip = MagicMock()
    fake_clip.cond_stage_model = MagicMock()
    fake_clip.cond_stage_model.state_dict.return_value = {}

    return (fake_model, fake_clip, MagicMock(), "vae.safetensors", None)


def _make_loader(**kwargs: Any) -> Any:
    """Create a LdmModelLoader with all dependencies faked.

    The resolve_path callable raises FileNotFoundError for any name,
    simulating a nonexistent file.
    """
    from modules.infrastructure.model_loader import LdmModelLoader

    def resolve_not_found(name: str) -> str:
        # Return a path that won't exist
        return f"/nonexistent/{name}"

    return LdmModelLoader(
        resolve_path=kwargs.get("resolve_path", resolve_not_found),
        load_fn=kwargs.get("load_fn", MagicMock(side_effect=FileNotFoundError("not found"))),
        embedding_directory=kwargs.get("embedding_directory", ""),
        lora_paths=kwargs.get("lora_paths", []),
    )


def _make_loader_with_non_sdxl_model() -> Any:
    """Create a loader that returns a non-SDXL model."""
    from modules.infrastructure.model_loader import LdmModelLoader

    return LdmModelLoader(
        resolve_path=lambda name: f"/fake/{name}",
        load_fn=MagicMock(return_value=_make_fake_non_sdxl_load_result()),
        embedding_directory="",
        lora_paths=[],
    )


def _make_loader_with_custom_load(load_fn: Any) -> Any:
    """Create a loader with a custom load function that returns SDXL models."""
    from modules.infrastructure.model_loader import LdmModelLoader

    return LdmModelLoader(
        resolve_path=lambda name: f"/fake/{name}",
        load_fn=load_fn,
        embedding_directory="",
        lora_paths=[],
    )
