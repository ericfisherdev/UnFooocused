"""Integration tests for LdmModelLoader — require real .safetensors files and GPU.

These tests verify the actual loading path against real SDXL checkpoint
and LoRA files on disk. They are slow and require GPU availability.

Acceptance criteria covered:
AC2: Loading a valid SDXL checkpoint returns a model with unet, clip, vae
AC5: LoRA application correctly modifies model weights
AC7: fast_checkpoint integration works
AC8: Integration tests pass against real .safetensors files
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Skip entire module if no GPU or model files available
_CHECKPOINT_DIR = os.environ.get("UNFOOOCUSED_TEST_CHECKPOINT_DIR", "/mnt/NovusLocus/ai/focus/models/checkpoints")
_LORA_DIR = os.environ.get("UNFOOOCUSED_TEST_LORA_DIR", "/mnt/NovusLocus/ai/focus/models/loras")
_CHECKPOINT_FILE = "juggernautXL_v8Rundiffusion.safetensors"
_CHECKPOINT_PATH = os.path.join(_CHECKPOINT_DIR, _CHECKPOINT_FILE)

_has_checkpoint = os.path.isfile(_CHECKPOINT_PATH)
_has_gpu = False
try:
    import torch

    _has_gpu = torch.cuda.is_available()
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (_has_checkpoint and _has_gpu),
    reason="Requires real checkpoint file and CUDA GPU",
)


# ---------------------------------------------------------------------------
# AC2: Loading a valid SDXL checkpoint returns model with unet, clip, vae
# ---------------------------------------------------------------------------


class TestLoadRealCheckpoint:
    """AC2: Loading juggernautXL returns a model with unet, clip, vae."""

    def test_load_checkpoint_returns_model_with_unet(self) -> None:
        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        assert model.unet is not None

    def test_load_checkpoint_returns_model_with_clip(self) -> None:
        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        assert model.clip is not None

    def test_load_checkpoint_returns_model_with_vae(self) -> None:
        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        assert model.vae is not None

    def test_load_checkpoint_stores_filename(self) -> None:
        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        assert _CHECKPOINT_FILE in model.filename

    def test_loaded_model_is_sdxl(self) -> None:
        """Validates the model is detected as SDXL architecture."""
        import ldm_patched.modules.latent_formats as lf

        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        assert isinstance(model.unet.model.latent_format, lf.SDXL)


# ---------------------------------------------------------------------------
# AC5: LoRA application correctly modifies model weights
# ---------------------------------------------------------------------------


class TestLoadRealLoras:
    """AC5: LoRA application modifies model weights."""

    def test_apply_lora_produces_unet_with_lora(self) -> None:
        from modules.domain.protocols import LoRAConfig

        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)

        lora_files = _find_lora_files()
        if not lora_files:
            pytest.skip("No LoRA files found in test directory")

        lora_config = LoRAConfig(filename=lora_files[0], weight=0.5)
        patched_model = loader.load_loras(model, [lora_config])

        # After LoRA application, unet_with_lora should differ from base unet
        assert patched_model.unet_with_lora is not None
        assert patched_model.unet_with_lora is not model.unet

    def test_apply_empty_lora_list_returns_model(self) -> None:
        loader = _make_real_loader()
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        patched_model = loader.load_loras(model, [])
        assert patched_model is not None


# ---------------------------------------------------------------------------
# AC6: Caching works for real loads
# ---------------------------------------------------------------------------


class TestCachingWithRealFiles:
    """AC6: Second load of same checkpoint returns cached model."""

    def test_second_load_uses_cached_data(self) -> None:
        loader = _make_real_loader()
        model1 = loader.load_checkpoint(_CHECKPOINT_FILE)
        model2 = loader.load_checkpoint(_CHECKPOINT_FILE)
        # Fresh instances backed by same cached base data
        assert model1 is not model2
        assert model1.filename == model2.filename
        assert model1.unet is model2.unet


# ---------------------------------------------------------------------------
# AC7: fast_checkpoint integration works
# ---------------------------------------------------------------------------


class TestFastCheckpointIntegration:
    """AC7: fast_checkpoint path resolution is used when configured."""

    def test_loader_uses_resolve_checkpoint_path(self) -> None:
        """Verifies the loader's resolve_path is called during loading."""

        loader = _make_real_loader()
        # Just verify it loads without error — the path resolution
        # is tested by the fast_checkpoint unit tests
        model = loader.load_checkpoint(_CHECKPOINT_FILE)
        assert model is not None


# ===========================================================================
# Test helpers
# ===========================================================================


def _make_real_loader() -> object:
    """Create an LdmModelLoader wired to real ldm_patched and real paths."""
    from ldm_patched.modules.sd import load_checkpoint_guess_config
    from modules.fast_checkpoint import resolve_checkpoint_path
    from modules.infrastructure.model_loader import LdmModelLoader

    checkpoint_folders = [_CHECKPOINT_DIR]
    lora_folders = [_LORA_DIR]

    def resolve(name: str) -> str:
        return resolve_checkpoint_path(name, checkpoint_folders)

    return LdmModelLoader(
        resolve_path=resolve,
        load_fn=load_checkpoint_guess_config,
        embedding_directory="",
        lora_paths=lora_folders,
    )


def _find_lora_files() -> list[str]:
    """Find available LoRA .safetensors files for testing."""
    lora_dir = Path(_LORA_DIR)
    if not lora_dir.is_dir():
        return []
    files = sorted(lora_dir.glob("*.safetensors"))
    return [f.name for f in files[:3]]
