"""Integration tests for VAE decoder — require real SDXL model and GPU.

These tests verify actual VAE decoding against a real checkpoint's VAE
and VAEApprox weights on disk. They are slow and require GPU availability.

Acceptance criteria covered:
  AC2: Decoding a valid latent produces numpy array shape (H, W, 3) uint8
  AC3: Tiled decoding works for large images (>2048px) without OOM
  AC4: VAEApprox preview produces usable preview image from partial latent
  AC5: VAEApprox weights loaded from models directory
  AC7: Integration tests pass with loaded SDXL model and VAE
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment detection — skip when no GPU or model files
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = os.environ.get(
    "UNFOOOCUSED_TEST_CHECKPOINT_DIR",
    "/mnt/NovusLocus/ai/focus/models/checkpoints",
)

_VAE_APPROX_DIR = os.environ.get(
    "UNFOOOCUSED_TEST_VAE_APPROX_DIR",
    "/mnt/NovusLocus/ai/focus/models/vae_approx",
)


def _discover_checkpoint() -> tuple[str, str]:
    """Resolve checkpoint file: env var > auto-discover > empty (skip)."""
    env_file = os.environ.get("UNFOOOCUSED_TEST_CHECKPOINT_FILE")
    if env_file:
        return env_file, os.path.join(_CHECKPOINT_DIR, env_file)

    if os.path.isdir(_CHECKPOINT_DIR):
        for name in sorted(os.listdir(_CHECKPOINT_DIR)):
            if name.endswith(".safetensors"):
                return name, os.path.join(_CHECKPOINT_DIR, name)

    return "", ""


_CHECKPOINT_FILE, _CHECKPOINT_PATH = _discover_checkpoint()
_has_checkpoint = bool(_CHECKPOINT_PATH) and os.path.isfile(_CHECKPOINT_PATH)
_has_vae_approx = os.path.isfile(os.path.join(_VAE_APPROX_DIR, "xlvaeapp.pth"))

_has_gpu = False
try:
    import torch

    _has_gpu = torch.cuda.is_available()
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (_has_gpu and _has_checkpoint),
    reason="Requires CUDA GPU and SDXL checkpoint file",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_model():
    """Load a real SDXL checkpoint — shared across the module."""
    from modules.infrastructure.model_loader import LdmModelLoader

    loader = LdmModelLoader()
    return loader.load_checkpoint(_CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def real_decoder():
    """Build a real LdmVAEDecoder backed by ldm_patched ops."""
    from ldm_patched.contrib.external import VAEDecode, VAEDecodeTiled
    from modules.infrastructure.vae_decoder import LdmVAEDecoder

    return LdmVAEDecoder(
        vae_decode_op=VAEDecode(),
        vae_decode_tiled_op=VAEDecodeTiled(),
    )


@pytest.fixture(scope="module")
def empty_latent():
    """Generate an empty latent tensor (noise) at 1024x1024."""
    import torch

    # SDXL latent: (1, 4, H/8, W/8) = (1, 4, 128, 128) for 1024x1024
    return {"samples": torch.randn(1, 4, 128, 128)}


@pytest.fixture(scope="module")
def large_latent():
    """Generate a large latent tensor for 2560x2560 tiled decoding test."""
    import torch

    # (1, 4, 320, 320) -> 2560x2560 pixel image
    return {"samples": torch.randn(1, 4, 320, 320)}


# ---------------------------------------------------------------------------
# AC2 + AC7: Decode produces correct output with real model
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestRealVAEDecode:
    """AC2/AC7: Real VAE decode produces numpy (H, W, 3) uint8."""

    def test_decode_shape_and_dtype(self, loaded_model, real_decoder, empty_latent) -> None:
        result = real_decoder.decode(vae=loaded_model, latent=empty_latent)

        assert len(result) >= 1
        arr = result[0]
        assert arr.ndim == 3
        assert arr.shape == (1024, 1024, 3)
        assert arr.dtype == np.uint8
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_decode_pil_compatible(self, loaded_model, real_decoder, empty_latent) -> None:
        from PIL import Image

        result = real_decoder.decode(vae=loaded_model, latent=empty_latent)
        img = Image.fromarray(result[0])
        assert img.mode == "RGB"
        assert img.size == (1024, 1024)


# ---------------------------------------------------------------------------
# AC3: Tiled decoding for large images
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestRealTiledDecode:
    """AC3: Tiled decode works for >2048px without OOM."""

    def test_tiled_decode_large_image(self, loaded_model, real_decoder, large_latent) -> None:
        result = real_decoder.decode(vae=loaded_model, latent=large_latent)

        arr = result[0]
        assert arr.ndim == 3
        assert arr.shape[0] == 2560
        assert arr.shape[1] == 2560
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8


# ---------------------------------------------------------------------------
# AC4 + AC5: VAEApprox preview with real weights
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _has_vae_approx, reason="VAEApprox weights not found")
class TestRealVAEApproxPreview:
    """AC4/AC5: VAEApprox preview with real xlvaeapp.pth weights."""

    def test_preview_produces_rgb_image(self, loaded_model) -> None:
        import torch
        from modules.infrastructure.vae_decoder import build_latent_previewer

        previewer = build_latent_previewer(
            vae_approx_path=os.path.join(_VAE_APPROX_DIR, "xlvaeapp.pth"),
        )
        # Simulate a partial latent during sampling: (1, 4, 128, 128)
        latent = torch.randn(1, 4, 128, 128).to(torch.device("cuda" if _has_gpu else "cpu"))

        result = previewer.preview(latent)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255
