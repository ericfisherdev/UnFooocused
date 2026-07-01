"""Unit tests for LdmVAEDecoder and LdmLatentPreviewer — no GPU, no real models.

Tests use fakes for ldm_patched's VAEDecode/VAEDecodeTiled and VAEApprox
so we can verify decoding orchestration, tiled fallback, preview generation,
and protocol conformance without loading any model weights.

Acceptance Criteria covered:
  AC1: modules/infrastructure/vae_decoder.py exists implementing VAEDecoder protocol
  AC2: Decoding a valid latent produces numpy array shape (H, W, 3) uint8 0-255
  AC3: Tiled decoding (tile_size=512) works for images larger than 2048px
  AC4: VAEApprox preview produces a usable preview image from a partial latent
  AC6: Output is compatible with PIL Image.fromarray() for saving
  AC8: Unit tests pass without GPU using fakes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Fakes — stand-ins for ldm_patched VAE infrastructure
# ---------------------------------------------------------------------------


class FakeVAEDecodeOp:
    """Fake for ldm_patched VAEDecode node.

    Returns a tuple of (image_batch,) where image_batch is a fake torch-like
    tensor that has a .cpu().numpy() chain returning a numpy array.
    """

    def __init__(self, output_h: int = 128, output_w: int = 128) -> None:
        self._output_h = output_h
        self._output_w = output_w

    def decode(self, *, samples: Any, vae: Any) -> tuple[Any]:
        """Mimic VAEDecode.decode() -> (image_batch_tensor,)."""
        # Return a fake tensor-like object that behaves like torch output
        # Shape: (batch, H, W, 3) float32 in [0, 1]
        arr = np.random.default_rng(42).random((1, self._output_h, self._output_w, 3), dtype=np.float32)
        return (FakeTorchTensor(arr),)


class FakeVAEDecodeTiledOp:
    """Fake for ldm_patched VAEDecodeTiled node."""

    def __init__(self, output_h: int = 128, output_w: int = 128) -> None:
        self._output_h = output_h
        self._output_w = output_w
        self.calls: list[dict[str, Any]] = []

    def decode(self, *, samples: Any, vae: Any, tile_size: int) -> tuple[Any]:
        """Mimic VAEDecodeTiled.decode() -> (image_batch_tensor,)."""
        self.calls.append({"samples": samples, "vae": vae, "tile_size": tile_size})
        arr = np.random.default_rng(42).random((1, self._output_h, self._output_w, 3), dtype=np.float32)
        return (FakeTorchTensor(arr),)


class FakeTorchTensor:
    """Minimal fake for a torch.Tensor supporting .cpu().numpy() chain."""

    def __init__(self, arr: NDArray[Any]) -> None:
        self._arr = arr

    def cpu(self) -> FakeTorchTensor:
        return self

    def numpy(self) -> NDArray[Any]:
        return self._arr


class FakeVAEApproxModel:
    """Fake VAEApprox model that produces a preview-sized image.

    Mimics the real VAEApprox forward pass: takes (B, 4, H, W) latent,
    returns (B, 3, H*2, W*2) preview (after interpolation + conv layers).
    """

    def __init__(self) -> None:
        self.current_type = "float32"
        self.call_count = 0

    def __call__(self, x: Any) -> Any:
        """Mimic VAEApprox.forward() -> preview tensor."""
        self.call_count += 1
        # Input: (B, 4, H, W) -> Output: (B, 3, H*2, W*2) approx
        if isinstance(x, FakeLatentInput):
            out_h = x.shape[2] * 2
            out_w = x.shape[3] * 2
        else:
            out_h = 32
            out_w = 32
        # Return values in [-1, 1] range (before * 127.5 + 127.5 rescaling)
        arr = np.random.default_rng(99).random((1, 3, out_h, out_w), dtype=np.float32) * 2 - 1
        return FakePreviewTensor(arr)


class FakeLatentInput:
    """Fake latent tensor with shape attribute for preview testing."""

    def __init__(self, batch: int = 1, channels: int = 4, h: int = 16, w: int = 16) -> None:
        self.shape = (batch, channels, h, w)

    def to(self, dtype: Any) -> FakeLatentInput:
        return self


class FakePreviewTensor:
    """Fake tensor for VAEApprox output supporting rearrange-like operations."""

    def __init__(self, arr: NDArray[Any]) -> None:
        # arr shape: (B, C, H, W)
        self._arr = arr

    def __mul__(self, other: float) -> FakePreviewTensor:
        return FakePreviewTensor(self._arr * other)

    def __add__(self, other: float) -> FakePreviewTensor:
        return FakePreviewTensor(self._arr + other)

    def rearrange_bchw_to_bhwc(self) -> FakePreviewTensor:
        """Simulates einops.rearrange(x, 'b c h w -> b h w c')."""
        return FakePreviewTensor(np.transpose(self._arr, (0, 2, 3, 1)))

    def first_sample(self) -> FakePreviewTensor:
        """Get first batch element."""
        return FakePreviewTensor(self._arr[0])

    def cpu(self) -> FakePreviewTensor:
        return self

    def numpy(self) -> NDArray[Any]:
        return self._arr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_latent() -> dict[str, Any]:
    """Build a fake latent dict mimicking ldm_patched's format."""
    return {"samples": "fake_latent_tensor"}


def _make_vae_decoder(
    *,
    vae_decode_op: Any = None,
    vae_decode_tiled_op: Any = None,
    tiled_threshold: int = 2048,
    gpu_loader: Any = None,
) -> Any:
    """Build an LdmVAEDecoder with injected fakes.

    gpu_loader defaults to a no-op so unit tests never exercise the real
    ldm_patched.modules.model_management call, which requires a real GPU
    model object (see UNF-91 — decode() must delegate GPU loading to an
    injected collaborator, not hardcode the ldm_patched import).
    """
    from modules.infrastructure.vae_decoder import LdmVAEDecoder

    return LdmVAEDecoder(
        vae_decode_op=vae_decode_op or FakeVAEDecodeOp(),
        vae_decode_tiled_op=vae_decode_tiled_op or FakeVAEDecodeTiledOp(),
        tiled_threshold=tiled_threshold,
        gpu_loader=gpu_loader or (lambda vae: None),
    )


def _make_latent_previewer(
    *,
    vae_approx_model: Any = None,
    rearrange_fn: Any = None,
) -> Any:
    """Build an LdmLatentPreviewer with injected fakes."""
    from modules.infrastructure.vae_decoder import LdmLatentPreviewer

    def default_rearrange(tensor: Any) -> Any:
        """Fake rearrange: (B, C, H, W) -> (B, H, W, C), return first sample."""
        arr = tensor._arr if hasattr(tensor, "_arr") else tensor
        if isinstance(arr, np.ndarray) and arr.ndim == 4:
            transposed = np.transpose(arr, (0, 2, 3, 1))
            return FakeTorchTensor(transposed[0])
        return FakeTorchTensor(arr)

    return LdmLatentPreviewer(
        vae_approx_model=vae_approx_model or FakeVAEApproxModel(),
        rearrange_fn=rearrange_fn or default_rearrange,
    )


# ---------------------------------------------------------------------------
# AC1: modules/infrastructure/vae_decoder.py exists implementing protocols
# ---------------------------------------------------------------------------


class TestVAEDecoderExistsAndSatisfiesProtocol:
    """AC1: vae_decoder.py exists and LdmVAEDecoder satisfies VAEDecoder protocol."""

    def test_module_importable(self) -> None:
        from modules.infrastructure.vae_decoder import LdmVAEDecoder

        assert LdmVAEDecoder is not None

    def test_satisfies_vae_decoder_protocol(self) -> None:
        from modules.domain.protocols import VAEDecoder

        decoder = _make_vae_decoder()
        assert isinstance(decoder, VAEDecoder)

    def test_previewer_importable(self) -> None:
        from modules.infrastructure.vae_decoder import LdmLatentPreviewer

        assert LdmLatentPreviewer is not None

    def test_previewer_satisfies_latent_previewer_protocol(self) -> None:
        from modules.domain.protocols import LatentPreviewer

        previewer = _make_latent_previewer()
        assert isinstance(previewer, LatentPreviewer)


# ---------------------------------------------------------------------------
# AC2: Decoding a valid latent produces numpy array shape (H, W, 3) uint8
# ---------------------------------------------------------------------------


class TestDecodeProducesCorrectOutput:
    """AC2: decode() returns list of numpy arrays with shape (H, W, 3) uint8."""

    def test_decode_returns_list_of_numpy_arrays(self) -> None:
        decoder = _make_vae_decoder()
        latent = _make_fake_latent()

        result = decoder.decode(vae="fake_vae", latent=latent)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_decode_output_shape_is_hwc(self) -> None:
        decoder = _make_vae_decoder(vae_decode_op=FakeVAEDecodeOp(output_h=256, output_w=384))
        latent = _make_fake_latent()

        result = decoder.decode(vae="fake_vae", latent=latent)

        arr = result[0]
        assert arr.ndim == 3
        assert arr.shape == (256, 384, 3)

    def test_decode_output_dtype_is_uint8(self) -> None:
        decoder = _make_vae_decoder()
        latent = _make_fake_latent()

        result = decoder.decode(vae="fake_vae", latent=latent)

        assert result[0].dtype == np.uint8

    def test_decode_output_values_in_0_255(self) -> None:
        decoder = _make_vae_decoder()
        latent = _make_fake_latent()

        result = decoder.decode(vae="fake_vae", latent=latent)

        arr = result[0]
        assert arr.min() >= 0
        assert arr.max() <= 255


# ---------------------------------------------------------------------------
# AC3: Tiled decoding for large images (>2048px)
# ---------------------------------------------------------------------------


class TestTiledDecoding:
    """AC3: Tiled decoding (tile_size=512) works for large images."""

    def test_tiled_decoding_used_for_large_images(self) -> None:
        """When latent implies image > tiled_threshold, use tiled decode."""
        tiled_op = FakeVAEDecodeTiledOp(output_h=2560, output_w=2560)
        decoder = _make_vae_decoder(
            vae_decode_tiled_op=tiled_op,
            tiled_threshold=2048,
        )
        # Latent for 2560x2560 image: latent is 1/8 size = 320x320
        latent = {"samples": FakeLatentInput(1, 4, 320, 320)}

        decoder.decode(vae="fake_vae", latent=latent)

        assert len(tiled_op.calls) == 1
        assert tiled_op.calls[0]["tile_size"] == 512

    def test_standard_decoding_used_for_small_images(self) -> None:
        """When latent implies image <= tiled_threshold, use standard decode."""
        tiled_op = FakeVAEDecodeTiledOp()
        decoder = _make_vae_decoder(
            vae_decode_tiled_op=tiled_op,
            tiled_threshold=2048,
        )
        # Latent for 1024x1024 image: latent is 128x128
        latent = {"samples": FakeLatentInput(1, 4, 128, 128)}

        decoder.decode(vae="fake_vae", latent=latent)

        assert len(tiled_op.calls) == 0  # tiled not used

    def test_tiled_output_still_correct_shape_and_dtype(self) -> None:
        tiled_op = FakeVAEDecodeTiledOp(output_h=2560, output_w=2560)
        decoder = _make_vae_decoder(
            vae_decode_tiled_op=tiled_op,
            tiled_threshold=2048,
        )
        latent = {"samples": FakeLatentInput(1, 4, 320, 320)}

        result = decoder.decode(vae="fake_vae", latent=latent)

        arr = result[0]
        assert arr.shape == (2560, 2560, 3)
        assert arr.dtype == np.uint8


# ---------------------------------------------------------------------------
# AC4: VAEApprox preview produces a usable preview image
# ---------------------------------------------------------------------------


class TestLatentPreviewGeneration:
    """AC4: VAEApprox preview produces low-res RGB image from latent."""

    def test_preview_returns_numpy_array(self) -> None:
        previewer = _make_latent_previewer()
        latent = FakeLatentInput(1, 4, 16, 16)

        result = previewer.preview(latent)

        assert isinstance(result, np.ndarray)

    def test_preview_output_is_rgb(self) -> None:
        previewer = _make_latent_previewer()
        latent = FakeLatentInput(1, 4, 16, 16)

        result = previewer.preview(latent)

        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB channels

    def test_preview_output_dtype_is_uint8(self) -> None:
        previewer = _make_latent_previewer()
        latent = FakeLatentInput(1, 4, 16, 16)

        result = previewer.preview(latent)

        assert result.dtype == np.uint8

    def test_preview_output_values_clipped_to_0_255(self) -> None:
        previewer = _make_latent_previewer()
        latent = FakeLatentInput(1, 4, 16, 16)

        result = previewer.preview(latent)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_preview_calls_vae_approx_model(self) -> None:
        model = FakeVAEApproxModel()
        previewer = _make_latent_previewer(vae_approx_model=model)
        latent = FakeLatentInput(1, 4, 16, 16)

        previewer.preview(latent)

        assert model.call_count == 1


# ---------------------------------------------------------------------------
# AC6: Output compatible with PIL Image.fromarray()
# ---------------------------------------------------------------------------


class TestPILCompatibility:
    """AC6: Output is compatible with PIL Image.fromarray() for saving."""

    def test_decode_output_pil_compatible(self) -> None:
        from PIL import Image

        decoder = _make_vae_decoder()
        latent = _make_fake_latent()

        result = decoder.decode(vae="fake_vae", latent=latent)

        # Should not raise
        img = Image.fromarray(result[0])
        assert img.mode == "RGB"

    def test_preview_output_pil_compatible(self) -> None:
        from PIL import Image

        previewer = _make_latent_previewer()
        latent = FakeLatentInput(1, 4, 16, 16)

        result = previewer.preview(latent)

        # Should not raise
        img = Image.fromarray(result)
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# AC8: Repr and basic construction
# ---------------------------------------------------------------------------


class TestDecoderRepr:
    """Verify repr and basic construction."""

    def test_vae_decoder_repr(self) -> None:
        decoder = _make_vae_decoder()
        assert "LdmVAEDecoder" in repr(decoder)

    def test_latent_previewer_repr(self) -> None:
        previewer = _make_latent_previewer()
        assert "LdmLatentPreviewer" in repr(previewer)


# ---------------------------------------------------------------------------
# UNF-91: _load_vae_to_gpu must resolve vae.patcher, not the raw VAE wrapper
# ---------------------------------------------------------------------------


class TestLoadVaeToGpuResolvesPatcher:
    """UNF-91: the default gpu_loader must load vae.patcher, not the bare
    ldm_patched VAE wrapper.

    ldm_patched's VAE wrapper has no ``load_device`` attribute of its own —
    only its internal ``.patcher`` (a ModelPatcher) does. Passing the raw
    VAE to load_models_gpu() crashes with
    ``AttributeError: 'VAE' object has no attribute 'load_device'``.
    """

    def test_loads_patcher_when_present(self, monkeypatch) -> None:
        pytest.importorskip("torch", reason="ldm_patched deps required for model_management tests")
        pytest.importorskip("psutil", reason="ldm_patched deps required for model_management tests")
        import ldm_patched.modules.model_management as model_management
        from modules.infrastructure.vae_decoder import _load_vae_to_gpu

        calls: list[Any] = []
        monkeypatch.setattr(model_management, "load_models_gpu", calls.append)

        patcher = object()
        vae = type("FakeVAE", (), {"patcher": patcher})()

        _load_vae_to_gpu(vae)

        assert calls == [[patcher]]

    def test_falls_back_to_vae_when_no_patcher_attribute(self, monkeypatch) -> None:
        pytest.importorskip("torch", reason="ldm_patched deps required for model_management tests")
        pytest.importorskip("psutil", reason="ldm_patched deps required for model_management tests")
        import ldm_patched.modules.model_management as model_management
        from modules.infrastructure.vae_decoder import _load_vae_to_gpu

        calls: list[Any] = []
        monkeypatch.setattr(model_management, "load_models_gpu", calls.append)

        vae = object()  # no .patcher attribute — defensive fallback

        _load_vae_to_gpu(vae)

        assert calls == [[vae]]


class TestDecoderUsesInjectedGpuLoader:
    """UNF-91: LdmVAEDecoder.decode() delegates GPU loading to an injected
    collaborator instead of hardcoding an ldm_patched import, so it stays
    unit-testable without a GPU (Dependency Inversion Principle).
    """

    def test_decode_calls_gpu_loader_with_inner_vae(self) -> None:
        calls: list[Any] = []
        decoder = _make_vae_decoder(gpu_loader=calls.append)
        latent = _make_fake_latent()

        decoder.decode(vae="fake_vae", latent=latent)

        assert calls == ["fake_vae"]

    def test_decode_uses_real_loader_by_default(self, monkeypatch) -> None:
        """Without an explicit gpu_loader, the production _load_vae_to_gpu is used."""
        from modules.infrastructure.vae_decoder import LdmVAEDecoder

        calls: list[Any] = []
        monkeypatch.setattr(
            "modules.infrastructure.vae_decoder._load_vae_to_gpu",
            calls.append,
        )
        decoder = LdmVAEDecoder(
            vae_decode_op=FakeVAEDecodeOp(),
            vae_decode_tiled_op=FakeVAEDecodeTiledOp(),
        )
        latent = _make_fake_latent()

        decoder.decode(vae="fake_vae", latent=latent)

        assert calls == ["fake_vae"]
