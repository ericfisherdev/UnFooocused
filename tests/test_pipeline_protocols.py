"""Unit tests for pipeline protocols — UNF-32 acceptance criteria.

Tests are organized by acceptance criterion:
AC1: protocols.py exists with ModelLoader, TextEncoder, Sampler, VAEDecoder,
     ProgressCallback, ModelManager
AC2: Each protocol is @runtime_checkable and uses typing.Protocol
AC3: Stub/fake implementations satisfy each protocol (isinstance checks)
AC4: Protocols have no imports from torch or ldm_patched
AC5: Type aliases (LatentTensor, Conditioning, TorchDevice) are defined
AC6: Docstrings document expected behavior and error conditions
AC7: Unit tests verify protocol structural matching with >95% coverage
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# AC1: protocols.py exists with the required protocol classes
# ---------------------------------------------------------------------------


class TestProtocolModuleExists:
    """AC1: modules/domain/protocols.py exists and exports required names."""

    def test_protocols_module_is_importable(self) -> None:
        from modules.domain import protocols  # noqa: F401

    def test_model_loader_protocol_exists(self) -> None:
        from modules.domain.protocols import ModelLoader

        assert ModelLoader is not None

    def test_text_encoder_protocol_exists(self) -> None:
        from modules.domain.protocols import TextEncoder

        assert TextEncoder is not None

    def test_sampler_protocol_exists(self) -> None:
        from modules.domain.protocols import Sampler

        assert Sampler is not None

    def test_vae_decoder_protocol_exists(self) -> None:
        from modules.domain.protocols import VAEDecoder

        assert VAEDecoder is not None

    def test_progress_callback_protocol_exists(self) -> None:
        from modules.domain.protocols import ProgressCallback

        assert ProgressCallback is not None

    def test_model_manager_protocol_exists(self) -> None:
        from modules.domain.protocols import ModelManager

        assert ModelManager is not None


# ---------------------------------------------------------------------------
# AC2: Each protocol uses typing.Protocol and @runtime_checkable
# ---------------------------------------------------------------------------


class TestProtocolsAreRuntimeCheckable:
    """AC2: Each protocol is defined using typing.Protocol with @runtime_checkable."""

    def test_model_loader_is_protocol(self) -> None:
        from modules.domain.protocols import ModelLoader

        assert _is_protocol(ModelLoader)

    def test_model_loader_is_runtime_checkable(self) -> None:
        from modules.domain.protocols import ModelLoader

        assert _is_runtime_checkable(ModelLoader)

    def test_text_encoder_is_protocol(self) -> None:
        from modules.domain.protocols import TextEncoder

        assert _is_protocol(TextEncoder)

    def test_text_encoder_is_runtime_checkable(self) -> None:
        from modules.domain.protocols import TextEncoder

        assert _is_runtime_checkable(TextEncoder)

    def test_sampler_is_protocol(self) -> None:
        from modules.domain.protocols import Sampler

        assert _is_protocol(Sampler)

    def test_sampler_is_runtime_checkable(self) -> None:
        from modules.domain.protocols import Sampler

        assert _is_runtime_checkable(Sampler)

    def test_vae_decoder_is_protocol(self) -> None:
        from modules.domain.protocols import VAEDecoder

        assert _is_protocol(VAEDecoder)

    def test_vae_decoder_is_runtime_checkable(self) -> None:
        from modules.domain.protocols import VAEDecoder

        assert _is_runtime_checkable(VAEDecoder)

    def test_progress_callback_is_protocol(self) -> None:
        from modules.domain.protocols import ProgressCallback

        assert _is_protocol(ProgressCallback)

    def test_progress_callback_is_runtime_checkable(self) -> None:
        from modules.domain.protocols import ProgressCallback

        assert _is_runtime_checkable(ProgressCallback)

    def test_model_manager_is_protocol(self) -> None:
        from modules.domain.protocols import ModelManager

        assert _is_protocol(ModelManager)

    def test_model_manager_is_runtime_checkable(self) -> None:
        from modules.domain.protocols import ModelManager

        assert _is_runtime_checkable(ModelManager)


# ---------------------------------------------------------------------------
# AC3: Stub/fake implementations satisfy each protocol (isinstance checks)
# ---------------------------------------------------------------------------


class TestFakeModelLoaderSatisfiesProtocol:
    """AC3: A FakeModelLoader satisfies the ModelLoader protocol."""

    def test_fake_model_loader_isinstance_check(self) -> None:
        from modules.domain.protocols import ModelLoader

        fake = _FakeModelLoader()
        assert isinstance(fake, ModelLoader)

    def test_fake_model_loader_load_checkpoint_returns_value(self) -> None:
        fake = _FakeModelLoader()
        result = fake.load_checkpoint("/fake/path.safetensors")
        assert result is not None

    def test_fake_model_loader_load_loras_returns_model(self) -> None:
        from modules.domain.protocols import LoRAConfig

        fake = _FakeModelLoader()
        model = fake.load_checkpoint("/fake/path.safetensors")
        lora = LoRAConfig(filename="lora.safetensors", weight=0.8)
        result = fake.load_loras(model, [lora])
        assert result is not None


class TestFakeTextEncoderSatisfiesProtocol:
    """AC3: A FakeTextEncoder satisfies the TextEncoder protocol."""

    def test_fake_text_encoder_isinstance_check(self) -> None:
        from modules.domain.protocols import TextEncoder

        fake = _FakeTextEncoder()
        assert isinstance(fake, TextEncoder)

    def test_fake_text_encoder_encode_returns_conditioning(self) -> None:
        fake = _FakeTextEncoder()
        result = fake.encode(["a photo of a cat"], clip_skip=1)
        assert result is not None

    def test_fake_text_encoder_clear_cache_returns_none(self) -> None:
        fake = _FakeTextEncoder()
        result = fake.clear_cache()
        assert result is None


class TestFakeSamplerSatisfiesProtocol:
    """AC3: A FakeSampler satisfies the Sampler protocol."""

    def test_fake_sampler_isinstance_check(self) -> None:
        from modules.domain.protocols import Sampler

        fake = _FakeSampler()
        assert isinstance(fake, Sampler)

    def test_fake_sampler_sample_returns_latent_tensor(self) -> None:
        from modules.domain.protocols import SamplerConfig

        fake = _FakeSampler()
        config = SamplerConfig(
            sampler_name="euler",
            scheduler="normal",
            steps=20,
            cfg_scale=7.0,
            seed=42,
            denoise=1.0,
        )
        result = fake.sample(
            model=object(),
            positive=object(),
            negative=object(),
            latent=object(),
            config=config,
            callback=None,
        )
        assert result is not None


class TestFakeVAEDecoderSatisfiesProtocol:
    """AC3: A FakeVAEDecoder satisfies the VAEDecoder protocol."""

    def test_fake_vae_decoder_isinstance_check(self) -> None:
        from modules.domain.protocols import VAEDecoder

        fake = _FakeVAEDecoder()
        assert isinstance(fake, VAEDecoder)

    def test_fake_vae_decoder_decode_returns_ndarray_list(self) -> None:
        fake = _FakeVAEDecoder()
        result = fake.decode(vae=object(), latent=object())
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)


class TestFakeProgressCallbackSatisfiesProtocol:
    """AC3: A FakeProgressCallback satisfies the ProgressCallback protocol."""

    def test_fake_progress_callback_isinstance_check(self) -> None:
        from modules.domain.protocols import ProgressCallback

        fake = _FakeProgressCallback()
        assert isinstance(fake, ProgressCallback)

    def test_fake_progress_callback_is_callable(self) -> None:
        fake = _FakeProgressCallback()
        fake(step=1, total=10, preview_image=None)


class TestFakeModelManagerSatisfiesProtocol:
    """AC3: A FakeModelManager satisfies the ModelManager protocol."""

    def test_fake_model_manager_isinstance_check(self) -> None:
        from modules.domain.protocols import ModelManager

        fake = _FakeModelManager()
        assert isinstance(fake, ModelManager)

    def test_fake_model_manager_get_torch_device_returns_value(self) -> None:
        fake = _FakeModelManager()
        result = fake.get_torch_device()
        assert result is not None

    def test_fake_model_manager_cleanup_models_returns_none(self) -> None:
        fake = _FakeModelManager()
        result = fake.cleanup_models()
        assert result is None


# ---------------------------------------------------------------------------
# AC4: No imports from torch or ldm_patched in protocols.py
# ---------------------------------------------------------------------------


class TestProtocolsHaveNoForbiddenImports:
    """AC4: Protocols have no imports from torch or ldm_patched."""

    def test_no_torch_import(self) -> None:
        source = _get_protocols_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("torch"), (
                        f"protocols.py must not import torch, found: import {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("torch"), (
                    f"protocols.py must not import from torch, found: from {node.module}"
                )

    def test_no_ldm_patched_import(self) -> None:
        source = _get_protocols_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("ldm_patched"), (
                        f"protocols.py must not import ldm_patched, found: import {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("ldm_patched"), (
                    f"protocols.py must not import from ldm_patched, found: from {node.module}"
                )


# ---------------------------------------------------------------------------
# AC5: Type aliases are defined for clarity
# ---------------------------------------------------------------------------


class TestTypeAliasesExist:
    """AC5: Type aliases (LatentTensor, Conditioning, TorchDevice) are defined."""

    def test_latent_tensor_alias_exists(self) -> None:
        from modules.domain.protocols import LatentTensor

        assert LatentTensor is not None

    def test_conditioning_alias_exists(self) -> None:
        from modules.domain.protocols import Conditioning

        assert Conditioning is not None

    def test_torch_device_alias_exists(self) -> None:
        from modules.domain.protocols import TorchDevice

        assert TorchDevice is not None


# ---------------------------------------------------------------------------
# AC6: Docstrings document expected behavior and error conditions
# ---------------------------------------------------------------------------


class TestProtocolsHaveDocstrings:
    """AC6: Docstrings document expected behavior and error conditions."""

    def test_model_loader_has_class_docstring(self) -> None:
        from modules.domain.protocols import ModelLoader

        assert ModelLoader.__doc__ is not None
        assert len(ModelLoader.__doc__.strip()) > 0

    def test_model_loader_load_checkpoint_has_docstring(self) -> None:
        from modules.domain.protocols import ModelLoader

        assert ModelLoader.load_checkpoint.__doc__ is not None

    def test_model_loader_load_loras_has_docstring(self) -> None:
        from modules.domain.protocols import ModelLoader

        assert ModelLoader.load_loras.__doc__ is not None

    def test_text_encoder_has_class_docstring(self) -> None:
        from modules.domain.protocols import TextEncoder

        assert TextEncoder.__doc__ is not None

    def test_text_encoder_encode_has_docstring(self) -> None:
        from modules.domain.protocols import TextEncoder

        assert TextEncoder.encode.__doc__ is not None

    def test_text_encoder_clear_cache_has_docstring(self) -> None:
        from modules.domain.protocols import TextEncoder

        assert TextEncoder.clear_cache.__doc__ is not None

    def test_sampler_has_class_docstring(self) -> None:
        from modules.domain.protocols import Sampler

        assert Sampler.__doc__ is not None

    def test_sampler_sample_has_docstring(self) -> None:
        from modules.domain.protocols import Sampler

        assert Sampler.sample.__doc__ is not None

    def test_vae_decoder_has_class_docstring(self) -> None:
        from modules.domain.protocols import VAEDecoder

        assert VAEDecoder.__doc__ is not None

    def test_vae_decoder_decode_has_docstring(self) -> None:
        from modules.domain.protocols import VAEDecoder

        assert VAEDecoder.decode.__doc__ is not None

    def test_progress_callback_has_class_docstring(self) -> None:
        from modules.domain.protocols import ProgressCallback

        assert ProgressCallback.__doc__ is not None

    def test_progress_callback_call_has_docstring(self) -> None:
        from modules.domain.protocols import ProgressCallback

        assert ProgressCallback.__call__.__doc__ is not None

    def test_model_manager_has_class_docstring(self) -> None:
        from modules.domain.protocols import ModelManager

        assert ModelManager.__doc__ is not None

    def test_model_manager_get_torch_device_has_docstring(self) -> None:
        from modules.domain.protocols import ModelManager

        assert ModelManager.get_torch_device.__doc__ is not None

    def test_model_manager_cleanup_models_has_docstring(self) -> None:
        from modules.domain.protocols import ModelManager

        assert ModelManager.cleanup_models.__doc__ is not None


# ---------------------------------------------------------------------------
# AC7 supplement: Domain model dataclasses exist and are well-formed
# ---------------------------------------------------------------------------


class TestDomainModelDataclasses:
    """Domain models (LoRAConfig, SamplerConfig, StableDiffusionModel) exist."""

    def test_lora_config_is_importable(self) -> None:
        from modules.domain.protocols import LoRAConfig

        assert LoRAConfig is not None

    def test_lora_config_has_filename_field(self) -> None:
        from modules.domain.protocols import LoRAConfig

        hints = get_type_hints(LoRAConfig)
        assert "filename" in hints

    def test_lora_config_has_weight_field(self) -> None:
        from modules.domain.protocols import LoRAConfig

        hints = get_type_hints(LoRAConfig)
        assert "weight" in hints

    def test_lora_config_is_frozen_dataclass(self) -> None:
        from modules.domain.protocols import LoRAConfig

        config = LoRAConfig(filename="test.safetensors", weight=0.8)
        try:
            config.filename = "other.safetensors"  # type: ignore[misc]
            raise AssertionError("LoRAConfig should be frozen")
        except AttributeError:
            pass  # expected — frozen dataclass

    def test_sampler_config_is_importable(self) -> None:
        from modules.domain.protocols import SamplerConfig

        assert SamplerConfig is not None

    def test_sampler_config_has_required_fields(self) -> None:
        from modules.domain.protocols import SamplerConfig

        hints = get_type_hints(SamplerConfig)
        expected_fields = {"sampler_name", "scheduler", "steps", "cfg_scale", "seed", "denoise"}
        assert expected_fields.issubset(hints.keys()), f"Missing fields: {expected_fields - hints.keys()}"

    def test_sampler_config_is_frozen_dataclass(self) -> None:
        from modules.domain.protocols import SamplerConfig

        config = SamplerConfig(
            sampler_name="euler",
            scheduler="normal",
            steps=20,
            cfg_scale=7.0,
            seed=42,
            denoise=1.0,
        )
        try:
            config.steps = 30  # type: ignore[misc]
            raise AssertionError("SamplerConfig should be frozen")
        except AttributeError:
            pass  # expected — frozen dataclass

    def test_stable_diffusion_model_alias_exists(self) -> None:
        from modules.domain.protocols import StableDiffusionModel

        assert StableDiffusionModel is not None


# ---------------------------------------------------------------------------
# AC7 supplement: Protocol method signatures have type annotations
# ---------------------------------------------------------------------------


class TestProtocolMethodsAreAnnotated:
    """AC7: Protocol methods have proper type annotations."""

    def test_model_loader_load_checkpoint_has_return_type(self) -> None:
        from modules.domain.protocols import ModelLoader

        hints = get_type_hints(ModelLoader.load_checkpoint)
        assert "return" in hints

    def test_model_loader_load_checkpoint_has_path_param(self) -> None:
        from modules.domain.protocols import ModelLoader

        hints = get_type_hints(ModelLoader.load_checkpoint)
        assert "path" in hints

    def test_model_loader_load_loras_has_return_type(self) -> None:
        from modules.domain.protocols import ModelLoader

        hints = get_type_hints(ModelLoader.load_loras)
        assert "return" in hints

    def test_text_encoder_encode_has_return_type(self) -> None:
        from modules.domain.protocols import TextEncoder

        hints = get_type_hints(TextEncoder.encode)
        assert "return" in hints

    def test_sampler_sample_has_return_type(self) -> None:
        from modules.domain.protocols import Sampler

        hints = get_type_hints(Sampler.sample)
        assert "return" in hints

    def test_vae_decoder_decode_has_return_type(self) -> None:
        from modules.domain.protocols import VAEDecoder

        hints = get_type_hints(VAEDecoder.decode)
        assert "return" in hints

    def test_model_manager_get_torch_device_has_return_type(self) -> None:
        from modules.domain.protocols import ModelManager

        hints = get_type_hints(ModelManager.get_torch_device)
        assert "return" in hints

    def test_progress_callback_call_has_step_param(self) -> None:
        from modules.domain.protocols import ProgressCallback

        hints = get_type_hints(ProgressCallback.__call__)
        assert "step" in hints

    def test_progress_callback_call_has_total_param(self) -> None:
        from modules.domain.protocols import ProgressCallback

        hints = get_type_hints(ProgressCallback.__call__)
        assert "total" in hints

    def test_progress_callback_call_has_preview_image_param(self) -> None:
        from modules.domain.protocols import ProgressCallback

        hints = get_type_hints(ProgressCallback.__call__)
        assert "preview_image" in hints


# ===========================================================================
# Test helpers — fake implementations for protocol satisfaction tests
# ===========================================================================


def _is_protocol(cls: type) -> bool:
    """Check if cls is a typing.Protocol subclass."""
    return getattr(cls, "_is_protocol", False)


def _is_runtime_checkable(cls: type) -> bool:
    """Check if cls is marked @runtime_checkable."""
    return getattr(cls, "_is_runtime_protocol", False)


def _get_protocols_source() -> str:
    """Read the source code of protocols.py for AST analysis."""
    protocols_path = Path(__file__).resolve().parent.parent / "modules" / "domain" / "protocols.py"
    return protocols_path.read_text()


class _FakeModelLoader:
    def load_checkpoint(self, path: str) -> Any:
        return {"unet": None, "clip": None, "vae": None}

    def load_loras(self, model: Any, loras: list[Any]) -> Any:
        return model


class _FakeTextEncoder:
    def encode(self, texts: list[str], clip_skip: int) -> Any:
        return [["fake_conditioning", {"pooled_output": "fake_pooled"}]]

    def clear_cache(self) -> None:
        pass


class _FakeSampler:
    def sample(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        latent: Any,
        config: Any,
        callback: Any,
    ) -> Any:
        return {"samples": "fake_latent_tensor"}


class _FakeVAEDecoder:
    def decode(self, vae: Any, latent: Any) -> list[NDArray[np.uint8]]:
        return [np.zeros((512, 512, 3), dtype=np.uint8)]


class _FakeProgressCallback:
    def __call__(self, step: int, total: int, preview_image: Any | None) -> None:
        pass


class _FakeModelManager:
    def get_torch_device(self) -> Any:
        return "cpu"

    def cleanup_models(self) -> None:
        pass
