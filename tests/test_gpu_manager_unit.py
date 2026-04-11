"""Unit tests for GPU memory management — no GPU required.

Tests use fakes and stubs for torch/ldm_patched dependencies.
Organized by acceptance criteria from UNF-39.

AC1: modules/infrastructure/gpu_manager.py exists implementing ModelManager protocol
AC2: get_torch_device() returns cuda:0 when CUDA available, cpu otherwise
AC3: load_models_to_gpu() moves models to GPU VRAM
AC4: cleanup() frees cached VRAM and runs garbage collection
AC5: get_vram_stats() returns accurate total/used/free VRAM
AC6: should_use_fp16() returns True for consumer GPUs
AC7: OOM does not crash — caught and reported as GPUMemoryError
AC9: Unit tests pass without GPU using fakes
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# AC1: Module exists and implements ModelManager protocol
# ---------------------------------------------------------------------------


class TestGPUManagerModuleExists:
    """AC1: modules/infrastructure/gpu_manager.py exists and exports LdmGPUManager."""

    def test_infrastructure_gpu_manager_is_importable(self) -> None:
        from modules.infrastructure import gpu_manager  # noqa: F401

    def test_ldm_gpu_manager_class_exists(self) -> None:
        from modules.infrastructure.gpu_manager import LdmGPUManager

        assert LdmGPUManager is not None

    def test_ldm_gpu_manager_satisfies_model_manager_protocol(self) -> None:
        from modules.domain.protocols import ModelManager

        manager = _make_gpu_manager()
        assert isinstance(manager, ModelManager)


# ---------------------------------------------------------------------------
# AC2: get_torch_device() returns correct device
# ---------------------------------------------------------------------------


class TestGetTorchDevice:
    """AC2: get_torch_device() returns cuda:0 when CUDA available, cpu otherwise."""

    def test_returns_cuda_device_when_cuda_available(self) -> None:
        fake_device = MagicMock()
        fake_device.type = "cuda"
        fake_ldm = MagicMock()
        fake_ldm.get_torch_device.return_value = fake_device

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        result = manager.get_torch_device()

        assert result is fake_device
        fake_ldm.get_torch_device.assert_called_once()

    def test_returns_cpu_device_when_no_cuda(self) -> None:
        fake_device = MagicMock()
        fake_device.type = "cpu"
        fake_ldm = MagicMock()
        fake_ldm.get_torch_device.return_value = fake_device

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        result = manager.get_torch_device()

        assert result.type == "cpu"


# ---------------------------------------------------------------------------
# AC3: load_models_to_gpu() moves models to GPU VRAM
# ---------------------------------------------------------------------------


class TestLoadModelsToGPU:
    """AC3: load_models_to_gpu() moves models to GPU VRAM."""

    def test_delegates_to_ldm_load_models_gpu(self) -> None:
        fake_ldm = MagicMock()
        manager = _make_gpu_manager(ldm_model_management=fake_ldm)

        model_a = MagicMock()
        model_b = MagicMock()
        manager.load_models_to_gpu([model_a, model_b])

        fake_ldm.load_models_gpu.assert_called_once_with([model_a, model_b])

    def test_empty_model_list_does_not_raise(self) -> None:
        fake_ldm = MagicMock()
        manager = _make_gpu_manager(ldm_model_management=fake_ldm)

        manager.load_models_to_gpu([])
        fake_ldm.load_models_gpu.assert_called_once_with([])


# ---------------------------------------------------------------------------
# AC4: cleanup() frees cached VRAM and runs garbage collection
# ---------------------------------------------------------------------------


class TestCleanup:
    """AC4: cleanup() frees cached VRAM and runs garbage collection."""

    def test_cleanup_calls_soft_empty_cache(self) -> None:
        fake_ldm = MagicMock()
        manager = _make_gpu_manager(ldm_model_management=fake_ldm)

        with patch("modules.infrastructure.gpu_manager.gc"):
            manager.cleanup()

        fake_ldm.soft_empty_cache.assert_called_once()

    def test_cleanup_calls_gc_collect(self) -> None:
        fake_ldm = MagicMock()
        manager = _make_gpu_manager(ldm_model_management=fake_ldm)

        with patch("modules.infrastructure.gpu_manager.gc") as mock_gc:
            manager.cleanup()
            mock_gc.collect.assert_called_once()

    def test_cleanup_calls_cleanup_models(self) -> None:
        fake_ldm = MagicMock()
        manager = _make_gpu_manager(ldm_model_management=fake_ldm)

        with patch("modules.infrastructure.gpu_manager.gc"):
            manager.cleanup()

        fake_ldm.cleanup_models.assert_called_once()


# ---------------------------------------------------------------------------
# AC5: get_vram_stats() returns VRAMStats value object
# ---------------------------------------------------------------------------


class TestGetVRAMStats:
    """AC5: get_vram_stats() returns accurate total/used/free VRAM."""

    def test_returns_vram_stats_value_object(self) -> None:
        from modules.domain.protocols import VRAMStats

        fake_ldm = MagicMock()
        fake_ldm.get_total_memory.return_value = 8 * 1024**3  # 8 GB
        fake_ldm.get_free_memory.return_value = 5 * 1024**3  # 5 GB free

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        stats = manager.get_vram_stats()

        assert isinstance(stats, VRAMStats)

    def test_total_matches_ldm_total_memory(self) -> None:
        fake_ldm = MagicMock()
        fake_ldm.get_total_memory.return_value = 8 * 1024**3
        fake_ldm.get_free_memory.return_value = 5 * 1024**3

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        stats = manager.get_vram_stats()

        assert stats.total_bytes == 8 * 1024**3

    def test_free_matches_ldm_free_memory(self) -> None:
        fake_ldm = MagicMock()
        fake_ldm.get_total_memory.return_value = 8 * 1024**3
        fake_ldm.get_free_memory.return_value = 5 * 1024**3

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        stats = manager.get_vram_stats()

        assert stats.free_bytes == 5 * 1024**3

    def test_used_is_total_minus_free(self) -> None:
        fake_ldm = MagicMock()
        fake_ldm.get_total_memory.return_value = 8 * 1024**3
        fake_ldm.get_free_memory.return_value = 5 * 1024**3

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        stats = manager.get_vram_stats()

        assert stats.used_bytes == 3 * 1024**3


# ---------------------------------------------------------------------------
# VRAMStats value object tests
# ---------------------------------------------------------------------------


class TestVRAMStatsValueObject:
    """VRAMStats is a frozen dataclass value object."""

    def test_vram_stats_is_frozen(self) -> None:
        from modules.domain.protocols import VRAMStats

        stats = VRAMStats(total_bytes=8_000, used_bytes=3_000, free_bytes=5_000)
        with pytest.raises(FrozenInstanceError):
            stats.total_bytes = 0  # type: ignore[misc]

    def test_vram_stats_rejects_negative_total(self) -> None:
        from modules.domain.protocols import VRAMStats

        with pytest.raises(ValueError, match="total_bytes"):
            VRAMStats(total_bytes=-1, used_bytes=0, free_bytes=0)

    def test_vram_stats_rejects_negative_used(self) -> None:
        from modules.domain.protocols import VRAMStats

        with pytest.raises(ValueError, match="used_bytes"):
            VRAMStats(total_bytes=100, used_bytes=-1, free_bytes=100)

    def test_vram_stats_rejects_negative_free(self) -> None:
        from modules.domain.protocols import VRAMStats

        with pytest.raises(ValueError, match="free_bytes"):
            VRAMStats(total_bytes=100, used_bytes=100, free_bytes=-1)

    def test_vram_stats_repr(self) -> None:
        from modules.domain.protocols import VRAMStats

        stats = VRAMStats(total_bytes=8_000_000_000, used_bytes=3_000_000_000, free_bytes=5_000_000_000)
        r = repr(stats)
        assert "total_bytes=8000000000" in r

    def test_total_mb_property(self) -> None:
        from modules.domain.protocols import VRAMStats

        stats = VRAMStats(total_bytes=8 * 1024**2, used_bytes=0, free_bytes=8 * 1024**2)
        assert stats.total_mb == pytest.approx(8.0)

    def test_used_mb_property(self) -> None:
        from modules.domain.protocols import VRAMStats

        stats = VRAMStats(total_bytes=8 * 1024**2, used_bytes=3 * 1024**2, free_bytes=5 * 1024**2)
        assert stats.used_mb == pytest.approx(3.0)

    def test_free_mb_property(self) -> None:
        from modules.domain.protocols import VRAMStats

        stats = VRAMStats(total_bytes=8 * 1024**2, used_bytes=3 * 1024**2, free_bytes=5 * 1024**2)
        assert stats.free_mb == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# AC6: should_use_fp16() returns True for consumer GPUs
# ---------------------------------------------------------------------------


class TestShouldUseFP16:
    """AC6: should_use_fp16() returns True for consumer GPUs."""

    def test_delegates_to_ldm_should_use_fp16(self) -> None:
        fake_ldm = MagicMock()
        fake_ldm.should_use_fp16.return_value = True

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        result = manager.should_use_fp16()

        assert result is True
        fake_ldm.should_use_fp16.assert_called_once()

    def test_returns_false_when_ldm_says_false(self) -> None:
        fake_ldm = MagicMock()
        fake_ldm.should_use_fp16.return_value = False

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        result = manager.should_use_fp16()

        assert result is False


# ---------------------------------------------------------------------------
# AC7: OOM does not crash — caught and reported as GPUMemoryError
# ---------------------------------------------------------------------------


class TestOOMHandling:
    """AC7: OOM is caught and reported as GPUMemoryError."""

    def test_gpu_memory_error_domain_exception_exists(self) -> None:
        from modules.domain.exceptions import GPUMemoryError

        assert GPUMemoryError is not None

    def test_gpu_memory_error_is_exception_subclass(self) -> None:
        from modules.domain.exceptions import GPUMemoryError

        assert issubclass(GPUMemoryError, Exception)

    def test_load_models_oom_raises_gpu_memory_error(self) -> None:
        from modules.domain.exceptions import GPUMemoryError

        fake_ldm = MagicMock()
        # Simulate torch.cuda.OutOfMemoryError via RuntimeError
        fake_ldm.load_models_gpu.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        with pytest.raises(GPUMemoryError, match="out of memory"):
            manager.load_models_to_gpu([MagicMock()])

    def test_load_models_non_oom_runtime_error_propagates(self) -> None:
        fake_ldm = MagicMock()
        fake_ldm.load_models_gpu.side_effect = RuntimeError("unrelated error")

        manager = _make_gpu_manager(ldm_model_management=fake_ldm)
        with pytest.raises(RuntimeError, match="unrelated error"):
            manager.load_models_to_gpu([MagicMock()])


# ---------------------------------------------------------------------------
# AC9: Protocol expansion — new methods on ModelManager
# ---------------------------------------------------------------------------


class TestModelManagerProtocolExpansion:
    """AC9: ModelManager protocol includes new GPU management methods."""

    def test_model_manager_has_load_models_to_gpu(self) -> None:
        from modules.domain.protocols import ModelManager

        assert hasattr(ModelManager, "load_models_to_gpu")

    def test_model_manager_has_get_vram_stats(self) -> None:
        from modules.domain.protocols import ModelManager

        assert hasattr(ModelManager, "get_vram_stats")

    def test_model_manager_has_should_use_fp16(self) -> None:
        from modules.domain.protocols import ModelManager

        assert hasattr(ModelManager, "should_use_fp16")

    def test_model_manager_has_cleanup(self) -> None:
        """cleanup() replaces cleanup_models() with broader scope."""
        from modules.domain.protocols import ModelManager

        assert hasattr(ModelManager, "cleanup")


# ---------------------------------------------------------------------------
# Test helpers — factory for LdmGPUManager with injected fakes
# ---------------------------------------------------------------------------


def _make_gpu_manager(
    *,
    ldm_model_management: Any = None,
) -> Any:
    """Create an LdmGPUManager with injected fake ldm_model_management."""
    from modules.infrastructure.gpu_manager import LdmGPUManager

    if ldm_model_management is None:
        ldm_model_management = MagicMock()
    return LdmGPUManager(ldm_model_management=ldm_model_management)
