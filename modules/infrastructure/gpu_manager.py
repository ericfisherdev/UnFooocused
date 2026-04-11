"""LdmGPUManager — concrete ModelManager using ldm_patched.

Bridges the domain ModelManager protocol to ldm_patched's model_management
module for GPU device queries, VRAM stats, model loading, and memory cleanup.
All torch and ldm_patched dependencies are confined to this module.

Domain errors raised:
    GPUMemoryError — CUDA out-of-memory during model loading.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any

from modules.domain.exceptions import GPUMemoryError
from modules.domain.protocols import VRAMStats

if TYPE_CHECKING:
    from modules.domain.protocols import TorchDevice

logger = logging.getLogger(__name__)

_OOM_MARKER = "out of memory"


class LdmGPUManager:
    """Concrete ModelManager adapter backed by ldm_patched.model_management.

    Dependencies are injected via constructor so the class can be tested
    with fakes in environments without a GPU.

    Args:
        ldm_model_management: The ldm_patched.modules.model_management module
            (or a fake with the same interface).
    """

    __slots__ = ("_ldm",)

    def __init__(self, *, ldm_model_management: Any) -> None:
        self._ldm = ldm_model_management

    def get_torch_device(self) -> TorchDevice:
        """Return the primary torch device for inference.

        Delegates to ldm_patched.modules.model_management.get_torch_device().

        Returns:
            A torch.device — typically cuda:0 or cpu.
        """
        return self._ldm.get_torch_device()

    def cleanup_models(self) -> None:
        """Free GPU memory by unloading cached models.

        Delegates to ldm_patched.modules.model_management.cleanup_models().
        """
        self._ldm.cleanup_models()

    def cleanup(self) -> None:
        """Full cleanup: unload cached models, empty GPU cache, run GC.

        Performs three operations in sequence:
        1. cleanup_models — unload unreferenced model patchers
        2. soft_empty_cache — flush the CUDA/MPS memory allocator cache
        3. gc.collect — trigger Python garbage collection
        """
        self._ldm.cleanup_models()
        self._ldm.soft_empty_cache()
        gc.collect()
        logger.debug("GPU cleanup complete: models unloaded, cache flushed, GC ran")

    def load_models_to_gpu(self, models: list[Any]) -> None:
        """Move specified models to GPU VRAM, offloading others if needed.

        Delegates to ldm_patched.modules.model_management.load_models_gpu().
        Catches CUDA OOM and translates to GPUMemoryError.

        Args:
            models: List of model patcher objects to load onto the GPU.

        Raises:
            GPUMemoryError: If GPU runs out of memory during loading.
            RuntimeError: For non-OOM runtime errors (re-raised as-is).
        """
        try:
            self._ldm.load_models_gpu(models)
        except RuntimeError as exc:
            if _OOM_MARKER in str(exc).lower():
                raise GPUMemoryError(f"GPU out of memory while loading models: {exc}") from exc
            raise

    def get_vram_stats(self) -> VRAMStats:
        """Return current VRAM usage statistics.

        Uses ldm_patched get_total_memory and get_free_memory to compute
        total, free, and used bytes.

        Returns:
            A frozen VRAMStats value object.
        """
        dev = self.get_torch_device()
        total = self._ldm.get_total_memory(dev=dev)
        free = self._ldm.get_free_memory(dev=dev)
        used = max(0, total - free)
        return VRAMStats(total_bytes=total, used_bytes=used, free_bytes=free)

    def should_use_fp16(self) -> bool:
        """Determine if fp16 inference should be used based on GPU capability.

        Delegates to ldm_patched.modules.model_management.should_use_fp16().

        Returns:
            True if fp16 is recommended for this GPU.
        """
        return self._ldm.should_use_fp16()
