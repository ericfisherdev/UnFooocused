"""Pipeline protocols (ports) for the UnFooocused image generation domain.

Defines Protocol-based interfaces that describe the contracts for model loading,
text encoding, sampling, and VAE decoding. These protocols allow the domain/service
layer to depend on abstractions rather than concrete ldm_patched implementations,
enabling unit testing without a GPU and following the Dependency Inversion Principle.

No imports from torch or ldm_patched are permitted in this module. Tensor types
are represented as opaque type aliases using Any.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray  # noqa: TC002 — needed at runtime for get_type_hints

# ---------------------------------------------------------------------------
# Type aliases — opaque wrappers for torch/ldm_patched types that the domain
# layer references without importing the concrete libraries.
# ---------------------------------------------------------------------------

LatentTensor = Any
"""Opaque alias for a latent-space tensor (torch.Tensor dict with 'samples' key)."""

Conditioning = Any
"""Opaque alias for CLIP conditioning output (list of [tensor, dict] pairs)."""

TorchDevice = Any
"""Opaque alias for a torch.device instance."""

StableDiffusionModel = Any
"""Opaque alias for a loaded Stable Diffusion model (unet, clip, vae bundle)."""


# ---------------------------------------------------------------------------
# Domain model dataclasses — value objects used by the protocols.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Configuration for a single LoRA adapter to apply to a model.

    Attributes:
        filename: Path or name of the LoRA weights file (e.g. 'detail.safetensors').
        weight: Strength multiplier for the LoRA, typically 0.0 to 1.0.

    """

    filename: str
    weight: float


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Configuration for a single sampling pass.

    Attributes:
        sampler_name: Name of the sampling algorithm (e.g. 'euler', 'dpm_2').
        scheduler: Noise schedule name (e.g. 'normal', 'karras').
        steps: Total number of denoising steps.
        cfg_scale: Classifier-free guidance scale.
        seed: Random seed for reproducibility.
        denoise: Denoising strength, 0.0 (no change) to 1.0 (full denoise).
        width: Output image width in pixels.
        height: Output image height in pixels.
    """

    sampler_name: str
    scheduler: str
    steps: int
    cfg_scale: float
    seed: int
    denoise: float
    width: int
    height: int


# ---------------------------------------------------------------------------
# Protocol definitions — ports that adapters must satisfy.
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelLoader(Protocol):
    """Port for loading Stable Diffusion checkpoints and applying LoRA adapters.

    Implementations wrap the concrete checkpoint loading and LoRA patching
    logic (e.g. ldm_patched's load_checkpoint_guess_config and model patcher).

    Errors:
        load_checkpoint may raise FileNotFoundError if the checkpoint path
        does not exist, or ValueError if the file is not a valid checkpoint.
        load_loras may raise FileNotFoundError for missing LoRA files.
    """

    def load_checkpoint(self, path: str) -> StableDiffusionModel:
        """Load a Stable Diffusion checkpoint from disk.

        Args:
            path: Absolute or resolved path to the checkpoint file
                  (.safetensors or .ckpt).

        Returns:
            An opaque model bundle containing unet, clip, vae, and metadata.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            ValueError: If the file cannot be parsed as a valid checkpoint.
        """
        ...

    def load_loras(self, model: StableDiffusionModel, loras: list[LoRAConfig]) -> StableDiffusionModel:
        """Apply one or more LoRA adapters to an already-loaded model.

        Args:
            model: The base model to patch with LoRA weights.
            loras: Ordered list of LoRA configurations to apply.

        Returns:
            A new or mutated model with LoRA weights applied.

        Raises:
            FileNotFoundError: If any LoRA file does not exist.
        """
        ...

    def apply_freeu(
        self,
        model: StableDiffusionModel,
        b1: float,
        b2: float,
        s1: float,
        s2: float,
    ) -> StableDiffusionModel:
        """Apply FreeU parameters to a loaded model.

        FreeU patches the model's U-Net in-place to improve generation quality.

        Args:
            model: The model to apply FreeU parameters to.
            b1: FreeU b1 backbone feature scaling factor.
            b2: FreeU b2 backbone feature scaling factor.
            s1: FreeU s1 skip feature scaling factor.
            s2: FreeU s2 skip feature scaling factor.

        Returns:
            The model with FreeU parameters applied.
        """
        ...


@runtime_checkable
class TextEncoder(Protocol):
    """Port for encoding text prompts into CLIP conditioning tensors.

    Implementations wrap the CLIP tokenization and encoding pipeline,
    including any caching of previously encoded prompts.

    Errors:
        encode may raise RuntimeError if the CLIP model is not loaded.
    """

    def encode(self, texts: list[str], clip_skip: int) -> Conditioning:
        """Encode a list of text prompts into conditioning tensors.

        Args:
            texts: One or more text prompts to encode.
            clip_skip: Number of final CLIP layers to skip (1 = use all layers).

        Returns:
            Conditioning data suitable for passing to a Sampler.

        Raises:
            RuntimeError: If the CLIP model is not loaded or available.
        """
        ...

    def clear_cache(self) -> None:
        """Discard all cached conditioning results.

        Call this after model or LoRA changes to avoid stale encodings.
        """
        ...


@runtime_checkable
class Sampler(Protocol):
    """Port for running the denoising/sampling loop on a latent tensor.

    Implementations wrap the k-sampler logic including noise scheduling,
    optional refiner switching, and progress reporting.

    Errors:
        sample may raise RuntimeError if the model is not loaded or if
        CUDA runs out of memory.
    """

    def sample(
        self,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        latent: LatentTensor,
        config: SamplerConfig,
        callback: ProgressCallback | None,
    ) -> LatentTensor:
        """Run the denoising loop to produce a sampled latent.

        Args:
            model: The model (with LoRAs applied) to sample from.
            positive: Positive (prompt) conditioning.
            negative: Negative (negative prompt) conditioning.
            latent: Initial latent tensor (empty noise or img2img input).
            config: Sampling parameters (steps, cfg, scheduler, etc.).
            callback: Optional progress callback invoked after each step.

        Returns:
            The denoised latent tensor ready for VAE decoding.

        Raises:
            RuntimeError: If the model is unavailable or GPU memory is exhausted.
        """
        ...


@runtime_checkable
class VAEDecoder(Protocol):
    """Port for decoding latent tensors into pixel-space images.

    Implementations wrap the VAE decode step, optionally supporting
    tiled decoding for large images.

    Errors:
        decode may raise RuntimeError if the VAE is not loaded or if
        CUDA runs out of memory.
    """

    def decode(self, vae: StableDiffusionModel, latent: LatentTensor) -> list[NDArray[Any]]:
        """Decode a latent tensor into one or more pixel-space images.

        Args:
            vae: The VAE model to use for decoding.
            latent: The latent tensor to decode.

        Returns:
            A list of numpy arrays, each with shape (H, W, 3) and dtype uint8,
            representing the decoded images.

        Raises:
            RuntimeError: If the VAE is unavailable or GPU memory is exhausted.
        """
        ...


@runtime_checkable
class LatentPreviewer(Protocol):
    """Port for generating fast preview images from partial latents during sampling.

    Implementations wrap a lightweight approximate VAE decoder (e.g. VAEApprox)
    that produces low-quality but fast RGB previews suitable for streaming
    to the client via WebSocket.

    Errors:
        preview may raise RuntimeError if the preview model is not loaded.
    """

    def preview(self, latent: LatentTensor) -> NDArray[Any]:
        """Generate a preview image from a partial latent tensor.

        Args:
            latent: A latent tensor (typically from an in-progress sampling step).

        Returns:
            A numpy array with shape (H, W, 3) and dtype uint8 representing
            the preview image.

        Raises:
            RuntimeError: If the preview model is not loaded.
        """
        ...


@runtime_checkable
class ProgressCallback(Protocol):
    """Port for receiving sampling progress updates.

    Implementations can update a progress bar, send WebSocket messages,
    or generate preview images.
    """

    def __call__(self, step: int, total: int, preview_image: Any | None) -> None:
        """Called after each sampling step completes.

        Args:
            step: The current step number (1-indexed).
            total: The total number of steps.
            preview_image: Optional preview image (numpy array or None).
        """
        ...


@dataclass(frozen=True, slots=True)
class VRAMStats:
    """Value object representing GPU VRAM usage statistics.

    Attributes:
        total_bytes: Total VRAM available on the device.
        used_bytes: VRAM currently in use.
        free_bytes: VRAM currently free.
    """

    total_bytes: int
    used_bytes: int
    free_bytes: int

    def __post_init__(self) -> None:
        if self.total_bytes < 0:
            raise ValueError(f"total_bytes must be non-negative, got {self.total_bytes}")
        if self.used_bytes < 0:
            raise ValueError(f"used_bytes must be non-negative, got {self.used_bytes}")
        if self.free_bytes < 0:
            raise ValueError(f"free_bytes must be non-negative, got {self.free_bytes}")

    @property
    def total_mb(self) -> float:
        """Total VRAM in megabytes."""
        return self.total_bytes / (1024 * 1024)

    @property
    def used_mb(self) -> float:
        """Used VRAM in megabytes."""
        return self.used_bytes / (1024 * 1024)

    @property
    def free_mb(self) -> float:
        """Free VRAM in megabytes."""
        return self.free_bytes / (1024 * 1024)


@runtime_checkable
class ModelManager(Protocol):
    """Port for managing GPU resources and model lifecycle.

    Implementations wrap ldm_patched's model_management module to
    provide device queries, memory cleanup, model loading, and VRAM reporting.

    Errors:
        cleanup / cleanup_models may log warnings if models cannot be freed.
        load_models_to_gpu may raise GPUMemoryError on OOM.
    """

    def get_torch_device(self) -> TorchDevice:
        """Return the primary torch device for inference (e.g. 'cuda:0', 'cpu').

        Returns:
            An opaque device identifier compatible with torch operations.
        """
        ...

    def cleanup_models(self) -> None:
        """Free GPU memory by unloading cached models.

        Call this between generation batches or when memory pressure is high.
        """
        ...

    def cleanup(self) -> None:
        """Full cleanup: unload cached models, empty GPU cache, run GC.

        Broader than cleanup_models — also flushes the CUDA cache and
        triggers Python garbage collection.
        """
        ...

    def load_models_to_gpu(self, models: list[Any]) -> None:
        """Move specified models to GPU VRAM, offloading others if needed.

        Args:
            models: List of model patcher objects to load onto the GPU.

        Raises:
            GPUMemoryError: If GPU runs out of memory during loading.
        """
        ...

    def get_vram_stats(self) -> VRAMStats:
        """Return current VRAM usage statistics.

        Returns:
            A VRAMStats value object with total, used, and free bytes.
        """
        ...

    def should_use_fp16(self) -> bool:
        """Determine if fp16 inference should be used based on GPU capability.

        Returns True for consumer GPUs where fp32 would cause OOM.

        Returns:
            True if fp16 is recommended, False for fp32.
        """
        ...
