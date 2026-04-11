"""DiffusionPipeline service — orchestrates model load through image decode.

Coordinates the full SDXL generation pipeline: checkpoint loading, LoRA
application, CLIP text encoding, diffusion sampling, and VAE decoding.
Contains zero torch code — only orchestration logic with injected dependencies.

No imports from torch or ldm_patched are permitted in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modules.domain.protocols import (
    LoRAConfig,
    SamplerConfig,
)
from numpy.typing import NDArray  # noqa: TC002 — needed at runtime for dataclass field

if TYPE_CHECKING:
    from modules.domain.protocols import (
        ModelLoader,
        Sampler,
        StableDiffusionModel,
        TextEncoder,
        VAEDecoder,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable configuration for a single generation run.

    Attributes:
        checkpoint_path: Path to the base SDXL checkpoint file.
        loras: Ordered list of LoRA adapters to apply.
        positive_prompt: Text prompt describing desired image content.
        negative_prompt: Text prompt describing undesired content.
        sampler_name: Sampling algorithm name (e.g. 'euler', 'dpmpp_2m_sde_gpu').
        scheduler: Noise schedule name (e.g. 'normal', 'karras').
        steps: Number of denoising steps.
        cfg_scale: Classifier-free guidance scale.
        seed: Base random seed for reproducibility.
        denoise: Denoising strength (0.0 to 1.0).
        image_number: Number of images to generate in this batch.
        clip_skip: Number of final CLIP layers to skip.
        width: Output image width in pixels.
        height: Output image height in pixels.
        disable_seed_increment: If True, use the same seed for all images.
        freeu_enabled: Whether to apply FreeU parameters.
        freeu_b1: FreeU b1 parameter.
        freeu_b2: FreeU b2 parameter.
        freeu_s1: FreeU s1 parameter.
        freeu_s2: FreeU s2 parameter.
    """

    checkpoint_path: str
    loras: list[LoRAConfig]
    positive_prompt: str
    negative_prompt: str
    sampler_name: str
    scheduler: str
    steps: int
    cfg_scale: float
    seed: int
    denoise: float
    image_number: int
    clip_skip: int
    width: int
    height: int
    disable_seed_increment: bool
    freeu_enabled: bool
    freeu_b1: float
    freeu_b2: float
    freeu_s1: float
    freeu_s2: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of generating a single image.

    Attributes:
        image: Decoded pixel-space image as numpy array (H, W, 3), dtype uint8.
        seed: The seed used for this specific image.
    """

    image: NDArray[Any]
    seed: int


# ---------------------------------------------------------------------------
# Cache key — tracks what model state is currently loaded
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ModelCacheKey:
    """Tracks loaded model identity for cache invalidation.

    Includes FreeU state because FreeU patches the model in-place —
    toggling it requires a fresh model reload.
    """

    checkpoint_path: str
    loras: tuple[LoRAConfig, ...]
    freeu_enabled: bool
    freeu_b1: float
    freeu_b2: float
    freeu_s1: float
    freeu_s2: float


# ---------------------------------------------------------------------------
# Progress callback type alias
# ---------------------------------------------------------------------------

ProgressCallbackFn = Callable[[int, int, int, Any], None]
"""Signature: (image_index, step, total, preview_image) -> None."""

CancelCheckFn = Callable[[], bool]
"""Returns True when generation should stop."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DiffusionPipeline:
    """Orchestrates the full SDXL generation pipeline.

    Accepts protocol-typed dependencies via constructor (Dependency Inversion).
    Contains no torch code — only coordination logic.

    Args:
        model_loader: Loads checkpoints and applies LoRAs.
        text_encoder: Encodes text prompts into CLIP conditioning.
        sampler: Runs the denoising/sampling loop.
        vae_decoder: Decodes latent tensors into pixel-space images.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        text_encoder: TextEncoder,
        sampler: Sampler,
        vae_decoder: VAEDecoder,
    ) -> None:
        self._model_loader = model_loader
        self._text_encoder = text_encoder
        self._sampler = sampler
        self._vae_decoder = vae_decoder

        self._cached_key: _ModelCacheKey | None = None
        self._cached_model: StableDiffusionModel | None = None

    def __repr__(self) -> str:
        cached = self._cached_key.checkpoint_path if self._cached_key else "none"
        return f"DiffusionPipeline(cached_checkpoint={cached!r})"

    def generate(
        self,
        config: PipelineConfig,
        progress_callback: ProgressCallbackFn | None = None,
        cancel_check: CancelCheckFn | None = None,
    ) -> list[GenerationResult]:
        """Run the full generation pipeline for the given config.

        Orchestration order:
        1. Load checkpoint (cached if unchanged)
        2. Apply LoRAs (cached if unchanged)
        3. Apply FreeU (if enabled)
        4. Encode positive and negative prompts via CLIP
        5. For each image: sample latent, decode via VAE

        Args:
            config: Generation parameters.
            progress_callback: Optional callback invoked per sampling step.
                Signature: (image_index, step, total, preview_image) -> None.
            cancel_check: Optional callable returning True to stop generation.

        Returns:
            List of GenerationResult, one per successfully generated image.
        """
        model = self._load_model(config)
        positive = self._text_encoder.encode([config.positive_prompt], config.clip_skip)
        negative = self._text_encoder.encode([config.negative_prompt], config.clip_skip)

        results: list[GenerationResult] = []
        for image_index in range(config.image_number):
            if cancel_check is not None and cancel_check():
                break

            seed = self._compute_seed(config, image_index)
            sampler_config = SamplerConfig(
                sampler_name=config.sampler_name,
                scheduler=config.scheduler,
                steps=config.steps,
                cfg_scale=config.cfg_scale,
                seed=seed,
                denoise=config.denoise,
                width=config.width,
                height=config.height,
            )

            step_callback = None
            if progress_callback is not None:
                step_callback = _make_step_callback(progress_callback, image_index)

            latent = self._sampler.sample(
                model=model,
                positive=positive,
                negative=negative,
                latent=None,
                config=sampler_config,
                callback=step_callback,
            )

            images = self._vae_decoder.decode(vae=model, latent=latent)
            for image in images:
                results.append(GenerationResult(image=image, seed=seed))

        return results

    def _load_model(self, config: PipelineConfig) -> StableDiffusionModel:
        """Load or return cached model based on checkpoint + LoRA + FreeU identity."""
        cache_key = _ModelCacheKey(
            checkpoint_path=config.checkpoint_path,
            loras=tuple(config.loras),
            freeu_enabled=config.freeu_enabled,
            freeu_b1=config.freeu_b1,
            freeu_b2=config.freeu_b2,
            freeu_s1=config.freeu_s1,
            freeu_s2=config.freeu_s2,
        )

        if cache_key == self._cached_key and self._cached_model is not None:
            return self._cached_model

        model = self._model_loader.load_checkpoint(config.checkpoint_path)
        self._text_encoder.clear_cache()

        if config.loras:
            model = self._model_loader.load_loras(model, config.loras)

        if config.freeu_enabled:
            model = self._model_loader.apply_freeu(
                model, config.freeu_b1, config.freeu_b2, config.freeu_s1, config.freeu_s2
            )

        self._cached_key = cache_key
        self._cached_model = model
        return model

    @staticmethod
    def _compute_seed(config: PipelineConfig, image_index: int) -> int:
        """Compute seed for a specific image in the batch."""
        if config.disable_seed_increment:
            return config.seed
        return config.seed + image_index


def _make_step_callback(
    progress_callback: ProgressCallbackFn,
    image_index: int,
) -> Callable[..., None]:
    """Create a per-step callback that includes the image index."""

    def callback(step: int, total: int, preview_image: Any) -> None:
        progress_callback(image_index, step, total, preview_image)

    return callback
