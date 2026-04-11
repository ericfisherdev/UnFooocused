"""DiffusionPipeline service — orchestrates model load through image decode.

Coordinates the full SDXL generation pipeline: checkpoint loading, LoRA
application, CLIP text encoding, diffusion sampling, and VAE decoding.
Supports optional refiner model handoff via three swap methods (joint,
separate, vae).

Contains zero torch code — only orchestration logic with injected dependencies.

No imports from torch or ldm_patched are permitted in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from modules.domain.protocols import (
    LoRAConfig,
    SamplerConfig,
)
from numpy.typing import NDArray  # noqa: TC002 — needed at runtime for dataclass field

if TYPE_CHECKING:
    from modules.domain.protocols import (
        Conditioning,
        LatentTensor,
        ModelLoader,
        Sampler,
        StableDiffusionModel,
        TextEncoder,
        VAEDecoder,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RefinerSwapMethod enum
# ---------------------------------------------------------------------------


class RefinerSwapMethod(Enum):
    """Strategy for switching from base model to refiner model.

    JOINT: Single ksampler call with refiner parameter and switch step.
    SEPARATE: Two sequential ksampler calls with noise preservation.
    VAE: Base samples, VAE interpose decode/reencode, refiner continues.
    """

    JOINT = "joint"
    SEPARATE = "separate"
    VAE = "vae"


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
        refiner_path: Path to the refiner checkpoint, or None/empty/'None' for no refiner.
        refiner_swap_method: Swap method string ('joint', 'separate', 'vae').
        refiner_switch: Fraction (0.0-1.0) of steps handled by the base model.
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
    refiner_path: str | None = None
    refiner_swap_method: str = "joint"
    refiner_switch: float = 0.5

    @property
    def has_refiner(self) -> bool:
        """Return True if this config specifies a real refiner checkpoint."""
        path = self.refiner_path
        if path is None:
            return False
        stripped = path.strip()
        return stripped != "" and stripped != "None"

    @classmethod
    def from_task(cls, task: Any) -> PipelineConfig:
        """Build a PipelineConfig from an AsyncTask.

        Extracts all generation parameters from the task. Sets image_number=1
        because the Worker iterates over images externally, calling the pipeline
        once per image.

        Args:
            task: An AsyncTask instance with parsed generation parameters.

        Returns:
            A frozen PipelineConfig ready for DiffusionPipeline.generate().
        """
        refiner_model_name = getattr(task, "refiner_model_name", None)
        refiner_path: str | None = None
        if isinstance(refiner_model_name, str) and refiner_model_name.strip() and refiner_model_name != "None":
            refiner_path = refiner_model_name

        return cls(
            checkpoint_path=task.base_model_name,
            loras=[LoRAConfig(filename=name, weight=weight) for name, weight in task.loras],
            positive_prompt=task.prompt,
            negative_prompt=task.negative_prompt,
            sampler_name=task.sampler_name,
            scheduler=task.scheduler_name,
            steps=task.effective_steps,
            cfg_scale=task.cfg_scale,
            seed=task.seed,
            denoise=1.0,
            image_number=1,
            clip_skip=task.clip_skip,
            width=task.width,
            height=task.height,
            disable_seed_increment=task.disable_seed_increment,
            freeu_enabled=task.freeu_enabled,
            freeu_b1=task.freeu_b1,
            freeu_b2=task.freeu_b2,
            freeu_s1=task.freeu_s1,
            freeu_s2=task.freeu_s2,
            refiner_path=refiner_path,
            refiner_swap_method=getattr(task, "refiner_swap_method", "joint"),
            refiner_switch=getattr(task, "refiner_switch", 0.5),
        )

    def with_seed(self, seed: int) -> PipelineConfig:
        """Return a copy with a different seed and disable_seed_increment=True.

        Used by the Worker to create per-image configs from a base config
        without reconstructing all fields manually.

        Args:
            seed: The seed to use for this specific image.

        Returns:
            A new frozen PipelineConfig identical to self except for seed
            and disable_seed_increment.
        """
        # dataclasses.replace not used because frozen + slots + list field
        # causes issues in some Python versions; explicit construction is safe.
        return PipelineConfig(
            checkpoint_path=self.checkpoint_path,
            loras=self.loras,
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            sampler_name=self.sampler_name,
            scheduler=self.scheduler,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            seed=seed,
            denoise=self.denoise,
            image_number=1,
            clip_skip=self.clip_skip,
            width=self.width,
            height=self.height,
            disable_seed_increment=True,
            freeu_enabled=self.freeu_enabled,
            freeu_b1=self.freeu_b1,
            freeu_b2=self.freeu_b2,
            freeu_s1=self.freeu_s1,
            freeu_s2=self.freeu_s2,
            refiner_path=self.refiner_path,
            refiner_swap_method=self.refiner_swap_method,
            refiner_switch=self.refiner_switch,
        )


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
# Pure helper functions
# ---------------------------------------------------------------------------


def compute_switch_step(refiner_switch: float, total_steps: int) -> int:
    """Compute the step at which the base model hands off to the refiner.

    Args:
        refiner_switch: Fraction (0.0-1.0) of total steps handled by the base.
        total_steps: Total number of denoising steps.

    Returns:
        Integer step number where the base model stops and refiner begins.
    """
    if not 0.0 <= refiner_switch <= 1.0:
        raise ValueError(f"refiner_switch must be between 0.0 and 1.0, got {refiner_switch}")
    return round(refiner_switch * total_steps)


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
# Callable type aliases for injected dependencies
# ---------------------------------------------------------------------------

ClipSeparateFn = Callable[..., Any]
"""Signature: (cond, target_model, target_clip) -> separated_cond."""

VAEInterposeFn = Callable[..., Any]
"""Signature: (latent) -> interposed_latent."""

ProgressCallbackFn = Callable[[int, int, int, Any], None]
"""Signature: (image_index, step, total, preview_image) -> None."""

CancelCheckFn = Callable[[], bool]
"""Returns True when generation should stop."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DiffusionPipeline:
    """Orchestrates the full SDXL generation pipeline with optional refiner.

    Accepts protocol-typed dependencies via constructor (Dependency Inversion).
    Contains no torch code — only coordination logic.

    Args:
        model_loader: Loads checkpoints and applies LoRAs.
        text_encoder: Encodes text prompts into CLIP conditioning.
        sampler: Runs the denoising/sampling loop.
        vae_decoder: Decodes latent tensors into pixel-space images.
        clip_separate: Separates CLIP conditioning for refiner UNet.
        vae_interpose: Transforms latent between base and refiner VAE spaces.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        text_encoder: TextEncoder,
        sampler: Sampler,
        vae_decoder: VAEDecoder,
        clip_separate: ClipSeparateFn | None = None,
        vae_interpose: VAEInterposeFn | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._text_encoder = text_encoder
        self._sampler = sampler
        self._vae_decoder = vae_decoder
        self._clip_separate = clip_separate
        self._vae_interpose = vae_interpose

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
        1. Load base checkpoint (cached if unchanged)
        2. Apply LoRAs (cached if unchanged)
        3. Apply FreeU (if enabled)
        4. Optionally load refiner checkpoint
        5. Encode positive and negative prompts via CLIP
        6. For each image: sample latent (with optional refiner handoff), decode via VAE

        Args:
            config: Generation parameters.
            progress_callback: Optional callback invoked per sampling step.
                Signature: (image_index, step, total, preview_image) -> None.
            cancel_check: Optional callable returning True to stop generation.

        Returns:
            List of GenerationResult, one per successfully generated image.
        """
        model = self._load_model(config)
        refiner_model = self._load_refiner(config)
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

            step_callback = _make_step_callback(progress_callback, image_index) if progress_callback else None

            latent = self._sample_with_refiner(
                model=model,
                refiner_model=refiner_model,
                positive=positive,
                negative=negative,
                swap_method=config.refiner_swap_method,
                sampler_config=sampler_config,
                refiner_switch=config.refiner_switch,
                callback=step_callback,
            )

            images = self._vae_decoder.decode(vae=model, latent=latent)
            for image in images:
                results.append(GenerationResult(image=image, seed=seed))

        return results

    # ------------------------------------------------------------------
    # Refiner dispatch
    # ------------------------------------------------------------------

    def _sample_with_refiner(
        self,
        model: StableDiffusionModel,
        refiner_model: StableDiffusionModel | None,
        positive: Conditioning,
        negative: Conditioning,
        swap_method: str,
        sampler_config: SamplerConfig,
        refiner_switch: float,
        callback: Any,
    ) -> LatentTensor:
        """Dispatch to the appropriate swap method or base-only sampling."""
        if refiner_model is None:
            return self._sample_base_only(model, positive, negative, sampler_config, callback)

        method = RefinerSwapMethod(swap_method)
        switch_step = compute_switch_step(refiner_switch, sampler_config.steps)

        # All three modes use two-pass scheduling with the computed switch step.
        # 'joint' and 'separate' pass the latent directly; 'vae' interposes via VAE.
        interpose_fn = self._vae_interpose if method is RefinerSwapMethod.VAE else None
        return self._sample_two_pass(
            base_model=model,
            refiner_model=refiner_model,
            positive=positive,
            negative=negative,
            sampler_config=sampler_config,
            switch_step=switch_step,
            callback=callback,
            latent_transform=interpose_fn,
        )

    def _sample_base_only(
        self,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        sampler_config: SamplerConfig,
        callback: Any,
    ) -> LatentTensor:
        """Single-pass sampling with the base model only."""
        return self._sampler.sample(
            model=model,
            positive=positive,
            negative=negative,
            latent=None,
            config=sampler_config,
            callback=callback,
        )

    def _sample_two_pass(
        self,
        base_model: StableDiffusionModel,
        refiner_model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        sampler_config: SamplerConfig,
        switch_step: int,
        callback: Any,
        latent_transform: VAEInterposeFn | None,
    ) -> LatentTensor:
        """Two-pass sampling: base model then refiner model.

        Used by both 'separate' and 'vae' swap methods. The only difference
        is whether a latent_transform (VAE interpose) is applied between passes.

        The base model runs for switch_step steps, then the refiner continues
        for the remaining steps.

        Args:
            base_model: Model for the first sampling pass.
            refiner_model: Model for the second sampling pass.
            positive: Positive CLIP conditioning.
            negative: Negative CLIP conditioning.
            sampler_config: Sampling parameters.
            switch_step: Step at which base hands off to refiner.
            callback: Progress callback or None.
            latent_transform: Optional transform applied to base output before refiner.
        """
        refiner_steps = sampler_config.steps - switch_step

        # Boundary: switch_step >= total means base handles everything
        if refiner_steps <= 0:
            return self._sampler.sample(
                model=base_model,
                positive=positive,
                negative=negative,
                latent=None,
                config=sampler_config,
                callback=callback,
            )

        # Boundary: switch_step <= 0 means refiner handles everything
        if switch_step <= 0:
            ref_positive = self._separate_clip(positive, refiner_model)
            ref_negative = self._separate_clip(negative, refiner_model)
            return self._sampler.sample(
                model=refiner_model,
                positive=ref_positive,
                negative=ref_negative,
                latent=None,
                config=sampler_config,
                callback=callback,
            )

        # Base model pass — runs for switch_step steps
        base_config = SamplerConfig(
            sampler_name=sampler_config.sampler_name,
            scheduler=sampler_config.scheduler,
            steps=switch_step,
            cfg_scale=sampler_config.cfg_scale,
            seed=sampler_config.seed,
            denoise=sampler_config.denoise,
            width=sampler_config.width,
            height=sampler_config.height,
        )
        base_latent = self._sampler.sample(
            model=base_model,
            positive=positive,
            negative=negative,
            latent=None,
            config=base_config,
            callback=callback,
        )

        # Optional latent transform (VAE interpose for 'vae' mode, None for 'separate')
        refiner_latent = latent_transform(base_latent) if latent_transform is not None else base_latent

        # Separate CLIP conditioning for refiner
        ref_positive = self._separate_clip(positive, refiner_model)
        ref_negative = self._separate_clip(negative, refiner_model)

        # Refiner pass — runs for remaining steps
        refiner_config = SamplerConfig(
            sampler_name=sampler_config.sampler_name,
            scheduler=sampler_config.scheduler,
            steps=refiner_steps,
            cfg_scale=sampler_config.cfg_scale,
            seed=sampler_config.seed,
            denoise=sampler_config.denoise,
            width=sampler_config.width,
            height=sampler_config.height,
        )
        return self._sampler.sample(
            model=refiner_model,
            positive=ref_positive,
            negative=ref_negative,
            latent=refiner_latent,
            config=refiner_config,
            callback=callback,
        )

    def _separate_clip(
        self,
        cond: Conditioning,
        refiner_model: StableDiffusionModel,
    ) -> Conditioning:
        """Separate CLIP conditioning for the refiner model."""
        if self._clip_separate is not None:
            return self._clip_separate(
                cond=cond,
                target_model=refiner_model,
                target_clip=self._cached_model,
            )
        return cond

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

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

        # Update the text encoder's CLIP reference after LoRA application
        # so that LoRA-patched CLIP weights are used for text encoding
        if hasattr(model, "clip_with_lora") and hasattr(self._text_encoder, "set_clip"):
            self._text_encoder.set_clip(model.clip_with_lora)

        if config.freeu_enabled:
            try:
                model = self._model_loader.apply_freeu(
                    model, config.freeu_b1, config.freeu_b2, config.freeu_s1, config.freeu_s2
                )
            except NotImplementedError:
                logger.warning("FreeU requested but model loader does not support it — skipping")

        self._cached_key = cache_key
        self._cached_model = model
        return model

    def _load_refiner(self, config: PipelineConfig) -> StableDiffusionModel | None:
        """Load refiner checkpoint if configured, else return None.

        Reuses the already-loaded base model when refiner_path matches
        checkpoint_path (synthetic refiner) to avoid duplicate VRAM usage.
        """
        if not config.has_refiner:
            return None

        assert config.refiner_path is not None  # guaranteed by has_refiner

        # Synthetic refiner: reuse the cached base model
        if config.refiner_path == config.checkpoint_path and self._cached_model is not None:
            return self._cached_model

        return self._model_loader.load_checkpoint(config.refiner_path)

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
