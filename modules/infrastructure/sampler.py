"""LdmSampler — concrete Sampler using ldm_patched ksampler infrastructure.

Bridges the domain Sampler protocol to ldm_patched's ksampler, sigma
calculation, and BrownianTreeNoiseSampler. Handles callback bridging
from ldm_patched's 5-arg step callback to the domain's 3-arg ProgressCallback.

All ldm_patched and torch dependencies are injected via constructor
callables, keeping this module testable with fakes.

Domain errors raised:
    RuntimeError — model unavailable or GPU memory exhausted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modules.domain.protocols import (
    Conditioning,
    LatentTensor,
    ProgressCallback,
    SamplerConfig,
    StableDiffusionModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PatchSettings — value object for diffusion patch parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatchSettings:
    """Configuration for diffusion patch parameters applied before sampling.

    These settings control sharpness, ADM scaling, ControlNet softness,
    and adaptive CFG behavior during the diffusion process. They map to
    the per-process patch_settings dict in FwdFooocus's modules/patch.py.

    Attributes:
        sharpness: Sharpness factor applied during diffusion (default 2.0).
        adm_scaler_end: ADM scaler end threshold (default 0.3).
        positive_adm_scale: Positive ADM scale factor (default 1.5).
        negative_adm_scale: Negative ADM scale factor (default 0.8).
        controlnet_softness: ControlNet softness factor (default 0.25).
        adaptive_cfg: Adaptive CFG threshold (default 7.0).
    """

    sharpness: float = 2.0
    adm_scaler_end: float = 0.3
    positive_adm_scale: float = 1.5
    negative_adm_scale: float = 0.8
    controlnet_softness: float = 0.25
    adaptive_cfg: float = 7.0


# ---------------------------------------------------------------------------
# Type aliases for injected callables
# ---------------------------------------------------------------------------

KSamplerFn = Callable[..., LatentTensor]
"""Signature matches core.ksampler(**kwargs) -> latent dict."""

SigmaCalculatorFn = Callable[..., Any]
"""Signature: (sampler, model, scheduler, steps, denoise) -> sigmas tensor."""

BrownianTreeInitFn = Callable[..., None]
"""Signature: (latent_tensor, sigma_min, sigma_max, seed=, cpu=) -> None."""

PatchSettingsApplierFn = Callable[..., None]
"""Signature: (**patch_settings_fields) -> None."""

GenerateEmptyLatentFn = Callable[[int, int], LatentTensor]
"""Signature: (width, height) -> latent dict."""


# ---------------------------------------------------------------------------
# LdmSampler — concrete Sampler adapter
# ---------------------------------------------------------------------------


class LdmSampler:
    """Concrete Sampler adapter backed by ldm_patched ksampler.

    Dependencies injected via constructor enable unit testing with fakes.

    Args:
        ksampler_fn: Callable wrapping core.ksampler.
        sigma_calculator: Callable wrapping calculate_sigmas.
        brownian_tree_init: Callable wrapping BrownianTreeNoiseSamplerPatched.global_init.
        patch_settings_applier: Callable that applies PatchSettings to the process.
        generate_empty_latent_fn: Callable that generates an empty latent tensor.
    """

    __slots__ = (
        "_brownian_tree_init",
        "_generate_empty_latent_fn",
        "_ksampler_fn",
        "_patch_settings_applier",
        "_sigma_calculator",
    )

    def __init__(
        self,
        *,
        ksampler_fn: KSamplerFn,
        sigma_calculator: SigmaCalculatorFn,
        brownian_tree_init: BrownianTreeInitFn,
        patch_settings_applier: PatchSettingsApplierFn,
        generate_empty_latent_fn: GenerateEmptyLatentFn,
    ) -> None:
        self._ksampler_fn = ksampler_fn
        self._sigma_calculator = sigma_calculator
        self._brownian_tree_init = brownian_tree_init
        self._patch_settings_applier = patch_settings_applier
        self._generate_empty_latent_fn = generate_empty_latent_fn

    def __repr__(self) -> str:
        return "LdmSampler()"

    def sample(
        self,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        latent: LatentTensor | None,
        config: SamplerConfig,
        callback: ProgressCallback | None,
        patch_settings: PatchSettings | None = None,
    ) -> LatentTensor:
        """Run the denoising loop to produce a sampled latent.

        Orchestration order:
        1. Apply PatchSettings (defaults if None)
        2. Generate empty latent if none provided
        3. Calculate sigmas for BrownianTree initialization
        4. Initialize BrownianTreeNoiseSampler
        5. Call ksampler with bridged callback

        Args:
            model: The model (with LoRAs applied) to sample from.
            positive: Positive (prompt) conditioning.
            negative: Negative (negative prompt) conditioning.
            latent: Initial latent tensor, or None to generate empty noise.
            config: Sampling parameters (steps, cfg, scheduler, etc.).
            callback: Optional progress callback invoked after each step.
            patch_settings: Optional diffusion patch parameters.

        Returns:
            The denoised latent tensor ready for VAE decoding.

        Raises:
            RuntimeError: If the model is unavailable or GPU memory is exhausted.
        """
        settings = patch_settings or PatchSettings()
        self._apply_patch_settings(settings)

        initial_latent = latent if latent is not None else self._generate_empty_latent_fn(config.width, config.height)

        sigmas = self._sigma_calculator(
            sampler=config.sampler_name,
            model=model.model,
            scheduler=config.scheduler,
            steps=config.steps,
            denoise=config.denoise,
        )

        sigma_min, sigma_max = _extract_sigma_range(sigmas)

        self._brownian_tree_init(
            initial_latent.get("samples", initial_latent) if isinstance(initial_latent, dict) else initial_latent,
            sigma_min,
            sigma_max,
            seed=config.seed,
            cpu=False,
        )

        bridged_callback = _bridge_callback(callback, config.steps) if callback is not None else None

        return self._ksampler_fn(
            model=model,
            positive=positive,
            negative=negative,
            latent=initial_latent,
            seed=config.seed,
            steps=config.steps,
            cfg=config.cfg_scale,
            sampler_name=config.sampler_name,
            scheduler=config.scheduler,
            denoise=config.denoise,
            callback_function=bridged_callback,
        )

    def _apply_patch_settings(self, settings: PatchSettings) -> None:
        """Forward PatchSettings fields to the patch settings applier."""
        self._patch_settings_applier(
            sharpness=settings.sharpness,
            adm_scaler_end=settings.adm_scaler_end,
            positive_adm_scale=settings.positive_adm_scale,
            negative_adm_scale=settings.negative_adm_scale,
            controlnet_softness=settings.controlnet_softness,
            adaptive_cfg=settings.adaptive_cfg,
        )


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _extract_sigma_range(sigmas: Any) -> tuple[float, float]:
    """Extract (sigma_min, sigma_max) from a sigmas sequence.

    Filters to positive values for sigma_min, takes absolute max for sigma_max.

    Args:
        sigmas: Sigma values (torch.Tensor or FakeSigmas).

    Returns:
        Tuple of (sigma_min, sigma_max) as floats.
    """
    positive_mask = sigmas > 0
    if hasattr(sigmas, "values"):
        # Fake sigmas path (unit tests)
        positive_values = [v for v, m in zip(sigmas.values, positive_mask, strict=False) if m]
        sigma_min = min(positive_values) if positive_values else 0.0
        sigma_max = sigmas.max()
    else:
        # Real torch tensor path
        positive_sigmas = sigmas[positive_mask]
        sigma_min = float(positive_sigmas.min().cpu().numpy())
        sigma_max = float(sigmas.max().cpu().numpy())
    return sigma_min, sigma_max


def _bridge_callback(
    domain_callback: ProgressCallback,
    total_steps: int,
) -> Callable[..., None]:
    """Bridge ldm_patched's 5-arg step callback to domain's 3-arg ProgressCallback.

    ldm_patched callback signature: (step, x0, x, total_steps, preview_image)
    Domain ProgressCallback signature: (step, total, preview_image)

    The bridge converts 0-indexed steps to 1-indexed.

    Args:
        domain_callback: The domain-level progress callback.
        total_steps: Total number of sampling steps.

    Returns:
        A callable matching ldm_patched's callback signature.
    """

    def bridged(step: int, _x0: Any, _x: Any, _total: int, preview_image: Any) -> None:
        domain_callback(step + 1, total_steps, preview_image)

    return bridged
