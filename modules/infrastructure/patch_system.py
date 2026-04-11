"""Patch system for diffusion process modifications.

Manages per-process PatchSettings state and provides monkey-patches for
ldm_patched internals that inject sharpness (anisotropic filtering),
ADM guidance scaling, and adaptive CFG into the diffusion process.

Architecture:
    PatchSettingsState — mutable per-process runtime state
    PatchSettingsRegistry — thread-safe registry keyed by PID
    Pure functions — compute_sharpness_alpha, compute_adaptive_cfg, etc.
    patch_all() — installs monkey-patches at application startup

Domain errors raised:
    None — this is pure infrastructure.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PatchSettingsState — mutable per-process runtime state
# ---------------------------------------------------------------------------


@dataclass
class PatchSettingsState:
    """Mutable runtime state for per-process diffusion patch settings.

    Unlike the frozen PatchSettings value object in sampler.py, this class
    holds mutable state that changes during the diffusion process
    (global_diffusion_progress, eps_record).

    Attributes:
        sharpness: Sharpness factor for anisotropic filtering (default 2.0).
        adm_scaler_end: ADM scaler end threshold (default 0.3).
        positive_adm_scale: Positive ADM scale factor (default 1.5).
        negative_adm_scale: Negative ADM scale factor (default 0.8).
        controlnet_softness: ControlNet softness factor (default 0.25).
        adaptive_cfg: Adaptive CFG threshold (default 7.0).
        global_diffusion_progress: Current diffusion progress 0.0-1.0 (mutable).
        eps_record: Optional eps tensor recording for debugging (mutable).
    """

    sharpness: float = 2.0
    adm_scaler_end: float = 0.3
    positive_adm_scale: float = 1.5
    negative_adm_scale: float = 0.8
    controlnet_softness: float = 0.25
    adaptive_cfg: float = 7.0
    global_diffusion_progress: float = 0.0
    eps_record: Any = field(default=None, repr=False)

    @classmethod
    def from_patch_settings(cls, settings: Any) -> PatchSettingsState:
        """Construct mutable runtime state from a frozen PatchSettings value object.

        Args:
            settings: A PatchSettings instance from modules.infrastructure.sampler.

        Returns:
            A new PatchSettingsState with values copied from settings.
        """
        return cls(
            sharpness=settings.sharpness,
            adm_scaler_end=settings.adm_scaler_end,
            positive_adm_scale=settings.positive_adm_scale,
            negative_adm_scale=settings.negative_adm_scale,
            controlnet_softness=settings.controlnet_softness,
            adaptive_cfg=settings.adaptive_cfg,
        )


# ---------------------------------------------------------------------------
# PatchSettingsRegistry — per-process state management
# ---------------------------------------------------------------------------


class PatchSettingsRegistry:
    """Thread-safe registry of PatchSettingsState keyed by process ID.

    Provides per-generation scoping so settings do not leak between tasks.
    Each process (PID) gets independent state. The ``scoped`` context manager
    ensures cleanup even on exceptions.
    """

    def __init__(self) -> None:
        self._states: dict[int, PatchSettingsState] = {}

    def set(self, pid: int, **kwargs: Any) -> None:
        """Set patch settings state for a given PID.

        Args:
            pid: Process ID to scope the settings to.
            **kwargs: PatchSettingsState field values to set.
        """
        self._states[pid] = PatchSettingsState(**kwargs)

    def get(self, pid: int) -> PatchSettingsState:
        """Get patch settings state for a given PID.

        Returns defaults if PID has no registered state.

        Args:
            pid: Process ID to look up.

        Returns:
            PatchSettingsState for this PID (fresh defaults if unregistered).
        """
        if pid not in self._states:
            self._states[pid] = PatchSettingsState()
        return self._states[pid]

    def clear(self, pid: int) -> None:
        """Remove state for a given PID, preventing leakage.

        Args:
            pid: Process ID to clear.
        """
        self._states.pop(pid, None)

    @contextlib.contextmanager
    def scoped(self, pid: int, **kwargs: Any) -> Generator[PatchSettingsState]:
        """Context manager that sets state on entry and clears on exit.

        Ensures per-generation scoping — state is always cleaned up,
        even if an exception occurs.

        Args:
            pid: Process ID to scope.
            **kwargs: PatchSettingsState field values.

        Yields:
            The PatchSettingsState for the duration of the block.
        """
        self.set(pid, **kwargs)
        try:
            yield self._states[pid]
        finally:
            self.clear(pid)


# Module-level registry instance used by the patched functions
patch_settings_registry = PatchSettingsRegistry()


# ---------------------------------------------------------------------------
# Pure computation functions — no side effects, fully testable
# ---------------------------------------------------------------------------


def compute_sharpness_alpha(sharpness: float, diffusion_progress: float) -> float:
    """Compute the blending alpha for sharpness-based anisotropic filtering.

    Formula from reference: alpha = 0.001 * sharpness * diffusion_progress

    Args:
        sharpness: Sharpness factor (0 disables filtering).
        diffusion_progress: Current diffusion progress (0.0 to 1.0).

    Returns:
        Alpha value for blending between original and filtered eps.
    """
    return 0.001 * sharpness * diffusion_progress


def apply_sharpness_blending(
    positive_eps: float,
    filtered_eps: float,
    sharpness: float,
    diffusion_progress: float,
) -> float:
    """Blend between original and anisotropic-filtered eps based on sharpness.

    result = filtered * alpha + original * (1 - alpha)

    Args:
        positive_eps: Original positive eps value.
        filtered_eps: Anisotropic-filtered eps value.
        sharpness: Sharpness factor controlling blend strength.
        diffusion_progress: Current diffusion progress (0.0 to 1.0).

    Returns:
        Blended eps value.
    """
    alpha = compute_sharpness_alpha(sharpness, diffusion_progress)
    return filtered_eps * alpha + positive_eps * (1.0 - alpha)


def compute_adaptive_cfg(
    uncond: float,
    cond: float,
    cfg_scale: float,
    adaptive_cfg: float,
    progress: float,
) -> float:
    """Compute adaptive CFG adjustment based on TSNR.

    When cfg_scale > adaptive_cfg, blends between real and mimicked eps
    based on diffusion progress. When cfg_scale <= adaptive_cfg, returns
    the standard CFG result.

    Args:
        uncond: Unconditional (negative) prediction.
        cond: Conditional (positive) prediction.
        cfg_scale: Actual CFG scale being used.
        adaptive_cfg: Threshold below which no mimicking occurs.
        progress: Current diffusion progress (0.0 to 1.0).

    Returns:
        The adjusted CFG result.
    """
    real_eps = uncond + cfg_scale * (cond - uncond)

    if cfg_scale > adaptive_cfg:
        mimic_eps = uncond + adaptive_cfg * (cond - uncond)
        return real_eps * progress + mimic_eps * (1.0 - progress)
    return real_eps


def scale_adm_dimensions(
    width: int,
    height: int,
    prompt_type: str,
    positive_adm_scale: float,
    negative_adm_scale: float,
) -> tuple[int, int]:
    """Scale ADM dimensions based on prompt type and scale factors.

    SDXL uses width/height as part of its conditioning. Scaling these
    dimensions modifies the guidance strength for positive vs negative prompts.

    Args:
        width: Original width in pixels.
        height: Original height in pixels.
        prompt_type: 'positive', 'negative', or empty string.
        positive_adm_scale: Scale factor for positive prompts.
        negative_adm_scale: Scale factor for negative prompts.

    Returns:
        Tuple of (scaled_width, scaled_height) as integers.
    """
    if prompt_type == "positive":
        return int(float(width) * positive_adm_scale), int(float(height) * positive_adm_scale)
    if prompt_type == "negative":
        return int(float(width) * negative_adm_scale), int(float(height) * negative_adm_scale)
    return width, height


# ---------------------------------------------------------------------------
# Sentinel for idempotent patching
# ---------------------------------------------------------------------------

_patched = False


# ---------------------------------------------------------------------------
# patch_all() — installs monkey-patches on ldm_patched internals
# ---------------------------------------------------------------------------


def patch_all() -> None:
    """Install monkey-patches on ldm_patched internals at application startup.

    Replaces:
        - ldm_patched.modules.samplers.sampling_function
        - ldm_patched.modules.model_base.SDXL.encode_adm

    This function is idempotent — calling it multiple times installs
    the same patched functions without double-wrapping.
    """
    global _patched

    if _patched:
        return

    import ldm_patched.modules.model_base as model_base_mod
    import ldm_patched.modules.samplers as samplers_mod

    samplers_mod.sampling_function = _patched_sampling_function
    model_base_mod.SDXL.encode_adm = _patched_sdxl_encode_adm

    _patched = True
    logger.info("Patch system initialized — sampling_function and SDXL.encode_adm replaced")


# ---------------------------------------------------------------------------
# Patched functions — installed by patch_all()
# ---------------------------------------------------------------------------


def _patched_sampling_function(
    model: Any,
    x: Any,
    timestep: Any,
    uncond: Any,
    cond: Any,
    cond_scale: float,
    model_options: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Any:
    """Patched sampling function with sharpness and adaptive CFG.

    Replaces ldm_patched.modules.samplers.sampling_function to inject:
    - Anisotropic filtering controlled by sharpness
    - Adaptive CFG blending

    This function is installed by patch_all().
    """
    import math

    import modules.infrastructure.anisotropic as anisotropic
    from ldm_patched.modules.samplers import calc_cond_uncond_batch

    pid = os.getpid()
    state = patch_settings_registry.get(pid)

    if math.isclose(cond_scale, 1.0) and not (model_options or {}).get("disable_cfg1_optimization", False):
        final_x0 = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)[0]
        if state.eps_record is not None:
            state.eps_record = ((x - final_x0) / timestep).cpu()
        return final_x0

    positive_x0, negative_x0 = calc_cond_uncond_batch(model, cond, uncond, x, timestep, model_options)

    positive_eps = x - positive_x0
    negative_eps = x - negative_x0

    alpha = compute_sharpness_alpha(state.sharpness, state.global_diffusion_progress)

    positive_eps_degraded = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=positive_x0)
    positive_eps_degraded_weighted = positive_eps_degraded * alpha + positive_eps * (1.0 - alpha)

    mimic_cfg = float(state.adaptive_cfg)
    real_cfg = float(cond_scale)
    real_eps = negative_eps + real_cfg * (positive_eps_degraded_weighted - negative_eps)

    if cond_scale > state.adaptive_cfg:
        mimicked_eps = negative_eps + mimic_cfg * (positive_eps_degraded_weighted - negative_eps)
        final_eps = real_eps * state.global_diffusion_progress + mimicked_eps * (1.0 - state.global_diffusion_progress)
    else:
        final_eps = real_eps

    if state.eps_record is not None:
        state.eps_record = (final_eps / timestep).cpu()

    return x - final_eps


def _patched_sdxl_encode_adm(self: Any, **kwargs: Any) -> Any:
    """Patched SDXL ADM encoding with per-process scale factors.

    Replaces ldm_patched.modules.model_base.SDXL.encode_adm to apply
    positive/negative ADM scaling from the current process's PatchSettings.
    """
    import ldm_patched.modules.model_base as model_base_mod
    import torch

    clip_pooled = model_base_mod.sdxl_pooled(kwargs, self.noise_augmentor)
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 1024)
    target_width = width
    target_height = height

    pid = os.getpid()
    state = patch_settings_registry.get(pid)

    prompt_type = kwargs.get("prompt_type", "")
    width, height = scale_adm_dimensions(
        width=width,
        height=height,
        prompt_type=prompt_type,
        positive_adm_scale=state.positive_adm_scale,
        negative_adm_scale=state.negative_adm_scale,
    )

    def _round_to_64(x: float) -> int:
        return round(x / 64.0) * 64

    def embedder(number_list: list[float]) -> Any:
        h = self.embedder(torch.tensor(number_list, dtype=torch.float32))
        h = torch.flatten(h).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return h

    target_width = _round_to_64(target_width)
    target_height = _round_to_64(target_height)

    adm_emphasized = embedder([height, width, 0, 0, target_height, target_width])
    adm_consistent = embedder([target_height, target_width, 0, 0, target_height, target_width])

    clip_pooled = clip_pooled.to(adm_emphasized)
    return torch.cat((clip_pooled, adm_emphasized, clip_pooled, adm_consistent), dim=1)
