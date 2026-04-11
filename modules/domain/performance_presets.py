"""Performance preset overrides for speed-optimized generation modes.

Defines the domain model for performance presets (LCM, Lightning, Hyper-SD)
that override sampler, scheduler, CFG, and LoRA settings to enable
fast generation with 4-8 steps instead of 30.

Each speed preset disables the refiner, zeroes sharpness, and applies
a specialized LoRA file tuned for few-step generation.

No imports from torch, ldm_patched, or any infrastructure package
are permitted in this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from modules.domain.exceptions import ModelNotFoundError
from modules.flags import Performance


class SchedulerPatchType(Enum):
    """Type of scheduler patching required for a performance preset.

    MODEL_SAMPLING_DISCRETE: Used by LCM/TCD and SGM-uniform schedulers.
        Patches the model's sampling to use discrete noise scheduling.
    CONTINUOUS_EDM: Used by Karras/EDM schedulers.
        Patches the model's sampling to use continuous EDM noise scheduling.
    """

    MODEL_SAMPLING_DISCRETE = "model_sampling_discrete"
    CONTINUOUS_EDM = "continuous_edm"


@dataclass(frozen=True, slots=True)
class PresetOverrides:
    """Immutable value object containing all overrides for a speed preset.

    Attributes:
        sampler_name: Sampling algorithm override.
        scheduler_name: Noise schedule override.
        steps: Denoising step count override.
        cfg_scale: Classifier-free guidance scale override.
        lora_filename: LoRA weights file to apply.
        lora_weight: Strength multiplier for the preset LoRA.
        refiner_disabled: Whether the refiner must be disabled.
        sharpness: Sharpness override (0.0 for speed presets).
        adaptive_cfg: Adaptive CFG override.
        adm_scaler_positive: ADM positive scaler override.
        adm_scaler_negative: ADM negative scaler override.
        adm_scaler_end: ADM end scaler override.
        scheduler_patch_type: Type of scheduler patching required.
    """

    sampler_name: str
    scheduler_name: str
    steps: int
    cfg_scale: float
    lora_filename: str
    lora_weight: float
    refiner_disabled: bool
    sharpness: float
    adaptive_cfg: float
    adm_scaler_positive: float
    adm_scaler_negative: float
    adm_scaler_end: float
    scheduler_patch_type: SchedulerPatchType


# ---------------------------------------------------------------------------
# Preset definitions — one per speed mode
# ---------------------------------------------------------------------------

_EXTREME_SPEED_OVERRIDES = PresetOverrides(
    sampler_name="lcm",
    scheduler_name="lcm",
    steps=8,
    cfg_scale=1.0,
    lora_filename="sdxl_lcm_lora.safetensors",
    lora_weight=1.0,
    refiner_disabled=True,
    sharpness=0.0,
    adaptive_cfg=1.0,
    adm_scaler_positive=1.0,
    adm_scaler_negative=1.0,
    adm_scaler_end=0.0,
    scheduler_patch_type=SchedulerPatchType.MODEL_SAMPLING_DISCRETE,
)

_LIGHTNING_OVERRIDES = PresetOverrides(
    sampler_name="euler",
    scheduler_name="sgm_uniform",
    steps=4,
    cfg_scale=1.0,
    lora_filename="sdxl_lightning_4step_lora.safetensors",
    lora_weight=1.0,
    refiner_disabled=True,
    sharpness=0.0,
    adaptive_cfg=1.0,
    adm_scaler_positive=1.0,
    adm_scaler_negative=1.0,
    adm_scaler_end=0.0,
    scheduler_patch_type=SchedulerPatchType.MODEL_SAMPLING_DISCRETE,
)

_HYPER_SD_OVERRIDES = PresetOverrides(
    sampler_name="dpmpp_sde_gpu",
    scheduler_name="karras",
    steps=4,
    cfg_scale=1.0,
    lora_filename="sdxl_hyper_sd_4step_lora.safetensors",
    lora_weight=0.8,
    refiner_disabled=True,
    sharpness=0.0,
    adaptive_cfg=1.0,
    adm_scaler_positive=1.0,
    adm_scaler_negative=1.0,
    adm_scaler_end=0.0,
    scheduler_patch_type=SchedulerPatchType.CONTINUOUS_EDM,
)

_PRESET_MAP: dict[Performance, PresetOverrides] = {
    Performance.EXTREME_SPEED: _EXTREME_SPEED_OVERRIDES,
    Performance.LIGHTNING: _LIGHTNING_OVERRIDES,
    Performance.HYPER_SD: _HYPER_SD_OVERRIDES,
}


def get_preset_overrides(preset: Performance) -> PresetOverrides | None:
    """Return the overrides for a performance preset, or None if no overrides apply.

    QUALITY and SPEED presets have no overrides — they use the user's
    configured sampler/scheduler/steps directly.

    Args:
        preset: The performance preset to look up.

    Returns:
        PresetOverrides for speed presets, None for QUALITY/SPEED.
    """
    return _PRESET_MAP.get(preset)


def validate_preset_lora(preset: Performance, lora_directory: str) -> None:
    """Validate that the LoRA file required by a preset exists on disk.

    No-op for presets that don't require a LoRA (QUALITY, SPEED).

    Args:
        preset: The performance preset to validate.
        lora_directory: Absolute path to the directory containing LoRA files.

    Raises:
        ModelNotFoundError: If the preset requires a LoRA file that does not
            exist in lora_directory.
    """
    overrides = get_preset_overrides(preset)
    if overrides is None:
        return

    lora_path = os.path.join(lora_directory, overrides.lora_filename)
    if not os.path.isfile(lora_path):
        raise ModelNotFoundError(
            f"Performance preset {preset.value} requires LoRA file "
            f"'{overrides.lora_filename}' but it was not found at {lora_path}"
        )
