"""Service layer for applying performance preset overrides to pipeline config.

Thin orchestration: looks up domain-defined preset overrides and applies
them to a PipelineConfig, producing a new config with speed-optimized
settings. Contains no domain logic — delegates to the domain module.

No imports from torch or ldm_patched are permitted in this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from modules.domain.performance_presets import get_preset_overrides
from modules.domain.protocols import LoRAConfig
from modules.services.diffusion_pipeline import PipelineConfig

if TYPE_CHECKING:
    from modules.flags import Performance

logger = logging.getLogger(__name__)


def apply_performance_preset(
    config: PipelineConfig,
    preset: Performance,
) -> PipelineConfig:
    """Apply performance preset overrides to a pipeline configuration.

    For QUALITY and SPEED presets, returns the original config unchanged.
    For speed presets (EXTREME_SPEED, LIGHTNING, HYPER_SD), returns a new
    PipelineConfig with sampler, scheduler, steps, CFG, refiner, and LoRA
    settings overridden according to the preset.

    The preset LoRA is appended to the existing LoRA list so user-selected
    LoRAs are preserved.

    Args:
        config: The base pipeline configuration to override.
        preset: The performance preset to apply.

    Returns:
        Original config if no overrides, otherwise a new PipelineConfig
        with preset overrides applied.
    """
    overrides = get_preset_overrides(preset)
    if overrides is None:
        return config

    if overrides.refiner_disabled and config.has_refiner:
        logger.info("Refiner disabled by %s preset", preset.value)

    preset_lora = LoRAConfig(
        filename=overrides.lora_filename,
        weight=overrides.lora_weight,
    )
    merged_loras = [*config.loras, preset_lora]

    return PipelineConfig(
        checkpoint_path=config.checkpoint_path,
        loras=merged_loras,
        positive_prompt=config.positive_prompt,
        negative_prompt=config.negative_prompt,
        sampler_name=overrides.sampler_name,
        scheduler=overrides.scheduler_name,
        steps=overrides.steps,
        cfg_scale=overrides.cfg_scale,
        seed=config.seed,
        denoise=config.denoise,
        image_number=config.image_number,
        clip_skip=config.clip_skip,
        width=config.width,
        height=config.height,
        disable_seed_increment=config.disable_seed_increment,
        freeu_enabled=config.freeu_enabled,
        freeu_b1=config.freeu_b1,
        freeu_b2=config.freeu_b2,
        freeu_s1=config.freeu_s1,
        freeu_s2=config.freeu_s2,
        refiner_path=None,
        refiner_swap_method=config.refiner_swap_method,
        refiner_switch=1.0,
    )
