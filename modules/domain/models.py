"""Domain models for pipeline configuration and model metadata.

Pure-Python value objects and entities representing the diffusion pipeline's
configuration and model metadata. These form the ubiquitous language for the
generation domain.

No imports from torch, ldm_patched, PIL, or any infrastructure package
are permitted in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.flags import SAMPLER_NAMES, SCHEDULER_NAMES

# ---------------------------------------------------------------------------
# Default weight range for LoRA — sourced from config defaults.
# Kept here to avoid importing config (infrastructure) into domain.
# ---------------------------------------------------------------------------

_DEFAULT_LORA_MIN_WEIGHT: float = -2.0
_DEFAULT_LORA_MAX_WEIGHT: float = 2.0


# ---------------------------------------------------------------------------
# CheckpointMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Metadata for a Stable Diffusion checkpoint file.

    Attributes:
        filename: Basename of the checkpoint file (e.g. 'juggernaut.safetensors').
        file_path: Absolute or resolved path to the checkpoint file.
        is_sdxl: Whether this checkpoint is an SDXL architecture model.
    """

    filename: str
    file_path: str
    is_sdxl: bool

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("filename must not be empty")
        if not self.file_path:
            raise ValueError("file_path must not be empty")


# ---------------------------------------------------------------------------
# LoRAConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Configuration for a single LoRA adapter.

    Attributes:
        filename: Path or name of the LoRA weights file.
        weight: Strength multiplier for the LoRA.
        min_weight: Minimum allowed weight (inclusive). Not included in equality.
        max_weight: Maximum allowed weight (inclusive). Not included in equality.
    """

    filename: str
    weight: float
    min_weight: float = field(default=_DEFAULT_LORA_MIN_WEIGHT, repr=False, compare=False)
    max_weight: float = field(default=_DEFAULT_LORA_MAX_WEIGHT, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("filename must not be empty")
        if not (self.min_weight <= self.weight <= self.max_weight):
            raise ValueError(f"weight {self.weight} outside allowed range [{self.min_weight}, {self.max_weight}]")


# ---------------------------------------------------------------------------
# SamplerConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Configuration for the sampling/denoising pass.

    Attributes:
        sampler_name: Sampling algorithm name (must be in SAMPLER_NAMES).
        scheduler_name: Noise schedule name (must be in SCHEDULER_NAMES).
        steps: Number of denoising steps (must be positive).
        cfg_scale: Classifier-free guidance scale (must be non-negative).
        seed: Random seed for reproducibility.
    """

    sampler_name: str
    scheduler_name: str
    steps: int
    cfg_scale: float
    seed: int

    def __post_init__(self) -> None:
        if self.sampler_name not in SAMPLER_NAMES:
            raise ValueError(f"sampler_name {self.sampler_name!r} not in known samplers: {SAMPLER_NAMES}")
        if self.scheduler_name not in SCHEDULER_NAMES:
            raise ValueError(f"scheduler_name {self.scheduler_name!r} not in known schedulers: {SCHEDULER_NAMES}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.cfg_scale < 0:
            raise ValueError(f"cfg_scale must be non-negative, got {self.cfg_scale}")


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Aggregate configuration for a single generation run.

    Composes CheckpointMetadata, LoRAConfig list, and SamplerConfig
    with image dimensions and optional refiner/FreeU settings.

    Attributes:
        checkpoint: Metadata for the base checkpoint.
        loras: Ordered list of LoRA adapters to apply.
        sampler: Sampling configuration.
        width: Output image width in pixels (must be positive).
        height: Output image height in pixels (must be positive).
        positive_prompt: Text prompt describing desired image content.
        negative_prompt: Text prompt describing undesired content.
        clip_skip: Number of final CLIP layers to skip.
        refiner: Optional refiner checkpoint metadata.
        refiner_switch: Point (0.0-1.0) at which to switch to refiner.
        freeu_enabled: Whether to apply FreeU parameters.
        freeu_b1: FreeU b1 parameter.
        freeu_b2: FreeU b2 parameter.
        freeu_s1: FreeU s1 parameter.
        freeu_s2: FreeU s2 parameter.
    """

    checkpoint: CheckpointMetadata
    loras: list[LoRAConfig]
    sampler: SamplerConfig
    width: int
    height: int
    positive_prompt: str
    negative_prompt: str
    clip_skip: int
    refiner: CheckpointMetadata | None = None
    refiner_switch: float = 0.5
    freeu_enabled: bool = False
    freeu_b1: float = 1.3
    freeu_b2: float = 1.4
    freeu_s1: float = 0.9
    freeu_s2: float = 0.2

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    @classmethod
    def from_task(cls, task: Any) -> PipelineConfig:
        """Build a PipelineConfig from an AsyncTask.

        Extracts all generation-relevant parameters from the task object.
        The task is typed as Any to avoid importing AsyncTask (infrastructure).

        Args:
            task: An AsyncTask instance with parsed generation parameters.

        Returns:
            A frozen PipelineConfig ready for pipeline consumption.
        """
        checkpoint = CheckpointMetadata(
            filename=task.base_model_name,
            file_path=task.base_model_name,
            is_sdxl=True,
        )

        loras = [LoRAConfig(filename=name, weight=weight) for name, weight in task.loras]

        sampler = SamplerConfig(
            sampler_name=task.sampler_name,
            scheduler_name=task.scheduler_name,
            steps=task.effective_steps,
            cfg_scale=task.cfg_scale,
            seed=task.seed,
        )

        refiner = None
        if hasattr(task, "refiner_model_name") and task.refiner_model_name != "None":
            refiner = CheckpointMetadata(
                filename=task.refiner_model_name,
                file_path=task.refiner_model_name,
                is_sdxl=True,
            )

        return cls(
            checkpoint=checkpoint,
            loras=loras,
            sampler=sampler,
            width=task.width,
            height=task.height,
            positive_prompt=task.prompt,
            negative_prompt=task.negative_prompt,
            clip_skip=task.clip_skip,
            refiner=refiner,
            refiner_switch=getattr(task, "refiner_switch", 0.5),
            freeu_enabled=getattr(task, "freeu_enabled", False),
            freeu_b1=getattr(task, "freeu_b1", 1.3),
            freeu_b2=getattr(task, "freeu_b2", 1.4),
            freeu_s1=getattr(task, "freeu_s1", 0.9),
            freeu_s2=getattr(task, "freeu_s2", 0.2),
        )


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of a completed generation run.

    Attributes:
        image_paths: List of absolute paths to generated image files.
        elapsed_time: Total generation time in seconds (must be non-negative).
        seed_used: The seed used for this generation.
    """

    image_paths: list[str]
    elapsed_time: float
    seed_used: int

    def __post_init__(self) -> None:
        if self.elapsed_time < 0:
            raise ValueError(f"elapsed_time must be non-negative, got {self.elapsed_time}")


# ---------------------------------------------------------------------------
# DiffusionProgress
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiffusionProgress:
    """Progress snapshot during diffusion sampling.

    Attributes:
        step: Current step number (0-indexed, 0 means not started).
        total_steps: Total number of steps (must be positive).
        preview_image: Optional preview image (opaque type, no PIL dependency).
    """

    step: int
    total_steps: int
    preview_image: Any = None

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        if self.step > self.total_steps:
            raise ValueError(f"step ({self.step}) must not exceed total_steps ({self.total_steps})")

    @property
    def percentage(self) -> float:
        """Compute progress percentage (0.0 to 100.0)."""
        return (self.step / self.total_steps) * 100.0
