"""Standalone flags and enums for UnFooocused.

Owns all enum, constant, and sentinel value definitions used during
image generation. This module has no external dependencies.

Domain concepts:
  - OutputFormat: image file format (png, jpeg, webp)
  - Performance: generation quality/speed presets with associated step counts
  - MetadataScheme: metadata embedding format
  - Sampler/scheduler lists: algorithm names for the diffusion process
  - SDXL aspect ratios: standard resolution pairs for SDXL models
  - Sentinel values: placeholder strings for disabled/enabled states
"""

from __future__ import annotations

from enum import Enum, IntEnum

# ---------------------------------------------------------------------------
# Sentinel values
# ---------------------------------------------------------------------------

disabled: str = "Disabled"
enabled: str = "Enabled"
subtle_variation: str = "Vary (Subtle)"
strong_variation: str = "Vary (Strong)"
upscale_15: str = "Upscale (1.5x)"
upscale_2: str = "Upscale (2x)"
upscale_fast: str = "Upscale (Fast 2x)"

uov_list: list[str] = [disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast]

# ---------------------------------------------------------------------------
# Enhancement processing order
# ---------------------------------------------------------------------------

enhancement_uov_before: str = "Before First Enhancement"
enhancement_uov_after: str = "After Last Enhancement"
enhancement_uov_processing_order: list[str] = [enhancement_uov_before, enhancement_uov_after]

enhancement_uov_prompt_type_original: str = "Original Prompts"
enhancement_uov_prompt_type_last_filled: str = "Last Filled Enhancement Prompts"
enhancement_uov_prompt_types: list[str] = [
    enhancement_uov_prompt_type_original,
    enhancement_uov_prompt_type_last_filled,
]

# ---------------------------------------------------------------------------
# Sampler definitions
# ---------------------------------------------------------------------------

CIVITAI_NO_KARRAS: list[str] = [
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_fast",
    "dpm_adaptive",
    "ddim",
    "uni_pc",
]

# fooocus: a1111 (Civitai) name mapping
KSAMPLER: dict[str, str] = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM",
    "tcd": "TCD",
    "restart": "Restart",
}

SAMPLER_EXTRA: dict[str, str] = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": "",
}

SAMPLERS: dict[str, str] = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES: list[str] = list(KSAMPLER.keys())
SAMPLER_NAMES: list[str] = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

SCHEDULER_NAMES: list[str] = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "lcm",
    "turbo",
    "align_your_steps",
    "tcd",
    "edm_playground_v2.5",
]

sampler_list: list[str] = SAMPLER_NAMES
scheduler_list: list[str] = SCHEDULER_NAMES

# ---------------------------------------------------------------------------
# Miscellaneous defaults
# ---------------------------------------------------------------------------

clip_skip_max: int = 12
default_vae: str = "Default (model)"
refiner_swap_method: str = "joint"

default_input_image_tab: str = "uov_tab"
input_image_tab_ids: list[str] = [
    "uov_tab",
    "ip_tab",
    "inpaint_tab",
    "describe_tab",
    "enhance_tab",
    "metadata_tab",
]

# ---------------------------------------------------------------------------
# ControlNet / ImagePrompt
# ---------------------------------------------------------------------------

cn_ip: str = "ImagePrompt"
cn_ip_face: str = "FaceSwap"
cn_canny: str = "PyraCanny"
cn_cpds: str = "CPDS"

ip_list: list[str] = [cn_ip, cn_canny, cn_cpds, cn_ip_face]
default_ip: str = cn_ip

default_parameters: dict[str, tuple[float, float]] = {
    cn_ip: (0.5, 0.6),
    cn_ip_face: (0.9, 0.75),
    cn_canny: (0.5, 1.0),
    cn_cpds: (0.5, 1.0),
}

# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

output_formats: list[str] = ["png", "jpeg", "webp"]  # kept in sync with OutputFormat.list()

# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------

inpaint_mask_models: list[str] = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
]
inpaint_mask_cloth_category: list[str] = ["full", "upper", "lower"]
inpaint_mask_sam_model: list[str] = ["vit_b", "vit_l", "vit_h"]

inpaint_engine_versions: list[str] = ["None", "v1", "v2.5", "v2.6"]
inpaint_option_default: str = "Inpaint or Outpaint (default)"
inpaint_option_detail: str = "Improve Detail (face, hand, eyes, etc.)"
inpaint_option_modify: str = "Modify Content (add objects, change background, etc.)"
inpaint_options: list[str] = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

# ---------------------------------------------------------------------------
# Describe types
# ---------------------------------------------------------------------------

describe_type_photo: str = "Photograph"
describe_type_anime: str = "Art/Anime"
describe_types: list[str] = [describe_type_photo, describe_type_anime]

# ---------------------------------------------------------------------------
# SDXL aspect ratios — full standard set (26 entries)
# ---------------------------------------------------------------------------

sdxl_aspect_ratios: list[str] = [
    "704*1408",
    "704*1344",
    "768*1344",
    "768*1280",
    "832*1216",
    "832*1152",
    "896*1152",
    "896*1088",
    "960*1088",
    "960*1024",
    "1024*1024",
    "1024*960",
    "1088*960",
    "1088*896",
    "1152*896",
    "1152*832",
    "1216*832",
    "1280*768",
    "1344*768",
    "1344*704",
    "1408*704",
    "1472*704",
    "1536*640",
    "1600*640",
    "1664*576",
    "1728*576",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MetadataScheme(Enum):
    """Metadata embedding format for generated images."""

    FOOOCUS = "fooocus"
    A1111 = "a1111"


metadata_scheme: list[tuple[str, str]] = [
    (f"{MetadataScheme.FOOOCUS.value} (json)", MetadataScheme.FOOOCUS.value),
    (f"{MetadataScheme.A1111.value} (plain text)", MetadataScheme.A1111.value),
]


class OutputFormat(Enum):
    """Image output file format."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"

    @classmethod
    def list(cls) -> list[str]:
        """Return all member values as a list."""
        return [member.value for member in cls]


class PerformanceLoRA(Enum):
    """LoRA filenames associated with each performance preset."""

    QUALITY = None
    SPEED = None
    EXTREME_SPEED = "sdxl_lcm_lora.safetensors"
    LIGHTNING = "sdxl_lightning_4step_lora.safetensors"
    HYPER_SD = "sdxl_hyper_sd_4step_lora.safetensors"


class Steps(IntEnum):
    """Step counts for each performance preset."""

    QUALITY = 60
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4

    @classmethod
    def keys(cls) -> list[str]:
        """Return member names as a list."""
        return list(cls.__members__)


class StepsUOV(IntEnum):
    """Step counts for upscale/variation at each performance preset."""

    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class Performance(Enum):
    """Generation quality/speed presets."""

    QUALITY = "Quality"
    SPEED = "Speed"
    EXTREME_SPEED = "Extreme Speed"
    LIGHTNING = "Lightning"
    HYPER_SD = "Hyper-SD"

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        """Return (name, value) tuples for all members."""
        return [(member.name, member.value) for member in cls]

    @classmethod
    def values(cls) -> list[str]:
        """Return display values for all members."""
        return [member.value for member in cls]

    @classmethod
    def by_steps(cls, steps: int | str) -> Performance:
        """Look up a Performance member by its step count."""
        return cls[Steps(int(steps)).name]

    @classmethod
    def has_restricted_features(cls, x: Performance | str) -> bool:
        """Return True if the preset restricts certain generation features."""
        value = x.value if isinstance(x, Performance) else x
        return value in _RESTRICTED_PERFORMANCE_VALUES

    def steps(self) -> int | None:
        """Return the step count for this preset, or None if not defined."""
        return Steps[self.name].value if self.name in Steps.__members__ else None

    def steps_uov(self) -> int | None:
        """Return the UOV step count for this preset, or None if not defined."""
        return StepsUOV[self.name].value if self.name in StepsUOV.__members__ else None

    def lora_filename(self) -> str | None:
        """Return the LoRA filename for this preset, or None."""
        return PerformanceLoRA[self.name].value if self.name in PerformanceLoRA.__members__ else None


# Presets that restrict generation features (e.g., no refiner support).
# Defined after the class so the enum members are available.
_RESTRICTED_PERFORMANCE_VALUES: frozenset[str] = frozenset(
    {
        Performance.EXTREME_SPEED.value,
        Performance.LIGHTNING.value,
        Performance.HYPER_SD.value,
    }
)
