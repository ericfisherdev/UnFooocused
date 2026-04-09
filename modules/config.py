"""UnFooocused standalone configuration module.

Loads settings from config.txt (JSON) in the working directory,
falling back to sensible defaults for SDXL image generation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDXL standard aspect ratios
# ---------------------------------------------------------------------------

SDXL_ASPECT_RATIOS: list[str] = [
    "704*1408",
    "704*1344",
    "768*1344",
    "768*1280",
    "832*1216",
    "896*1152",
    "1024*1024",
    "1152*896",
    "1216*832",
    "1280*768",
    "1344*768",
    "1344*704",
    "1408*704",
]

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "default_model": "juggernautXL_v8Rundiffusion.safetensors",
    "default_refiner": "None",
    "default_refiner_switch": 0.5,
    "default_performance": "Speed",
    "default_aspect_ratio": "1152*896",
    "available_aspect_ratios": SDXL_ASPECT_RATIOS,
    "default_image_number": 1,
    "default_max_image_number": 32,
    "default_output_format": "png",
    "default_prompt": "",
    "default_prompt_negative": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Enhance",
        "Fooocus Sharp",
    ],
    "default_cfg_scale": 4.0,
    "default_sample_sharpness": 2.0,
    "default_sampler": "dpmpp_2m_sde_gpu",
    "default_scheduler": "karras",
    "default_loras": [
        [True, "sd_xl_offset_example-lora_1.0.safetensors", 0.1],
        [True, "None", 1.0],
        [True, "None", 1.0],
        [True, "None", 1.0],
        [True, "None", 1.0],
    ],
    "default_loras_min_weight": -2.0,
    "default_loras_max_weight": 2.0,
    "default_max_lora_number": 5,
    "default_controlnet_image_count": 4,
    "default_enhance_tabs": 3,
    # Paths
    "paths_checkpoints": ["./models/checkpoints"],
    "paths_loras": ["./models/loras"],
    "path_embeddings": "./models/embeddings",
    "path_outputs": "./outputs",
}


def load_config(config_path: str = "config.txt") -> dict[str, Any]:
    """Load configuration from a JSON file, merged over defaults.

    Args:
        config_path: Path to the JSON config file. Defaults to
            ``config.txt`` in the current working directory.

    Returns:
        Merged configuration dictionary.

    Raises:
        ValueError: If the config file exists but contains invalid JSON
            or is not a JSON object.
    """
    config = dict(_DEFAULTS)

    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                user_config = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"config.txt contains invalid JSON: {exc}"
            ) from exc

        if not isinstance(user_config, dict):
            raise ValueError(
                "config.txt must contain a JSON object, "
                f"got {type(user_config).__name__}"
            )

        config.update(user_config)

    return config


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_files(paths: list[str], extension: str = ".safetensors") -> list[str]:
    """Scan directories for files with the given extension.

    Returns:
        Sorted list of filenames (not full paths).
    """
    found: set[str] = set()
    for directory in paths:
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            if entry.lower().endswith(extension):
                found.add(entry)
    return sorted(found)


# ---------------------------------------------------------------------------
# Module-level state — populated on import, matches app.py's expectations
# ---------------------------------------------------------------------------

_config = load_config()

# Generation defaults (accessed as config.default_base_model_name, etc.)
default_base_model_name: str = _config["default_model"]
default_refiner_model_name: str = _config["default_refiner"]
default_refiner_switch: float = _config["default_refiner_switch"]
default_performance: str = _config["default_performance"]
default_aspect_ratio: str = _config["default_aspect_ratio"]
available_aspect_ratios: list[str] = _config["available_aspect_ratios"]
default_image_number: int = _config["default_image_number"]
default_max_image_number: int = _config["default_max_image_number"]
default_output_format: str = _config["default_output_format"]
default_prompt: str = _config["default_prompt"]
default_prompt_negative: str = _config["default_prompt_negative"]
default_styles: list[str] = _config["default_styles"]
default_cfg_scale: float = _config["default_cfg_scale"]
default_sample_sharpness: float = _config["default_sample_sharpness"]
default_sampler: str = _config["default_sampler"]
default_scheduler: str = _config["default_scheduler"]
default_loras: list = _config["default_loras"]
default_loras_min_weight: float = _config["default_loras_min_weight"]
default_loras_max_weight: float = _config["default_loras_max_weight"]
default_max_lora_number: int = _config["default_max_lora_number"]
default_controlnet_image_count: int = _config["default_controlnet_image_count"]
default_enhance_tabs: int = _config["default_enhance_tabs"]

# Paths
paths_checkpoints: list[str] = _config["paths_checkpoints"]
paths_loras: list[str] = _config["paths_loras"]
path_embeddings: str = _config["path_embeddings"]
path_outputs: str = _config["path_outputs"]

# Discovered model files
model_filenames: list[str] = _discover_files(paths_checkpoints)
lora_filenames: list[str] = _discover_files(paths_loras)


def update_model_filenames() -> None:
    """Re-scan checkpoint directories and update model_filenames."""
    global model_filenames
    model_filenames = _discover_files(paths_checkpoints)


def update_lora_filenames() -> None:
    """Re-scan LoRA directories and update lora_filenames."""
    global lora_filenames
    lora_filenames = _discover_files(paths_loras)
