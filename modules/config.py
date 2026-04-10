"""UnFooocused standalone configuration module.

Loads settings from config.txt (JSON) in the project root,
falling back to sensible defaults for SDXL image generation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parents[1] / "config.txt"

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


def load_config(
    config_path: str | os.PathLike[str] = _DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load configuration from a JSON file, merged over defaults.

    Args:
        config_path: Path to the JSON config file. Defaults to
            ``config.txt`` in the project root (next to ``modules/``).

    Returns:
        Merged configuration dictionary.

    Raises:
        ValueError: If the config file exists but contains invalid JSON,
            is not a JSON object, or has invalid override types.
    """
    config = dict(_DEFAULTS)
    config_path = Path(config_path)

    if config_path.is_file():
        try:
            with config_path.open(encoding="utf-8") as fh:
                user_config = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"config.txt contains invalid JSON: {exc}") from exc

        if not isinstance(user_config, dict):
            raise ValueError(f"config.txt must contain a JSON object, got {type(user_config).__name__}")

        config.update(user_config)

    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Validate critical config values that would cause import-time crashes."""
    if not isinstance(config.get("paths_checkpoints"), list):
        raise ValueError("config.txt: paths_checkpoints must be a list of paths")
    if not isinstance(config.get("paths_loras"), list):
        raise ValueError("config.txt: paths_loras must be a list of paths")
    if config["default_loras_min_weight"] >= config["default_loras_max_weight"]:
        raise ValueError("config.txt: default_loras_min_weight must be < default_loras_max_weight")


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
        try:
            entries = os.listdir(directory)
        except OSError as exc:
            logger.warning("Skipping unreadable directory %s: %s", directory, exc)
            continue
        for entry in entries:
            if entry.lower().endswith(extension):
                found.add(entry)
    return sorted(found)


# ---------------------------------------------------------------------------
# AppConfig — frozen dataclass holding all configuration values
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable configuration object for the UnFooocused application.

    Constructed via ``AppConfig.from_dict()`` (from a ``load_config()`` result)
    or directly in tests.  Fields mirror the module-level globals that existed
    previously, so the migration can proceed incrementally.
    """

    # Generation defaults
    default_base_model_name: str
    default_refiner_model_name: str
    default_refiner_switch: float
    default_performance: str
    default_aspect_ratio: str
    available_aspect_ratios: tuple[str, ...]
    default_image_number: int
    default_max_image_number: int
    default_output_format: str
    default_prompt: str
    default_prompt_negative: str
    default_styles: tuple[str, ...]
    default_cfg_scale: float
    default_sample_sharpness: float
    default_sampler: str
    default_scheduler: str
    default_loras: tuple[tuple, ...]
    default_loras_min_weight: float
    default_loras_max_weight: float
    default_max_lora_number: int
    default_controlnet_image_count: int
    default_enhance_tabs: int

    # Paths
    paths_checkpoints: tuple[str, ...]
    paths_loras: tuple[str, ...]
    path_embeddings: str
    path_outputs: str

    # Discovered model files
    model_filenames: tuple[str, ...]
    lora_filenames: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        """Create an ``AppConfig`` from a ``load_config()`` result dict.

        Runs file discovery for model and LoRA filenames using the
        checkpoint/LoRA paths found in *raw*.
        """
        paths_checkpoints: list[str] = raw["paths_checkpoints"]
        paths_loras: list[str] = raw["paths_loras"]

        return cls(
            default_base_model_name=raw["default_model"],
            default_refiner_model_name=raw["default_refiner"],
            default_refiner_switch=float(raw["default_refiner_switch"]),
            default_performance=raw["default_performance"],
            default_aspect_ratio=raw["default_aspect_ratio"],
            available_aspect_ratios=tuple(raw["available_aspect_ratios"]),
            default_image_number=int(raw["default_image_number"]),
            default_max_image_number=int(raw["default_max_image_number"]),
            default_output_format=raw["default_output_format"],
            default_prompt=raw["default_prompt"],
            default_prompt_negative=raw["default_prompt_negative"],
            default_styles=tuple(raw["default_styles"]),
            default_cfg_scale=float(raw["default_cfg_scale"]),
            default_sample_sharpness=float(raw["default_sample_sharpness"]),
            default_sampler=raw["default_sampler"],
            default_scheduler=raw["default_scheduler"],
            default_loras=tuple(tuple(lora) for lora in raw["default_loras"]),
            default_loras_min_weight=float(raw["default_loras_min_weight"]),
            default_loras_max_weight=float(raw["default_loras_max_weight"]),
            default_max_lora_number=int(raw["default_max_lora_number"]),
            default_controlnet_image_count=int(raw["default_controlnet_image_count"]),
            default_enhance_tabs=int(raw["default_enhance_tabs"]),
            paths_checkpoints=tuple(paths_checkpoints),
            paths_loras=tuple(paths_loras),
            path_embeddings=raw["path_embeddings"],
            path_outputs=raw["path_outputs"],
            model_filenames=tuple(_discover_files(paths_checkpoints)),
            lora_filenames=tuple(_discover_files(paths_loras)),
        )


# ---------------------------------------------------------------------------
# Singleton accessor — replaces module-level mutable globals
# ---------------------------------------------------------------------------

_app_config: AppConfig | None = None
_config_lock = threading.Lock()


def get_config() -> AppConfig:
    """Return the application configuration singleton.

    Lazily initializes on first call by loading from ``config.txt``.
    Uses double-checked locking for thread safety.
    """
    global _app_config
    if _app_config is None:
        with _config_lock:
            if _app_config is None:
                _app_config = AppConfig.from_dict(load_config())
    return _app_config


def set_config(config: AppConfig) -> None:
    """Override the singleton with *config* (for testing)."""
    global _app_config
    with _config_lock:
        _app_config = config


def reset_config() -> None:
    """Clear the singleton so the next ``get_config()`` reinitializes."""
    global _app_config
    with _config_lock:
        _app_config = None


# ---------------------------------------------------------------------------
# Pure refresh functions — return new lists, no global mutation
# ---------------------------------------------------------------------------


def refresh_model_filenames(paths: list[str]) -> list[str]:
    """Re-scan *paths* and return discovered checkpoint filenames."""
    return _discover_files(paths)


def refresh_lora_filenames(paths: list[str]) -> list[str]:
    """Re-scan *paths* and return discovered LoRA filenames."""
    return _discover_files(paths)
