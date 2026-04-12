"""UnFooocused standalone configuration module.

Loads settings from config.txt (JSON) in the project root,
falling back to sensible defaults for SDXL image generation.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.flags import SAMPLER_NAMES, SCHEDULER_NAMES, clip_skip_max, sdxl_aspect_ratios

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parents[1] / "config.txt"

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "default_model": "juggernautXL_v8Rundiffusion.safetensors",
    "default_base_model": None,
    "default_refiner": "None",
    "default_refiner_switch": 0.5,
    "previous_default_models": [],
    "default_performance": "Speed",
    "default_aspect_ratio": "1152*896",
    "available_aspect_ratios": sdxl_aspect_ratios,
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
    "default_cfg_scale": 7.0,
    "default_cfg_tsnr": 7.0,
    "default_sample_sharpness": 2.0,
    "default_sampler": "dpmpp_2m_sde_gpu",
    "default_scheduler": "karras",
    "default_clip_skip": 2,
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
    "default_steps": 30,
    "default_enhance_tabs": 3,
    # Paths
    "paths_checkpoints": ["./models/checkpoints"],
    "paths_loras": ["./models/loras"],
    "path_embeddings": "./models/embeddings",
    "path_outputs": "./outputs",
    "path_fast_checkpoints": "",
}


def load_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
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
    config = copy.deepcopy(_DEFAULTS)
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
    if not isinstance(config.get("path_fast_checkpoints"), str):
        raise ValueError("config.txt: path_fast_checkpoints must be a string path")
    _validate_lora_config(config)
    if config["default_loras_min_weight"] >= config["default_loras_max_weight"]:
        raise ValueError("config.txt: default_loras_min_weight must be < default_loras_max_weight")
    _validate_model_refiner_config(config)
    _validate_sampling_config(config)


_LORA_WEIGHT_BOUND: float = 10.0


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_lora_entry(index: int, entry: Any) -> None:
    if not isinstance(entry, list | tuple) or len(entry) != 3:
        raise ValueError(f"config.txt: default_loras[{index}] must be a [enabled, filename, weight] triple")
    enabled, filename, weight = entry
    if not isinstance(enabled, bool):
        raise ValueError(f"config.txt: default_loras[{index}] enabled flag must be a bool")
    if not isinstance(filename, str):
        raise ValueError(f"config.txt: default_loras[{index}] filename must be a string")
    if not _is_number(weight):
        raise ValueError(f"config.txt: default_loras[{index}] weight must be a number")


def _validate_lora_weight_bound(key: str, value: Any) -> None:
    if not _is_number(value):
        raise ValueError(f"config.txt: {key} must be a number in [-10, 10]")
    if not -_LORA_WEIGHT_BOUND <= float(value) <= _LORA_WEIGHT_BOUND:
        raise ValueError(f"config.txt: {key} must be within [-10, 10]")


def _validate_lora_config(config: dict[str, Any]) -> None:
    """Validate LoRA config keys (UNF-54)."""
    loras = config.get("default_loras")
    if not isinstance(loras, list):
        raise ValueError("config.txt: default_loras must be a list of [enabled, filename, weight] triples")
    for index, entry in enumerate(loras):
        _validate_lora_entry(index, entry)

    _validate_lora_weight_bound("default_loras_min_weight", config.get("default_loras_min_weight"))
    _validate_lora_weight_bound("default_loras_max_weight", config.get("default_loras_max_weight"))

    max_loras = config.get("default_max_lora_number")
    if not isinstance(max_loras, int) or isinstance(max_loras, bool):
        raise ValueError("config.txt: default_max_lora_number must be a positive integer")
    if max_loras <= 0:
        raise ValueError("config.txt: default_max_lora_number must be > 0")
    if len(loras) > max_loras:
        raise ValueError(
            f"config.txt: default_loras has {len(loras)} entries but default_max_lora_number is {max_loras}"
        )


def _validate_model_refiner_config(config: dict[str, Any]) -> None:
    """Validate model, refiner, and model-history config keys (UNF-52)."""
    if not isinstance(config.get("default_model"), str):
        raise ValueError("config.txt: default_model must be a string filename")
    if not isinstance(config.get("default_refiner"), str):
        raise ValueError("config.txt: default_refiner must be a string filename or 'None'")

    switch = config.get("default_refiner_switch")
    if not isinstance(switch, int | float) or isinstance(switch, bool):
        raise ValueError("config.txt: default_refiner_switch must be a number in [0, 1]")
    if not 0.0 <= float(switch) <= 1.0:
        raise ValueError("config.txt: default_refiner_switch must be in [0, 1]")

    base_model = config.get("default_base_model")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError("config.txt: default_base_model must be a string or null")

    previous_models = config.get("previous_default_models")
    if not isinstance(previous_models, list):
        raise ValueError("config.txt: previous_default_models must be a list of filenames")
    if not all(isinstance(item, str) for item in previous_models):
        raise ValueError("config.txt: previous_default_models entries must all be strings")


def _validate_sampling_config(config: dict[str, Any]) -> None:
    """Validate sampler, scheduler, CFG, and clip_skip override values."""
    sampler = config.get("default_sampler")
    if sampler not in SAMPLER_NAMES:
        raise ValueError(
            f"config.txt: default_sampler={sampler!r} is not a valid sampler. Valid options: {', '.join(SAMPLER_NAMES)}"
        )

    scheduler = config.get("default_scheduler")
    if scheduler not in SCHEDULER_NAMES:
        raise ValueError(
            f"config.txt: default_scheduler={scheduler!r} is not a valid scheduler. "
            f"Valid options: {', '.join(SCHEDULER_NAMES)}"
        )

    clip_skip = config.get("default_clip_skip")
    if not isinstance(clip_skip, int) or isinstance(clip_skip, bool):
        raise ValueError(f"config.txt: default_clip_skip must be an integer in 1..{clip_skip_max}, got {clip_skip!r}")
    if not (1 <= clip_skip <= clip_skip_max):
        raise ValueError(f"config.txt: default_clip_skip must be in 1..{clip_skip_max}, got {clip_skip}")

    _require_finite_number(config, "default_cfg_tsnr")
    cfg_scale = _require_finite_number(config, "default_cfg_scale")
    if cfg_scale <= 0:
        raise ValueError(f"config.txt: default_cfg_scale must be > 0, got {cfg_scale}")

    sharpness = _require_finite_number(config, "default_sample_sharpness")
    if sharpness < 0:
        raise ValueError(f"config.txt: default_sample_sharpness must be >= 0, got {sharpness}")


def _require_finite_number(config: dict[str, Any], key: str) -> float:
    """Return *config[key]* as float, raising ValueError unless finite and numeric."""
    value = config.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"config.txt: {key} must be a number, got {value!r}")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"config.txt: {key} must be finite, got {value}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"config.txt: {key} must be finite, got {value}")
    return converted


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_files(paths: list[str], extension: str = ".safetensors") -> list[str]:
    """Scan directories recursively for files with the given extension.

    Returns:
        Sorted list of paths relative to each search root.  Root-level files
        are returned as plain filenames (e.g. ``model.safetensors``);
        subdirectory files use forward-slash separators
        (e.g. ``pony/model.safetensors``).
    """
    found: set[str] = set()
    ext_lower = extension.lower()
    for directory in paths:
        root_path = Path(directory)
        if not root_path.is_dir():
            continue
        try:
            for file_path in root_path.rglob("*"):
                if not file_path.is_file():
                    continue
                if not file_path.name.lower().endswith(ext_lower):
                    continue
                relative = file_path.relative_to(root_path)
                found.add(relative.as_posix())
        except OSError as exc:
            logger.warning("Skipping unreadable directory %s: %s", directory, exc)
            continue
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
    default_base_model: str | None
    previous_default_models: tuple[str, ...]
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
    default_cfg_tsnr: float
    default_sample_sharpness: float
    default_sampler: str
    default_scheduler: str
    default_clip_skip: int
    default_loras: tuple[tuple, ...]
    default_loras_min_weight: float
    default_loras_max_weight: float
    default_max_lora_number: int
    default_steps: int
    default_controlnet_image_count: int
    default_enhance_tabs: int

    # Paths
    paths_checkpoints: tuple[str, ...]
    paths_loras: tuple[str, ...]
    path_embeddings: str
    path_outputs: str
    path_fast_checkpoints: str

    # Discovered model files
    model_filenames: tuple[str, ...]
    lora_filenames: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        """Create an ``AppConfig`` from a ``load_config()`` result dict.

        Runs file discovery for model and LoRA filenames using the
        checkpoint/LoRA paths found in *raw*.
        """
        from modules.flags import Performance

        paths_checkpoints: list[str] = raw["paths_checkpoints"]
        paths_loras: list[str] = raw["paths_loras"]

        raw_steps = int(raw.get("default_steps", -1))
        if raw_steps <= 0:
            perf = Performance(raw["default_performance"])
            raw_steps = perf.steps() or 30

        return cls(
            default_base_model_name=raw["default_model"],
            default_base_model=raw.get("default_base_model"),
            previous_default_models=tuple(raw.get("previous_default_models", [])),
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
            default_cfg_tsnr=float(raw["default_cfg_tsnr"]),
            default_sample_sharpness=float(raw["default_sample_sharpness"]),
            default_sampler=raw["default_sampler"],
            default_scheduler=raw["default_scheduler"],
            default_clip_skip=int(raw["default_clip_skip"]),
            default_loras=tuple(tuple(lora) for lora in raw["default_loras"]),
            default_loras_min_weight=float(raw["default_loras_min_weight"]),
            default_loras_max_weight=float(raw["default_loras_max_weight"]),
            default_max_lora_number=int(raw["default_max_lora_number"]),
            default_steps=raw_steps,
            default_controlnet_image_count=int(raw["default_controlnet_image_count"]),
            default_enhance_tabs=int(raw["default_enhance_tabs"]),
            paths_checkpoints=tuple(paths_checkpoints),
            paths_loras=tuple(paths_loras),
            path_embeddings=raw["path_embeddings"],
            path_outputs=raw["path_outputs"],
            path_fast_checkpoints=raw.get("path_fast_checkpoints", ""),
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
