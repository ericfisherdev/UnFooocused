"""UnFooocused standalone configuration module.

Loads settings from config.txt (JSON) in the project root,
falling back to sensible defaults for SDXL image generation.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    "default_vae": "Default (model)",
    "default_performance": "Speed",
    "default_overwrite_step": -1,
    "default_overwrite_switch": -1,
    "default_overwrite_upscale": -1,
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
    "path_vae": "./models/vae",
    "path_vae_approx": "./models/vae_approx",
    "path_upscale_models": "./models/upscale_models",
    "path_inpaint": "./models/inpaint",
    "path_controlnet": "./models/controlnet",
    "path_clip_vision": "./models/clip_vision",
    "path_fooocus_expansion": "./models/prompt_expansion/fooocus_expansion",
    "path_wildcards": "./wildcards",
    "path_safety_checker": "./models/safety_checker",
    "path_sam": "./models/sam",
    "path_lora_presets": "./lora_presets",
    "path_outputs": "./outputs",
    "path_fast_checkpoints": "",
    "temp_path": "./temp",
    "temp_path_cleanup_on_launch": True,
    # UI / metadata / advanced (UNF-57)
    "default_advanced_checkbox": False,
    "default_developer_debug_mode_checkbox": False,
    "default_black_out_nsfw": False,
    "default_save_metadata_to_images": False,
    "default_save_only_final_enhanced_image": False,
    "default_metadata_scheme": "fooocus",
    "metadata_created_by": "",
    "default_describe_apply_prompts_checkbox": True,
    "default_describe_content_type": ["Photograph"],
    # Base model preset (UNF-59) — populated below from _DEFAULT_BASE_MODEL_PRESET
}

_VALID_METADATA_SCHEMES: frozenset[str] = frozenset({"fooocus", "a111", "comfy"})
_VALID_DESCRIBE_CONTENT_TYPES: frozenset[str] = frozenset({"Photograph", "Art/Anime"})
_VALID_BASE_MODEL_PRESETS: frozenset[str] = frozenset({"SDXL", "Pony", "Illustrious"})
_DEFAULT_BASE_MODEL_PRESET: str = "SDXL"
_DEFAULTS["base_model_preset"] = _DEFAULT_BASE_MODEL_PRESET

# Download/cache maps (UNF-60): dict[filename, url]
_DOWNLOAD_CONFIG_KEYS: tuple[str, ...] = (
    "checkpoint_downloads",
    "lora_downloads",
    "embeddings_downloads",
    "vae_downloads",
)
for _dl_key in _DOWNLOAD_CONFIG_KEYS:
    _DEFAULTS[_dl_key] = {}
_UI_ADVANCED_BOOL_KEYS: tuple[str, ...] = (
    "default_advanced_checkbox",
    "default_developer_debug_mode_checkbox",
    "default_black_out_nsfw",
    "default_save_metadata_to_images",
    "default_save_only_final_enhanced_image",
    "default_describe_apply_prompts_checkbox",
)

_NEW_PATH_KEYS: tuple[str, ...] = (
    "path_vae",
    "path_vae_approx",
    "path_upscale_models",
    "path_inpaint",
    "path_controlnet",
    "path_clip_vision",
    "path_fooocus_expansion",
    "path_wildcards",
    "path_safety_checker",
    "path_sam",
    "path_lora_presets",
    "temp_path",
)


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

    _apply_path_env_overrides(config)
    _validate_config(config)
    _warn_missing_paths(config)
    if config.get("temp_path_cleanup_on_launch"):
        cleanup_temp_path(config["temp_path"])
    _write_config_template_next_to(config_path)
    return config


def _write_config_template_next_to(config_path: Path) -> None:
    """Write full_config_template.txt alongside *config_path* (UNF-61)."""
    from modules.config_template import write_template

    try:
        write_template(config_path.parent / "full_config_template.txt")
    except OSError as exc:
        logger.warning("failed to write full_config_template.txt: %s", exc)


def _apply_path_env_overrides(config: dict[str, Any]) -> None:
    for key in _NEW_PATH_KEYS:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value


def _warn_missing_paths(config: dict[str, Any]) -> None:
    for key in _NEW_PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            expanded = Path(value).expanduser()
        except OSError, RuntimeError:
            logger.warning("config.txt: %s=%r could not be expanded", key, value)
            continue
        if not expanded.is_dir():
            logger.warning("config.txt: %s=%r does not exist on disk", key, value)


_UNSAFE_TEMP_PATHS: frozenset[str] = frozenset({"", ".", "..", "/", "~"})


def _is_safe_temp_path(temp_path: str) -> bool:
    """Reject empty, root, cwd, and other destructive temp_path values (UNF-58)."""
    if not isinstance(temp_path, str) or temp_path.strip() in _UNSAFE_TEMP_PATHS:
        return False
    try:
        resolved = Path(temp_path).expanduser().resolve()
    except OSError, RuntimeError:
        return False
    if resolved == resolved.parent:
        return False
    cwd = Path.cwd().resolve()
    return not (resolved == cwd or cwd.is_relative_to(resolved))


def cleanup_temp_path(temp_path: str) -> None:
    """Remove all contents of *temp_path* on launch (UNF-58)."""
    if not _is_safe_temp_path(temp_path):
        logger.error("refusing to clean temp_path=%r: unsafe or destructive value", temp_path)
        return
    path = Path(temp_path).expanduser()
    if path.is_symlink():
        logger.warning("refusing to clean temp_path=%r because it is a symlink", temp_path)
        return
    if not path.is_dir():
        return
    try:
        children = tuple(path.iterdir())
    except OSError as exc:
        logger.warning("failed to enumerate temp_path=%r: %s", temp_path, exc)
        return
    for child in children:
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            logger.warning("failed to remove temp entry %s: %s", child, exc)


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
    _validate_image_gen_config(config)
    _validate_vae_perf_config(config)
    _validate_ui_metadata_config(config)
    _validate_base_model_preset_config(config)
    _validate_directory_paths(config)
    _validate_download_config(config)


def _validate_directory_paths(config: dict[str, Any]) -> None:
    """Validate directory path config keys (UNF-58)."""
    for key in _NEW_PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str):
            raise ValueError(f"config.txt: {key} must be a string path")
    cleanup = config.get("temp_path_cleanup_on_launch")
    if not isinstance(cleanup, bool):
        raise ValueError("config.txt: temp_path_cleanup_on_launch must be a bool")


_LORA_WEIGHT_BOUND: float = 10.0


def _is_number(value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    return math.isfinite(value)


def _validate_lora_entry(index: int, entry: Any) -> None:
    if not isinstance(entry, list | tuple) or len(entry) != 3:
        raise ValueError(f"config.txt: default_loras[{index}] must be a [enabled, filename, weight] triple")
    enabled, filename, weight = entry
    if not isinstance(enabled, bool):
        raise ValueError(f"config.txt: default_loras[{index}] enabled flag must be a bool")
    if not isinstance(filename, str):
        raise ValueError(f"config.txt: default_loras[{index}] filename must be a string")
    if not _is_number(weight):
        raise ValueError(f"config.txt: default_loras[{index}] weight must be a finite number")


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


_ASPECT_RATIO_RE = re.compile(r"^(\d+)[*x](\d+)$")
_VALID_OUTPUT_FORMATS: frozenset[str] = frozenset({"png", "jpeg", "webp"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_aspect_ratio_string(key: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(f"config.txt: {key} must be a string like 'WIDTH*HEIGHT' or 'WIDTHxHEIGHT'")
    match = _ASPECT_RATIO_RE.match(value)
    if not match:
        raise ValueError(f"config.txt: {key}={value!r} must match 'WIDTH*HEIGHT' or 'WIDTHxHEIGHT'")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"config.txt: {key}={value!r} dimensions must be positive")


def _validate_image_gen_config(config: dict[str, Any]) -> None:
    """Validate image generation default config keys (UNF-55)."""
    for key in ("default_prompt", "default_prompt_negative"):
        if not isinstance(config.get(key), str):
            raise ValueError(f"config.txt: {key} must be a string")

    styles = config.get("default_styles")
    if not isinstance(styles, list):
        raise ValueError("config.txt: default_styles must be a list of strings")
    if not all(isinstance(s, str) for s in styles):
        raise ValueError("config.txt: default_styles entries must all be strings")

    available = config.get("available_aspect_ratios")
    if not isinstance(available, list):
        raise ValueError("config.txt: available_aspect_ratios must be a list of aspect-ratio strings")
    for entry in available:
        _validate_aspect_ratio_string("available_aspect_ratios", entry)

    _validate_aspect_ratio_string("default_aspect_ratio", config.get("default_aspect_ratio"))
    if config["default_aspect_ratio"] not in available:
        raise ValueError(
            f"config.txt: default_aspect_ratio={config['default_aspect_ratio']!r} "
            f"must be one of available_aspect_ratios"
        )

    max_images = config.get("default_max_image_number")
    if not _is_positive_int(max_images):
        raise ValueError("config.txt: default_max_image_number must be a positive integer")

    image_number = config.get("default_image_number")
    if not _is_positive_int(image_number):
        raise ValueError("config.txt: default_image_number must be a positive integer")
    if image_number > max_images:
        logger.warning(
            "config.txt: default_image_number=%d exceeds default_max_image_number=%d; clamping to %d",
            image_number,
            max_images,
            max_images,
        )
        config["default_image_number"] = max_images

    output_format = config.get("default_output_format")
    if not isinstance(output_format, str):
        raise ValueError(f"config.txt: default_output_format must be a string, one of {sorted(_VALID_OUTPUT_FORMATS)}")
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"config.txt: default_output_format={output_format!r} must be one of {sorted(_VALID_OUTPUT_FORMATS)}"
        )


def _validate_overwrite_int(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config.txt: {key} must be an integer >= -1 (use -1 for auto)")
    if value < -1:
        raise ValueError(f"config.txt: {key}={value} must be >= -1")


def _validate_overwrite_upscale(config: dict[str, Any], key: str) -> None:
    converted = _require_finite_number(config, key)
    if converted < -1:
        raise ValueError(f"config.txt: {key}={converted} must be >= -1")


def _validate_vae_perf_config(config: dict[str, Any]) -> None:
    """Validate VAE and performance config keys (UNF-56)."""
    from modules.flags import Performance

    if not isinstance(config.get("default_vae"), str):
        raise ValueError("config.txt: default_vae must be a string")

    performance = config.get("default_performance")
    valid_values = Performance.values()
    if not isinstance(performance, str) or performance not in valid_values:
        raise ValueError(f"config.txt: default_performance={performance!r} must be one of {valid_values}")

    _validate_overwrite_int(config, "default_overwrite_step")
    _validate_overwrite_int(config, "default_overwrite_switch")
    _validate_overwrite_upscale(config, "default_overwrite_upscale")


def _validate_ui_metadata_config(config: dict[str, Any]) -> None:
    """Validate UI / metadata / advanced config keys (UNF-57)."""
    for key in _UI_ADVANCED_BOOL_KEYS:
        value = config.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"config.txt: {key} must be a boolean, got {value!r}")

    scheme = config.get("default_metadata_scheme")
    if not isinstance(scheme, str):
        raise ValueError(f"config.txt: default_metadata_scheme must be a string, got {scheme!r}")
    if scheme not in _VALID_METADATA_SCHEMES:
        raise ValueError(
            f"config.txt: default_metadata_scheme={scheme!r} must be one of {sorted(_VALID_METADATA_SCHEMES)}"
        )

    created_by = config.get("metadata_created_by")
    if not isinstance(created_by, str):
        raise ValueError(f"config.txt: metadata_created_by must be a string, got {created_by!r}")

    content_types = config.get("default_describe_content_type")
    if not isinstance(content_types, list):
        raise ValueError("config.txt: default_describe_content_type must be a list of strings")
    for entry in content_types:
        if not isinstance(entry, str):
            raise ValueError(f"config.txt: default_describe_content_type entries must be strings, got {entry!r}")
        if entry not in _VALID_DESCRIBE_CONTENT_TYPES:
            raise ValueError(
                f"config.txt: default_describe_content_type entry {entry!r} must be one of "
                f"{sorted(_VALID_DESCRIBE_CONTENT_TYPES)}"
            )


def _validate_download_config(config: dict[str, Any]) -> None:
    """Validate download/cache map config keys (UNF-60).

    Each of ``checkpoint_downloads``, ``lora_downloads``, ``embeddings_downloads``,
    ``vae_downloads`` must be a ``dict[str, str]`` mapping filename to download URL.
    """
    for key in _DOWNLOAD_CONFIG_KEYS:
        mapping = config.get(key)
        if not isinstance(mapping, dict):
            raise ValueError(f"config.txt: {key} must be a mapping of filename to URL, got {type(mapping).__name__}")
        for filename, url in mapping.items():
            if not isinstance(filename, str):
                raise ValueError(f"config.txt: {key} keys must be strings, got {filename!r}")
            if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
                raise ValueError(
                    f"config.txt: {key} keys must be plain filenames without path components, got {filename!r}"
                )
            if not isinstance(url, str):
                raise ValueError(f"config.txt: {key}[{filename!r}] URL must be a string, got {url!r}")


@dataclass(frozen=True, slots=True)
class PendingDownload:
    """A file queued for download on startup (UNF-60)."""

    filename: str
    url: str
    target_dir: Path


def plan_missing_downloads(downloads: Mapping[str, str], target_dir: str | Path) -> tuple[PendingDownload, ...]:
    """Return downloads whose target file does not yet exist.

    Pure function: checks ``target_dir / filename`` for each entry in
    *downloads* and yields a ``PendingDownload`` for every missing file.
    """
    target = Path(target_dir)
    return tuple(
        PendingDownload(filename=name, url=url, target_dir=target)
        for name, url in downloads.items()
        if not (target / name).is_file()
    )


def _validate_base_model_preset_config(config: dict[str, Any]) -> None:
    """Validate base_model_preset with warn+fallback semantics (UNF-59).

    Unlike other validators, invalid values do not raise — they log a warning
    and fall back to the default preset so startup never crashes on a typo.
    """
    value = config.get("base_model_preset")
    if isinstance(value, str) and value in _VALID_BASE_MODEL_PRESETS:
        return
    logger.warning(
        "config.txt: base_model_preset=%r is not one of %s; falling back to %r",
        value,
        sorted(_VALID_BASE_MODEL_PRESETS),
        _DEFAULT_BASE_MODEL_PRESET,
    )
    config["base_model_preset"] = _DEFAULT_BASE_MODEL_PRESET


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
    default_vae: str
    default_performance: str
    default_overwrite_step: int
    default_overwrite_switch: int
    default_overwrite_upscale: float
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
    path_vae: str
    path_vae_approx: str
    path_upscale_models: str
    path_inpaint: str
    path_controlnet: str
    path_clip_vision: str
    path_fooocus_expansion: str
    path_wildcards: str
    path_safety_checker: str
    path_sam: str
    path_lora_presets: str
    path_outputs: str
    path_fast_checkpoints: str
    temp_path: str
    temp_path_cleanup_on_launch: bool

    # UI / metadata / advanced (UNF-57)
    default_advanced_checkbox: bool
    default_developer_debug_mode_checkbox: bool
    default_black_out_nsfw: bool
    default_save_metadata_to_images: bool
    default_save_only_final_enhanced_image: bool
    default_metadata_scheme: str
    metadata_created_by: str
    default_describe_apply_prompts_checkbox: bool
    default_describe_content_type: tuple[str, ...]

    # Base model preset (UNF-59)
    base_model_preset: str

    # Download/cache maps (UNF-60) — read-only views over dict[filename, url]
    checkpoint_downloads: Mapping[str, str]
    lora_downloads: Mapping[str, str]
    embeddings_downloads: Mapping[str, str]
    vae_downloads: Mapping[str, str]

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
            default_vae=raw["default_vae"],
            default_performance=raw["default_performance"],
            default_overwrite_step=int(raw["default_overwrite_step"]),
            default_overwrite_switch=int(raw["default_overwrite_switch"]),
            default_overwrite_upscale=float(raw["default_overwrite_upscale"]),
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
            path_vae=raw["path_vae"],
            path_vae_approx=raw["path_vae_approx"],
            path_upscale_models=raw["path_upscale_models"],
            path_inpaint=raw["path_inpaint"],
            path_controlnet=raw["path_controlnet"],
            path_clip_vision=raw["path_clip_vision"],
            path_fooocus_expansion=raw["path_fooocus_expansion"],
            path_wildcards=raw["path_wildcards"],
            path_safety_checker=raw["path_safety_checker"],
            path_sam=raw["path_sam"],
            path_lora_presets=raw["path_lora_presets"],
            path_outputs=raw["path_outputs"],
            path_fast_checkpoints=raw.get("path_fast_checkpoints", ""),
            temp_path=raw["temp_path"],
            temp_path_cleanup_on_launch=bool(raw["temp_path_cleanup_on_launch"]),
            default_advanced_checkbox=raw["default_advanced_checkbox"],
            default_developer_debug_mode_checkbox=raw["default_developer_debug_mode_checkbox"],
            default_black_out_nsfw=raw["default_black_out_nsfw"],
            default_save_metadata_to_images=raw["default_save_metadata_to_images"],
            default_save_only_final_enhanced_image=raw["default_save_only_final_enhanced_image"],
            default_metadata_scheme=raw["default_metadata_scheme"],
            metadata_created_by=raw["metadata_created_by"],
            default_describe_apply_prompts_checkbox=raw["default_describe_apply_prompts_checkbox"],
            default_describe_content_type=tuple(raw["default_describe_content_type"]),
            base_model_preset=raw["base_model_preset"],
            checkpoint_downloads=MappingProxyType(dict(raw["checkpoint_downloads"])),
            lora_downloads=MappingProxyType(dict(raw["lora_downloads"])),
            embeddings_downloads=MappingProxyType(dict(raw["embeddings_downloads"])),
            vae_downloads=MappingProxyType(dict(raw["vae_downloads"])),
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
