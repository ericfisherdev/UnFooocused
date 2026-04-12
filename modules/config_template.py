"""UNF-61: full_config_template.txt generation.

Produces a JSONC reference file that documents every recognized config key
with type, default, valid values, and description. Auto-generated next to
``config.txt`` on startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class KeyDoc:
    """Documentation metadata for a single config key."""

    category: str
    type_hint: str
    description: str
    valid_values: str = ""


_CATEGORY_ORDER: tuple[str, ...] = (
    "Model & Refiner",
    "Sampling",
    "LoRA",
    "Image Generation",
    "VAE & Performance",
    "UI, Metadata & Advanced",
    "Base Model Preset",
    "Downloads & Cache",
    "Paths",
)


KEY_DOCS: dict[str, KeyDoc] = {
    # Model & Refiner
    "default_model": KeyDoc("Model & Refiner", "string", "Default checkpoint filename loaded on startup."),
    "default_base_model": KeyDoc(
        "Model & Refiner", "string|null", "Alternate base-model pointer; null to use default_model."
    ),
    "default_refiner": KeyDoc("Model & Refiner", "string", 'Refiner checkpoint filename, or "None" to disable.'),
    "default_refiner_switch": KeyDoc(
        "Model & Refiner",
        "float",
        "Fraction of denoising steps before switching to refiner.",
        "0.0 - 1.0",
    ),
    "previous_default_models": KeyDoc(
        "Model & Refiner", "list[string]", "Historical model filenames to migrate away from."
    ),
    "default_vae": KeyDoc(
        "Model & Refiner", "string", "VAE filename or 'Default (model)' to use the model's built-in VAE."
    ),
    "base_model_preset": KeyDoc(
        "Base Model Preset",
        "string",
        "Active preset that seeds sampler/scheduler defaults.",
        "SDXL | Pony | Illustrious",
    ),
    # Sampling
    "default_performance": KeyDoc(
        "Sampling", "string", "Performance preset.", "Speed | Quality | Extreme Speed | Lightning | Hyper-SD"
    ),
    "default_overwrite_step": KeyDoc(
        "Sampling", "int", "Override performance preset steps. -1 leaves preset unchanged."
    ),
    "default_overwrite_switch": KeyDoc("Sampling", "int", "Override refiner switch step. -1 leaves preset unchanged."),
    "default_overwrite_upscale": KeyDoc("Sampling", "float", "Override upscale factor. -1 leaves preset unchanged."),
    "default_sampler": KeyDoc("Sampling", "string", "Default sampler name."),
    "default_scheduler": KeyDoc("Sampling", "string", "Default scheduler name."),
    "default_clip_skip": KeyDoc("Sampling", "int", "CLIP skip value.", "1 - 12"),
    "default_cfg_scale": KeyDoc("Sampling", "float", "Default classifier-free guidance scale."),
    "default_cfg_tsnr": KeyDoc("Sampling", "float", "TSNR-adaptive CFG target value."),
    "default_sample_sharpness": KeyDoc("Sampling", "float", "Sample sharpness (negative prompt strength)."),
    "default_steps": KeyDoc("Sampling", "int", "Default total denoising steps."),
    # LoRA
    "default_loras": KeyDoc(
        "LoRA",
        "list[[bool,string,float]]",
        "Default LoRA entries as (enabled, filename, weight) triples.",
    ),
    "default_loras_min_weight": KeyDoc("LoRA", "float", "Minimum allowed LoRA weight slider bound."),
    "default_loras_max_weight": KeyDoc("LoRA", "float", "Maximum allowed LoRA weight slider bound."),
    "default_max_lora_number": KeyDoc("LoRA", "int", "Maximum number of LoRA slots in the UI."),
    # Image Generation
    "default_aspect_ratio": KeyDoc("Image Generation", "string", "Default aspect ratio expressed as 'width*height'."),
    "available_aspect_ratios": KeyDoc(
        "Image Generation", "list[string]", "Selectable aspect ratios for the UI dropdown."
    ),
    "default_image_number": KeyDoc("Image Generation", "int", "Default batch size (images per generation)."),
    "default_max_image_number": KeyDoc("Image Generation", "int", "Upper limit for the batch-size slider."),
    "default_output_format": KeyDoc("Image Generation", "string", "Output file format.", "png | jpg | webp"),
    "default_prompt": KeyDoc("Image Generation", "string", "Starter positive prompt."),
    "default_prompt_negative": KeyDoc("Image Generation", "string", "Starter negative prompt."),
    "default_styles": KeyDoc("Image Generation", "list[string]", "Styles enabled by default."),
    "default_controlnet_image_count": KeyDoc("Image Generation", "int", "Number of ControlNet image slots."),
    "default_enhance_tabs": KeyDoc("Image Generation", "int", "Number of enhance tabs in the UI."),
    # VAE & Performance — (kept under Sampling/Image Gen categories above where relevant)
    # UI, Metadata & Advanced
    "default_advanced_checkbox": KeyDoc("UI, Metadata & Advanced", "bool", "Expand advanced controls by default."),
    "default_developer_debug_mode_checkbox": KeyDoc(
        "UI, Metadata & Advanced", "bool", "Enable developer debug mode checkbox default."
    ),
    "default_black_out_nsfw": KeyDoc("UI, Metadata & Advanced", "bool", "Automatically black out NSFW detections."),
    "default_save_metadata_to_images": KeyDoc("UI, Metadata & Advanced", "bool", "Embed metadata in generated images."),
    "default_save_only_final_enhanced_image": KeyDoc(
        "UI, Metadata & Advanced",
        "bool",
        "Save only the final enhanced image instead of every stage.",
    ),
    "default_metadata_scheme": KeyDoc(
        "UI, Metadata & Advanced",
        "string",
        "Metadata schema used when saving images.",
        "fooocus | a111 | comfy",
    ),
    "metadata_created_by": KeyDoc("UI, Metadata & Advanced", "string", "Creator name stamped in image metadata."),
    "default_describe_apply_prompts_checkbox": KeyDoc(
        "UI, Metadata & Advanced",
        "bool",
        "Auto-apply describe prompts to the prompt field.",
    ),
    "default_describe_content_type": KeyDoc(
        "UI, Metadata & Advanced",
        "list[string]",
        "Content types enabled for the describe feature.",
        "subset of: Photograph, Art/Anime",
    ),
    # Downloads & Cache
    "checkpoint_downloads": KeyDoc(
        "Downloads & Cache",
        "dict[string,string]",
        "Filename → URL map; missing files are queued for download on startup.",
    ),
    "lora_downloads": KeyDoc("Downloads & Cache", "dict[string,string]", "LoRA filename → URL download map."),
    "embeddings_downloads": KeyDoc(
        "Downloads & Cache", "dict[string,string]", "Embedding filename → URL download map."
    ),
    "vae_downloads": KeyDoc("Downloads & Cache", "dict[string,string]", "VAE filename → URL download map."),
    # Paths
    "paths_checkpoints": KeyDoc("Paths", "list[string]", "Directories searched for checkpoints."),
    "paths_loras": KeyDoc("Paths", "list[string]", "Directories searched for LoRAs."),
    "path_embeddings": KeyDoc("Paths", "string", "Directory for textual-inversion embeddings."),
    "path_vae": KeyDoc("Paths", "string", "Directory for VAE checkpoints."),
    "path_vae_approx": KeyDoc("Paths", "string", "Directory for VAE approximation models."),
    "path_upscale_models": KeyDoc("Paths", "string", "Directory for upscaler models."),
    "path_inpaint": KeyDoc("Paths", "string", "Directory for inpainting models."),
    "path_controlnet": KeyDoc("Paths", "string", "Directory for ControlNet models."),
    "path_clip_vision": KeyDoc("Paths", "string", "Directory for CLIP vision models."),
    "path_fooocus_expansion": KeyDoc("Paths", "string", "Directory for the prompt-expansion model."),
    "path_wildcards": KeyDoc("Paths", "string", "Directory for wildcard text files."),
    "path_safety_checker": KeyDoc("Paths", "string", "Directory for safety-checker models."),
    "path_sam": KeyDoc("Paths", "string", "Directory for Segment Anything models."),
    "path_lora_presets": KeyDoc("Paths", "string", "Directory for LoRA presets."),
    "path_outputs": KeyDoc("Paths", "string", "Directory where generated images are saved."),
    "path_fast_checkpoints": KeyDoc("Paths", "string", "Optional alternate directory for fast-loading checkpoints."),
    "temp_path": KeyDoc("Paths", "string", "Directory for temporary files created during generation."),
    "temp_path_cleanup_on_launch": KeyDoc("Paths", "bool", "Remove contents of temp_path when the app starts."),
}


def _validate_registry(defaults: dict[str, Any], docs: dict[str, KeyDoc]) -> None:
    """Raise if any default key lacks documentation metadata."""
    missing = sorted(set(defaults) - set(docs))
    if missing:
        raise ValueError(
            f"config_template.KEY_DOCS missing entries for keys: {missing}. Add a KeyDoc for each new config key."
        )


def _format_default(value: Any) -> str:
    """Render *value* as a JSON literal (trailing comma handled by caller)."""
    return json.dumps(value, indent=2, sort_keys=False)


def generate_template() -> str:
    """Return the full_config_template.txt contents as a JSONC string."""
    from modules.config import _DEFAULTS

    _validate_registry(_DEFAULTS, KEY_DOCS)

    by_category: dict[str, list[str]] = {cat: [] for cat in _CATEGORY_ORDER}
    for key, doc in KEY_DOCS.items():
        by_category.setdefault(doc.category, []).append(key)

    lines: list[str] = [
        "// UnFooocused full config template — auto-generated reference (UNF-61).",
        "// Every option shown here is recognized by config.txt.",
        "// Lines starting with // are comments and must be stripped before JSON parsing.",
        "{",
    ]

    flat_entries: list[tuple[str, str, KeyDoc, Any]] = []
    for category in _CATEGORY_ORDER:
        for key in by_category.get(category, ()):
            flat_entries.append((category, key, KEY_DOCS[key], _DEFAULTS[key]))

    last_index = len(flat_entries) - 1
    current_category: str | None = None
    for idx, (category, key, doc, value) in enumerate(flat_entries):
        if category != current_category:
            lines.append(f"  // === {category} ===")
            current_category = category
        lines.append(f"  // {doc.type_hint}: {doc.description}")
        if doc.valid_values:
            lines.append(f"  //   valid: {doc.valid_values}")
        rendered = _format_default(value)
        indented = "\n".join("  " + line for line in rendered.splitlines())
        suffix = "" if idx == last_index else ","
        lines.append(f'  "{key}": {indented.lstrip()}{suffix}')

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_template(target_path: str | Path) -> None:
    """Write the generated template to *target_path*, overwriting any existing file."""
    Path(target_path).write_text(generate_template(), encoding="utf-8")
