"""Standalone SDXL styles module for UnFooocused.

Loads style definitions from JSON files in the sdxl_styles/ directory.
Each style maps a name to a (positive_prompt, negative_prompt) tuple.
The positive prompt may contain a ``{prompt}`` placeholder that
``apply_style`` replaces with the user's prompt text.

Exports:
    styles: dict[str, tuple[str, str]]  — all loaded styles
    legal_style_names: list[str]        — ordered list for the UI picker
    apply_style(style, positive) -> (list[str], list[str], bool)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_STYLES_DIR = Path(__file__).resolve().parent.parent / "sdxl_styles"

# Canonical load order — files listed here are sorted to the end so they
# override earlier entries when names collide after normalize_key().
_LOAD_ORDER_SUFFIX: list[str] = [
    "sdxl_styles_fooocus.json",
    "sdxl_styles_sai.json",
    "sdxl_styles_mre.json",
    "sdxl_styles_twri.json",
    "sdxl_styles_diva.json",
    "sdxl_styles_marc_k3nt3l.json",
]

# The virtual style name for the Fooocus V2 expansion prompt.
FOOOCUS_EXPANSION = "Fooocus V2"


def _normalize_key(raw: str) -> str:
    """Normalize a style name to Title Case with known acronym fixups."""
    key = raw.replace("-", " ")
    words = key.split(" ")
    words = [w[:1].upper() + w[1:].lower() for w in words]
    key = " ".join(words)
    key = key.replace("3d", "3D")
    key = key.replace("Sai", "SAI")
    key = key.replace("Mre", "MRE")
    key = key.replace("(s", "(S")
    return key


def _discover_style_files(directory: Path) -> list[str]:
    """Return JSON filenames in *directory*, ordered for deterministic loading."""
    if not directory.is_dir():
        logger.warning("Styles directory not found: %s", directory)
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".json"))
    # Move canonical files to the end so they are loaded last (wins on collision).
    for name in _LOAD_ORDER_SUFFIX:
        if name in files:
            files.remove(name)
            files.append(name)
    return files


def _load_styles(directory: Path) -> dict[str, tuple[str, str]]:
    """Load all style JSON files from *directory* into a name -> (pos, neg) dict."""
    result: dict[str, tuple[str, str]] = {}
    for filename in _discover_style_files(directory):
        filepath = directory / filename
        try:
            with open(filepath, encoding="utf-8") as fh:
                entries = json.load(fh)
            for entry in entries:
                name = _normalize_key(entry["name"])
                positive = entry.get("prompt", "")
                negative = entry.get("negative_prompt", "")
                result[name] = (positive, negative)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Failed to load style file %s: %s", filename, exc)
    return result


# ---------------------------------------------------------------------------
# Module-level state — loaded once at import time
# ---------------------------------------------------------------------------

styles: dict[str, tuple[str, str]] = _load_styles(_STYLES_DIR)

legal_style_names: list[str] = [FOOOCUS_EXPANSION, *styles.keys()]


def apply_style(style: str, positive: str) -> tuple[list[str], list[str], bool]:
    """Apply a named style to a user prompt.

    Args:
        style: A key from :data:`styles`.
        positive: The user's positive prompt text.

    Returns:
        A 3-tuple of:
        - positive prompt lines (list[str]) with ``{prompt}`` replaced
        - negative prompt lines (list[str])
        - whether the style's positive template contained ``{prompt}``

    Raises:
        KeyError: If *style* is not in :data:`styles`.
    """
    style_positive, style_negative = styles[style]
    has_placeholder = "{prompt}" in style_positive
    applied_positive = style_positive.replace("{prompt}", positive)
    return applied_positive.splitlines(), style_negative.splitlines(), has_placeholder
