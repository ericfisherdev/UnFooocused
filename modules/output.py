"""Output saving module for UnFooocused.

Saves generated images to date-based folders and maintains a log.html
viewer with thumbnails, metadata tables, and copy-to-clipboard support.

Domain concepts:
  - generate_temp_filename: creates a unique filepath under {output_dir}/{YYYY-MM-DD}/
  - save_image: persists a PIL Image with format-appropriate metadata embedding
  - update_log_html: creates/updates a self-contained HTML log in the date folder
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import urllib.parse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename generation (AC1, AC2)
# ---------------------------------------------------------------------------


def generate_temp_filename(
    folder: str = "./outputs/",
    extension: str = "png",
) -> tuple[str, str, str]:
    """Generate a unique timestamped filename inside a date-based subfolder.

    Args:
        folder: Base output directory.
        extension: File extension without dot (png, jpeg, webp).

    Returns:
        Tuple of (date_string, absolute_filepath, filename_only).
    """
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    random_number = random.randint(1000, 9999)  # noqa: S311  # nosec B311
    filename = f"{time_string}_{random_number}.{extension}"
    result = os.path.join(folder, date_string, filename)
    return date_string, os.path.abspath(result), filename


# ---------------------------------------------------------------------------
# Image saving with metadata (AC5, AC6, AC7)
# ---------------------------------------------------------------------------


def save_image(
    *,
    image: Image,
    filepath: str,
    output_format: str,
    metadata: list[tuple[str, str, str]],
    parsed_parameters: str,
) -> str:
    """Save a PIL Image to disk with format-appropriate metadata embedding.

    Args:
        image: PIL Image object to save.
        filepath: Absolute destination path.
        output_format: One of "png", "jpeg", "webp".
        metadata: List of (label, key, value) triples for log display.
        parsed_parameters: Serialized metadata string to embed in the file.

    Returns:
        The absolute filepath of the saved image.

    Domain errors:
        OSError: If the file cannot be written to disk.
    """
    if output_format == "png":
        _save_png(image, filepath, parsed_parameters)
    elif output_format == "jpeg":
        _save_jpeg(image, filepath, parsed_parameters)
    elif output_format == "webp":
        _save_webp(image, filepath, parsed_parameters)
    else:
        image.save(filepath)

    logger.info("Image saved: %s", filepath)
    return os.path.abspath(filepath)


def _save_png(image: Image, filepath: str, parsed_parameters: str) -> None:
    """Save PNG with optional PngInfo metadata."""
    from PIL.PngImagePlugin import PngInfo

    pnginfo = None
    if parsed_parameters:
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", parsed_parameters)

    image.save(filepath, pnginfo=pnginfo)


def _save_jpeg(image: Image, filepath: str, parsed_parameters: str) -> None:
    """Save JPEG with optional EXIF metadata."""
    exif_bytes = _build_exif(parsed_parameters) if parsed_parameters else b""
    if exif_bytes:
        image.save(filepath, quality=95, optimize=True, progressive=True, exif=exif_bytes)
    else:
        image.save(filepath, quality=95, optimize=True, progressive=True)


def _save_webp(image: Image, filepath: str, parsed_parameters: str) -> None:
    """Save WebP with optional EXIF metadata."""
    exif_bytes = _build_exif(parsed_parameters) if parsed_parameters else b""
    if exif_bytes:
        image.save(filepath, quality=95, lossless=False, exif=exif_bytes)
    else:
        image.save(filepath, quality=95, lossless=False)


def _build_exif(parsed_parameters: str) -> bytes:
    """Build EXIF bytes with parameters stored in ImageDescription (tag 0x010E).

    Uses PIL's built-in Exif class — no external ``piexif`` dependency.

    Returns:
        EXIF bytes suitable for PIL's ``exif=`` parameter, or empty bytes
        if the parameters string is empty.
    """
    if not parsed_parameters:
        return b""

    from PIL.Image import Exif

    exif = Exif()
    # 0x010E = ImageDescription
    exif[0x010E] = parsed_parameters
    return exif.tobytes()


# ---------------------------------------------------------------------------
# log.html generation (AC3, AC4)
# ---------------------------------------------------------------------------

_LOG_SPLIT_MARKER = "<!--unfooocused-log-split-->"

_CSS_STYLES = (
    "<style>"
    "body { background-color: #121212; color: #E0E0E0; } "
    "a { color: #BB86FC; } "
    ".metadata { border-collapse: collapse; width: 100%; } "
    ".metadata .label { width: 15%; } "
    ".metadata .value { width: 85%; font-weight: bold; } "
    ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
    ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
    ".image-container div { text-align: center; padding: 4px; } "
    "hr { border-color: gray; } "
    "button { background-color: black; color: white; border: 1px solid grey; "
    "border-radius: 5px; padding: 5px 10px; text-align: center; "
    "display: inline-block; font-size: 16px; cursor: pointer; }"
    "button:hover { background-color: grey; color: black; }"
    "</style>"
)

_JS_CLIPBOARD = """<script>
function to_clipboard(txt) {
    txt = decodeURIComponent(txt);
    if (navigator.clipboard && navigator.permissions) {
        navigator.clipboard.writeText(txt);
    } else {
        const textArea = document.createElement('textArea');
        textArea.value = txt;
        textArea.style.width = 0;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999px';
        textArea.style.top = '10px';
        textArea.setAttribute('readonly', 'readonly');
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
    }
    alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\n\\n' + txt);
}
</script>"""


def update_log_html(
    *,
    html_path: str,
    image_filename: str,
    metadata: list[tuple[str, str, str]],
    date_string: str,
) -> None:
    """Create or update a log.html file with a new image entry prepended.

    The HTML is self-contained with inline CSS (dark theme) and JS
    (copy-to-clipboard). New entries appear at the top so the most
    recent image is always first.

    Args:
        html_path: Absolute path to the log.html file.
        image_filename: Filename only (not full path) of the image.
        metadata: List of (label, key, value) triples.
        date_string: The YYYY-MM-DD date string for the page title.
    """
    begin_part = (
        f"<!DOCTYPE html><html><head><title>UnFooocused Log {date_string}</title>"
        f"{_CSS_STYLES}</head><body>{_JS_CLIPBOARD}"
        f"<p>UnFooocused Log {date_string}</p>\n"
        f"<p>Metadata is embedded if enabled in config.</p>"
        f"{_LOG_SPLIT_MARKER}\n\n"
    )
    end_part = f"\n{_LOG_SPLIT_MARKER}</body></html>"

    middle_part = _read_existing_middle(html_path)
    new_entry = _build_image_entry(image_filename, metadata)
    middle_part = new_entry + middle_part

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(begin_part + middle_part + end_part)

    logger.info("Log updated: %s", html_path)


def _read_existing_middle(html_path: str) -> str:
    """Extract the middle content section from an existing log.html, if any."""
    if not os.path.exists(html_path):
        return ""

    with open(html_path, encoding="utf-8") as fh:
        content = fh.read()

    parts = content.split(_LOG_SPLIT_MARKER)
    if len(parts) == 3:
        return parts[1]
    return ""


def _build_image_entry(
    image_filename: str,
    metadata: list[tuple[str, str, str]],
) -> str:
    """Build the HTML fragment for a single image entry."""
    div_name = image_filename.replace(".", "_")
    parts: list[str] = [
        f'<div id="{div_name}" class="image-container"><hr><table><tr>\n',
        f'<td><a href="{image_filename}" target="_blank">'
        f"<img src='{image_filename}' "
        f"onerror=\"this.closest('.image-container').style.display='none';\" "
        f"loading='lazy'/></a><div>{image_filename}</div></td>",
        "<td><table class='metadata'>",
    ]

    for label, _key, value in metadata:
        value_txt = str(value).replace("\n", " </br> ")
        parts.append(f"<tr><td class='label'>{label}</td><td class='value'>{value_txt}</td></tr>\n")

    parts.append("</table>")

    js_txt = urllib.parse.quote(
        json.dumps({k: v for _, k, v in metadata}, indent=0),
        safe="",
    )
    parts.append(f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>")
    parts.append("</td></tr></table></div>\n\n")

    return "".join(parts)
