"""FwdFooocus-compatible HTML log writer.

Writes generation metadata to a log.html file using the same format as
FwdFooocus, so entries from both applications can coexist in the same file.

The HTML uses the ``<!--fooocus-log-split-->`` sentinel marker (matching
FwdFooocus) to delimit the header, entry content, and footer sections.

Domain concepts:
  - LogEntry: value object holding image filename, date, and metadata triples
  - write_log_entry: appends a new entry to an existing or new log.html
  - FOOOCUS_LOG_SPLIT_MARKER: the sentinel shared with FwdFooocus
"""

from __future__ import annotations

import html
import json
import logging
import os
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FOOOCUS_LOG_SPLIT_MARKER = "<!--fooocus-log-split-->"


@dataclass(frozen=True, slots=True)
class LogEntry:
    """Value object representing a single image generation log entry.

    Attributes:
        image_filename: Filename only (not full path) of the generated image.
        date_string: YYYY-MM-DD date string for the page title.
        metadata: List of (label, key, value) triples matching FwdFooocus format.
    """

    image_filename: str
    date_string: str
    metadata: list[tuple[str, str, str]]


# ---------------------------------------------------------------------------
# CSS and JS — identical to FwdFooocus for visual compatibility
# ---------------------------------------------------------------------------

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
    "button:hover {background-color: grey; color: black;}"
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
    alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.'
        + '\\nCurrent clipboard content is:\\n\\n' + txt);
}
</script>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_log_entry(*, html_path: str, entry: LogEntry) -> None:
    """Create or update a log.html file with a new image entry prepended.

    Uses the FwdFooocus ``<!--fooocus-log-split-->`` sentinel so that
    entries from both FwdFooocus and UnFooocused coexist in the same file.

    New entries are prepended (newest first) to match FwdFooocus behavior.

    Args:
        html_path: Absolute path to the log.html file.
        entry: LogEntry value object with image filename, date, and metadata.

    Domain errors:
        OSError: If the file cannot be written to disk.
    """
    begin_part = _build_header(entry.date_string)
    end_part = f"\n{FOOOCUS_LOG_SPLIT_MARKER}</body></html>"

    middle_part = _read_existing_middle(html_path)
    new_entry = _build_image_entry(entry.image_filename, entry.metadata)
    middle_part = new_entry + middle_part

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(begin_part + middle_part + end_part)

    logger.info("Log updated: %s", html_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_header(date_string: str) -> str:
    """Build the HTML header section with CSS, JS, and opening split marker."""
    return (
        f"<!DOCTYPE html><html><head><title>FwdFooocus Log {date_string}</title>"
        f"{_CSS_STYLES}</head><body>{_JS_CLIPBOARD}"
        f"<p>FwdFooocus Log {date_string} (private)</p>\n"
        f"<p>Metadata is embedded if enabled in the config or developer debug mode."
        f" You can find the information for each image in line Metadata Scheme.</p>"
        f"{FOOOCUS_LOG_SPLIT_MARKER}\n\n"
    )


def _read_existing_middle(html_path: str) -> str:
    """Extract the middle content section from an existing log.html, if any.

    Handles both FwdFooocus and UnFooocused log files by splitting on the
    shared ``<!--fooocus-log-split-->`` sentinel marker.
    """
    if not os.path.exists(html_path):
        return ""

    with open(html_path, encoding="utf-8") as fh:
        content = fh.read()

    parts = content.split(FOOOCUS_LOG_SPLIT_MARKER)
    if len(parts) == 3:
        return parts[1]
    return ""


def _build_image_entry(
    image_filename: str,
    metadata: list[tuple[str, str, str]],
) -> str:
    """Build the HTML fragment for a single image entry.

    Matches FwdFooocus format: image thumbnail + metadata table + clipboard button.
    """
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
        escaped_label = html.escape(str(label))
        value_txt = html.escape(str(value)).replace("\n", " </br> ")
        parts.append(f"<tr><td class='label'>{escaped_label}</td><td class='value'>{value_txt}</td></tr>\n")

    parts.append("</table>")

    js_txt = urllib.parse.quote(
        json.dumps({k: v for _, k, v in metadata}, indent=0),
        safe="",
    )
    parts.append(f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>")
    parts.append("</td></tr></table></div>\n\n")

    return "".join(parts)
