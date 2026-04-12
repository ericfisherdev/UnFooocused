"""Tests for FwdFooocus-compatible HTML log writer.

Acceptance criteria under test (from UNF-47):
  AC1: After generation, a log.html file exists in the date-based output directory
  AC2: Log entries contain all metadata fields matching FwdFooocus format
  AC3: Existing FwdFooocus log entries are preserved when UnFooocused appends
  AC4: Copy-to-clipboard button works with URL-encoded JSON parameters
  AC5: Multiple generations on the same day append to the same log.html
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Module under test — does not exist yet (RED phase).
from modules.html_log_writer import (
    FOOOCUS_LOG_SPLIT_MARKER,
    LogEntry,
    write_log_entry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FOOOCUS_METADATA_FIELDS = (
    "Prompt",
    "Negative Prompt",
    "Styles",
    "Performance",
    "Steps",
    "Resolution",
    "Guidance Scale",
    "Sharpness",
    "ADM Guidance",
    "Base Model",
    "Refiner Model",
    "Refiner Switch",
    "Sampler",
    "Scheduler",
    "VAE",
    "Seed",
    "Version",
)


@pytest.fixture()
def date_folder(tmp_path: Path) -> Path:
    """Provide a temporary date-based output directory."""
    folder = tmp_path / "outputs" / "2026-04-11"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture()
def sample_entry() -> LogEntry:
    """A LogEntry with all FwdFooocus-compatible metadata fields populated."""
    return LogEntry(
        image_filename="2026-04-11_14-30-45_1234.png",
        date_string="2026-04-11",
        metadata=[
            ("Prompt", "prompt", "a beautiful landscape"),
            ("Negative Prompt", "negative_prompt", "blurry, low quality"),
            ("Styles", "styles", "['Fooocus V2', 'Fooocus Enhance']"),
            ("Performance", "performance", "Speed"),
            ("Steps", "steps", "30"),
            ("Resolution", "resolution", "(1024, 1024)"),
            ("Guidance Scale", "guidance_scale", "7.0"),
            ("Sharpness", "sharpness", "2.0"),
            ("ADM Guidance", "adm_guidance", "(1.5, 0.8, 0.3)"),
            ("Base Model", "base_model", "juggernautXL_v8Rundiffusion.safetensors"),
            ("Refiner Model", "refiner_model", "None"),
            ("Refiner Switch", "refiner_switch", "0.5"),
            ("Sampler", "sampler", "dpmpp_2m_sde_gpu"),
            ("Scheduler", "scheduler", "karras"),
            ("VAE", "vae", "Default (model)"),
            ("Seed", "seed", "12345"),
            ("Version", "version", "UnFooocused v0.1.0"),
        ],
    )


@pytest.fixture()
def fwdfooocus_log_html() -> str:
    """A minimal but valid FwdFooocus-generated log.html file."""
    return (
        "<!DOCTYPE html><html><head><title>FwdFooocus Log 2026-04-11</title>"
        "<style>body { background-color: #121212; color: #E0E0E0; }</style>"
        "</head><body>"
        "<script>function to_clipboard(txt) { alert(txt); }</script>"
        "<p>FwdFooocus Log 2026-04-11 (private)</p>\n"
        "<p>Metadata is embedded if enabled in the config or developer debug mode."
        " You can find the information for each image in line Metadata Scheme.</p>"
        "<!--fooocus-log-split-->\n\n"
        '<div id="fwd_image_png" class="image-container"><hr><table><tr>\n'
        '<td><a href="fwd_image.png" target="_blank">'
        "<img src='fwd_image.png' loading='lazy'/></a>"
        "<div>fwd_image.png</div></td>"
        "<td><table class='metadata'>"
        "<tr><td class='label'>Prompt</td><td class='value'>fwd prompt</td></tr>\n"
        "<tr><td class='label'>Version</td>"
        "<td class='value'>FwdFooocus v0.0.1</td></tr>\n"
        "</table></td></tr></table></div>\n\n"
        "\n<!--fooocus-log-split--></body></html>"
    )


# ---------------------------------------------------------------------------
# AC1: log.html file exists in date-based output directory after write
# ---------------------------------------------------------------------------


class TestLogFileCreation:
    """AC1: After generation, a log.html file exists in the date-based output directory."""

    def test_creates_log_html_in_date_folder(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        assert html_path.exists()

    def test_creates_valid_html_document(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content


# ---------------------------------------------------------------------------
# AC2: Log entries contain all metadata fields matching FwdFooocus format
# ---------------------------------------------------------------------------


class TestMetadataFields:
    """AC2: Log entries contain all metadata fields matching FwdFooocus format."""

    def test_entry_contains_all_fooocus_metadata_labels(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        for label in _FOOOCUS_METADATA_FIELDS:
            assert label in content, f"Missing metadata label: {label}"

    def test_entry_has_image_thumbnail_link(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert f'href="{sample_entry.image_filename}"' in content
        assert f"src='{sample_entry.image_filename}'" in content

    def test_entry_has_metadata_table_with_css_classes(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert "class='metadata'" in content
        assert "class='label'" in content
        assert "class='value'" in content

    def test_version_field_shows_unfooocused(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert "UnFooocused" in content

    def test_uses_fooocus_log_split_marker(self, date_folder: Path, sample_entry: LogEntry) -> None:
        """Must use <!--fooocus-log-split--> for cross-compatibility."""
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert FOOOCUS_LOG_SPLIT_MARKER in content
        assert content.count(FOOOCUS_LOG_SPLIT_MARKER) == 2

    def test_has_dark_theme_css(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert "background-color: #121212" in content

    def test_has_clipboard_javascript(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert "function to_clipboard" in content


# ---------------------------------------------------------------------------
# AC3: Existing FwdFooocus log entries are preserved
# ---------------------------------------------------------------------------


class TestFwdFooocusPreservation:
    """AC3: Existing FwdFooocus log entries are preserved when UnFooocused appends."""

    def test_preserves_fwdfooocus_entries(
        self,
        date_folder: Path,
        sample_entry: LogEntry,
        fwdfooocus_log_html: str,
    ) -> None:
        html_path = date_folder / "log.html"
        html_path.write_text(fwdfooocus_log_html, encoding="utf-8")

        write_log_entry(html_path=str(html_path), entry=sample_entry)

        content = html_path.read_text(encoding="utf-8")
        assert "fwd_image.png" in content, "FwdFooocus image entry was lost"
        assert "fwd prompt" in content, "FwdFooocus metadata was lost"
        assert "FwdFooocus v0.0.1" in content, "FwdFooocus version was lost"

    def test_unfooocused_entry_prepended_before_fwdfooocus(
        self,
        date_folder: Path,
        sample_entry: LogEntry,
        fwdfooocus_log_html: str,
    ) -> None:
        html_path = date_folder / "log.html"
        html_path.write_text(fwdfooocus_log_html, encoding="utf-8")

        write_log_entry(html_path=str(html_path), entry=sample_entry)

        content = html_path.read_text(encoding="utf-8")
        pos_new = content.index(sample_entry.image_filename)
        pos_old = content.index("fwd_image.png")
        assert pos_new < pos_old, "New entry must appear before existing FwdFooocus entries"

    def test_result_still_has_two_split_markers(
        self,
        date_folder: Path,
        sample_entry: LogEntry,
        fwdfooocus_log_html: str,
    ) -> None:
        html_path = date_folder / "log.html"
        html_path.write_text(fwdfooocus_log_html, encoding="utf-8")

        write_log_entry(html_path=str(html_path), entry=sample_entry)

        content = html_path.read_text(encoding="utf-8")
        assert content.count(FOOOCUS_LOG_SPLIT_MARKER) == 2


# ---------------------------------------------------------------------------
# AC4: Copy-to-clipboard button works with URL-encoded JSON parameters
# ---------------------------------------------------------------------------


class TestCopyToClipboard:
    """AC4: Copy-to-clipboard button works with URL-encoded JSON parameters."""

    def test_button_element_present(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")
        assert "<button" in content
        assert "Copy to Clipboard" in content

    def test_button_contains_url_encoded_json(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")

        # Extract the URL-encoded string from to_clipboard('...')

        match = re.search(r"to_clipboard\('([^']+)'\)", content)
        assert match is not None, "to_clipboard call not found in HTML"

        encoded_json = match.group(1)
        decoded_json = urllib.parse.unquote(encoded_json)
        params = json.loads(decoded_json)

        assert params["prompt"] == "a beautiful landscape"
        assert params["seed"] == "12345"

    def test_all_metadata_keys_in_clipboard_json(self, date_folder: Path, sample_entry: LogEntry) -> None:
        html_path = date_folder / "log.html"
        write_log_entry(html_path=str(html_path), entry=sample_entry)
        content = html_path.read_text(encoding="utf-8")

        match = re.search(r"to_clipboard\('([^']+)'\)", content)
        assert match is not None
        params = json.loads(urllib.parse.unquote(match.group(1)))

        expected_keys = {k for _, k, _ in sample_entry.metadata}
        assert set(params.keys()) == expected_keys


# ---------------------------------------------------------------------------
# AC5: Multiple generations on the same day append to the same log.html
# ---------------------------------------------------------------------------


class TestMultipleGenerations:
    """AC5: Multiple generations on the same day append to the same log.html."""

    def test_two_entries_in_same_file(self, date_folder: Path) -> None:
        html_path = date_folder / "log.html"

        entry1 = LogEntry(
            image_filename="first.png",
            date_string="2026-04-11",
            metadata=[
                ("Prompt", "prompt", "first prompt"),
                ("Seed", "seed", "111"),
                ("Version", "version", "UnFooocused v0.1.0"),
            ],
        )
        entry2 = LogEntry(
            image_filename="second.png",
            date_string="2026-04-11",
            metadata=[
                ("Prompt", "prompt", "second prompt"),
                ("Seed", "seed", "222"),
                ("Version", "version", "UnFooocused v0.1.0"),
            ],
        )

        write_log_entry(html_path=str(html_path), entry=entry1)
        write_log_entry(html_path=str(html_path), entry=entry2)

        content = html_path.read_text(encoding="utf-8")
        assert "first.png" in content
        assert "second.png" in content

    def test_newest_entry_appears_first(self, date_folder: Path) -> None:
        html_path = date_folder / "log.html"

        entry1 = LogEntry(
            image_filename="older.png",
            date_string="2026-04-11",
            metadata=[("Prompt", "prompt", "old"), ("Version", "version", "v1")],
        )
        entry2 = LogEntry(
            image_filename="newer.png",
            date_string="2026-04-11",
            metadata=[("Prompt", "prompt", "new"), ("Version", "version", "v1")],
        )

        write_log_entry(html_path=str(html_path), entry=entry1)
        write_log_entry(html_path=str(html_path), entry=entry2)

        content = html_path.read_text(encoding="utf-8")
        assert content.index("newer.png") < content.index("older.png")

    def test_three_entries_all_preserved(self, date_folder: Path) -> None:
        html_path = date_folder / "log.html"

        for i in range(3):
            entry = LogEntry(
                image_filename=f"img_{i}.png",
                date_string="2026-04-11",
                metadata=[
                    ("Prompt", "prompt", f"prompt {i}"),
                    ("Version", "version", "v1"),
                ],
            )
            write_log_entry(html_path=str(html_path), entry=entry)

        content = html_path.read_text(encoding="utf-8")
        for i in range(3):
            assert f"img_{i}.png" in content
            assert f"prompt {i}" in content
