"""Tests for the output saving module.

Acceptance criteria under test:
  AC1: Generated images are saved to {path_outputs}/{YYYY-MM-DD}/ directories
  AC2: Filenames follow the pattern {YYYY-MM-DD_HH-MM-SS}_{random}.{format}
  AC3: A log.html file is created/updated in each date folder with image entries
  AC4: log.html displays image thumbnails, metadata table, and copy-to-clipboard button
  AC5: PNG images embed metadata in PngInfo when enabled
  AC6: JPEG/WebP images embed metadata in EXIF when enabled
  AC7: Gallery receives correct file paths to display generated images
"""

from __future__ import annotations

import datetime
import os
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# The module under test — does not exist yet (RED phase).
from modules.output import generate_temp_filename, save_image, update_log_html

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    return tmp_path / "outputs"


@pytest.fixture()
def fixed_now() -> datetime.datetime:
    """A deterministic datetime for reproducible filename tests."""
    return datetime.datetime(2026, 4, 10, 14, 30, 45)


@pytest.fixture()
def sample_metadata() -> list[tuple[str, str, str]]:
    """Metadata triples as (label, key, value) for log display."""
    return [
        ("Prompt", "prompt", "a beautiful landscape"),
        ("Negative", "negative_prompt", "blurry"),
        ("Steps", "steps", "30"),
        ("Sampler", "sampler", "dpmpp_2m_sde_gpu"),
    ]


# ---------------------------------------------------------------------------
# AC1 + AC2: generate_temp_filename
# ---------------------------------------------------------------------------


class TestGenerateTempFilename:
    """generate_temp_filename creates date folders and timestamped filenames."""

    def test_returns_date_string_matching_today(self, output_dir: Path, fixed_now: datetime.datetime) -> None:
        with patch("modules.output.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fixed_now
            date_string, _, _ = generate_temp_filename(folder=str(output_dir), extension="png")
        assert date_string == "2026-04-10"

    def test_filepath_contains_date_subdirectory(self, output_dir: Path, fixed_now: datetime.datetime) -> None:
        with patch("modules.output.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fixed_now
            _, filepath, _ = generate_temp_filename(folder=str(output_dir), extension="png")
        assert f"{os.sep}2026-04-10{os.sep}" in filepath

    def test_filename_matches_timestamp_random_pattern(self, output_dir: Path, fixed_now: datetime.datetime) -> None:
        with patch("modules.output.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fixed_now
            _, _, only_name = generate_temp_filename(folder=str(output_dir), extension="png")
        # Pattern: YYYY-MM-DD_HH-MM-SS_{4-digit-random}.{ext}
        assert re.match(r"2026-04-10_14-30-45_\d{4}\.png$", only_name)

    def test_filepath_is_absolute(self, output_dir: Path, fixed_now: datetime.datetime) -> None:
        with patch("modules.output.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fixed_now
            _, filepath, _ = generate_temp_filename(folder=str(output_dir), extension="png")
        assert os.path.isabs(filepath)

    def test_extension_matches_requested_format(self, output_dir: Path, fixed_now: datetime.datetime) -> None:
        for ext in ("png", "jpeg", "webp"):
            with patch("modules.output.datetime") as mock_dt:
                mock_dt.datetime.now.return_value = fixed_now
                _, filepath, only_name = generate_temp_filename(folder=str(output_dir), extension=ext)
            assert filepath.endswith(f".{ext}")
            assert only_name.endswith(f".{ext}")


# ---------------------------------------------------------------------------
# AC3 + AC4: update_log_html
# ---------------------------------------------------------------------------


class TestUpdateLogHtml:
    """update_log_html creates and appends to log.html in the date folder."""

    def test_creates_log_html_file(self, output_dir: Path, sample_metadata: list[tuple[str, str, str]]) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        update_log_html(
            html_path=str(date_folder / "log.html"),
            image_filename="2026-04-10_14-30-45_1234.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )
        assert (date_folder / "log.html").exists()

    def test_log_html_contains_image_reference(
        self, output_dir: Path, sample_metadata: list[tuple[str, str, str]]
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        update_log_html(
            html_path=str(date_folder / "log.html"),
            image_filename="2026-04-10_14-30-45_1234.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )
        content = (date_folder / "log.html").read_text(encoding="utf-8")
        assert "2026-04-10_14-30-45_1234.png" in content

    def test_log_html_has_dark_theme_css(self, output_dir: Path, sample_metadata: list[tuple[str, str, str]]) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        update_log_html(
            html_path=str(date_folder / "log.html"),
            image_filename="test.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )
        content = (date_folder / "log.html").read_text(encoding="utf-8")
        assert "background-color: #121212" in content

    def test_log_html_contains_metadata_table(
        self, output_dir: Path, sample_metadata: list[tuple[str, str, str]]
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        update_log_html(
            html_path=str(date_folder / "log.html"),
            image_filename="test.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )
        content = (date_folder / "log.html").read_text(encoding="utf-8")
        assert "a beautiful landscape" in content
        assert "class='metadata'" in content

    def test_log_html_contains_copy_to_clipboard_button(
        self, output_dir: Path, sample_metadata: list[tuple[str, str, str]]
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        update_log_html(
            html_path=str(date_folder / "log.html"),
            image_filename="test.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )
        content = (date_folder / "log.html").read_text(encoding="utf-8")
        assert "to_clipboard" in content
        assert "<button" in content

    def test_log_html_prepends_new_entries(self, output_dir: Path, sample_metadata: list[tuple[str, str, str]]) -> None:
        """Newest image should appear before older images in log.html."""
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        html_path = str(date_folder / "log.html")

        update_log_html(
            html_path=html_path,
            image_filename="first_image.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )
        update_log_html(
            html_path=html_path,
            image_filename="second_image.png",
            metadata=sample_metadata,
            date_string="2026-04-10",
        )

        content = (date_folder / "log.html").read_text(encoding="utf-8")
        pos_second = content.index("second_image.png")
        pos_first = content.index("first_image.png")
        assert pos_second < pos_first, "Newest entry must appear first"


# ---------------------------------------------------------------------------
# AC5: PNG metadata embedding
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL/Pillow not installed")
class TestSaveImagePng:
    """save_image embeds PngInfo metadata in PNG files."""

    def test_saves_png_file_to_disk(
        self,
        output_dir: Path,
        sample_metadata: list[tuple[str, str, str]],
        fixed_now: datetime.datetime,
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "test.png")
        img = Image.new("RGB", (64, 64), color="red")

        result = save_image(
            image=img,
            filepath=filepath,
            output_format="png",
            metadata=sample_metadata,
            parsed_parameters="prompt: a beautiful landscape",
        )
        assert os.path.isfile(result)

    def test_png_contains_parameters_metadata(
        self,
        output_dir: Path,
        sample_metadata: list[tuple[str, str, str]],
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "test_meta.png")
        img = Image.new("RGB", (64, 64), color="blue")

        save_image(
            image=img,
            filepath=filepath,
            output_format="png",
            metadata=sample_metadata,
            parsed_parameters="prompt: a beautiful landscape",
        )

        saved = Image.open(filepath)
        assert "parameters" in saved.info
        assert "a beautiful landscape" in saved.info["parameters"]


# ---------------------------------------------------------------------------
# AC6: JPEG/WebP EXIF metadata embedding
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL/Pillow not installed")
class TestSaveImageJpeg:
    """save_image embeds EXIF metadata in JPEG files."""

    def test_saves_jpeg_file_to_disk(
        self,
        output_dir: Path,
        sample_metadata: list[tuple[str, str, str]],
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "test.jpeg")
        img = Image.new("RGB", (64, 64), color="green")

        result = save_image(
            image=img,
            filepath=filepath,
            output_format="jpeg",
            metadata=sample_metadata,
            parsed_parameters="prompt: a beautiful landscape",
        )
        assert os.path.isfile(result)

    def test_jpeg_contains_exif_data(
        self,
        output_dir: Path,
        sample_metadata: list[tuple[str, str, str]],
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "test_exif.jpeg")
        img = Image.new("RGB", (64, 64), color="green")

        save_image(
            image=img,
            filepath=filepath,
            output_format="jpeg",
            metadata=sample_metadata,
            parsed_parameters="prompt: a beautiful landscape",
        )

        saved = Image.open(filepath)
        exif = saved.getexif()
        # EXIF tag 0x9286 (UserComment) or 0x010e (ImageDescription)
        # We check that some EXIF data exists
        assert len(exif) > 0, "JPEG should contain EXIF metadata"


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL/Pillow not installed")
class TestSaveImageWebp:
    """save_image embeds EXIF metadata in WebP files."""

    def test_saves_webp_file_to_disk(
        self,
        output_dir: Path,
        sample_metadata: list[tuple[str, str, str]],
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "test.webp")
        img = Image.new("RGB", (64, 64), color="yellow")

        result = save_image(
            image=img,
            filepath=filepath,
            output_format="webp",
            metadata=sample_metadata,
            parsed_parameters="prompt: a beautiful landscape",
        )
        assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# AC7: save_image returns correct file path for gallery
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL/Pillow not installed")
class TestSaveImageReturnPath:
    """save_image returns the absolute file path for gallery display."""

    def test_returns_absolute_path_of_saved_file(
        self,
        output_dir: Path,
        sample_metadata: list[tuple[str, str, str]],
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "gallery_test.png")
        img = Image.new("RGB", (64, 64), color="purple")

        result = save_image(
            image=img,
            filepath=filepath,
            output_format="png",
            metadata=sample_metadata,
            parsed_parameters="",
        )
        assert os.path.isabs(result)
        assert os.path.isfile(result)

    def test_save_without_metadata_still_saves_file(
        self,
        output_dir: Path,
    ) -> None:
        date_folder = output_dir / "2026-04-10"
        date_folder.mkdir(parents=True)
        filepath = str(date_folder / "no_meta.png")
        img = Image.new("RGB", (64, 64), color="white")

        result = save_image(
            image=img,
            filepath=filepath,
            output_format="png",
            metadata=[],
            parsed_parameters="",
        )
        assert os.path.isfile(result)
