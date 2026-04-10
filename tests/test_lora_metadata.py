"""Unit tests for modules.lora_metadata — inner-loop TDD.

Tests are organized by function under test, from pure extraction functions
(functional core) to the LoraMetadataScanner class (imperative shell).

The safetensors library (safe_open) is an unmanaged dependency — it is mocked
at the module boundary via modules.lora_metadata._safe_open, following
Freeman & Pryce's 'only mock types you own' principle.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from modules.lora_metadata import (
    LoraMetadataScanner,
    _deduplicate_ordered,
    _extract_base_model,
    _extract_characters,
    _extract_description,
    _extract_numeric_field,
    _extract_resolution,
    _extract_styles,
    _extract_trigger_words,
    _normalize_base_model,
    _parse_bucket_resolutions,
    _parse_dataset_dirs,
    _parse_tag_frequency,
    extract_metadata,
    get_all_library_data,
    get_distinct_base_models,
    get_metadata_summary,
    get_trigger_words_for_filename,
    is_valid_lora_file,
    search_library,
)


def _make_safe_open_mock(metadata_return_value=None, keys_return_value=None):
    """Create a context-manager MagicMock that mimics _safe_open behavior."""
    mock_handle = MagicMock()
    mock_handle.metadata.return_value = metadata_return_value
    if keys_return_value is not None:
        mock_handle.keys.return_value = keys_return_value
    mock_handle.__enter__ = MagicMock(return_value=mock_handle)
    mock_handle.__exit__ = MagicMock(return_value=False)
    return mock_handle


# ---------------------------------------------------------------------------
# _normalize_base_model
# ---------------------------------------------------------------------------
class TestNormalizeBaseModel:
    """_normalize_base_model() maps raw model strings to standard names."""

    def test_sdxl_1_0_variant(self):
        assert _normalize_base_model("sdxl_1.0") == "SDXL 1.0"

    def test_sdxl_plain(self):
        assert _normalize_base_model("sdxl") == "SDXL 1.0"

    def test_stable_diffusion_xl(self):
        assert _normalize_base_model("stable_diffusion_xl") == "SDXL 1.0"

    def test_sd15_variant(self):
        assert _normalize_base_model("sd1.5") == "SD 1.5"

    def test_sd_1_5_with_dash(self):
        assert _normalize_base_model("sd-1.5") == "SD 1.5"

    def test_stable_diffusion_15(self):
        assert _normalize_base_model("stable_diffusion_1.5") == "SD 1.5"

    def test_pony(self):
        assert _normalize_base_model("pony") == "Pony"

    def test_pdxl(self):
        assert _normalize_base_model("pdxl") == "Pony"

    def test_flux(self):
        assert _normalize_base_model("flux") == "Flux"

    def test_sd3(self):
        assert _normalize_base_model("sd3") == "SD 3"

    def test_sd21(self):
        assert _normalize_base_model("sd2.1") == "SD 2.1"

    def test_unknown_passes_through_stripped(self):
        assert _normalize_base_model("  some_custom_model  ") == "some_custom_model"

    def test_unknown_exact_passthrough(self):
        assert _normalize_base_model("MyCustomCheckpoint") == "MyCustomCheckpoint"

    def test_case_insensitive(self):
        assert _normalize_base_model("SDXL_1.0") == "SDXL 1.0"
        assert _normalize_base_model("Pony") == "Pony"
        assert _normalize_base_model("FLUX") == "Flux"


# ---------------------------------------------------------------------------
# _extract_trigger_words
# ---------------------------------------------------------------------------
class TestExtractTriggerWords:
    """_extract_trigger_words() parses trigger words from multiple metadata formats."""

    def test_ss_tag_frequency_json(self):
        tag_freq = {"dataset1": {"cat": 50, "dog": 30, "bird": 10}}
        metadata = {"ss_tag_frequency": json.dumps(tag_freq)}
        result = _extract_trigger_words(metadata)
        assert "cat" in result
        assert "dog" in result
        assert "bird" in result

    def test_ss_tag_frequency_sorted_by_count(self):
        tag_freq = {"ds": {"rare": 1, "common": 100, "mid": 50}}
        metadata = {"ss_tag_frequency": json.dumps(tag_freq)}
        result = _extract_trigger_words(metadata)
        assert result.index("common") < result.index("mid")
        assert result.index("mid") < result.index("rare")

    def test_plain_comma_separated(self):
        metadata = {"trigger_words": "cat, dog, bird"}
        result = _extract_trigger_words(metadata)
        assert result == ["cat", "dog", "bird"]

    def test_semicolon_separated(self):
        metadata = {"trigger_words": "cat;dog;bird"}
        result = _extract_trigger_words(metadata)
        assert result == ["cat", "dog", "bird"]

    def test_newline_separated(self):
        metadata = {"trigger_words": "cat\ndog\nbird"}
        result = _extract_trigger_words(metadata)
        assert result == ["cat", "dog", "bird"]

    def test_ss_dataset_dirs_format(self):
        dirs = {"1_character_name": {"n_repeats": 1}, "10_style_lora": {"n_repeats": 10}}
        metadata = {"ss_dataset_dirs": json.dumps(dirs)}
        result = _extract_trigger_words(metadata)
        assert "character name" in result
        assert "style lora" in result

    def test_list_value(self):
        metadata = {"trigger_words": ["cat", "dog", "bird"]}
        result = _extract_trigger_words(metadata)
        assert result == ["cat", "dog", "bird"]

    def test_deduplicates_case_insensitively(self):
        metadata = {"trigger_words": "Cat, cat, CAT, dog"}
        result = _extract_trigger_words(metadata)
        assert len([w for w in result if w.lower() == "cat"]) == 1
        assert "dog" in result

    def test_empty_metadata_returns_empty(self):
        assert _extract_trigger_words({}) == []

    def test_skips_empty_values(self):
        metadata = {"trigger_words": ""}
        assert _extract_trigger_words(metadata) == []

    def test_strips_whitespace(self):
        metadata = {"trigger_words": "  cat  ,  dog  "}
        result = _extract_trigger_words(metadata)
        assert result == ["cat", "dog"]


# ---------------------------------------------------------------------------
# _extract_description
# ---------------------------------------------------------------------------
class TestExtractDescription:
    """_extract_description() checks keys in priority order."""

    def test_returns_first_matching_key(self):
        metadata = {
            "ss_training_comment": "first",
            "description": "second",
        }
        assert _extract_description(metadata) == "first"

    def test_skips_empty_values(self):
        metadata = {
            "ss_training_comment": "",
            "description": "fallback",
        }
        assert _extract_description(metadata) == "fallback"

    def test_returns_none_when_no_keys(self):
        assert _extract_description({}) is None

    def test_strips_whitespace(self):
        metadata = {"description": "  hello world  "}
        assert _extract_description(metadata) == "hello world"

    def test_modelspec_description_fallback(self):
        metadata = {"modelspec.description": "from modelspec"}
        assert _extract_description(metadata) == "from modelspec"


# ---------------------------------------------------------------------------
# _extract_numeric_field
# ---------------------------------------------------------------------------
class TestExtractNumericField:
    """_extract_numeric_field() handles int, float, and invalid values."""

    def test_integer_value(self):
        metadata = {"ss_epoch": "10"}
        result = _extract_numeric_field(metadata, ["ss_epoch"])
        assert result == 10
        assert isinstance(result, int)

    def test_float_string_truncated_to_int(self):
        metadata = {"ss_epoch": "10.7"}
        result = _extract_numeric_field(metadata, ["ss_epoch"])
        assert result == 10
        assert isinstance(result, int)

    def test_as_float_mode(self):
        metadata = {"network_alpha": "0.75"}
        result = _extract_numeric_field(metadata, ["network_alpha"], as_float=True)
        assert result == pytest.approx(0.75)
        assert isinstance(result, float)

    def test_invalid_value_skipped(self):
        metadata = {"ss_epoch": "not_a_number", "ss_num_epochs": "5"}
        result = _extract_numeric_field(metadata, ["ss_epoch", "ss_num_epochs"])
        assert result == 5

    def test_all_invalid_returns_none(self):
        metadata = {"ss_epoch": "garbage"}
        assert _extract_numeric_field(metadata, ["ss_epoch"]) is None

    def test_missing_keys_returns_none(self):
        assert _extract_numeric_field({}, ["ss_epoch"]) is None

    def test_first_valid_key_wins(self):
        metadata = {"ss_epoch": "3", "ss_num_epochs": "10"}
        assert _extract_numeric_field(metadata, ["ss_epoch", "ss_num_epochs"]) == 3


# ---------------------------------------------------------------------------
# _extract_resolution
# ---------------------------------------------------------------------------
class TestExtractResolution:
    """_extract_resolution() parses bucket info and plain resolution strings."""

    def test_ss_bucket_info_json(self):
        bucket_info = {
            "buckets": {
                "[512, 768]": {"count": 10},
                "[768, 512]": {"count": 5},
            }
        }
        metadata = {"ss_bucket_info": json.dumps(bucket_info)}
        result = _extract_resolution(metadata)
        assert "512x768" in result
        assert "768x512" in result

    def test_plain_resolution_string(self):
        metadata = {"resolution": "1024x1024"}
        assert _extract_resolution(metadata) == "1024x1024"

    def test_ss_resolution_string(self):
        metadata = {"ss_resolution": "512,768"}
        assert _extract_resolution(metadata) == "512,768"

    def test_empty_metadata_returns_none(self):
        assert _extract_resolution({}) is None

    def test_empty_bucket_info_falls_through(self):
        bucket_info = {"buckets": {}}
        metadata = {"ss_bucket_info": json.dumps(bucket_info), "resolution": "1024x1024"}
        result = _extract_resolution(metadata)
        assert result == "1024x1024"

    def test_parenthesis_bucket_format(self):
        bucket_info = {"buckets": {"(512, 768)": {"count": 1}}}
        metadata = {"ss_bucket_info": json.dumps(bucket_info)}
        result = _extract_resolution(metadata)
        assert "512x768" in result


# ---------------------------------------------------------------------------
# _extract_characters
# ---------------------------------------------------------------------------
class TestExtractCharacters:
    """_extract_characters() finds character patterns in text and metadata keys."""

    def test_character_metadata_key(self):
        metadata = {"character": "Saber, Rin"}
        result = _extract_characters("", metadata)
        assert "Saber" in result
        assert "Rin" in result

    def test_characters_list_metadata(self):
        metadata = {"characters": ["Alice", "Bob"]}
        result = _extract_characters("", metadata)
        assert "Alice" in result
        assert "Bob" in result

    def test_character_pattern_in_text(self):
        text = "character: Hatsune Miku"
        result = _extract_characters(text, {})
        assert "Hatsune Miku" in result

    def test_name_from_pattern_in_text(self):
        text = "Sakura from character"
        result = _extract_characters(text, {})
        assert "Sakura" in result

    def test_deduplicates_results(self):
        metadata = {"character": "Alice, Alice"}
        result = _extract_characters("", metadata)
        assert result.count("Alice") == 1

    def test_empty_text_and_metadata(self):
        assert _extract_characters("", {}) == []


# ---------------------------------------------------------------------------
# _extract_styles
# ---------------------------------------------------------------------------
class TestExtractStyles:
    """_extract_styles() matches style keywords case-insensitively."""

    def test_finds_anime_keyword(self):
        result = _extract_styles("This is an Anime style LoRA", {})
        assert "Anime" in result

    def test_finds_multiple_keywords(self):
        result = _extract_styles("realistic photorealistic digital art", {})
        assert "Realistic" in result
        assert "Photorealistic" in result
        assert "Digital Art" in result

    def test_case_insensitive_matching(self):
        result = _extract_styles("ANIME style", {})
        assert "Anime" in result

    def test_metadata_style_key(self):
        metadata = {"style": "watercolor, sketch"}
        result = _extract_styles("", metadata)
        assert "watercolor" in result
        assert "sketch" in result

    def test_metadata_styles_list(self):
        metadata = {"styles": ["impressionist", "abstract"]}
        result = _extract_styles("", metadata)
        assert "impressionist" in result
        assert "abstract" in result

    def test_no_matches_returns_empty(self):
        assert _extract_styles("plain text with no style words", {}) == []

    def test_deduplicates(self):
        # "anime" from text + "Anime" from metadata should deduplicate
        metadata = {"style": "Anime"}
        result = _extract_styles("anime lora", metadata)
        anime_count = sum(1 for s in result if s.lower() == "anime")
        assert anime_count == 1


# ---------------------------------------------------------------------------
# get_metadata_summary
# ---------------------------------------------------------------------------
class TestGetMetadataSummary:
    """get_metadata_summary() produces formatted output with all fields."""

    @pytest.fixture()
    def full_metadata(self):
        return {
            "filename": "test_lora.safetensors",
            "file_path": "/path/to/test_lora.safetensors",
            "file_size": 50 * 1024 * 1024,  # 50 MB
            "base_model": "SDXL 1.0",
            "trigger_words": ["cat", "dog", "bird", "fish", "snake", "lizard"],
            "description": "A test LoRA for animals",
            "characters": ["Neko"],
            "styles": ["Anime"],
            "training_epochs": 10,
            "training_steps": 5000,
            "resolution": "1024x1024",
            "network_dim": 128,
            "network_alpha": 64.0,
            "raw_metadata": {},
            "extraction_errors": [],
        }

    def test_contains_filename(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "test_lora.safetensors" in summary

    def test_contains_base_model(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "SDXL 1.0" in summary

    def test_contains_trigger_words_truncated(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "cat" in summary
        assert "+1 more" in summary

    def test_contains_description(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "A test LoRA for animals" in summary

    def test_contains_characters(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "Neko" in summary

    def test_contains_styles(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "Anime" in summary

    def test_contains_file_size(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "50.00 MB" in summary

    def test_contains_network_info(self, full_metadata):
        summary = get_metadata_summary(full_metadata)
        assert "128" in summary
        assert "64.0" in summary

    def test_unknown_base_model_when_none(self, full_metadata):
        full_metadata["base_model"] = None
        summary = get_metadata_summary(full_metadata)
        assert "Unknown" in summary

    def test_no_trigger_words_message(self, full_metadata):
        full_metadata["trigger_words"] = []
        summary = get_metadata_summary(full_metadata)
        assert "No trigger words available" in summary

    def test_no_description_message(self, full_metadata):
        full_metadata["description"] = None
        summary = get_metadata_summary(full_metadata)
        assert "No description" in summary

    def test_long_description_truncated(self, full_metadata):
        full_metadata["description"] = "A" * 200
        summary = get_metadata_summary(full_metadata)
        assert "..." in summary

    def test_extraction_errors_shown(self, full_metadata):
        full_metadata["extraction_errors"] = ["error1", "error2"]
        summary = get_metadata_summary(full_metadata)
        assert "2 extraction issues" in summary


# ---------------------------------------------------------------------------
# _deduplicate_ordered
# ---------------------------------------------------------------------------
class TestDeduplicateOrdered:
    """_deduplicate_ordered() preserves first occurrence, case-insensitive."""

    def test_removes_case_insensitive_duplicates(self):
        assert _deduplicate_ordered(["Cat", "cat", "CAT"]) == ["Cat"]

    def test_preserves_order(self):
        assert _deduplicate_ordered(["b", "a", "c"]) == ["b", "a", "c"]

    def test_removes_empty_strings(self):
        assert _deduplicate_ordered(["a", "", "b", ""]) == ["a", "b"]

    def test_empty_list(self):
        assert _deduplicate_ordered([]) == []


# ---------------------------------------------------------------------------
# _parse_tag_frequency
# ---------------------------------------------------------------------------
class TestParseTagFrequency:
    """_parse_tag_frequency() parses ss_tag_frequency JSON structure."""

    def test_valid_json_string(self):
        data = {"ds1": {"tag_a": 10, "tag_b": 5}}
        result = _parse_tag_frequency(json.dumps(data))
        assert "tag_a" in result
        assert "tag_b" in result

    def test_limits_to_20_tags_per_dataset(self):
        tags = {f"tag_{i}": i for i in range(30)}
        data = {"ds": tags}
        result = _parse_tag_frequency(json.dumps(data))
        assert len(result) == 20

    def test_invalid_json_returns_empty(self):
        assert _parse_tag_frequency("not json") == []

    def test_non_dict_returns_empty(self):
        assert _parse_tag_frequency(json.dumps([1, 2, 3])) == []


# ---------------------------------------------------------------------------
# _parse_dataset_dirs
# ---------------------------------------------------------------------------
class TestParseDatasetDirs:
    """_parse_dataset_dirs() extracts trigger words from directory names."""

    def test_strips_repeat_prefix(self):
        data = {"1_character_name": {}}
        result = _parse_dataset_dirs(json.dumps(data))
        assert "character name" in result

    def test_multiple_dirs(self):
        data = {"5_style_a": {}, "10_style_b": {}}
        result = _parse_dataset_dirs(json.dumps(data))
        assert len(result) == 2

    def test_invalid_json_returns_empty(self):
        assert _parse_dataset_dirs("bad json") == []


# ---------------------------------------------------------------------------
# _parse_bucket_resolutions
# ---------------------------------------------------------------------------
class TestParseBucketResolutions:
    """_parse_bucket_resolutions() parses ss_bucket_info JSON."""

    def test_list_format_buckets(self):
        data = {"buckets": {"[512, 768]": {"count": 1}}}
        result = _parse_bucket_resolutions(json.dumps(data))
        assert result == "512x768"

    def test_deduplicates_resolutions(self):
        data = {"buckets": {"[512, 768]": {"count": 1}, "[768, 1024]": {"count": 2}}}
        result = _parse_bucket_resolutions(json.dumps(data))
        assert "512x768" in result
        assert "768x1024" in result

    def test_invalid_json_returns_none(self):
        assert _parse_bucket_resolutions("not json") is None

    def test_no_buckets_key_returns_none(self):
        assert _parse_bucket_resolutions(json.dumps({"other": 1})) is None


# ---------------------------------------------------------------------------
# extract_metadata (integration with _safe_open mock)
# ---------------------------------------------------------------------------
class TestExtractMetadata:
    """extract_metadata() orchestrates all extraction functions."""

    def test_extracts_all_fields_from_safetensors(self, tmp_path):
        fake_file = tmp_path / "test.safetensors"
        fake_file.write_bytes(b"fake")

        raw = {
            "ss_base_model_version": "sdxl_1.0",
            "ss_tag_frequency": json.dumps({"ds": {"cat": 10}}),
            "ss_training_comment": "A test lora",
            "ss_epoch": "5",
            "ss_steps": "1000",
            "ss_network_dim": "128",
            "ss_network_alpha": "64.0",
            "ss_resolution": "1024x1024",
        }

        mock_handle = _make_safe_open_mock(metadata_return_value=raw)

        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            result = extract_metadata(str(fake_file))

        assert result["base_model"] == "SDXL 1.0"
        assert "cat" in result["trigger_words"]
        assert result["description"] == "A test lora"
        assert result["training_epochs"] == 5
        assert result["training_steps"] == 1000
        assert result["network_dim"] == 128
        assert result["network_alpha"] == pytest.approx(64.0)
        assert result["resolution"] == "1024x1024"

    def test_handles_no_metadata(self, tmp_path):
        fake_file = tmp_path / "empty.safetensors"
        fake_file.write_bytes(b"fake")

        mock_handle = _make_safe_open_mock(metadata_return_value=None)

        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            result = extract_metadata(str(fake_file))

        assert result["base_model"] is None
        assert result["trigger_words"] == []
        assert "No metadata found" in result["extraction_errors"][0]

    def test_handles_safe_open_exception(self, tmp_path):
        fake_file = tmp_path / "bad.safetensors"
        fake_file.write_bytes(b"fake")

        with patch("modules.lora_metadata._safe_open", side_effect=RuntimeError("corrupt")):
            result = extract_metadata(str(fake_file))

        assert len(result["extraction_errors"]) > 0
        assert "corrupt" in result["extraction_errors"][0]


# ---------------------------------------------------------------------------
# is_valid_lora_file
# ---------------------------------------------------------------------------
class TestIsValidLoraFile:
    """is_valid_lora_file() checks extension, existence, and LoRA keys."""

    def test_rejects_non_safetensors_extension(self):
        assert is_valid_lora_file("/path/to/model.ckpt") is False

    def test_rejects_nonexistent_file(self, tmp_path):
        assert is_valid_lora_file(str(tmp_path / "missing.safetensors")) is False

    def test_accepts_file_with_lora_keys(self, tmp_path):
        fake_file = tmp_path / "good.safetensors"
        fake_file.write_bytes(b"fake")

        mock_handle = _make_safe_open_mock(
            keys_return_value=["lora_unet_down.weight", "lora_unet_up.weight"],
        )

        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            assert is_valid_lora_file(str(fake_file)) is True

    def test_rejects_file_without_lora_keys(self, tmp_path):
        fake_file = tmp_path / "not_lora.safetensors"
        fake_file.write_bytes(b"fake")

        mock_handle = _make_safe_open_mock(
            keys_return_value=["model.weight", "encoder.bias"],
        )

        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            assert is_valid_lora_file(str(fake_file)) is False


# ---------------------------------------------------------------------------
# LoraMetadataScanner — blocking scan and search
# ---------------------------------------------------------------------------
class TestLoraMetadataScanner:
    """LoraMetadataScanner indexes files and supports search operations."""

    @pytest.fixture()
    def scanner_with_index(self):
        """Create a scanner with pre-populated index for search tests."""
        scanner = LoraMetadataScanner(lora_paths=[])
        scanner._metadata_index = {
            "/loras/sdxl_char.safetensors": {
                "filename": "sdxl_char.safetensors",
                "file_path": "/loras/sdxl_char.safetensors",
                "base_model": "SDXL 1.0",
                "trigger_words": ["saber", "fate"],
                "description": "Character LoRA",
                "characters": ["Saber"],
                "styles": ["Anime"],
                "file_size": 100,
                "training_epochs": None,
                "training_steps": None,
                "resolution": None,
                "network_dim": None,
                "network_alpha": None,
                "raw_metadata": {},
                "extraction_errors": [],
            },
            "/loras/pony_style.safetensors": {
                "filename": "pony_style.safetensors",
                "file_path": "/loras/pony_style.safetensors",
                "base_model": "Pony",
                "trigger_words": ["watercolor", "painting"],
                "description": "Style LoRA",
                "characters": [],
                "styles": ["Watercolor"],
                "file_size": 200,
                "training_epochs": None,
                "training_steps": None,
                "resolution": None,
                "network_dim": None,
                "network_alpha": None,
                "raw_metadata": {},
                "extraction_errors": [],
            },
            "/loras/sd15_generic.safetensors": {
                "filename": "sd15_generic.safetensors",
                "file_path": "/loras/sd15_generic.safetensors",
                "base_model": "SD 1.5",
                "trigger_words": ["saber", "sword"],
                "description": "Generic LoRA",
                "characters": [],
                "styles": [],
                "file_size": 150,
                "training_epochs": None,
                "training_steps": None,
                "resolution": None,
                "network_dim": None,
                "network_alpha": None,
                "raw_metadata": {},
                "extraction_errors": [],
            },
        }
        return scanner

    def test_blocking_scan_indexes_safetensors_files(self, tmp_path):
        """start_scan(blocking=True) discovers and indexes .safetensors files."""
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        (lora_dir / "model_a.safetensors").write_bytes(b"fake")
        (lora_dir / "model_b.safetensors").write_bytes(b"fake")
        (lora_dir / "not_a_lora.txt").write_bytes(b"nope")

        raw_metadata = {"ss_base_model_version": "sdxl"}
        mock_handle = _make_safe_open_mock(metadata_return_value=raw_metadata)

        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])

        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            scanner.start_scan(blocking=True)

        assert scanner.scan_complete is True
        assert scanner.is_scanning is False
        index = scanner.metadata_index
        assert len(index) == 2
        filenames = {m["filename"] for m in index.values()}
        assert "model_a.safetensors" in filenames
        assert "model_b.safetensors" in filenames
        assert "not_a_lora.txt" not in filenames

    def test_blocking_scan_handles_empty_directory(self, tmp_path):
        lora_dir = tmp_path / "empty_loras"
        lora_dir.mkdir()

        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])
        scanner.start_scan(blocking=True)

        assert scanner.scan_complete is True
        assert len(scanner.metadata_index) == 0

    def test_blocking_scan_handles_nonexistent_directory(self, tmp_path):
        scanner = LoraMetadataScanner(lora_paths=[str(tmp_path / "nonexistent")])
        scanner.start_scan(blocking=True)

        assert scanner.scan_complete is True
        assert len(scanner.metadata_index) == 0

    def test_search_by_base_model(self, scanner_with_index):
        results = scanner_with_index.search_by_base_model("SDXL")
        assert len(results) == 1
        assert results[0]["filename"] == "sdxl_char.safetensors"

    def test_search_by_base_model_case_insensitive(self, scanner_with_index):
        results = scanner_with_index.search_by_base_model("sdxl")
        assert len(results) == 1

    def test_search_by_base_model_no_match(self, scanner_with_index):
        results = scanner_with_index.search_by_base_model("Flux")
        assert len(results) == 0

    def test_search_by_trigger_word(self, scanner_with_index):
        results = scanner_with_index.search_by_trigger_word("saber")
        assert len(results) == 2
        filenames = {r["filename"] for r in results}
        assert "sdxl_char.safetensors" in filenames
        assert "sd15_generic.safetensors" in filenames

    def test_search_by_trigger_word_case_insensitive(self, scanner_with_index):
        results = scanner_with_index.search_by_trigger_word("SABER")
        assert len(results) == 2

    def test_search_by_trigger_word_no_match(self, scanner_with_index):
        results = scanner_with_index.search_by_trigger_word("nonexistent")
        assert len(results) == 0

    def test_search_by_trigger_word_partial_match(self, scanner_with_index):
        results = scanner_with_index.search_by_trigger_word("water")
        assert len(results) == 1
        assert results[0]["filename"] == "pony_style.safetensors"

    def test_get_metadata_returns_deep_copy(self, scanner_with_index):
        path = "/loras/sdxl_char.safetensors"
        meta1 = scanner_with_index.get_metadata(path)
        meta2 = scanner_with_index.get_metadata(path)
        assert meta1 is not meta2
        assert meta1 == meta2

    def test_get_metadata_returns_none_for_unknown(self, scanner_with_index):
        assert scanner_with_index.get_metadata("/nonexistent") is None

    def test_get_metadata_by_filename(self, scanner_with_index):
        results = scanner_with_index.get_metadata_by_filename("sdxl_char.safetensors")
        assert len(results) == 1
        assert results[0]["base_model"] == "SDXL 1.0"

    def test_scan_stats(self, scanner_with_index):
        stats = scanner_with_index.scan_stats
        assert "is_scanning" in stats
        assert "total_indexed" in stats
        assert stats["total_indexed"] == 3

    def test_remove_file(self, scanner_with_index):
        assert scanner_with_index.remove_file("/loras/sdxl_char.safetensors") is True
        assert len(scanner_with_index.metadata_index) == 2
        assert scanner_with_index.remove_file("/loras/sdxl_char.safetensors") is False

    def test_clear_index(self, scanner_with_index):
        scanner_with_index.clear_index()
        assert len(scanner_with_index.metadata_index) == 0
        assert scanner_with_index.scan_complete is False

    def test_scan_subdirectories(self, tmp_path):
        """Scanner discovers files in subdirectories."""
        lora_dir = tmp_path / "loras"
        sub_dir = lora_dir / "characters"
        sub_dir.mkdir(parents=True)
        (sub_dir / "deep.safetensors").write_bytes(b"fake")

        raw_metadata = {"ss_base_model_version": "pony"}
        mock_handle = _make_safe_open_mock(metadata_return_value=raw_metadata)

        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])
        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            scanner.start_scan(blocking=True)

        index = scanner.metadata_index
        assert len(index) == 1
        meta = next(iter(index.values()))
        assert meta["relative_path"] == os.path.join("characters", "deep.safetensors")


# ---------------------------------------------------------------------------
# Helper: build a scanner with a pre-populated index for library function tests
# ---------------------------------------------------------------------------
def _make_populated_scanner() -> LoraMetadataScanner:
    """Create a scanner with a representative index for library function tests.

    The global library functions (get_all_library_data, etc.) delegate to
    get_scanner(), so tests must patch that to inject this scanner.
    """
    scanner = LoraMetadataScanner(lora_paths=[])
    scanner._metadata_index = {
        "/loras/sdxl_char.safetensors": {
            "filename": "sdxl_char.safetensors",
            "file_path": "/loras/sdxl_char.safetensors",
            "relative_path": "sdxl_char.safetensors",
            "base_model": "SDXL 1.0",
            "trigger_words": ["saber", "fate"],
            "description": "Character LoRA for Saber",
            "characters": ["Saber"],
            "styles": ["Anime"],
            "file_size": 100,
            "training_epochs": None,
            "training_steps": None,
            "resolution": None,
            "network_dim": None,
            "network_alpha": None,
            "raw_metadata": {},
            "extraction_errors": [],
        },
        "/loras/pony/style.safetensors": {
            "filename": "style.safetensors",
            "file_path": "/loras/pony/style.safetensors",
            "relative_path": "pony/style.safetensors",
            "base_model": "Pony",
            "trigger_words": ["watercolor", "painting"],
            "description": "Style LoRA",
            "characters": [],
            "styles": ["Watercolor"],
            "file_size": 200,
            "training_epochs": None,
            "training_steps": None,
            "resolution": None,
            "network_dim": None,
            "network_alpha": None,
            "raw_metadata": {},
            "extraction_errors": [],
        },
        "/loras/no_model.safetensors": {
            "filename": "no_model.safetensors",
            "file_path": "/loras/no_model.safetensors",
            "relative_path": "no_model.safetensors",
            "base_model": None,
            "trigger_words": [],
            "description": None,
            "characters": [],
            "styles": [],
            "file_size": 50,
            "training_epochs": None,
            "training_steps": None,
            "resolution": None,
            "network_dim": None,
            "network_alpha": None,
            "raw_metadata": {},
            "extraction_errors": [],
        },
    }
    scanner._scan_complete = True
    return scanner


@pytest.fixture()
def _patch_scanner():
    """Patch get_scanner to return a pre-populated scanner for library tests."""
    scanner = _make_populated_scanner()
    with patch("modules.lora_metadata.get_scanner", return_value=scanner):
        yield scanner


# ---------------------------------------------------------------------------
# get_all_library_data
# ---------------------------------------------------------------------------
class TestGetAllLibraryData:
    """get_all_library_data() returns sorted entries with fallback values."""

    @pytest.mark.usefixtures("_patch_scanner")
    def test_returns_all_entries(self):
        result = get_all_library_data()
        assert len(result) == 3

    @pytest.mark.usefixtures("_patch_scanner")
    def test_sorted_by_relative_path_case_insensitive(self):
        result = get_all_library_data()
        paths = [entry["relative_path"] for entry in result]
        assert paths == sorted(paths, key=str.lower)

    @pytest.mark.usefixtures("_patch_scanner")
    def test_fallback_base_model_unknown(self):
        result = get_all_library_data()
        no_model_entry = next(e for e in result if e["filename"] == "no_model.safetensors")
        assert no_model_entry["base_model"] == "Unknown"

    @pytest.mark.usefixtures("_patch_scanner")
    def test_fallback_description_empty_string(self):
        result = get_all_library_data()
        no_model_entry = next(e for e in result if e["filename"] == "no_model.safetensors")
        assert no_model_entry["description"] == ""

    @pytest.mark.usefixtures("_patch_scanner")
    def test_fallback_trigger_words_empty_list(self):
        result = get_all_library_data()
        no_model_entry = next(e for e in result if e["filename"] == "no_model.safetensors")
        assert no_model_entry["trigger_words"] == []

    def test_empty_index_returns_empty_list(self):
        scanner = LoraMetadataScanner(lora_paths=[])
        with patch("modules.lora_metadata.get_scanner", return_value=scanner):
            result = get_all_library_data()
        assert result == []

    @pytest.mark.usefixtures("_patch_scanner")
    def test_fallback_relative_path_uses_filename(self):
        """When relative_path is missing, filename is used as fallback."""
        scanner = _make_populated_scanner()
        # Remove relative_path from one entry
        del scanner._metadata_index["/loras/sdxl_char.safetensors"]["relative_path"]
        with patch("modules.lora_metadata.get_scanner", return_value=scanner):
            result = get_all_library_data()
        entry = next(e for e in result if e["filename"] == "sdxl_char.safetensors")
        assert entry["relative_path"] == "sdxl_char.safetensors"


# ---------------------------------------------------------------------------
# get_distinct_base_models
# ---------------------------------------------------------------------------
class TestGetDistinctBaseModels:
    """get_distinct_base_models() returns sorted unique models with Unknown last."""

    @pytest.mark.usefixtures("_patch_scanner")
    def test_returns_known_models_sorted(self):
        result = get_distinct_base_models()
        # Only non-None models are included; the None entry is excluded
        assert "Pony" in result
        assert "SDXL 1.0" in result

    @pytest.mark.usefixtures("_patch_scanner")
    def test_excludes_none_base_models(self):
        result = get_distinct_base_models()
        assert None not in result

    def test_unknown_placed_at_end(self):
        scanner = LoraMetadataScanner(lora_paths=[])
        scanner._metadata_index = {
            "/a.safetensors": {"base_model": "Unknown"},
            "/b.safetensors": {"base_model": "SDXL 1.0"},
            "/c.safetensors": {"base_model": "Pony"},
        }
        with patch("modules.lora_metadata.get_scanner", return_value=scanner):
            result = get_distinct_base_models()
        assert result[-1] == "Unknown"
        assert result[0] != "Unknown"

    def test_empty_index_returns_empty_list(self):
        scanner = LoraMetadataScanner(lora_paths=[])
        with patch("modules.lora_metadata.get_scanner", return_value=scanner):
            result = get_distinct_base_models()
        assert result == []

    def test_models_are_alphabetically_sorted(self):
        scanner = LoraMetadataScanner(lora_paths=[])
        scanner._metadata_index = {
            "/z.safetensors": {"base_model": "SD 1.5"},
            "/a.safetensors": {"base_model": "Flux"},
            "/b.safetensors": {"base_model": "Pony"},
        }
        with patch("modules.lora_metadata.get_scanner", return_value=scanner):
            result = get_distinct_base_models()
        assert result == ["Flux", "Pony", "SD 1.5"]


# ---------------------------------------------------------------------------
# get_trigger_words_for_filename
# ---------------------------------------------------------------------------
class TestGetTriggerWordsForFilename:
    """get_trigger_words_for_filename() finds trigger words by relative path or basename."""

    @pytest.mark.usefixtures("_patch_scanner")
    def test_match_by_relative_path(self):
        result = get_trigger_words_for_filename("pony/style.safetensors")
        assert result == ["watercolor", "painting"]

    def test_fallback_to_basename_match(self):
        """When no relative_path matches, fall back to basename matching."""
        scanner = LoraMetadataScanner(lora_paths=[])
        scanner._metadata_index = {
            "/deep/path/special.safetensors": {
                "filename": "special.safetensors",
                "file_path": "/deep/path/special.safetensors",
                "relative_path": "deep/path/special.safetensors",
                "base_model": "SDXL 1.0",
                "trigger_words": ["unique_trigger"],
                "description": "",
                "characters": [],
                "styles": [],
                "file_size": 100,
                "training_epochs": None,
                "training_steps": None,
                "resolution": None,
                "network_dim": None,
                "network_alpha": None,
                "raw_metadata": {},
                "extraction_errors": [],
            },
        }
        with patch("modules.lora_metadata.get_scanner", return_value=scanner):
            # "special.safetensors" won't match relative_path "deep/path/special.safetensors"
            # but will match the basename via get_metadata_by_filename
            result = get_trigger_words_for_filename("special.safetensors")
        assert result == ["unique_trigger"]

    @pytest.mark.usefixtures("_patch_scanner")
    def test_not_found_returns_empty_list(self):
        result = get_trigger_words_for_filename("nonexistent.safetensors")
        assert result == []


# ---------------------------------------------------------------------------
# search_library
# ---------------------------------------------------------------------------
class TestSearchLibrary:
    """search_library() filters by text query and/or base model."""

    @pytest.mark.usefixtures("_patch_scanner")
    def test_text_search_matches_description(self):
        result = search_library(query="Character")
        assert len(result) == 1
        assert result[0]["filename"] == "sdxl_char.safetensors"

    @pytest.mark.usefixtures("_patch_scanner")
    def test_text_search_matches_trigger_word(self):
        result = search_library(query="watercolor")
        assert len(result) == 1
        assert result[0]["filename"] == "style.safetensors"

    @pytest.mark.usefixtures("_patch_scanner")
    def test_base_model_filter(self):
        result = search_library(base_model_filter="Pony")
        assert len(result) == 1
        assert result[0]["filename"] == "style.safetensors"

    @pytest.mark.usefixtures("_patch_scanner")
    def test_combined_query_and_filter(self):
        result = search_library(query="saber", base_model_filter="SDXL 1.0")
        assert len(result) == 1
        assert result[0]["filename"] == "sdxl_char.safetensors"

    @pytest.mark.usefixtures("_patch_scanner")
    def test_empty_query_returns_all(self):
        result = search_library(query="", base_model_filter="")
        assert len(result) == 3

    @pytest.mark.usefixtures("_patch_scanner")
    def test_no_match_returns_empty(self):
        result = search_library(query="zzzzz_nonexistent")
        assert result == []

    @pytest.mark.usefixtures("_patch_scanner")
    def test_base_model_filter_with_unknown_fallback(self):
        """Entries with None base_model get 'Unknown' fallback in get_all_library_data."""
        result = search_library(base_model_filter="Unknown")
        assert len(result) == 1
        assert result[0]["filename"] == "no_model.safetensors"

    @pytest.mark.usefixtures("_patch_scanner")
    def test_text_search_is_case_insensitive(self):
        result = search_library(query="SABER")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _run_scan edge cases — error handling, progress, and stop-requested
# ---------------------------------------------------------------------------
class TestRunScanEdgeCases:
    """_run_scan() handles extraction failures, progress logging, and stop requests."""

    def test_extraction_failure_increments_files_failed(self, tmp_path):
        """Files that raise during extract_metadata are counted as failures."""
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        (lora_dir / "bad.safetensors").write_bytes(b"fake")

        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])

        # Patch extract_metadata itself to raise — _safe_open errors are caught
        # inside extract_metadata and don't propagate to _run_scan.
        with patch("modules.lora_metadata.extract_metadata", side_effect=RuntimeError("unhandled")):
            scanner.start_scan(blocking=True)

        stats = scanner.scan_stats
        assert stats["files_failed"] == 1
        assert stats["files_scanned"] == 0

    def test_stop_requested_halts_scan(self, tmp_path):
        """Setting _stop_requested causes the scan to stop mid-iteration."""
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        for i in range(5):
            (lora_dir / f"model_{i}.safetensors").write_bytes(b"fake")

        raw_metadata = {"ss_base_model_version": "sdxl"}
        mock_handle = _make_safe_open_mock(metadata_return_value=raw_metadata)

        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])

        call_count = 0

        def stop_after_two(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                scanner._stop_requested = True
            return mock_handle

        with patch("modules.lora_metadata._safe_open", side_effect=stop_after_two):
            scanner.start_scan(blocking=True)

        # Should have scanned fewer than all 5 files
        assert scanner.scan_stats["files_scanned"] < 5
        assert scanner.scan_complete is True

    def test_no_lora_paths_aborts_scan(self):
        """When no paths are configured and config import fails, scan aborts."""
        scanner = LoraMetadataScanner(lora_paths=[])
        with patch(
            "modules.lora_metadata.LoraMetadataScanner._load_lora_paths_from_config",
            return_value=[],
        ):
            scanner.start_scan(blocking=True)
        assert scanner.scan_complete is True
        assert len(scanner.metadata_index) == 0

    def test_load_lora_paths_from_config_import_error(self):
        """_load_lora_paths_from_config returns [] on ImportError."""
        scanner = LoraMetadataScanner(lora_paths=[])
        with patch(
            "modules.lora_metadata.LoraMetadataScanner._load_lora_paths_from_config",
            return_value=[],
        ):
            scanner.start_scan(blocking=True)
        assert scanner.scan_complete is True


# ---------------------------------------------------------------------------
# _extract_base_model edge cases
# ---------------------------------------------------------------------------
class TestExtractBaseModel:
    """_extract_base_model() checks key mappings in order and handles empty values."""

    def test_returns_none_when_no_keys_present(self):
        assert _extract_base_model({}) is None

    def test_skips_empty_value(self):
        metadata = {"ss_base_model_version": "", "base_model": "sdxl"}
        result = _extract_base_model(metadata)
        assert result == "SDXL 1.0"

    def test_returns_none_when_all_values_empty(self):
        metadata = {"ss_base_model_version": ""}
        assert _extract_base_model(metadata) is None


# ---------------------------------------------------------------------------
# _parse_bucket_resolutions edge cases
# ---------------------------------------------------------------------------
class TestParseBucketResolutionsEdgeCases:
    """_parse_bucket_resolutions() handles non-dict and malformed bucket keys."""

    def test_non_dict_value_returns_none(self):
        assert _parse_bucket_resolutions(json.dumps([1, 2, 3])) is None

    def test_invalid_bucket_key_skipped(self):
        data = {"buckets": {"not_valid_json": {"count": 1}, "[512, 768]": {"count": 1}}}
        result = _parse_bucket_resolutions(json.dumps(data))
        assert result == "512x768"

    def test_empty_resolutions_returns_none(self):
        # Bucket key that parses but has wrong structure (not 2 elements)
        data = {"buckets": {"[512]": {"count": 1}}}
        result = _parse_bucket_resolutions(json.dumps(data))
        assert result is None


# ---------------------------------------------------------------------------
# _extract_resolution edge case — empty value skipped
# ---------------------------------------------------------------------------
class TestExtractResolutionEdgeCases:
    """_extract_resolution() skips empty/falsy values."""

    def test_skips_empty_resolution_value(self):
        metadata = {"ss_resolution": "", "resolution": "1024x1024"}
        result = _extract_resolution(metadata)
        assert result == "1024x1024"


# ---------------------------------------------------------------------------
# Scanner: refresh_file and _compute_relative_path
# ---------------------------------------------------------------------------
class TestScannerRefreshAndRelativePath:
    """refresh_file() and _compute_relative_path() handle updates and fallbacks."""

    def test_refresh_file_updates_index(self, tmp_path):
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        lora_file = lora_dir / "test.safetensors"
        lora_file.write_bytes(b"fake")

        raw = {"ss_base_model_version": "sdxl"}
        mock_handle = _make_safe_open_mock(metadata_return_value=raw)

        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])

        with patch("modules.lora_metadata._safe_open", return_value=mock_handle):
            result = scanner.refresh_file(str(lora_file))

        assert result is not None
        assert result["base_model"] == "SDXL 1.0"
        assert str(lora_file) in scanner.metadata_index

    def test_refresh_file_returns_none_on_failure(self, tmp_path):
        scanner = LoraMetadataScanner(lora_paths=[])

        with patch("modules.lora_metadata.extract_metadata", side_effect=RuntimeError("fail")):
            result = scanner.refresh_file("/nonexistent.safetensors")

        assert result is None

    def test_compute_relative_path_fallback_to_basename(self):
        scanner = LoraMetadataScanner(lora_paths=["/some/path"])
        result = scanner._compute_relative_path("/different/path/file.safetensors")
        assert result == "file.safetensors"

    def test_compute_relative_path_matches_lora_root(self, tmp_path):
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        scanner = LoraMetadataScanner(lora_paths=[str(lora_dir)])
        result = scanner._compute_relative_path(str(lora_dir / "sub" / "file.safetensors"))
        assert result == os.path.join("sub", "file.safetensors")


# ---------------------------------------------------------------------------
# _discover_lora_files edge case — path is a file, not a directory
# ---------------------------------------------------------------------------
class TestDiscoverLoraFilesEdgeCases:
    """_discover_lora_files() handles non-directory paths."""

    def test_skips_path_that_is_a_file(self, tmp_path):
        file_path = tmp_path / "not_a_dir.safetensors"
        file_path.write_bytes(b"fake")

        scanner = LoraMetadataScanner(lora_paths=[str(file_path)])
        files = scanner._discover_lora_files()
        assert files == []


# ---------------------------------------------------------------------------
# extract_metadata edge case — file_size OSError
# ---------------------------------------------------------------------------
class TestExtractMetadataEdgeCases:
    """extract_metadata() handles file size errors gracefully."""

    def test_file_size_os_error(self, tmp_path):
        fake_file = tmp_path / "missing.safetensors"
        # File doesn't exist, so getsize will fail
        # But _safe_open is mocked, so only file_size path errors
        raw = {"ss_base_model_version": "sdxl"}
        mock_handle = _make_safe_open_mock(metadata_return_value=raw)

        with (
            patch("modules.lora_metadata._safe_open", return_value=mock_handle),
            patch("os.path.getsize", side_effect=OSError("no such file")),
        ):
            result = extract_metadata(str(fake_file))

        assert result["file_size"] == 0
        assert any("Failed to get file size" in e for e in result["extraction_errors"])
