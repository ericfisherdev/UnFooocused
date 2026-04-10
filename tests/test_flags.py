"""Unit tests for the standalone flags and enums module.

Inner-loop TDD (Beck): tests for enum members, constant lists, and sentinel values.
These tests define the contract for modules/flags.py before it exists.

Tests are organized by domain concept:
  - OutputFormat enum
  - Performance enum (with Steps, StepsUOV, PerformanceLoRA)
  - MetadataScheme enum
  - Sampler and scheduler constant lists
  - SDXL aspect ratio list
  - Sentinel values (disabled, enabled, etc.)
"""

from enum import Enum, IntEnum

# ---------------------------------------------------------------------------
# OutputFormat enum
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """OutputFormat enum represents image output file formats."""

    def test_has_png_member(self):
        from modules.flags import OutputFormat

        assert OutputFormat.PNG.value == "png"

    def test_has_jpeg_member(self):
        from modules.flags import OutputFormat

        assert OutputFormat.JPEG.value == "jpeg"

    def test_has_webp_member(self):
        from modules.flags import OutputFormat

        assert OutputFormat.WEBP.value == "webp"

    def test_has_exactly_three_members(self):
        from modules.flags import OutputFormat

        assert len(OutputFormat) == 3

    def test_is_an_enum(self):
        from modules.flags import OutputFormat

        assert issubclass(OutputFormat, Enum)

    def test_list_returns_all_values(self):
        from modules.flags import OutputFormat

        assert OutputFormat.list() == ["png", "jpeg", "webp"]


# ---------------------------------------------------------------------------
# Performance enum
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance enum represents generation quality/speed presets."""

    def test_has_quality_member(self):
        from modules.flags import Performance

        assert Performance.QUALITY.value == "Quality"

    def test_has_speed_member(self):
        from modules.flags import Performance

        assert Performance.SPEED.value == "Speed"

    def test_has_extreme_speed_member(self):
        from modules.flags import Performance

        assert Performance.EXTREME_SPEED.value == "Extreme Speed"

    def test_has_lightning_member(self):
        from modules.flags import Performance

        assert Performance.LIGHTNING.value == "Lightning"

    def test_has_hyper_sd_member(self):
        from modules.flags import Performance

        assert Performance.HYPER_SD.value == "Hyper-SD"

    def test_has_exactly_five_members(self):
        from modules.flags import Performance

        assert len(Performance) == 5

    def test_list_returns_name_value_tuples(self):
        from modules.flags import Performance

        result = Performance.list()
        assert ("QUALITY", "Quality") in result
        assert ("SPEED", "Speed") in result

    def test_values_returns_display_strings(self):
        from modules.flags import Performance

        result = Performance.values()
        assert "Quality" in result
        assert "Speed" in result
        assert "Extreme Speed" in result

    def test_steps_returns_correct_value_for_quality(self):
        from modules.flags import Performance

        assert Performance.QUALITY.steps() == 60

    def test_steps_returns_correct_value_for_speed(self):
        from modules.flags import Performance

        assert Performance.SPEED.steps() == 30

    def test_steps_returns_correct_value_for_extreme_speed(self):
        from modules.flags import Performance

        assert Performance.EXTREME_SPEED.steps() == 8

    def test_has_restricted_features_for_extreme_speed(self):
        from modules.flags import Performance

        assert Performance.has_restricted_features(Performance.EXTREME_SPEED) is True

    def test_has_restricted_features_false_for_quality(self):
        from modules.flags import Performance

        assert Performance.has_restricted_features(Performance.QUALITY) is False

    def test_by_steps_maps_step_count_to_performance(self):
        from modules.flags import Performance

        assert Performance.by_steps(60) == Performance.QUALITY
        assert Performance.by_steps(30) == Performance.SPEED


# ---------------------------------------------------------------------------
# Steps enum
# ---------------------------------------------------------------------------


class TestSteps:
    """Steps IntEnum maps performance presets to step counts."""

    def test_quality_is_60(self):
        from modules.flags import Steps

        assert Steps.QUALITY == 60

    def test_speed_is_30(self):
        from modules.flags import Steps

        assert Steps.SPEED == 30

    def test_extreme_speed_is_8(self):
        from modules.flags import Steps

        assert Steps.EXTREME_SPEED == 8

    def test_is_int_enum(self):
        from modules.flags import Steps

        assert issubclass(Steps, IntEnum)

    def test_keys_returns_member_names(self):
        from modules.flags import Steps

        keys = Steps.keys()
        assert "QUALITY" in keys
        assert "SPEED" in keys


# ---------------------------------------------------------------------------
# MetadataScheme enum
# ---------------------------------------------------------------------------


class TestMetadataScheme:
    """MetadataScheme enum represents metadata embedding formats."""

    def test_has_fooocus_member(self):
        from modules.flags import MetadataScheme

        assert MetadataScheme.FOOOCUS.value == "fooocus"

    def test_has_a1111_member(self):
        from modules.flags import MetadataScheme

        assert MetadataScheme.A1111.value == "a1111"

    def test_has_exactly_two_members(self):
        from modules.flags import MetadataScheme

        assert len(MetadataScheme) == 2


# ---------------------------------------------------------------------------
# Sampler and scheduler lists
# ---------------------------------------------------------------------------


class TestSamplerList:
    """sampler_list contains all available sampler algorithm names."""

    def test_sampler_list_is_a_list(self):
        from modules.flags import sampler_list

        assert isinstance(sampler_list, list)

    def test_sampler_list_contains_dpmpp_2m_sde_gpu(self):
        """dpmpp_2m_sde_gpu is the default sampler in config."""
        from modules.flags import sampler_list

        assert "dpmpp_2m_sde_gpu" in sampler_list

    def test_sampler_list_contains_euler(self):
        from modules.flags import sampler_list

        assert "euler" in sampler_list

    def test_sampler_list_contains_ddim(self):
        """DDIM is in the SAMPLER_EXTRA group."""
        from modules.flags import sampler_list

        assert "ddim" in sampler_list

    def test_sampler_list_has_at_least_15_entries(self):
        from modules.flags import sampler_list

        assert len(sampler_list) >= 15


class TestSchedulerList:
    """scheduler_list contains all available scheduler algorithm names."""

    def test_scheduler_list_is_a_list(self):
        from modules.flags import scheduler_list

        assert isinstance(scheduler_list, list)

    def test_scheduler_list_contains_karras(self):
        """karras is the default scheduler in config."""
        from modules.flags import scheduler_list

        assert "karras" in scheduler_list

    def test_scheduler_list_contains_normal(self):
        from modules.flags import scheduler_list

        assert "normal" in scheduler_list

    def test_scheduler_list_has_at_least_5_entries(self):
        from modules.flags import scheduler_list

        assert len(scheduler_list) >= 5


# ---------------------------------------------------------------------------
# SDXL aspect ratios
# ---------------------------------------------------------------------------


class TestSdxlAspectRatios:
    """sdxl_aspect_ratios contains the standard SDXL resolution pairs."""

    def test_is_a_list(self):
        from modules.flags import sdxl_aspect_ratios

        assert isinstance(sdxl_aspect_ratios, list)

    def test_contains_832x1216(self):
        from modules.flags import sdxl_aspect_ratios

        assert "832*1216" in sdxl_aspect_ratios

    def test_contains_1024x1024(self):
        from modules.flags import sdxl_aspect_ratios

        assert "1024*1024" in sdxl_aspect_ratios

    def test_contains_1216x832(self):
        from modules.flags import sdxl_aspect_ratios

        assert "1216*832" in sdxl_aspect_ratios

    def test_has_26_standard_ratios(self):
        """The full SDXL aspect ratio list has 26 entries."""
        from modules.flags import sdxl_aspect_ratios

        assert len(sdxl_aspect_ratios) == 26

    def test_all_entries_match_width_star_height_format(self):
        from modules.flags import sdxl_aspect_ratios

        for ratio in sdxl_aspect_ratios:
            parts = ratio.split("*")
            assert len(parts) == 2, f"Expected W*H format, got: {ratio}"
            assert parts[0].isdigit() and parts[1].isdigit(), f"Non-numeric dimensions: {ratio}"


# ---------------------------------------------------------------------------
# Sentinel values
# ---------------------------------------------------------------------------


class TestSentinelValues:
    """Sentinel string constants used as placeholder values in generation args."""

    def test_disabled_sentinel(self):
        from modules.flags import disabled

        assert disabled == "Disabled"

    def test_enabled_sentinel(self):
        from modules.flags import enabled

        assert enabled == "Enabled"


# ---------------------------------------------------------------------------
# output_formats list (legacy compat)
# ---------------------------------------------------------------------------


class TestOutputFormats:
    """output_formats is the legacy list form of OutputFormat values."""

    def test_output_formats_matches_enum_list(self):
        from modules.flags import OutputFormat, output_formats

        assert output_formats == OutputFormat.list()
