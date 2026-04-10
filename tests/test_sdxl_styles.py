"""Unit tests for modules.sdxl_styles — the standalone SDXL styles module.

Inner-loop TDD (Beck): these tests define the contract for style definitions,
legal_style_names, and apply_style() before the module exists.

Test structure follows Arrange-Act-Assert (Osherove) and tests observable
behavior, not implementation details (Khorikov).
"""

import pytest

# ---------------------------------------------------------------------------
# Style data structure
# ---------------------------------------------------------------------------


class TestStyleDefinitions:
    """The styles dict maps style names to (positive_prompt, negative_prompt) tuples."""

    def test_styles_dict_exists_and_is_nonempty(self):
        from modules.sdxl_styles import styles

        assert isinstance(styles, dict)
        assert len(styles) > 0

    def test_each_style_value_is_a_two_element_tuple(self):
        from modules.sdxl_styles import styles

        for name, value in styles.items():
            assert isinstance(value, tuple), f"Style {name!r} value is not a tuple"
            assert len(value) == 2, f"Style {name!r} tuple has {len(value)} elements, expected 2"

    def test_style_prompts_are_strings(self):
        from modules.sdxl_styles import styles

        for name, (positive, negative) in styles.items():
            assert isinstance(positive, str), f"Style {name!r} positive prompt is not a string"
            assert isinstance(negative, str), f"Style {name!r} negative prompt is not a string"

    def test_fooocus_v2_is_virtual_style(self):
        """Fooocus V2 is a virtual expansion style — in legal_style_names but NOT in styles dict."""
        from modules.sdxl_styles import FOOOCUS_EXPANSION, legal_style_names, styles

        assert FOOOCUS_EXPANSION == "Fooocus V2"
        assert FOOOCUS_EXPANSION in legal_style_names
        assert FOOOCUS_EXPANSION not in styles

    def test_fooocus_enhance_style_exists(self):
        """Fooocus Enhance is a default style in config.default_styles."""
        from modules.sdxl_styles import styles

        assert "Fooocus Enhance" in styles

    def test_fooocus_sharp_style_exists(self):
        """Fooocus Sharp is a default style in config.default_styles."""
        from modules.sdxl_styles import styles

        assert "Fooocus Sharp" in styles


# ---------------------------------------------------------------------------
# legal_style_names
# ---------------------------------------------------------------------------


class TestLegalStyleNames:
    """legal_style_names is the list of style names available to the UI."""

    def test_legal_style_names_is_a_list(self):
        from modules.sdxl_styles import legal_style_names

        assert isinstance(legal_style_names, list)

    def test_legal_style_names_is_nonempty(self):
        from modules.sdxl_styles import legal_style_names

        assert len(legal_style_names) > 0

    def test_legal_style_names_contains_only_strings(self):
        from modules.sdxl_styles import legal_style_names

        for name in legal_style_names:
            assert isinstance(name, str), f"Expected str, got {type(name)}: {name!r}"

    def test_fooocus_v2_is_first_entry(self):
        """Fooocus V2 must be the first style (the default expansion)."""
        from modules.sdxl_styles import legal_style_names

        assert legal_style_names[0] == "Fooocus V2"

    def test_default_styles_are_all_present(self):
        """All config.default_styles must be available in legal_style_names."""
        from modules.sdxl_styles import legal_style_names

        defaults = ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
        for style in defaults:
            assert style in legal_style_names, f"Default style {style!r} missing from legal_style_names"

    def test_no_duplicate_names(self):
        from modules.sdxl_styles import legal_style_names

        assert len(legal_style_names) == len(set(legal_style_names)), "Duplicate style names found"


# ---------------------------------------------------------------------------
# apply_style()
# ---------------------------------------------------------------------------


class TestApplyStyle:
    """apply_style() injects style prompts into user prompts using {prompt} placeholder."""

    def test_apply_style_returns_three_element_tuple(self):
        from modules.sdxl_styles import apply_style

        result = apply_style("Fooocus Sharp", "a cat")
        assert isinstance(result, tuple)
        assert len(result) == 3, f"Expected 3-element tuple, got {len(result)}"

    def test_apply_style_positive_is_list_of_strings(self):
        from modules.sdxl_styles import apply_style

        positive, _negative, _has_placeholder = apply_style("Fooocus Sharp", "a cat")
        assert isinstance(positive, list)
        assert all(isinstance(line, str) for line in positive)

    def test_apply_style_negative_is_list_of_strings(self):
        from modules.sdxl_styles import apply_style

        _positive, negative, _has_placeholder = apply_style("Fooocus Sharp", "a cat")
        assert isinstance(negative, list)
        assert all(isinstance(line, str) for line in negative)

    def test_apply_style_third_element_is_bool(self):
        """Third element indicates whether {prompt} placeholder was found."""
        from modules.sdxl_styles import apply_style

        _positive, _negative, has_placeholder = apply_style("Fooocus Sharp", "a cat")
        assert isinstance(has_placeholder, bool)

    def test_prompt_placeholder_is_replaced(self):
        """When the style has {prompt}, the user prompt should appear in the output."""
        from modules.sdxl_styles import apply_style

        positive, _negative, has_placeholder = apply_style("Fooocus Sharp", "a beautiful sunset")
        combined = " ".join(positive)
        assert "a beautiful sunset" in combined
        assert "{prompt}" not in combined
        assert has_placeholder is True

    def test_style_without_positive_prompt_returns_user_prompt(self):
        """Styles with empty positive prompt should still work (user prompt passed through)."""
        from modules.sdxl_styles import apply_style, styles

        # Find a style with no positive prompt (empty string — no {prompt} placeholder)
        no_positive_styles = [name for name, (p, _n) in styles.items() if not p]
        if not no_positive_styles:
            pytest.skip("No styles with empty positive prompt found")
        style_name = no_positive_styles[0]
        _positive, _negative, has_placeholder = apply_style(style_name, "test prompt")
        assert has_placeholder is False

    def test_apply_style_raises_for_unknown_style(self):
        from modules.sdxl_styles import apply_style

        with pytest.raises(KeyError):
            apply_style("Nonexistent Style That Does Not Exist", "a cat")
