"""UNF-61: full_config_template.txt generation tests."""

from __future__ import annotations

import json
import re

import pytest


class TestKeyDocRegistry:
    """Every _DEFAULTS key must have documentation metadata."""

    def test_every_default_key_has_docs(self):
        from modules.config import _DEFAULTS
        from modules.config_template import KEY_DOCS

        missing = sorted(set(_DEFAULTS) - set(KEY_DOCS))
        assert not missing, f"KEY_DOCS missing entries for: {missing}"

    def test_no_extra_docs_without_default(self):
        from modules.config import _DEFAULTS
        from modules.config_template import KEY_DOCS

        orphan = sorted(set(KEY_DOCS) - set(_DEFAULTS))
        assert not orphan, f"KEY_DOCS has entries with no _DEFAULTS counterpart: {orphan}"

    def test_every_doc_has_category_type_description(self):
        from modules.config_template import KEY_DOCS, KeyDoc

        for key, doc in KEY_DOCS.items():
            assert isinstance(doc, KeyDoc), f"{key} is not a KeyDoc"
            assert doc.category, f"{key} missing category"
            assert doc.type_hint, f"{key} missing type_hint"
            assert doc.description, f"{key} missing description"


class TestGenerateTemplate:
    """Template generation produces JSONC with every default key."""

    def test_contains_every_default_key(self):
        from modules.config import _DEFAULTS
        from modules.config_template import generate_template

        output = generate_template()
        for key in _DEFAULTS:
            assert f'"{key}"' in output, f"template missing key {key!r}"

    def test_contains_category_headers(self):
        from modules.config_template import generate_template

        output = generate_template()
        assert "// === Model" in output
        assert "// === Paths" in output

    def test_body_is_valid_json_after_stripping_comments(self):
        from modules.config_template import generate_template

        output = generate_template()
        stripped = re.sub(r"^\s*//.*$", "", output, flags=re.MULTILINE)
        parsed = json.loads(stripped)
        assert isinstance(parsed, dict)

    def test_every_key_preceded_by_doc_comment(self):
        from modules.config import _DEFAULTS
        from modules.config_template import generate_template

        output = generate_template()
        for key in _DEFAULTS:
            pattern = re.compile(rf"//[^\n]*\n\s*\"{re.escape(key)}\"")
            assert pattern.search(output), f"{key} missing preceding doc comment"


class TestWriteTemplate:
    """Template is written to disk and regenerated each call."""

    def test_write_creates_file_at_target_path(self, tmp_path):
        from modules.config_template import write_template

        target = tmp_path / "full_config_template.txt"
        write_template(target)
        assert target.is_file()
        content = target.read_text()
        assert '"default_model"' in content

    def test_write_overwrites_existing_file(self, tmp_path):
        from modules.config_template import write_template

        target = tmp_path / "full_config_template.txt"
        target.write_text("stale")
        write_template(target)
        assert "stale" not in target.read_text()

    def test_load_config_writes_template_next_to_config_file(self, tmp_path):
        from modules.config import load_config

        config_path = tmp_path / "config.txt"
        load_config(config_path=config_path)
        template_path = tmp_path / "full_config_template.txt"
        assert template_path.is_file()

    def test_template_regenerates_on_each_load(self, tmp_path):
        from modules.config import load_config

        config_path = tmp_path / "config.txt"
        template_path = tmp_path / "full_config_template.txt"
        load_config(config_path=config_path)
        first_mtime = template_path.stat().st_mtime_ns
        template_path.write_text("tampered")
        load_config(config_path=config_path)
        assert template_path.read_text() != "tampered"
        assert template_path.stat().st_mtime_ns >= first_mtime


class TestNewKeyGuardrail:
    """Adding a new _DEFAULTS key without docs must fail the registry check."""

    def test_validate_registry_raises_when_key_missing_from_docs(self):
        from modules.config_template import _validate_registry

        fake_defaults = {"new_unknown_key": 0}
        fake_docs = {}
        with pytest.raises(ValueError, match="new_unknown_key"):
            _validate_registry(fake_defaults, fake_docs)
