"""Tests for Python 3.14 version requirements (UNF-45).

Acceptance criteria:
  AC1: CI workflows use Python 3.14
  AC2: pyproject.toml requires-python is >=3.14
  AC3: No tomli dependency or conditional import code remains
  AC5: ruff target-version is updated to highest supported version
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


@pytest.fixture()
def pyproject_data() -> dict:
    """Parse pyproject.toml and return the data dictionary."""
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# AC1: CI workflows use Python 3.14
# ---------------------------------------------------------------------------


class TestCIWorkflowsPython314:
    """AC1: CI workflows must use Python 3.14."""

    def test_test_workflow_uses_python_314(self) -> None:
        """test.yml must specify python-version 3.14."""
        content = (WORKFLOWS_DIR / "test.yml").read_text()
        assert 'python-version: "3.14"' in content, (
            "test.yml does not use Python 3.14. Found content that should contain 'python-version: \"3.14\"'."
        )

    def test_test_workflow_does_not_use_python_310(self) -> None:
        """test.yml must not reference Python 3.10."""
        content = (WORKFLOWS_DIR / "test.yml").read_text()
        assert 'python-version: "3.10"' not in content, "test.yml still references Python 3.10."

    def test_lint_workflow_uses_python_314(self) -> None:
        """lint.yml bandit job must specify python-version 3.14."""
        content = (WORKFLOWS_DIR / "lint.yml").read_text()
        assert 'python-version: "3.14"' in content, (
            "lint.yml does not use Python 3.14. Found content that should contain 'python-version: \"3.14\"'."
        )

    def test_lint_workflow_does_not_use_python_310(self) -> None:
        """lint.yml must not reference Python 3.10."""
        content = (WORKFLOWS_DIR / "lint.yml").read_text()
        assert 'python-version: "3.10"' not in content, "lint.yml still references Python 3.10."


# ---------------------------------------------------------------------------
# AC2: pyproject.toml requires-python is >=3.14
# ---------------------------------------------------------------------------


class TestRequiresPython314:
    """AC2: pyproject.toml requires-python must be >=3.14."""

    def test_requires_python_is_3_14(self, pyproject_data: dict) -> None:
        """requires-python must be '>=3.14'."""
        requires_python = pyproject_data.get("project", {}).get("requires-python", "")
        assert requires_python == ">=3.14", f"requires-python is '{requires_python}', expected '>=3.14'."


# ---------------------------------------------------------------------------
# AC3: No tomli dependency or conditional import code remains
# ---------------------------------------------------------------------------


class TestNoTomliReferences:
    """AC3: No tomli dependency or conditional import code remains."""

    def test_no_tomli_in_dev_dependencies(self, pyproject_data: dict) -> None:
        """tomli must not appear in [project.optional-dependencies].dev."""
        dev_deps = pyproject_data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
        tomli_deps = [d for d in dev_deps if "tomli" in d.lower()]
        assert not tomli_deps, (
            f"tomli still listed in dev dependencies: {tomli_deps}. Python 3.14 includes tomllib in stdlib."
        )

    def test_no_tomli_conditional_import_in_test_files(self) -> None:
        """No test file should contain 'import tomli as tomllib' conditional."""
        # Build the forbidden string from parts so this file's own assertion
        # messages do not trigger a false positive when scanned.
        forbidden = "import tomli " + "as tomllib"
        tests_dir = PROJECT_ROOT / "tests"
        for test_file in tests_dir.glob("*.py"):
            if test_file.name == Path(__file__).name:
                continue
            content = test_file.read_text()
            assert forbidden not in content, (
                f"{test_file.name} still contains '{forbidden}'. With Python >=3.14, use 'import tomllib' directly."
            )

    def test_no_version_check_for_tomllib(self) -> None:
        """No test file should contain sys.version_info checks for tomllib."""
        version_check = "sys.version" + "_info"
        tomli_ref = "tom" + "li"
        tests_dir = PROJECT_ROOT / "tests"
        for test_file in tests_dir.glob("*.py"):
            if test_file.name == Path(__file__).name:
                continue
            content = test_file.read_text()
            if version_check in content and tomli_ref in content:
                raise AssertionError(
                    f"{test_file.name} contains a sys.version_info check related to tomli. "
                    "With Python >=3.14, use 'import tomllib' directly."
                )


# ---------------------------------------------------------------------------
# AC5: ruff target-version is updated
# ---------------------------------------------------------------------------


class TestRuffTargetVersion:
    """AC5: ruff target-version must be updated from py310."""

    def test_ruff_target_version_not_py310(self, pyproject_data: dict) -> None:
        """ruff target-version must not be py310."""
        target = pyproject_data.get("tool", {}).get("ruff", {}).get("target-version", "")
        assert target != "py310", (
            "ruff target-version is still 'py310'. Update to 'py314' or the highest version ruff supports."
        )

    def test_ruff_target_version_is_py314_or_higher(self, pyproject_data: dict) -> None:
        """ruff target-version should be py314 (or highest supported)."""
        target = pyproject_data.get("tool", {}).get("ruff", {}).get("target-version", "")
        # Accept py312, py313, or py314 — whichever ruff supports
        acceptable = {"py312", "py313", "py314"}
        assert target in acceptable, f"ruff target-version is '{target}', expected one of {acceptable}."
