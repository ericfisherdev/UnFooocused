"""Tests for pyproject.toml dependency declarations (UNF-22).

Acceptance criteria:
  AC1: pip install -e . in a clean venv installs safetensors and Pillow
       without manual intervention.
  AC2: import modules.lora_metadata and import modules.output succeed
       after a clean install.
  AC3: All existing tests continue to pass.

AC3 is verified by the full test suite run, not a dedicated test here.
AC1 and AC2 are tested by parsing pyproject.toml and verifying the
declared dependencies include safetensors and Pillow.

We test at the packaging metadata level (parsing pyproject.toml) rather
than spawning subprocesses with clean venvs, because:
  - The dependency resolver (pip) is a managed dependency we trust
  - What we control is the declaration in pyproject.toml
  - If the declaration is correct, pip install -e . will work
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


@pytest.fixture()
def pyproject_data() -> dict:
    """Parse pyproject.toml and return the data dictionary."""
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


@pytest.fixture()
def declared_dependencies(pyproject_data: dict) -> list[str]:
    """Return the list of declared runtime dependencies (lowercased)."""
    raw = pyproject_data.get("project", {}).get("dependencies", [])
    return [dep.lower() for dep in raw]


class TestSafetensorsDependency:
    """AC1: safetensors must be a declared runtime dependency."""

    def test_safetensors_in_runtime_dependencies(self, declared_dependencies: list[str]) -> None:
        """safetensors must appear in [project].dependencies."""
        matches = [d for d in declared_dependencies if d.startswith("safetensors")]
        assert matches, (
            "safetensors is not declared in [project].dependencies. "
            "modules.lora_metadata imports safetensors.safe_open at runtime."
        )

    def test_safetensors_has_minimum_version(self, declared_dependencies: list[str]) -> None:
        """safetensors dependency must specify a minimum version."""
        matches = [d for d in declared_dependencies if d.startswith("safetensors")]
        assert matches, "safetensors not found in dependencies"
        dep = matches[0]
        assert ">=" in dep, (
            f"safetensors dependency '{dep}' lacks a minimum version constraint. Use safetensors>=0.4 or similar."
        )


class TestPillowDependency:
    """AC1: Pillow must be a declared runtime dependency."""

    def test_pillow_in_runtime_dependencies(self, declared_dependencies: list[str]) -> None:
        """Pillow must appear in [project].dependencies."""
        matches = [d for d in declared_dependencies if d.startswith("pillow")]
        assert matches, (
            "Pillow is not declared in [project].dependencies. "
            "modules.output and modules.async_worker import PIL at runtime."
        )

    def test_pillow_has_minimum_version(self, declared_dependencies: list[str]) -> None:
        """Pillow dependency must specify a minimum version."""
        matches = [d for d in declared_dependencies if d.startswith("pillow")]
        assert matches, "Pillow not found in dependencies"
        dep = matches[0]
        assert ">=" in dep, (
            f"Pillow dependency '{dep}' lacks a minimum version constraint. Use Pillow>=10.0 or similar."
        )


class TestModuleImportability:
    """AC2: modules that depend on safetensors/Pillow must be importable."""

    def test_lora_metadata_importable(self) -> None:
        """modules.lora_metadata must import without error.

        Note: The safetensors import is deferred (lazy), so this test
        verifies the module-level import succeeds. The lazy wrapper
        _safe_open is only called when actually reading .safetensors files.
        """
        import modules.lora_metadata  # noqa: F401

    def test_output_importable(self) -> None:
        """modules.output must import without error."""
        import modules.output  # noqa: F401

    def test_async_worker_importable(self) -> None:
        """modules.async_worker must import without error."""
        import modules.async_worker  # noqa: F401
