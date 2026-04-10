"""Tests for vendored ldm_patched package.

Validates acceptance criteria for UNF-30:
1. ldm_patched/ is importable as a Python package
2. ldm_patched.modules.sd.load_checkpoint_guess_config is callable
3. ldm_patched.modules.model_management.get_torch_device returns a torch.device
4. ldm_patched.modules.samplers exposes sampler and scheduler names
5. torch, einops, safetensors, transformers are declared in pyproject.toml
6. No upstream-specific references remain in vendored code
7. Existing tests continue to pass (covered by running full suite)
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

_torch_available = bool(importlib.util.find_spec("torch"))
requires_torch = pytest.mark.skipif(not _torch_available, reason="torch not available")


class TestLdmPatchedImportable:
    """AC-1: ldm_patched/ directory exists and is importable."""

    def test_ldm_patched_directory_exists(self) -> None:
        ldm_dir = PROJECT_ROOT / "ldm_patched"
        assert ldm_dir.is_dir(), f"ldm_patched/ directory not found at {ldm_dir}"

    def test_ldm_patched_has_init(self) -> None:
        init_file = PROJECT_ROOT / "ldm_patched" / "__init__.py"
        assert init_file.exists(), "ldm_patched/__init__.py not found"

    def test_ldm_patched_importable(self) -> None:
        import ldm_patched

        assert ldm_patched is not None


class TestSdModuleCallable:
    """AC-2: ldm_patched.modules.sd.load_checkpoint_guess_config is callable."""

    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch", reason="torch required for sd module tests")

    def test_load_checkpoint_guess_config_exists(self) -> None:
        from ldm_patched.modules import sd

        assert hasattr(sd, "load_checkpoint_guess_config")

    def test_load_checkpoint_guess_config_is_callable(self) -> None:
        from ldm_patched.modules.sd import load_checkpoint_guess_config

        assert callable(load_checkpoint_guess_config)


class TestModelManagement:
    """AC-3: get_torch_device returns a torch.device."""

    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch", reason="torch required for model_management tests")

    def test_get_torch_device_exists(self) -> None:
        from ldm_patched.modules import model_management

        assert hasattr(model_management, "get_torch_device")

    def test_get_torch_device_returns_torch_device(self) -> None:
        import torch
        from ldm_patched.modules.model_management import get_torch_device

        device = get_torch_device()
        assert isinstance(device, torch.device)


class TestSamplerNames:
    """AC-4: samplers module exposes sampler and scheduler names."""

    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch", reason="torch required for samplers tests")

    def test_sampler_names_exists(self) -> None:
        from ldm_patched.modules import samplers

        assert hasattr(samplers, "SAMPLER_NAMES")

    def test_sampler_names_is_nonempty_list(self) -> None:
        from ldm_patched.modules.samplers import SAMPLER_NAMES

        assert isinstance(SAMPLER_NAMES, list)
        assert len(SAMPLER_NAMES) > 0

    def test_scheduler_names_exists(self) -> None:
        from ldm_patched.modules import samplers

        assert hasattr(samplers, "SCHEDULER_NAMES")

    def test_scheduler_names_is_nonempty_list(self) -> None:
        from ldm_patched.modules.samplers import SCHEDULER_NAMES

        assert isinstance(SCHEDULER_NAMES, list)
        assert len(SCHEDULER_NAMES) > 0

    def test_ksampler_class_exists(self) -> None:
        from ldm_patched.modules.samplers import KSampler

        assert KSampler is not None


class TestDependenciesDeclared:
    """AC-5: Required dependencies declared in pyproject.toml."""

    def _read_pyproject_dependencies(self) -> list[str]:
        """Read the dependencies list from pyproject.toml."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        return data.get("project", {}).get("dependencies", [])

    def _extract_dep_names(self, deps: list[str]) -> list[str]:
        return [d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower() for d in deps]

    def test_torch_declared(self) -> None:
        deps = self._read_pyproject_dependencies()
        dep_names = self._extract_dep_names(deps)
        assert "torch" in dep_names, f"torch not found in dependencies: {deps}"

    def test_einops_declared(self) -> None:
        deps = self._read_pyproject_dependencies()
        dep_names = self._extract_dep_names(deps)
        assert "einops" in dep_names, f"einops not found in dependencies: {deps}"

    def test_safetensors_declared(self) -> None:
        deps = self._read_pyproject_dependencies()
        dep_names = self._extract_dep_names(deps)
        assert "safetensors" in dep_names, f"safetensors not found in dependencies: {deps}"

    def test_transformers_declared(self) -> None:
        deps = self._read_pyproject_dependencies()
        dep_names = self._extract_dep_names(deps)
        assert "transformers" in dep_names, f"transformers not found in dependencies: {deps}"


class TestNoUpstreamReferences:
    """AC-6: No upstream-specific references remain in vendored code."""

    def _grep_ldm_patched(self, pattern: str) -> str:
        """Search ldm_patched/ for a pattern, return matching lines."""
        ldm_dir = PROJECT_ROOT / "ldm_patched"
        if not ldm_dir.is_dir():
            pytest.skip("ldm_patched/ not yet vendored")

        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, str(ldm_dir)],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_no_args_manager_references(self) -> None:
        matches = self._grep_ldm_patched("args_manager")
        assert matches == "", f"Found args_manager references:\n{matches}"

    def test_no_fwdfooocus_shared_references(self) -> None:
        """No imports of upstream 'shared' module."""
        matches = self._grep_ldm_patched(r"from modules\.shared\|import modules\.shared\|from shared import")
        assert matches == "", f"Found shared module references:\n{matches}"

    def test_no_gradio_references(self) -> None:
        matches = self._grep_ldm_patched(r"import gradio\|from gradio")
        assert matches == "", f"Found gradio references:\n{matches}"

    def test_no_fwdfooocus_config_references(self) -> None:
        """No imports of upstream 'modules.config'."""
        matches = self._grep_ldm_patched(r"from modules\.config\|import modules\.config")
        assert matches == "", f"Found upstream modules.config references:\n{matches}"
