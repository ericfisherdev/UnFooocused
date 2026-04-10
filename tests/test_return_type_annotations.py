"""Tests for UNF-21: every function in ui/app.py must have a return type annotation.

Acceptance criteria:
  1. Every def and async def in ui/app.py has an explicit -> return type annotation.
  2. All existing tests pass without modification.
  3. ruff lint passes with no new warnings.

These tests encode criterion 1 by parsing the AST and inspecting each
function definition for the presence of a `returns` annotation.
"""

import ast
from pathlib import Path

import pytest

APP_MODULE_PATH = Path(__file__).resolve().parents[1] / "ui" / "app.py"


def _get_function_defs() -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Parse ui/app.py and return all top-level and nested function definitions."""
    source = APP_MODULE_PATH.read_text()
    tree = ast.parse(source, filename=str(APP_MODULE_PATH))
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(node)
    return functions


class TestReturnTypeAnnotations:
    """Every function in ui/app.py must declare an explicit return type annotation."""

    def test_all_functions_have_return_annotations(self):
        """AC-1: Every def and async def in ui/app.py has an explicit -> return type."""
        functions = _get_function_defs()
        assert len(functions) > 0, "No functions found in ui/app.py — AST parsing may be broken"

        missing = [f"{f.name} (line {f.lineno})" for f in functions if f.returns is None]
        assert not missing, "Functions missing return type annotations:\n  " + "\n  ".join(missing)

    @pytest.mark.parametrize(
        "func_name",
        [
            "index",
            "lora_library_rescan",
            "lora_library_scan_status",
            "lora_library_data",
            "lora_trigger_words",
            "heartbeat_ping",
            "get_app_config",
            "get_models",
            "get_styles",
            "get_samplers",
            "generate",
            "generate_stop",
            "ws_generation",
        ],
    )
    def test_endpoint_function_has_return_annotation(self, func_name: str):
        """Each endpoint function must have a return type annotation."""
        functions = _get_function_defs()
        func = next((f for f in functions if f.name == func_name), None)
        assert func is not None, f"Function {func_name!r} not found in ui/app.py"
        assert func.returns is not None, (
            f"Endpoint {func_name!r} (line {func.lineno}) is missing a return type annotation"
        )

    @pytest.mark.parametrize(
        "func_name",
        [
            "_find_processing_task",
            "_drain_remaining_yields",
        ],
    )
    def test_helper_function_has_return_annotation(self, func_name: str):
        """Internal helpers that were missing return annotations must now have them."""
        functions = _get_function_defs()
        func = next((f for f in functions if f.name == func_name), None)
        assert func is not None, f"Function {func_name!r} not found in ui/app.py"
        assert func.returns is not None, (
            f"Helper {func_name!r} (line {func.lineno}) is missing a return type annotation"
        )


class TestParameterTypeAnnotations:
    """_find_processing_task and _drain_remaining_yields must have fully typed parameters."""

    def test_find_processing_task_params_are_typed(self):
        """AC-1 (extended): _find_processing_task parameters must have type annotations."""
        functions = _get_function_defs()
        func = next((f for f in functions if f.name == "_find_processing_task"), None)
        assert func is not None

        untyped = [arg.arg for arg in func.args.args if arg.annotation is None]
        assert not untyped, f"_find_processing_task has untyped parameters: {untyped}"

    def test_drain_remaining_yields_params_are_typed(self):
        """AC-1 (extended): _drain_remaining_yields parameters must have type annotations."""
        functions = _get_function_defs()
        func = next((f for f in functions if f.name == "_drain_remaining_yields"), None)
        assert func is not None

        untyped = [arg.arg for arg in func.args.args if arg.annotation is None]
        assert not untyped, f"_drain_remaining_yields has untyped parameters: {untyped}"
