"""Tests for UNF-15: Rename FwdFooocus references to UnFooocused.

Acceptance criteria encoded as tests:
1. FastAPI app.title is 'UnFooocused'
2. HTML <title> tag says 'UnFooocused'
3. No docstrings reference FwdFooocus as the current product name
4. grep -r 'FwdFooocus' finds zero matches in .py and .html files
   (excluding comments explaining migration history)
5. All existing tests still pass after renaming
"""

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFastAPIAppTitle:
    """AC-1: FastAPI app.title is 'UnFooocused'."""

    def test_app_title_is_unfooocused(self):
        from ui.app import app

        assert app.title == "UnFooocused"


class TestHTMLTitle:
    """AC-2: HTML <title> tag says 'UnFooocused'."""

    def test_base_html_title_is_unfooocused(self):
        base_html = PROJECT_ROOT / "ui" / "templates" / "base.html"
        content = base_html.read_text()
        match = re.search(r"<title>(.*?)</title>", content)
        assert match is not None, "<title> tag not found in base.html"
        assert match.group(1) == "UnFooocused"


class TestDocstringsDoNotReferenceFwdFooocus:
    """AC-3: No docstrings reference FwdFooocus as the current product name."""

    @staticmethod
    def _collect_docstrings(filepath: Path) -> list[tuple[int, str]]:
        """Extract all docstrings from a Python file with their line numbers."""
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    lineno = getattr(node, "lineno", 0)
                    results.append((lineno, docstring))
        return results

    @staticmethod
    def _python_files() -> list[Path]:
        """Return all .py files in the project (excluding .venv and this test file)."""
        this_file = Path(__file__).resolve()
        return [
            p
            for p in PROJECT_ROOT.rglob("*.py")
            if ".venv" not in p.parts and "__pycache__" not in p.parts and p.resolve() != this_file
        ]

    def test_no_docstrings_reference_fwdfooocus(self):
        violations = []
        for filepath in self._python_files():
            for lineno, docstring in self._collect_docstrings(filepath):
                if "FwdFooocus" in docstring:
                    rel = filepath.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel}:{lineno}")
        assert violations == [], f"Docstrings still reference FwdFooocus: {violations}"


class TestNoFwdFooocusInSourceFiles:
    """AC-4: No FwdFooocus references in .py, .html, or .css files.

    Excludes:
    - Comments that explain migration history (intentional context)
    - Test files (this file references FwdFooocus in test names/assertions)
    - .venv directory
    """

    PATTERN = re.compile(r"FwdFooocus|Fwd Fooocus")

    @staticmethod
    def _source_files() -> list[Path]:
        """Collect all .py, .html, and .css files, excluding tests and .venv."""
        extensions = {".py", ".html", ".css"}
        return [
            p
            for p in PROJECT_ROOT.rglob("*")
            if p.suffix in extensions
            and ".venv" not in p.parts
            and "__pycache__" not in p.parts
            and "tests" not in p.parts
        ]

    @staticmethod
    def _is_migration_comment(line: str) -> bool:
        """True if the line is a comment explaining migration history."""
        stripped = line.strip()
        # Python single-line comments
        if stripped.startswith("#") and ("migrat" in stripped.lower() or "from fwdfooocus" in stripped.lower()):
            return True
        # CSS comments containing migration context
        if "/*" in stripped and ("migrat" in stripped.lower() or "from fwdfooocus" in stripped.lower()):
            return True
        # HTML comments
        return "<!--" in stripped and ("migrat" in stripped.lower() or "from fwdfooocus" in stripped.lower())

    def test_no_fwdfooocus_in_source_files(self):
        violations = []
        for filepath in self._source_files():
            for lineno, line in enumerate(filepath.read_text().splitlines(), start=1):
                if self.PATTERN.search(line) and not self._is_migration_comment(line):
                    rel = filepath.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel}:{lineno}: {line.strip()}")
        assert violations == [], "FwdFooocus references found in source files:\n" + "\n".join(violations)
