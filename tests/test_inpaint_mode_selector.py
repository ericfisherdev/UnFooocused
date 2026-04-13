"""UNF-65 — Inpainting mode selector (template-only).

Mode change is a client-side setter of advanced-control default values;
the server reads whatever the user submitted. Backend per-mode forcing
was reverted in UNF-67 per architectural realignment — advanced slider
behaviour is covered by ``test_advanced_inpaint_controls.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

MODE_DEFAULT = "Inpaint or Outpaint (default)"
MODE_DETAIL = "Improve Detail (face, hand, eyes, etc.)"
MODE_MODIFY = "Modify Content (add objects, change background, etc.)"


@pytest.fixture
def client() -> TestClient:
    from ui.app import app

    return TestClient(app)


class TestInpaintModeTemplate:
    """The inpaint panel must expose a mode selector bound to Alpine state
    so the template can toggle outpaint controls vs the additional prompt."""

    @pytest.fixture
    def html(self, client: TestClient) -> str:
        response = client.get("/")
        assert response.status_code == 200
        return response.text

    def test_mode_select_present(self, html: str) -> None:
        assert 'data-testid="inpaint-mode-select"' in html

    def test_mode_select_has_three_options(self, html: str) -> None:
        assert MODE_DEFAULT in html
        assert MODE_DETAIL in html
        assert MODE_MODIFY in html

    def test_mode_select_bound_to_alpine_state(self, html: str) -> None:
        assert 'x-model="inpaintMode"' in html

    def test_additional_prompt_visibility_bound_to_detail_or_modify(self, html: str) -> None:
        assert 'data-testid="inpaint-additional-prompt"' in html
        assert "inpaintMode === 'detail'" in html
        assert "inpaintMode === 'modify'" in html

    def test_outpaint_controls_visibility_bound_to_default_mode(self, html: str) -> None:
        assert 'data-testid="inpaint-outpaint-controls"' in html
        assert "inpaintMode === 'default'" in html

    def test_outpaint_direction_checkboxes_present(self, html: str) -> None:
        for direction in ("left", "right", "top", "bottom"):
            assert f'data-testid="outpaint-direction-{direction}"' in html
