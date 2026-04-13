"""UNF-65 — Inpainting mode selector.

Outside-in TDD: template-level tests assert the dropdown renders with
three modes wired to per-mode visibility state; unit tests assert
``_build_generate_args`` enforces the forced parameter values for each
mode regardless of what the client sends for engine/strength/respective_field.

Reference: FwdFooocus ``modules/flags.py`` (mode constants) and
``FwdFooocus/webui.py::inpaint_mode_change`` (forced value matrix).
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


class TestGenerateArgsAppliesModeForcedValues:
    """``_build_generate_args`` must force engine/strength/respective_field
    per the mode table below, ignoring whatever the client posted:

    default → engine=body, strength=1.0, respective_field=0.618, disable_initial_latent=False
    detail  → engine='None', strength=0.5, respective_field=0.0,   disable_initial_latent=False
    modify  → engine=body,   strength=1.0, respective_field=0.0,   disable_initial_latent=True
    """

    def _args(self, body: dict) -> list:
        from ui.app import _build_generate_args

        return _build_generate_args(body)

    def _unpack(self, args: list) -> dict:
        from modules.async_worker import AsyncTask

        task = AsyncTask(list(args))
        return {
            "inpaint_disable_initial_latent": task.inpaint_disable_initial_latent,
            "inpaint_engine": task.inpaint_engine,
            "inpaint_strength": task.inpaint_strength,
            "inpaint_respective_field": task.inpaint_respective_field,
        }

    def test_default_mode_forces_values(self) -> None:
        fields = self._unpack(
            self._args(
                {
                    "prompt": "x",
                    "inpaint_mode": "default",
                    "inpaint_engine": "v2.6",
                    "inpaint_strength": 0.2,
                    "inpaint_respective_field": 0.9,
                    "inpaint_disable_initial_latent": True,
                }
            )
        )
        assert fields["inpaint_engine"] == "v2.6"
        assert fields["inpaint_strength"] == pytest.approx(1.0)
        assert fields["inpaint_respective_field"] == pytest.approx(0.618)
        assert fields["inpaint_disable_initial_latent"] is False

    def test_detail_mode_forces_values(self) -> None:
        fields = self._unpack(
            self._args(
                {
                    "prompt": "x",
                    "inpaint_mode": "detail",
                    "inpaint_engine": "v2.6",
                    "inpaint_strength": 0.9,
                    "inpaint_respective_field": 0.5,
                    "inpaint_disable_initial_latent": True,
                }
            )
        )
        assert fields["inpaint_engine"] == "None"
        assert fields["inpaint_strength"] == pytest.approx(0.5)
        assert fields["inpaint_respective_field"] == pytest.approx(0.0)
        assert fields["inpaint_disable_initial_latent"] is False

    def test_modify_mode_forces_values(self) -> None:
        fields = self._unpack(
            self._args(
                {
                    "prompt": "x",
                    "inpaint_mode": "modify",
                    "inpaint_engine": "v2.6",
                    "inpaint_strength": 0.2,
                    "inpaint_respective_field": 0.9,
                    "inpaint_disable_initial_latent": False,
                }
            )
        )
        assert fields["inpaint_engine"] == "v2.6"
        assert fields["inpaint_strength"] == pytest.approx(1.0)
        assert fields["inpaint_respective_field"] == pytest.approx(0.0)
        assert fields["inpaint_disable_initial_latent"] is True

    def test_omitted_mode_defaults_to_default(self) -> None:
        fields = self._unpack(self._args({"prompt": "x"}))
        assert fields["inpaint_strength"] == pytest.approx(1.0)
        assert fields["inpaint_respective_field"] == pytest.approx(0.618)
        assert fields["inpaint_disable_initial_latent"] is False

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="inpaint_mode"):
            self._args({"prompt": "x", "inpaint_mode": "nonsense"})
