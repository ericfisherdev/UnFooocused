"""UNF-67 — Advanced inpainting controls.

The mode selector drives *client-side* default values for the advanced
sliders; the user can then override any slider before submitting. The
server is the consumer of whatever the user actually sent — no server-
side forcing. This reverts the UNF-65 server resolver and exposes five
sliders/checkbox/select controls bound to Alpine state.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from ui.app import app

    return TestClient(app)


class TestAdvancedInpaintTemplate:
    """The inpaint panel must expose five advanced controls bound to
    Alpine state: engine select, denoising strength slider, respective
    field slider, erode/dilate slider, and disable-initial-latent
    checkbox."""

    @pytest.fixture
    def html(self, client: TestClient) -> str:
        response = client.get("/")
        assert response.status_code == 200
        return response.text

    def test_engine_select_present(self, html: str) -> None:
        assert 'data-testid="inpaint-engine-select"' in html
        assert 'x-model="inpaintEngine"' in html

    def test_engine_select_lists_all_versions(self, html: str) -> None:
        for version in ("None", "v1", "v2.5", "v2.6"):
            assert f'value="{version}"' in html

    def test_strength_slider_present(self, html: str) -> None:
        assert 'data-testid="inpaint-strength-slider"' in html
        assert 'x-model.number="inpaintStrength"' in html

    def test_respective_field_slider_present(self, html: str) -> None:
        assert 'data-testid="inpaint-respective-field-slider"' in html
        assert 'x-model.number="inpaintRespectiveField"' in html

    def test_erode_or_dilate_slider_present(self, html: str) -> None:
        assert 'data-testid="inpaint-erode-or-dilate-slider"' in html
        assert 'x-model.number="inpaintErodeOrDilate"' in html

    def test_disable_initial_latent_checkbox_present(self, html: str) -> None:
        assert 'data-testid="inpaint-disable-initial-latent"' in html
        assert 'x-model="inpaintDisableInitialLatent"' in html


class TestAlpineModeWatcher:
    """Mode change is a client-side setter of slider initial values."""

    @pytest.fixture
    def js(self) -> str:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "input-image-tabs.js"
        return path.read_text()

    def test_watch_inpaint_mode_registered(self, js: str) -> None:
        assert '$watch("inpaintMode"' in js or "$watch('inpaintMode'" in js

    def test_alpine_state_declares_advanced_fields(self, js: str) -> None:
        for field in (
            "inpaintEngine",
            "inpaintStrength",
            "inpaintRespectiveField",
            "inpaintErodeOrDilate",
            "inpaintDisableInitialLatent",
        ):
            assert field in js

    def test_detail_mode_branch_sets_strength_and_field(self, js: str) -> None:
        assert '"detail"' in js or "'detail'" in js
        assert "0.5" in js

    def test_modify_mode_branch_sets_disable_initial_latent(self, js: str) -> None:
        assert '"modify"' in js or "'modify'" in js


class TestGenerateArgsReadsRawBodyValues:
    """``_build_generate_args`` must pass through user-supplied values
    for the five advanced inpaint fields. No per-mode server forcing."""

    def _unpack(self, body: dict) -> dict:
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        task = AsyncTask(list(_build_generate_args(body)))
        return {
            "inpaint_disable_initial_latent": task.inpaint_disable_initial_latent,
            "inpaint_engine": task.inpaint_engine,
            "inpaint_strength": task.inpaint_strength,
            "inpaint_respective_field": task.inpaint_respective_field,
            "inpaint_erode_or_dilate": task.inpaint_erode_or_dilate,
        }

    def test_reads_engine_from_body(self) -> None:
        fields = self._unpack({"prompt": "x", "inpaint_engine": "v1"})
        assert fields["inpaint_engine"] == "v1"

    def test_reads_strength_from_body(self) -> None:
        fields = self._unpack({"prompt": "x", "inpaint_strength": 0.42})
        assert fields["inpaint_strength"] == pytest.approx(0.42)

    def test_reads_respective_field_from_body(self) -> None:
        fields = self._unpack({"prompt": "x", "inpaint_respective_field": 0.33})
        assert fields["inpaint_respective_field"] == pytest.approx(0.33)

    def test_reads_erode_or_dilate_from_body(self) -> None:
        fields = self._unpack({"prompt": "x", "inpaint_erode_or_dilate": 17})
        assert fields["inpaint_erode_or_dilate"] == 17

    def test_reads_disable_initial_latent_from_body(self) -> None:
        fields = self._unpack({"prompt": "x", "inpaint_disable_initial_latent": True})
        assert fields["inpaint_disable_initial_latent"] is True

    def test_engine_default_from_config(self) -> None:
        import modules.config as config

        fields = self._unpack({"prompt": "x"})
        assert fields["inpaint_engine"] == config.get_config().default_inpaint_engine_version

    def test_strength_default_is_one(self) -> None:
        fields = self._unpack({"prompt": "x"})
        assert fields["inpaint_strength"] == pytest.approx(1.0)

    def test_respective_field_default_is_0_618(self) -> None:
        fields = self._unpack({"prompt": "x"})
        assert fields["inpaint_respective_field"] == pytest.approx(0.618)

    def test_erode_or_dilate_default_is_zero(self) -> None:
        fields = self._unpack({"prompt": "x"})
        assert fields["inpaint_erode_or_dilate"] == 0

    def test_disable_initial_latent_default_is_false(self) -> None:
        fields = self._unpack({"prompt": "x"})
        assert fields["inpaint_disable_initial_latent"] is False

    def test_mode_field_does_not_override_user_values(self) -> None:
        """The UNF-65 server resolver is gone. inpaint_mode in the body
        must not clobber any of the advanced fields."""
        fields = self._unpack(
            {
                "prompt": "x",
                "inpaint_mode": "detail",
                "inpaint_engine": "v2.6",
                "inpaint_strength": 0.9,
                "inpaint_respective_field": 0.7,
                "inpaint_disable_initial_latent": True,
            }
        )
        assert fields["inpaint_engine"] == "v2.6"
        assert fields["inpaint_strength"] == pytest.approx(0.9)
        assert fields["inpaint_respective_field"] == pytest.approx(0.7)
        assert fields["inpaint_disable_initial_latent"] is True
