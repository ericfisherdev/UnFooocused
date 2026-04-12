"""UNF-64 — Input Image tab with inpaint image upload.

Outside-in TDD: template-level tests assert the UI container exists and
wires upload + brush canvas; unit tests assert the API forwards
uov/inpaint image payloads through ``_build_generate_args`` instead of
hardcoding ``None``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from ui.app import app

    return TestClient(app)


class TestInputImageTemplate:
    """The rendered index page must expose the Input Image UI container."""

    @pytest.fixture
    def html(self, client: TestClient) -> str:
        response = client.get("/")
        assert response.status_code == 200
        return response.text

    def test_page_contains_input_image_section(self, html: str) -> None:
        assert 'data-testid="input-image-section"' in html

    def test_page_contains_three_input_image_subtabs(self, html: str) -> None:
        assert 'data-testid="input-image-tab-uov"' in html
        assert 'data-testid="input-image-tab-ip"' in html
        assert 'data-testid="input-image-tab-inpaint"' in html

    def test_subtab_labels_match_spec(self, html: str) -> None:
        assert "Upscale or Variation" in html
        assert "Image Prompt" in html
        assert "Inpaint or Outpaint" in html

    def test_inpaint_subtab_has_file_upload(self, html: str) -> None:
        assert 'data-testid="inpaint-image-upload"' in html
        assert 'type="file"' in html

    def test_inpaint_subtab_has_mask_canvas(self, html: str) -> None:
        assert 'data-testid="inpaint-mask-canvas"' in html
        assert "<canvas" in html

    def test_input_image_component_registered(self, html: str) -> None:
        assert "input-image-tabs.js" in html

    @pytest.mark.parametrize("slug", ["uov", "ip", "inpaint"])
    def test_tabpanel_aria_labelledby_matches_tab_id(self, html: str, slug: str) -> None:
        btn_id = f'id="input-image-tab-{slug}-btn"'
        panel_labelled = f'aria-labelledby="input-image-tab-{slug}-btn"'
        aria_controls = f'aria-controls="input-image-panel-{slug}"'
        assert btn_id in html
        assert panel_labelled in html
        assert aria_controls in html


class TestGenerateArgsForwardsInputImages:
    """``_build_generate_args`` must forward inpaint + uov payloads from
    the request body instead of hardcoding them to ``None`` — otherwise
    the UI cannot submit an uploaded image to the backend."""

    def _args(self, body: dict) -> list:
        from ui.app import _build_generate_args

        return _build_generate_args(body)

    def _unpack_input_image_slice(self, args: list) -> dict:
        """Extract the input-image positional slice using the known order.

        Mirrors ``AsyncTask.__init__``: after base config, loras, and three
        scalar inputs (checkbox, current_tab, uov_method) we hit the image
        payloads.
        """
        from modules.async_worker import AsyncTask

        task = AsyncTask(list(args))
        return {
            "input_image_checkbox": task.input_image_checkbox,
            "current_tab": task.current_tab,
            "uov_method": task.uov_method,
            "uov_input_image": task.uov_input_image,
            "inpaint_input_image": task.inpaint_input_image,
        }

    def test_inpaint_input_image_forwarded_from_body(self) -> None:
        payload = {"image": "data:image/png;base64,AAAA", "mask": "data:image/png;base64,BBBB"}
        args = self._args({"prompt": "x", "inpaint_input_image": payload})
        fields = self._unpack_input_image_slice(args)
        assert fields["inpaint_input_image"] == payload

    def test_uov_input_image_forwarded_from_body(self) -> None:
        payload = {"image": "data:image/png;base64,CCCC"}
        args = self._args({"prompt": "x", "uov_input_image": payload})
        fields = self._unpack_input_image_slice(args)
        assert fields["uov_input_image"] == payload

    def test_input_image_checkbox_defaults_false_when_omitted(self) -> None:
        args = self._args({"prompt": "x"})
        fields = self._unpack_input_image_slice(args)
        assert fields["input_image_checkbox"] is False
        assert fields["inpaint_input_image"] is None
        assert fields["uov_input_image"] is None

    def test_current_tab_forwarded_from_body(self) -> None:
        args = self._args({"prompt": "x", "current_tab": "inpaint", "input_image_checkbox": True})
        fields = self._unpack_input_image_slice(args)
        assert fields["current_tab"] == "inpaint"
        assert fields["input_image_checkbox"] is True
