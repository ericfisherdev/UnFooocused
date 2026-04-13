"""UNF-68 — Advanced masking UI controls.

Outside-in TDD: template tests assert the advanced-masking toggle
renders the mask generation panel with all eight mask models, the
conditional cloth-category and SAM dropdowns, invert-mask, and a
separate mask-image upload. JS-level tests assert the Alpine state
and mask-combination helper are present. Backend regression tests
confirm ``_build_generate_args`` still passes ``inpaint_advanced_
masking_checkbox`` and ``invert_mask_checkbox`` through.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MASK_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
]
CLOTH_CATEGORIES = ["full", "upper", "lower"]
SAM_MODELS = ["vit_b", "vit_l", "vit_h"]


@pytest.fixture
def client() -> TestClient:
    from ui.app import app

    return TestClient(app)


class TestAdvancedMaskingTemplate:
    @pytest.fixture
    def html(self, client: TestClient) -> str:
        response = client.get("/")
        assert response.status_code == 200
        return response.text

    def test_toggle_checkbox_present(self, html: str) -> None:
        assert 'data-testid="inpaint-advanced-masking-checkbox"' in html
        assert 'x-model="inpaintAdvancedMaskingEnabled"' in html

    def test_mask_panel_gated_on_toggle(self, html: str) -> None:
        assert 'data-testid="inpaint-mask-generation-panel"' in html
        assert 'x-show="inpaintAdvancedMaskingEnabled"' in html

    def test_mask_model_select_present(self, html: str) -> None:
        assert 'data-testid="inpaint-mask-model-select"' in html
        assert 'x-model="inpaintMaskModel"' in html

    def test_mask_model_lists_all_eight_options(self, html: str) -> None:
        for model in MASK_MODELS:
            assert f'value="{model}"' in html

    def test_mask_model_default_is_isnet_general_use(self, html: str) -> None:
        # The Alpine state default is the source of truth — a template
        # <option selected> hint is optional.
        js = (Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "input-image-tabs.js").read_text()
        assert 'inpaintMaskModel: "isnet-general-use"' in js

    def test_cloth_category_select_present(self, html: str) -> None:
        assert 'data-testid="inpaint-mask-cloth-category-select"' in html
        for option in CLOTH_CATEGORIES:
            assert f'value="{option}"' in html

    def test_cloth_category_conditional_visibility(self, html: str) -> None:
        assert "inpaintMaskModel === 'u2net_cloth_seg'" in html

    def test_sam_model_select_present(self, html: str) -> None:
        assert 'data-testid="inpaint-mask-sam-model-select"' in html
        for option in SAM_MODELS:
            assert f'value="{option}"' in html

    def test_sam_model_conditional_visibility(self, html: str) -> None:
        assert "inpaintMaskModel === 'sam'" in html

    def test_invert_mask_checkbox_present(self, html: str) -> None:
        assert 'data-testid="inpaint-invert-mask"' in html
        assert 'x-model="inpaintInvertMask"' in html

    def test_mask_image_upload_present(self, html: str) -> None:
        assert 'data-testid="inpaint-mask-image-upload"' in html


class TestAlpineMaskingState:
    @pytest.fixture
    def js(self) -> str:
        path = Path(__file__).resolve().parents[1] / "ui" / "static" / "js" / "input-image-tabs.js"
        return path.read_text()

    def test_state_declares_masking_fields(self, js: str) -> None:
        for field in (
            "inpaintAdvancedMaskingEnabled",
            "inpaintInvertMask",
            "inpaintMaskModel",
            "inpaintMaskClothCategory",
            "inpaintMaskSamModel",
            "inpaintMaskImageUpload",
        ):
            assert field in js

    def test_masking_defaults(self, js: str) -> None:
        assert "inpaintAdvancedMaskingEnabled: false" in js
        assert "inpaintInvertMask: false" in js
        assert 'inpaintMaskClothCategory: "full"' in js
        assert 'inpaintMaskSamModel: "vit_b"' in js

    def test_combine_masks_helper_present(self, js: str) -> None:
        """Uploaded mask must be combined with the drawn mask via
        element-wise max (np.maximum semantics)."""
        assert "_combineMasks" in js or "combineMasks" in js

    def test_on_mask_upload_handler_present(self, js: str) -> None:
        assert "onInpaintMaskUpload" in js


class TestGenerateArgsMaskingPassthrough:
    def _unpack(self, body: dict) -> dict:
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        task = AsyncTask(list(_build_generate_args(body)))
        return {
            "inpaint_advanced_masking_checkbox": task.inpaint_advanced_masking_checkbox,
            "invert_mask_checkbox": task.invert_mask_checkbox,
        }

    def test_advanced_masking_checkbox_default_false(self) -> None:
        assert self._unpack({"prompt": "x"})["inpaint_advanced_masking_checkbox"] is False

    def test_invert_mask_checkbox_default_false(self) -> None:
        assert self._unpack({"prompt": "x"})["invert_mask_checkbox"] is False

    def test_advanced_masking_checkbox_passthrough_true(self) -> None:
        fields = self._unpack({"prompt": "x", "inpaint_advanced_masking_checkbox": True})
        assert fields["inpaint_advanced_masking_checkbox"] is True

    def test_invert_mask_checkbox_passthrough_true(self) -> None:
        fields = self._unpack({"prompt": "x", "invert_mask_checkbox": True})
        assert fields["invert_mask_checkbox"] is True
