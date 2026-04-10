"""Unit tests for domain types — UNF-17 acceptance criteria.

Tests are organized by acceptance criterion:
1. SessionState TypedDict exists defining the session state schema
2. save_state() and load_state() use the typed session state
3. All functions have complete type annotations (params and return types)
4. BaseModelFamily NewType prevents passing arbitrary strings
5. Heartbeat module has complete type annotations
"""

import typing
from typing import get_type_hints


class TestSessionStateTypedDictExists:
    """AC1: SessionState TypedDict exists defining the session state schema."""

    def test_session_state_is_importable(self) -> None:
        from modules.session_state import SessionState

        assert SessionState is not None

    def test_session_state_is_typed_dict(self) -> None:
        from modules.session_state import SessionState

        # TypedDicts are subclasses of dict at runtime
        assert issubclass(SessionState, dict)

    def test_session_state_has_prompt_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "prompt" in hints

    def test_session_state_has_negative_prompt_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "negative_prompt" in hints

    def test_session_state_has_styles_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "styles" in hints

    def test_session_state_has_loras_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "loras" in hints

    def test_session_state_has_sampler_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "sampler" in hints

    def test_session_state_has_scheduler_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "scheduler" in hints

    def test_session_state_has_cfg_scale_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "cfg_scale" in hints

    def test_session_state_has_sharpness_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "sharpness" in hints

    def test_session_state_has_steps_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "steps" in hints

    def test_session_state_has_seed_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "seed" in hints

    def test_session_state_has_base_model_name_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "base_model_name" in hints

    def test_session_state_has_refiner_model_name_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "refiner_model_name" in hints

    def test_session_state_has_performance_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "performance" in hints

    def test_session_state_has_image_number_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "image_number" in hints

    def test_session_state_has_aspect_ratios_selection_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "aspect_ratios_selection" in hints

    def test_session_state_has_vae_name_field(self) -> None:
        from modules.session_state import SessionState

        hints = get_type_hints(SessionState)
        assert "vae_name" in hints


class TestBaseModelFamilyNewType:
    """AC4 (impl plan item 4): BaseModelFamily NewType exists."""

    def test_base_model_family_is_importable(self) -> None:
        from modules.session_state import BaseModelFamily

        assert BaseModelFamily is not None

    def test_base_model_family_creates_str_value(self) -> None:
        from modules.session_state import BaseModelFamily

        value = BaseModelFamily("pony")
        # NewType returns the value unchanged at runtime
        assert value == "pony"
        assert isinstance(value, str)


class TestSaveStateUsesTypedSignature:
    """AC2: save_state() uses the typed session state, not dict[str, Any]."""

    def test_save_state_base_model_param_is_base_model_family(self) -> None:
        from modules.session_state import BaseModelFamily, save_state

        hints = get_type_hints(save_state)
        assert hints["base_model"] is BaseModelFamily

    def test_save_state_state_param_is_session_state(self) -> None:
        from modules.session_state import SessionState, save_state

        hints = get_type_hints(save_state)
        assert hints["state"] is SessionState

    def test_save_state_return_type_is_none(self) -> None:
        from modules.session_state import save_state

        hints = get_type_hints(save_state)
        assert hints["return"] is type(None)


class TestLoadStateUsesTypedSignature:
    """AC2: load_state() uses the typed session state, not dict[str, Any]."""

    def test_load_state_base_model_param_is_base_model_family(self) -> None:
        from modules.session_state import BaseModelFamily, load_state

        hints = get_type_hints(load_state)
        assert hints["base_model"] is BaseModelFamily

    def test_load_state_return_type_includes_session_state(self) -> None:
        from modules.session_state import SessionState, load_state

        hints = get_type_hints(load_state)
        return_type = hints["return"]
        # Should be SessionState | None
        args = typing.get_args(return_type)
        assert SessionState in args
        assert type(None) in args


class TestGetConnectionAnnotations:
    """AC3: _get_connection has complete type annotations."""

    def test_get_connection_has_return_annotation(self) -> None:
        from modules.session_state import _get_connection

        hints = get_type_hints(_get_connection)
        assert "return" in hints


class TestHeartbeatAnnotations:
    """AC3/5: All functions in heartbeat.py have complete type annotations."""

    def test_update_heartbeat_return_type(self) -> None:
        from modules.heartbeat import update_heartbeat

        hints = get_type_hints(update_heartbeat)
        assert hints["return"] is type(None)

    def test_is_browser_connected_return_type(self) -> None:
        from modules.heartbeat import is_browser_connected

        hints = get_type_hints(is_browser_connected)
        assert hints["return"] is bool

    def test_is_browser_connected_timeout_param_type(self) -> None:
        from modules.heartbeat import is_browser_connected

        hints = get_type_hints(is_browser_connected)
        assert hints["timeout_seconds"] is float

    def test_last_heartbeat_time_has_type_annotation(self) -> None:
        """Module-level variable _last_heartbeat_time should be annotated."""
        import modules.heartbeat as hb

        module_annotations = get_type_hints(hb)
        assert "_last_heartbeat_time" in module_annotations


class TestLoraEntryTypedDict:
    """LoraEntry TypedDict should exist to type the lora list entries."""

    def test_lora_entry_is_importable(self) -> None:
        from modules.session_state import LoraEntry

        assert LoraEntry is not None

    def test_lora_entry_is_typed_dict(self) -> None:
        from modules.session_state import LoraEntry

        assert issubclass(LoraEntry, dict)

    def test_lora_entry_has_name_field(self) -> None:
        from modules.session_state import LoraEntry

        hints = get_type_hints(LoraEntry)
        assert "name" in hints

    def test_lora_entry_has_weight_field(self) -> None:
        from modules.session_state import LoraEntry

        hints = get_type_hints(LoraEntry)
        assert "weight" in hints
