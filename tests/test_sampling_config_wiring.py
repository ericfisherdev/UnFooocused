"""UNF-53: sampling/generation config pass-through into AsyncTask.

Verifies that values from AppConfig (default_sampler, default_scheduler,
default_cfg_scale, default_cfg_tsnr, default_sample_sharpness, default_clip_skip)
propagate through ui.app._build_generate_args() into the constructed AsyncTask.
"""

from __future__ import annotations

import dataclasses

import pytest


@pytest.fixture(autouse=True)
def _isolate_config_singleton():
    from modules.config import reset_config

    reset_config()
    yield
    reset_config()


def _inject_config(**overrides) -> None:
    """Swap the singleton with an AppConfig overriding sampling defaults."""
    from modules.config import get_config, set_config

    base = get_config()
    set_config(dataclasses.replace(base, **overrides))


class TestSamplingDefaultsReachAsyncTask:
    """UNF-53 AC3: Config sampling values pass through to AsyncTask fields."""

    def test_sampler_default_reaches_task(self):
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(default_sampler="euler")
        task = AsyncTask(_build_generate_args({"prompt": "x"}))
        assert task.sampler_name == "euler"

    def test_scheduler_default_reaches_task(self):
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(default_scheduler="exponential")
        task = AsyncTask(_build_generate_args({"prompt": "x"}))
        assert task.scheduler_name == "exponential"

    def test_cfg_scale_default_reaches_task(self):
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(default_cfg_scale=8.25)
        task = AsyncTask(_build_generate_args({"prompt": "x"}))
        assert task.cfg_scale == pytest.approx(8.25)

    def test_cfg_tsnr_default_reaches_task_adaptive_cfg(self):
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(default_cfg_tsnr=6.75)
        task = AsyncTask(_build_generate_args({"prompt": "x"}))
        assert task.adaptive_cfg == pytest.approx(6.75)

    def test_sharpness_default_reaches_task(self):
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(default_sample_sharpness=3.5)
        task = AsyncTask(_build_generate_args({"prompt": "x"}))
        assert task.sharpness == pytest.approx(3.5)

    def test_clip_skip_default_reaches_task(self):
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(default_clip_skip=5)
        task = AsyncTask(_build_generate_args({"prompt": "x"}))
        assert task.clip_skip == 5

    def test_request_body_overrides_config_defaults(self):
        """Explicit body values still win over config defaults."""
        from modules.async_worker import AsyncTask
        from ui.app import _build_generate_args

        _inject_config(
            default_sampler="euler",
            default_scheduler="karras",
            default_cfg_scale=7.0,
            default_cfg_tsnr=7.0,
            default_clip_skip=2,
        )
        body = {
            "prompt": "x",
            "sampler_name": "dpmpp_2m",
            "scheduler_name": "normal",
            "cfg_scale": 9.0,
            "adaptive_cfg": 4.0,
            "clip_skip": 1,
        }
        task = AsyncTask(_build_generate_args(body))
        assert task.sampler_name == "dpmpp_2m"
        assert task.scheduler_name == "normal"
        assert task.cfg_scale == pytest.approx(9.0)
        assert task.adaptive_cfg == pytest.approx(4.0)
        assert task.clip_skip == 1
