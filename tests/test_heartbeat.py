"""Unit tests for modules.heartbeat — inner-loop TDD.

Each test drives one behavior. Tests are ordered by TDD cycle:
1. update_heartbeat() sets the last heartbeat time to approximately now
2. is_browser_connected() returns True immediately after update_heartbeat()
3. is_browser_connected() returns False when timeout has elapsed
4. is_browser_connected() respects custom timeout_seconds parameter
5. is_browser_connected() default timeout is 15 seconds
"""

import modules.heartbeat as heartbeat_module


class TestUpdateHeartbeat:
    """Cycle 1: update_heartbeat() records the current time."""

    def test_sets_last_heartbeat_to_current_time(self, monkeypatch):
        fake_now = 1000.0
        monkeypatch.setattr(heartbeat_module, "time", _FakeTimeModule(fake_now))

        heartbeat_module.update_heartbeat()

        assert heartbeat_module._last_heartbeat_time == fake_now


class TestIsBrowserConnectedAfterHeartbeat:
    """Cycle 2: is_browser_connected() returns True right after a heartbeat."""

    def test_returns_true_immediately_after_heartbeat(self, monkeypatch):
        fake_now = 1000.0
        fake_time = _FakeTimeModule(fake_now)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        # Time has not advanced — should be connected
        assert heartbeat_module.is_browser_connected() is True


class TestIsBrowserConnectedTimeout:
    """Cycle 3: is_browser_connected() returns False after timeout elapses."""

    def test_returns_false_when_timeout_elapsed(self, monkeypatch):
        fake_time = _FakeTimeModule(1000.0)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        # Advance time beyond the default 15-second timeout
        fake_time.advance(16.0)

        assert heartbeat_module.is_browser_connected() is False


class TestIsBrowserConnectedCustomTimeout:
    """Cycle 4: is_browser_connected() respects a custom timeout_seconds."""

    def test_connected_within_custom_timeout(self, monkeypatch):
        fake_time = _FakeTimeModule(1000.0)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        fake_time.advance(4.0)

        assert heartbeat_module.is_browser_connected(timeout_seconds=5.0) is True

    def test_disconnected_beyond_custom_timeout(self, monkeypatch):
        fake_time = _FakeTimeModule(1000.0)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        fake_time.advance(6.0)

        assert heartbeat_module.is_browser_connected(timeout_seconds=5.0) is False


class TestIsBrowserConnectedDefaultTimeout:
    """Cycle 5: The default timeout is exactly 15 seconds."""

    def test_connected_at_14_seconds(self, monkeypatch):
        fake_time = _FakeTimeModule(1000.0)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        fake_time.advance(14.0)

        assert heartbeat_module.is_browser_connected() is True

    def test_disconnected_at_16_seconds(self, monkeypatch):
        fake_time = _FakeTimeModule(1000.0)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        fake_time.advance(16.0)

        assert heartbeat_module.is_browser_connected() is False

    def test_boundary_disconnected_at_exactly_15_seconds(self, monkeypatch):
        """At exactly 15s the condition (elapsed < timeout) is False."""
        fake_time = _FakeTimeModule(1000.0)
        monkeypatch.setattr(heartbeat_module, "time", fake_time)

        heartbeat_module.update_heartbeat()
        fake_time.advance(15.0)

        assert heartbeat_module.is_browser_connected() is False


# ---------------------------------------------------------------------------
# Test double: fake time module
# ---------------------------------------------------------------------------

class _FakeTimeModule:
    """Replaces the `time` module within heartbeat to control time.time()."""

    def __init__(self, start: float) -> None:
        self._now = start

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds
