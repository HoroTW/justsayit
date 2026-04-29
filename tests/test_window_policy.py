"""Tests for the window-class clipboard policy.

Mocks the focused-window lookup so we can assert the policy decisions
without needing a real Wayland/X11 session.
"""

from __future__ import annotations

import pytest

from justsayit.config import Config
from justsayit.cli import App


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubOverlay:
    def __init__(self) -> None:
        self.armed_calls: list[bool] = []

    def push_clipboard_context_armed(self, armed: bool) -> None:
        self.armed_calls.append(armed)


def _app_with_policy(
    *,
    enabled: bool = True,
    auto_arm=None,
    block=None,
) -> App:
    cfg = Config()
    cfg.window_clipboard_policy.enabled = enabled
    cfg.window_clipboard_policy.auto_arm = auto_arm or []
    cfg.window_clipboard_policy.block = block or []
    app = App(cfg, no_overlay=True, no_paste=True)
    app.overlay = _StubOverlay()
    return app


# ---------------------------------------------------------------------------
# Policy disabled / no rules
# ---------------------------------------------------------------------------


def test_policy_disabled_is_noop(monkeypatch):
    app = _app_with_policy(enabled=False, auto_arm=["thunderbird"])
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: "thunderbird")

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is False
    assert app.overlay.armed_calls == []


def test_policy_with_no_rules_is_noop(monkeypatch):
    app = _app_with_policy(enabled=True)
    called = []

    def _spy():
        called.append("called")
        return "thunderbird"

    monkeypatch.setattr("justsayit.cli.active_window_id", _spy)

    app._apply_window_clipboard_policy()

    # Should short-circuit before even calling active_window_id.
    assert called == []
    assert app._clipboard_context_armed is False


# ---------------------------------------------------------------------------
# auto_arm
# ---------------------------------------------------------------------------


def test_auto_arm_arms_when_class_matches(monkeypatch):
    app = _app_with_policy(auto_arm=["thunderbird"])
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: "thunderbird")

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is True
    assert app.overlay.armed_calls == [True]


def test_auto_arm_substring_match(monkeypatch):
    """The policy uses case-insensitive substring matching so users can
    list 'firefox' and have it match 'firefox-developer-edition'."""
    app = _app_with_policy(auto_arm=["thunderbird"])
    monkeypatch.setattr(
        "justsayit.cli.active_window_id", lambda: "org.mozilla.thunderbird"
    )

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is True


def test_auto_arm_no_match_leaves_state(monkeypatch):
    app = _app_with_policy(auto_arm=["thunderbird"])
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: "firefox")

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is False
    assert app.overlay.armed_calls == []


def test_auto_arm_idempotent_when_already_armed(monkeypatch):
    app = _app_with_policy(auto_arm=["konsole"])
    app._clipboard_context_armed = True
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: "konsole")

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is True
    # We don't re-push when already armed.
    assert app.overlay.armed_calls == []


# ---------------------------------------------------------------------------
# block
# ---------------------------------------------------------------------------


def test_block_disarms_when_class_matches(monkeypatch):
    app = _app_with_policy(block=["keepassxc"])
    app._clipboard_context_armed = True
    monkeypatch.setattr(
        "justsayit.cli.active_window_id", lambda: "org.keepassxc.KeePassXC"
    )

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is False
    assert app.overlay.armed_calls == [False]


def test_block_clears_arm_next(monkeypatch):
    app = _app_with_policy(block=["keepassxc"])
    app._clipboard_context_arm_next = True
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: "keepassxc")

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_arm_next is False


def test_block_wins_over_auto_arm(monkeypatch):
    """If a class is in both lists, block prevails."""
    app = _app_with_policy(
        auto_arm=["sensitive"],
        block=["sensitive"],
    )
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: "sensitive")

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is False


# ---------------------------------------------------------------------------
# Lookup failure
# ---------------------------------------------------------------------------


def test_unknown_window_does_not_change_state(monkeypatch):
    app = _app_with_policy(auto_arm=["thunderbird"])
    monkeypatch.setattr("justsayit.cli.active_window_id", lambda: None)

    app._apply_window_clipboard_policy()

    assert app._clipboard_context_armed is False
    assert app.overlay.armed_calls == []
