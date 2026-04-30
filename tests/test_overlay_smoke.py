"""Real-instantiation smoke test for OverlayWindow.

The structural tests in ``test_overlay_scroll.py`` and the arity tests in
``test_overlay_callback_signatures.py`` cover specific shapes by string
matching the source — but they pass even when an API call uses a name
that doesn't exist on the actual GTK widget. This file constructs the
real ``OverlayWindow`` under a ``Gtk.Application`` to catch that whole
class of "the API I assumed doesn't exist" bugs at unit-test time.

Concrete trigger: 0.24.0 used ``ScrolledWindow.set_hscrollbar_policy`` /
``set_vscrollbar_policy`` (a GTK3-style fiction) — the AttributeError
only fired at app startup. A user reported it. This test would have
caught it before the merge.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gtk  # noqa: E402

from justsayit.config import Config
from justsayit.overlay import OverlayWindow


def _construct_overlay() -> OverlayWindow | Exception:
    """Run a Gtk.Application activate cycle that constructs an
    OverlayWindow and immediately quits. Returns the constructed window
    on success, or the exception that fired during construction."""
    cfg = Config()
    captured: list[OverlayWindow | Exception] = []

    def on_activate(app: Gtk.Application) -> None:
        try:
            overlay = OverlayWindow(app, cfg)
            captured.append(overlay)
        except Exception as exc:
            captured.append(exc)
        finally:
            app.quit()

    app = Gtk.Application(application_id="dev.justsayit.smoketest")
    app.connect("activate", on_activate)
    app.run([])
    if not captured:
        raise AssertionError("on_activate never fired")
    return captured[0]


def test_overlay_constructs_without_error():
    """OverlayWindow.__init__ must run to completion without raising —
    no AttributeError from GTK3-style API names, no NotImplementedError
    from missing widgets, no signature mismatch."""
    result = _construct_overlay()
    if isinstance(result, Exception):
        raise AssertionError(
            f"OverlayWindow construction raised: {type(result).__name__}: {result}"
        )
    assert isinstance(result, OverlayWindow)


def test_scrolled_content_area_is_actually_a_scrolled_window():
    """The ``_content_scroll`` attribute must be a real Gtk.ScrolledWindow
    with the policies the design calls for (not just any widget that
    happens to have a ``set_max_content_height`` attribute)."""
    overlay = _construct_overlay()
    assert isinstance(overlay, OverlayWindow), overlay
    assert isinstance(overlay._content_scroll, Gtk.ScrolledWindow)
    h_policy, v_policy = overlay._content_scroll.get_policy()
    assert h_policy == Gtk.PolicyType.NEVER, (
        f"horizontal scrollbar policy must be NEVER, got {h_policy}"
    )
    assert v_policy == Gtk.PolicyType.AUTOMATIC, (
        f"vertical scrollbar policy must be AUTOMATIC, got {v_policy}"
    )
    # Max content height must come from the config (cap beyond which
    # scroll engages instead of growing the window further).
    assert overlay._content_scroll.get_max_content_height() == 1000


def test_overlay_methods_exist():
    """All push_* methods that the pipeline / cli call into must exist on
    the constructed instance with the right names. Catches typos and
    forgotten renames."""
    overlay = _construct_overlay()
    assert isinstance(overlay, OverlayWindow), overlay
    for name in (
        "push_state",
        "push_level",
        "push_detected_text",
        "push_llm_text",
        "push_tool_call",
        "push_linger_start",
        "push_hide",
        "push_error",
        "push_update_available",
        "push_clipboard_context_armed",
        "push_continue_armed",
        "push_assistant_mode",
        "push_llm_profile",
        "push_redo_buttons",
    ):
        assert callable(getattr(overlay, name, None)), (
            f"OverlayWindow.{name} missing or not callable"
        )
