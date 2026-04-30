"""Regression tests for callback-shape mismatches in OverlayWindow.

The overlay registers Python methods as GTK callbacks (tick callbacks,
DrawingArea draw_func, gesture handlers, etc.). GTK calls these with
specific argument shapes; if a method's signature doesn't accept those
args, the failure only surfaces at runtime under a real GTK loop —
which our normal pytest run never exercises.

These tests check the signatures via ``inspect`` so a wrong arity is
caught at unit-test time. Concrete trigger that motivated this file:
``add_tick_callback`` invokes its callback with
``(widget, frame_clock, user_data)`` (3 args + bound self), but a
previous version of ``_tick`` only accepted ``(self, widget, frame_clock)``,
which silently passed import + tests but crashed the moment the user
opened the overlay.
"""

from __future__ import annotations

import inspect

from justsayit.overlay import OverlayWindow


def _positional_count(method) -> int:
    """Count parameters that can be passed positionally (excluding self)."""
    sig = inspect.signature(method)
    return sum(
        1
        for p in list(sig.parameters.values())[1:]  # drop self
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    )


def test_tick_accepts_add_tick_callback_signature():
    """``self._dot.add_tick_callback(self._tick, None)`` invokes
    ``self._tick(widget, frame_clock, user_data)`` — 3 positional args
    plus the bound self. The method must accept all 3."""
    assert _positional_count(OverlayWindow._tick) >= 3, (
        "_tick must accept (widget, frame_clock, user_data) — see "
        "GTK add_tick_callback signature"
    )


def test_draw_funcs_accept_drawing_area_signature():
    """DrawingArea ``set_draw_func`` invokes ``cb(area, cr, w, h, user_data)``
    — 5 positional args plus the bound self."""
    for name in ("_draw_dot", "_draw_meter"):
        method = getattr(OverlayWindow, name)
        assert _positional_count(method) >= 5, (
            f"{name} must accept (area, cr, w, h, user_data) — see "
            f"GTK DrawingArea.set_draw_func signature"
        )


def test_gesture_pressed_handler_signature():
    """``GestureClick.connect('pressed', cb)`` invokes
    ``cb(gesture, n_press, x, y)`` — 4 positional args plus self."""
    method = OverlayWindow._on_result_clicked
    assert _positional_count(method) >= 4, (
        "_on_result_clicked must accept (gesture, n_press, x, y) — see "
        "GTK GestureClick.pressed signature"
    )


def test_label_has_selection_notify_handler_signature():
    """``label.connect('notify::has-selection', cb)`` invokes
    ``cb(label, pspec)`` — 2 positional args plus self."""
    method = OverlayWindow._on_label_has_selection
    assert _positional_count(method) >= 2, (
        "_on_label_has_selection must accept (label, pspec)"
    )
