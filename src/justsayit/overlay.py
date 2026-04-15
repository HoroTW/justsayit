"""Transparent wlr-layer-shell recording overlay.

A small bottom-anchored bar with:
  * a colored status dot on the left (idle / validating / recording / manual),
  * a state label above the mic-level meter,
  * a center-outward mic-level meter driven by the RMS callback from audio.py.

All updates from non-UI threads must go through ``GLib.idle_add`` —
helpers ``push_state`` / ``push_level`` handle that for you.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")

from gi.repository import Gdk, GLib, Gtk, Gtk4LayerShell  # noqa: E402

from justsayit.audio import State
from justsayit.config import Config

log = logging.getLogger(__name__)


CSS = b"""
window.justsayit-overlay {
    background: transparent;
}
.justsayit-overlay-box {
    background-color: rgb(20, 20, 30);
    border-radius: 14px;
    padding: 8px 14px;
}
.justsayit-overlay-label {
    color: rgba(255, 255, 255, 0.85);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 11px;
    font-weight: 500;
}
"""


@dataclass(frozen=True)
class _DotColor:
    r: float
    g: float
    b: float


_STATE_STYLE = {
    State.IDLE: ("idle", _DotColor(0.50, 0.50, 0.55)),
    State.VALIDATING: ("listening…", _DotColor(0.95, 0.82, 0.26)),
    State.RECORDING: ("recording", _DotColor(0.95, 0.30, 0.30)),
    State.MANUAL: ("recording (manual)", _DotColor(0.40, 0.72, 1.00)),
}


def _install_css_once() -> None:
    display = Gdk.Display.get_default()
    if display is None:
        return
    if getattr(_install_css_once, "_done", False):
        return
    provider = Gtk.CssProvider()
    try:
        provider.load_from_string(CSS.decode("utf-8"))
    except AttributeError:
        provider.load_from_data(CSS)
    Gtk.StyleContext.add_provider_for_display(
        display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    _install_css_once._done = True  # type: ignore[attr-defined]


class OverlayWindow(Gtk.ApplicationWindow):
    """Bottom-anchored layer-shell window."""

    def __init__(self, application: Gtk.Application, cfg: Config) -> None:
        super().__init__(application=application)
        self._cfg = cfg
        self._state = State.IDLE
        self._level = 0.0
        self._level_smoothed = 0.0
        self._pulse = 0.0

        self.add_css_class("justsayit-overlay")
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(cfg.overlay.width, cfg.overlay.height)

        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
        edge = (
            Gtk4LayerShell.Edge.TOP
            if cfg.overlay.anchor == "top"
            else Gtk4LayerShell.Edge.BOTTOM
        )
        Gtk4LayerShell.set_anchor(self, edge, True)
        Gtk4LayerShell.set_margin(self, edge, cfg.overlay.margin)
        # Don't reserve any workarea space; we want to float over content.
        Gtk4LayerShell.set_exclusive_zone(self, 0)

        # Outer horizontal box: [dot] [label / meter stack]
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.add_css_class("justsayit-overlay-box")
        box.set_hexpand(True)
        box.set_vexpand(True)
        box.set_opacity(max(0.0, min(1.0, cfg.overlay.opacity)))

        # Status dot — left side, vertically centered
        self._dot = Gtk.DrawingArea()
        self._dot.set_content_width(16)
        self._dot.set_content_height(16)
        self._dot.set_valign(Gtk.Align.CENTER)
        self._dot.set_draw_func(self._draw_dot, None)
        box.append(self._dot)

        # Right side: label above, meter below
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        right_box.set_hexpand(True)
        right_box.set_vexpand(True)

        self._label = Gtk.Label(label=_STATE_STYLE[State.IDLE][0])
        self._label.add_css_class("justsayit-overlay-label")
        self._label.set_xalign(0.0)
        self._label.set_hexpand(True)
        right_box.append(self._label)

        self._meter = Gtk.DrawingArea()
        self._meter.set_content_height(10)
        self._meter.set_hexpand(True)
        self._meter.set_valign(Gtk.Align.CENTER)
        self._meter.set_draw_func(self._draw_meter, None)
        right_box.append(self._meter)

        box.append(right_box)
        self.set_child(box)

        _install_css_once()

        # ~60 Hz UI tick for level smoothing + recording pulse.
        GLib.timeout_add(33, self._tick)

    # --- threadsafe entry points ------------------------------------------

    def push_state(self, state: State) -> None:
        GLib.idle_add(self._apply_state, state, priority=GLib.PRIORITY_DEFAULT)

    def push_level(self, rms: float) -> None:
        # Don't bounce a GLib.idle per chunk; just store and let the tick use it.
        self._level = rms

    # --- UI thread ---------------------------------------------------------

    def _apply_state(self, state: State) -> bool:
        self._state = state
        label, _ = _STATE_STYLE[state]
        self._label.set_label(label)
        self._dot.queue_draw()
        # Only visible while recording / listening.
        if state is State.IDLE:
            self.set_visible(False)
        else:
            if not self.get_visible():
                self.set_visible(True)
            self.present()
        return False  # one-shot

    def _tick(self) -> bool:
        # Scale raw RMS (~0..0.3 for normal speech) to [0, 1] with soft cap,
        # then apply user sensitivity multiplier.
        sensitivity = self._cfg.overlay.visualizer_sensitivity
        target = min(1.0, self._level * 8.0 * sensitivity)
        self._level_smoothed += (target - self._level_smoothed) * 0.25
        if self._state in (State.RECORDING, State.MANUAL, State.VALIDATING):
            self._pulse = (self._pulse + 0.08) % (2 * math.pi)
        else:
            self._pulse *= 0.9
        self._meter.queue_draw()
        self._dot.queue_draw()
        return True

    def _draw_dot(self, _area, cr, w, h, _user_data):
        _, color = _STATE_STYLE[self._state]
        radius = min(w, h) / 2.0 - 1.5
        # Subtle halo when active.
        if self._state in (State.RECORDING, State.MANUAL, State.VALIDATING):
            halo = 0.35 + 0.25 * (math.sin(self._pulse) + 1) / 2
            cr.set_source_rgba(color.r, color.g, color.b, halo * 0.5)
            cr.arc(w / 2, h / 2, radius + 3, 0, 2 * math.pi)
            cr.fill()
        cr.set_source_rgba(color.r, color.g, color.b, 1.0)
        cr.arc(w / 2, h / 2, radius, 0, 2 * math.pi)
        cr.fill()

    def _draw_meter(self, _area, cr, w, h, _user_data):
        # background track (full width)
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.10)
        self._round_rect(cr, 0, 0, w, h, h / 2)
        cr.fill()
        # filled level — grows symmetrically from the center outward
        fill = max(0.0, min(1.0, self._level_smoothed))
        if fill <= 0.0:
            return
        # Color shifts from green to amber as level rises.
        g = 0.85 - 0.45 * fill
        r = 0.35 + 0.50 * fill
        cr.set_source_rgba(r, g, 0.35, 0.92)
        bar_w = w * fill
        bar_x = (w - bar_w) / 2
        self._round_rect(cr, bar_x, 0, bar_w, h, h / 2)
        cr.fill()

    @staticmethod
    def _round_rect(cr, x, y, w, h, r):
        if w < r * 2:
            r = w / 2
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()
