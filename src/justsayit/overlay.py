"""Transparent wlr-layer-shell recording overlay.

Layout has two modes:

  Compact (recording / listening / processing):
    ┌──────────────────────────────────┐
    │  recording                       │  ← state label
    │  ● ▓▓▓▓──────────────────▓▓▓▓    │  ← dot + meter
    └──────────────────────────────────┘

  Expanded result view (after transcription):
    ┌──────────────────────────────────────────────────────┐
    │  This is the regex-filtered detected text.           │  ← top field
    │  ──────────────────────────────────────────────────  │
    │  This is the LLM-cleaned result.                     │  ← bottom field
    │  ──────────────────────────────────────────────────  │
    │  ● ▓▓──────────────────────────────────────────▓▓    │  ← dot + flat meter
    └──────────────────────────────────────────────────────┘

All updates from non-UI threads must go through ``GLib.idle_add`` —
the ``push_*`` helpers handle that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

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
.justsayit-detected-label {
    color: rgba(255, 255, 255, 0.92);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 11px;
    font-weight: 400;
}
.justsayit-llm-label {
    color: rgba(180, 255, 200, 0.90);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 11px;
    font-weight: 400;
}
.justsayit-abort-button {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    color: rgba(255, 255, 255, 0.55);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
}
.justsayit-abort-button:hover {
    color: rgba(255, 120, 120, 0.95);
}
.justsayit-clip-button {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    color: rgba(255, 255, 255, 0.55);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
}
.justsayit-clip-button:hover {
    color: rgba(180, 220, 255, 0.95);
}
.justsayit-clip-button.armed {
    color: rgba(120, 200, 255, 1.0);
}
.justsayit-clip-button.armed:hover {
    color: rgba(255, 120, 120, 0.95);
}
.justsayit-cont-button {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    color: rgba(255, 255, 255, 0.55);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
}
.justsayit-cont-button:hover {
    color: rgba(180, 255, 180, 0.95);
}
.justsayit-cont-button.armed {
    color: rgba(120, 255, 140, 1.0);
}
.justsayit-cont-button.armed:hover {
    color: rgba(255, 120, 120, 0.95);
}
.justsayit-assistant-button {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    color: rgba(255, 255, 255, 0.55);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
}
.justsayit-assistant-button:hover {
    color: rgba(200, 180, 255, 0.95);
}
.justsayit-assistant-button.armed {
    color: rgba(180, 140, 255, 1.0);
}
.justsayit-assistant-button.armed:hover {
    color: rgba(255, 120, 120, 0.95);
}
.justsayit-copy-result-button {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    color: rgba(180, 255, 200, 0.75);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
}
.justsayit-copy-result-button:hover {
    color: rgba(180, 255, 200, 1.0);
}
.justsayit-update-badge {
    color: rgba(255, 215, 90, 0.95);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 10px;
    font-weight: 600;
    padding: 0 4px;
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

_DOT_RESULT = _DotColor(0.35, 0.85, 0.45)   # green during result / linger

_SAFETY_MS = 30_000
_LLM_WAITING = "Wait for LLM processing…"
_CHAR_WIDTH_PX = 6.5   # approximate for Inter 11px


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

    def __init__(
        self,
        application: Gtk.Application,
        cfg: Config,
        *,
        on_abort: Callable[[], None] | None = None,
        on_toggle_clipboard_context: Callable[[], None] | None = None,
        on_toggle_continue_window: Callable[[], None] | None = None,
        on_toggle_assistant_mode: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(application=application)
        self._cfg = cfg
        self._state = State.IDLE
        self._on_abort = on_abort
        self._on_toggle_clipboard_context = on_toggle_clipboard_context
        self._on_toggle_continue_window = on_toggle_continue_window
        self._on_toggle_assistant_mode = on_toggle_assistant_mode

        self._level = 0.0
        self._level_smoothed = 0.0
        self._pulse = 0.0
        self._assistant_mode = False
        self._last_llm_text = ""

        self._dot_color_override: _DotColor | None = None
        self._linger_source: int | None = None
        self._safety_source: int | None = None
        # Set by ``_force_hide`` so a delayed engine-IDLE callback (× abort
        # while RECORDING/MANUAL, or transcribe-thread ``push_hide`` from
        # short-segment skip / empty transcription) cannot re-open the
        # overlay with "processing…" + 30s safety hide. Cleared on the
        # next non-IDLE state or after the suppressed IDLE is consumed.
        self._suppress_next_idle_processing = False

        self.add_css_class("justsayit-overlay")
        self.set_decorated(False)
        self.set_size_request(cfg.overlay.width, cfg.overlay.height)
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
        Gtk4LayerShell.set_exclusive_zone(self, 0)

        # ── Root: vertical stack ─────────────────────────────────────────────
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        root.add_css_class("justsayit-overlay-box")
        root.set_hexpand(True)
        root.set_vexpand(True)
        root.set_opacity(max(0.0, min(1.0, cfg.overlay.opacity)))

        # ── Top row: state label (left) + abort × button (right) ─────────────
        # CenterBox anchors start/end widgets to the row's left/right edges
        # regardless of which is visible — so when the state label is hidden
        # in the expanded result view the × stays pinned top-right (a plain
        # HBox would collapse the hidden label's allocation and the button,
        # which only sets halign=END, would render at the row's start).
        top_row = Gtk.CenterBox()
        top_row.set_hexpand(True)

        self._state_label = Gtk.Label(label=_STATE_STYLE[State.IDLE][0])
        self._state_label.add_css_class("justsayit-overlay-label")
        self._state_label.set_xalign(0.0)
        self._state_label.set_halign(Gtk.Align.START)
        top_row.set_start_widget(self._state_label)

        # Right-anchored cluster: [update badge?] [× button]. The badge
        # is hidden until the GitHub version check finds something newer
        # (see push_update_available); the button is always present.
        end_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        end_box.set_valign(Gtk.Align.START)
        end_box.set_halign(Gtk.Align.END)

        self._update_badge = Gtk.Label(label="update available")
        self._update_badge.add_css_class("justsayit-update-badge")
        self._update_badge.set_visible(False)
        end_box.append(self._update_badge)

        # Continue-session button. Arms a timed window during which each
        # recording continues the previous LLM conversation thread.
        self._cont_button = Gtk.Button(label="↩")
        self._cont_button.add_css_class("justsayit-cont-button")
        self._cont_button.set_tooltip_text(
            "Continue previous LLM session (starts 5 min window)"
        )
        self._cont_button.connect("clicked", self._on_cont_clicked)
        self._cont_button.set_visible(False)
        end_box.append(self._cont_button)

        # Clipboard-context arming button. One click → next LLM call
        # receives the clipboard contents under a "Clipboard as additional
        # context" section in the system prompt; click again to disarm
        # before recording. The armed state has its own CSS class so it's
        # visually obvious whether the next transcription will be enriched.
        self._clip_button = Gtk.Button(label="📋")
        self._clip_button.add_css_class("justsayit-clip-button")
        self._clip_button.set_tooltip_text(
            "Use clipboard contents, as LLM context (just once for this recording)"
        )
        self._clip_button.connect("clicked", self._on_clip_clicked)
        # Only useful while a manual recording is in progress — hidden in
        # idle / VAD-recording / result phases. Visibility is toggled by
        # ``_apply_state`` when entering / leaving ``State.MANUAL``.
        self._clip_button.set_visible(False)
        end_box.append(self._clip_button)

        # Copy-result button: copies the last LLM response to clipboard.
        # Visible only in assistant mode after a result is shown.
        self._copy_result_button = Gtk.Button(label="📄")
        self._copy_result_button.add_css_class("justsayit-copy-result-button")
        self._copy_result_button.set_tooltip_text("Copy response to clipboard")
        self._copy_result_button.connect("clicked", self._on_copy_result_clicked)
        self._copy_result_button.set_visible(False)
        end_box.append(self._copy_result_button)

        # Assistant-mode toggle: keeps the overlay open after each result
        # so it can be used as an interactive chat. Arms continue-session
        # automatically for every recording while active.
        self._assistant_button = Gtk.Button(label="💬")
        self._assistant_button.add_css_class("justsayit-assistant-button")
        self._assistant_button.set_tooltip_text(
            "Toggle assistant mode — overlay stays open for interactive chat"
        )
        self._assistant_button.connect("clicked", self._on_assistant_clicked)
        end_box.append(self._assistant_button)

        self._abort_button = Gtk.Button(label="×")
        self._abort_button.add_css_class("justsayit-abort-button")
        self._abort_button.set_tooltip_text("Abort recording (discard, no paste)")
        self._abort_button.connect("clicked", self._on_abort_clicked)
        end_box.append(self._abort_button)

        top_row.set_end_widget(end_box)

        root.append(top_row)

        # ── Text area (result mode only, hidden by default) ───────────────────
        # Separator + top field
        self._sep1 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self._sep1.set_margin_bottom(4)
        self._sep1.set_visible(False)
        root.append(self._sep1)

        self._detected_label = Gtk.Label()
        self._detected_label.add_css_class("justsayit-detected-label")
        self._detected_label.set_xalign(0.0)
        self._detected_label.set_hexpand(True)
        self._detected_label.set_wrap(True)
        self._detected_label.set_visible(False)
        root.append(self._detected_label)

        # Separator + bottom (LLM) field
        self._sep2 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self._sep2.set_margin_top(4)
        self._sep2.set_margin_bottom(4)
        self._sep2.set_visible(False)
        root.append(self._sep2)

        self._llm_label = Gtk.Label()
        self._llm_label.add_css_class("justsayit-llm-label")
        self._llm_label.set_xalign(0.0)
        self._llm_label.set_hexpand(True)
        self._llm_label.set_wrap(True)
        self._llm_label.set_visible(False)
        root.append(self._llm_label)

        # Separator above bottom row (only shown in result mode)
        self._sep_bottom = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self._sep_bottom.set_margin_top(4)
        self._sep_bottom.set_margin_bottom(4)
        self._sep_bottom.set_visible(False)
        root.append(self._sep_bottom)

        # ── Bottom row: dot + meter (always visible) ──────────────────────────
        bottom_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        bottom_row.set_hexpand(True)

        self._dot = Gtk.DrawingArea()
        self._dot.set_content_width(16)
        self._dot.set_content_height(16)
        self._dot.set_valign(Gtk.Align.CENTER)
        self._dot.set_draw_func(self._draw_dot, None)
        bottom_row.append(self._dot)

        self._meter = Gtk.DrawingArea()
        self._meter.set_content_height(10)
        self._meter.set_hexpand(True)
        self._meter.set_valign(Gtk.Align.CENTER)
        self._meter.set_draw_func(self._draw_meter, None)
        bottom_row.append(self._meter)

        root.append(bottom_row)

        # Click anywhere on the pill during result-display → pause auto-close.
        _click_ctrl = Gtk.GestureClick.new()
        _click_ctrl.connect("pressed", self._on_result_clicked)
        root.add_controller(_click_ctrl)

        self.set_child(root)
        _install_css_once()

        GLib.timeout_add(33, self._tick)

    # ── Thread-safe entry points ─────────────────────────────────────────────

    def push_state(self, state: State) -> None:
        GLib.idle_add(self._apply_state, state, priority=GLib.PRIORITY_DEFAULT)

    def push_level(self, rms: float) -> None:
        self._level = rms

    def push_detected_text(self, text: str, llm_pending: bool = False) -> None:
        """Show regex-filtered text in the top field.

        *llm_pending* = True → bottom field shows "Wait for LLM processing…"
        until ``push_llm_text`` is called.
        """
        GLib.idle_add(
            self._apply_detected_text, text, llm_pending,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_llm_text(self, text: str, thought: str = "") -> None:
        """Update the bottom field with the LLM-cleaned result.

        *thought* — optional reasoning preamble (the part stripped by
        ``paste_strip_regex`` before paste). When non-empty it is shown
        italicised on its own line above *text* so the user sees the
        full model reply at a glance, even though only *text* will be
        pasted into the focused window.
        """
        GLib.idle_add(
            self._apply_llm_text, text, thought,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_tool_call(self, name: str, params: dict) -> None:
        """Show a tool-call annotation in the LLM text area. Called from the
        transcription thread during the tool-call loop before the final answer
        is ready. Thread-safe via GLib.idle_add."""
        GLib.idle_add(
            self._apply_tool_call, name, params,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_linger_start(self) -> None:
        GLib.idle_add(self._start_linger, priority=GLib.PRIORITY_DEFAULT)

    def push_hide(self) -> None:
        GLib.idle_add(self._force_hide, priority=GLib.PRIORITY_DEFAULT)

    def push_update_available(self, latest_version: str) -> None:
        """Show the small yellow "update available" badge to the left of
        the × button. Safe to call from any thread; idempotent."""
        GLib.idle_add(
            self._apply_update_available, latest_version,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_clipboard_context_armed(self, armed: bool) -> None:
        """Visually indicate whether the next LLM call will pick up the
        clipboard. Toggles the ``armed`` CSS class on the 📋 button."""
        GLib.idle_add(
            self._apply_clipboard_armed, armed,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_continue_armed(self, armed: bool) -> None:
        """Visually indicate whether the continue window is active.
        Toggles the ``armed`` CSS class on the ↩ button."""
        GLib.idle_add(
            self._apply_cont_armed, armed,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_assistant_mode(self, active: bool) -> None:
        """Enable or disable assistant mode. When active the overlay will not
        auto-dismiss after showing a result; the copy-result button appears."""
        GLib.idle_add(
            self._apply_assistant_mode, active,
            priority=GLib.PRIORITY_DEFAULT,
        )

    # ── User actions ─────────────────────────────────────────────────────────

    def _on_cont_clicked(self, _button: Gtk.Button) -> None:
        if self._on_toggle_continue_window is not None:
            try:
                self._on_toggle_continue_window()
            except Exception:
                log.exception("on_toggle_continue_window callback raised")

    def _on_assistant_clicked(self, _button: Gtk.Button) -> None:
        if self._on_toggle_assistant_mode is not None:
            try:
                self._on_toggle_assistant_mode()
            except Exception:
                log.exception("on_toggle_assistant_mode callback raised")

    def _on_copy_result_clicked(self, _button: Gtk.Button) -> None:
        if not self._last_llm_text:
            return
        import subprocess
        try:
            subprocess.run(
                ["wl-copy"],
                input=self._last_llm_text.encode("utf-8"),
                timeout=3.0,
                check=True,
            )
            log.info("copied LLM result to clipboard (%d chars)", len(self._last_llm_text))
        except Exception:
            log.exception("failed to copy result to clipboard")

    def _on_clip_clicked(self, _button: Gtk.Button) -> None:
        if self._on_toggle_clipboard_context is not None:
            try:
                self._on_toggle_clipboard_context()
            except Exception:
                log.exception("on_toggle_clipboard_context callback raised")

    def _on_result_clicked(self, _gesture, _n_press, _x, _y) -> None:
        """Clicking the pill during result display activates assistant mode."""
        if self._detected_label.get_visible() and not self._assistant_mode:
            log.debug("result clicked — activating assistant mode")
            if self._on_toggle_assistant_mode is not None:
                try:
                    self._on_toggle_assistant_mode()
                except Exception:
                    log.exception("on_toggle_assistant_mode callback raised")

    def _on_abort_clicked(self, _button: Gtk.Button) -> None:
        """× button: abort an active recording (discard, no paste). If
        the overlay is in the post-result linger phase, just dismiss."""
        if self._state in (State.VALIDATING, State.RECORDING, State.MANUAL):
            log.info("abort button clicked during %s — discarding", self._state.value)
            if self._on_abort is not None:
                try:
                    self._on_abort()
                except Exception:
                    log.exception("on_abort callback raised")
            # Engine will transition to IDLE; hide immediately so the user
            # gets instant feedback without waiting for the state callback.
            self._force_hide()
        else:
            log.info("abort button clicked outside recording — dismissing overlay")
            self._force_hide()

    # ── UI-thread handlers ───────────────────────────────────────────────────

    def _apply_state(self, state: State) -> bool:
        prev = self._state
        self._state = state

        # Both context buttons are only meaningful while a manual recording
        # is in progress; hidden in idle / VAD / result phases.
        self._cont_button.set_visible(state is State.MANUAL)
        self._clip_button.set_visible(state is State.MANUAL)

        if state is State.IDLE:
            if self._suppress_next_idle_processing:
                # Overlay was already explicitly dismissed (× abort, or
                # transcribe thread called push_hide for skip / empty);
                # don't re-open with "processing…" — there's no result.
                self._suppress_next_idle_processing = False
                self._force_hide()
                return False
            if prev in (State.RECORDING, State.MANUAL):
                self._cancel_linger()
                self._dot_color_override = None
                self._state_label.set_label("processing…")
                self._state_label.set_visible(True)
                self._hide_text_areas()
                self._collapse_window()
                self._schedule_safety_hide()
                self._dot.queue_draw()
                if not self.get_visible():
                    self.set_visible(True)
            else:
                self._force_hide()
        else:
            # New recording — any leftover suppression from a prior cycle
            # is stale; honour the fresh state transition normally.
            self._suppress_next_idle_processing = False
            self._cancel_linger()
            self._cancel_safety()
            self._dot_color_override = None
            label, _ = _STATE_STYLE[state]
            self._state_label.set_label(label)
            self._state_label.set_visible(True)
            self._hide_text_areas()
            self._collapse_window()
            self._dot.queue_draw()
            if not self.get_visible():
                self.set_visible(True)
            self.present()
        return False

    def _apply_detected_text(self, text: str, llm_pending: bool) -> bool:
        self._cancel_safety()
        self._dot_color_override = _DOT_RESULT

        # Hide state label, show text fields above the bottom row.
        self._state_label.set_visible(False)
        self._detected_label.set_label(text)
        self._sep1.set_visible(True)
        self._detected_label.set_visible(True)

        if llm_pending:
            self._llm_label.set_label(_LLM_WAITING)
            self._sep2.set_visible(True)
            self._llm_label.set_visible(True)
        else:
            self._sep2.set_visible(False)
            self._llm_label.set_visible(False)

        self._sep_bottom.set_visible(True)

        # Pre-size: height_for(text) × 2 + static, capped at max_height.
        self._expand_window(text, two_fields=llm_pending)

        self._dot.queue_draw()
        if not self.get_visible():
            self.set_visible(True)
        self.present()
        return False

    def _apply_tool_call(self, name: str, params: dict) -> bool:
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        annotation = f"⚙ {name}({param_str})"
        # Show in the LLM field as an intermediate status update.
        # If the field is already showing a previous annotation, append.
        existing = self._llm_label.get_text() if not self._llm_label.get_use_markup() else ""
        if existing and existing not in (_LLM_WAITING,):
            text = existing + "\n" + annotation
        else:
            text = annotation
        self._llm_label.set_label(text)
        if not self._llm_label.get_visible():
            self._sep2.set_visible(True)
            self._llm_label.set_visible(True)
        return False

    def _apply_llm_text(self, text: str, thought: str = "") -> bool:
        self._last_llm_text = text
        if thought:
            from html import escape
            # Blue-green / teal italic for the thought, then a newline and
            # the normal-weight body that will actually be pasted.
            markup = (
                f'<span foreground="#5ed1c4"><i>{escape(thought)}</i></span>'
                f"\n\n{escape(text)}"
            )
            self._llm_label.set_markup(markup)
        else:
            self._llm_label.set_label(text)
        if not self._llm_label.get_visible():
            self._sep2.set_visible(True)
            self._llm_label.set_visible(True)
        if text:
            self._copy_result_button.set_visible(True)
        return False

    def _apply_update_available(self, latest_version: str) -> bool:
        self._update_badge.set_tooltip_text(
            f"Update available: v{latest_version} — see GitHub releases"
        )
        self._update_badge.set_visible(True)
        return False

    def _apply_clipboard_armed(self, armed: bool) -> bool:
        if armed:
            self._clip_button.add_css_class("armed")
            self._clip_button.set_tooltip_text(
                "Use clipboard contents, as LLM context (just once for this recording) "
                "recording — click to disarm"
            )
        else:
            self._clip_button.remove_css_class("armed")
            self._clip_button.set_tooltip_text(
                "Use clipboard contents, as LLM context (just once for this recording) "
                "recording"
            )
        return False

    def _apply_cont_armed(self, armed: bool) -> bool:
        if armed:
            self._cont_button.add_css_class("armed")
            self._cont_button.set_tooltip_text("Continue window active — click to disarm")
        else:
            self._cont_button.remove_css_class("armed")
            self._cont_button.set_tooltip_text("Continue previous LLM session (starts 5 min window)")
        return False

    def _apply_assistant_mode(self, active: bool) -> bool:
        self._assistant_mode = active
        if active:
            self._assistant_button.add_css_class("armed")
            self._assistant_button.set_tooltip_text(
                "Assistant mode active — click to deactivate"
            )
        else:
            self._assistant_button.remove_css_class("armed")
            self._assistant_button.set_tooltip_text(
                "Toggle assistant mode — overlay stays open for interactive chat"
            )
        return False

    def _start_linger(self) -> bool:
        if self._assistant_mode:
            return False  # stay open until manually dismissed
        self._cancel_linger()
        ms = self._cfg.overlay.result_linger_ms
        if ms > 0:
            self._linger_source = GLib.timeout_add(ms, self._finish_linger)
        else:
            self._force_hide()
        return False

    def _finish_linger(self) -> bool:
        self._linger_source = None
        self._force_hide()
        return False

    def _force_hide(self) -> bool:
        self._cancel_linger()
        self._cancel_safety()
        self._dot_color_override = None
        self._hide_text_areas()
        self._collapse_window()
        self.set_visible(False)
        self._last_llm_text = ""
        # An audio-thread IDLE callback may still be queued behind us
        # (× abort during RECORDING/MANUAL, or skip-short / empty
        # transcription firing push_hide before the engine state event
        # reaches GLib). Tell _apply_state to honour the dismissal
        # instead of re-opening the overlay with "processing…".
        self._suppress_next_idle_processing = True
        return False

    # ── Layout helpers ───────────────────────────────────────────────────────

    def _hide_text_areas(self) -> None:
        self._sep1.set_visible(False)
        self._detected_label.set_visible(False)
        self._sep2.set_visible(False)
        self._llm_label.set_visible(False)
        self._sep_bottom.set_visible(False)
        self._copy_result_button.set_visible(False)
        self._state_label.set_visible(True)

    def _collapse_window(self) -> None:
        self.set_default_size(self._cfg.overlay.width, self._cfg.overlay.height)

    def _expand_window(self, text: str, two_fields: bool) -> None:
        """Pre-size the window using: height_for(text) × 2 + static_height."""
        max_w = self._cfg.overlay.max_width
        max_h = self._cfg.overlay.max_height

        # Usable text width: subtract padding (14 × 2) + dot (16) + spacing (10).
        usable_w = max_w - 14 * 2 - 16 - 10
        chars_per_line = max(1, int(usable_w / _CHAR_WIDTH_PX))
        n_lines = max(1, math.ceil(len(text) / chars_per_line))
        line_h = 16  # px
        text_h = n_lines * line_h

        # static = compact pill (state-label + bottom-row) + separators.
        static_h = self._cfg.overlay.height + 3 * 12  # 3 separator rows
        multiplier = 2 if two_fields else 1
        estimated_h = min(max_h, static_h + text_h * multiplier + 8)
        self.set_default_size(max_w, estimated_h)

    # ── Timers ───────────────────────────────────────────────────────────────

    def _schedule_safety_hide(self) -> None:
        self._cancel_safety()
        self._safety_source = GLib.timeout_add(_SAFETY_MS, self._safety_hide)

    def _safety_hide(self) -> bool:
        self._safety_source = None
        log.warning("overlay safety-hide fired (no text after %ds)", _SAFETY_MS // 1000)
        self._force_hide()
        return False

    def _cancel_linger(self) -> None:
        if self._linger_source is not None:
            GLib.source_remove(self._linger_source)
            self._linger_source = None

    def _cancel_safety(self) -> None:
        if self._safety_source is not None:
            GLib.source_remove(self._safety_source)
            self._safety_source = None

    # ── Tick / draw ──────────────────────────────────────────────────────────

    def _tick(self) -> bool:
        sensitivity = self._cfg.overlay.visualizer_sensitivity
        # Only animate the meter while actively recording / listening.
        # In result / linger phase (state=IDLE, dot_color_override set) or
        # while idle, decay the smoothed level to zero so the bar goes flat.
        if self._state in (State.RECORDING, State.MANUAL, State.VALIDATING):
            target = min(1.0, self._level * 8.0 * sensitivity)
            self._level_smoothed += (target - self._level_smoothed) * 0.25
            self._pulse = (self._pulse + 0.08) % (2 * math.pi)
        else:
            # Decay to flat: meter goes quiet; pulse slows but keeps halo in
            # result phase via dot_color_override check in _draw_dot.
            self._level_smoothed *= 0.85
            if self._dot_color_override is not None:
                self._pulse = (self._pulse + 0.04) % (2 * math.pi)
            else:
                self._pulse *= 0.9
        self._meter.queue_draw()
        self._dot.queue_draw()
        return True

    def _draw_dot(self, _area, cr, w, h, _user_data):
        color = (
            self._dot_color_override
            if self._dot_color_override is not None
            else _STATE_STYLE[self._state][1]
        )
        radius = min(w, h) / 2.0 - 1.5
        show_halo = (
            self._state in (State.RECORDING, State.MANUAL, State.VALIDATING)
            or self._dot_color_override is not None
        )
        if show_halo:
            halo = 0.35 + 0.25 * (math.sin(self._pulse) + 1) / 2
            cr.set_source_rgba(color.r, color.g, color.b, halo * 0.5)
            cr.arc(w / 2, h / 2, radius + 3, 0, 2 * math.pi)
            cr.fill()
        cr.set_source_rgba(color.r, color.g, color.b, 1.0)
        cr.arc(w / 2, h / 2, radius, 0, 2 * math.pi)
        cr.fill()

    def _draw_meter(self, _area, cr, w, h, _user_data):
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.10)
        self._round_rect(cr, 0, 0, w, h, h / 2)
        cr.fill()
        fill = max(0.0, min(1.0, self._level_smoothed))
        if fill <= 0.0:
            return
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
