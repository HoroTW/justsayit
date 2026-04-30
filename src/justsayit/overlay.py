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
import re
import textwrap
from dataclasses import dataclass
from html import escape
from typing import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
gi.require_version("Pango", "1.0")

from gi.repository import Gdk, GLib, Gtk, Gtk4LayerShell, Pango  # noqa: E402

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
.justsayit-profile-label {
    color: rgba(180, 180, 210, 0.45);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 9px;
    font-weight: 400;
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
.justsayit-overlay-btn {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.55);
}
.justsayit-abort-button:hover {
    color: rgba(255, 120, 120, 0.95);
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
.justsayit-cont-button:hover {
    color: rgba(180, 255, 180, 0.95);
}
.justsayit-cont-button.armed {
    color: rgba(120, 255, 140, 1.0);
}
.justsayit-cont-button.armed:hover {
    color: rgba(255, 120, 120, 0.95);
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
    color: rgba(180, 255, 200, 0.75);
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
.justsayit-error-label {
    color: rgba(255, 200, 90, 1.0);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 11px;
    font-weight: 600;
}
.justsayit-retry-button {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 4px;
    margin: 0;
    min-height: 16px;
    min-width: 16px;
    color: rgba(255, 200, 90, 0.85);
    font-family: "Inter", "Cantarell", "Noto Sans", sans-serif;
    font-size: 12px;
    font-weight: 600;
}
.justsayit-retry-button:hover {
    color: rgba(255, 200, 90, 1.0);
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
    State.MANUAL: ("recording…", _DotColor(0.40, 0.72, 1.00)),
}

_DOT_RESULT = _DotColor(0.35, 0.85, 0.45)   # green during result / linger
_DOT_ERROR = _DotColor(1.00, 0.78, 0.35)    # amber during error pill

_SAFETY_MS = 30_000
_LLM_WAITING = "Wait for LLM processing…"
_CHAR_WIDTH_PX = 6.5   # approximate for Inter 11px


# ── Markdown → Pango ────────────────────────────────────────────────────────
# GtkLabel's set_markup() only understands Pango markup, not Markdown. The
# LLM frequently emits **bold**, `code`, lists and headings — render those
# inline rather than showing the raw asterisks. Block elements Pango doesn't
# have (headings, lists, quotes) are reproduced via <span> / bullet glyphs.

_MD_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_MD_QUOTE_RE = re.compile(r"^&gt;\s*(.*)$")
_MD_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_MD_BOLD_UNDER_RE = re.compile(r"(?<![A-Za-z0-9_])__([^_\n]+?)__(?![A-Za-z0-9_])")
_MD_ITALIC_RE = re.compile(r"(?<![*A-Za-z0-9])\*([^*\n]+?)\*(?![*A-Za-z0-9])")
_MD_ITALIC_UNDER_RE = re.compile(r"(?<![A-Za-z0-9_])_([^_\n]+?)_(?![A-Za-z0-9_])")
_MD_STRIKE_RE = re.compile(r"~~([^~\n]+?)~~")
_MD_TABLE_SEP_RE = re.compile(r"^\|[-| :]+\|?\s*$")
_MD_TABLE_MAX_COL_WIDTH = 40   # cells longer than this wrap to multiple lines

_HEADING_SIZES = {1: "x-large", 2: "large", 3: "large"}


def _render_md_table(lines: list[str]) -> str:
    """Render a markdown table block as a monospace ``<tt>`` block with
    aligned columns. Cell contents are HTML-escaped; inline markdown
    inside cells is intentionally NOT processed (kept as literal text).

    Long cells (over ``_MD_TABLE_MAX_COL_WIDTH`` chars) are wrapped onto
    multiple display rows so wide content doesn't blow out the overlay
    horizontally. The whole table is wrapped in
    ``<span allow_breaks="false">`` so Pango doesn't re-break the
    carefully-aligned rows on whitespace inside the table."""

    def _parse_row(line: str) -> list[str]:
        inner = line.strip()
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|"):
            inner = inner[:-1]
        return [cell.strip() for cell in inner.split("|")]

    rows: list[list[str] | None] = []
    for line in lines:
        if _MD_TABLE_SEP_RE.match(line.strip()):
            rows.append(None)  # placeholder — replaced with ─ divider
        else:
            rows.append(_parse_row(line))

    col_count = max((len(r) for r in rows if r is not None), default=1)
    for i, row in enumerate(rows):
        if row is not None:
            rows[i] = (row + [""] * col_count)[:col_count]

    # Per-column width = max natural width across cells, capped at
    # MAX_COL_WIDTH. Cells longer than the cap wrap to multiple lines.
    widths = [1] * col_count
    for row in rows:
        if row is None:
            continue
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], min(len(cell), _MD_TABLE_MAX_COL_WIDTH))

    def _wrap_cell(cell: str, width: int) -> list[str]:
        if len(cell) <= width:
            return [cell] if cell else [""]
        return textwrap.wrap(
            cell, width=width, break_long_words=True, break_on_hyphens=False,
        ) or [""]

    out: list[str] = []
    for row in rows:
        if row is None:
            # Separator row: each column's dashes match that column's
            # width; the inter-column "─┼─" lines up with " │ " in the
            # data rows so ┼ sits exactly where │ does. No leading/
            # trailing dash — data rows don't have leading/trailing
            # chars either, so adding them here would shift all the ┼
            # columns one position right.
            segs = ["─" * widths[j] for j in range(col_count)]
            out.append("─┼─".join(segs))
        else:
            wrapped = [_wrap_cell(cell, widths[j]) for j, cell in enumerate(row)]
            n_disp = max(len(w) for w in wrapped)
            for k in range(n_disp):
                segs = [
                    escape(wrapped[j][k] if k < len(wrapped[j]) else "").ljust(widths[j])
                    for j in range(col_count)
                ]
                out.append(" │ ".join(segs))
    return f'<span allow_breaks="false"><tt>{chr(10).join(out)}</tt></span>'


def _md_to_pango(text: str) -> str:
    """Translate a small subset of Markdown to Pango markup.

    Handles fenced + inline code, bold, italic, strikethrough, links,
    ATX headings, bullet lists, and blockquotes. Anything else falls
    through as escaped plain text. Code spans/blocks are stashed first
    so their contents are never re-interpreted as markdown."""
    if not text:
        return ""

    stash: dict[str, str] = {}

    def _stash(value: str) -> str:
        key = f"\x00MD{len(stash)}\x00"
        stash[key] = value
        return key

    # Fences first — stash their contents before the table pass so that
    # "|"-prefixed lines INSIDE a fence are not mistakenly detected as a
    # markdown table (which would embed a stash key inside a fence stash,
    # producing unresolvable nested keys and invalid Pango markup).
    def _fence(m: re.Match[str]) -> str:
        return _stash(f"<tt>{escape(m.group(1).rstrip())}</tt>")

    text = _MD_FENCE_RE.sub(_fence, text)

    # Tables — collect contiguous "|"-prefixed lines that include a
    # separator row (|---|---|), render to a stashed <tt> block so later
    # passes (escape, bold, etc.) don't touch them. Fenced content is
    # already replaced with stash keys so it is invisible to this pass.
    src_lines = text.splitlines()
    out_table_pass: list[str] = []
    i = 0
    while i < len(src_lines):
        line = src_lines[i]
        if line.strip().startswith("|"):
            block: list[str] = []
            j = i
            while j < len(src_lines) and src_lines[j].strip().startswith("|"):
                block.append(src_lines[j])
                j += 1
            has_sep = any(_MD_TABLE_SEP_RE.match(b.strip()) for b in block)
            if len(block) >= 2 and has_sep:
                out_table_pass.append(_stash(_render_md_table(block)))
                i = j
                continue
        out_table_pass.append(line)
        i += 1
    text = "\n".join(out_table_pass)

    def _inline_code(m: re.Match[str]) -> str:
        return _stash(f"<tt>{escape(m.group(1))}</tt>")

    text = _MD_INLINE_CODE_RE.sub(_inline_code, text)

    def _link(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        return _stash(
            f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'
        )

    text = _MD_LINK_RE.sub(_link, text)

    text = escape(text)

    out_lines: list[str] = []
    for line in text.split("\n"):
        m = _MD_HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            size = _HEADING_SIZES.get(level, "medium")
            line = f'<span size="{size}" weight="bold">{m.group(2)}</span>'
            out_lines.append(line)
            continue
        m = _MD_BULLET_RE.match(line)
        if m:
            out_lines.append(f"{m.group(1)}• {m.group(2)}")
            continue
        m = _MD_QUOTE_RE.match(line)
        if m:
            out_lines.append(f"<i>│ {m.group(1)}</i>")
            continue
        out_lines.append(line)
    text = "\n".join(out_lines)

    text = _MD_BOLD_RE.sub(r"<b>\1</b>", text)
    text = _MD_BOLD_UNDER_RE.sub(r"<b>\1</b>", text)
    text = _MD_STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _MD_ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _MD_ITALIC_UNDER_RE.sub(r"<i>\1</i>", text)

    for key, value in stash.items():
        text = text.replace(key, value)
    return text


_GTKLABEL_ONLY_TAGS_RE = re.compile(r"</?a\b[^>]*>", re.IGNORECASE)


def _set_label_markup_safe(label: Gtk.Label, markup: str, fallback: str) -> None:
    """``set_markup`` but if Pango rejects the string, fall back to plain
    text. The markdown converter is best-effort; an LLM emitting an
    unbalanced asterisk inside HTML-looking text shouldn't blank the pill.

    ``Gtk.Label.set_markup()`` does NOT raise on invalid markup — it logs
    a Gtk-WARNING and leaves the label unchanged (causing a frozen pill).
    ``Pango.parse_markup`` DOES raise, so we validate first.

    BUT — ``Pango.parse_markup`` is stricter than ``GtkLabel.set_markup``:
    GtkLabel accepts ``<a href="...">`` for clickable links as a GTK
    extension, while pure Pango doesn't recognise ``<a>`` at all and
    rejects it. Strip ``<a>`` open/close tags for the validation pass
    only; the original markup (with links intact) is what goes to
    ``set_markup``. Otherwise any LLM response containing a link gets
    every other markdown style (bold, code, tables) silently stripped."""
    validation_markup = _GTKLABEL_ONLY_TAGS_RE.sub("", markup)
    try:
        Pango.parse_markup(validation_markup, -1, "\0")
        label.set_markup(markup)
    except Exception:
        log.exception("Pango markup invalid; falling back to plain text")
        label.set_text(fallback)


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
        self._last_tick_us = 0   # frame-clock timestamp from previous _tick
        self._assistant_mode = False
        self._last_llm_text = ""
        # Pending retry for the current error pill (or None).
        self._retry_cb: Callable[[], None] | None = None
        self._error_active: bool = False

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
        # Default: don't intercept keyboard. Switched to ON_DEMAND while the
        # result pill is visible so Ctrl+C on selected text works (the
        # compositor routes key events to us only when the user clicks the
        # pill, then back to the previous window when they click elsewhere).
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
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

        start_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        start_box.set_valign(Gtk.Align.CENTER)
        start_box.set_margin_end(11)

        self._profile_label = Gtk.Label(label="")
        self._profile_label.add_css_class("justsayit-profile-label")
        self._profile_label.set_xalign(0.0)
        self._profile_label.set_halign(Gtk.Align.START)
        self._profile_label.set_visible(False)
        start_box.append(self._profile_label)

        self._state_label = Gtk.Label(label=_STATE_STYLE[State.IDLE][0])
        self._state_label.add_css_class("justsayit-overlay-label")
        self._state_label.set_xalign(0.0)
        self._state_label.set_halign(Gtk.Align.START)
        start_box.append(self._state_label)

        top_row.set_start_widget(start_box)

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
        self._cont_button.add_css_class("justsayit-overlay-btn")
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
        self._clip_button.add_css_class("justsayit-overlay-btn")
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
        self._copy_result_button.add_css_class("justsayit-overlay-btn")
        self._copy_result_button.add_css_class("justsayit-copy-result-button")
        self._copy_result_button.set_tooltip_text("Copy response to clipboard")
        self._copy_result_button.connect("clicked", self._on_copy_result_clicked)
        self._copy_result_button.set_visible(False)
        end_box.append(self._copy_result_button)

        # Assistant-mode toggle: keeps the overlay open after each result
        # so it can be used as an interactive chat. Arms continue-session
        # automatically for every recording while active.
        self._assistant_button = Gtk.Button(label="💬")
        self._assistant_button.add_css_class("justsayit-overlay-btn")
        self._assistant_button.add_css_class("justsayit-assistant-button")
        self._assistant_button.set_tooltip_text(
            "Toggle assistant mode — overlay stays open for interactive chat"
        )
        self._assistant_button.connect("clicked", self._on_assistant_clicked)
        end_box.append(self._assistant_button)

        # Retry button — only visible during the error pill, alongside ×.
        self._retry_button = Gtk.Button(label="🔁")
        self._retry_button.add_css_class("justsayit-retry-button")
        self._retry_button.set_tooltip_text("Retry the failed step")
        self._retry_button.connect("clicked", self._on_retry_clicked)
        self._retry_button.set_visible(False)
        end_box.append(self._retry_button)

        self._abort_button = Gtk.Button(label="×")
        self._abort_button.add_css_class("justsayit-overlay-btn")
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
        self._detected_label.set_selectable(True)
        # Default non-focusable: ON_DEMAND keyboard mode would otherwise
        # let the compositor grant focus to the layer-shell as soon as
        # we set ON_DEMAND, stealing keyboard focus from the user's
        # focused window — which breaks the paste target. Focus is
        # enabled on demand in `_on_result_clicked`, only after the
        # user explicitly clicks the result pill.
        self._detected_label.set_can_focus(False)
        self._detected_label.connect(
            "notify::has-selection", self._on_label_has_selection
        )
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
        self._llm_label.set_selectable(True)
        self._llm_label.set_can_focus(False)  # see _detected_label note
        self._llm_label.connect(
            "notify::has-selection", self._on_label_has_selection
        )
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

        # Click anywhere on the pill (including on selectable text labels)
        # during result-display → pause auto-close. CAPTURE-phase fires
        # from root DOWN to target before any child gesture, so it sees
        # clicks even on selectable labels which would otherwise eat them.
        _click_ctrl = Gtk.GestureClick.new()
        _click_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        _click_ctrl.connect("pressed", self._on_result_clicked)
        root.add_controller(_click_ctrl)

        self.set_child(root)
        _install_css_once()

        # Drive the meter + dot pulse off the frame clock of the dot widget.
        # Tick callbacks auto-pause when the widget is unmapped, so the
        # animation loop doesn't run when the overlay is hidden.
        self._dot.add_tick_callback(self._tick, None)

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

    def push_error(
        self,
        stage: str,
        msg: str,
        retry_cb: Callable[[], None] | None = None,
    ) -> None:
        """Show an amber error pill with *msg* and optional retry button.

        Stays visible for ``result_linger_ms * 3`` (errors deserve more
        reading time) before auto-dismissing. Clicking the pill cancels
        the auto-close (reuses ``_on_result_clicked``).
        """
        GLib.idle_add(
            self._apply_error, stage, msg, retry_cb,
            priority=GLib.PRIORITY_DEFAULT,
        )

    def push_llm_profile(self, backend: str | None, name: str | None) -> None:
        """Show or hide the tiny LLM profile label (e.g. 'local/gemma4-cleanup').
        Pass (None, None) to hide it when postprocessing is off."""
        GLib.idle_add(
            self._apply_profile_label, backend, name,
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

    def _on_retry_clicked(self, _button: Gtk.Button) -> None:
        cb = self._retry_cb
        # Hide the error pill immediately so the user sees the click
        # take effect even if the retry takes a moment to start.
        self._force_hide()
        if cb is not None:
            try:
                cb()
            except Exception:
                log.exception("retry callback raised")

    def _on_clip_clicked(self, _button: Gtk.Button) -> None:
        if self._on_toggle_clipboard_context is not None:
            try:
                self._on_toggle_clipboard_context()
            except Exception:
                log.exception("on_toggle_clipboard_context callback raised")

    def _on_result_clicked(self, _gesture, _n_press, _x, _y) -> None:
        """Clicking the result pill cancels the auto-dismiss AND enables
        keyboard interaction (so Ctrl+C copies the selected text). Wired
        in CAPTURE phase on the root box so it fires for clicks on
        selectable labels too — which would otherwise eat the click
        before any BUBBLE-phase gesture sees it.

        Focus-grant is deferred to this moment (rather than enabled when
        the result first appears) so the layer-shell never steals
        keyboard focus from the user's focused window before the user
        deliberately interacts with the pill — that would break paste.

        We do NOT call ``grab_focus()`` here. CAPTURE phase fires before
        the label's own press handler, so by the time the label sees the
        event, ``can_focus`` is already True — GTK's natural click-to-
        focus behavior moves focus to the actual clicked label as part
        of the same press, and the drag-to-select sequence completes
        normally. Calling ``grab_focus`` explicitly would (a) trigger
        GTK's select-all-on-focus-gain on selectable labels and (b)
        interrupt the in-flight press→drag→release sequence, turning
        every drag-select into a single click."""
        if not self._detected_label.get_visible():
            return
        log.info("result pill clicked — enabling Ctrl+C and cancelling auto-dismiss")
        self._cancel_linger()
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.ON_DEMAND)
        self._detected_label.set_can_focus(True)
        self._llm_label.set_can_focus(True)

    def _on_label_has_selection(self, label: Gtk.Label, _pspec) -> None:
        """Selecting text in a label is itself a strong signal of user
        engagement — also cancel the linger. Belt-and-braces with the
        root CAPTURE-phase gesture: even if a click somehow fails to
        reach _on_result_clicked, dragging out a selection still keeps
        the pill open."""
        if label.get_property("has-selection"):
            self._cancel_linger()

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
                Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
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
            Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
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
        # Stay at NONE keyboard mode + can_focus=False on labels until the
        # user explicitly clicks the pill — otherwise the compositor would
        # grant the layer-shell window focus the moment we present the
        # result, stealing it from the user's focused window and breaking
        # the paste target. The click handler enables ON_DEMAND + focus.
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
        self._detected_label.set_can_focus(False)
        self._llm_label.set_can_focus(False)

        # Hide state label, show text fields above the bottom row.
        self._state_label.set_visible(False)
        self._detected_label.set_text(text)
        self._sep1.set_visible(True)
        self._detected_label.set_visible(True)

        if llm_pending:
            self._llm_label.set_text(_LLM_WAITING)
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
        existing = self._llm_label.get_text()
        if existing and existing != _LLM_WAITING:
            text = existing + "\n" + annotation
        else:
            text = annotation
        self._llm_label.set_text(text)
        if not self._llm_label.get_visible():
            self._sep2.set_visible(True)
            self._llm_label.set_visible(True)
        return False

    def _apply_llm_text(self, text: str, thought: str = "") -> bool:
        self._last_llm_text = text
        body_markup = _md_to_pango(text)
        if thought:
            # Blue-green / teal italic for the thought, then a newline and
            # the markdown-rendered body that will actually be pasted.
            markup = (
                f'<span foreground="#5ed1c4"><i>{escape(thought)}</i></span>'
                f"\n\n{body_markup}"
            )
        else:
            markup = body_markup
        _set_label_markup_safe(self._llm_label, markup, text)
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

    def _set_armed(
        self, button: Gtk.Button, armed: bool, tip_off: str, tip_on: str
    ) -> None:
        if armed:
            button.add_css_class("armed")
            button.set_tooltip_text(tip_on)
        else:
            button.remove_css_class("armed")
            button.set_tooltip_text(tip_off)

    def _apply_clipboard_armed(self, armed: bool) -> bool:
        self._set_armed(
            self._clip_button,
            armed,
            tip_off="Use clipboard contents as LLM context (just once for the next recording)",
            tip_on="Use clipboard contents as LLM context (just once for the next recording) — click to disarm",
        )
        return False

    def _apply_cont_armed(self, armed: bool) -> bool:
        self._set_armed(
            self._cont_button,
            armed,
            tip_off="Continue previous LLM session (starts 5 min window)",
            tip_on="Continue window active — click to disarm",
        )
        return False

    def _apply_profile_label(self, backend: str | None, name: str | None) -> bool:
        if backend and name:
            self._profile_label.set_label(f"{backend}/{name}")
        else:
            self._profile_label.set_label("direct (no LLM)")
        self._profile_label.set_visible(True)
        return False

    def _apply_error(
        self,
        stage: str,
        msg: str,
        retry_cb: Callable[[], None] | None,
    ) -> bool:
        # Cancel any running timers — this is a fresh result/error display.
        self._cancel_safety()
        self._cancel_linger()
        self._error_active = True
        self._retry_cb = retry_cb
        self._dot_color_override = _DOT_ERROR
        # Same focus-stealing avoidance as _apply_detected_text: stay at
        # NONE until user explicitly clicks the pill.
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
        self._detected_label.set_can_focus(False)
        self._llm_label.set_can_focus(False)

        # State label: "error: <stage>" in amber.
        self._state_label.remove_css_class("justsayit-overlay-label")
        self._state_label.add_css_class("justsayit-error-label")
        self._state_label.set_label(f"error: {stage}")
        self._state_label.set_visible(True)

        # Body: plain-text message in the detected label (no markup).
        self._detected_label.set_use_markup(False)
        self._detected_label.set_text(msg or "(unknown error)")
        self._sep1.set_visible(True)
        self._detected_label.set_visible(True)

        # Hide the LLM field — error has no second body.
        self._sep2.set_visible(False)
        self._llm_label.set_visible(False)
        self._sep_bottom.set_visible(True)

        self._retry_button.set_visible(retry_cb is not None)

        self._expand_window(msg or "", two_fields=False)
        self._dot.queue_draw()
        if not self.get_visible():
            self.set_visible(True)
        self.present()

        # Linger 3× the normal result time before auto-dismissing.
        ms = self._cfg.overlay.result_linger_ms * 3
        if ms > 0:
            self._linger_source = GLib.timeout_add(ms, self._finish_linger)
        return False

    def _apply_assistant_mode(self, active: bool) -> bool:
        self._assistant_mode = active
        self._set_armed(
            self._assistant_button,
            active,
            tip_off="Toggle assistant mode — overlay stays open for interactive chat",
            tip_on="Assistant mode active — click to deactivate",
        )
        return False

    def _start_linger(self) -> bool:
        if self._assistant_mode:
            return False  # stay open until manually dismissed
        if self._error_active:
            # _apply_error already started a longer (3x) linger; don't
            # let a happy-path push_linger_start shorten it.
            return False
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
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
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
        self._retry_button.set_visible(False)
        self._state_label.set_visible(True)
        # Restore the state label's normal styling — error pill swaps in
        # the amber CSS class which would otherwise persist into the
        # next recording's "listening" / "recording" label.
        if self._error_active:
            self._state_label.remove_css_class("justsayit-error-label")
            self._state_label.add_css_class("justsayit-overlay-label")
            self._error_active = False
            self._retry_cb = None

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

    def _tick(self, _widget, frame_clock, _user_data=None) -> bool:
        # Time-based animation: derive a per-frame scale from the frame
        # clock so behaviour matches the previous fixed-33ms cadence
        # regardless of the actual refresh rate. Original constants were
        # tuned for ~33ms ticks.
        now_us = frame_clock.get_frame_time()
        if self._last_tick_us == 0:
            dt_frames = 1.0
        else:
            dt_frames = max(0.0, (now_us - self._last_tick_us) / 33_000.0)
        self._last_tick_us = now_us

        active = self._state in (State.RECORDING, State.MANUAL, State.VALIDATING)
        in_result = self._dot_color_override is not None

        # Fast-path: truly idle — no recording, no result halo, meter and
        # pulse already collapsed to zero. Skip queue_draw to stop the
        # DrawingAreas from re-rendering every frame at idle.
        if (
            not active
            and not in_result
            and self._level_smoothed < 0.001
            and abs(self._pulse) < 0.001
        ):
            return True

        sensitivity = self._cfg.overlay.visualizer_sensitivity
        if active:
            target = min(1.0, self._level * 8.0 * sensitivity)
            # First-order IIR; per-frame factor scaled by dt for stability.
            alpha = min(1.0, 0.25 * dt_frames)
            self._level_smoothed += (target - self._level_smoothed) * alpha
            self._pulse = (self._pulse + 0.08 * dt_frames) % (2 * math.pi)
        else:
            # Decay to flat: meter goes quiet; pulse slows but keeps halo in
            # result phase via dot_color_override check in _draw_dot.
            self._level_smoothed *= 0.85 ** dt_frames
            if in_result:
                self._pulse = (self._pulse + 0.04 * dt_frames) % (2 * math.pi)
            else:
                self._pulse *= 0.9 ** dt_frames
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
