"""Markdown table rendering in the overlay's _md_to_pango.

Tables are detected as contiguous `|`-prefixed lines containing a
separator row (`|---|---|`) and rendered as a monospace `<tt>` block
with aligned columns. The user accepted the monospace fallback over
attempting full table rendering inside Pango markup."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, call

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Pango", "1.0")
from gi.repository import GLib, Gtk, Pango  # noqa: E402

from justsayit.overlay import _md_to_pango, _render_md_table, _set_label_markup_safe


SIMPLE_TABLE = (
    "| col1 | col2 |\n"
    "|------|------|\n"
    "| a    | b    |\n"
    "| cc   | dd   |\n"
)


def test_simple_table_renders_as_tt_block():
    out = _md_to_pango(SIMPLE_TABLE)
    assert "<tt>" in out and "</tt>" in out
    assert "│" in out  # column separator
    assert "─" in out  # header divider


def _strip_table_wrappers(out: str) -> str:
    """Strip the table's Pango wrapping (``<span allow_breaks="false">``
    + ``<tt>``) so the inner aligned text can be inspected as plain
    monospace lines."""
    return re.sub(r"<[^>]+>", "", out)


def test_table_columns_are_aligned():
    out = _md_to_pango(SIMPLE_TABLE)
    # Every data line must have the same display width.
    body_lines = [
        ln for ln in _strip_table_wrappers(out).split("\n")
        if "│" in ln
    ]
    assert len(body_lines) >= 3, body_lines  # header + 2 body rows
    widths = {len(ln) for ln in body_lines}
    assert len(widths) == 1, f"data rows misaligned: {widths}"


def test_separator_aligns_with_data_rows():
    """The ``─┼─`` separator must be the same length as the data rows so
    each ``┼`` sits exactly under a ``│`` in the data rows. Earlier code
    prepended/appended an extra ``─`` that shifted columns right by one."""
    out = _md_to_pango(SIMPLE_TABLE)
    body = _strip_table_wrappers(out)
    lines = [ln for ln in body.split("\n") if ln.strip()]
    # Find the separator row (contains ┼) and a data row (contains │ but not ┼).
    sep_lines = [ln for ln in lines if "┼" in ln]
    data_lines = [ln for ln in lines if "│" in ln and "┼" not in ln]
    assert sep_lines and data_lines
    assert len(sep_lines[0]) == len(data_lines[0]), (
        f"sep len {len(sep_lines[0])} != data len {len(data_lines[0])}\n"
        f"sep:  {sep_lines[0]!r}\ndata: {data_lines[0]!r}"
    )
    # Every ┼ in the separator must sit at the same column index as a │
    # in the data rows.
    sep_pluses = [i for i, c in enumerate(sep_lines[0]) if c == "┼"]
    data_pipes = [i for i, c in enumerate(data_lines[0]) if c == "│"]
    assert sep_pluses == data_pipes, (
        f"┼ at {sep_pluses} but │ at {data_pipes}"
    )


def test_no_separator_falls_through_unchanged():
    """A `|`-line block without a separator should NOT be table-rendered."""
    src = "| just  | text |\n| more  | text |\n"
    out = _md_to_pango(src)
    assert "<tt>" not in out


def test_table_cell_content_is_escaped():
    src = (
        "| name | sym |\n"
        "|------|-----|\n"
        "| <foo>| & 5 < 6 |\n"
    )
    out = _md_to_pango(src)
    assert "&lt;foo&gt;" in out
    assert "&amp;" in out
    assert "<foo>" not in out  # no raw <


def test_table_inline_markdown_in_cells_kept_literal():
    src = (
        "| key  | val   |\n"
        "|------|-------|\n"
        "| **a**| _b_   |\n"
    )
    out = _md_to_pango(src)
    # Cell-internal **a** stays as escaped literal, not <b>a</b>.
    assert "**a**" in out
    assert "<b>a</b>" not in out


def test_table_output_parses_as_pango_markup():
    """Real GtkLabel.set_markup acceptance is the bar Pango cares about."""
    label = Gtk.Label()
    label.set_markup(_md_to_pango(SIMPLE_TABLE))  # raises on parse error


def test_render_md_table_one_column():
    out = _render_md_table(["| only |", "|------|", "| row1 |", "| row2 |"])
    assert "<tt>" in out
    assert "only" in out


def test_table_with_empty_cells():
    src = (
        "| a  | b  |\n"
        "|----|----|\n"
        "|    | bb |\n"
        "| aa |    |\n"
    )
    out = _md_to_pango(src)
    label = Gtk.Label()
    label.set_markup(out)  # parse-validates


# ── Bug regression: fence containing a markdown table ───────────────────────

FENCE_WITH_TABLE_INPUT = (
    "Sure — here's a simple Markdown table example:\n"
    "\n"
    "```markdown\n"
    "| Name  | Age |\n"
    "|-------|-----|\n"
    "| Alice | 28  |\n"
    "```\n"
    "\n"
    "Rendered, it looks like this:\n"
    "\n"
    "| Name  | Age |\n"
    "|-------|-----|\n"
    "| Alice | 28  |\n"
)


def test_fence_containing_table_parses_as_valid_pango():
    """A fenced block that contains a markdown table must produce valid
    Pango markup — previously the table pass ran before the fence pass,
    turning ``|``-lines inside fences into stash keys that nested inside
    the fence stash, leaving literal NUL bytes and unbalanced tags."""
    out = _md_to_pango(FENCE_WITH_TABLE_INPUT)
    # Pango.parse_markup raises GLib.Error on bad markup (unlike set_markup).
    Pango.parse_markup(out, -1, "\0")


def test_md_to_pango_no_nul_bytes():
    """_md_to_pango output must never contain NUL bytes for any reasonable
    input — stash keys use NUL as delimiters and must be fully resolved."""
    inputs = [
        FENCE_WITH_TABLE_INPUT,
        SIMPLE_TABLE,
        "plain text with **bold** and `code`",
        "```python\nprint('hello')\n```",
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "",
    ]
    for src in inputs:
        out = _md_to_pango(src)
        assert "\x00" not in out, f"NUL byte in output for input: {src!r}"


def test_set_label_markup_safe_falls_back_on_invalid_markup():
    """When markup is invalid, _set_label_markup_safe must call set_text
    with the fallback rather than leaving set_markup to silently fail."""
    label = MagicMock(spec=Gtk.Label)
    bad_markup = "<b>unclosed bold"
    fallback = "unclosed bold"
    _set_label_markup_safe(label, bad_markup, fallback)
    label.set_markup.assert_not_called()
    label.set_text.assert_called_once_with(fallback)


def test_set_label_markup_safe_calls_set_markup_on_valid():
    """When markup is valid, _set_label_markup_safe must call set_markup."""
    label = MagicMock(spec=Gtk.Label)
    good_markup = "Hello <b>world</b>"
    _set_label_markup_safe(label, good_markup, "fallback")
    label.set_markup.assert_called_once_with(good_markup)
    label.set_text.assert_not_called()


# ── Wide-cell wrapping ──────────────────────────────────────────────────────

LONG_CELL = "This item is significantly longer than the others and takes up a lot of horizontal space."


def test_wide_cells_wrap_to_multiple_lines():
    """A cell longer than _MD_TABLE_MAX_COL_WIDTH must wrap onto multiple
    display rows so the overlay doesn't grow horizontally past max_width."""
    src = (
        "| col1  | col2 |\n"
        "|-------|------|\n"
        f"| short | {LONG_CELL} |\n"
        "| a     | b    |\n"
    )
    out = _md_to_pango(src)
    body = _strip_table_wrappers(out)
    # No single rendered line should be longer than ~50 chars (cap is 40
    # per col, plus separators); originally this would be ~100+.
    max_len = max(len(ln) for ln in body.split("\n"))
    assert max_len < 60, f"row too wide: {max_len} chars"
    # Every word from the long cell must still appear somewhere in the
    # rendered body — wrapping shouldn't drop content.
    for word in LONG_CELL.split():
        assert word in body, f"word {word!r} missing after wrap"


def test_wide_cells_preserve_alignment():
    """When one cell wraps and others don't, the data rows that result
    must still have aligned ``│`` columns."""
    src = (
        "| col1  | col2 |\n"
        "|-------|------|\n"
        f"| short | {LONG_CELL} |\n"
    )
    out = _md_to_pango(src)
    body = _strip_table_wrappers(out)
    data_lines = [ln for ln in body.split("\n") if "│" in ln and "┼" not in ln]
    assert len(data_lines) >= 3, "expected wrap to produce extra data rows"
    pipe_positions = {tuple(i for i, c in enumerate(ln) if c == "│") for ln in data_lines}
    assert len(pipe_positions) == 1, f"misaligned after wrap: {pipe_positions}"


def test_table_uses_allow_breaks_false():
    """Pango must not be allowed to wrap inside the table on whitespace —
    that would break the careful column alignment. We mark the whole
    table with ``allow_breaks="false"``."""
    out = _md_to_pango(SIMPLE_TABLE)
    assert 'allow_breaks="false"' in out
    # Pango must still parse it.
    Pango.parse_markup(out, -1, "\0")
