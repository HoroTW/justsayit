"""Markdown table rendering in the overlay's _md_to_pango.

Tables are detected as contiguous `|`-prefixed lines containing a
separator row (`|---|---|`) and rendered as a monospace `<tt>` block
with aligned columns. The user accepted the monospace fallback over
attempting full table rendering inside Pango markup."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from justsayit.overlay import _md_to_pango, _render_md_table


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


def test_table_columns_are_aligned():
    out = _md_to_pango(SIMPLE_TABLE)
    # Every data line must have the same display width.
    body_lines = [
        ln for ln in out.replace("<tt>", "").replace("</tt>", "").split("\n")
        if "│" in ln
    ]
    assert len(body_lines) >= 3, body_lines  # header + 2 body rows
    widths = {len(ln) for ln in body_lines}
    assert len(widths) == 1, f"data rows misaligned: {widths}"


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
