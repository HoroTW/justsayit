"""Extra markdown rendering tests: horizontal rules and conflict checks."""

from __future__ import annotations

import re

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Pango", "1.0")
from gi.repository import Pango  # noqa: E402

from justsayit.overlay import _md_to_pango


def test_horizontal_rule_renders_as_dash_line():
    """A line of three or more `-` (or `*` or `_`) on its own should
    render as a horizontal divider, not raw text."""
    src = "Above\n\n---\n\nBelow"
    out = _md_to_pango(src)
    assert "─" in out
    # Pango must accept the result.
    Pango.parse_markup(out, -1, "\0")


def test_horizontal_rule_star():
    """Three or more `*` on their own line should also render as a divider."""
    src = "Before\n\n***\n\nAfter"
    out = _md_to_pango(src)
    assert "─" in out
    Pango.parse_markup(out, -1, "\0")


def test_horizontal_rule_underscore():
    """Three or more `_` on their own line should also render as a divider."""
    src = "Before\n\n___\n\nAfter"
    out = _md_to_pango(src)
    assert "─" in out
    Pango.parse_markup(out, -1, "\0")


def test_horizontal_rule_five_dashes():
    """Five or more dashes should still be caught."""
    src = "Title\n\n-----\n\nBody"
    out = _md_to_pango(src)
    assert "─" in out
    Pango.parse_markup(out, -1, "\0")


def test_horizontal_rule_width_uses_hr_chars_arg():
    """The HR character count comes from the converter's hr_chars
    keyword so the overlay can size the divider to its actual content
    width (Pango has no stretch-to-width markup)."""
    out = _md_to_pango("---", hr_chars=10)
    assert "─" * 10 in out
    assert "─" * 11 not in out  # not longer than asked

    out = _md_to_pango("---", hr_chars=120)
    assert "─" * 120 in out


def test_overlay_passes_hr_chars_from_max_width():
    """The OverlayWindow caller of _md_to_pango must derive hr_chars from
    cfg.overlay.max_width so HR rendering stays roughly the width of the
    pill across config changes (default 1100 px → ~85 chars)."""
    import inspect
    from justsayit.overlay import OverlayWindow
    src = inspect.getsource(OverlayWindow._apply_llm_text)
    assert "hr_chars=" in src
    assert "self._cfg.overlay.max_width" in src


def test_horizontal_rule_does_not_break_tables():
    """A markdown table's `|---|---|` separator must NOT be misdetected
    as a horizontal rule (which would corrupt the table)."""
    src = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = _md_to_pango(src)
    # Expect a rendered table (with ┼ chars), NOT a series of HR divider lines.
    assert "┼" in out


def test_bullet_dashes_not_treated_as_hr():
    """Lines starting with `- item` (bullet list) must not be misdetected
    as horizontal rules."""
    src = "- one\n- two\n- three"
    out = _md_to_pango(src)
    # Should produce bullets (• prefix), not HR lines.
    assert "•" in out
    assert "─" not in out  # no HR divider should appear


def test_hr_raw_text_not_in_output():
    """The literal `---` string must not appear in the output as plain text."""
    src = "Above\n\n---\n\nBelow"
    out = _md_to_pango(src)
    assert "---" not in out


def test_hr_grows_to_match_widest_content_line():
    """When the response has a wide table that forces the overlay past its
    configured max_width, the HR should grow to match the table's width
    instead of staying at the small configured length."""
    # Build a wide-rendered table (12 columns, each ~5 chars wide).
    table_src = (
        "| " + " | ".join(f"col{i:02d}" for i in range(12)) + " |\n"
        + "|" + "|".join(["-----"] * 12) + "|\n"
        + "| " + " | ".join(f"v{i:02d}aa" for i in range(12)) + " |\n"
    )
    src = f"{table_src}\n---\n"
    # Use a small hr_chars so we know any growth came from the table.
    out = _md_to_pango(src, hr_chars=20)
    body = re.sub(r"<[^>]+>", "", out)
    # Find the HR row (a run of ─ chars on its own line, with NO ┼).
    hr_lines = [
        ln for ln in body.split("\n")
        if ln and set(ln) == {"─"}
    ]
    assert hr_lines, f"no HR-only line found in body: {body!r}"
    hr_len = len(hr_lines[0])
    # HR should be MUCH longer than 20 (the configured min) since the
    # table is wide.
    assert hr_len > 50, f"HR too short for wide-table content: {hr_len}"


def test_hr_capped_at_300_chars():
    """A single very long line must not produce an absurdly long HR
    (Pango handles paragraph wrap, so the cap protects against runaway
    HRs from raw text length)."""
    long_line = "x" * 500
    src = f"{long_line}\n\n---"
    out = _md_to_pango(src, hr_chars=20)
    body = re.sub(r"<[^>]+>", "", out)
    hr_lines = [ln for ln in body.split("\n") if ln and set(ln) == {"─"}]
    assert hr_lines
    assert len(hr_lines[0]) <= 300


def test_hr_at_least_hr_chars_when_content_is_short():
    """When the response has no wide content, the HR uses the configured
    width (not zero / nothing / falling back to content width)."""
    out = _md_to_pango("hi\n\n---\n\nbye", hr_chars=40)
    body = re.sub(r"<[^>]+>", "", out)
    hr_lines = [ln for ln in body.split("\n") if ln and set(ln) == {"─"}]
    assert hr_lines
    assert len(hr_lines[0]) == 40


def test_hr_marker_does_not_leak_into_output():
    """The internal \\x02 marker for HRs must always be substituted out
    of the final output, including when there are zero non-HR lines."""
    out = _md_to_pango("---")
    assert "\x02" not in out
    out = _md_to_pango("---\n---\n---")
    assert "\x02" not in out
