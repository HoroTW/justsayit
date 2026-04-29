"""Tests for justsayit.snippets — loader + matcher behaviour.

These tests exercise the snippet matching logic in isolation. The
pipeline integration is covered by tests/test_pipeline_routing.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from justsayit.snippets import (
    Snippet,
    SnippetMatch,
    load_snippets,
    match_snippet,
)


# ---------------------------------------------------------------------------
# load_snippets
# ---------------------------------------------------------------------------


def test_load_snippets_missing_file_returns_empty(tmp_path: Path):
    assert load_snippets(tmp_path / "missing.toml") == []


def test_load_snippets_basic(tmp_path: Path):
    f = tmp_path / "snippets.toml"
    f.write_text(
        """
[[snippet]]
trigger = "insert email signature"
replacement = "Best,\\nHoro"
""",
        encoding="utf-8",
    )
    snips = load_snippets(f)
    assert len(snips) == 1
    s = snips[0]
    assert s.trigger == "insert email signature"
    assert s.replacement == "Best,\nHoro"
    assert s.mode == "replace"
    assert s.bypass_llm is True
    assert s.literal is True


def test_load_snippets_skips_invalid_entries(tmp_path: Path):
    f = tmp_path / "snippets.toml"
    f.write_text(
        """
[[snippet]]
trigger = ""
replacement = "no trigger"

[[snippet]]
trigger = "ok"
replacement = "ok!"
""",
        encoding="utf-8",
    )
    snips = load_snippets(f)
    assert [s.trigger for s in snips] == ["ok"]


def test_load_snippets_invalid_mode_skipped(tmp_path: Path):
    f = tmp_path / "snippets.toml"
    f.write_text(
        """
[[snippet]]
trigger = "bad"
replacement = "x"
mode = "wrong"

[[snippet]]
trigger = "good"
replacement = "y"
""",
        encoding="utf-8",
    )
    snips = load_snippets(f)
    assert [s.trigger for s in snips] == ["good"]


def test_load_snippets_malformed_toml(tmp_path: Path):
    f = tmp_path / "snippets.toml"
    f.write_text("not = valid = toml\n", encoding="utf-8")
    assert load_snippets(f) == []


# ---------------------------------------------------------------------------
# match_snippet — replace mode
# ---------------------------------------------------------------------------


def test_replace_mode_whole_match():
    snips = [Snippet(trigger="hello", replacement="HI THERE", mode="replace")]
    m = match_snippet("hello", snips)
    assert m == SnippetMatch("HI THERE", True)


def test_replace_mode_case_insensitive_and_punctuation_stripped():
    snips = [Snippet(trigger="insert email signature", replacement="SIG", mode="replace")]
    m = match_snippet("Insert, Email Signature.", snips)
    assert m is not None
    assert m.replacement == "SIG"


def test_replace_mode_partial_does_not_match():
    snips = [Snippet(trigger="hello", replacement="x", mode="replace")]
    assert match_snippet("hello world", snips) is None


# ---------------------------------------------------------------------------
# match_snippet — expand mode
# ---------------------------------------------------------------------------


def test_expand_mode_partial_match_appends_remainder():
    snips = [Snippet(trigger="todo", replacement="TODO:", mode="expand")]
    m = match_snippet("todo finish the test", snips)
    assert m is not None
    assert m.replacement == "TODO: finish the test"


def test_expand_mode_exact_match_no_remainder():
    snips = [Snippet(trigger="todo", replacement="TODO:", mode="expand")]
    m = match_snippet("todo", snips)
    assert m is not None
    assert m.replacement == "TODO:"


def test_expand_mode_no_match():
    snips = [Snippet(trigger="todo", replacement="TODO:", mode="expand")]
    assert match_snippet("write some code", snips) is None


def test_expand_mode_only_matches_at_start():
    snips = [Snippet(trigger="todo", replacement="TODO:", mode="expand")]
    assert match_snippet("please todo this", snips) is None


# ---------------------------------------------------------------------------
# match_snippet — regex mode
# ---------------------------------------------------------------------------


def test_regex_replace_mode():
    snips = [
        Snippet(
            trigger=r"hello \w+",
            replacement="GREETED",
            mode="replace",
            literal=False,
        )
    ]
    m = match_snippet("Hello world", snips)
    assert m is not None
    assert m.replacement == "GREETED"


def test_regex_expand_mode_keeps_remainder():
    snips = [
        Snippet(
            trigger=r"qa",
            replacement="QA:",
            mode="expand",
            literal=False,
        )
    ]
    m = match_snippet("QA the new feature", snips)
    assert m is not None
    assert m.replacement == "QA: the new feature"


def test_regex_invalid_pattern_returns_none():
    snips = [
        Snippet(trigger="(", replacement="x", mode="replace", literal=False),
    ]
    assert match_snippet("anything", snips) is None


# ---------------------------------------------------------------------------
# bypass_llm flag
# ---------------------------------------------------------------------------


def test_bypass_llm_true():
    snips = [Snippet(trigger="x", replacement="y", bypass_llm=True)]
    m = match_snippet("x", snips)
    assert m is not None
    assert m.bypass_llm is True


def test_bypass_llm_false():
    snips = [Snippet(trigger="x", replacement="y", bypass_llm=False)]
    m = match_snippet("x", snips)
    assert m is not None
    assert m.bypass_llm is False


# ---------------------------------------------------------------------------
# match_snippet — first match wins
# ---------------------------------------------------------------------------


def test_first_matching_snippet_wins():
    snips = [
        Snippet(trigger="foo", replacement="A"),
        Snippet(trigger="foo", replacement="B"),
    ]
    m = match_snippet("foo", snips)
    assert m is not None
    assert m.replacement == "A"


def test_empty_text_no_match():
    snips = [Snippet(trigger="x", replacement="y")]
    assert match_snippet("", snips) is None
    assert match_snippet("   ", snips) is None
