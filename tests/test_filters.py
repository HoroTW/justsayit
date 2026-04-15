"""Tests for the regex post-processing filter chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from justsayit.filters import (
    Filter,
    FilterError,
    apply_filters,
    build_filter,
    build_filters,
    load_filters,
)


# --- parsing / validation ---------------------------------------------------


def test_build_filter_minimal():
    f = build_filter({"name": "noop", "pattern": "a", "replacement": "b"})
    assert isinstance(f, Filter)
    assert f.name == "noop"
    assert f.enabled is True
    assert f.apply("aaa") == "bbb"


def test_build_filter_missing_name_rejected():
    with pytest.raises(FilterError):
        build_filter({"pattern": "a", "replacement": "b"})


def test_build_filter_empty_name_rejected():
    with pytest.raises(FilterError):
        build_filter({"name": "", "pattern": "a", "replacement": "b"})


def test_build_filter_non_string_pattern_rejected():
    with pytest.raises(FilterError):
        build_filter({"name": "x", "pattern": 5, "replacement": "b"})


def test_build_filter_bad_regex_rejected():
    with pytest.raises(FilterError):
        build_filter({"name": "broken", "pattern": "(", "replacement": ""})


def test_build_filter_unknown_flag_rejected():
    with pytest.raises(FilterError):
        build_filter(
            {"name": "x", "pattern": "a", "replacement": "b", "flags": ["NOPE"]}
        )


def test_build_filter_flags_must_be_list_of_strings():
    with pytest.raises(FilterError):
        build_filter(
            {"name": "x", "pattern": "a", "replacement": "b", "flags": "IGNORECASE"}
        )


def test_build_filters_requires_array():
    with pytest.raises(FilterError):
        build_filters({"not": "a list"})  # type: ignore[arg-type]


# --- matching group replacement --------------------------------------------


def test_numeric_backreference():
    f = build_filter(
        {
            "name": "email-spoken",
            "pattern": r"(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)",
            "replacement": r"\1@\2.\3",
            "flags": ["IGNORECASE"],
        }
    )
    assert f.apply("send it to alice at example dot com please") == (
        "send it to alice@example.com please"
    )


def test_named_group_backreference():
    f = build_filter(
        {
            "name": "swap",
            "pattern": r"(?P<first>\w+)\s+(?P<second>\w+)",
            "replacement": r"\g<second> \g<first>",
        }
    )
    assert f.apply("hello world") == "world hello"


def test_multiple_matches_replaced():
    f = build_filter(
        {
            "name": "num",
            "pattern": r"\d+",
            "replacement": "#",
        }
    )
    assert f.apply("room 101 at 202") == "room # at #"


def test_no_match_is_passthrough():
    f = build_filter({"name": "nop", "pattern": r"xyz", "replacement": "!"})
    assert f.apply("hello world") == "hello world"


# --- flags -------------------------------------------------------------------


def test_ignorecase_flag():
    f = build_filter(
        {
            "name": "ci",
            "pattern": r"hello",
            "replacement": "hi",
            "flags": ["IGNORECASE"],
        }
    )
    assert f.apply("HELLO there") == "hi there"


def test_multiline_flag():
    f = build_filter(
        {
            "name": "ml",
            "pattern": r"^foo",
            "replacement": "bar",
            "flags": ["MULTILINE"],
        }
    )
    assert f.apply("foo\nfoo") == "bar\nbar"


def test_dotall_flag():
    f = build_filter(
        {
            "name": "da",
            "pattern": r"a.b",
            "replacement": "X",
            "flags": ["DOTALL"],
        }
    )
    assert f.apply("a\nb") == "X"


# --- enabled toggle ----------------------------------------------------------


def test_disabled_filter_skipped():
    f = build_filter(
        {"name": "off", "pattern": "a", "replacement": "b", "enabled": False}
    )
    assert f.apply("aaa") == "aaa"


# --- chain semantics ---------------------------------------------------------


def test_filters_apply_in_order():
    filters = build_filters(
        [
            {"name": "first", "pattern": "foo", "replacement": "bar"},
            {"name": "second", "pattern": "bar", "replacement": "baz"},
        ]
    )
    assert apply_filters("foo", filters) == "baz"


def test_filters_order_second_wins():
    # If order were reversed, "foo" would never become "baz" via this chain.
    filters = build_filters(
        [
            {"name": "second", "pattern": "bar", "replacement": "baz"},
            {"name": "first", "pattern": "foo", "replacement": "bar"},
        ]
    )
    assert apply_filters("foo", filters) == "bar"


def test_empty_chain_passthrough():
    assert apply_filters("unchanged", []) == "unchanged"


# --- unicode / edge cases ----------------------------------------------------


def test_unicode_match():
    f = build_filter(
        {"name": "u", "pattern": r"café", "replacement": "coffee"}
    )
    assert f.apply("I want café now") == "I want coffee now"


def test_empty_input():
    f = build_filter({"name": "x", "pattern": "a", "replacement": "b"})
    assert f.apply("") == ""


def test_replacement_can_be_empty_string():
    f = build_filter({"name": "strip", "pattern": r"\s+um\s+", "replacement": " "})
    assert f.apply("so um this") == "so this"


# --- file loading ------------------------------------------------------------


def test_load_filters_missing_file_returns_empty(tmp_path: Path):
    assert load_filters(tmp_path / "nope.json") == []


def test_load_filters_roundtrip(tmp_path: Path):
    p = tmp_path / "filters.json"
    p.write_text(
        json.dumps(
            [
                {"name": "dots", "pattern": r"\.\.\.", "replacement": "…"},
                {
                    "name": "question",
                    "pattern": r"\bwhat\b",
                    "replacement": "WHAT",
                    "flags": ["IGNORECASE"],
                },
            ]
        ),
        encoding="utf-8",
    )
    filters = load_filters(p)
    assert [f.name for f in filters] == ["dots", "question"]
    assert apply_filters("What... now", filters) == "WHAT… now"


def test_load_filters_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(FilterError):
        load_filters(p)


def test_load_filters_must_be_list(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"name":"x"}', encoding="utf-8")
    with pytest.raises(FilterError):
        load_filters(p)
