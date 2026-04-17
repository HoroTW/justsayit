"""Tests for the regex post-processing filter chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from justsayit.config import (
    _default_filter_chain,
    defaults_baseline_path,
    ensure_filters_file,
)
from justsayit.filters import (
    Filter,
    FilterError,
    apply_filters,
    build_filter,
    build_filters,
    load_filters,
)


def _default_chain():
    """Compile the shipped default chain for end-to-end assertions."""
    return build_filters(_default_filter_chain())


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


# --- shipped default filter chain -------------------------------------------


def test_default_chain_compiles():
    # Every filter must compile without raising.
    chain = _default_chain()
    assert len(chain) > 0


def test_default_chain_headline_example():
    chain = _default_chain()
    text = "Hallo, neue Zeile. Ich komme nicht. Punkt. Neue Zeile, eure Katja."
    expected = "Hallo,\nIch komme nicht.\neure Katja."
    assert apply_filters(text, chain) == expected


def test_default_chain_replaces_punkt_mid_sentence():
    chain = _default_chain()
    assert apply_filters("Hello Punkt next", chain) == "Hello. next"


def test_default_chain_drops_redundant_punkt():
    chain = _default_chain()
    # STT already wrote "." after "ich"; spoken "Punkt" must be dropped, not stacked.
    assert apply_filters("ich. Punkt. weiter", chain) == "ich. weiter"


def test_default_chain_replaces_komma():
    chain = _default_chain()
    assert apply_filters("Hello Komma world", chain) == "Hello, world"


def test_default_chain_drops_redundant_komma():
    chain = _default_chain()
    assert apply_filters("Hallo, Komma weiter", chain) == "Hallo, weiter"


def test_default_chain_question_mark_de_and_en():
    chain = _default_chain()
    assert apply_filters("ist das so Fragezeichen", chain) == "ist das so?"
    assert apply_filters("really question mark", chain) == "really?"


def test_default_chain_exclamation():
    chain = _default_chain()
    assert apply_filters("super Ausrufezeichen", chain) == "super!"
    assert apply_filters("yes exclamation mark", chain) == "yes!"


def test_default_chain_colon_and_semicolon():
    chain = _default_chain()
    assert apply_filters("Liste Doppelpunkt eins", chain) == "Liste: eins"
    assert apply_filters("a Semikolon b", chain) == "a; b"


def test_default_chain_new_paragraph():
    chain = _default_chain()
    text = "Erster Absatz. Neuer Absatz. Zweiter."
    assert apply_filters(text, chain) == "Erster Absatz.\n\nZweiter."


def test_default_chain_new_line_english():
    chain = _default_chain()
    assert apply_filters("Hello comma new line greetings", chain) == (
        "Hello,\ngreetings"
    )


def test_default_chain_full_stop_en():
    chain = _default_chain()
    assert apply_filters("Hello full stop next", chain) == "Hello. next"


def test_default_chain_preserves_punctuation_inside_words():
    # `\b` boundary should keep "Diskussionspunkt" intact.
    chain = _default_chain()
    assert apply_filters("Der Diskussionspunkt war wichtig.", chain) == (
        "Der Diskussionspunkt war wichtig."
    )


def test_default_chain_collapse_preserves_newlines():
    chain = _default_chain()
    # The collapse-spaces rule must not turn `\n` into a single space.
    assert apply_filters("a    neue Zeile    b", chain) == "a\nb"


def test_default_chain_drops_stray_punct_only_line():
    chain = _default_chain()
    # A literal "." sitting alone on a line should be removed.
    assert apply_filters("Hallo\n.\nWelt", chain) == "Hallo\nWelt"


def test_ensure_filters_file_writes_default(tmp_path: Path):
    p = tmp_path / "filters.json"
    ensure_filters_file(p)
    assert p.exists()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    names = [entry["name"] for entry in raw]
    assert any("Punkt" in n for n in names)
    assert any("new line" in n for n in names)
    # And it must compile end-to-end.
    chain = build_filters(raw)
    assert apply_filters(
        "Test Komma noch eins Punkt", chain
    ) == "Test, noch eins."


def test_ensure_filters_file_does_not_overwrite(tmp_path: Path):
    p = tmp_path / "filters.json"
    p.write_text("[]", encoding="utf-8")
    ensure_filters_file(p)
    assert p.read_text(encoding="utf-8") == "[]"


def test_defaults_baseline_path_format(tmp_path: Path):
    # Convention is "<stem>.defaults-baseline<suffix>" so install.sh's
    # baseline_path_for() can derive the same path with shell parameter
    # expansion.
    assert defaults_baseline_path(tmp_path / "filters.json") == (
        tmp_path / "filters.defaults-baseline.json"
    )
    assert defaults_baseline_path(tmp_path / "config.toml") == (
        tmp_path / "config.defaults-baseline.toml"
    )


def test_ensure_filters_file_writes_baseline_on_fresh_install(tmp_path: Path):
    p = tmp_path / "filters.json"
    ensure_filters_file(p)
    baseline = defaults_baseline_path(p)
    assert baseline.exists(), "baseline sidecar should be written alongside filters.json"
    assert baseline.read_text(encoding="utf-8") == p.read_text(encoding="utf-8")


def test_ensure_filters_file_heals_baseline_when_user_in_sync(tmp_path: Path):
    # Pre-baseline install: user has the current shipped defaults verbatim
    # but no baseline file. Next app start should snapshot one silently.
    p = tmp_path / "filters.json"
    p.write_text(
        json.dumps(_default_filter_chain(), indent=2) + "\n", encoding="utf-8"
    )
    ensure_filters_file(p)
    assert defaults_baseline_path(p).exists()


def test_ensure_filters_file_does_not_heal_when_user_customised(tmp_path: Path):
    # Pre-baseline install with a customised file: no auto-baseline. The
    # update path will degrade to a plain diff prompt and write the
    # baseline only after the user accepts/rejects.
    p = tmp_path / "filters.json"
    p.write_text("[]", encoding="utf-8")
    ensure_filters_file(p)
    assert not defaults_baseline_path(p).exists()
