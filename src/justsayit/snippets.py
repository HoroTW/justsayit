"""Canned-text snippets that expand from spoken triggers.

A snippet is a (trigger, replacement) pair stored in
``~/.config/justsayit/snippets.toml``. After regex-filter postprocessing
runs, the pipeline matches the cleaned text against every snippet; the
first match either replaces the utterance entirely (``mode="replace"``)
or expands a prefix and keeps the trailing dictation (``mode="expand"``).
When ``bypass_llm = true`` the snippet's replacement is pasted directly
without going through the LLM cleanup pass.

File schema::

    [[snippet]]
    trigger     = "insert email signature"
    replacement = "Best,\\nHoro"
    mode        = "replace"   # or "expand"
    bypass_llm  = true        # default true
    literal     = true        # default true; false → trigger is a regex

Trigger matching is case-insensitive and ignores ASCII punctuation by
default (``literal=true``). For ``mode="expand"`` the trigger only has
to match the START of the utterance; the remaining text is appended to
the replacement.
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip ASCII punctuation, collapse whitespace."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", cleaned).strip()


@dataclass(frozen=True)
class Snippet:
    trigger: str
    replacement: str
    mode: str = "replace"      # "replace" | "expand"
    bypass_llm: bool = True
    literal: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("replace", "expand"):
            raise ValueError(
                f"snippet {self.trigger!r}: mode must be 'replace' or 'expand', "
                f"got {self.mode!r}"
            )


@dataclass(frozen=True)
class SnippetMatch:
    replacement: str
    bypass_llm: bool


def _config_path() -> Path:
    from justsayit.config import config_dir
    return config_dir() / "snippets.toml"


def load_snippets(path: str | Path | None = None) -> list[Snippet]:
    """Load snippets from *path* (defaults to ``<config_dir>/snippets.toml``).

    Returns ``[]`` if the file does not exist or is empty. Malformed
    entries are logged and skipped — one bad snippet does not prevent
    the rest from loading.
    """
    if path is None:
        p = _config_path()
    else:
        p = Path(path)
    if not p.exists():
        log.debug("no snippets file at %s; using empty snippet list", p)
        return []
    try:
        with p.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        log.error("snippets file %s: invalid TOML: %s", p, e)
        return []
    entries = raw.get("snippet") or []
    if not isinstance(entries, list):
        log.error("snippets file %s: 'snippet' must be an array of tables", p)
        return []
    snippets: list[Snippet] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            log.warning("snippets[%d]: not a table; skipping", i)
            continue
        trigger = entry.get("trigger")
        replacement = entry.get("replacement")
        if not isinstance(trigger, str) or not trigger:
            log.warning("snippets[%d]: missing/empty 'trigger'; skipping", i)
            continue
        if not isinstance(replacement, str):
            log.warning("snippets[%d]: 'replacement' must be a string; skipping", i)
            continue
        try:
            snippets.append(
                Snippet(
                    trigger=trigger,
                    replacement=replacement,
                    mode=str(entry.get("mode", "replace")),
                    bypass_llm=bool(entry.get("bypass_llm", True)),
                    literal=bool(entry.get("literal", True)),
                )
            )
        except ValueError as e:
            log.warning("snippets[%d]: %s; skipping", i, e)
    log.info("loaded %d snippet(s) from %s", len(snippets), p)
    return snippets


def _match_one(text: str, snip: Snippet) -> SnippetMatch | None:
    """Return a SnippetMatch if *snip* matches *text*, else None."""
    if snip.literal:
        norm_text = _normalize(text)
        norm_trigger = _normalize(snip.trigger)
        if not norm_trigger:
            return None
        if snip.mode == "replace":
            if norm_text == norm_trigger:
                return SnippetMatch(snip.replacement, snip.bypass_llm)
            return None
        # expand: trigger must match the START of the normalized text.
        if norm_text == norm_trigger:
            return SnippetMatch(snip.replacement, snip.bypass_llm)
        prefix = norm_trigger + " "
        if norm_text.startswith(prefix):
            remainder = norm_text[len(prefix):].strip()
            replacement = (
                snip.replacement + " " + remainder if remainder else snip.replacement
            )
            return SnippetMatch(replacement, snip.bypass_llm)
        return None

    # Regex mode: don't normalize; let users author the regex precisely.
    flags = re.IGNORECASE
    try:
        if snip.mode == "replace":
            pattern = re.compile(rf"\A\s*(?:{snip.trigger})\s*\Z", flags)
            if pattern.match(text):
                return SnippetMatch(snip.replacement, snip.bypass_llm)
            return None
        # expand: anchor at start, capture rest.
        pattern = re.compile(rf"\A\s*(?:{snip.trigger})(?:\s+(?P<rest>.+))?\s*\Z", flags | re.DOTALL)
        m = pattern.match(text)
        if m is None:
            return None
        rest = (m.group("rest") or "").strip()
        replacement = (
            snip.replacement + " " + rest if rest else snip.replacement
        )
        return SnippetMatch(replacement, snip.bypass_llm)
    except re.error as e:
        log.warning("snippet %r: invalid regex: %s", snip.trigger, e)
        return None


def match_snippet(
    text: str, snippets: Iterable[Snippet]
) -> SnippetMatch | None:
    """Return the first matching snippet for *text*, or None."""
    if not text:
        return None
    for snip in snippets:
        m = _match_one(text, snip)
        if m is not None:
            return m
    return None
