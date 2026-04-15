"""Regex post-processing for transcribed text.

Filters are defined in a JSON file as an ordered list of objects with
``name``, ``pattern``, ``replacement`` and optional ``flags``/``enabled``
fields. Filters run in file order, and Python's ``re.sub`` replacement
syntax is used, so ``\\1`` / ``\\g<name>`` back-references work.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# Flag names users can put in the JSON "flags" array.
_FLAG_MAP: dict[str, re.RegexFlag] = {
    "IGNORECASE": re.IGNORECASE,
    "I": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "M": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "S": re.DOTALL,
    "UNICODE": re.UNICODE,
    "U": re.UNICODE,
    "VERBOSE": re.VERBOSE,
    "X": re.VERBOSE,
    "ASCII": re.ASCII,
    "A": re.ASCII,
}


class FilterError(ValueError):
    """Raised when a filter definition is malformed."""


@dataclass(frozen=True)
class Filter:
    name: str
    pattern: re.Pattern[str]
    replacement: str
    enabled: bool = True
    raw_pattern: str = ""
    flags: tuple[str, ...] = field(default_factory=tuple)

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        return self.pattern.sub(self.replacement, text)


def _parse_flags(values: Iterable[str] | None) -> int:
    if not values:
        return 0
    combined = 0
    for v in values:
        key = str(v).strip().upper()
        if key not in _FLAG_MAP:
            raise FilterError(f"unknown regex flag: {v!r}")
        combined |= _FLAG_MAP[key]
    return combined


def build_filter(entry: dict) -> Filter:
    """Build a single Filter from a parsed JSON object."""
    if not isinstance(entry, dict):
        raise FilterError(f"filter entry must be an object, got {type(entry).__name__}")

    name = entry.get("name")
    pattern = entry.get("pattern")
    replacement = entry.get("replacement", "")
    flags = entry.get("flags")
    enabled = bool(entry.get("enabled", True))

    if not isinstance(name, str) or not name:
        raise FilterError("filter 'name' must be a non-empty string")
    if not isinstance(pattern, str):
        raise FilterError(f"filter {name!r}: 'pattern' must be a string")
    if not isinstance(replacement, str):
        raise FilterError(f"filter {name!r}: 'replacement' must be a string")

    flag_tuple: tuple[str, ...] = ()
    if flags is not None:
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            raise FilterError(f"filter {name!r}: 'flags' must be a list of strings")
        flag_tuple = tuple(flags)

    try:
        compiled = re.compile(pattern, _parse_flags(flag_tuple))
    except re.error as e:
        raise FilterError(f"filter {name!r}: invalid regex: {e}") from e

    return Filter(
        name=name,
        pattern=compiled,
        replacement=replacement,
        enabled=enabled,
        raw_pattern=pattern,
        flags=flag_tuple,
    )


def load_filters(path: str | Path) -> list[Filter]:
    """Load filters from a JSON file. Returns [] if the file does not exist."""
    p = Path(path)
    if not p.exists():
        log.debug("no filter file at %s; using empty filter chain", p)
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise FilterError(f"{p}: invalid JSON: {e}") from e
    return build_filters(raw)


def build_filters(entries: list) -> list[Filter]:
    """Build a list of Filters from already-parsed JSON data."""
    if not isinstance(entries, list):
        raise FilterError(
            f"filter file must be a JSON array, got {type(entries).__name__}"
        )
    return [build_filter(e) for e in entries]


def apply_filters(text: str, filters: Iterable[Filter]) -> str:
    """Run ``text`` through every filter in order."""
    for f in filters:
        try:
            text = f.apply(text)
        except re.error as e:  # pragma: no cover - compile already checks
            log.warning("filter %r failed at runtime: %s", f.name, e)
    return text
