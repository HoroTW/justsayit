"""Configuration loading for justsayit.

Reads ``$XDG_CONFIG_HOME/justsayit/config.toml`` (with sensible defaults)
and resolves the filter-file and cache-dir paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

from platformdirs import user_cache_dir, user_config_dir

from ._schema import (
    AudioConfig,
    VadConfig,
    ShortcutConfig,
    PasteConfig,
    ModelConfig,
    OverlayConfig,
    SoundConfig,
    PostprocessConfig,
    WindowClipboardPolicy,
    LogConfig,
    Config,
    _coerce_section,
)

APP_NAME = "justsayit"

# Distinctive header line written into commented-defaults files. Used as
# a stable marker so ``ensure_config_file`` can tell "already in
# commented form" apart from "still on the legacy fully-populated form"
# without re-checking content equality (which would re-trigger every
# time we ship a new default value).
_COMMENTED_FORM_MARKER = "# justsayit configuration (commented-defaults form)."


def config_dir() -> Path:
    return Path(user_config_dir(APP_NAME))


# --- .env / secret resolution ---------------------------------------------
#
# API keys for the OpenAI-compatible LLM and Whisper backends can come from
# three places, in priority order:
#   1. an explicit literal in the config file / profile (``api_key = "..."``)
#   2. a process environment variable (``api_key_env = "OPENAI_API_KEY"``)
#   3. ``$XDG_CONFIG_HOME/justsayit/.env`` — same KEY=VALUE format as
#      python-dotenv, loaded into ``os.environ`` on first call so anything
#      already exported in the shell still wins.
#
# Reading the .env at first ``resolve_secret`` call (instead of at import
# time) keeps test isolation straightforward — tests that monkeypatch
# ``config_dir`` don't have to worry about ordering.

_DOTENV_LOADED = False


def _dotenv_path() -> Path:
    return config_dir() / ".env"


def load_dotenv(*, force: bool = False) -> None:
    """Merge KEY=VALUE pairs from ``<config_dir>/.env`` into ``os.environ``.

    Process env wins — values already exported in the shell are preserved.
    Call is idempotent; pass ``force=True`` to re-read (mainly for tests).
    Lines beginning with ``#`` are comments; values may be wrapped in
    matched single or double quotes (which are stripped). Anything else
    is left as-is, preserving embedded whitespace inside the value.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED and not force:
        return
    _DOTENV_LOADED = True
    path = _dotenv_path()
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Strip an optional leading "export " for shell compatibility.
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ[key] = val


def resolve_secret(literal: str, env_var: str) -> str:
    """Return the secret value, preferring explicit literals.

    Order: literal config value > process env (after .env is merged in).
    Returns ``""`` if neither yields a value, so callers can branch on
    truthiness.
    """
    if literal:
        return literal
    if not env_var:
        return ""
    load_dotenv()
    return os.environ.get(env_var, "")


def cache_dir() -> Path:
    return Path(user_cache_dir(APP_NAME))


def models_dir() -> Path:
    return cache_dir() / "models"


# --- loading ---------------------------------------------------------------


def load_config(path: Path | None = None) -> Config:
    """Load config from ``path`` (defaults to ``$XDG_CONFIG_HOME/justsayit/config.toml``).

    After reading ``config.toml`` (the user's authored settings — never
    rewritten by the app), overlays the runtime-mutable subset from a
    sibling ``state.toml`` if present. State wins. See :func:`save_state`
    for the list of fields persisted as state.
    """
    if path is None:
        path = config_dir() / "config.toml"

    cfg = Config()
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)

        cfg.audio = _coerce_section(AudioConfig, raw.get("audio"))
        cfg.vad = _coerce_section(VadConfig, raw.get("vad"))
        cfg.shortcut = _coerce_section(ShortcutConfig, raw.get("shortcut"))
        cfg.paste = _coerce_section(PasteConfig, raw.get("paste"))
        cfg.model = _coerce_section(ModelConfig, raw.get("model"))
        cfg.overlay = _coerce_section(OverlayConfig, raw.get("overlay"))
        cfg.sound = _coerce_section(SoundConfig, raw.get("sound"))
        cfg.log = _coerce_section(LogConfig, raw.get("log"))
        cfg.postprocess = _coerce_section(PostprocessConfig, raw.get("postprocess"))
        cfg.window_clipboard_policy = _coerce_section(
            WindowClipboardPolicy, raw.get("window_clipboard_policy")
        )

        if "filters_path" in raw:
            cfg.filters_path = Path(raw["filters_path"]).expanduser()

    _apply_state_overlay(cfg, _state_path_for(path))
    return cfg


def _state_path_for(config_path: Path) -> Path:
    """Sibling ``state.toml`` next to *config_path*. Pairing rule shared
    by :func:`load_config` and :func:`save_state` so callers passing a
    custom config path (tests, alternate dirs) get matching state."""
    return config_path.parent / "state.toml"


def state_path() -> Path:
    """Default state-file location: ``$XDG_CONFIG_HOME/justsayit/state.toml``."""
    return config_dir() / "state.toml"


def _apply_state_overlay(cfg: Config, path: Path) -> None:
    """Read *path* and overlay the runtime-mutable fields onto *cfg*.
    Silent no-op if the file is missing or malformed — state is
    best-effort, the on-disk config.toml is the source of truth for
    everything else."""
    if not path.exists():
        return
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return
    vad = raw.get("vad") or {}
    if "enabled" in vad:
        cfg.vad.enabled = bool(vad["enabled"])
    ps = raw.get("postprocess") or {}
    if "enabled" in ps:
        cfg.postprocess.enabled = bool(ps["enabled"])
    if "profile" in ps:
        cfg.postprocess.profile = str(ps["profile"])


def save_state(cfg: Config, path: Path | None = None) -> None:
    """Persist the runtime-mutable subset of *cfg* to ``state.toml``.

    Fields written: ``vad.enabled``, ``postprocess.enabled``,
    ``postprocess.profile``. The rest of *cfg* is ignored — those are
    user-authored settings that live in ``config.toml`` and the app
    never rewrites that file.

    Best-effort: OSError is swallowed (state-tracking is non-essential;
    on next start the user just gets whatever's in config.toml)."""
    if path is None:
        path = state_path()
    content = (
        "# justsayit runtime state — written by the app whenever you toggle\n"
        "# auto-listen, postprocess, or switch profile (via the tray menu or\n"
        "# the equivalent hotkey). Editing by hand is fine but expect the\n"
        "# app to overwrite on the next change. Settings (audio, model,\n"
        "# overlay, …) live in config.toml — the app never rewrites that\n"
        "# file, so comments and customisations there survive forever.\n"
        "\n"
        "[vad]\n"
        f"enabled = {'true' if cfg.vad.enabled else 'false'}\n"
        "\n"
        "[postprocess]\n"
        f"enabled = {'true' if cfg.postprocess.enabled else 'false'}\n"
        f'profile = "{cfg.postprocess.profile}"\n'
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


def ensure_dirs(cfg: Config | None = None) -> None:
    """Create config and cache directories if they don't exist."""
    config_dir().mkdir(parents=True, exist_ok=True)
    cache_dir().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)


def render_config_toml(cfg: Config | None = None, *, commented: bool = False) -> str:
    """Render ``cfg`` (or the dataclass defaults) as a TOML document.

    Default mode emits every setting with its current value (good for
    ``show-defaults`` output and for inspecting a runtime config).

    With ``commented=True`` every value line is prefixed with ``# ``,
    leaving section headers uncommented. This is the "commented-defaults"
    form shipped on fresh install: only knobs the user explicitly cares
    about live in the file as uncommented lines, so future updates that
    move a default never collide with their settings.
    """
    from dataclasses import fields as dc_fields
    if cfg is None:
        cfg = Config()
    if commented:
        lines = [
            _COMMENTED_FORM_MARKER,
            "# Every key below is the shipped default, commented out.",
            "# Uncomment a line and change the value to override it.",
            "# Lines you don't touch keep tracking the shipped defaults,",
            "# so future updates that tweak a default just work.",
            "",
        ]
    else:
        lines = [
            "# justsayit configuration. Every setting is listed with its",
            "# current value. Delete or comment a line to fall back to the",
            "# built-in default (the app will not rewrite unchanged sections).",
            "",
        ]
    prefix = "# " if commented else ""

    def _render_scalar(val) -> str:
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, list):
            inner = ", ".join(_render_scalar(v) for v in val)
            return f"[{inner}]"
        return repr(val)

    for section_name in (
        "audio",
        "vad",
        "shortcut",
        "paste",
        "model",
        "overlay",
        "sound",
        "log",
        "postprocess",
        "window_clipboard_policy",
    ):
        section = getattr(cfg, section_name)
        lines.append(f"[{section_name}]")
        for f in dc_fields(section):
            val = getattr(section, f.name)
            if val is None:
                # Already commented (Optional/None default). No extra prefix.
                lines.append(f'# {f.name} = ""')
                continue
            if isinstance(val, dict):
                # Inline-table form keeps everything in one line for the
                # commented-defaults file. Empty dict: keep the line as
                # an empty inline table the user can fill in.
                if val:
                    items = ", ".join(
                        f"{k} = {_render_scalar(v)}" for k, v in val.items()
                    )
                    rendered = "{ " + items + " }"
                else:
                    rendered = "{}"
                lines.append(f"{prefix}{f.name} = {rendered}")
                continue
            rendered = _render_scalar(val)
            lines.append(f"{prefix}{f.name} = {rendered}")
        lines.append("")
    lines.append(f'{prefix}filters_path = "{cfg.filters_path}"')
    return "\n".join(lines) + "\n"


def default_config_toml() -> str:
    """Back-compat wrapper — same as ``render_config_toml(Config())``."""
    return render_config_toml(None)


def save_config(cfg: Config, path: Path | None = None) -> None:
    """Persist the runtime-mutable subset of *cfg* (back-compat wrapper).

    Historically wrote the entire merged config back to ``config.toml``,
    nuking any inline comments the user had written. Now delegates to
    :func:`save_state`, which writes only ``vad.enabled``,
    ``postprocess.enabled``, and ``postprocess.profile`` to a sibling
    ``state.toml``. The user's ``config.toml`` is never touched.

    *path* is interpreted as the config.toml location (for test
    parity); the state file is derived as a sibling.
    """
    if path is None:
        path = config_dir() / "config.toml"
    save_state(cfg, _state_path_for(path))


def _has_uncommented_assignment(text: str) -> bool:
    """True if *text* contains at least one ``key = value`` line that
    isn't commented out — the heuristic for "still on legacy fully-
    populated form"."""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        if "=" in stripped:
            return True
    return False


def ensure_commented_form_file(
    path: Path,
    commented: str,
    marker: str,
    *,
    suffix: str = ".bak-pre-commented-form",
    validator: Callable[[str], None] | None = None,
) -> bool:
    """Ensure *path* exists in commented-defaults form, migrating from
    legacy fully-populated form once if necessary.

    The marker is a stable header line embedded in *commented*; finding
    it in the user file means migration already happened, so we leave
    the file alone (the user may have uncommented overrides). For files
    that lack the marker AND contain uncommented ``key = value`` lines
    (legacy form), the existing file is backed up to
    ``<path><suffix>`` (if no backup exists yet) and overwritten with
    *commented*. Pure-comment / empty files get the commented template
    written without backup.

    If *validator* is given, it is called on the existing file content
    even when the marker is present; raising any exception means "this
    file is corrupt despite the marker" and triggers re-migration. This
    rescues files written by an earlier buggy template that happened to
    embed the marker.

    Returns ``True`` if the file was just written / migrated, ``False``
    if it was found already in commented form.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commented, encoding="utf-8")
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    was_marked_but_corrupt = False
    if marker in head[:8192]:
        if validator is None:
            return False
        try:
            validator(head)
            return False
        except Exception:
            was_marked_but_corrupt = True
    if was_marked_but_corrupt or _has_uncommented_assignment(head):
        backup = path.with_name(path.name + suffix)
        if not backup.exists():
            try:
                backup.write_bytes(path.read_bytes())
            except OSError:
                pass
    try:
        path.write_text(commented, encoding="utf-8")
    except OSError:
        pass
    return True


def ensure_config_file(path: Path | None = None) -> Path:
    """Write the commented-defaults ``config.toml`` if it doesn't
    exist yet, so the file is always available for inspection /
    editing. Returns the resolved path.

    One-shot migration: a pre-existing ``config.toml`` in the legacy
    fully-populated form (every key uncommented) is backed up to
    ``config.toml.bak-pre-commented-form`` and rewritten in the new
    commented form. Files that are already in the new form (marker
    present) or that have only commented content are left alone.
    """
    if path is None:
        path = config_dir() / "config.toml"
    commented = render_config_toml(None, commented=True)
    ensure_commented_form_file(path, commented, _COMMENTED_FORM_MARKER)
    return path


def ensure_filters_file(path: Path | None = None) -> Path:
    """Write the default ``filters.json`` if it doesn't exist. Returns the
    resolved path.

    The default chain handles dictated punctuation and line-break words
    (DE+EN), so the LLM postprocess step doesn't have to — and so the
    feature works without an LLM at all. Disable any rule by setting
    ``"enabled": false`` on it.
    """
    if path is None:
        path = config_dir() / "filters.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_default_filter_chain(), indent=2) + "\n",
            encoding="utf-8",
        )
    return path


def _default_filter_chain() -> list[dict]:
    """Built-in spoken-punctuation + cleanup chain written to filters.json
    on first run. Kept as a function so tests can apply it directly without
    touching disk."""

    # Per spoken word we ship a pair: a "drop when redundant" rule (when
    # STT already inserted the matching character right before the spoken
    # word) followed by a "replace" rule for everything else. The pair is
    # listed adjacent so a later rule's replacement can satisfy the next
    # word's lookbehind.
    def _pair(name: str, alternation: str, char: str) -> list[dict]:
        # Escape the character if it needs escaping inside a regex.
        esc = re.escape(char)
        return [
            {
                "name": f"spoken: drop redundant {name}",
                "pattern": rf"(?<=[.!?,:;])[ \t]*\b(?:{alternation})\b{esc}?",
                "replacement": "",
                "flags": ["IGNORECASE"],
            },
            {
                "name": f"spoken: {name} -> {char}",
                "pattern": rf"[ \t]*\b(?:{alternation})\b{esc}?",
                "replacement": char,
                "flags": ["IGNORECASE"],
            },
        ]

    chain: list[dict] = []

    # Line-break words first: their trailing punctuation (e.g. STT's "." or
    # "," after the dictation marker) gets absorbed by the optional
    # punctuation slot in the pattern, so we don't need a separate "drop
    # redundant" pair for them.
    chain.append(
        {
            "name": "spoken: new paragraph",
            "pattern": r"[ \t]*\b(?:neuer\s+Absatz|new\s+paragraph)\b[ \t]*[.,;:!?]?[ \t]*",
            "replacement": "\n\n",
            "flags": ["IGNORECASE"],
        }
    )
    chain.append(
        {
            "name": "spoken: new line",
            "pattern": r"[ \t]*\b(?:neue\s+Zeile|new\s+line)\b[ \t]*[.,;:!?]?[ \t]*",
            "replacement": "\n",
            "flags": ["IGNORECASE"],
        }
    )

    chain += _pair("Punkt/period", r"Punkt|period|full\s+stop", ".")
    chain += _pair("Komma/comma", r"Komma|comma", ",")
    chain += _pair("Fragezeichen/question mark", r"Fragezeichen|question\s+mark", "?")
    chain += _pair(
        "Ausrufezeichen/exclamation mark", r"Ausrufezeichen|exclamation\s+mark", "!"
    )
    chain += _pair("Doppelpunkt/colon", r"Doppelpunkt|colon", ":")
    chain += _pair("Semikolon/semicolon", r"Semikolon|semicolon", ";")

    # Cleanup. Order matters: drop punctuation-only lines BEFORE trimming
    # leading punctuation on a line, because the latter would turn ", "
    # into "" and leave the newline alone.
    chain.append(
        {
            "name": "drop punctuation-only line",
            "pattern": r"^[ \t]*[.,;:!?]+[ \t]*\n?",
            "replacement": "",
            "flags": ["MULTILINE"],
        }
    )
    chain.append(
        {
            "name": "drop leading punctuation on line",
            "pattern": r"(?<=\n)[ \t]*[.,;:!?]+[ \t]*",
            "replacement": "",
        }
    )
    chain.append(
        {
            "name": "trim trailing whitespace per line",
            "pattern": r"[ \t]+$",
            "replacement": "",
            "flags": ["MULTILINE"],
        }
    )
    chain.append(
        {
            "name": "collapse spaces (preserves newlines)",
            "pattern": r"[ \t]{2,}",
            "replacement": " ",
        }
    )
    chain.append(
        {
            "name": "trim whitespace",
            "pattern": r"^\s+|\s+$",
            "replacement": "",
        }
    )
    return chain


def ensure_after_llm_filters_file(path: Path | None = None) -> Path:
    """Write the default ``after_LLM_filters.json`` if it doesn't exist.

    These filters run after the LLM response, before paste — useful for
    normalizing typographic characters that models tend to emit (em dashes,
    curly quotes, …) to their plain-ASCII equivalents.
    Disable any rule by setting ``"enabled": false`` on it.
    """
    if path is None:
        path = config_dir() / "after_LLM_filters.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_default_after_llm_filter_chain(), indent=2) + "\n",
            encoding="utf-8",
        )
    return path


def _default_after_llm_filter_chain() -> list[dict]:
    return [
        {
            "name": "em dash to hyphen",
            "pattern": "—",
            "replacement": " - ",
            "enabled": True,
        },
        {
            "name": "en dash to hyphen",
            "pattern": "–",
            "replacement": "-",
            "enabled": True,
        },
        {
            "name": "left double quote to straight",
            "pattern": "[“„]",
            "replacement": '"',
            "enabled": True,
        },
        {
            "name": "right double quote to straight",
            "pattern": "”",
            "replacement": '"',
            "enabled": True,
        },
        {
            "name": "curly single quotes to straight",
            "pattern": "[‘’]",
            "replacement": "'",
            "enabled": True,
        },
        {
            "name": "ellipsis to three dots",
            "pattern": "…",
            "replacement": "...",
            "enabled": False,
        },
    ]


def ensure_tools_file(path: Path | None = None) -> Path:
    """Write an example ``tools.json`` if it doesn't exist. Returns the path.

    The file starts as an empty array so no tools are active by default.
    Users add tool objects to enable function calling from the LLM.
    """
    if path is None:
        path = config_dir() / "tools.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        example = [
            {
                "_comment": "Remove this entry and add your own tools. Each tool needs name, description, parameters (JSON Schema), and exec (shell command with {param} placeholders).",
                "name": "open_url",
                "description": "Open a URL in the default web browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to open",
                        }
                    },
                    "required": ["url"],
                },
                "exec": "xdg-open {url}",
                "enabled": False,
            }
        ]
        path.write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    return path
