"""Configuration loading for justsayit.

Reads ``$XDG_CONFIG_HOME/justsayit/config.toml`` (with sensible defaults)
and resolves the filter-file and cache-dir paths.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir

APP_NAME = "justsayit"


def config_dir() -> Path:
    return Path(user_config_dir(APP_NAME))


def cache_dir() -> Path:
    return Path(user_cache_dir(APP_NAME))


def models_dir() -> Path:
    return cache_dir() / "models"


# --- dataclasses -----------------------------------------------------------


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    device: str | int | None = None  # None = system default
    block_ms: int = 30  # audio callback block size in ms
    # Rolling buffer of the last N ms of audio kept while idle. When a
    # recording starts (VAD-triggered or hotkey-triggered) the ring is
    # prepended to the segment so we don't clip the first phoneme while
    # Silero is still deciding "is this speech?" or while the user's
    # finger is still on the hotkey.
    lookback_ms: int = 300


@dataclass
class VadConfig:
    # Master switch. When False, recording is purely hotkey-driven:
    # toggle starts, toggle stops, transcribe the buffer. No Silero, no
    # auto-open, no 3s validation. This is the simple path and the
    # default while we stabilise the app.
    enabled: bool = False
    # RMS energy in [0, 1] that opens a candidate recording window.
    # Silero then confirms speech from there.
    open_rms: float = 0.015
    # Silero VAD thresholds
    silero_threshold: float = 0.5
    min_silence_seconds: float = 0.8
    min_speech_seconds: float = 0.25
    # How long (s) we transcribe after opening to validate "real speech";
    # if no words come out, we discard and return to waiting.
    validation_seconds: float = 3.0
    # Hard ceiling on a single recording segment.
    max_segment_seconds: float = 60.0


@dataclass
class ShortcutConfig:
    # Unique ID used when registering with the portal.
    id: str = "toggle-dictation"
    description: str = "Toggle justsayit dictation"
    # Preferred accelerator in XDG format, e.g. "SUPER+backslash".
    # The user can always rebind in their desktop settings.
    preferred: str = "SUPER+backslash"


@dataclass
class PasteConfig:
    enabled: bool = True
    # Keystroke-injection tool. Only "dotool" is currently supported.
    backend: str = "dotool"
    # Key combination to trigger paste in the focused app.
    paste_combo: str = "shift+insert"
    # Minimum time (ms) to wait between the stop-hotkey being pressed and
    # the synthetic paste firing, so the user has time to release the
    # modifier keys from that hotkey. If transcription + filtering took
    # longer than this, no extra wait happens.
    release_delay_ms: int = 250
    # Brief pause (ms) between wl-copy and the synthetic keystroke to let
    # the clipboard settle; some apps (Electron) race without this.
    settle_ms: int = 40
    # Hard timeout (s) for each subprocess call. Keeps a broken setup
    # from blocking the transcription worker forever.
    subprocess_timeout: float = 5.0
    # If non-zero: prepend a space before the transcription when the
    # previous successful transcription finished within this many
    # milliseconds. Lets you dictate continuously without manually
    # inserting spaces between phrases. Ignored when
    # append_trailing_space is True (the trailing space already serves
    # as the separator).
    auto_space_timeout_ms: int = 0
    # Always append a trailing space after every transcription so the
    # cursor is ready for the next word. When enabled together with
    # auto_space_timeout_ms, this takes precedence (a warning
    # notification is shown and the prefix behaviour is suppressed).
    append_trailing_space: bool = False
    # When True, ``wl-copy --sensitive`` is used instead of plain ``wl-copy``.
    # The ``--sensitive`` flag tells clipboard managers (e.g. KDE Klipper) to
    # skip recording this entry.  Text IS still available for a manual Ctrl+V
    # paste immediately after dictation. Hint: We paste into both primary
    # and clipboard selections, so the text is available for middle-click and
    # shift+insert paste on most applications.
    skip_clipboard_history: bool = True
    # When True, text is injected directly via ``dotool type`` — the
    # clipboard is never used at all, so clipboard managers cannot record
    # anything.  Text will NOT be available for manual re-paste after
    # dictation.  Requires backend = "dotool".
    # Takes precedence over skip_clipboard_history if both are True.
    type_directly: bool = False
    # When True (default), the regular clipboard is restored to its previous
    # content after the synthetic paste keystroke fires, so the user's own
    # copied content is not clobbered by dictation.  Primary/selection
    # clipboard is not restored.  No-op when type_directly is True.
    restore_clipboard: bool = True


@dataclass
class ModelConfig:
    # Transcription backend. "parakeet" uses sherpa-onnx (bundled dep,
    # default). "whisper" uses faster-whisper (optional dep — install with
    # the [whisper] extra or run: uv pip install faster-whisper).
    backend: str = "parakeet"

    # --- Parakeet (sherpa-onnx) -------------------------------------------
    # sherpa-onnx publishes packaged model bundles as tar.bz2 release assets.
    # Default: Parakeet TDT v3 multilingual INT8.
    parakeet_archive_url: str = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    )
    # Name of the top-level directory inside the archive.
    parakeet_archive_dir: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    # Filenames inside the unpacked directory.
    parakeet_encoder: str = "encoder.int8.onnx"
    parakeet_decoder: str = "decoder.int8.onnx"
    parakeet_joiner: str = "joiner.int8.onnx"
    parakeet_tokens: str = "tokens.txt"

    # --- faster-whisper / distil-whisper ------------------------------------
    # HuggingFace model ID or a local directory path. Good options:
    #   "Systran/faster-distil-whisper-large-v3"  (multilingual, default)
    #   "Systran/faster-whisper-large-v3"          (full large-v3)
    #   "Systran/faster-whisper-large-v3-turbo"    (faster, slightly lower quality)
    #   "Systran/faster-distil-whisper-medium.en"  (English-only, small)
    whisper_model: str = "Systran/faster-distil-whisper-large-v3"
    # Inference device: "cpu" or "cuda". Auto to "cpu" on systems without GPU.
    whisper_device: str = "cpu"
    # CTranslate2 quantisation. "int8" is fastest on CPU with little quality
    # loss. Use "float16" on CUDA, "float32" for maximum accuracy.
    whisper_compute_type: str = "int8"

    # --- Shared ---------------------------------------------------------------
    # Silero VAD ONNX (tiny file, downloaded directly).
    vad_url: str = (
        "https://github.com/snakers4/silero-vad/raw/master/"
        "src/silero_vad/data/silero_vad.onnx"
    )
    # Inference threads (0 = library default). Applies to both backends.
    num_threads: int = 2


@dataclass
class OverlayConfig:
    enabled: bool = True
    # "bottom" or "top" edge of the output.
    anchor: str = "bottom"
    margin: int = 24
    width: int = 174
    height: int = 56
    # Multiplier applied to the raw microphone level before it is
    # displayed in the visualizer bar. Increase above 1.0 if your mic
    # records quietly and the bar barely moves; decrease below 1.0 if
    # the bar clips on every word.
    visualizer_sensitivity: float = 1.0
    # Background opacity of the overlay pill (0.0 = fully transparent,
    # 1.0 = fully opaque).
    opacity: float = 0.78
    # How long (ms) the overlay stays visible after a successful paste so
    # the user can read the transcribed / LLM-cleaned text.
    # 0 = hide immediately after paste (original behaviour).
    result_linger_ms: int = 5_000
    # Maximum width/height (px) the overlay may expand to when showing the
    # result text fields.  The overlay always starts at width × height and
    # grows to fit the content up to these limits.
    max_width: int = 600
    max_height: int = 400


@dataclass
class SoundConfig:
    enabled: bool = True
    # Playback volume for notification sounds (0.0 = silent, 1.0 = full).
    volume: float = 1.0
    # Volume scale for the soft start chime played when VAD enters the
    # VALIDATING state (relative to sound.volume). Kept quieter than the
    # confirmed-recording chime because the result is still uncertain.
    validating_volume_scale: float = 0.4


@dataclass
class PostprocessConfig:
    # Master switch. When False the LLM step is skipped entirely.
    enabled: bool = False
    # Profile name (resolved to config_dir()/postprocess/<profile>.toml)
    # or a direct path to a .toml file.
    profile: str = "gemma4-cleanup"


@dataclass
class LogConfig:
    # Rotating debug log written to disk. Off by default — turn this on
    # when you need to share a trace of a bug. Console logging is always
    # on and controlled independently by --log-level.
    file_enabled: bool = False
    # Empty string = default to <cache_dir>/justsayit.log.
    file_path: str = ""
    file_level: str = "DEBUG"
    file_max_bytes: int = 5_000_000
    file_backup_count: int = 3


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    shortcut: ShortcutConfig = field(default_factory=ShortcutConfig)
    paste: PasteConfig = field(default_factory=PasteConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    log: LogConfig = field(default_factory=LogConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # File path for user regex filters.
    filters_path: Path = field(default_factory=lambda: config_dir() / "filters.json")


# --- loading ---------------------------------------------------------------


def _coerce_section(section_cls, data: dict[str, Any] | None):
    """Create a dataclass instance from a TOML table, ignoring unknown keys."""
    if data is None:
        return section_cls()
    if not is_dataclass(section_cls):  # pragma: no cover
        raise TypeError(f"{section_cls} is not a dataclass")
    kwargs: dict[str, Any] = {}
    for f in fields(section_cls):
        if f.name in data:
            kwargs[f.name] = data[f.name]
    return section_cls(**kwargs)


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


def render_config_toml(cfg: Config | None = None) -> str:
    """Render ``cfg`` (or the dataclass defaults) as a commented TOML
    document with every setting present — good for explorability and for
    round-tripping a runtime-modified config back to disk."""
    if cfg is None:
        cfg = Config()
    lines = [
        "# justsayit configuration. Every setting is listed with its",
        "# current value. Delete or comment a line to fall back to the",
        "# built-in default (the app will not rewrite unchanged sections).",
        "",
    ]
    for section_name in ("audio", "vad", "shortcut", "paste", "model", "overlay", "sound", "log", "postprocess"):
        section = getattr(cfg, section_name)
        lines.append(f"[{section_name}]")
        for f in fields(section):
            val = getattr(section, f.name)
            if val is None:
                lines.append(f'# {f.name} = ""')
                continue
            if isinstance(val, bool):
                rendered = "true" if val else "false"
            elif isinstance(val, str):
                rendered = f'"{val}"'
            else:
                rendered = repr(val)
            lines.append(f"{f.name} = {rendered}")
        lines.append("")
    lines.append(f'filters_path = "{cfg.filters_path}"')
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


def defaults_baseline_path(user_path: Path) -> Path:
    """Sidecar that records the shipped defaults the user file derives from.

    ``install.sh --update`` reads this to tell three cases apart that
    otherwise look identical:
      1. user file == baseline (never customised, defaults moved → safe to update)
      2. user file != baseline, baseline == new defaults (user customised, defaults didn't move → leave alone)
      3. both diverged (true 3-way: show what changed in defaults vs what user customised)

    Layout: each directory gets a hidden ``.baseline/`` subdir holding
    snapshots that share the user-file's name verbatim. So
    ``~/.config/justsayit/filters.json`` →
    ``~/.config/justsayit/.baseline/filters.json`` and
    ``~/.config/justsayit/postprocess/gemma4-cleanup.toml`` →
    ``~/.config/justsayit/postprocess/.baseline/gemma4-cleanup.toml``.
    install.sh derives the same path with shell parameter expansion.

    Best-effort state — if the baseline goes missing, install.sh degrades
    cleanly to a plain diff prompt, then writes a fresh baseline.
    """
    return user_path.parent / ".baseline" / user_path.name


def _legacy_baseline_path(user_path: Path) -> Path:
    """Pre-0.8.7 sidecar layout: ``foo.json`` → ``foo.defaults-baseline.json``
    in the same directory. Kept for one-shot migration."""
    return user_path.with_name(
        f"{user_path.stem}.defaults-baseline{user_path.suffix}"
    )


def _migrate_legacy_baseline(user_path: Path) -> None:
    """Move a pre-0.8.7 sidecar (``foo.defaults-baseline.json``) into the
    new ``.baseline/`` subdir if present. Best-effort, idempotent — a
    no-op if the legacy file is missing or the new-layout file already
    exists. Called lazily from baseline read/write paths so users who
    never re-run ``install.sh --update`` still get migrated on next app
    start."""
    legacy = _legacy_baseline_path(user_path)
    if not legacy.exists():
        return
    new = defaults_baseline_path(user_path)
    if new.exists():
        try:
            legacy.unlink()
        except OSError:
            pass
        return
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(new)
    except OSError:
        pass


def _write_baseline(user_path: Path, content: str) -> None:
    """Write the defaults baseline next to *user_path*. Best-effort —
    OSError is swallowed since baseline-tracking is non-essential."""
    _migrate_legacy_baseline(user_path)
    try:
        target = defaults_baseline_path(user_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError:
        pass


def _heal_baseline(user_path: Path, current_defaults: str) -> None:
    """Migration helper: if the user file matches current shipped
    defaults verbatim and no baseline exists yet, snapshot one. Lets
    pre-baseline installs auto-upgrade silently for users who never
    customised."""
    _migrate_legacy_baseline(user_path)
    baseline = defaults_baseline_path(user_path)
    if baseline.exists():
        return
    try:
        if user_path.read_text(encoding="utf-8") == current_defaults:
            _write_baseline(user_path, current_defaults)
    except OSError:
        pass


def ensure_config_file(path: Path | None = None) -> Path:
    """Write the fully-populated default ``config.toml`` if it doesn't
    exist yet, so the file is always available for inspection / editing.
    Returns the resolved path."""
    if path is None:
        path = config_dir() / "config.toml"
    rendered = render_config_toml(None)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        _write_baseline(path, rendered)
    else:
        _heal_baseline(path, rendered)
    return path


def ensure_filters_file(path: Path | None = None) -> Path:
    """Write the default ``filters.json`` if it doesn't exist. Returns the
    resolved path.

    The default chain handles dictated punctuation and line-break words
    (DE+EN), so the LLM postprocess step doesn't have to — and so the
    feature works without an LLM at all. Disable any rule by setting
    ``"enabled": false`` on it.
    """
    import json

    if path is None:
        path = config_dir() / "filters.json"
    rendered = json.dumps(_default_filter_chain(), indent=2) + "\n"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        _write_baseline(path, rendered)
    else:
        _heal_baseline(path, rendered)
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
    chain.append({
        "name": "spoken: new paragraph",
        "pattern": r"[ \t]*\b(?:neuer\s+Absatz|new\s+paragraph)\b[ \t]*[.,;:!?]?[ \t]*",
        "replacement": "\n\n",
        "flags": ["IGNORECASE"],
    })
    chain.append({
        "name": "spoken: new line",
        "pattern": r"[ \t]*\b(?:neue\s+Zeile|new\s+line)\b[ \t]*[.,;:!?]?[ \t]*",
        "replacement": "\n",
        "flags": ["IGNORECASE"],
    })

    chain += _pair("Punkt/period", r"Punkt|period|full\s+stop", ".")
    chain += _pair("Komma/comma", r"Komma|comma", ",")
    chain += _pair("Fragezeichen/question mark", r"Fragezeichen|question\s+mark", "?")
    chain += _pair("Ausrufezeichen/exclamation mark", r"Ausrufezeichen|exclamation\s+mark", "!")
    chain += _pair("Doppelpunkt/colon", r"Doppelpunkt|colon", ":")
    chain += _pair("Semikolon/semicolon", r"Semikolon|semicolon", ";")

    # Cleanup. Order matters: drop punctuation-only lines BEFORE trimming
    # leading punctuation on a line, because the latter would turn ", "
    # into "" and leave the newline alone.
    chain.append({
        "name": "drop punctuation-only line",
        "pattern": r"^[ \t]*[.,;:!?]+[ \t]*\n?",
        "replacement": "",
        "flags": ["MULTILINE"],
    })
    chain.append({
        "name": "drop leading punctuation on line",
        "pattern": r"(?<=\n)[ \t]*[.,;:!?]+[ \t]*",
        "replacement": "",
    })
    chain.append({
        "name": "trim trailing whitespace per line",
        "pattern": r"[ \t]+$",
        "replacement": "",
        "flags": ["MULTILINE"],
    })
    chain.append({
        "name": "collapse spaces (preserves newlines)",
        "pattern": r"[ \t]{2,}",
        "replacement": " ",
    })
    chain.append({
        "name": "trim whitespace",
        "pattern": r"^\s+|\s+$",
        "replacement": "",
    })
    return chain


# silence "unused" warning on MISSING import
_ = MISSING
