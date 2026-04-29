"""Configuration dataclasses for justsayit."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from pathlib import Path
from typing import Any


# silence "unused" warning on MISSING import
_ = MISSING


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
    # Segments shorter than this are dropped before transcription,
    # regex filters, and LLM cleanup.  Set to 0 to disable.
    skip_segments_below_seconds: float = 1.0


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
    # Delay (ms) between the synthetic paste keystroke firing and the
    # previous clipboard being restored. Needs to be long enough for the
    # target app to actually read the dictated text off the clipboard
    # before we overwrite it — 250ms covers most slow Electron apps.
    # No-op when restore_clipboard is False or type_directly is True.
    restore_delay_ms: int = 250


@dataclass
class ModelConfig:
    # Transcription backend.
    #   "parakeet" — local sherpa-onnx (bundled dep, default).
    #   "whisper"  — local faster-whisper (optional [whisper] extra).
    #   "openai"   — OpenAI-compatible /audio/transcriptions endpoint
    #                (no local model loaded; needs openai_endpoint +
    #                openai_model + an API key via openai_api_key /
    #                openai_api_key_env / .env).
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

    # --- OpenAI-compatible /audio/transcriptions endpoint -----------------
    # Base URL of an OpenAI-compatible API (no trailing slash). Examples:
    #   "https://api.openai.com/v1"            (OpenAI Whisper)
    #   "https://api.groq.com/openai/v1"       (Groq whisper-large-v3)
    #   "http://localhost:8000/v1"             (self-hosted: vLLM, faster-
    #                                           whisper-server, whisper.cpp …)
    openai_endpoint: str = ""
    # Model name passed in the multipart form (e.g. "whisper-1",
    # "whisper-large-v3", "Systran/faster-whisper-large-v3").
    openai_model: str = "whisper-1"
    # Inline API key. Empty by default — prefer openai_api_key_env / .env.
    openai_api_key: str = ""
    # Process env var to read the key from when openai_api_key is empty.
    # Falls through to ``<config_dir>/.env`` (loaded once into os.environ).
    openai_api_key_env: str = "OPENAI_API_KEY"
    # Optional ISO-639-1 language hint sent to the API (empty = auto).
    openai_language: str = ""
    # HTTP timeout (seconds) for the transcription request.
    openai_timeout: float = 60.0
    # Retry count + delay for transient HTTP errors (408, 429, 5xx, …).
    openai_retries: int = 3
    openai_retry_delay: float = 1.0

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
    visualizer_sensitivity: float = 2.5
    # Background opacity of the overlay pill (0.0 = fully transparent,
    # 1.0 = fully opaque).
    opacity: float = 0.7
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
    # Bash script executed on every LLM request; stdout is prepended to the
    # system prompt as a dynamic state block when non-empty.
    dynamic_context_script: str = field(
        default_factory=lambda: str(_lazy_config_dir() / "dynamic-context.sh")
    )
    # How long (minutes) the continue window stays open after being armed.
    continue_window_minutes: int = 5


@dataclass
class WindowClipboardPolicy:
    """Auto-arm or block clipboard-context based on the focused window.

    When ``enabled`` is True, the focused-window's class / app-id is
    queried at the start of every recording. If the class is in
    ``block``, clipboard-context is forcibly disarmed for that
    recording. If it is in ``auto_arm``, clipboard-context is armed.
    Comparison is case-insensitive substring on the lowercased class.
    """

    enabled: bool = False
    auto_arm: list[str] = field(default_factory=list)
    block: list[str] = field(default_factory=list)


@dataclass
class PrefixRouterConfig:
    """Spoken-prefix routing.

    When enabled, the pipeline matches a leading ``word:`` (or ``word,``)
    against ``prefixes`` to switch the LLM profile for one recording, or
    against the special ``quick`` prefix (when ``quick_skip_llm=True``)
    to skip the LLM entirely. The prefix is stripped from the text
    before snippet matching and the LLM call.

    Example config snippet::

        [prefix_router]
        enabled = true
        quick_skip_llm = true
        [prefix_router.prefixes]
        code  = "code-cleanup"
        email = "email-polish"
    """

    enabled: bool = False
    # When True, the literal "quick" prefix routes around the LLM
    # (regardless of whether it appears in the prefixes mapping).
    quick_skip_llm: bool = True
    # Spoken-word → profile-name mapping. Keys are matched
    # case-insensitively.
    prefixes: dict[str, str] = field(default_factory=dict)


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
    prefix_router: PrefixRouterConfig = field(default_factory=PrefixRouterConfig)
    window_clipboard_policy: WindowClipboardPolicy = field(
        default_factory=WindowClipboardPolicy
    )
    # File path for user regex filters (applied after transcription, before LLM).
    filters_path: Path = field(default_factory=lambda: _lazy_config_dir() / "filters.json")
    # File path for post-LLM normalization filters (applied after LLM, before paste).
    after_llm_filters_path: Path = field(default_factory=lambda: _lazy_config_dir() / "after_LLM_filters.json")
    # File path for custom tool definitions (JSON array). Used by LLM backends
    # that support function calling. An empty array or missing file disables tools.
    tools_path: Path = field(default_factory=lambda: _lazy_config_dir() / "tools.json")


def _lazy_config_dir() -> Path:
    """Deferred call to config_dir() — avoids circular import at module level.
    Calls through the _io module so monkeypatching _io.config_dir is
    observed by these defaults (used in tests)."""
    from . import _io
    return _io.config_dir()


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
