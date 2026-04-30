"""Tests for App._handle_segment — space logic and output formatting.

Imports ``justsayit.cli`` which loads GTK bindings at module level.
The ``conftest.py`` env flags prevent the module from re-exec'ing the
test process under a systemd scope or with LD_PRELOAD set.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from justsayit.audio import Segment
from justsayit.config import Config
from justsayit.pipeline import SegmentPipeline
from justsayit.postprocess import ProcessResult
from justsayit.transcribe import TranscriberBase

# Import App after conftest.py has set the env guards.
from justsayit.cli import App


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTranscriber(TranscriberBase):
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return self._text


class _CountingTranscriber(TranscriberBase):
    def __init__(self, text: str = "ignored") -> None:
        self._text = text
        self.calls = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        self.calls += 1
        return self._text


class _StubOverlay:
    def __init__(self) -> None:
        self.hide_calls = 0
        self.detected = []
        self.llm = []
        self.linger_calls = 0
        self.clip_armed_calls: list[bool] = []

    def push_hide(self) -> None:
        self.hide_calls += 1

    def push_detected_text(self, text: str, llm_pending: bool = False) -> None:
        self.detected.append((text, llm_pending))

    def push_llm_text(self, text: str, thought: str = "") -> None:
        self.llm.append((text, thought))

    def push_linger_start(self) -> None:
        self.linger_calls += 1

    def push_clipboard_context_armed(self, armed: bool) -> None:
        self.clip_armed_calls.append(armed)


class _RaisingPostprocessor:
    def __init__(self, message: str = "remote exploded") -> None:
        self.message = message
        self.strip_calls = 0

    def process_with_reasoning(
        self, text: str, *, extra_context: str = "", extra_image=None, extra_image_mime: str = "", previous_session=None, tools=None, tool_caller=None, assistant_mode: bool = False
    ) -> ProcessResult:
        raise RuntimeError(self.message)

    def strip_for_paste(self, text: str) -> str:
        self.strip_calls += 1
        return text.replace("secret", "")

    def find_strip_matches(self, text: str) -> list[str]:
        return []


class _StubPostprocessor:
    """Returns a fixed ProcessResult — lets tests exercise reasoning wiring."""

    def __init__(self, text: str, reasoning: str = "") -> None:
        self._text = text
        self._reasoning = reasoning
        self.calls = 0
        self.extra_contexts: list[str] = []

    def process_with_reasoning(
        self, text: str, *, extra_context: str = "", extra_image=None, extra_image_mime: str = "", previous_session=None, tools=None, tool_caller=None, assistant_mode: bool = False
    ) -> ProcessResult:
        self.calls += 1
        self.extra_contexts.append(extra_context)
        return ProcessResult(text=self._text, reasoning=self._reasoning)

    def strip_for_paste(self, text: str) -> str:
        return text

    def find_strip_matches(self, text: str) -> list[str]:
        return []


def _make_seg(duration_s: float = 1.01) -> Segment:
    """Return a minimal Segment with the given duration."""
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="manual")


def _app(cfg: Config) -> App:
    app = App(cfg, no_overlay=True, no_paste=True)
    app.filters = []
    return app


def _wire_pipeline(app: App) -> None:
    """Build a SegmentPipeline from the current App state. Tests must call
    this after setting ``app.transcriber`` (and any optional postprocessor /
    overlay) — the production path goes through ``setup_transcriber`` which
    builds the same wiring."""
    assert app.transcriber is not None
    app.pipeline = SegmentPipeline(
        app.cfg,
        app.transcriber,
        app.filters,
        app.paster,
        no_paste=app.no_paste,
        after_llm_filters=app.after_llm_filters,
    )
    app.pipeline.postprocessor = app.postprocessor
    app.pipeline.overlay = app.overlay


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------


def test_handle_segment_prints_transcription(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hello world")
    _wire_pipeline(app)
    app._handle_segment(_make_seg())
    assert capsys.readouterr().out.strip() == "hello world"


def test_handle_segment_empty_transcription_prints_nothing(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("")
    _wire_pipeline(app)
    app._handle_segment(_make_seg())
    assert capsys.readouterr().out == ""


def test_handle_segment_updates_last_transcription_time():
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hi")
    _wire_pipeline(app)
    assert app.pipeline._last_transcription_time is None
    app._handle_segment(_make_seg())
    assert app.pipeline._last_transcription_time is not None


def test_handle_segment_empty_does_not_update_last_time():
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("")
    _wire_pipeline(app)
    app._handle_segment(_make_seg())
    assert app.pipeline._last_transcription_time is None


def test_handle_segment_skips_short_segments_before_transcription(capsys):
    cfg = Config()
    cfg.audio.skip_segments_below_seconds = 1.0
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _CountingTranscriber("hello")
    _wire_pipeline(app)

    app._handle_segment(_make_seg(duration_s=0.25))

    assert app.transcriber.calls == 0
    assert app.overlay.hide_calls == 1
    assert app.pipeline._last_transcription_time is None
    assert capsys.readouterr().out == ""


def test_handle_segment_does_not_skip_when_threshold_disabled(capsys):
    cfg = Config()
    cfg.audio.skip_segments_below_seconds = 0.0
    app = _app(cfg)
    app.transcriber = _CountingTranscriber("hello")
    _wire_pipeline(app)

    app._handle_segment(_make_seg(duration_s=0.25))

    assert app.transcriber.calls == 1
    assert capsys.readouterr().out == "hello\n"


# ---------------------------------------------------------------------------
# append_trailing_space
# ---------------------------------------------------------------------------


def test_trailing_space_appended(capsys):
    cfg = Config()
    cfg.paste.append_trailing_space = True
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hello")
    _wire_pipeline(app)
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    assert out == "hello \n"


def test_trailing_space_off_by_default(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hello")
    _wire_pipeline(app)
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    assert out == "hello\n"


# ---------------------------------------------------------------------------
# auto_space_timeout_ms
# ---------------------------------------------------------------------------


def test_auto_space_prepended_within_timeout(capsys):
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 5000  # 5 s window
    app = _app(cfg)
    app.transcriber = _StubTranscriber("world")
    _wire_pipeline(app)
    # Set last_transcription_time to just now → elapsed ≈ 0ms → within timeout.
    app.pipeline._last_transcription_time = time.monotonic()
    app._handle_segment(_make_seg(duration_s=1.01))
    out = capsys.readouterr().out
    assert out == " world\n"


def test_auto_space_not_prepended_when_timeout_exceeded(capsys):
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 1000  # 1 s window
    app = _app(cfg)
    app.transcriber = _StubTranscriber("world")
    _wire_pipeline(app)
    # Set last_transcription_time to 100 s ago → elapsed >> timeout.
    app.pipeline._last_transcription_time = time.monotonic() - 100.0
    app._handle_segment(_make_seg(duration_s=1.01))
    out = capsys.readouterr().out
    assert out == "world\n"


def test_auto_space_not_prepended_on_first_transcription(capsys):
    """No previous transcription time → no auto-space, regardless of timeout."""
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 5000
    app = _app(cfg)
    app.transcriber = _StubTranscriber("first")
    _wire_pipeline(app)
    assert app.pipeline._last_transcription_time is None
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    assert out == "first\n"


def test_auto_space_disabled_when_zero(capsys):
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 0  # disabled
    app = _app(cfg)
    app.transcriber = _StubTranscriber("word")
    _wire_pipeline(app)
    app.pipeline._last_transcription_time = time.monotonic()
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    assert out == "word\n"


# ---------------------------------------------------------------------------
# Interaction: both flags set → trailing_space wins
# ---------------------------------------------------------------------------


def test_trailing_space_takes_precedence_over_auto_space(capsys):
    """When append_trailing_space is True, auto_space_timeout_ms is ignored
    (the trailing space already acts as separator)."""
    cfg = Config()
    cfg.paste.append_trailing_space = True
    cfg.paste.auto_space_timeout_ms = 5000
    app = _app(cfg)
    app.transcriber = _StubTranscriber("word")
    _wire_pipeline(app)
    app.pipeline._last_transcription_time = time.monotonic()
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    # Trailing space appended, no leading space.
    assert out == "word \n"


def test_llm_failure_shows_overlay_error_but_prints_original_text(capsys):
    cfg = Config()
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _StubTranscriber("hello world")
    app.postprocessor = _RaisingPostprocessor("HTTP 503: upstream timeout")
    _wire_pipeline(app)

    app._handle_segment(_make_seg())

    assert capsys.readouterr().out == "hello world\n"
    assert app.overlay.detected == [("hello world", True)]
    assert app.overlay.llm == [("LLM error: HTTP 503: upstream timeout", "")]


def test_llm_failure_does_not_strip_fallback_text(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("keep secret text")
    pp = _RaisingPostprocessor("boom")
    app.postprocessor = pp
    _wire_pipeline(app)

    app._handle_segment(_make_seg())

    assert capsys.readouterr().out == "keep secret text\n"
    assert pp.strip_calls == 0


def test_remote_reasoning_field_is_shown_in_overlay_thought(capsys):
    """Structured reasoning from remote backends (DeepSeek/vLLM/OpenRouter)
    should land in the overlay's `thought` slot — not the pasted body."""
    cfg = Config()
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _StubTranscriber("hello world")
    app.postprocessor = _StubPostprocessor(
        text="cleaned hello world",
        reasoning="model decided punctuation needed a comma",
    )
    _wire_pipeline(app)

    app._handle_segment(_make_seg())

    assert capsys.readouterr().out == "cleaned hello world\n"
    assert app.overlay.llm == [
        ("cleaned hello world", "model decided punctuation needed a comma")
    ]


def test_no_reasoning_field_leaves_overlay_thought_empty(capsys):
    cfg = Config()
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _StubTranscriber("hello")
    app.postprocessor = _StubPostprocessor(text="cleaned hello", reasoning="")
    _wire_pipeline(app)

    app._handle_segment(_make_seg())

    assert capsys.readouterr().out == "cleaned hello\n"
    assert app.overlay.llm == [("cleaned hello", "")]


def test_armed_clipboard_context_is_passed_to_postprocessor_and_disarms(
    capsys, monkeypatch
):
    cfg = Config()
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _StubTranscriber("hello")
    pp = _StubPostprocessor(text="cleaned hello")
    app.postprocessor = pp
    _wire_pipeline(app)
    monkeypatch.setattr(
        "justsayit.cli.read_clipboard",
        lambda **_kw: "extra context from clipboard",
    )
    app._clipboard_context_armed = True

    app._handle_segment(_make_seg())

    assert pp.extra_contexts == ["extra context from clipboard"]
    assert app._clipboard_context_armed is False
    # Overlay was told to disarm the visual.
    assert app.overlay.clip_armed_calls == [False]


def test_unarmed_clipboard_context_does_not_read_clipboard(capsys, monkeypatch):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hello")
    pp = _StubPostprocessor(text="cleaned hello")
    app.postprocessor = pp
    _wire_pipeline(app)
    calls = []

    def _boom():
        calls.append("called")
        return "should not be read"

    monkeypatch.setattr("justsayit.cli.read_clipboard", _boom)

    app._handle_segment(_make_seg())

    assert pp.extra_contexts == [""]
    assert calls == []


def test_armed_with_empty_clipboard_still_disarms(capsys, monkeypatch):
    cfg = Config()
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _StubTranscriber("hello")
    pp = _StubPostprocessor(text="cleaned hello")
    app.postprocessor = pp
    _wire_pipeline(app)
    monkeypatch.setattr("justsayit.cli.read_clipboard", lambda **_kw: None)
    app._clipboard_context_armed = True

    app._handle_segment(_make_seg())

    assert pp.extra_contexts == [""]
    assert app._clipboard_context_armed is False
    assert app.overlay.clip_armed_calls == [False]


def test_toggle_clipboard_context_flips_flag_and_pushes_overlay():
    cfg = Config()
    app = _app(cfg)
    app.overlay = _StubOverlay()

    assert app._clipboard_context_armed is False
    app._toggle_clipboard_context()
    assert app._clipboard_context_armed is True
    assert app.overlay.clip_armed_calls == [True]

    app._toggle_clipboard_context()
    assert app._clipboard_context_armed is False
    assert app.overlay.clip_armed_calls == [True, False]
