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

    def push_hide(self) -> None:
        self.hide_calls += 1


def _make_seg(duration_s: float = 1.01) -> Segment:
    """Return a minimal Segment with the given duration."""
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="manual")


def _app(cfg: Config) -> App:
    app = App(cfg, no_overlay=True, no_paste=True)
    app.filters = []
    return app


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------


def test_handle_segment_prints_transcription(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hello world")
    app._handle_segment(_make_seg())
    assert capsys.readouterr().out.strip() == "hello world"


def test_handle_segment_empty_transcription_prints_nothing(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("")
    app._handle_segment(_make_seg())
    assert capsys.readouterr().out == ""


def test_handle_segment_updates_last_transcription_time():
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hi")
    assert app._last_transcription_time is None
    app._handle_segment(_make_seg())
    assert app._last_transcription_time is not None


def test_handle_segment_empty_does_not_update_last_time():
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("")
    app._handle_segment(_make_seg())
    assert app._last_transcription_time is None


def test_handle_segment_skips_short_segments_before_transcription(capsys):
    cfg = Config()
    cfg.audio.skip_segments_below_seconds = 1.0
    app = _app(cfg)
    app.overlay = _StubOverlay()
    app.transcriber = _CountingTranscriber("hello")

    app._handle_segment(_make_seg(duration_s=0.25))

    assert app.transcriber.calls == 0
    assert app.overlay.hide_calls == 1
    assert app._last_transcription_time is None
    assert capsys.readouterr().out == ""


def test_handle_segment_does_not_skip_when_threshold_disabled(capsys):
    cfg = Config()
    cfg.audio.skip_segments_below_seconds = 0.0
    app = _app(cfg)
    app.transcriber = _CountingTranscriber("hello")

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
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    assert out == "hello \n"


def test_trailing_space_off_by_default(capsys):
    cfg = Config()
    app = _app(cfg)
    app.transcriber = _StubTranscriber("hello")
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
    # Set last_transcription_time to just now → elapsed ≈ 0ms → within timeout.
    app._last_transcription_time = time.monotonic()
    app._handle_segment(_make_seg(duration_s=1.01))
    out = capsys.readouterr().out
    assert out == " world\n"


def test_auto_space_not_prepended_when_timeout_exceeded(capsys):
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 1000  # 1 s window
    app = _app(cfg)
    app.transcriber = _StubTranscriber("world")
    # Set last_transcription_time to 100 s ago → elapsed >> timeout.
    app._last_transcription_time = time.monotonic() - 100.0
    app._handle_segment(_make_seg(duration_s=1.01))
    out = capsys.readouterr().out
    assert out == "world\n"


def test_auto_space_not_prepended_on_first_transcription(capsys):
    """No previous transcription time → no auto-space, regardless of timeout."""
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 5000
    app = _app(cfg)
    app.transcriber = _StubTranscriber("first")
    assert app._last_transcription_time is None
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    assert out == "first\n"


def test_auto_space_disabled_when_zero(capsys):
    cfg = Config()
    cfg.paste.auto_space_timeout_ms = 0  # disabled
    app = _app(cfg)
    app.transcriber = _StubTranscriber("word")
    app._last_transcription_time = time.monotonic()
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
    app._last_transcription_time = time.monotonic()
    app._handle_segment(_make_seg())
    out = capsys.readouterr().out
    # Trailing space appended, no leading space.
    assert out == "word \n"
