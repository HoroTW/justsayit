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
from justsayit.pipeline import SegmentPipeline, _lowercase_preview_start
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
        self.partials: list[str] = []
        self.word_previews: list[tuple[str, str, str]] = []
        self.chunk_previews: list[tuple[str, str]] = []
        self.finalized_chunks: list[str] = []
        self.llm = []
        self.linger_calls = 0
        self.clip_armed_calls: list[bool] = []
        self.errors: list[tuple[str, str, object]] = []

    def push_hide(self) -> None:
        self.hide_calls += 1

    def push_detected_text(self, text: str, llm_pending: bool = False) -> None:
        self.detected.append((text, llm_pending))

    def push_partial_text(self, text: str) -> None:
        self.partials.append(text)

    def push_word_preview_text(
        self,
        finalized_text: str,
        chunk_preview_text: str,
        word_preview_text: str,
    ) -> None:
        self.word_previews.append((finalized_text, chunk_preview_text, word_preview_text))

    def push_chunk_preview_text(self, committed_text: str, preview_text: str) -> None:
        self.chunk_previews.append((committed_text, preview_text))

    def push_finalized_chunk_text(self, text: str) -> None:
        self.finalized_chunks.append(text)

    def push_llm_text(self, text: str, thought: str = "") -> None:
        self.llm.append((text, thought))

    def push_linger_start(self) -> None:
        self.linger_calls += 1

    def push_clipboard_context_armed(self, armed: bool) -> None:
        self.clip_armed_calls.append(armed)

    def push_error(self, stage: str, msg: str, retry_cb=None) -> None:
        self.errors.append((stage, msg, retry_cb))

    def push_tool_call(self, name: str, params: dict) -> None:
        pass

    def push_redo_buttons(self, visible: bool, last_run_was_assistant: bool) -> None:
        pass


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
    monkeypatch.setattr("justsayit.cli.read_clipboard_image", lambda: None)
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
    monkeypatch.setattr("justsayit.cli.read_clipboard_image", lambda: None)

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
    monkeypatch.setattr("justsayit.cli.read_clipboard_image", lambda: None)
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


# ---------------------------------------------------------------------------
# Pipeline: error surface + undo/re-paste
# ---------------------------------------------------------------------------


class _RaisingTranscriber(TranscriberBase):
    def __init__(self, message: str = "asr exploded") -> None:
        self.message = message
        self.calls = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        self.calls += 1
        raise RuntimeError(self.message)


def test_pipeline_transcribe_failure_emits_on_error():
    """A direct SegmentPipeline test — when the transcriber raises, on_error
    fires with the stage tag and the retry callback re-enqueues the segment."""
    from justsayit.pipeline import SegmentPipeline

    cfg = Config()
    captured: list[tuple[str, str, object]] = []
    enqueued: list = []

    pipe = SegmentPipeline(
        cfg,
        _RaisingTranscriber("network down"),
        filters=[],
        paster=None,
        no_paste=True,
        on_error=lambda stage, msg, cb: captured.append((stage, msg, cb)),
        enqueue_segment=lambda s: enqueued.append(s),
    )

    seg = _make_seg()
    pipe.handle(seg)

    assert len(captured) == 1
    stage, msg, cb = captured[0]
    assert stage == "transcribe"
    assert "network down" in msg
    assert cb is not None
    cb()
    assert enqueued == [seg]


def test_pipeline_llm_failure_also_emits_on_error(capsys):
    """LLM exception still falls back to original text but additionally fires on_error."""
    from justsayit.pipeline import SegmentPipeline

    cfg = Config()
    captured: list[tuple[str, str, object]] = []

    pipe = SegmentPipeline(
        cfg,
        _StubTranscriber("hello world"),
        filters=[],
        paster=None,
        no_paste=True,
        on_error=lambda stage, msg, cb: captured.append((stage, msg, cb)),
    )
    pipe.postprocessor = _RaisingPostprocessor("HTTP 503: upstream timeout")

    pipe.handle(_make_seg())

    # Original text still printed (fallback preserved).
    assert capsys.readouterr().out == "hello world\n"
    # And the error pill was fired.
    assert [(s, m) for s, m, _cb in captured] == [
        ("llm", "HTTP 503: upstream timeout")
    ]


# ---------------------------------------------------------------------------
# Streaming partial previews
# ---------------------------------------------------------------------------


class _SequenceTranscriber(TranscriberBase):
    """Returns successive strings from a list on each transcribe() call."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self._idx = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._idx < len(self._texts):
            text = self._texts[self._idx]
            self._idx += 1
            return text
        return ""


class _RecordingSequenceTranscriber(_SequenceTranscriber):
    def __init__(self, texts: list[str]) -> None:
        super().__init__(texts)
        self.durations: list[float] = []

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        self.durations.append(len(samples) / float(sample_rate))
        return super().transcribe(samples, sample_rate)


class _DurationLabelTranscriber(TranscriberBase):
    def __init__(self) -> None:
        self.durations: list[float] = []
        self.finalized_count = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        duration = len(samples) / float(sample_rate)
        self.durations.append(duration)
        if duration <= 4.0:
            return "rough preview"
        if 20.0 <= duration <= 30.0:
            self.finalized_count += 1
            return f"finalized {self.finalized_count}"
        return "tail"


def _make_partial_seg(duration_s: float = 22.5) -> Segment:
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="stream-chunk", is_final=False)


def _make_pipeline(texts: list[str]) -> "SegmentPipeline":
    from justsayit.pipeline import SegmentPipeline
    cfg = Config()
    return SegmentPipeline(
        cfg,
        _SequenceTranscriber(texts),
        filters=[],
        paster=None,
        no_paste=True,
    )


def _make_pipeline_with_transcriber(transcriber: TranscriberBase) -> "SegmentPipeline":
    from justsayit.pipeline import SegmentPipeline
    cfg = Config()
    return SegmentPipeline(
        cfg,
        transcriber,
        filters=[],
        paster=None,
        no_paste=True,
    )


def test_streaming_partials_are_replaced_by_final_transcription(capsys):
    """Preview chunks update the UI, but final paste uses the final ASR pass."""
    pipe = _make_pipeline(["rough first", "rough second", "final corrected"])

    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_seg(), is_continue=False)  # is_final=True by default

    out = capsys.readouterr().out.strip()
    assert out == "final corrected"
    assert pipe._partial_raws == []


def test_mid_sentence_preview_artifacts_do_not_reach_final_text(capsys):
    """Partial ASR can invent sentence boundaries; final text must replace it."""
    pipe = _make_pipeline([
        "Please send this message.",
        "To the home channel.",
        "Please send this message to the home channel.",
    ])

    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_seg(), is_continue=False)

    out = capsys.readouterr().out.strip()
    assert out == "Please send this message to the home channel."
    assert "message. To" not in out


def test_preview_text_starts_lowercase_when_it_looks_like_a_word():
    assert _lowercase_preview_start("Weil ja") == "weil ja"
    assert _lowercase_preview_start("Ärgerlich") == "ärgerlich"
    assert _lowercase_preview_start("PDF export") == "PDF export"
    assert _lowercase_preview_start("I/O") == "I/O"


def test_streaming_word_and_chunk_previews_start_lowercase():
    pipe = _make_pipeline([
        "Weil ja",
        "Maybe this is better",
        "Final output",
    ])
    overlay = _StubOverlay()
    pipe.overlay = overlay

    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_partial_seg(5.0), is_continue=False)

    assert overlay.word_previews[0][2] == "weil ja"
    assert overlay.chunk_previews[0][1] == "maybe this is better"


def test_preview_start_lowercase_keeps_acronyms_uppercase():
    pipe = _make_pipeline(["PDF", "PDF export preview", "Final output"])
    overlay = _StubOverlay()
    pipe.overlay = overlay

    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_partial_seg(5.0), is_continue=False)

    assert overlay.word_previews[0][2] == "PDF"
    assert overlay.chunk_previews[0][1] == "PDF export preview"


def test_streaming_word_preview_is_replaced_by_quality_chunk_preview():
    transcriber = _DurationLabelTranscriber()
    pipe = _make_pipeline_with_transcriber(transcriber)
    overlay = _StubOverlay()
    pipe.overlay = overlay

    for _ in range(11):
        pipe.handle(_make_partial_seg(3.0), is_continue=False)

    assert overlay.word_previews
    assert overlay.chunk_previews
    assert overlay.finalized_chunks[-1].startswith("finalized")


def test_streaming_finalization_waits_for_stable_tail_context():
    transcriber = _DurationLabelTranscriber()
    pipe = _make_pipeline_with_transcriber(transcriber)
    overlay = _StubOverlay()
    pipe.overlay = overlay

    for _ in range(10):
        pipe.handle(_make_partial_seg(3.0), is_continue=False)

    assert not overlay.finalized_chunks

    pipe.handle(_make_partial_seg(3.0), is_continue=False)

    assert overlay.finalized_chunks[-1].startswith("finalized")


def test_word_preview_keeps_rolling_chunk_preview_as_committed_text():
    transcriber = _DurationLabelTranscriber()
    pipe = _make_pipeline_with_transcriber(transcriber)
    overlay = _StubOverlay()
    pipe.overlay = overlay

    for _ in range(7):
        pipe.handle(_make_partial_seg(3.0), is_continue=False)
    assert overlay.chunk_previews[-1][1] == "finalized 1"

    pipe.handle(_make_partial_seg(3.0), is_continue=False)

    assert overlay.word_previews[-1][0] == ""
    assert overlay.word_previews[-1][1] == "finalized 1"
    assert overlay.word_previews[-1][2] == "rough preview"


def test_streaming_final_stop_transcribes_only_tail_after_finalized_chunks(capsys):
    transcriber = _DurationLabelTranscriber()
    pipe = _make_pipeline_with_transcriber(transcriber)

    for _ in range(11):
        pipe.handle(_make_partial_seg(3.0), is_continue=False)
    pipe.handle(_make_seg(duration_s=2.0), is_continue=False)

    out = capsys.readouterr().out.strip()
    assert out == "finalized 1 tail"
    assert len(transcriber.durations) == 13
    assert sum(1 for d in transcriber.durations if d == pytest.approx(3.0)) == 11
    assert sum(1 for d in transcriber.durations if 20.0 <= d <= 30.0) == 1
    assert transcriber.durations[-1] <= 10.0


def test_streaming_finalizer_drains_multiple_ready_chunks_before_stop(capsys):
    transcriber = _DurationLabelTranscriber()
    pipe = _make_pipeline_with_transcriber(transcriber)

    pipe.handle(_make_partial_seg(60.0), is_continue=False)
    pipe.handle(_make_seg(duration_s=2.0), is_continue=False)

    out = capsys.readouterr().out.strip()
    assert out == "finalized 1 finalized 2 tail"
    assert len(transcriber.durations) == 4
    assert transcriber.durations[0] == pytest.approx(60.0)
    assert 20.0 <= transcriber.durations[1] <= 30.0
    assert 20.0 <= transcriber.durations[2] <= 30.0
    assert transcriber.durations[3] <= 12.0


def test_clear_partials_prevents_leaking_into_next_final(capsys):
    """After clear_partials(), the next final segment gets only its own text."""
    pipe = _make_pipeline(["leaked1", "leaked2", "alone"])

    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.handle(_make_partial_seg(5.0), is_continue=False)
    pipe.clear_partials()
    pipe.handle(_make_seg(), is_continue=False)

    out = capsys.readouterr().out.strip()
    assert out == "alone"
    assert pipe._partial_raws == []
