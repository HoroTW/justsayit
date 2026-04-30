"""Tests for SegmentPipeline.redo_with_override (F5b feature)."""
from __future__ import annotations

import numpy as np
import pytest

from justsayit.audio import Segment
from justsayit.config import Config
from justsayit.pipeline import SegmentPipeline
from justsayit.postprocess import ProcessResult
from justsayit.transcribe import TranscriberBase


# ---------------------------------------------------------------------------
# Helpers (copied / adapted from test_handle_segment.py)
# ---------------------------------------------------------------------------


class _StubTranscriber(TranscriberBase):
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return self._text


class _RecordingPostprocessor:
    """Captures calls to process_with_reasoning for assertion."""

    def __init__(self, reply: str = "processed") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def process_with_reasoning(
        self,
        text: str,
        *,
        extra_context: str = "",
        extra_image=None,
        extra_image_mime: str = "",
        previous_session=None,
        tools=None,
        tool_caller=None,
        assistant_mode: bool = False,
        extra_system_prompt: str = "",
    ) -> ProcessResult:
        self.calls.append(
            {
                "text": text,
                "assistant_mode": assistant_mode,
                "extra_system_prompt": extra_system_prompt,
            }
        )
        return ProcessResult(text=self.reply)

    def strip_for_paste(self, text: str) -> str:
        return text

    def find_strip_matches(self, text: str) -> list[str]:
        return []


class _StubOverlay:
    def __init__(self) -> None:
        self.detected: list[tuple] = []
        self.llm: list[tuple] = []
        self.linger_calls = 0
        self.redo_buttons: list[tuple] = []

    def push_hide(self) -> None:
        pass

    def push_detected_text(self, text: str, llm_pending: bool = False) -> None:
        self.detected.append((text, llm_pending))

    def push_llm_text(self, text: str, thought: str = "") -> None:
        self.llm.append((text, thought))

    def push_linger_start(self) -> None:
        self.linger_calls += 1

    def push_error(self, stage: str, msg: str, retry_cb=None) -> None:
        pass

    def push_tool_call(self, name: str, params: dict) -> None:
        pass

    def push_redo_buttons(self, visible: bool, last_run_was_assistant: bool) -> None:
        self.redo_buttons.append((visible, last_run_was_assistant))


def _make_seg(duration_s: float = 1.01) -> Segment:
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="manual")


def _make_pipeline(
    transcribed: str = "hello world",
    *,
    no_paste: bool = True,
) -> SegmentPipeline:
    cfg = Config()
    pipe = SegmentPipeline(
        cfg,
        _StubTranscriber(transcribed),
        filters=[],
        paster=None,
        no_paste=no_paste,
    )
    return pipe


# ---------------------------------------------------------------------------
# 1. After handle(), _last_detected_text is set
# ---------------------------------------------------------------------------


def test_handle_sets_last_detected_text(capsys):
    pipe = _make_pipeline("hello world")
    pipe.handle(_make_seg())
    assert pipe._last_detected_text == "hello world"


def test_handle_sets_last_detected_text_post_filter(capsys):
    """Filters run before the cache is set — the cached text is post-filter."""
    from justsayit.filters import Filter

    cfg = Config()
    # Filter that uppercases everything.
    class _UpperFilter:
        def apply(self, text: str) -> str:
            return text.upper()

    pipe = SegmentPipeline(
        cfg,
        _StubTranscriber("hello"),
        filters=[_UpperFilter()],
        paster=None,
        no_paste=True,
    )
    pipe.handle(_make_seg())
    assert pipe._last_detected_text == "HELLO"


# ---------------------------------------------------------------------------
# 2. redo_with_override calls postprocessor with correct args
# ---------------------------------------------------------------------------


def test_redo_calls_postprocessor_with_assistant_true(capsys):
    pipe = _make_pipeline("some text")
    pp = _RecordingPostprocessor("assistant reply")
    pipe.postprocessor = pp
    pipe.handle(_make_seg())
    # Clear call log from handle().
    pp.calls.clear()

    pipe.redo_with_override(assistant_mode_override=True)

    assert len(pp.calls) == 1
    call = pp.calls[0]
    assert call["text"] == "some text"
    assert call["assistant_mode"] is True


def test_redo_calls_postprocessor_with_assistant_false(capsys):
    pipe = _make_pipeline("some text")
    pp = _RecordingPostprocessor("cleaned text")
    pipe.postprocessor = pp
    pipe.assistant_mode = True  # simulate prior run was assistant mode
    pipe.handle(_make_seg())
    pp.calls.clear()

    pipe.redo_with_override(assistant_mode_override=False)

    assert len(pp.calls) == 1
    assert pp.calls[0]["assistant_mode"] is False


def test_redo_result_is_pushed_to_overlay(capsys):
    pipe = _make_pipeline("some text")
    overlay = _StubOverlay()
    pipe.overlay = overlay
    pp = _RecordingPostprocessor("redo result")
    pipe.postprocessor = pp
    pipe.handle(_make_seg())

    pipe.redo_with_override(assistant_mode_override=False)

    # Last llm push should be "redo result".
    assert overlay.llm[-1][0] == "redo result"


# ---------------------------------------------------------------------------
# 3. redo_with_override with no cached text is a no-op (logs warning)
# ---------------------------------------------------------------------------


def test_redo_noop_when_no_cached_text(caplog):
    import logging

    pipe = _make_pipeline()
    pp = _RecordingPostprocessor()
    pipe.postprocessor = pp
    assert pipe._last_detected_text is None

    with caplog.at_level(logging.WARNING, logger="justsayit.pipeline"):
        pipe.redo_with_override(assistant_mode_override=True)

    assert pp.calls == []
    assert any("no cached" in r.message for r in caplog.records)


def test_redo_noop_when_no_postprocessor(caplog):
    import logging

    pipe = _make_pipeline()
    pipe._last_detected_text = "something"

    with caplog.at_level(logging.WARNING, logger="justsayit.pipeline"):
        pipe.redo_with_override(assistant_mode_override=False)

    assert any("no postprocessor" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. REDO nudge ends up in extra_system_prompt seen by postprocessor
# ---------------------------------------------------------------------------


def test_redo_cleanup_nudge_in_extra_system_prompt(capsys):
    pipe = _make_pipeline("text to redo")
    pp = _RecordingPostprocessor("cleaned")
    pipe.postprocessor = pp
    pipe.handle(_make_seg())
    pp.calls.clear()

    pipe.redo_with_override(assistant_mode_override=False)

    assert len(pp.calls) == 1
    nudge = pp.calls[0]["extra_system_prompt"]
    assert "REDO" in nudge
    assert "cleanup" in nudge.lower()
    assert "DO NOT" in nudge


def test_redo_assistant_nudge_in_extra_system_prompt(capsys):
    pipe = _make_pipeline("text to redo")
    pp = _RecordingPostprocessor("assistant reply")
    pipe.postprocessor = pp
    pipe.handle(_make_seg())
    pp.calls.clear()

    pipe.redo_with_override(assistant_mode_override=True)

    assert len(pp.calls) == 1
    nudge = pp.calls[0]["extra_system_prompt"]
    assert "REDO" in nudge
    assert "assistant" in nudge.lower()
    assert "DO NOT" in nudge


# ---------------------------------------------------------------------------
# 5. last_was_assistant_mode is updated by redo
# ---------------------------------------------------------------------------


def test_redo_updates_last_was_assistant_mode(capsys):
    pipe = _make_pipeline("text")
    pp = _RecordingPostprocessor()
    pipe.postprocessor = pp
    pipe.handle(_make_seg())
    assert pipe.last_was_assistant_mode is False

    pipe.redo_with_override(assistant_mode_override=True)
    assert pipe.last_was_assistant_mode is True

    pipe.redo_with_override(assistant_mode_override=False)
    assert pipe.last_was_assistant_mode is False
