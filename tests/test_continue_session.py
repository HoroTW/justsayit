"""Tests for the continue-session feature (session.json + multi-turn history)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from justsayit.audio import Segment
from justsayit.config import Config
from justsayit.pipeline import (
    SegmentPipeline,
    _clear_session,
    _load_session,
    _save_session,
    _session_path,
)
from justsayit.postprocess._processor import PostprocessorBase
from justsayit.postprocess._profile import PostprocessProfile, ProcessResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seg(duration_s: float = 1.01) -> Segment:
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="manual")


class _StubTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples, sr) -> str:
        return self._text

    def has_words(self, samples, sr) -> bool:
        return True

    def warmup(self) -> None:
        pass


class _RecordingPostprocessor:
    """Captures arguments passed to process_with_reasoning; returns fixed result."""

    def __init__(self, text: str = "cleaned", session_data: dict | None = None) -> None:
        self._text = text
        self._session_data = session_data
        self.calls: list[dict] = []

    def process_with_reasoning(
        self,
        text: str,
        *,
        extra_context: str = "",
        extra_image=None,
        extra_image_mime: str = "",
        previous_session=None,
    ) -> ProcessResult:
        self.calls.append({"text": text, "previous_session": previous_session})
        return ProcessResult(text=self._text, session_data=self._session_data)

    def strip_for_paste(self, text: str) -> str:
        return text

    def find_strip_matches(self, text: str) -> list[str]:
        return []


def _make_pipeline(cfg: Config, text: str = "hello", pp=None) -> SegmentPipeline:
    pl = SegmentPipeline(cfg, _StubTranscriber(text), [], None, no_paste=True)
    pl.postprocessor = pp
    return pl


# ---------------------------------------------------------------------------
# Session file helpers
# ---------------------------------------------------------------------------


def test_save_and_load_session(tmp_path):
    fake_cache = tmp_path / "cache"
    with patch("justsayit.pipeline._session_path", return_value=fake_cache / "session.json"):
        data = {"backend": "remote", "prev_messages": [], "ts": 1.0}
        _save_session(data)
        loaded = _load_session()
    assert loaded == data


def test_load_session_returns_none_when_missing(tmp_path):
    with patch("justsayit.pipeline._session_path", return_value=tmp_path / "no_file.json"):
        assert _load_session() is None


def test_clear_session_deletes_file(tmp_path):
    p = tmp_path / "session.json"
    p.write_text("{}", encoding="utf-8")
    with patch("justsayit.pipeline._session_path", return_value=p):
        _clear_session()
    assert not p.exists()


def test_clear_session_tolerates_missing(tmp_path):
    with patch("justsayit.pipeline._session_path", return_value=tmp_path / "no.json"):
        _clear_session()  # must not raise


# ---------------------------------------------------------------------------
# Pipeline: session save / clear
# ---------------------------------------------------------------------------


def test_pipeline_saves_session_when_is_continue(tmp_path):
    cfg = Config()
    session_data = {"backend": "remote", "prev_messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], "ts": 1.0}
    pp = _RecordingPostprocessor(session_data=session_data)
    pl = _make_pipeline(cfg, "hello", pp)

    saved = {}
    with patch("justsayit.pipeline._load_session", return_value=None), \
         patch("justsayit.pipeline._save_session", side_effect=lambda d: saved.update(d)), \
         patch("justsayit.pipeline._clear_session") as mock_clear:
        pl.handle(_make_seg(), is_continue=True)

    assert saved["backend"] == "remote"
    mock_clear.assert_not_called()


def test_pipeline_clears_session_when_not_continue(tmp_path):
    cfg = Config()
    session_data = {"backend": "remote", "prev_messages": [], "ts": 1.0}
    pp = _RecordingPostprocessor(session_data=session_data)
    pl = _make_pipeline(cfg, "hello", pp)

    with patch("justsayit.pipeline._load_session", return_value=None), \
         patch("justsayit.pipeline._save_session") as mock_save, \
         patch("justsayit.pipeline._clear_session") as mock_clear:
        pl.handle(_make_seg(), is_continue=False)

    mock_save.assert_not_called()
    mock_clear.assert_called_once()


def test_pipeline_passes_previous_session_to_postprocessor():
    cfg = Config()
    prev_session = {"backend": "remote", "prev_messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "ts": 1.0}
    pp = _RecordingPostprocessor()
    pl = _make_pipeline(cfg, "follow up", pp)

    with patch("justsayit.pipeline._load_session", return_value=prev_session), \
         patch("justsayit.pipeline._save_session"), \
         patch("justsayit.pipeline._clear_session"):
        pl.handle(_make_seg(), is_continue=True)

    assert len(pp.calls) == 1
    assert pp.calls[0]["previous_session"] is prev_session


def test_pipeline_passes_none_when_not_continue():
    cfg = Config()
    pp = _RecordingPostprocessor()
    pl = _make_pipeline(cfg, "hello", pp)

    with patch("justsayit.pipeline._load_session") as mock_load, \
         patch("justsayit.pipeline._clear_session"):
        pl.handle(_make_seg(), is_continue=False)

    mock_load.assert_not_called()
    assert pp.calls[0]["previous_session"] is None


def test_pipeline_clears_session_on_llm_exception():
    cfg = Config()

    class _RaisingPP:
        def process_with_reasoning(self, text, *, extra_context="", extra_image=None, extra_image_mime="", previous_session=None):
            raise RuntimeError("boom")

        def strip_for_paste(self, text):
            return text

        def find_strip_matches(self, text):
            return []

    pl = _make_pipeline(cfg, "hello", _RaisingPP())

    with patch("justsayit.pipeline._load_session", return_value=None), \
         patch("justsayit.pipeline._clear_session") as mock_clear:
        pl.handle(_make_seg(), is_continue=False)

    mock_clear.assert_called_once()


def test_pipeline_does_not_clear_session_on_llm_exception_when_continue():
    cfg = Config()

    class _RaisingPP:
        def process_with_reasoning(self, text, *, extra_context="", extra_image=None, extra_image_mime="", previous_session=None):
            raise RuntimeError("boom")

        def strip_for_paste(self, text):
            return text

        def find_strip_matches(self, text):
            return []

    pl = _make_pipeline(cfg, "hello", _RaisingPP())

    with patch("justsayit.pipeline._load_session", return_value={"backend": "remote"}), \
         patch("justsayit.pipeline._clear_session") as mock_clear:
        pl.handle(_make_seg(), is_continue=True)

    mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# PostprocessorBase: _build_messages_continued + _format_history_text
# ---------------------------------------------------------------------------


def _make_profile(**kw) -> PostprocessProfile:
    defaults = {
        "system_prompt": "You are a helpful assistant.",
        "system_prompt_file": "",
    }
    defaults.update(kw)
    return PostprocessProfile(**defaults)


class _ConcreteProcessor(PostprocessorBase):
    def _run(self, text, extra_context="", extra_image=None, extra_image_mime="", previous_session=None):
        return ProcessResult(text=text)


def test_format_history_text_basic():
    proc = _ConcreteProcessor(_make_profile())
    msgs = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And Germany?"},
        {"role": "assistant", "content": "Berlin."},
    ]
    result = proc._format_history_text(msgs)
    assert result.startswith("## PREVIOUS SESSION HISTORY")
    assert "User: What is the capital of France?" in result
    assert "Assistant: Paris." in result
    assert "User: And Germany?" in result
    assert "Assistant: Berlin." in result


def test_build_messages_continued_order():
    proc = _ConcreteProcessor(_make_profile())
    prev = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    msgs = proc._build_messages_continued("second", "", prev)
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "first"}
    assert msgs[2] == {"role": "assistant", "content": "answer"}
    assert msgs[3]["role"] == "user"
    assert "second" in msgs[3]["content"]


def test_build_messages_with_history_text_goes_into_system_prompt():
    proc = _ConcreteProcessor(_make_profile())
    prev = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    history = proc._format_history_text(prev)
    msgs = proc._build_messages("new question", history_text=history)
    system_content = msgs[0]["content"]
    assert "## PREVIOUS SESSION HISTORY" in system_content
    assert "User: q" in system_content
    assert "Assistant: a" in system_content
