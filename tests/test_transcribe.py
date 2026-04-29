"""Tests for the transcription factory and base class."""

from __future__ import annotations

import numpy as np
import pytest

from justsayit.config import Config
from justsayit.transcribe import TranscriberBase, make_transcriber
from justsayit.transcribe_parakeet import ParakeetTranscriber
from justsayit.transcribe_whisper import WhisperTranscriber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTranscriber(TranscriberBase):
    """Minimal concrete subclass that returns a fixed string."""

    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return self._text


# ---------------------------------------------------------------------------
# make_transcriber factory
# ---------------------------------------------------------------------------


def test_make_transcriber_parakeet_returns_correct_type():
    cfg = Config()
    cfg.model.backend = "parakeet"
    t = make_transcriber(cfg)
    assert isinstance(t, ParakeetTranscriber)


def test_make_transcriber_whisper_returns_correct_type():
    cfg = Config()
    cfg.model.backend = "whisper"
    t = make_transcriber(cfg)
    assert isinstance(t, WhisperTranscriber)


def test_make_transcriber_whisper_does_not_load_model():
    """WhisperTranscriber is lazy; construction alone must not import faster-whisper."""
    cfg = Config()
    cfg.model.backend = "whisper"
    t = make_transcriber(cfg)
    assert t._model is None  # model not loaded yet


def test_make_transcriber_unknown_backend_raises():
    cfg = Config()
    cfg.model.backend = "unknown-backend"
    with pytest.raises(ValueError, match="unknown transcription backend"):
        make_transcriber(cfg)


def test_make_transcriber_parakeet_resolves_paths_internally():
    """ParakeetTranscriber must resolve its own ModelPaths from cfg —
    callers shouldn't have to pre-compute them."""
    cfg = Config()
    cfg.model.backend = "parakeet"
    t = make_transcriber(cfg)
    assert isinstance(t, ParakeetTranscriber)
    assert t.paths.encoder.name == cfg.model.parakeet_encoder


def test_make_transcriber_parakeet_does_not_load_model():
    """ParakeetTranscriber is lazy; construction alone must not import sherpa-onnx."""
    cfg = Config()
    t = make_transcriber(cfg)
    assert t._recog is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TranscriberBase interface
# ---------------------------------------------------------------------------


def test_transcriber_base_warmup_is_noop():
    """Default warmup() must not raise."""
    t = _StubTranscriber("hello")
    t.warmup()  # should not raise


def test_transcriber_base_has_words_empty_samples():
    t = _StubTranscriber("anything")
    assert t.has_words(np.array([], dtype=np.float32), 16000) is False


def test_transcriber_base_has_words_with_real_word():
    t = _StubTranscriber("hello world")
    assert t.has_words(np.zeros(160, dtype=np.float32), 16000) is True


def test_transcriber_base_has_words_no_words():
    # Single-char tokens don't count as words (_WORD_RE requires \w{2,})
    t = _StubTranscriber("a b c")
    assert t.has_words(np.zeros(160, dtype=np.float32), 16000) is False


def test_transcriber_base_has_words_empty_text():
    t = _StubTranscriber("")
    assert t.has_words(np.zeros(160, dtype=np.float32), 16000) is False


def test_transcriber_base_has_words_minimum_two_char_word():
    t = _StubTranscriber("ok")
    assert t.has_words(np.zeros(160, dtype=np.float32), 16000) is True
