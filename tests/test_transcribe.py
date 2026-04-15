"""Tests for the transcription factory and base class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from justsayit.config import Config
from justsayit.model import ModelPaths
from justsayit.transcribe import TranscriberBase, make_transcriber
from justsayit.transcribe_parakeet import ParakeetTranscriber
from justsayit.transcribe_whisper import WhisperTranscriber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_paths() -> ModelPaths:
    """Return a ModelPaths pointing at non-existent files (constructor only)."""
    return ModelPaths(
        encoder=Path("/fake/encoder.onnx"),
        decoder=Path("/fake/decoder.onnx"),
        joiner=Path("/fake/joiner.onnx"),
        tokens=Path("/fake/tokens.txt"),
        vad=Path("/fake/silero_vad.onnx"),
    )


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
    t = make_transcriber(cfg, _fake_paths())
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


def test_make_transcriber_parakeet_requires_model_paths():
    cfg = Config()
    cfg.model.backend = "parakeet"
    with pytest.raises(ValueError, match="model_paths"):
        make_transcriber(cfg, model_paths=None)


def test_make_transcriber_parakeet_does_not_load_model():
    """ParakeetTranscriber is lazy; construction alone must not import sherpa-onnx."""
    cfg = Config()
    t = make_transcriber(cfg, _fake_paths())
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
