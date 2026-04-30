"""Tests for AudioEngine debug dump functionality."""

import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from justsayit.audio import AudioEngine
from justsayit.config import Config, AudioConfig


def _make_engine(tmp_path: Path) -> AudioEngine:
    cfg = Config(audio=AudioConfig(debug_dump_dir=str(tmp_path)))
    engine = AudioEngine(
        cfg,
        vad_model_path=None,
        validate_words=lambda s, r: True,
        on_segment=lambda seg: None,
    )
    return engine


def test_emit_writes_wav(tmp_path):
    engine = _make_engine(tmp_path)
    engine._append(np.zeros(8000, dtype=np.float32))
    engine._emit("manual")

    wav_files = list(tmp_path.glob("*.wav"))
    assert len(wav_files) == 1, f"expected 1 WAV, got {wav_files}"

    with wave.open(str(wav_files[0]), "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        assert w.getnframes() == 8000
