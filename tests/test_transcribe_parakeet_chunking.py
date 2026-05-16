from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from justsayit.transcribe_parakeet import (
    _CHUNK_SILENCE_RMS,
    _CHUNK_SILENCE_RMS_CAP,
    _adaptive_silence_rms,
    _chunk_at_silence,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    dtype = np.int16 if sampwidth == 2 else np.int32
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        samples = samples[::n_channels]
    samples /= float(np.iinfo(dtype).max)
    return samples, sr


def test_chunker_uses_noisy_pause_instead_of_hard_25s_split():
    wav_path = FIXTURES_DIR / "long_42s_chunking.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    chunks = _chunk_at_silence(samples, sr)

    assert len(chunks) == 2
    split_s = chunks[0][1] / float(sr)
    assert 23.0 <= split_s <= 24.5


def test_chunker_uses_100ms_silence_before_force_split():
    sr = 16_000
    samples = np.full(50 * sr, 0.1, dtype=np.float32)
    samples[30 * sr:int(30.1 * sr)] = 0.0

    chunks = _chunk_at_silence(samples, sr)

    split_s = chunks[0][1] / float(sr)
    assert split_s == pytest.approx(30.05, abs=0.1)


def test_chunker_hard_splits_at_45s_without_silence():
    sr = 16_000
    samples = np.full(50 * sr, 0.1, dtype=np.float32)

    chunks = _chunk_at_silence(samples, sr)

    split_s = chunks[0][1] / float(sr)
    assert split_s == pytest.approx(45.0)


def test_adaptive_silence_floor_tracks_room_noise_with_cap():
    assert _adaptive_silence_rms([]) == pytest.approx(_CHUNK_SILENCE_RMS)

    quiet_room = [0.001, 0.002, 0.003, 0.004]
    assert _adaptive_silence_rms(quiet_room) == pytest.approx(_CHUNK_SILENCE_RMS)

    noisy_room = [0.009, 0.010, 0.011, 0.012]
    assert _adaptive_silence_rms(noisy_room) > _CHUNK_SILENCE_RMS

    loud_input = [0.10, 0.11, 0.12, 0.13]
    assert _adaptive_silence_rms(loud_input) == pytest.approx(_CHUNK_SILENCE_RMS_CAP)
