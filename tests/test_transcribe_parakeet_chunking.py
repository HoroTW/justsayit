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
    _compress_internal_silence,
    _trim_silence,
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


def test_chunker_uses_best_valid_pause_without_collapsing_tail():
    wav_path = FIXTURES_DIR / "long_42s_chunking.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    chunks = _chunk_at_silence(samples, sr)

    assert len(chunks) >= 2
    split_s = chunks[0][1] / float(sr)
    chunk_lengths = [(end - start) / float(sr) for start, end in chunks]
    assert 10.0 <= split_s <= 35.0
    assert all(length >= 5.0 for length in chunk_lengths)


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


def test_chunker_does_not_create_tiny_final_tail_after_silence_split():
    sr = 16_000
    samples = np.full(int(28.0 * sr), 0.1, dtype=np.float32)
    samples[int(26.0 * sr):int(26.25 * sr)] = 0.0

    chunks = _chunk_at_silence(samples, sr)

    assert chunks == [(0, len(samples))]


def test_chunker_relaxes_to_shorter_valid_pause_instead_of_tiny_tail():
    sr = 16_000
    samples = np.full(int(63.0 * sr), 0.1, dtype=np.float32)
    samples[int(29.3 * sr):int(29.55 * sr)] = 0.0
    samples[int(53.0 * sr):int(53.1 * sr)] = 0.0
    samples[int(62.75 * sr):int(63.0 * sr)] = 0.0

    chunks = _chunk_at_silence(samples, sr)

    assert len(chunks) == 3
    assert chunks[1][1] / float(sr) == pytest.approx(53.05, abs=0.1)


def test_adaptive_silence_floor_tracks_room_noise_with_cap():
    assert _adaptive_silence_rms([]) == pytest.approx(_CHUNK_SILENCE_RMS)

    quiet_room = [0.001, 0.002, 0.003, 0.004]
    assert _adaptive_silence_rms(quiet_room) == pytest.approx(_CHUNK_SILENCE_RMS)

    noisy_room = [0.009, 0.010, 0.011, 0.012]
    assert _adaptive_silence_rms(noisy_room) > _CHUNK_SILENCE_RMS

    loud_input = [0.10, 0.11, 0.12, 0.13]
    assert _adaptive_silence_rms(loud_input) == pytest.approx(_CHUNK_SILENCE_RMS_CAP)


def test_trim_silence_keeps_small_context_pad():
    sr = 16_000
    silence = np.zeros(int(0.20 * sr), dtype=np.float32)
    speech = np.full(int(1.0 * sr), 0.1, dtype=np.float32)
    samples = np.concatenate([silence, speech, silence])

    trimmed, head_s, tail_s = _trim_silence(samples, sr, 0.005, 1.0)

    assert head_s == pytest.approx(0.10)
    assert tail_s == pytest.approx(0.10)
    assert len(trimmed) == pytest.approx(int(1.2 * sr), abs=sr * 0.01)


def test_compress_internal_silence_caps_long_pauses():
    sr = 16_000
    speech = np.full(int(0.5 * sr), 0.1, dtype=np.float32)
    pause = np.zeros(int(3.0 * sr), dtype=np.float32)
    samples = np.concatenate([speech, pause, speech])

    compressed, removed_s = _compress_internal_silence(samples, sr, 0.005, 0.8)

    assert removed_s == pytest.approx(2.2, abs=0.1)
    assert len(compressed) / float(sr) == pytest.approx(1.8, abs=0.1)
