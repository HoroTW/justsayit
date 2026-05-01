"""Burn tests for parakeet_normalize presets against real WAV fixtures.

Skipped by default (pytest config deselects ``burn`` marker). Run with:

    pytest tests/test_burn_parakeet_normalize.py -m burn -v

Requires the Parakeet model to be present locally (run
``justsayit download-models`` first). Also skipped gracefully when the
fixture WAVs are absent (clean envs without audio dumps).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from justsayit.config import Config
from justsayit.transcribe_parakeet import ParakeetTranscriber

pytestmark = pytest.mark.burn

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"

# (fixture_basename, normalize_preset, trim_rms, expected_substring_or_empty_for_empty_result)
CASES = [
    # Raw baselines — locks in that without ANY of our fixes, the failing files are empty.
    ("failed_quiet_gpt_attack",   "off", 0.0,    ""),
    ("failed_long_silence_tail",  "off", 0.0,    ""),
    # New file is NOT fixed by normalize alone (any preset) — needs trim.
    ("failed_long_silence_tail",  "A",   0.0,    ""),
    ("failed_long_silence_tail",  "B",   0.0,    ""),
    # Default config (normalize="A", trim=0.005) must transcribe everything correctly.
    ("quiet_silent_label_a",      "A",   0.005, "silent"),
    ("quiet_silent_label_b",      "A",   0.005, "silent"),
    ("varianz_de",                "A",   0.005, "leiser"),
    ("failed_quiet_gpt_attack",   "A",   0.005, "near zero overlap"),
    ("failed_long_silence_tail",  "A",   0.005, "level three attack"),
]


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
    # Normalize to [-1, 1]
    samples /= float(np.iinfo(dtype).max)
    return samples, sr


@pytest.fixture(scope="module")
def _transcriber_cache():
    """Cache one transcriber per (preset, trim_rms) pair to avoid reloading for each case."""
    return {}


def _get_transcriber(preset: str, trim_rms: float, cache: dict) -> ParakeetTranscriber:
    key = (preset, trim_rms)
    if key not in cache:
        cfg = Config()
        cfg.model.backend = "parakeet"
        cfg.model.parakeet_normalize = preset
        cfg.model.parakeet_trim_silence_rms = trim_rms
        t = ParakeetTranscriber(cfg)
        # Check model files exist before attempting warmup
        if not t.paths.encoder.exists():
            pytest.skip(f"Parakeet model not downloaded (missing {t.paths.encoder})")
        t.warmup()
        cache[key] = t
    return cache[key]


@pytest.mark.parametrize("basename,preset,trim_rms,expected", CASES, ids=[
    f"{b}-{p}-trim{t}" for b, p, t, _ in CASES
])
def test_normalize_preset(basename, preset, trim_rms, expected, _transcriber_cache):
    wav_path = FIXTURES_DIR / f"{basename}.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    transcriber = _get_transcriber(preset, trim_rms, _transcriber_cache)
    out = transcriber.transcribe(samples, sr).strip()

    if expected == "":
        assert out == "", f"expected empty output (preset={preset!r}, trim_rms={trim_rms}), got: {out!r}"
    else:
        assert expected.lower() in out.lower(), (
            f"expected {expected!r} in output (preset={preset!r}, trim_rms={trim_rms}), got: {out!r}"
        )
