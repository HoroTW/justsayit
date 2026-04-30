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

# (fixture_basename, preset, expected_substring_or_empty_for_empty_result)
CASES = [
    # Preset "off" baseline — failed file MUST be empty (locks in the regression).
    ("failed_quiet_gpt_attack", "off", ""),
    # Preset "A" (default): all 4 files must transcribe correctly.
    ("quiet_silent_label_a",    "A", "silent"),
    ("quiet_silent_label_b",    "A", "silent"),
    ("varianz_de",              "A", "leiser"),
    ("failed_quiet_gpt_attack", "A", "near zero overlap"),
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
    """Cache one transcriber per preset to avoid reloading for each case."""
    return {}


def _get_transcriber(preset: str, cache: dict) -> ParakeetTranscriber:
    if preset not in cache:
        cfg = Config()
        cfg.model.backend = "parakeet"
        cfg.model.parakeet_normalize = preset
        t = ParakeetTranscriber(cfg)
        # Check model files exist before attempting warmup
        if not t.paths.encoder.exists():
            pytest.skip(f"Parakeet model not downloaded (missing {t.paths.encoder})")
        t.warmup()
        cache[preset] = t
    return cache[preset]


@pytest.mark.parametrize("basename,preset,expected", CASES, ids=[
    f"{b}-{p}" for b, p, _ in CASES
])
def test_normalize_preset(basename, preset, expected, _transcriber_cache):
    wav_path = FIXTURES_DIR / f"{basename}.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    transcriber = _get_transcriber(preset, _transcriber_cache)
    out = transcriber.transcribe(samples, sr).strip()

    if expected == "":
        assert out == "", f"expected empty output (preset={preset!r}), got: {out!r}"
    else:
        assert expected.lower() in out.lower(), (
            f"expected {expected!r} in output (preset={preset!r}), got: {out!r}"
        )
