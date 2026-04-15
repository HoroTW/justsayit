#!/usr/bin/env python3
"""Generate the notification WAV files bundled with justsayit.

Outputs start.wav and stop.wav into src/justsayit/sounds/.  Those files
are committed to the repo — end-users never need to run this script.

    uv run python sounds/generate_sounds.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent.parent / "src" / "justsayit" / "sounds"
SAMPLE_RATE = 44100


def chime(
    freq: float,
    duration: float,
    *,
    decay: float,
    harmonics: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return int16 mono samples for a bell-like chime.

    *harmonics* is a list of (frequency_ratio, amplitude) pairs that are
    summed to form the waveform.  Defaults to a gentle overtone series that
    gives a metallic warmth without sounding harsh.
    """
    if harmonics is None:
        harmonics = [(1.0, 0.65), (2.0, 0.22), (3.0, 0.09), (4.0, 0.04)]

    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    signal: np.ndarray = sum(  # type: ignore[assignment]
        amp * np.sin(2 * np.pi * freq * ratio * t)
        for ratio, amp in harmonics
    )

    # 4 ms linear ramp-up to kill the click at onset, then exponential decay.
    attack = int(0.004 * SAMPLE_RATE)
    envelope = np.exp(-decay * t)
    envelope[:attack] = np.linspace(0.0, 1.0, attack)
    signal *= envelope

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal /= peak
    signal *= 0.80  # leave a little headroom

    return (signal * 32767).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    ms = len(samples) / SAMPLE_RATE * 1000
    print(f"  wrote {path.relative_to(Path(__file__).parent.parent)}  ({ms:.0f} ms)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("generating sounds …")

    # A4 (440 Hz) — full, mid-range, signals "recording started"
    write_wav(
        OUT_DIR / "start.wav",
        chime(440.0, 0.38, decay=10.0, harmonics=[(1.0, 0.78), (2.0, 0.18), (3.0, 0.04)]),
    )

    # E4 (329.63 Hz) — a perfect fourth lower, deep and settled;
    # slower decay so it rings noticeably longer than the start sound
    write_wav(
        OUT_DIR / "stop.wav",
        chime(329.63, 0.53, decay=7.8, harmonics=[(1.0, 0.78), (2.0, 0.18), (3.0, 0.04)]),
    )

    print("done.")


if __name__ == "__main__":
    main()
