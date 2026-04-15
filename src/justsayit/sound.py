"""Notification sound playback for justsayit.

Sounds are loaded once at startup from the bundled WAV files and played
asynchronously (fire-and-forget) via sounddevice so audio callbacks are
never blocked.
"""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

_SOUNDS_DIR = Path(__file__).parent / "sounds"
_SAMPLE_RATE = 44100


def _load_wav(path: Path) -> np.ndarray:
    """Return a float32 array in [-1, 1] from a 16-bit mono WAV file."""
    with wave.open(str(path)) as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


class SoundPlayer:
    def __init__(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        self._start: np.ndarray | None = None
        self._stop: np.ndarray | None = None
        self._mute: np.ndarray | None = None
        self._unmute: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        for name, attr in [("start", "_start"), ("stop", "_stop"), ("mute", "_mute"), ("unmute", "_unmute")]:
            path = _SOUNDS_DIR / f"{name}.wav"
            try:
                setattr(self, attr, _load_wav(path))
                log.debug("loaded sound: %s", path)
            except Exception:
                log.warning("could not load sound %s", path, exc_info=True)

    def play_start(self, volume_scale: float = 1.0) -> None:
        self._play(self._start, volume_scale)

    def play_stop(self) -> None:
        self._play(self._stop)

    def play_mute(self) -> None:
        self._play(self._mute)

    def play_unmute(self) -> None:
        self._play(self._unmute)

    def _play(self, samples: np.ndarray | None, volume_scale: float = 1.0) -> None:
        if samples is None or self._volume <= 0.0:
            return
        data = samples * self._volume * volume_scale
        # Spawn a tiny daemon thread so the caller (audio engine callback)
        # is never blocked by sd.play() stream setup.
        threading.Thread(target=_fire, args=(data,), daemon=True).start()


def _fire(data: np.ndarray) -> None:
    try:
        sd.play(data, samplerate=_SAMPLE_RATE)
    except Exception:
        log.debug("sound playback error", exc_info=True)
