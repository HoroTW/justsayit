"""Parakeet TDT v3 transcription via sherpa-onnx."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from justsayit.config import Config
from justsayit.model import paths as _resolve_paths
from justsayit.transcribe import TranscriberBase

log = logging.getLogger(__name__)

_NORMALIZE_PRESETS: dict[str, tuple[float, float]] = {
    "off": (0.0, 1.0),
    "A":   (0.15, 8.0),
    "B":   (0.30, 8.0),
    "C":   (0.30, 4.0),
}

_CHUNK_MAX_SECONDS = 25.0        # safe upper bound (margin from 35s breakpoint)
_CHUNK_MIN_SECONDS = 5.0         # don't produce micro-chunks
_CHUNK_SILENCE_RMS = 0.005       # same threshold as the existing trim default
_CHUNK_SILENCE_WINDOW_MS = 250   # silence must be >= this to count as a sentence boundary


def _chunk_at_silence(samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Return (start, end) sample-index pairs covering ``samples``.

    If the total duration is <= _CHUNK_MAX_SECONDS, returns a single chunk.
    Otherwise, walks forward greedily: for each chunk, scan the window
    [start + _CHUNK_MIN_SECONDS, start + _CHUNK_MAX_SECONDS] for the LAST
    silence trough (RMS below _CHUNK_SILENCE_RMS over _CHUNK_SILENCE_WINDOW_MS).
    Splits at the middle of that silence window; if none found, force-splits
    at start + _CHUNK_MAX_SECONDS.
    """
    total = len(samples) / float(sample_rate)
    if total <= _CHUNK_MAX_SECONDS:
        return [(0, len(samples))]

    win = max(1, int(_CHUNK_SILENCE_WINDOW_MS / 1000.0 * sample_rate))
    max_samples = int(_CHUNK_MAX_SECONDS * sample_rate)
    min_samples = int(_CHUNK_MIN_SECONDS * sample_rate)

    chunks: list[tuple[int, int]] = []
    start = 0
    n = len(samples)

    while start < n:
        remaining = n - start
        if remaining <= max_samples:
            chunks.append((start, n))
            break

        # Scan [start+min, start+max] for the last silence trough
        scan_start = start + min_samples
        scan_end = start + max_samples

        last_silence_mid: int | None = None
        pos = scan_start
        while pos + win <= scan_end:
            window = samples[pos:pos + win]
            rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
            if rms < _CHUNK_SILENCE_RMS:
                last_silence_mid = pos + win // 2
            pos += win // 2  # half-overlap for finer detection

        if last_silence_mid is not None:
            split = last_silence_mid
        else:
            split = start + max_samples

        chunks.append((start, split))
        start = split

    return chunks


def _trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    threshold_rms: float,
    min_keep_seconds: float,
    window_ms: int = 50,
) -> tuple[np.ndarray, float, float]:
    """Strip leading/trailing 50-ms windows whose RMS is below threshold.

    Returns ``(samples, head_seconds_trimmed, tail_seconds_trimmed)``. If
    the trim would shrink the buffer below ``min_keep_seconds``, returns
    the original buffer untouched (and (0.0, 0.0)).
    """
    if threshold_rms <= 0.0:
        return samples, 0.0, 0.0
    win = max(1, int(window_ms / 1000.0 * sample_rate))
    n = len(samples)
    if n == 0:
        return samples, 0.0, 0.0
    rms = lambda c: float(np.sqrt(np.mean(np.square(c, dtype=np.float64))))
    start = 0
    while start + win <= n and rms(samples[start:start + win]) < threshold_rms:
        start += win
    end = n
    while end - win >= start and rms(samples[end - win:end]) < threshold_rms:
        end -= win
    kept_seconds = (end - start) / float(sample_rate)
    if kept_seconds < min_keep_seconds:
        return samples, 0.0, 0.0
    return samples[start:end], start / float(sample_rate), (n - end) / float(sample_rate)


def _normalize(samples: np.ndarray, preset: str) -> tuple[np.ndarray, float]:
    """Boost ``samples`` toward the preset's min_peak, capped by max_gain.
    Returns (samples, gain). Unknown presets fall back to "A"."""
    min_peak, max_gain = _NORMALIZE_PRESETS.get(preset, _NORMALIZE_PRESETS["A"])
    if min_peak <= 0.0:
        return samples, 1.0
    peak = float(np.abs(samples).max())
    if peak <= 0.0 or peak >= min_peak:
        return samples, 1.0
    gain = min(min_peak / peak, max_gain)
    return samples * gain, gain


class ParakeetTranscriber(TranscriberBase):
    """Thin wrapper around sherpa_onnx.OfflineRecognizer for Parakeet TDT."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.paths = _resolve_paths(cfg)
        self._recog = None  # lazy
        self._lock = threading.Lock()

    def _build(self):
        import sherpa_onnx

        def _p(p: Path) -> str:
            return str(p)

        log.info("loading Parakeet recognizer from %s", self.paths.encoder.parent)
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_p(self.paths.encoder),
            decoder=_p(self.paths.decoder),
            joiner=_p(self.paths.joiner),
            tokens=_p(self.paths.tokens),
            num_threads=max(1, int(self.cfg.model.num_threads)),
            sample_rate=int(self.cfg.audio.sample_rate),
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            model_type="nemo_transducer",
        )

    def warmup(self) -> None:
        with self._lock:
            if self._recog is None:
                self._recog = self._build()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        with self._lock:
            if self._recog is None:
                self._recog = self._build()
            samples, head_s, tail_s = _trim_silence(
                samples,
                int(sample_rate),
                self.cfg.model.parakeet_trim_silence_rms,
                self.cfg.model.parakeet_trim_min_keep_seconds,
            )
            if head_s > 0.0 or tail_s > 0.0:
                log.info(
                    "parakeet trim-silence: cut %.2fs head + %.2fs tail (kept %.2fs)",
                    head_s, tail_s, len(samples) / float(sample_rate),
                )
            samples, gain = _normalize(samples, self.cfg.model.parakeet_normalize)
            if gain != 1.0:
                log.info(
                    "parakeet input boost: %.2fx (preset=%s, peak %.4f -> %.4f)",
                    gain, self.cfg.model.parakeet_normalize,
                    float(np.abs(samples).max()) / gain,
                    float(np.abs(samples).max()),
                )
            chunks = _chunk_at_silence(samples, int(sample_rate))
            if len(chunks) > 1:
                total_s = len(samples) / float(sample_rate)
                log.info(
                    "parakeet auto-chunk: split %.2fs into %d pieces", total_s, len(chunks)
                )
                for i, (cs, ce) in enumerate(chunks):
                    log.debug(
                        "  chunk %d: %.2fs–%.2fs",
                        i, cs / float(sample_rate), ce / float(sample_rate),
                    )
            parts: list[str] = []
            for cs, ce in chunks:
                chunk_samples = samples[cs:ce].astype(np.float32, copy=False)
                stream = self._recog.create_stream()
                stream.accept_waveform(int(sample_rate), chunk_samples)
                self._recog.decode_stream(stream)
                t = (stream.result.text or "").strip()
                if t:
                    parts.append(t)
        return " ".join(parts)
