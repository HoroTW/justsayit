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

_CHUNK_TARGET_SECONDS = 25.0     # start looking for a natural split here
_CHUNK_FORCE_SECONDS = 45.0      # hard split only if no silence is found by here
_CHUNK_MIN_SECONDS = 5.0         # don't produce micro-chunks
_CHUNK_SILENCE_RMS = 0.005       # absolute floor; adaptive scan handles noisy rooms
_CHUNK_SILENCE_RMS_CAP = 0.020   # don't treat quiet speech as a split point
_CHUNK_SILENCE_NOISE_MULT = 1.35 # allow pauses above the absolute floor
_CHUNK_SILENCE_WINDOW_MS = (250, 200, 150, 100)
_CHUNK_TARGET_LOOKBACK_SECONDS = 2.0


def _adaptive_silence_rms(window_rms: list[float]) -> float:
    """Return a local silence threshold for the current scan window."""
    if not window_rms:
        return _CHUNK_SILENCE_RMS
    low_rms = float(np.percentile(window_rms, 10))
    return max(
        _CHUNK_SILENCE_RMS,
        min(_CHUNK_SILENCE_RMS_CAP, low_rms * _CHUNK_SILENCE_NOISE_MULT),
    )


def _find_silence_split(
    samples: np.ndarray,
    scan_start: int,
    scan_end: int,
    sample_rate: int,
) -> int | None:
    """Return the first silence split in ``[scan_start, scan_end]``.

    The search starts with conservative 250-ms pauses and relaxes down to
    100-ms pauses before the caller falls back to a hard split.
    """
    for window_ms in _CHUNK_SILENCE_WINDOW_MS:
        win = max(1, int(window_ms / 1000.0 * sample_rate))
        if scan_start + win > scan_end:
            continue

        rms_windows: list[tuple[int, float]] = []
        pos = scan_start
        while pos + win <= scan_end:
            window = samples[pos:pos + win]
            rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
            rms_windows.append((pos + win // 2, rms))
            pos += max(1, win // 2)  # half-overlap for finer detection

        if not rms_windows:
            continue

        # Real mic captures often have a room-noise floor above the fixed
        # 0.005 threshold. Estimate the local floor from the scan window,
        # but cap it so low-energy speech is not treated as silence.
        silence_rms = _adaptive_silence_rms([rms for _, rms in rms_windows])

        for mid, rms in rms_windows:
            if rms < silence_rms:
                return mid

    return None


def _chunk_at_silence(samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Return (start, end) sample-index pairs covering ``samples``.

    Chunks aim to split on natural pauses after _CHUNK_TARGET_SECONDS. If no
    pause appears by _CHUNK_FORCE_SECONDS, the chunk is hard-split there.
    """
    total = len(samples) / float(sample_rate)
    if total <= _CHUNK_TARGET_SECONDS:
        return [(0, len(samples))]

    target_samples = int(_CHUNK_TARGET_SECONDS * sample_rate)
    force_samples = int(_CHUNK_FORCE_SECONDS * sample_rate)
    min_samples = int(_CHUNK_MIN_SECONDS * sample_rate)
    target_lookback_samples = int(_CHUNK_TARGET_LOOKBACK_SECONDS * sample_rate)

    chunks: list[tuple[int, int]] = []
    start = 0
    n = len(samples)

    while start < n:
        remaining = n - start
        if remaining <= target_samples:
            chunks.append((start, n))
            break

        scan_start = start + max(min_samples, target_samples - target_lookback_samples)
        scan_end = min(n, start + force_samples)

        split = _find_silence_split(samples, scan_start, scan_end, sample_rate)
        if split is not None:
            split = max(start + min_samples, min(split, scan_end))
        elif remaining > force_samples:
            split = start + force_samples
        else:
            chunks.append((start, n))
            break

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

    def prime(self) -> None:
        """Force a forward pass with a tiny silence buffer to page the
        model weights into RAM. Skips trim/normalize/chunk — pure model touch."""
        with self._lock:
            if self._recog is None:
                self._recog = self._build()
            sr = int(self.cfg.audio.sample_rate)
            # 100ms of silence is enough to traverse the encoder weights.
            dummy = np.zeros(int(0.1 * sr), dtype=np.float32)
            stream = self._recog.create_stream()
            stream.accept_waveform(sr, dummy)
            self._recog.decode_stream(stream)

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
            chunks = _chunk_at_silence(samples, int(sample_rate))
            samples, gain = _normalize(samples, self.cfg.model.parakeet_normalize)
            if gain != 1.0:
                log.info(
                    "parakeet input boost: %.2fx (preset=%s, peak %.4f -> %.4f)",
                    gain, self.cfg.model.parakeet_normalize,
                    float(np.abs(samples).max()) / gain,
                    float(np.abs(samples).max()),
                )
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
