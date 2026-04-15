"""Parakeet TDT v3 transcription via sherpa-onnx."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import numpy as np

from justsayit.config import Config
from justsayit.model import ModelPaths

log = logging.getLogger(__name__)

# "Has words" heuristic for the 3-second validation window.
_WORD_RE = re.compile(r"\w{2,}", re.UNICODE)


class Transcriber:
    """Thin wrapper around sherpa_onnx.OfflineRecognizer for Parakeet TDT."""

    def __init__(self, cfg: Config, paths: ModelPaths) -> None:
        self.cfg = cfg
        self.paths = paths
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
        """Eagerly load the model so the first real transcription isn't slow."""
        with self._lock:
            if self._recog is None:
                self._recog = self._build()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Synchronous single-shot transcription."""
        with self._lock:
            if self._recog is None:
                self._recog = self._build()
            stream = self._recog.create_stream()
            stream.accept_waveform(int(sample_rate), samples.astype(np.float32, copy=False))
            self._recog.decode_stream(stream)
            text = stream.result.text or ""
        return text.strip()

    def has_words(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Cheap validation hook: True iff transcription produced ≥1 word token."""
        if len(samples) == 0:
            return False
        text = self.transcribe(samples, sample_rate)
        return bool(_WORD_RE.search(text))
