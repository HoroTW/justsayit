"""faster-whisper / distil-whisper transcription backend."""

from __future__ import annotations

import logging
import threading

import numpy as np

from justsayit.config import Config
from justsayit.model import models_dir
from justsayit.transcribe import TranscriberBase

log = logging.getLogger(__name__)


class WhisperTranscriber(TranscriberBase):
    """Transcription via faster-whisper (CTranslate2).

    The model is downloaded from HuggingFace Hub on first use into
    ``<cache>/justsayit/models/whisper/``.  No explicit download step
    is needed — just set ``model.backend = "whisper"`` in config.toml.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None  # lazy
        self._lock = threading.Lock()

    def _build(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: uv pip install 'justsayit[whisper]'  "
                "or re-run install.sh --model whisper"
            ) from exc

        model_id = self.cfg.model.whisper_model
        download_root = str(models_dir() / "whisper")
        log.info(
            "loading faster-whisper model %s  device=%s  compute_type=%s",
            model_id,
            self.cfg.model.whisper_device,
            self.cfg.model.whisper_compute_type,
        )
        return WhisperModel(
            model_id,
            device=self.cfg.model.whisper_device,
            compute_type=self.cfg.model.whisper_compute_type,
            cpu_threads=max(1, int(self.cfg.model.num_threads)),
            download_root=download_root,
        )

    def warmup(self) -> None:
        with self._lock:
            if self._model is None:
                self._model = self._build()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        with self._lock:
            if self._model is None:
                self._model = self._build()
            segments, _ = self._model.transcribe(
                samples.astype(np.float32, copy=False),
                beam_size=5,
            )
            text = "".join(seg.text for seg in segments)
        return text.strip()
