"""Transcription backends for justsayit.

Provides ``TranscriberBase`` (the common interface used throughout the app)
and ``make_transcriber(cfg, model_paths)`` which instantiates the right
backend based on ``cfg.model.backend``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from justsayit.config import Config
    from justsayit.model import ModelPaths

# "Has words" heuristic for the 3-second validation window.
_WORD_RE = re.compile(r"\w{2,}", re.UNICODE)


class TranscriberBase:
    """Common interface for all transcription backends."""

    def warmup(self) -> None:
        """Eagerly load the model so the first real transcription isn't slow."""

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Synchronous single-shot transcription. Returns stripped text."""
        raise NotImplementedError

    def has_words(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Validation hook: True iff transcription produced ≥1 word token."""
        if len(samples) == 0:
            return False
        return bool(_WORD_RE.search(self.transcribe(samples, sample_rate)))


def make_transcriber(
    cfg: "Config", model_paths: "ModelPaths | None" = None
) -> TranscriberBase:
    """Instantiate the transcriber for ``cfg.model.backend``.

    ``model_paths`` is required when backend is ``"parakeet"`` and ignored
    for ``"whisper"`` (faster-whisper handles its own model loading).
    """
    backend = cfg.model.backend
    if backend == "parakeet":
        from justsayit.transcribe_parakeet import ParakeetTranscriber

        if model_paths is None:
            raise ValueError("model_paths is required for the parakeet backend")
        return ParakeetTranscriber(cfg, model_paths)
    if backend == "whisper":
        from justsayit.transcribe_whisper import WhisperTranscriber

        return WhisperTranscriber(cfg)
    if backend == "openai":
        from justsayit.transcribe_openai import OpenAIWhisperTranscriber

        return OpenAIWhisperTranscriber(cfg)
    raise ValueError(f"unknown transcription backend: {backend!r}")
