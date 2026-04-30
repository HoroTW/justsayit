"""Transcription backends for justsayit.

Provides ``TranscriberBase`` (the common interface used throughout the app)
and ``make_transcriber(cfg)`` which instantiates the right backend based on
``cfg.model.backend``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from justsayit.config import Config

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


def make_transcriber(cfg: "Config") -> TranscriberBase:
    """Instantiate the transcriber for ``cfg.model.backend``."""
    backend = cfg.model.backend
    if backend == "parakeet":
        from justsayit.transcribe_parakeet import ParakeetTranscriber

        return ParakeetTranscriber(cfg)
    if backend == "whisper":
        from justsayit.transcribe_whisper import WhisperTranscriber

        return WhisperTranscriber(cfg)
    if backend == "openai":
        from justsayit.transcribe_openai import OpenAIWhisperTranscriber

        return OpenAIWhisperTranscriber(cfg)
    raise ValueError(f"unknown transcription backend: {backend!r}")
