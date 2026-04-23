"""OpenAI-compatible /audio/transcriptions transcription backend.

Sends the captured audio buffer as a multipart-form WAV upload to any
endpoint that speaks the OpenAI Whisper schema.  Compatible providers
include OpenAI, Groq, self-hosted faster-whisper-server, vLLM with the
audio extension, and whisper.cpp's bundled HTTP server.

Pure stdlib: no ``openai`` dependency.  Audio is encoded in-memory as
16-bit PCM WAV (the format every Whisper-style server accepts) and
posted directly via ``urllib.request``.
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import threading
import urllib.request
import wave

import numpy as np

from justsayit._http import request_with_retry
from justsayit.config import Config, resolve_secret
from justsayit.transcribe import TranscriberBase

log = logging.getLogger(__name__)


def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Return *samples* (mono float32 in [-1, 1]) as a 16-bit PCM WAV blob."""
    if samples.ndim != 1:
        samples = np.mean(samples, axis=1)
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _build_multipart(
    *,
    wav_bytes: bytes,
    model: str,
    language: str = "",
    response_format: str = "json",
) -> tuple[bytes, str]:
    """Hand-built multipart/form-data payload.

    Returns ``(body, content_type)``.  Avoids pulling in ``requests`` or
    ``email.mime`` for what is in practice three form fields.
    """
    boundary = "----justsayit-" + secrets.token_hex(16)
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )

    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode("utf-8")
    )
    parts.append(wav_bytes)
    parts.append(b"\r\n")
    field("model", model)
    field("response_format", response_format)
    if language:
        field("language", language)
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class OpenAIWhisperTranscriber(TranscriberBase):
    """Transcription via an OpenAI-compatible /audio/transcriptions API.

    All knobs (endpoint, model, key source, timeout, language hint) live
    on ``cfg.model`` so the user can switch from local Parakeet/Whisper
    to a hosted endpoint by editing config.toml — no extra install step.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        if not cfg.model.openai_endpoint:
            raise ValueError(
                "model.backend = \"openai\" requires model.openai_endpoint to be set"
            )

    # warmup() is a no-op — there's nothing to load locally and we don't
    # want a fake call burning quota / latency every cold start.

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if len(samples) == 0:
            return ""
        api_key = resolve_secret(
            self.cfg.model.openai_api_key, self.cfg.model.openai_api_key_env
        )
        if not api_key:
            raise RuntimeError(
                "openai backend requires an API key.\n"
                f"  Set model.openai_api_key, export {self.cfg.model.openai_api_key_env},\n"
                "  or put it in ~/.config/justsayit/.env."
            )
        wav_bytes = _encode_wav(samples, sample_rate)
        body, content_type = _build_multipart(
            wav_bytes=wav_bytes,
            model=self.cfg.model.openai_model,
            language=self.cfg.model.openai_language,
        )
        url = self.cfg.model.openai_endpoint.rstrip("/") + "/audio/transcriptions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "justsayit",
            },
            method="POST",
        )
        with self._lock:  # serialise so we don't fan out parallel requests
            raw = request_with_retry(
                req,
                timeout=self.cfg.model.openai_timeout,
                retries=self.cfg.model.openai_retries,
                delay=self.cfg.model.openai_retry_delay,
                label="transcription",
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Some servers return text/plain when response_format=text was
            # negotiated. We requested json, but tolerate the fallback.
            return raw.decode("utf-8", errors="replace").strip()
        if isinstance(data, dict) and "text" in data:
            return str(data["text"]).strip()
        return str(data).strip()
