"""Burn tests for clipboard image input via the OpenAI Responses API.

Tests that a real base64-encoded PNG is accepted and produces a coherent
response. Skipped by default; run explicitly with:

    pytest -m burn tests/test_burn_responses_image.py

Cost per run: ~1–4 requests × gpt-5.4-mini with detail="high" image
(~512–765 tokens per 512×512 tile) — roughly $0.001–0.005.
"""

from __future__ import annotations

import base64
import io
import struct
import zlib
from pathlib import Path

import pytest

from justsayit.config import resolve_secret
from justsayit.postprocess import load_profile

pytestmark = pytest.mark.burn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_openai_key() -> str:
    key = resolve_secret("", "OPENAI_API_KEY")
    if not key:
        pytest.skip(
            "OPENAI_API_KEY not set (env or ~/.config/justsayit/.env) — "
            "skipping Responses API image burn test"
        )
    return key


def _make_test_png(text: str = "Hello OCR", width: int = 200, height: int = 50) -> bytes:
    """Build a minimal valid PNG in-memory. No PIL dependency.

    Produces a solid white image with a thick black border — enough for
    the API to prove it can decode the image, even without visible text
    (the model will say it sees a white/black rectangle, which is coherent).
    """
    def _pack_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), ...
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _pack_chunk(b"IHDR", ihdr_data)

    # Build raw image data: solid white with black border
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            on_border = x < 2 or x >= width - 2 or y < 2 or y >= height - 2
            px = b"\x00\x00\x00" if on_border else b"\xff\xff\xff"
            row += px
        rows.append(b"\x00" + bytes(row))  # filter byte = None

    raw = b"".join(rows)
    compressed = zlib.compress(raw)
    idat = _pack_chunk(b"IDAT", compressed)
    iend = _pack_chunk(b"IEND", b"")

    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _responses_profile(tmp_path: Path, **overrides) -> object:
    """Write a minimal Responses API profile and load it."""
    lines = [
        'base = "responses"\n',
        'endpoint = "https://api.openai.com/v1"\n',
        'model = "gpt-5.4-mini"\n',
        'api_key_env = "OPENAI_API_KEY"\n',
        'system_prompt = "You are a helpful assistant. Reply in one short sentence."\n',
        "max_tokens = 64\n",
        "request_timeout = 30.0\n",
        "remote_retries = 0\n",
    ]
    for key, val in overrides.items():
        if isinstance(val, str):
            lines.append(f'{key} = "{val}"\n')
        else:
            lines.append(f"{key} = {val}\n")
    p = tmp_path / "responses-burn.toml"
    p.write_text("".join(lines), encoding="utf-8")
    return load_profile(str(p))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_burn_responses_image_detail_high(tmp_path):
    """Send a real PNG via base64 with detail='high' and confirm the model
    returns a non-empty coherent response about the image content."""
    _require_openai_key()
    profile = _responses_profile(tmp_path, image_detail="high")
    assert profile.image_detail == "high"

    from justsayit.postprocess import make_postprocessor
    pp = make_postprocessor(profile)

    png = _make_test_png()
    result = pp.process_with_reasoning(
        "What do you see in this image?",
        extra_image=png,
        extra_image_mime="image/png",
    )

    assert result.text.strip(), "model returned empty response for image input"
    # The model should describe a rectangle or image — not refuse or error.
    assert len(result.text) > 10, f"suspiciously short reply: {result.text!r}"


def test_burn_responses_image_detail_low(tmp_path):
    """Same as high but with detail='low' (cheaper, fewer tokens)."""
    _require_openai_key()
    profile = _responses_profile(tmp_path, image_detail="low")

    from justsayit.postprocess import make_postprocessor
    pp = make_postprocessor(profile)

    png = _make_test_png()
    result = pp.process_with_reasoning(
        "Describe this image briefly.",
        extra_image=png,
        extra_image_mime="image/png",
    )

    assert result.text.strip()


def test_burn_responses_image_off_ignores_image(tmp_path):
    """When image_detail='off', the image is not sent and the
    model answers based on text alone."""
    _require_openai_key()
    profile = _responses_profile(tmp_path, image_detail="off")
    assert profile.image_detail == "off"

    from justsayit.postprocess import make_postprocessor
    pp = make_postprocessor(profile)

    png = _make_test_png()
    # The image is present but should be silently dropped.
    result = pp.process_with_reasoning(
        "hello world",
        extra_image=png,
        extra_image_mime="image/png",
    )

    # Should still return a cleaned/echoed response without crashing.
    assert result.text.strip()


def test_burn_responses_image_no_image_still_works(tmp_path):
    """Sanity: detail='high' with no image provided works like a plain call."""
    _require_openai_key()
    profile = _responses_profile(tmp_path, image_detail="high")

    from justsayit.postprocess import make_postprocessor
    pp = make_postprocessor(profile)

    result = pp.process_with_reasoning("hello world")

    assert result.text.strip()
