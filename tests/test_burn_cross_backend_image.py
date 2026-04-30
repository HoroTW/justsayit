"""Burn tests for cross-backend session continuation with images.

Verifies that images stored in session.json via _canonical_to_responses_input
(Responses API) and _build_messages (Chat Completions) are actually
visible to the model in turn 2 after a backend switch.

    pytest -m burn tests/test_burn_cross_backend_image.py

Cost per run: 4 API calls (2 turns × 2 directions), small PNG at detail=low.
~$0.002–0.01 depending on model pricing.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from justsayit.config import resolve_secret
from justsayit.postprocess import load_profile, make_postprocessor

pytestmark = pytest.mark.burn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_openai_key() -> str:
    key = resolve_secret("", "OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — skipping cross-backend image burn tests")
    return key


def _make_test_png(width: int = 200, height: int = 50) -> bytes:
    """Minimal valid PNG: white background with thick black border."""
    def _pack_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _pack_chunk(b"IHDR", ihdr_data)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            on_border = x < 2 or x >= width - 2 or y < 2 or y >= height - 2
            px = b"\x00\x00\x00" if on_border else b"\xff\xff\xff"
            row += px
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    idat = _pack_chunk(b"IDAT", zlib.compress(raw))
    iend = _pack_chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _responses_profile(tmp_path: Path) -> object:
    p = tmp_path / "responses.toml"
    p.write_text(
        'base = "responses"\n'
        'endpoint = "https://api.openai.com/v1"\n'
        'model = "gpt-5.4-mini"\n'
        'api_key_env = "OPENAI_API_KEY"\n'
        'system_prompt = "You are a helpful assistant. Follow the user\'s instructions exactly."\n'
        "max_tokens = 128\n"
        "request_timeout = 30.0\n"
        "remote_retries = 0\n"
        'image_detail = "low"\n',
        encoding="utf-8",
    )
    return load_profile(str(p))


def _remote_profile(tmp_path: Path) -> object:
    p = tmp_path / "remote.toml"
    p.write_text(
        'base = "remote"\n'
        'endpoint = "https://api.openai.com/v1"\n'
        'model = "gpt-4o-mini"\n'
        'api_key_env = "OPENAI_API_KEY"\n'
        'system_prompt = "You are a helpful assistant. Follow the user\'s instructions exactly."\n'
        "max_tokens = 128\n"
        "request_timeout = 30.0\n"
        "remote_retries = 0\n"
        'image_detail = "low"\n',
        encoding="utf-8",
    )
    return load_profile(str(p))


def _mentions_image_content(text: str) -> bool:
    """Return True if the response describes visual content of the test image."""
    keywords = {
        "image", "picture", "photo", "bild", "foto",
        "white", "black", "border", "rectangle", "background",
        "weiß", "schwarz", "rahmen", "rechteck", "hintergrund",
        "png", "graphic",
    }
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_burn_responses_to_remote_cross_backend_image(tmp_path):
    """Turn 1 (Responses API): image sent, model told NOT to describe it yet.
    Turn 2 (Chat Completions, cross-backend continue): model asked to now describe
    the image from turn 1. Verifies _build_messages injects the prev_messages."""
    _require_openai_key()
    png = _make_test_png()

    # Turn 1: Responses backend — image attached, model holds off describing it
    resp_pp = make_postprocessor(_responses_profile(tmp_path))
    turn1 = resp_pp.process_with_reasoning(
        "I am sending you an image. Do NOT describe it — just say 'acknowledged'.",
        extra_image=png,
        extra_image_mime="image/png",
    )
    assert turn1.text.strip(), "turn 1 returned empty"
    assert turn1.session_data is not None, "turn 1 produced no session_data"
    assert turn1.session_data.get("prev_messages"), "turn 1 session has no prev_messages"

    # Confirm image is in session history in canonical format
    prev_msgs = turn1.session_data["prev_messages"]
    user_msg = prev_msgs[0]
    assert isinstance(user_msg["content"], list), "expected list content (text + image) in session"
    assert any(b.get("type") == "image_url" for b in user_msg["content"]), (
        "image_url block missing from Responses API session history"
    )

    # Turn 2: Chat Completions backend, cross-backend continue — ask for description
    remote_pp = make_postprocessor(_remote_profile(tmp_path))
    turn2 = remote_pp.process_with_reasoning(
        "Now describe the image from the previous message.",
        previous_session=turn1.session_data,
    )
    assert turn2.text.strip(), "turn 2 returned empty"
    assert _mentions_image_content(turn2.text), (
        f"turn 2 (remote, cross-backend) should describe the image but got: {turn2.text!r}"
    )


def test_burn_remote_to_responses_cross_backend_image(tmp_path):
    """Turn 1 (Chat Completions): image sent, model told NOT to describe it yet.
    Turn 2 (Responses API, cross-backend continue): model asked to now describe
    the image from turn 1. Verifies _canonical_to_responses_input converts correctly."""
    _require_openai_key()
    png = _make_test_png()

    # Turn 1: Chat Completions backend — image attached, model holds off
    remote_pp = make_postprocessor(_remote_profile(tmp_path))
    turn1 = remote_pp.process_with_reasoning(
        "I am sending you an image. Do NOT describe it — just say 'acknowledged'.",
        extra_image=png,
        extra_image_mime="image/png",
    )
    assert turn1.text.strip(), "turn 1 returned empty"
    assert turn1.session_data is not None, "turn 1 produced no session_data"
    assert turn1.session_data.get("prev_messages"), "turn 1 session has no prev_messages"

    # Confirm image is in session history in canonical format
    prev_msgs = turn1.session_data["prev_messages"]
    user_msg = prev_msgs[0]
    assert isinstance(user_msg["content"], list), "expected list content (text + image) in session"
    assert any(b.get("type") == "image_url" for b in user_msg["content"]), (
        "image_url block missing from Chat Completions session history"
    )

    # Turn 2: Responses backend, cross-backend continue — ask for description
    resp_pp = make_postprocessor(_responses_profile(tmp_path))
    turn2 = resp_pp.process_with_reasoning(
        "Now describe the image from the previous message.",
        previous_session=turn1.session_data,
    )
    assert turn2.text.strip(), "turn 2 returned empty"
    assert _mentions_image_content(turn2.text), (
        f"turn 2 (responses, cross-backend) should describe the image but got: {turn2.text!r}"
    )


def test_burn_three_turn_responses_remote_responses(tmp_path):
    """Three turns alternating backends, each turn 1+2 adds an image:
    Turn 1 (Responses): image1, don't describe
    Turn 2 (Remote, cross-backend): image2, don't describe
    Turn 3 (Responses, cross-backend): describe both images from turns 1+2."""
    _require_openai_key()
    png1 = _make_test_png(width=200, height=50)
    png2 = _make_test_png(width=100, height=100)

    # Turn 1: Responses
    turn1 = make_postprocessor(_responses_profile(tmp_path)).process_with_reasoning(
        "I am sending you image A. Do NOT describe it — just say 'acknowledged A'.",
        extra_image=png1, extra_image_mime="image/png",
    )
    assert turn1.session_data, "turn 1 no session_data"

    # Turn 2: Remote (cross-backend), also sends an image
    turn2 = make_postprocessor(_remote_profile(tmp_path)).process_with_reasoning(
        "I am sending you image B. Do NOT describe it — just say 'acknowledged B'.",
        extra_image=png2, extra_image_mime="image/png",
        previous_session=turn1.session_data,
    )
    assert turn2.session_data, "turn 2 no session_data"
    assert len(turn2.session_data["prev_messages"]) == 4, (
        f"expected 4 prev_messages after turn 2, got {len(turn2.session_data['prev_messages'])}"
    )

    # Turn 3: Responses (cross-backend), asks about both images
    turn3 = make_postprocessor(_responses_profile(tmp_path)).process_with_reasoning(
        "Now describe image A and image B from the previous messages.",
        previous_session=turn2.session_data,
    )
    assert turn3.text.strip(), "turn 3 returned empty"
    assert _mentions_image_content(turn3.text), (
        f"turn 3 (responses, 3-turn) should describe images but got: {turn3.text!r}"
    )


def test_burn_three_turn_remote_responses_remote(tmp_path):
    """Three turns alternating backends, each turn 1+2 adds an image:
    Turn 1 (Remote): image1, don't describe
    Turn 2 (Responses, cross-backend): image2, don't describe
    Turn 3 (Remote, cross-backend): describe both images from turns 1+2."""
    _require_openai_key()
    png1 = _make_test_png(width=200, height=50)
    png2 = _make_test_png(width=100, height=100)

    # Turn 1: Remote
    turn1 = make_postprocessor(_remote_profile(tmp_path)).process_with_reasoning(
        "I am sending you image A. Do NOT describe it — just say 'acknowledged A'.",
        extra_image=png1, extra_image_mime="image/png",
    )
    assert turn1.session_data, "turn 1 no session_data"

    # Turn 2: Responses (cross-backend), also sends an image
    turn2 = make_postprocessor(_responses_profile(tmp_path)).process_with_reasoning(
        "I am sending you image B. Do NOT describe it — just say 'acknowledged B'.",
        extra_image=png2, extra_image_mime="image/png",
        previous_session=turn1.session_data,
    )
    assert turn2.session_data, "turn 2 no session_data"
    assert len(turn2.session_data["prev_messages"]) == 4, (
        f"expected 4 prev_messages after turn 2, got {len(turn2.session_data['prev_messages'])}"
    )

    # Turn 3: Remote (cross-backend), asks about both images
    turn3 = make_postprocessor(_remote_profile(tmp_path)).process_with_reasoning(
        "Now describe image A and image B from the previous messages.",
        previous_session=turn2.session_data,
    )
    assert turn3.text.strip(), "turn 3 returned empty"
    assert _mentions_image_content(turn3.text), (
        f"turn 3 (remote, 3-turn) should describe images but got: {turn3.text!r}"
    )
