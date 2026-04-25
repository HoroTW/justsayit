"""Tests for the continue-session feature (session.json + multi-turn history)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from justsayit.audio import Segment
from justsayit.config import Config
from justsayit.pipeline import (
    SegmentPipeline,
    _clear_session,
    _load_session,
    _save_session,
    _session_path,
)
from justsayit.postprocess._processor import PostprocessorBase
from justsayit.postprocess.backend_responses import ResponsesBackend
from justsayit.postprocess._profile import PostprocessProfile, ProcessResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seg(duration_s: float = 1.01) -> Segment:
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="manual")


class _StubTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples, sr) -> str:
        return self._text

    def has_words(self, samples, sr) -> bool:
        return True

    def warmup(self) -> None:
        pass


class _RecordingPostprocessor:
    """Captures arguments passed to process_with_reasoning; returns fixed result."""

    def __init__(self, text: str = "cleaned", session_data: dict | None = None) -> None:
        self._text = text
        self._session_data = session_data
        self.calls: list[dict] = []

    def process_with_reasoning(
        self,
        text: str,
        *,
        extra_context: str = "",
        extra_image=None,
        extra_image_mime: str = "",
        previous_session=None,
        tools=None,
        tool_caller=None,
        assistant_mode: bool = False,
    ) -> ProcessResult:
        self.calls.append({"text": text, "previous_session": previous_session})
        return ProcessResult(text=self._text, session_data=self._session_data)

    def strip_for_paste(self, text: str) -> str:
        return text

    def find_strip_matches(self, text: str) -> list[str]:
        return []


def _make_pipeline(cfg: Config, text: str = "hello", pp=None) -> SegmentPipeline:
    pl = SegmentPipeline(cfg, _StubTranscriber(text), [], None, no_paste=True)
    pl.postprocessor = pp
    return pl


# ---------------------------------------------------------------------------
# Session file helpers
# ---------------------------------------------------------------------------


def test_save_and_load_session(tmp_path):
    fake_cache = tmp_path / "cache"
    with patch("justsayit.pipeline._session_path", return_value=fake_cache / "session.json"):
        data = {"backend": "remote", "prev_messages": [], "ts": 1.0}
        _save_session(data)
        loaded = _load_session()
    assert loaded == data


def test_load_session_returns_none_when_missing(tmp_path):
    with patch("justsayit.pipeline._session_path", return_value=tmp_path / "no_file.json"):
        assert _load_session() is None


def test_clear_session_deletes_file(tmp_path):
    p = tmp_path / "session.json"
    p.write_text("{}", encoding="utf-8")
    with patch("justsayit.pipeline._session_path", return_value=p):
        _clear_session()
    assert not p.exists()


def test_clear_session_tolerates_missing(tmp_path):
    with patch("justsayit.pipeline._session_path", return_value=tmp_path / "no.json"):
        _clear_session()  # must not raise


# ---------------------------------------------------------------------------
# Pipeline: session save / clear
# ---------------------------------------------------------------------------


def test_pipeline_saves_session_when_is_continue(tmp_path):
    cfg = Config()
    session_data = {"backend": "remote", "prev_messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], "ts": 1.0}
    pp = _RecordingPostprocessor(session_data=session_data)
    pl = _make_pipeline(cfg, "hello", pp)

    saved = {}
    with patch("justsayit.pipeline._load_session", return_value=None), \
         patch("justsayit.pipeline._save_session", side_effect=lambda d: saved.update(d)):
        pl.handle(_make_seg(), is_continue=True)

    assert saved["backend"] == "remote"


def test_pipeline_saves_session_when_not_continue(tmp_path):
    """Non-continue calls save the current exchange so a future continue can pick it up."""
    cfg = Config()
    session_data = {"backend": "remote", "prev_messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], "ts": 1.0}
    pp = _RecordingPostprocessor(session_data=session_data)
    pl = _make_pipeline(cfg, "hello", pp)

    saved = {}
    with patch("justsayit.pipeline._save_session", side_effect=lambda d: saved.update(d)):
        pl.handle(_make_seg(), is_continue=False)

    assert saved["backend"] == "remote"


def test_pipeline_passes_previous_session_to_postprocessor():
    cfg = Config()
    prev_session = {"backend": "remote", "prev_messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "ts": 1.0}
    pp = _RecordingPostprocessor()
    pl = _make_pipeline(cfg, "follow up", pp)

    with patch("justsayit.pipeline._load_session", return_value=prev_session), \
         patch("justsayit.pipeline._save_session"):
        pl.handle(_make_seg(), is_continue=True)

    assert len(pp.calls) == 1
    assert pp.calls[0]["previous_session"] is prev_session


def test_pipeline_passes_none_when_not_continue():
    cfg = Config()
    pp = _RecordingPostprocessor()
    pl = _make_pipeline(cfg, "hello", pp)

    with patch("justsayit.pipeline._load_session") as mock_load, \
         patch("justsayit.pipeline._save_session"):
        pl.handle(_make_seg(), is_continue=False)

    mock_load.assert_not_called()
    assert pp.calls[0]["previous_session"] is None


def test_pipeline_does_not_save_on_llm_exception():
    """If the LLM fails the session is left unchanged — don't save, don't clear."""
    cfg = Config()

    class _RaisingPP:
        def process_with_reasoning(self, text, *, extra_context="", extra_image=None, extra_image_mime="", previous_session=None, tools=None, tool_caller=None, assistant_mode=False):
            raise RuntimeError("boom")

        def strip_for_paste(self, text):
            return text

        def find_strip_matches(self, text):
            return []

    pl = _make_pipeline(cfg, "hello", _RaisingPP())

    with patch("justsayit.pipeline._save_session") as mock_save, \
         patch("justsayit.pipeline._clear_session") as mock_clear:
        pl.handle(_make_seg(), is_continue=False)

    mock_save.assert_not_called()
    mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# PostprocessorBase: _build_messages_continued + _format_history_text
# ---------------------------------------------------------------------------


def _make_profile(**kw) -> PostprocessProfile:
    defaults = {
        "system_prompt": "You are a helpful assistant.",
        "system_prompt_file": "",
    }
    defaults.update(kw)
    return PostprocessProfile(**defaults)


class _ConcreteProcessor(PostprocessorBase):
    def _run(self, text, extra_context="", extra_image=None, extra_image_mime="", previous_session=None):
        return ProcessResult(text=text)


def test_format_history_text_basic():
    proc = _ConcreteProcessor(_make_profile())
    msgs = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And Germany?"},
        {"role": "assistant", "content": "Berlin."},
    ]
    result = proc._format_history_text(msgs)
    assert result.startswith("## PREVIOUS SESSION HISTORY")
    assert "User: What is the capital of France?" in result
    assert "Assistant: Paris." in result
    assert "User: And Germany?" in result
    assert "Assistant: Berlin." in result


def test_build_messages_continued_order():
    proc = _ConcreteProcessor(_make_profile())
    prev = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    msgs = proc._build_messages_continued("second", "", prev)
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "first"}
    assert msgs[2] == {"role": "assistant", "content": "answer"}
    assert msgs[3]["role"] == "user"
    assert "second" in msgs[3]["content"]


def test_format_history_text_handles_list_content():
    """List content (image + text blocks) should extract text only, no Python repr."""
    proc = _ConcreteProcessor(_make_profile())
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"}},
        ]},
        {"role": "assistant", "content": "It's a cat."},
    ]
    result = proc._format_history_text(msgs)
    assert "describe this image" in result
    assert "It's a cat." in result
    assert "image_url" not in result  # no raw dict repr


def test_build_messages_with_history_text_goes_into_system_prompt():
    proc = _ConcreteProcessor(_make_profile())
    prev = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    history = proc._format_history_text(prev)
    msgs = proc._build_messages("new question", history_text=history)
    system_content = msgs[0]["content"]
    assert "## PREVIOUS SESSION HISTORY" in system_content
    assert "User: q" in system_content
    assert "Assistant: a" in system_content


# ---------------------------------------------------------------------------
# ResponsesBackend: _canonical_to_responses_input
# ---------------------------------------------------------------------------


def test_canonical_to_responses_input_text_only():
    prev = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    out = ResponsesBackend._canonical_to_responses_input(prev)
    assert out[0] == {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    assert out[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]}


def test_canonical_to_responses_input_with_image():
    prev = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"}},
        ]},
        {"role": "assistant", "content": "It's a cat."},
    ]
    out = ResponsesBackend._canonical_to_responses_input(prev)
    assert out[0]["role"] == "user"
    user_content = out[0]["content"]
    assert {"type": "input_text", "text": "describe this"} in user_content
    assert {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "auto"} in user_content
    assert out[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "It's a cat."}]}


# ---------------------------------------------------------------------------
# RemoteBackend: cross-backend uses _build_messages_continued directly
# ---------------------------------------------------------------------------


def test_remote_cross_backend_uses_build_messages_continued():
    """RemoteBackend must use _build_messages_continued even for cross-backend sessions,
    since prev_messages is always in canonical chat-completions format."""
    from justsayit.postprocess.backend_remote import RemoteBackend
    from justsayit.postprocess._profile import PostprocessProfile
    import unittest.mock as mock

    profile = PostprocessProfile(
        system_prompt="sys",
        system_prompt_file="",
        model="gpt-4o-mini",
        endpoint="http://fake/v1",
        api_key="sk-test",
    )
    backend = RemoteBackend(profile)

    prev_session = {
        "backend": "responses",  # different backend!
        "prev_messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "ts": 1.0,
    }

    captured_messages = []

    def fake_http_post(url, body, headers, **kw):
        captured_messages.extend(body["messages"])
        return {"choices": [{"message": {"content": "ok", "role": "assistant"}}], "usage": {}}

    with mock.patch("justsayit.postprocess.backend_remote._http_post", side_effect=fake_http_post):
        backend._run("follow up", previous_session=prev_session)

    # Should have: system, user(question), assistant(answer), user(follow up)
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1] == {"role": "user", "content": "question"}
    assert captured_messages[2] == {"role": "assistant", "content": "answer"}
    assert captured_messages[3]["role"] == "user"


# ---------------------------------------------------------------------------
# Session storage: clipboard text + images in prev_messages
# ---------------------------------------------------------------------------


def _fake_remote_profile() -> PostprocessProfile:
    return PostprocessProfile(
        system_prompt="sys",
        system_prompt_file="",
        model="gpt-4o-mini",
        endpoint="http://fake/v1",
        api_key="sk-test",
    )


def _fake_responses_response() -> dict:
    return {
        "id": "resp_123",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_extra_context_stored_in_remote_prev_messages():
    """Clipboard text (extra_context) must appear in prev_messages so future
    continue turns know what context the user provided in that turn."""
    import unittest.mock as mock
    from justsayit.postprocess.backend_remote import RemoteBackend

    backend = RemoteBackend(_fake_remote_profile())
    clipboard = "def foo():\n    pass"

    with mock.patch(
        "justsayit.postprocess.backend_remote._http_post",
        return_value={"choices": [{"message": {"content": "ok", "role": "assistant"}}], "usage": {}},
    ):
        result = backend._run("please refactor this", extra_context=clipboard)

    user_content = result.session_data["prev_messages"][0]["content"]
    assert isinstance(user_content, list), "expected list content when extra_context provided"
    all_text = " ".join(b.get("text", "") for b in user_content if b.get("type") == "text")
    assert "def foo" in all_text, f"clipboard text missing from prev_messages: {user_content!r}"


def test_extra_context_stored_in_responses_prev_messages():
    """Same as above but for the Responses API backend."""
    import unittest.mock as mock

    profile = PostprocessProfile(
        system_prompt="sys",
        system_prompt_file="",
        model="gpt-5.4-mini",
        endpoint="http://fake/v1",
        api_key="sk-test",
    )
    backend = ResponsesBackend(profile)
    clipboard = "some clipboard content"

    with mock.patch(
        "justsayit.postprocess.backend_responses._http_post",
        return_value=_fake_responses_response(),
    ):
        result = backend._run("help me with this", extra_context=clipboard)

    user_content = result.session_data["prev_messages"][0]["content"]
    assert isinstance(user_content, list), "expected list content when extra_context provided"
    all_text = " ".join(b.get("text", "") for b in user_content if b.get("type") == "text")
    assert "clipboard content" in all_text, f"clipboard text missing from responses prev_messages: {user_content!r}"


def test_local_backend_stores_image_in_prev_messages():
    """LocalBackend stores extra_image in prev_messages (canonical image_url format)
    even though the local model cannot see it — so switching to a vision backend
    later preserves the image."""
    import unittest.mock as mock
    from justsayit.postprocess.backend_local import LocalBackend

    backend = LocalBackend(PostprocessProfile(system_prompt="sys", system_prompt_file=""))
    backend._llm = mock.MagicMock()
    backend._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    result = backend._run("describe this", extra_image=png, extra_image_mime="image/png")

    user_content = result.session_data["prev_messages"][0]["content"]
    assert isinstance(user_content, list), "expected list content (text + image) in local session"
    assert any(b.get("type") == "image_url" for b in user_content), (
        f"image_url block missing from local backend prev_messages: {user_content!r}"
    )


def test_local_to_remote_cross_backend_image_visible():
    """After local → remote backend switch, the remote backend must include
    the image from the local session in the messages sent to the API."""
    import unittest.mock as mock
    from justsayit.postprocess.backend_local import LocalBackend
    from justsayit.postprocess.backend_remote import RemoteBackend

    profile = _fake_remote_profile()
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    # Turn 1: local backend with image
    local = LocalBackend(profile)
    local._llm = mock.MagicMock()
    local._llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ack"}}]}
    turn1 = local._run("see this image", extra_image=png, extra_image_mime="image/png")

    # Sanity: local session has the image
    assert any(
        b.get("type") == "image_url"
        for b in turn1.session_data["prev_messages"][0]["content"]
    )

    # Turn 2: remote backend, cross-backend continue
    remote = RemoteBackend(profile)
    captured: list[dict] = []

    with mock.patch(
        "justsayit.postprocess.backend_remote._http_post",
        side_effect=lambda url, body, headers, **kw: (
            captured.extend(body["messages"]) or
            {"choices": [{"message": {"content": "it's a png", "role": "assistant"}}], "usage": {}}
        ),
    ):
        remote._run("now describe it", previous_session=turn1.session_data)

    # Image must be visible in the messages forwarded to the remote API
    all_blocks = [
        b
        for msg in captured
        for b in (msg["content"] if isinstance(msg["content"], list) else [])
    ]
    assert any(b.get("type") == "image_url" for b in all_blocks), (
        "image_url missing from remote API messages after local→remote switch"
    )


# ---------------------------------------------------------------------------
# Two-turn image continuations: no raw bytes leaking into text fields
# ---------------------------------------------------------------------------

# Minimal distinct fake PNGs (different fill byte so base64 differs)
_PNG1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_PNG2 = b"\x89PNG\r\n\x1a\n" + b"\xff" * 32


def _is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:image/") and ";base64," in s


def _has_no_binary(s: str) -> bool:
    """Check that a string contains no raw binary bytes (only printable ASCII)."""
    return all(0x20 <= ord(c) <= 0x7E for c in s[:200])


def _url_from_session(content_or_msg) -> str:
    """Extract first image_url data URL from a prev_messages user entry."""
    content = content_or_msg if isinstance(content_or_msg, list) else content_or_msg.get("content", [])
    return next(b["image_url"]["url"] for b in content if b.get("type") == "image_url")


def test_remote_continue_second_image_preserves_both_as_data_urls():
    """Turn 1 (remote): image1 stored in session.
    Turn 2 (remote, continue): image2 sent while loading turn 1 session.
    Both images must be proper base64 data URLs in the session and in the
    API messages — no raw bytes in any text field."""
    import base64
    import json
    import unittest.mock as mock
    from justsayit.postprocess.backend_remote import RemoteBackend

    backend = RemoteBackend(_fake_remote_profile())

    # Turn 1
    with mock.patch(
        "justsayit.postprocess.backend_remote._http_post",
        return_value={"choices": [{"message": {"content": "ack", "role": "assistant"}}], "usage": {}},
    ):
        turn1 = backend._run("image one", extra_image=_PNG1, extra_image_mime="image/png")

    user1_content = turn1.session_data["prev_messages"][0]["content"]
    assert isinstance(user1_content, list)
    img_url_1 = next(b["image_url"]["url"] for b in user1_content if b.get("type") == "image_url")
    assert _is_data_url(img_url_1), f"turn 1 image not a data URL: {img_url_1[:80]!r}"
    assert _has_no_binary(img_url_1), "binary chars in turn 1 image URL"

    # JSON round-trip (same as pipeline save → load)
    session = json.loads(json.dumps(turn1.session_data))

    # Turn 2 — capture messages sent to API
    captured: list[dict] = []

    def fake_post(url, body, headers, **kw):
        captured.extend(body["messages"])
        return {"choices": [{"message": {"content": "both seen", "role": "assistant"}}], "usage": {}}

    with mock.patch("justsayit.postprocess.backend_remote._http_post", side_effect=fake_post):
        turn2 = backend._run("image two", extra_image=_PNG2, extra_image_mime="image/png",
                             previous_session=session)

    # Session after turn 2 must have 4 entries: u1 a1 u2 a2
    prev = turn2.session_data["prev_messages"]
    assert len(prev) == 4, f"expected 4 prev_messages, got {len(prev)}"

    # Both user messages must have their images as data URLs
    for idx in (0, 2):
        content = prev[idx]["content"]
        assert isinstance(content, list), f"prev_messages[{idx}] content should be a list"
        img_blocks = [b for b in content if b.get("type") == "image_url"]
        assert img_blocks, f"no image_url block in prev_messages[{idx}]"
        url = img_blocks[0]["image_url"]["url"]
        assert _is_data_url(url), f"prev_messages[{idx}] image not a data URL: {url[:80]!r}"
        assert _has_no_binary(url), f"binary chars in prev_messages[{idx}] image URL"

    # The two stored images must encode different bytes
    assert _url_from_session(prev[0]) != _url_from_session(prev[2]), \
        "turn 1 and turn 2 images must differ"

    b64_png1 = base64.b64encode(_PNG1).decode()
    b64_png2 = base64.b64encode(_PNG2).decode()
    assert b64_png1 in _url_from_session(prev[0]), "turn 1 image data doesn't match PNG1"
    assert b64_png2 in _url_from_session(prev[2]), "turn 2 image data doesn't match PNG2"

    # API messages must contain image_url blocks (not raw bytes squeezed into text)
    api_image_blocks = [
        b
        for msg in captured
        for b in (msg["content"] if isinstance(msg["content"], list) else [])
        if b.get("type") == "image_url"
    ]
    assert len(api_image_blocks) == 2, \
        f"expected 2 image_url blocks in API call, got {len(api_image_blocks)}"
    for block in api_image_blocks:
        assert _is_data_url(block["image_url"]["url"])

    # No text block may contain raw PNG magic bytes
    for msg in captured:
        content = msg["content"]
        if isinstance(content, str):
            assert "\x89PNG" not in content, f"raw PNG bytes in text message: {content[:60]!r}"
        elif isinstance(content, list):
            for b in content:
                if b.get("type") == "text":
                    assert "\x89PNG" not in b.get("text", ""), \
                        f"raw PNG bytes in text block: {b['text'][:60]!r}"


def test_responses_continue_second_image_both_stored_as_data_urls():
    """Turn 1 (responses, same-backend chain): image1 → session.
    Turn 2 (responses, same-backend chain): image2 added.
    Both images must be proper data URLs in the session — no raw bytes."""
    import base64
    import unittest.mock as mock

    profile = PostprocessProfile(
        system_prompt="sys", system_prompt_file="",
        model="gpt-5.4-mini", endpoint="http://fake/v1", api_key="sk-test",
    )
    backend = ResponsesBackend(profile)

    def fake_post(url, body, headers, **kw):
        return _fake_responses_response()

    # Turn 1
    with mock.patch("justsayit.postprocess.backend_responses._http_post", side_effect=fake_post):
        turn1 = backend._run("image one", extra_image=_PNG1, extra_image_mime="image/png")

    img_url_1 = _url_from_session(turn1.session_data["prev_messages"][0])
    assert _is_data_url(img_url_1)
    assert _has_no_binary(img_url_1)

    # Turn 2 — same backend (uses response_id chain)
    with mock.patch("justsayit.postprocess.backend_responses._http_post", side_effect=fake_post):
        turn2 = backend._run("image two", extra_image=_PNG2, extra_image_mime="image/png",
                             previous_session=turn1.session_data)

    prev = turn2.session_data["prev_messages"]
    assert len(prev) == 4, f"expected 4 prev_messages, got {len(prev)}"

    b64_png1 = base64.b64encode(_PNG1).decode()
    b64_png2 = base64.b64encode(_PNG2).decode()

    img1_url = _url_from_session(prev[0])
    img2_url = _url_from_session(prev[2])
    assert _is_data_url(img1_url) and _has_no_binary(img1_url)
    assert _is_data_url(img2_url) and _has_no_binary(img2_url)
    assert b64_png1 in img1_url, "turn 1 image doesn't match PNG1"
    assert b64_png2 in img2_url, "turn 2 image doesn't match PNG2"
    assert img1_url != img2_url, "turn 1 and turn 2 images must differ"
