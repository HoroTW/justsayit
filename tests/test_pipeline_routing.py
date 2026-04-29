"""Tests for pipeline integration of snippets and prefix-router.

These exercise SegmentPipeline.handle() end-to-end with stub
postprocessors and transcribers, so we can assert which path the
pipeline takes (LLM vs bypass) and which profile got used.
"""

from __future__ import annotations

import numpy as np
import pytest

from justsayit.audio import Segment
from justsayit.config import Config
from justsayit.pipeline import SegmentPipeline
from justsayit.postprocess import ProcessResult
from justsayit.snippets import Snippet
from justsayit.transcribe import TranscriberBase


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTranscriber(TranscriberBase):
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return self._text


class _StubOverlay:
    def __init__(self) -> None:
        self.detected: list = []
        self.llm: list = []
        self.linger_calls = 0
        self.hide_calls = 0

    def push_hide(self) -> None:
        self.hide_calls += 1

    def push_detected_text(self, text: str, llm_pending: bool = False) -> None:
        self.detected.append((text, llm_pending))

    def push_llm_text(self, text: str, thought: str = "") -> None:
        self.llm.append((text, thought))

    def push_linger_start(self) -> None:
        self.linger_calls += 1

    def push_clipboard_context_armed(self, armed: bool) -> None:
        pass


class _StubPP:
    """Stub postprocessor that records every call and the texts it sees."""

    def __init__(self, label: str = "default") -> None:
        self.label = label
        self.calls: list[str] = []

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
        self.calls.append(text)
        return ProcessResult(text=f"[{self.label}] {text}")

    def strip_for_paste(self, text: str) -> str:
        return text

    def find_strip_matches(self, text: str) -> list[str]:
        return []


def _seg(duration_s: float = 1.01) -> Segment:
    sr = 16_000
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return Segment(samples=samples, sample_rate=sr, reason="manual")


def _make_pipeline(
    cfg: Config,
    text: str,
    *,
    snippets=None,
    pp: _StubPP | None = None,
    resolve_profile=None,
) -> tuple[SegmentPipeline, _StubOverlay]:
    pl = SegmentPipeline(
        cfg,
        _StubTranscriber(text),
        [],
        None,
        no_paste=True,
        snippets=snippets or [],
    )
    pl.postprocessor = pp
    overlay = _StubOverlay()
    pl.overlay = overlay
    if resolve_profile is not None:
        pl.resolve_profile = resolve_profile
    return pl, overlay


# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------


def test_snippet_bypass_llm_skips_postprocessor(capsys):
    cfg = Config()
    pp = _StubPP("default")
    snip = Snippet(trigger="hello", replacement="HI!", bypass_llm=True)
    pl, overlay = _make_pipeline(
        cfg, "hello", snippets=[snip], pp=pp
    )

    pl.handle(_seg())

    # LLM was skipped
    assert pp.calls == []
    # Replacement was printed verbatim
    assert capsys.readouterr().out.strip() == "HI!"
    # Overlay shows the replacement, no LLM-pending placeholder
    assert overlay.detected == [("HI!", False)]


def test_snippet_without_bypass_passes_replacement_to_llm(capsys):
    cfg = Config()
    pp = _StubPP("default")
    snip = Snippet(trigger="hello", replacement="HI!", bypass_llm=False)
    pl, _ = _make_pipeline(
        cfg, "hello", snippets=[snip], pp=pp
    )

    pl.handle(_seg())

    # LLM saw the replacement, not the original transcription
    assert pp.calls == ["HI!"]
    assert capsys.readouterr().out.strip() == "[default] HI!"


def test_no_snippet_match_falls_through_to_llm(capsys):
    cfg = Config()
    pp = _StubPP("default")
    snip = Snippet(trigger="something else", replacement="x")
    pl, _ = _make_pipeline(
        cfg, "hello world", snippets=[snip], pp=pp
    )

    pl.handle(_seg())

    assert pp.calls == ["hello world"]
    assert capsys.readouterr().out.strip() == "[default] hello world"


def test_expand_snippet_appends_remainder_then_llm(capsys):
    cfg = Config()
    pp = _StubPP("default")
    snip = Snippet(
        trigger="todo", replacement="TODO:", mode="expand", bypass_llm=False
    )
    pl, _ = _make_pipeline(
        cfg, "todo finish the test", snippets=[snip], pp=pp
    )

    pl.handle(_seg())

    assert pp.calls == ["TODO: finish the test"]


# ---------------------------------------------------------------------------
# Prefix router
# ---------------------------------------------------------------------------


def test_prefix_router_disabled_by_default(capsys):
    cfg = Config()
    pp = _StubPP("default")
    pl, _ = _make_pipeline(cfg, "code: print hello", pp=pp)

    pl.handle(_seg())

    # No routing — LLM saw the raw text including the prefix.
    assert pp.calls == ["code: print hello"]


def test_prefix_router_quick_skips_llm(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.quick_skip_llm = True
    pp = _StubPP("default")
    pl, _ = _make_pipeline(cfg, "quick: hello world", pp=pp)

    pl.handle(_seg())

    # LLM bypassed, prefix stripped from output.
    assert pp.calls == []
    assert capsys.readouterr().out.strip() == "hello world"


def test_prefix_router_quick_with_comma_separator(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.quick_skip_llm = True
    pp = _StubPP("default")
    pl, _ = _make_pipeline(cfg, "quick, hello world", pp=pp)

    pl.handle(_seg())

    assert pp.calls == []
    assert capsys.readouterr().out.strip() == "hello world"


def test_prefix_router_routes_to_named_profile(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.prefixes = {"code": "code-cleanup"}

    default_pp = _StubPP("default")
    code_pp = _StubPP("code")

    def _resolve(name: str):
        if name == "code-cleanup":
            return code_pp
        return None

    pl, _ = _make_pipeline(
        cfg, "code: print hello", pp=default_pp, resolve_profile=_resolve
    )

    pl.handle(_seg())

    # Default pp untouched; code pp got the stripped text.
    assert default_pp.calls == []
    assert code_pp.calls == ["print hello"]
    assert capsys.readouterr().out.strip() == "[code] print hello"


def test_prefix_router_unknown_prefix_falls_through(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.prefixes = {"code": "code-cleanup"}
    pp = _StubPP("default")
    pl, _ = _make_pipeline(cfg, "banana: hello", pp=pp)

    pl.handle(_seg())

    # Unknown prefix → no routing, LLM sees full text.
    assert pp.calls == ["banana: hello"]


def test_prefix_router_resolve_profile_returns_none_uses_active(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.prefixes = {"code": "missing-profile"}
    pp = _StubPP("default")

    pl, _ = _make_pipeline(
        cfg,
        "code: print hello",
        pp=pp,
        resolve_profile=lambda name: None,
    )
    pl.handle(_seg())

    # Prefix stripped, but the active pp processes the text since the
    # resolver returned None.
    assert pp.calls == ["print hello"]


def test_prefix_router_then_snippet_match(capsys):
    """Prefix is stripped first, snippet matches the stripped text."""
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.prefixes = {"code": "code-cleanup"}
    pp = _StubPP("default")
    snip = Snippet(trigger="print hello", replacement="print('hi')", bypass_llm=True)
    pl, _ = _make_pipeline(
        cfg,
        "code: print hello",
        snippets=[snip],
        pp=pp,
        resolve_profile=lambda name: _StubPP("code"),
    )

    pl.handle(_seg())

    # Snippet bypass wins → no LLM call at all.
    assert pp.calls == []
    assert capsys.readouterr().out.strip() == "print('hi')"


def test_prefix_router_case_insensitive_keyword(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.quick_skip_llm = True
    pp = _StubPP("default")
    pl, _ = _make_pipeline(cfg, "Quick: hi", pp=pp)

    pl.handle(_seg())

    assert pp.calls == []
    assert capsys.readouterr().out.strip() == "hi"


def test_prefix_router_quick_skip_disabled(capsys):
    cfg = Config()
    cfg.prefix_router.enabled = True
    cfg.prefix_router.quick_skip_llm = False
    pp = _StubPP("default")
    pl, _ = _make_pipeline(cfg, "quick: hello", pp=pp)

    pl.handle(_seg())

    # quick: not in prefixes mapping and quick_skip_llm is off → no route.
    assert pp.calls == ["quick: hello"]
