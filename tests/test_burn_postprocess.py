"""End-to-end burn tests that exercise the real llama-cpp-python stack
against locally cached GGUFs, plus a real OpenAI HTTP smoke test.
Skipped by default (pytest config only collects tests not marked
``burn``); run explicitly with:

    pytest -m burn tests/test_burn_postprocess.py

Rationale: the mock-based unit tests in ``test_postprocess.py`` have
let integration bugs ship more than once — a ``chat_template_kwargs=``
kwarg that ``Llama.create_chat_completion()`` rejects (no ``**kwargs``
in its signature), a chat-handler lookup that missed the
``_chat_handlers`` dict where GGUF-embedded Jinja templates live
under the magic name ``chat_template.default``, and an OpenAI HTTP 400
because the same ``chat_template_kwargs`` field tripped OpenAI's
strict body validation. All three would have blown up the first time
these tests ran against the real backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from justsayit.config import resolve_secret
from justsayit.postprocess import (
    LLMPostprocessor,
    PostprocessProfile,
    load_profile,
)


MODELS_DIR = Path("~/.cache/justsayit/models/llm").expanduser()


pytestmark = pytest.mark.burn


def _require_gguf(name: str) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        pytest.skip(f"missing GGUF: {path} (run `justsayit setup-llm`)")
    return path


@pytest.fixture(scope="module")
def _llama_available():
    pytest.importorskip("llama_cpp")


def _mini_profile(model_path: Path, **overrides) -> PostprocessProfile:
    # Tight generation budget — we only care about the handshake
    # (template render + a few tokens back), not quality.
    return PostprocessProfile(
        model_path=str(model_path),
        system_prompt="Reply with exactly one short sentence.",
        max_tokens=32,
        temperature=0.0,
        n_ctx=2048,
        **overrides,
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-4-E4B-it-Q4_K_M.gguf",
        "Qwen3.5-0.8B-Q4_K_M.gguf",
    ],
)
def test_burn_process_with_default_thinking_kwargs(_llama_available, model_name):
    """Real end-to-end: load GGUF, install chat_template_kwargs
    wrapper, run a full ``process()`` call. Regression target: the
    production crash where ``chat_template.default`` wasn't resolvable
    via the static handler registry."""
    model_path = _require_gguf(model_name)
    profile = _mini_profile(model_path)
    # Default built-in profile has chat_template_kwargs={"enable_thinking": True};
    # Gemma's template ignores it, Qwen 3.5's honours it — both must not crash.
    assert profile.chat_template_kwargs == {"enable_thinking": True}

    pp = LLMPostprocessor(profile)
    result = pp.process("hello world")

    assert isinstance(result, str)
    assert result.strip(), "model returned empty string"


def test_burn_process_with_empty_template_kwargs(_llama_available):
    """Handler wrapping is a no-op when chat_template_kwargs is empty —
    must still work end-to-end."""
    model_path = _require_gguf("gemma-4-E4B-it-Q4_K_M.gguf")
    profile = _mini_profile(model_path, chat_template_kwargs={})

    pp = LLMPostprocessor(profile)
    result = pp.process("hello")

    assert result.strip()


def test_burn_qwen_thinking_toggle(_llama_available):
    """Qwen 3.5's template reads ``enable_thinking`` — flipping it
    should change the prompt the model sees, even if we can't assert
    on the final text. The important bit is that neither value
    crashes, proving the kwarg genuinely reaches the Jinja template."""
    model_path = _require_gguf("Qwen3.5-0.8B-Q4_K_M.gguf")

    for flag in (True, False):
        profile = _mini_profile(
            model_path, chat_template_kwargs={"enable_thinking": flag}
        )
        pp = LLMPostprocessor(profile)
        result = pp.process("say hi")
        assert result.strip(), f"empty result with enable_thinking={flag}"


# ---------------------------------------------------------------------------
# Real OpenAI HTTP smoke test
# ---------------------------------------------------------------------------


def _require_openai_key() -> str:
    key = resolve_secret("", "OPENAI_API_KEY")
    if not key:
        pytest.skip(
            "OPENAI_API_KEY not set (env or ~/.config/justsayit/.env) — "
            "skipping real-OpenAI burn test"
        )
    return key


_OPENAI_PROFILE_TOML_BASE = (
    'base = "remote"\n'
    'endpoint = "https://api.openai.com/v1"\n'
    'model = "gpt-4o-mini"\n'
    'api_key_env = "OPENAI_API_KEY"\n'
    'system_prompt = "Reply with exactly one short sentence."\n'
    "max_tokens = 32\n"
    "temperature = 0.0\n"
    "request_timeout = 30.0\n"
    "remote_retries = 0\n"
)


def _write_openai_profile(tmp_path: Path, extra_toml: str = "") -> PostprocessProfile:
    """Write a temp profile and load it via ``load_profile`` — this is
    the real production codepath, which merges in ``remote-defaults.toml``
    before constructing the dataclass. Direct ``PostprocessProfile(...)``
    would skip that merge and inherit the ``builtin`` dataclass defaults,
    hiding the exact bug we're testing for."""
    p = tmp_path / "openai-burn.toml"
    p.write_text(_OPENAI_PROFILE_TOML_BASE + extra_toml, encoding="utf-8")
    return load_profile(str(p))


def test_burn_openai_default_profile_round_trips(tmp_path):
    """Real OpenAI /chat/completions handshake using the same defaults
    a freshly-installed ``openai-cleanup`` profile would produce.

    Regression target: 0.13.7 shipped ``chat_template_kwargs =
    {enable_thinking = true}`` in remote-defaults.toml, which OpenAI
    rejects with HTTP 400 ("Unrecognized request argument supplied:
    chat_template_kwargs"). The fix is the {} default; this test is
    what catches a regression of that decision."""
    _require_openai_key()
    profile = _write_openai_profile(tmp_path)
    # Critical: remote default must NOT send chat_template_kwargs.
    assert profile.chat_template_kwargs == {}, (
        "remote default leaked chat_template_kwargs — OpenAI will 400"
    )

    pp = LLMPostprocessor(profile)
    result = pp.process("hello world")

    assert isinstance(result, str)
    assert result.strip(), "OpenAI returned empty string"


def test_burn_openai_explicit_chat_template_kwargs_400s(tmp_path):
    """Confirms why the default has to be {}: opting in to
    chat_template_kwargs against real OpenAI does in fact raise
    HTTP 400. If OpenAI ever starts silently dropping unknown body
    fields again, this test will start failing — at which point we
    can re-evaluate the default."""
    _require_openai_key()
    profile = _write_openai_profile(
        tmp_path,
        extra_toml="chat_template_kwargs = { enable_thinking = true }\n",
    )
    assert profile.chat_template_kwargs == {"enable_thinking": True}

    pp = LLMPostprocessor(profile)
    with pytest.raises(RuntimeError, match=r"HTTP 400"):
        pp.process("hello")
