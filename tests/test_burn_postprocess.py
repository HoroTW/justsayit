"""End-to-end burn tests that exercise the real llama-cpp-python stack
against locally cached GGUFs. Skipped by default (pytest config only
collects tests not marked ``burn``); run explicitly with:

    pytest -m burn tests/test_burn_postprocess.py

Rationale: the mock-based unit tests in ``test_postprocess.py`` have
twice let integration bugs ship — first a ``chat_template_kwargs=``
kwarg that ``Llama.create_chat_completion()`` rejects (no ``**kwargs``
in its signature), then a chat-handler lookup that missed the
``_chat_handlers`` dict where GGUF-embedded Jinja templates live
under the magic name ``chat_template.default``. Both would have
blown up the first time these tests ran against a real GGUF.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from justsayit.postprocess import LLMPostprocessor, PostprocessProfile


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
