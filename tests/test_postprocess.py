"""Tests for LLM postprocessing — profile loading, config integration, and
the LLMPostprocessor's process() method (using a mock llama_cpp.Llama)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from justsayit.config import (
    Config,
    load_config,
    render_config_toml,
)
from justsayit.postprocess import (
    KNOWN_LLM_MODELS,
    LLMPostprocessor,
    PostprocessProfile,
    _CLEANUP_PROFILE_TOML,
    _DYNAMIC_CONTEXT_SCRIPT,
    _FUN_PROFILE_TOML,
    _load_prompt,
    context_file_path,
    dynamic_context_script_path,
    ensure_context_file,
    ensure_default_profile,
    ensure_dynamic_context_script,
    ensure_fun_profile,
    find_hf_q4_filename,
    load_context_sidecar,
    load_profile,
    profiles_dir,
)

# Convenience: read the shipped prompt files exactly the way the
# postprocess module does. Tests assert on prompt content, so we want
# the same text the runtime sees.
_DEFAULT_SYSTEM_PROMPT = _load_prompt("cleanup_local.md")
_REMOTE_CLEANUP_SYSTEM_PROMPT = _load_prompt("cleanup_remote.md")


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


def test_load_profile_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    profile_dir = tmp_path / "postprocess"
    profile_dir.mkdir()
    (profile_dir / "mymodel.toml").write_text(
        '[model]\nmodel_path = "/fake/model.gguf"\ntemperature = 0.05\n',
        encoding="utf-8",
    )
    # TOML has no [model] section — fields are at top level
    (profile_dir / "mymodel.toml").write_text(
        'model_path = "/fake/model.gguf"\ntemperature = 0.05\n',
        encoding="utf-8",
    )
    profile = load_profile("mymodel")
    assert profile.model_path == "/fake/model.gguf"
    assert profile.temperature == pytest.approx(0.05)


def test_load_profile_by_path(tmp_path):
    p = tmp_path / "custom.toml"
    p.write_text(
        'model_path = "/tmp/model.gguf"\nn_gpu_layers = 0\n',
        encoding="utf-8",
    )
    profile = load_profile(str(p))
    assert profile.model_path == "/tmp/model.gguf"
    assert profile.n_gpu_layers == 0


def test_load_profile_unknown_keys_ignored(tmp_path):
    p = tmp_path / "extra.toml"
    p.write_text(
        'model_path = "/x.gguf"\nfuture_option = 99\n',
        encoding="utf-8",
    )
    profile = load_profile(str(p))  # must not raise
    assert profile.model_path == "/x.gguf"


def test_load_profile_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent")


def test_load_profile_defaults_are_applied(tmp_path):
    p = tmp_path / "minimal.toml"
    p.write_text('model_path = "/x.gguf"\n', encoding="utf-8")
    profile = load_profile(str(p))
    assert profile.temperature == pytest.approx(0.08)
    assert profile.n_gpu_layers == -1
    assert profile.max_tokens == 4096
    assert "{text}" in profile.user_template


def test_ensure_default_profile_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    path = ensure_default_profile()
    assert path.exists()
    assert path.name == "gemma4-cleanup.toml"
    content = path.read_text(encoding="utf-8")
    # The recommended cleanup profile picks the builtin backend and
    # references the canonical prompt by file rather than embedding it.
    assert 'base = "builtin"' in content
    assert "system_prompt_file" in content
    assert "temperature" in content
    assert "cleanup_local.md" in content


def test_ensure_default_profiles_writes_all_four(tmp_path, monkeypatch):
    """`init` ships four profiles side-by-side: the recommended local
    Gemma cleanup profile, the playful emoji sibling, the OpenAI-
    compatible endpoint variant, and the Ollama-served-Gemma example.
    All four must exist after a single call so the tray menu has them
    to offer."""
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    from justsayit.postprocess import ensure_default_profiles

    cleanup, fun, openai, ollama_gemma = ensure_default_profiles()
    assert cleanup.name == "gemma4-cleanup.toml"
    assert fun.name == "gemma4-fun.toml"
    assert openai.name == "openai-cleanup.toml"
    assert ollama_gemma.name == "ollama-gemma.toml"
    assert (
        cleanup.exists() and fun.exists() and openai.exists() and ollama_gemma.exists()
    )
    fun_text = fun.read_text(encoding="utf-8")
    # Fun profile points users back at cleanup and disables the strip
    # regex (the fun prompt has no <|think|> channel).
    assert "gemma4-cleanup" in fun_text
    assert 'paste_strip_regex = ""' in fun_text
    assert 'system_prompt_file = "fun.md"' in fun_text


def test_openai_profile_template_has_base_endpoint_and_model_uncommented(
    tmp_path, monkeypatch
):
    """The openai-cleanup.toml ships with `base = "remote"`, `endpoint`,
    and `model` already uncommented (they're what makes this the openai
    variant); everything else stays commented so users only override
    what they need. The system prompt comes from `cleanup_remote.md`
    via the `remote-defaults.toml` overlay — no auto-swap, no
    embedded copy."""
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    from justsayit.postprocess import ensure_openai_profile, load_profile

    p = ensure_openai_profile()
    text = p.read_text(encoding="utf-8")
    # Defining keys live as bare assignments at the top of the file.
    assert '\nbase = "remote"' in text
    assert '\nendpoint = "https://api.openai.com/v1"' in text
    assert '\nmodel = "gpt-4o-mini"' in text
    assert "# remote_retries = 3" in text
    assert "# remote_retry_delay_seconds = 1.0" in text
    # No embedded system prompt block — `system_prompt_file` is the
    # source of truth, and it's set in remote-defaults.toml.
    assert "\nsystem_prompt =" not in text
    # Loads cleanly + carries the right backend defaults.
    profile = load_profile("openai-cleanup")
    assert profile.base == "remote"
    assert profile.endpoint == "https://api.openai.com/v1"
    assert profile.model == "gpt-4o-mini"
    assert profile.api_key == ""  # falls through to env / .env
    # System prompt: file reference picked up from remote-defaults.toml.
    assert profile.system_prompt_file == "cleanup_remote.md"
    assert profile.system_prompt == ""  # inline override empty


def test_ollama_gemma_profile_demonstrates_orthogonal_backend_and_prompt(
    tmp_path, monkeypatch
):
    """Backend (`base = "remote"`) and prompt (`system_prompt_file =
    "cleanup_local.md"`) are independent. The ollama-gemma profile is
    the worked example: HTTP backend (Ollama), Gemma's <|think|>
    cleanup prompt, channel stripper re-enabled."""
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    from justsayit.postprocess import ensure_ollama_gemma_profile, load_profile

    p = ensure_ollama_gemma_profile()
    profile = load_profile("ollama-gemma")
    assert profile.base == "remote"
    assert profile.endpoint == "http://localhost:11434/v1"
    assert profile.system_prompt_file == "cleanup_local.md"
    # Re-enables the channel stripper that remote-defaults.toml leaves blank.
    assert profile.paste_strip_regex == r"<\|channel>thought(.*?)<channel\|>"


def test_ensure_default_profile_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    p1 = ensure_default_profile()
    original = p1.read_text(encoding="utf-8")
    p2 = ensure_default_profile()
    assert p1 == p2
    assert p2.read_text(encoding="utf-8") == original  # not overwritten


# ---------------------------------------------------------------------------
# LLMPostprocessor (mock llama_cpp)
# ---------------------------------------------------------------------------


def _make_mock_llama(response_text: str):
    """Return a mock Llama instance that returns *response_text*."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response_text}}]
    }
    return llm


def test_process_returns_cleaned_text():
    profile = PostprocessProfile(model_path="/fake/model.gguf")
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("Cleaned text.")
    result = pp.process("ähm cleaned text")
    assert result == "Cleaned text."


def test_process_falls_back_on_empty_response():
    profile = PostprocessProfile(model_path="/fake/model.gguf")
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("")  # empty model output
    result = pp.process("original text")
    assert result == "original text"


def test_process_uses_system_prompt():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="My custom prompt.",
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("ok")
    pp.process("some text")
    call_kwargs = pp._llm.create_chat_completion.call_args
    messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][0]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] == "My custom prompt."


def test_default_prompts_include_full_emoji_collapse_guidance():
    for prompt in (_DEFAULT_SYSTEM_PROMPT, _REMOTE_CLEANUP_SYSTEM_PROMPT):
        assert "collapse the WHOLE phrase to only the emoji" in prompt
        assert "Fragen da Emoji" in prompt
        assert "not `Fragen da 🤔`" in prompt


def test_process_uses_temperature_and_max_tokens():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        temperature=0.03,
        max_tokens=128,
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("out")
    pp.process("in")
    call_kwargs = pp._llm.create_chat_completion.call_args[1]
    assert call_kwargs["temperature"] == pytest.approx(0.03)
    assert call_kwargs["max_tokens"] == 128


def test_process_substitutes_text_in_user_template():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        user_template="Bitte korrigiere: {text}",
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("korrigiert")
    pp.process("roher text")
    messages = pp._llm.create_chat_completion.call_args[1]["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert user_msg["content"] == "Bitte korrigiere: roher text"


def test_build_messages_keeps_leading_hey_computer_request_as_is():
    profile = PostprocessProfile(model_path="/fake/model.gguf")
    pp = LLMPostprocessor(profile)

    messages = pp._build_messages("Hey Computer, make this sound more formal")

    user_msg = next(m for m in messages if m["role"] == "user")
    assert user_msg["content"] == "Hey Computer, make this sound more formal"


@pytest.mark.parametrize("prompt", [_DEFAULT_SYSTEM_PROMPT, _REMOTE_CLEANUP_SYSTEM_PROMPT])
def test_assistant_mode_requires_literal_word_computer(prompt):
    """Regression test for the Gemma misfire on bare leading `Hey ...`
    (German example: `Hey, ich habe gesehen, wir haben ganz viel
    geschrieben.`). The shipped prompts must explicitly require the
    literal word `Computer` and must include counter-examples for bare
    greetings (`Hey`, `Hi`, `Hallo`) so the model doesn't fuzzy-match
    leading interjections to the trigger phrase."""
    # The hard requirement must be stated, not just implied.
    assert "HARD REQUIREMENT" in prompt
    assert "literal word `Computer`" in prompt
    # Bare-greeting counter-examples — these are what guard against the
    # bug case (`Hey, ...` triggering assistant mode).
    assert "bare `Hey`" in prompt
    assert "bare `Hi`" in prompt
    assert "bare `Hallo`" in prompt
    # The exact German example that misfired in the wild belongs in the
    # examples block as the most prominent counter-example.
    assert "Hey, ich habe gesehen" in prompt


@pytest.mark.parametrize("prompt", [_DEFAULT_SYSTEM_PROMPT, _REMOTE_CLEANUP_SYSTEM_PROMPT])
def test_bare_questions_are_not_assistant_triggers(prompt):
    """Regression test for Gemma answering bare questions like `Wie viel
    Uhr ist es gerade?` instead of cleaning them up. The prompt must
    state explicitly that questions without `Computer` stay CLEANUP, and
    must include German + English question counter-examples."""
    assert "bare QUESTION" in prompt
    assert "Wie viel Uhr ist es gerade?" in prompt
    assert "What time is it?" in prompt
    assert "Was meinst du dazu?" in prompt


def test_warmup_loads_model(tmp_path):
    """warmup() should call _build() and cache the result in _llm."""
    profile = PostprocessProfile(model_path=str(tmp_path / "model.gguf"))
    (tmp_path / "model.gguf").write_bytes(b"fake")
    pp = LLMPostprocessor(profile)
    mock_llm = MagicMock()
    with patch("justsayit.postprocess.LLMPostprocessor._build", return_value=mock_llm):
        pp.warmup()
    assert pp._llm is mock_llm


def test_strip_for_paste_noop_when_unset():
    profile = PostprocessProfile(model_path="/fake/model.gguf")
    pp = LLMPostprocessor(profile)
    assert pp.strip_for_paste("hello world") == "hello world"


def test_strip_for_paste_removes_match_dotall():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex=r"<\|channel\|>.*?<\|message\|>",
    )
    pp = LLMPostprocessor(profile)
    raw = "<|channel|>analysis\nthinking lines\nmore lines<|message|>real reply"
    assert pp.strip_for_paste(raw) == "real reply"


def test_strip_for_paste_strip_before_token():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex=r"(?s).*<\|message\|>",
    )
    pp = LLMPostprocessor(profile)
    raw = "preamble\n<|channel|>x<|message|>middle<|message|>final"
    assert pp.strip_for_paste(raw) == "final"


def test_strip_for_paste_invalid_regex_disabled(caplog):
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex="[unterminated",
    )
    pp = LLMPostprocessor(profile)
    assert pp.strip_for_paste("anything") == "anything"


def test_find_strip_matches_no_group_returns_full_match():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex=r"<\|channel>.*?<channel\|>",
    )
    pp = LLMPostprocessor(profile)
    raw = "<|channel>thinking<channel|>real reply"
    assert pp.find_strip_matches(raw) == ["<|channel>thinking<channel|>"]
    # strip still removes the whole match
    assert pp.strip_for_paste(raw) == "real reply"


def test_find_strip_matches_capture_group_returns_inner():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex=r"<\|channel>(.*?)<channel\|>",
    )
    pp = LLMPostprocessor(profile)
    raw = "<|channel>thinking<channel|>real reply"
    # group(1) = inner content, framing tags stripped from display
    assert pp.find_strip_matches(raw) == ["thinking"]
    # strip still removes the whole match (tags + content)
    assert pp.strip_for_paste(raw) == "real reply"


def test_find_strip_matches_multiple_blocks():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex=r"<\|channel>(.*?)<channel\|>",
    )
    pp = LLMPostprocessor(profile)
    raw = "<|channel>first<channel|>body<|channel>second<channel|>more"
    assert pp.find_strip_matches(raw) == ["first", "second"]


def test_find_strip_matches_empty_when_unset():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        paste_strip_regex="",
    )
    pp = LLMPostprocessor(profile)
    assert pp.find_strip_matches("anything") == []


def test_context_appended_to_system_prompt():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="Base prompt.",
        context="Name: Alice\nCountry: NL",
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("ok")
    pp.process("input")
    messages = pp._llm.create_chat_completion.call_args[1]["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"].startswith("Base prompt.")
    assert "# User context" in system_msg["content"]
    assert "Name: Alice" in system_msg["content"]
    assert "Country: NL" in system_msg["content"]


def test_context_empty_no_heading():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="Base prompt.",
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_mock_llama("ok")
    pp.process("input")
    messages = pp._llm.create_chat_completion.call_args[1]["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] == "Base prompt."
    assert "User context" not in system_msg["content"]


def test_dynamic_context_prepended_before_system_prompt_and_user_context(monkeypatch):
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="Base prompt.",
        context="Name: Alice",
    )
    pp = LLMPostprocessor(profile, dynamic_context_script="~/dynamic-context.sh")
    monkeypatch.setattr(
        pp, "_dynamic_context", lambda: "Date: 2026-04-17\nTimezone: Europe/Berlin"
    )

    prompt = pp._system_prompt()

    assert prompt == (
        "# STATE (DYNAMIC CONTEXT):\n"
        "Date: 2026-04-17\n"
        "Timezone: Europe/Berlin\n\n"
        "----\n\n"
        "Base prompt.\n\n"
        "# User context\n"
        "Name: Alice"
    )


def test_dynamic_context_empty_omits_state_block(monkeypatch):
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="Base prompt.",
    )
    pp = LLMPostprocessor(profile, dynamic_context_script="~/dynamic-context.sh")
    monkeypatch.setattr(pp, "_dynamic_context", lambda: "")

    assert pp._system_prompt() == "Base prompt."


def test_append_to_system_prompt_adds_paragraph_after_base():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="Base prompt.",
        append_to_system_prompt="Always reply in English.",
    )
    pp = LLMPostprocessor(profile)

    assert pp._system_prompt() == "Base prompt.\n\nAlways reply in English."


def test_append_to_system_prompt_works_without_base_prompt():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt_file="",
        append_to_system_prompt="Only this.",
    )
    pp = LLMPostprocessor(profile)

    assert pp._system_prompt() == "Only this."


def test_append_to_system_prompt_sits_between_base_and_context():
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        system_prompt="Base prompt.",
        append_to_system_prompt="Addition.",
        context="Name: Alice",
    )
    pp = LLMPostprocessor(profile)

    assert pp._system_prompt() == (
        "Base prompt.\n\nAddition.\n\n# User context\nName: Alice"
    )


def _make_strict_llama(response_text: str):
    """Mock Llama whose ``create_chat_completion`` mirrors the real
    fixed-signature behaviour (no ``**kwargs``), so any attempt to pass
    ``chat_template_kwargs=`` straight in raises ``TypeError`` exactly
    like llama-cpp-python 0.3.x does in production. Forwards the call
    into ``llm.chat_handler(**kwargs)`` so the wrapper installed by
    ``_install_chat_template_kwargs`` is exercised end-to-end."""
    llm = MagicMock()
    llm.chat_format = "stub"
    # Pre-install a permissive base handler so the wrapper's
    # ``llama_chat_format.get_chat_completion_handler(chat_format)``
    # fallback isn't exercised (no real "stub" handler exists).
    llm.chat_handler = MagicMock()

    def _create_chat_completion(
        *,
        messages,
        temperature=0.2,
        max_tokens=None,
    ):
        # Forward into the (possibly wrapped) handler so the test can
        # observe the kwargs the handler ultimately sees.
        llm.chat_handler(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return {"choices": [{"message": {"content": response_text}}]}

    llm.create_chat_completion = MagicMock(side_effect=_create_chat_completion)
    return llm


def test_chat_template_kwargs_not_passed_to_create_chat_completion():
    """Regression: ``Llama.create_chat_completion()`` rejects
    ``chat_template_kwargs`` (no ``**kwargs`` in its signature). The
    wrapper must inject the template kwargs at chat-handler level
    instead, leaving the top-level call clean."""
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        chat_template_kwargs={"enable_thinking": True},
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_strict_llama("ok")
    pp._install_chat_template_kwargs()

    pp.process("in")  # would raise TypeError without the wrapper

    call_kwargs = pp._llm.create_chat_completion.call_args[1]
    assert "chat_template_kwargs" not in call_kwargs


def test_chat_template_kwargs_reach_chat_handler():
    """Wrapper must surface template kwargs (e.g. ``enable_thinking``)
    to the underlying chat handler, since that's where the Jinja
    template picks them up."""
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        chat_template_kwargs={"enable_thinking": True},
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_strict_llama("ok")
    handler_mock = MagicMock()
    pp._llm.chat_handler = handler_mock
    pp._install_chat_template_kwargs()

    pp.process("in")

    handler_call = handler_mock.call_args[1]
    assert handler_call["enable_thinking"] is True


def test_chat_template_kwargs_install_noop_when_empty():
    """Empty ``chat_template_kwargs`` must leave the chat_handler
    untouched so we don't pay wrapping cost or risk breaking the
    default handler path."""
    profile = PostprocessProfile(
        model_path="/fake/model.gguf",
        chat_template_kwargs={},
    )
    pp = LLMPostprocessor(profile)
    pp._llm = _make_strict_llama("ok")
    sentinel = pp._llm.chat_handler
    pp._install_chat_template_kwargs()

    assert pp._llm.chat_handler is sentinel


def test_chat_template_kwargs_default_enables_thinking():
    """Built-in defaults opt into thinking so Qwen 3.5 works out of the
    box. Gemma ignores the flag (its Jinja template doesn't read it),
    so the same default is safe for every shipped model."""
    profile = PostprocessProfile(model_path="/fake/model.gguf")
    assert profile.chat_template_kwargs == {"enable_thinking": True}


def test_dynamic_context_script_empty_stdout_omitted(monkeypatch):
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(model_path="/fake/model.gguf")
    pp = LLMPostprocessor(profile, dynamic_context_script="~/dynamic-context.sh")

    class _Proc:
        returncode = 0
        stdout = "\n"
        stderr = ""

    monkeypatch.setattr(pp_mod.subprocess, "run", lambda *a, **k: _Proc())

    assert pp._dynamic_context() == ""


def test_dynamic_context_script_failure_logged_and_ignored(monkeypatch, caplog):
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(model_path="/fake/model.gguf")
    pp = LLMPostprocessor(profile, dynamic_context_script="~/dynamic-context.sh")

    class _Proc:
        returncode = 7
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(pp_mod.subprocess, "run", lambda *a, **k: _Proc())

    assert pp._dynamic_context() == ""
    assert "dynamic context script exited with 7" in caplog.text


def test_dynamic_context_success_and_assembled_prompt_logged(monkeypatch, caplog):
    caplog.set_level("INFO")
    profile = PostprocessProfile(
        model_path="/fake/model.gguf", system_prompt="Base prompt."
    )
    pp = LLMPostprocessor(profile, dynamic_context_script="~/dynamic-context.sh")
    pp._llm = _make_mock_llama("ok")
    monkeypatch.setattr(
        pp, "_dynamic_context", lambda: "Date: 2026-04-17\nTimezone: Europe/Berlin"
    )

    pp.process("input")

    assert "assembled LLM system prompt:" in caplog.text
    assert "# STATE (DYNAMIC CONTEXT):" in caplog.text
    assert "Date: 2026-04-17" in caplog.text
    assert "Timezone: Europe/Berlin" in caplog.text


def test_build_raises_without_llama_cpp():
    profile = PostprocessProfile(model_path="/nonexistent/model.gguf")
    pp = LLMPostprocessor(profile)
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            pp._build()


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_postprocess_config_defaults():
    cfg = Config()
    assert cfg.postprocess.enabled is False
    assert cfg.postprocess.profile == "gemma4-cleanup"


def test_render_includes_postprocess_section():
    import tomllib

    raw = tomllib.loads(render_config_toml())
    assert "postprocess" in raw
    assert raw["postprocess"]["enabled"] is False
    assert raw["postprocess"]["profile"] == "gemma4-cleanup"


def test_load_config_postprocess_settings(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[postprocess]\nenabled = true\nprofile = "my-model"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.postprocess.enabled is True
    assert cfg.postprocess.profile == "my-model"


# ---------------------------------------------------------------------------
# Personal-context sidecar
# ---------------------------------------------------------------------------


def test_ensure_context_file_writes_template(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    p = ensure_context_file()
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert 'context = ""' in body, "template must define an empty context value"
    assert "User context" not in body or "appended" in body, (
        "template should explain the field"
    )


def test_ensure_dynamic_context_script_writes_template(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    p = ensure_dynamic_context_script()
    assert p == dynamic_context_script_path()
    assert p.exists()
    assert p.read_text(encoding="utf-8") == _DYNAMIC_CONTEXT_SCRIPT
    assert p.stat().st_mode & 0o111


def test_ensure_dynamic_context_script_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    target = dynamic_context_script_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("#!/bin/sh\nprintf 'custom\\n'\n", encoding="utf-8")

    ensure_dynamic_context_script()

    assert target.read_text(encoding="utf-8") == "#!/bin/sh\nprintf 'custom\\n'\n"


def test_ensure_context_file_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    target = context_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('context = "my notes"\n', encoding="utf-8")
    ensure_context_file()
    assert target.read_text(encoding="utf-8") == 'context = "my notes"\n'


def test_load_context_sidecar_returns_value(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    target = context_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('context = "Name: Jane"\n', encoding="utf-8")
    assert load_context_sidecar() == "Name: Jane"


def test_load_context_sidecar_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    assert load_context_sidecar() == ""


def test_load_context_sidecar_malformed_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    target = context_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not = valid = toml = at all", encoding="utf-8")
    # Must not raise — returns "" so the LLM call still works.
    assert load_context_sidecar() == ""


def test_load_profile_falls_back_to_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    # Profile with no context field
    pdir = tmp_path / "postprocess"
    pdir.mkdir()
    (pdir / "demo.toml").write_text(
        'model_path = "/fake/model.gguf"\nsystem_prompt = "x"\n',
        encoding="utf-8",
    )
    # Sidecar with context
    target = context_file_path()
    target.write_text('context = "Name: Sidecar"\n', encoding="utf-8")
    profile = load_profile("demo")
    assert profile.context == "Name: Sidecar"


def test_load_profile_context_field_overrides_sidecar(tmp_path, monkeypatch):
    """Per-profile `context = "..."` wins over the sidecar — backward
    compat for users who already have context inline in their profile."""
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    pdir = tmp_path / "postprocess"
    pdir.mkdir()
    (pdir / "demo.toml").write_text(
        'model_path = "/fake/model.gguf"\ncontext = "Profile-level"\n',
        encoding="utf-8",
    )
    target = context_file_path()
    target.write_text('context = "Sidecar-level"\n', encoding="utf-8")
    profile = load_profile("demo")
    assert profile.context == "Profile-level"


# ---------------------------------------------------------------------------
# Commented-defaults form: fresh write, legacy migration, post-migration noop
# ---------------------------------------------------------------------------


def test_ensure_default_profile_writes_commented_template_on_fresh_install(
    tmp_path: Path,
):
    p = tmp_path / "gemma4-cleanup.toml"
    ensure_default_profile(p)
    assert p.read_text(encoding="utf-8") == _CLEANUP_PROFILE_TOML


def test_ensure_fun_profile_writes_commented_template_on_fresh_install(tmp_path: Path):
    p = tmp_path / "gemma4-fun.toml"
    ensure_fun_profile(p)
    assert p.read_text(encoding="utf-8") == _FUN_PROFILE_TOML


def test_ensure_default_profile_migrates_legacy_fully_populated_file(tmp_path: Path):
    """A legacy profile in the old fully-populated form (uncommented
    key=value lines, no commented-form marker) gets backed up exactly
    once and rewritten in commented-defaults form."""
    p = tmp_path / "gemma4-cleanup.toml"
    legacy = '# my custom profile\nmodel_path = "/tmp/x.gguf"\ntemperature = 0.5\n'
    p.write_text(legacy, encoding="utf-8")
    ensure_default_profile(p)
    backup = p.with_name(p.name + ".bak-pre-commented-form")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == legacy
    assert p.read_text(encoding="utf-8") == _CLEANUP_PROFILE_TOML


def test_ensure_default_profile_preserves_user_overrides_post_migration(tmp_path: Path):
    """After migration the file carries the commented-form marker. Even
    if the user uncomments and edits a key, subsequent ensure_*() calls
    must NOT back up + reset (that would discard their override) — the
    marker tells us we're already in the new form."""
    p = tmp_path / "gemma4-cleanup.toml"
    user_edited = (
        _CLEANUP_PROFILE_TOML.rstrip() + "\n\n# user override\ntemperature = 0.42\n"
    )
    p.write_text(user_edited, encoding="utf-8")
    ensure_default_profile(p)
    assert p.read_text(encoding="utf-8") == user_edited
    backup = p.with_name(p.name + ".bak-pre-commented-form")
    assert not backup.exists()


def test_shipped_profile_templates_parse_as_valid_toml():
    """Both templates in the source must parse with tomllib — regression
    guard for the f-string bug where a multi-line default leaked raw
    lines into the file at column 1, breaking TOML parsing and making
    the profile silently disappear from the tray menu."""
    import tomllib

    tomllib.loads(_CLEANUP_PROFILE_TOML)
    tomllib.loads(_FUN_PROFILE_TOML)


def test_ensure_default_profile_re_migrates_marker_carrying_corrupt_file(
    tmp_path: Path,
):
    """A file that bears the commented-form marker but fails TOML parse
    (i.e. was written by an earlier buggy template) must be backed up
    and rewritten — otherwise the user is stuck with a broken file that
    the tray will silently skip."""
    from justsayit.postprocess import _PROFILE_COMMENTED_FORM_MARKER

    p = tmp_path / "gemma4-cleanup.toml"
    corrupt = (
        f"{_PROFILE_COMMENTED_FORM_MARKER}\n"
        "# leading comments\n"
        "<|think|> stray uncommented junk that breaks TOML parsing\n"
    )
    p.write_text(corrupt, encoding="utf-8")
    ensure_default_profile(p)
    backup = p.with_name(p.name + ".bak-pre-commented-form")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == corrupt
    assert p.read_text(encoding="utf-8") == _CLEANUP_PROFILE_TOML


# ---------------------------------------------------------------------------
# Network: verify each built-in model repo exposes a Q4_K_M GGUF
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("model_key", list(KNOWN_LLM_MODELS.keys()))
def test_hf_model_has_q4_k_m_gguf(network, model_key):
    """Each entry in KNOWN_LLM_MODELS must have a Q4_K_M .gguf on HuggingFace."""
    hf_repo = KNOWN_LLM_MODELS[model_key]["hf_repo"]
    filename = find_hf_q4_filename(hf_repo)
    assert filename.endswith(".gguf"), (
        f"{model_key} ({hf_repo}): expected .gguf, got {filename!r}"
    )
    assert "Q4_K_M" in filename, (
        f"{model_key} ({hf_repo}): expected Q4_K_M in filename, got {filename!r}"
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM endpoint
# ---------------------------------------------------------------------------


def test_profile_has_openai_fields():
    p = PostprocessProfile()
    assert p.endpoint == ""
    assert p.model == ""
    assert p.api_key == ""
    assert p.api_key_env == "OPENAI_API_KEY"
    assert p.request_timeout == 60.0
    assert p.remote_retries == 3
    assert p.remote_retry_delay_seconds == pytest.approx(1.0)


def test_load_profile_defaults_include_remote_retry_fields(tmp_path):
    p = tmp_path / "minimal.toml"
    p.write_text(
        'endpoint = "https://api.example.com/v1"\nmodel = "m"\n', encoding="utf-8"
    )

    profile = load_profile(str(p))

    assert profile.remote_retries == 3
    assert profile.remote_retry_delay_seconds == pytest.approx(1.0)


def test_remote_process_posts_chat_completions(monkeypatch):
    """Happy path: profile.endpoint set → urllib POST with the right body
    and Authorization header; returned content is stripped and surfaced."""
    import json
    from unittest.mock import MagicMock
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test",
        system_prompt="Clean it up.",
        temperature=0.05,
        max_tokens=128,
    )
    pp = LLMPostprocessor(profile)

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        body = json.dumps(
            {"choices": [{"message": {"content": "  cleaned reply  "}}]}
        ).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *a: False
        return resp

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)

    out = pp.process("raw text")
    assert out == "cleaned reply"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    # Authorization is title-cased by urllib's header_items
    assert captured["headers"].get("Authorization") == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["temperature"] == pytest.approx(0.05)
    assert captured["body"]["max_tokens"] == 128
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][0]["content"] == "Clean it up."
    assert captured["body"]["messages"][1]["content"] == "raw text"
    assert captured["timeout"] == pytest.approx(60.0)


def test_remote_process_uses_env_key_when_literal_empty(monkeypatch, tmp_path):
    """No api_key in profile → resolve_secret reads from process env
    (which now also includes anything the .env loader merged in)."""
    from unittest.mock import MagicMock
    import json
    import justsayit.postprocess as pp_mod
    import justsayit.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "config_dir", lambda: tmp_path)
    cfg_mod._DOTENV_LOADED = False
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-shell")

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="m",
    )
    pp = LLMPostprocessor(profile)

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *a: False
        return resp

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)
    pp.process("x")
    assert captured["headers"]["Authorization"] == "Bearer key-from-shell"


def test_remote_process_raises_when_no_key(monkeypatch, tmp_path):
    """Endpoint set but no key anywhere → clear error message that names
    the env var the user should set."""
    import justsayit.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "config_dir", lambda: tmp_path)
    cfg_mod._DOTENV_LOADED = False
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="m",
    )
    pp = LLMPostprocessor(profile)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        pp.process("x")


def test_remote_process_raises_when_model_empty(monkeypatch):
    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        api_key="sk-x",
    )
    pp = LLMPostprocessor(profile)
    with pytest.raises(RuntimeError, match="profile.model"):
        pp.process("x")


def test_remote_process_falls_back_to_input_on_empty_response(monkeypatch):
    """Same contract as the local path: empty content → return input."""
    from unittest.mock import MagicMock
    import json
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="m",
        api_key="sk",
    )
    pp = LLMPostprocessor(profile)

    def fake_urlopen(req, timeout=None):
        body = json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *a: False
        return resp

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)
    assert pp.process("original") == "original"


def test_remote_process_retries_transient_http_error_then_succeeds(monkeypatch):
    import json
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="m",
        api_key="sk",
        remote_retries=2,
        remote_retry_delay_seconds=0.25,
    )
    pp = LLMPostprocessor(profile)

    calls = {"count": 0}
    sleeps = []

    def fake_sleep(delay):
        sleeps.append(delay)

    def fake_urlopen(req, timeout=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise pp_mod.urllib.error.HTTPError(
                req.full_url,
                503,
                "Service Unavailable",
                hdrs=None,
                fp=None,
            )

        class _Resp:
            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "done"}}]}
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Resp()

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pp_mod.time, "sleep", fake_sleep)

    assert pp.process("original") == "done"
    assert calls["count"] == 3
    assert sleeps == [0.25, 0.25]


def test_remote_process_exhausts_retries_on_transient_error(monkeypatch):
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="m",
        api_key="sk",
        remote_retries=2,
        remote_retry_delay_seconds=0.5,
    )
    pp = LLMPostprocessor(profile)

    calls = {"count": 0}
    sleeps = []

    def fake_sleep(delay):
        sleeps.append(delay)

    def fake_urlopen(req, timeout=None):
        calls["count"] += 1
        raise pp_mod.urllib.error.URLError("temporary dns failure")

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pp_mod.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="temporary dns failure"):
        pp.process("original")
    assert calls["count"] == 3
    assert sleeps == [0.5, 0.5]


def test_warmup_skipped_for_remote_endpoint():
    """warmup() must NOT touch llama-cpp-python when endpoint is set —
    there is no local model and we don't want a probe request."""
    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="m",
        api_key="sk",
    )
    pp = LLMPostprocessor(profile)
    pp.warmup()  # would raise RuntimeError("llama-cpp-python is not installed")
    assert pp._llm is None  # never tried to build


def test_remote_default_resolves_channel_free_prompt_via_file_reference():
    """A profile loaded with `base = "remote"` (or with `endpoint` set,
    auto-inferred to remote) should resolve `system_prompt_file =
    "cleanup_remote.md"` from `remote-defaults.toml` and emit the
    channel-free prompt — no auto-swap, just an overlay lookup."""
    profile = PostprocessProfile(
        base="remote",
        endpoint="https://api.example.com/v1",
        model="gpt-4o-mini",
        api_key="sk",
        system_prompt_file="cleanup_remote.md",
    )
    pp = LLMPostprocessor(profile)
    out = pp._system_prompt()
    assert out == _REMOTE_CLEANUP_SYSTEM_PROMPT.strip()
    # The remote prompt drops Gemma's `<|think|>` channel and explicitly
    # forbids the literal `No changes.` shortcut on both paths.
    assert "<|think|>" not in out
    assert "do NOT write `No changes.`" in out
    # Local prompt stays distinct — proves the resolution actually
    # picked the remote file rather than colliding with the local one.
    assert "<|think|>" in _DEFAULT_SYSTEM_PROMPT
    assert "NEVER respond with a status string" in _DEFAULT_SYSTEM_PROMPT
    assert "`No changes.`" in _DEFAULT_SYSTEM_PROMPT


def test_remote_endpoint_keeps_user_overridden_prompt():
    """If the user customised system_prompt, respect it verbatim — even
    on the remote path. The inline override always beats the file
    reference."""
    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="gpt-4o-mini",
        api_key="sk",
        system_prompt="Translate everything to pirate.",
    )
    pp = LLMPostprocessor(profile)
    assert pp._system_prompt() == "Translate everything to pirate."


def test_local_endpoint_keeps_default_prompt_with_channel_directives():
    """Without an endpoint set, the local llama-cpp path must still see
    the Gemma `<|think|>`-channel prompt — that's what `paste_strip_regex`
    is paired with."""
    profile = PostprocessProfile()  # no endpoint → builtin base
    pp = LLMPostprocessor(profile)
    assert pp._system_prompt() == _DEFAULT_SYSTEM_PROMPT.strip()


def test_ollama_gemma_combo_resolves_local_prompt_over_remote_backend(tmp_path):
    """The whole point of the orthogonal design: `base = "remote"`
    routes the call to the HTTP backend, but `system_prompt_file =
    "cleanup_local.md"` makes the model see Gemma's <|think|>-channel
    prompt. This is what makes Ollama-served Gemma work."""
    profile = PostprocessProfile(
        base="remote",
        endpoint="http://localhost:11434/v1",
        model="gemma3:4b",
        api_key="ollama",
        system_prompt_file="cleanup_local.md",
    )
    pp = LLMPostprocessor(profile)
    out = pp._system_prompt()
    # Got the Gemma channel prompt, not the channel-free one.
    assert "<|think|>" in out
    assert out == _DEFAULT_SYSTEM_PROMPT.strip()


def test_remote_process_forwards_chat_template_kwargs_when_set(monkeypatch):
    """Non-empty chat_template_kwargs must land in the JSON body so
    template toggles like Qwen 3.5's enable_thinking reach the server."""
    import json
    from unittest.mock import MagicMock
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="qwen3.5-0.8b",
        api_key="sk",
        chat_template_kwargs={"enable_thinking": True},
    )
    pp = LLMPostprocessor(profile)

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        body = json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}
        ).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *a: False
        return resp

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)

    pp.process("in")
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_remote_process_omits_chat_template_kwargs_when_empty(monkeypatch):
    """Empty dict → key must NOT appear in the body, so providers that
    don't understand it aren't confused (and we don't risk 400s from
    stricter gateways)."""
    import json
    from unittest.mock import MagicMock
    import justsayit.postprocess as pp_mod

    profile = PostprocessProfile(
        endpoint="https://api.example.com/v1",
        model="gpt-4o-mini",
        api_key="sk",
        chat_template_kwargs={},
    )
    pp = LLMPostprocessor(profile)

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        body = json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}
        ).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *a: False
        return resp

    monkeypatch.setattr(pp_mod.urllib.request, "urlopen", fake_urlopen)

    pp.process("in")
    assert "chat_template_kwargs" not in captured["body"]


def test_load_profile_chat_template_kwargs_from_toml(tmp_path, monkeypatch):
    """User profiles can set chat_template_kwargs via inline TOML table."""
    import justsayit.postprocess as pp_mod

    monkeypatch.setattr(pp_mod, "profiles_dir", lambda: tmp_path)
    p = tmp_path / "qwen-thinking.toml"
    p.write_text(
        "model_path = '/m'\n"
        "chat_template_kwargs = { enable_thinking = true }\n",
        encoding="utf-8",
    )
    profile = load_profile("qwen-thinking")
    assert profile.chat_template_kwargs == {"enable_thinking": True}
