"""Tests for LLM postprocessing — profile loading, config integration, and
the LLMPostprocessor's process() method (using a mock llama_cpp.Llama)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from justsayit.config import Config, load_config, render_config_toml
from justsayit.postprocess import (
    KNOWN_LLM_MODELS,
    LLMPostprocessor,
    PostprocessProfile,
    ensure_default_profile,
    find_hf_q4_filename,
    load_profile,
    profiles_dir,
)


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
    assert "system_prompt" in content
    assert "temperature" in content
    # The recommended cleanup prompt has the conservative guardrails baked in
    assert "CONSERVATIVE CLEANUP" in content
    assert "modal particles" in content


def test_ensure_default_profiles_writes_both(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.postprocess.config_dir", lambda: tmp_path)
    from justsayit.postprocess import ensure_default_profiles

    cleanup, fun = ensure_default_profiles()
    assert cleanup.name == "gemma4-cleanup.toml"
    assert fun.name == "gemma4-fun.toml"
    assert cleanup.exists() and fun.exists()
    fun_text = fun.read_text(encoding="utf-8")
    # Fun profile is the emojify stub and points users back at cleanup.
    assert "Emojify" in fun_text
    assert "gemma4-cleanup" in fun_text
    # No <|think|> in fun → no strip regex needed.
    assert 'paste_strip_regex = ""' in fun_text


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
        "[postprocess]\nenabled = true\nprofile = \"my-model\"\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.postprocess.enabled is True
    assert cfg.postprocess.profile == "my-model"


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
