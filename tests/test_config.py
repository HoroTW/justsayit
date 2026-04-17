"""Tests for config loading, rendering, and saving."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from justsayit.config import (
    Config,
    LogConfig,
    ModelConfig,
    OverlayConfig,
    PasteConfig,
    SoundConfig,
    VadConfig,
    load_config,
    render_config_toml,
    save_config,
    save_state,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_model_backend():
    assert Config().model.backend == "parakeet"


def test_default_whisper_model():
    cfg = ModelConfig()
    assert cfg.whisper_model == "Systran/faster-distil-whisper-large-v3"
    assert cfg.whisper_device == "cpu"
    assert cfg.whisper_compute_type == "int8"


def test_default_sound_config():
    s = SoundConfig()
    assert s.enabled is True
    assert s.volume == 1.0
    assert s.validating_volume_scale == 0.4


def test_default_paste_space_fields():
    p = PasteConfig()
    assert p.auto_space_timeout_ms == 0
    assert p.append_trailing_space is False


def test_default_overlay_fields():
    o = OverlayConfig()
    assert o.visualizer_sensitivity == 1.0
    assert o.opacity == 0.78


def test_default_log_config():
    l = LogConfig()
    assert l.file_enabled is False
    assert l.file_path == ""


# ---------------------------------------------------------------------------
# render_config_toml
# ---------------------------------------------------------------------------


def _parsed(cfg=None):
    """Render cfg (or defaults) to TOML and parse it back as a dict."""
    return tomllib.loads(render_config_toml(cfg))


def test_render_produces_valid_toml():
    raw = _parsed()
    assert isinstance(raw, dict)


def test_render_includes_model_section_with_backend():
    raw = _parsed()
    assert raw["model"]["backend"] == "parakeet"


def test_render_includes_whisper_fields():
    raw = _parsed()
    assert "whisper_model" in raw["model"]
    assert "whisper_device" in raw["model"]
    assert "whisper_compute_type" in raw["model"]


def test_render_includes_sound_section():
    raw = _parsed()
    assert "sound" in raw
    assert raw["sound"]["enabled"] is True
    assert raw["sound"]["validating_volume_scale"] == pytest.approx(0.4)


def test_render_booleans_as_toml_literals():
    """TOML booleans must be lowercase true/false, not Python True/False."""
    toml_str = render_config_toml()
    assert "True" not in toml_str
    assert "False" not in toml_str
    assert "true" in toml_str or "false" in toml_str


def test_render_non_default_backend():
    cfg = Config()
    cfg.model.backend = "whisper"
    raw = _parsed(cfg)
    assert raw["model"]["backend"] == "whisper"


def test_render_includes_all_parakeet_file_fields():
    raw = _parsed()
    model = raw["model"]
    for key in ("parakeet_encoder", "parakeet_decoder", "parakeet_joiner", "parakeet_tokens"):
        assert key in model, f"missing key: {key}"


# ---------------------------------------------------------------------------
# load_config round-trips
# ---------------------------------------------------------------------------


def test_load_config_missing_file_returns_defaults(tmp_path):
    result = load_config(tmp_path / "nope.toml")
    assert result.model.backend == "parakeet"
    assert result.sound.enabled is True


def test_load_config_whisper_backend(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[model]\nbackend = "whisper"\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.model.backend == "whisper"


def test_load_config_whisper_model_override(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[model]\nwhisper_model = "Systran/faster-whisper-large-v3"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.model.whisper_model == "Systran/faster-whisper-large-v3"


def test_load_config_sound_settings(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[sound]\nenabled = false\nvolume = 0.5\nvalidating_volume_scale = 0.2\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.sound.enabled is False
    assert cfg.sound.volume == pytest.approx(0.5)
    assert cfg.sound.validating_volume_scale == pytest.approx(0.2)


def test_load_config_paste_space_settings(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[paste]\nauto_space_timeout_ms = 3000\nappend_trailing_space = true\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.paste.auto_space_timeout_ms == 3000
    assert cfg.paste.append_trailing_space is True


def test_load_config_unknown_keys_ignored(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nbackend = \"parakeet\"\nfuture_setting = 99\n", encoding="utf-8")
    cfg = load_config(p)  # must not raise
    assert cfg.model.backend == "parakeet"


def test_render_load_roundtrip_defaults(tmp_path):
    original = Config()
    p = tmp_path / "config.toml"
    p.write_text(render_config_toml(original), encoding="utf-8")
    restored = load_config(p)

    assert restored.model.backend == original.model.backend
    assert restored.model.whisper_model == original.model.whisper_model
    assert restored.sound.validating_volume_scale == pytest.approx(
        original.sound.validating_volume_scale
    )
    assert restored.paste.auto_space_timeout_ms == original.paste.auto_space_timeout_ms
    assert restored.overlay.opacity == pytest.approx(original.overlay.opacity)
    assert restored.vad.enabled == original.vad.enabled


def test_render_load_roundtrip_whisper(tmp_path):
    cfg = Config()
    cfg.model.backend = "whisper"
    cfg.model.whisper_model = "my-local-model"
    cfg.model.whisper_device = "cuda"
    p = tmp_path / "config.toml"
    p.write_text(render_config_toml(cfg), encoding="utf-8")
    restored = load_config(p)
    assert restored.model.backend == "whisper"
    assert restored.model.whisper_model == "my-local-model"
    assert restored.model.whisper_device == "cuda"


# ---------------------------------------------------------------------------
# save_config / state.toml split
# ---------------------------------------------------------------------------


def test_save_config_writes_state_not_config(tmp_path):
    """save_config persists the runtime-mutable subset to a sibling
    state.toml and never touches config.toml."""
    p = tmp_path / "config.toml"
    cfg = Config()
    cfg.vad.enabled = True
    save_config(cfg, p)
    # config.toml is untouched (didn't exist before, doesn't exist after).
    assert not p.exists()
    # state.toml carries the toggle.
    assert (tmp_path / "state.toml").exists()
    restored = load_config(p)
    assert restored.vad.enabled is True


def test_save_config_does_not_overwrite_existing_config_toml(tmp_path):
    """If the user has authored config.toml (with comments, custom values,
    extra fields…), save_config must not rewrite it. Only state.toml
    moves."""
    p = tmp_path / "config.toml"
    authored = (
        "# my carefully written config\n"
        "[sound]\n"
        "volume = 0.3  # quieter than default\n"
        "\n"
        "[model]\n"
        'backend = "whisper"\n'
    )
    p.write_text(authored, encoding="utf-8")

    runtime = Config()
    runtime.vad.enabled = True
    save_config(runtime, p)

    # config.toml is byte-identical to what the user authored.
    assert p.read_text(encoding="utf-8") == authored


def test_state_overlay_wins_over_config_toml(tmp_path):
    """State.toml's runtime fields override what's in config.toml; non-state
    fields fall through from config.toml."""
    p = tmp_path / "config.toml"
    disk_cfg = Config()
    disk_cfg.model.backend = "whisper"
    disk_cfg.sound.volume = 0.3
    disk_cfg.vad.enabled = False  # config.toml says off
    p.write_text(render_config_toml(disk_cfg), encoding="utf-8")

    # Toggle VAD on at runtime.
    runtime = Config()
    runtime.vad.enabled = True
    save_config(runtime, p)

    restored = load_config(p)
    assert restored.vad.enabled is True            # state wins
    assert restored.model.backend == "whisper"     # config.toml preserved
    assert restored.sound.volume == pytest.approx(0.3)


def test_load_config_falls_back_to_config_toml_when_state_missing(tmp_path):
    """No state.toml → config.toml's runtime fields are used as-is."""
    p = tmp_path / "config.toml"
    disk_cfg = Config()
    disk_cfg.vad.enabled = True
    disk_cfg.postprocess.enabled = True
    disk_cfg.postprocess.profile = "gemma4-cleanup"
    p.write_text(render_config_toml(disk_cfg), encoding="utf-8")

    restored = load_config(p)
    assert restored.vad.enabled is True
    assert restored.postprocess.enabled is True
    assert restored.postprocess.profile == "gemma4-cleanup"


def test_save_state_persists_postprocess_profile(tmp_path):
    """save_state covers all three runtime fields, not just vad.enabled."""
    sp = tmp_path / "state.toml"
    cfg = Config()
    cfg.postprocess.enabled = True
    cfg.postprocess.profile = "gemma4-fun"
    save_state(cfg, sp)

    raw = tomllib.loads(sp.read_text(encoding="utf-8"))
    assert raw["postprocess"]["enabled"] is True
    assert raw["postprocess"]["profile"] == "gemma4-fun"


def test_state_overlay_ignores_malformed_file(tmp_path):
    """A broken state.toml should never crash the app — load_config falls
    back to config.toml values."""
    p = tmp_path / "config.toml"
    disk_cfg = Config()
    disk_cfg.vad.enabled = True
    p.write_text(render_config_toml(disk_cfg), encoding="utf-8")
    (tmp_path / "state.toml").write_text("this is not valid TOML [[[", encoding="utf-8")

    restored = load_config(p)
    assert restored.vad.enabled is True
