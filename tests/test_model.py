"""Tests for model path resolution and ensure_vad download logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from justsayit.config import Config
from justsayit.model import ModelPaths, ensure_vad, paths


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_vad_path_is_under_models_dir():
    cfg = Config()
    p = paths(cfg)
    import justsayit.model as _m
    assert p.vad.parent == _m.models_dir()


def test_paths_encoder_uses_config_filename():
    cfg = Config()
    cfg.model.parakeet_encoder = "my_encoder.onnx"
    p = paths(cfg)
    assert p.encoder.name == "my_encoder.onnx"


def test_paths_all_inside_parakeet_dir():
    cfg = Config()
    p = paths(cfg)
    base = p.encoder.parent
    for attr in ("encoder", "decoder", "joiner", "tokens"):
        assert getattr(p, attr).parent == base


def test_model_paths_is_frozen():
    cfg = Config()
    p = paths(cfg)
    with pytest.raises((TypeError, AttributeError)):
        p.encoder = Path("/other")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ensure_vad
# ---------------------------------------------------------------------------


def test_ensure_vad_skips_download_if_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.model.models_dir", lambda: tmp_path)
    vad_file = tmp_path / "silero_vad.onnx"
    vad_file.write_bytes(b"fake-model")

    download_calls: list[str] = []

    def _no_download(url, dest, **kw):
        download_calls.append(url)

    monkeypatch.setattr("justsayit.model._download", _no_download)
    result = ensure_vad(Config())
    assert result == vad_file
    assert download_calls == []


def test_ensure_vad_downloads_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.model.models_dir", lambda: tmp_path)

    def _fake_download(url, dest, **kw):
        # Simulate a successful download.
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"downloaded")

    monkeypatch.setattr("justsayit.model._download", _fake_download)
    result = ensure_vad(Config())
    assert result.name == "silero_vad.onnx"
    assert result.exists()


def test_ensure_vad_raises_if_download_produced_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.model.models_dir", lambda: tmp_path)

    def _broken_download(url, dest, **kw):
        pass  # doesn't create the file

    monkeypatch.setattr("justsayit.model._download", _broken_download)
    with pytest.raises(RuntimeError, match="VAD model still missing"):
        ensure_vad(Config())


def test_ensure_vad_force_re_downloads_existing(tmp_path, monkeypatch):
    monkeypatch.setattr("justsayit.model.models_dir", lambda: tmp_path)
    vad_file = tmp_path / "silero_vad.onnx"
    vad_file.write_bytes(b"old-model")

    download_calls: list[str] = []

    def _fake_download(url, dest, **kw):
        download_calls.append(url)
        Path(dest).write_bytes(b"new-model")

    monkeypatch.setattr("justsayit.model._download", _fake_download)
    ensure_vad(Config(), force=True)
    assert len(download_calls) == 1
