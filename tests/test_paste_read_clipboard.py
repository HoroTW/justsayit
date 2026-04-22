"""Coverage for ``read_clipboard``'s text-only guard.

Without ``text_only=True`` the function preserves the old behavior
(used by paste's clipboard-restore snapshot). With ``text_only=True``
the function must refuse non-text clipboards (e.g. images) instead
of decoding raw bytes as UTF-8 and emitting kilobytes of replacement
characters — which would then be fed to the LLM as context.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from justsayit import paste


def _fake_run_factory(responses: dict[tuple[str, ...], SimpleNamespace]):
    def _run(argv, **kwargs):
        key = tuple(argv[1:])  # drop the wl-paste path
        if key not in responses:
            raise AssertionError(f"unexpected wl-paste call: {argv!r}")
        return responses[key]

    return _run


def test_read_clipboard_text_only_skips_image_clipboard(monkeypatch):
    monkeypatch.setattr(paste.shutil, "which", lambda _: "/usr/bin/wl-paste")
    monkeypatch.setattr(
        paste.subprocess,
        "run",
        _fake_run_factory(
            {
                ("--list-types",): SimpleNamespace(
                    returncode=0, stdout=b"image/png\nimage/bmp\n"
                ),
            }
        ),
    )

    assert paste.read_clipboard(text_only=True) is None


def test_read_clipboard_text_only_picks_utf8_over_plain(monkeypatch):
    captured = []

    def _run(argv, **kwargs):
        captured.append(argv)
        if argv[1:] == ["--list-types"]:
            return SimpleNamespace(
                returncode=0,
                stdout=b"text/plain\ntext/plain;charset=utf-8\nUTF8_STRING\n",
            )
        return SimpleNamespace(returncode=0, stdout="hello".encode("utf-8"))

    monkeypatch.setattr(paste.shutil, "which", lambda _: "/usr/bin/wl-paste")
    monkeypatch.setattr(paste.subprocess, "run", _run)

    assert paste.read_clipboard(text_only=True) == "hello"
    # Second call is the actual fetch — must target the utf-8 MIME.
    assert captured[1][-2:] == ["--type", "text/plain;charset=utf-8"]


def test_read_clipboard_default_does_not_probe_types(monkeypatch):
    """Paste's restore-clipboard snapshot still wants the raw 'whatever
    is there' behavior — don't add an extra subprocess call for it."""
    calls = []

    def _run(argv, **kwargs):
        calls.append(argv[1:])
        return SimpleNamespace(returncode=0, stdout=b"hi")

    monkeypatch.setattr(paste.shutil, "which", lambda _: "/usr/bin/wl-paste")
    monkeypatch.setattr(paste.subprocess, "run", _run)

    assert paste.read_clipboard() == "hi"
    assert calls == [["--no-newline"]]


def test_read_clipboard_missing_wl_paste_returns_none(monkeypatch):
    monkeypatch.setattr(paste.shutil, "which", lambda _: None)
    # text_only shouldn't matter when wl-paste isn't installed.
    assert paste.read_clipboard() is None
    assert paste.read_clipboard(text_only=True) is None


def test_read_clipboard_list_types_timeout_returns_none(monkeypatch):
    monkeypatch.setattr(paste.shutil, "which", lambda _: "/usr/bin/wl-paste")

    def _run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=2.0)

    monkeypatch.setattr(paste.subprocess, "run", _run)
    assert paste.read_clipboard(text_only=True) is None
