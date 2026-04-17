"""Tests for the GitHub version-check helper."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from justsayit import __version__, update_check
from justsayit.cli import App
from justsayit.update_check import (
    UpdateInfo,
    check_for_update,
    is_newer,
    parse_version_from_pyproject,
)


# --- pure helpers -----------------------------------------------------------


def test_parse_version_from_pyproject():
    body = '[project]\nname = "x"\nversion = "1.2.3"\n'
    assert parse_version_from_pyproject(body) == "1.2.3"


def test_parse_version_first_match_wins():
    # The Build [tool.hatch.build.targets.wheel] section can also have a
    # version field in some packaging setups; we want the [project] one,
    # which is always first.
    body = '[project]\nversion = "0.7.2"\n[tool.foo]\nversion = "9.9.9"\n'
    assert parse_version_from_pyproject(body) == "0.7.2"


def test_parse_version_missing_returns_none():
    assert parse_version_from_pyproject("[project]\nname = 'x'\n") is None


@pytest.mark.parametrize(
    "latest,current,expected",
    [
        ("1.0.0", "0.9.0", True),
        ("0.7.2", "0.7.1", True),
        ("1.0.0", "1.0.0", False),
        ("0.9.0", "1.0.0", False),
        ("1.2.3", "1.2.4", False),
        ("garbage", "0.7.0", False),
        ("0.7.0", "garbage", False),
    ],
)
def test_is_newer(latest: str, current: str, expected: bool):
    assert is_newer(latest, current) is expected


# --- check_for_update -------------------------------------------------------


def test_check_for_update_returns_info_when_newer(tmp_path: Path):
    cache = tmp_path / "cache.json"
    with patch.object(update_check, "_fetch_latest", return_value="1.0.0"):
        info = check_for_update("0.7.2", cache_path=cache)
    assert isinstance(info, UpdateInfo)
    assert info.current == "0.7.2"
    assert info.latest == "1.0.0"
    # Cache must now exist with the latest version.
    assert cache.exists()
    cached = json.loads(cache.read_text())
    assert cached["latest"] == "1.0.0"


def test_check_for_update_returns_none_when_same(tmp_path: Path):
    cache = tmp_path / "cache.json"
    with patch.object(update_check, "_fetch_latest", return_value="0.7.2"):
        assert check_for_update("0.7.2", cache_path=cache) is None


def test_check_for_update_returns_none_on_fetch_failure(tmp_path: Path):
    cache = tmp_path / "cache.json"
    with patch.object(update_check, "_fetch_latest", return_value=None):
        assert check_for_update("0.7.2", cache_path=cache) is None
    # No cache write on failure (so we retry sooner instead of the
    # 24h-cached "no update" response).
    assert not cache.exists()


def test_check_for_update_uses_cache_within_interval(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"checked_at": int(time.time()), "latest": "9.9.9"}))
    # Mocking _fetch_latest to a sentinel ensures we DON'T hit the network.
    with patch.object(
        update_check,
        "_fetch_latest",
        side_effect=AssertionError("should not have fetched — cache is fresh"),
    ):
        info = check_for_update("0.7.2", cache_path=cache)
    assert info is not None
    assert info.latest == "9.9.9"


def test_check_for_update_force_bypasses_cache(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"checked_at": int(time.time()), "latest": "0.7.2"}))
    with patch.object(update_check, "_fetch_latest", return_value="9.9.9"):
        info = check_for_update("0.7.2", cache_path=cache, force=True)
    assert info is not None and info.latest == "9.9.9"


def test_check_for_update_expired_cache_refetches(tmp_path: Path):
    cache = tmp_path / "cache.json"
    # Older than CHECK_INTERVAL_SECONDS — cache is stale.
    cache.write_text(
        json.dumps(
            {
                "checked_at": int(time.time())
                - update_check.CHECK_INTERVAL_SECONDS
                - 10,
                "latest": "0.7.2",
            }
        )
    )
    with patch.object(update_check, "_fetch_latest", return_value="0.8.0"):
        info = check_for_update("0.7.2", cache_path=cache)
    assert info is not None and info.latest == "0.8.0"


def test_check_for_update_corrupt_cache_is_ignored(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text("{not valid json")
    with patch.object(update_check, "_fetch_latest", return_value="1.0.0"):
        info = check_for_update("0.7.2", cache_path=cache)
    assert info is not None and info.latest == "1.0.0"


def test_kick_off_update_check_logs_startup_message_and_starts_async(caplog):
    app = object.__new__(App)
    app.gtk_app = object()
    app.overlay = None

    captured: dict[str, object] = {}

    def fake_check_async(current_version, on_result, *, timeout=5.0):
        captured["current_version"] = current_version
        captured["on_result"] = on_result
        captured["timeout"] = timeout
        return object()

    caplog.set_level("INFO", logger="justsayit")

    with (
        patch("justsayit.update_check.check_async", side_effect=fake_check_async),
        patch("justsayit.update_check.detect_install_dir", return_value=None),
    ):
        app._kick_off_update_check()

    assert "checking for updates on GitHub..." in caplog.text
    assert captured["current_version"] == __version__
    assert callable(captured["on_result"])
    assert captured["timeout"] == 5.0


def test_kick_off_update_check_logs_no_update_result(caplog):
    app = object.__new__(App)
    app.gtk_app = object()
    app.overlay = None

    captured: dict[str, object] = {}

    def fake_check_async(current_version, on_result, *, timeout=5.0):
        captured["on_result"] = on_result
        return object()

    caplog.set_level("INFO", logger="justsayit")

    with (
        patch("justsayit.update_check.check_async", side_effect=fake_check_async),
        patch("justsayit.update_check.detect_install_dir", return_value=None),
    ):
        app._kick_off_update_check()

    captured["on_result"](None, True)

    assert "checking for updates on GitHub..." in caplog.text
    assert "no update available on GitHub" in caplog.text


def test_kick_off_update_check_logs_update_available_result(caplog):
    app = object.__new__(App)
    app.gtk_app = object()
    app.overlay = None
    app._notify_update_available = lambda info, install_dir: None

    captured: dict[str, object] = {}

    def fake_check_async(current_version, on_result, *, timeout=5.0):
        captured["on_result"] = on_result
        return object()

    info = UpdateInfo(current="0.11.5", latest="0.11.6", url="https://example.com")
    caplog.set_level("INFO", logger="justsayit")

    with (
        patch("justsayit.update_check.check_async", side_effect=fake_check_async),
        patch("justsayit.update_check.detect_install_dir", return_value=None),
        patch("justsayit.cli.GLib.idle_add", side_effect=lambda fn: fn()),
    ):
        app._kick_off_update_check()
        captured["on_result"](info, True)

    assert "update available: v0.11.5 -> v0.11.6" in caplog.text
