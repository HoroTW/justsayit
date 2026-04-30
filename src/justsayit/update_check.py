"""Best-effort version check via pyproject.toml on the main branch.

Reads ``pyproject.toml`` from the ``main`` branch (simple commit-based workflow)
and compares the ``version`` field against ``__version__``.

A failed check (no network, GitHub down, malformed response, broken cache
file) is silently ignored from the user's point of view — startup never
blocks on this. The latest seen version is cached in the cache dir for 3h
so repeated launches don't hammer the API.

The release/endpoint URL comment is kept for clarity, but the check defaults
to main-branch pyproject.toml behavior.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from justsayit.config import cache_dir

log = logging.getLogger(__name__)

PYPROJECT_URL = "https://raw.githubusercontent.com/HoroTW/justsayit/main/pyproject.toml"
RELEASE_PAGE_URL = "https://github.com/HoroTW/justsayit/releases"
CHECK_INTERVAL_SECONDS = 3 * 60 * 60


@dataclass(frozen=True)
class UpdateInfo:
    current: str
    latest: str
    url: str


def _cache_path() -> Path:
    return cache_dir() / "update_check.json"


def detect_install_dir() -> Path | None:
    """Return the project root if we look like an editable install with
    ``install.sh`` + ``.git`` present, otherwise ``None``.

    Used to compose a "how to update" hint in the notification — Nix or
    pip-installed users get a generic "see releases page" hint instead
    because they can't just ``git pull``.
    """
    import justsayit as _j

    init = Path(_j.__file__).resolve()
    # editable install: <root>/src/justsayit/__init__.py -> root is parent.parent.parent
    candidate = init.parent.parent.parent
    if (candidate / "install.sh").exists() and (candidate / ".git").exists():
        return candidate
    return None


def parse_version_from_pyproject(text: str) -> str | None:
    """Extract the ``version = "..."`` line from a pyproject.toml body."""
    # Look for version = "x.y.z" (allow optional whitespace, single quotes)
    m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        return None
    return m.group(1)


def is_newer(latest: str, current: str) -> bool:
    """True if *latest* is a strictly higher X.Y.Z semver than *current*.

    Returns False if either version is unparseable — we'd rather miss an
    update than nag about a bogus comparison.
    """
    try:
        a = tuple(int(p) for p in latest.strip().split("."))
        b = tuple(int(p) for p in current.strip().split("."))
    except ValueError:
        return False
    if len(a) != 3 or len(b) != 3:
        return False
    return a > b


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(path: Path, latest: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"checked_at": int(time.time()), "latest": latest}),
            encoding="utf-8",
        )
    except OSError:
        log.debug("could not write update-check cache", exc_info=True)


def _fetch_latest(timeout: float) -> str | None:
    """Fetch the version from main-branch pyproject.toml. Returns the
    bare version string, or None on any failure."""
    req = urllib.request.Request(
        PYPROJECT_URL,
        headers={"User-Agent": "justsayit/update-check"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.info("update check: pyproject.toml fetch failed: %s", exc)
        return None

    version = parse_version_from_pyproject(body)
    if version is None:
        log.info("update check: pyproject.toml missing a usable version")
        return None

    log.info("update check: fetched version=%s from pyproject.toml", version)
    return version


def _check_for_update_with_status(
    current_version: str,
    *,
    timeout: float = 5.0,
    force: bool = False,
    cache_path: Path | None = None,
) -> tuple[UpdateInfo | None, bool]:
    """Return ``(result, checked)`` for an update check.

    ``checked`` is ``True`` when we successfully determined the latest
    version (including a cached "no update" result), and ``False`` when
    the check failed and should stay silent.
    """
    path = cache_path if cache_path is not None else _cache_path()
    cache = _load_cache(path)
    now = int(time.time())
    if not force and cache:
        last = int(cache.get("checked_at", 0))
        cached_latest = cache.get("latest")
        if cached_latest and now - last < CHECK_INTERVAL_SECONDS:
            log.info(
                "update check: using cached latest=%s age=%ss",
                cached_latest,
                now - last,
            )
            if is_newer(cached_latest, current_version):
                log.info(
                    "update check: decision=update source=cache current=%s latest=%s",
                    current_version,
                    cached_latest,
                )
                return UpdateInfo(
                    current_version, cached_latest, RELEASE_PAGE_URL
                ), True
            log.info(
                "update check: decision=no-update source=cache current=%s latest=%s",
                current_version,
                cached_latest,
            )
            return None, True

    latest = _fetch_latest(timeout)
    if latest is None:
        log.info("update check: decision=fetch-failed")
        return None, False

    _save_cache(path, latest)
    if is_newer(latest, current_version):
        log.info(
            "update check: decision=update source=network current=%s latest=%s",
            current_version,
            latest,
        )
        return UpdateInfo(current_version, latest, RELEASE_PAGE_URL), True

    log.info(
        "update check: decision=no-update source=network current=%s latest=%s",
        current_version,
        latest,
    )
    return None, True


def check_for_update(
    current_version: str,
    *,
    timeout: float = 5.0,
    force: bool = False,
    cache_path: Path | None = None,
) -> UpdateInfo | None:
    """Synchronously check the version in pyproject.toml on the main branch.

    Returns :class:`UpdateInfo` if the latest seen version is strictly
    newer than *current_version*; ``None`` otherwise (no update, network
    error, or short-circuited by the 3h cache).

    *force* skips the cache check; *cache_path* overrides the default
    location for tests.
    """
    result, _checked = _check_for_update_with_status(
        current_version,
        timeout=timeout,
        force=force,
        cache_path=cache_path,
    )
    return result


def check_async(
    current_version: str,
    on_result: Callable[[UpdateInfo | None, bool], None],
    *,
    timeout: float = 5.0,
) -> threading.Thread:
    """Run :func:`check_for_update` on a daemon thread.

    *on_result* is invoked with ``(result, checked)`` on the worker
    thread — wrap it in ``GLib.idle_add`` if it touches the UI.
    """

    def _run() -> None:
        try:
            result, checked = _check_for_update_with_status(
                current_version, timeout=timeout
            )
        except Exception:
            log.exception("update check raised")
            result = None
            checked = False
        try:
            on_result(result, checked)
        except Exception:
            log.exception("update-check callback raised")

    t = threading.Thread(target=_run, name="justsayit-update-check", daemon=True)
    t.start()
    return t
