"""Best-effort GitHub release-version check.

Reads ``pyproject.toml`` from the ``main`` branch on GitHub (the canonical
source of truth — gets bumped with every release commit, no separate
GitHub Release object required) and compares against ``__version__``.

A failed check (no network, GitHub down, malformed response, broken cache
file) is silently ignored — startup never blocks on this. The latest
seen version is cached for 24h in the cache dir so launching repeatedly
doesn't hammer the API.
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

PYPROJECT_URL = (
    "https://raw.githubusercontent.com/HoroTW/justsayit/main/pyproject.toml"
)
RELEASE_PAGE_URL = "https://github.com/HoroTW/justsayit/releases"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


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
    """Extract the first ``version = "..."`` line from a pyproject body."""
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


def _semver_tuple(v: str) -> tuple[int, int, int] | None:
    m = _SEMVER_RE.match(v)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def is_newer(latest: str, current: str) -> bool:
    """True if *latest* is a strictly higher semver than *current*.

    Returns False if either version is unparseable — we'd rather miss an
    update than nag about a bogus comparison.
    """
    a = _semver_tuple(latest)
    b = _semver_tuple(current)
    if a is None or b is None:
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
    req = urllib.request.Request(
        PYPROJECT_URL, headers={"User-Agent": "justsayit/update-check"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.debug("update check: fetch failed: %s", exc)
        return None
    return parse_version_from_pyproject(body)


def check_for_update(
    current_version: str,
    *,
    timeout: float = 5.0,
    force: bool = False,
    cache_path: Path | None = None,
) -> UpdateInfo | None:
    """Synchronously check GitHub for a newer version.

    Returns :class:`UpdateInfo` if the latest seen version is strictly
    newer than *current_version*; ``None`` otherwise (no update, network
    error, or short-circuited by the 24h cache).

    *force* skips the cache check; *cache_path* overrides the default
    location for tests.
    """
    path = cache_path if cache_path is not None else _cache_path()
    cache = _load_cache(path)
    now = int(time.time())
    if not force and cache:
        last = int(cache.get("checked_at", 0))
        cached_latest = cache.get("latest")
        if cached_latest and now - last < CHECK_INTERVAL_SECONDS:
            if is_newer(cached_latest, current_version):
                return UpdateInfo(current_version, cached_latest, RELEASE_PAGE_URL)
            return None

    latest = _fetch_latest(timeout)
    if latest is None:
        return None
    _save_cache(path, latest)
    if is_newer(latest, current_version):
        return UpdateInfo(current_version, latest, RELEASE_PAGE_URL)
    return None


def check_async(
    current_version: str,
    on_result: Callable[[UpdateInfo | None], None],
    *,
    timeout: float = 5.0,
) -> threading.Thread:
    """Run :func:`check_for_update` on a daemon thread.

    *on_result* is invoked with the result on the worker thread — wrap
    it in ``GLib.idle_add`` if it touches the UI.
    """

    def _run() -> None:
        try:
            result = check_for_update(current_version, timeout=timeout)
        except Exception:
            log.exception("update check raised")
            result = None
        try:
            on_result(result)
        except Exception:
            log.exception("update-check callback raised")

    t = threading.Thread(target=_run, name="justsayit-update-check", daemon=True)
    t.start()
    return t
