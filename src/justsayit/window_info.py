"""Best-effort focused-window class lookup.

Used by the window-clipboard policy to auto-arm or block clipboard
context based on which application is focused when a recording starts.

We try ``kdotool`` first (the Wayland-native KDE port of xdotool) and
fall back to ``xdotool`` for X11 sessions. Either tool missing is fine
— the function returns ``None`` and the caller treats the policy as a
no-op for that recording.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger(__name__)

# Hard upper bound; the lookup runs on the audio worker thread at
# recording-start, so anything slower than this would be perceived as a
# delay before the chime.
_TIMEOUT_S = 0.2


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


def active_window_id() -> str | None:
    """Return the focused window's class / app-id (lowercased), or None.

    One-shot, no caching. ``None`` covers every failure mode (tool
    missing, timeout, no focused window, parse error) so callers can
    treat the policy as opt-in: missing class → policy never triggers.
    """
    if shutil.which("kdotool"):
        out = _run(["kdotool", "getactivewindow", "getwindowclassname"])
        if out:
            return out.strip().lower()
    if shutil.which("xdotool"):
        out = _run(["xdotool", "getactivewindow", "getwindowclassname"])
        if out:
            return out.strip().lower()
    log.debug("active_window_id: neither kdotool nor xdotool available")
    return None
