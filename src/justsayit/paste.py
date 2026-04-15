"""Clipboard + auto-paste helpers for Wayland.

Supports two keystroke-injection backends, selectable via config:

* ``dotool`` — uinput-based, works on KDE Plasma / sway / Hyprland / niri.
  Requires the user to be in the ``input`` group (or the ``dotoold``
  user service running).
* ``wtype`` — virtual-keyboard protocol. Fine on sway/Hyprland; broken
  on Plasma 6 at time of writing.

``wl-copy`` is always used for the clipboard itself.

Important: ``wl-copy`` forks a background daemon to hold the selection.
That daemon inherits the subprocess pipes, so we *must* redirect
stdout/stderr to ``DEVNULL`` — otherwise ``subprocess.run`` waits for
EOF on a pipe the daemon keeps open, and the caller hangs forever.

Also important: each cold ``dotool`` invocation pays ~500-900ms creating
the uinput device. We keep **one** long-running ``dotool`` process and
write paste commands to its stdin — that makes subsequent pastes near
instant.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

log = logging.getLogger(__name__)


class PasteError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise PasteError(f"required tool {tool!r} not found on PATH")
    return path


def copy_to_clipboard(text: str, *, timeout: float = 5.0) -> None:
    """Put ``text`` on the Wayland clipboard via wl-copy."""
    if not text:
        return
    wl_copy = _require("wl-copy")
    try:
        proc = subprocess.run(
            [wl_copy],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise PasteError(f"wl-copy timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise PasteError(f"wl-copy exited {proc.returncode}")


def _dotool_input(combo: str) -> bytes:
    # dotool reads commands from stdin; `key` accepts XKB-style
    # modifier+key joined by '+'.
    return f"key {combo}\n".encode("utf-8")


def _wtype_argv(combo: str) -> list[str]:
    parts = [p for p in combo.split("+") if p]
    if not parts:
        raise PasteError(f"empty paste combo: {combo!r}")
    *modifiers, key = parts
    argv = ["wtype"]
    for m in modifiers:
        argv += ["-M", m]
    argv += ["-k", key]
    for m in reversed(modifiers):
        argv += ["-m", m]
    return argv


def send_paste_shortcut(
    combo: str = "ctrl+shift+v",
    *,
    backend: str = "dotool",
    timeout: float = 5.0,
) -> None:
    """Synthesise ``combo`` as a keystroke via the chosen backend."""
    if backend == "dotool":
        dotool = _require("dotool")
        argv = [dotool]
        stdin = _dotool_input(combo)
    elif backend == "wtype":
        _require("wtype")
        argv = _wtype_argv(combo)
        stdin = None
    else:
        raise PasteError(f"unknown paste backend: {backend!r}")

    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise PasteError(
            f"{backend} timed out after {timeout}s — is the service running?"
        ) from e
    if proc.returncode != 0:
        raise PasteError(f"{backend} exited {proc.returncode}")


def paste_text(
    text: str,
    *,
    combo: str = "ctrl+shift+v",
    backend: str = "dotool",
    settle_ms: int = 40,
    timeout: float = 5.0,
) -> None:
    """One-shot copy + paste. Prefer :class:`Paster` if you paste repeatedly —
    it keeps ``dotool`` warm and saves hundreds of ms per paste."""
    if not text:
        return
    copy_to_clipboard(text, timeout=timeout)
    if settle_ms > 0:
        time.sleep(settle_ms / 1000)
    send_paste_shortcut(combo, backend=backend, timeout=timeout)


class Paster:
    """Long-lived paster. Keeps a persistent ``dotool`` process warm so
    each paste skips the uinput cold-start cost. Thread-safe: ``paste``
    can be called from the transcription worker."""

    def __init__(
        self,
        *,
        backend: str = "dotool",
        combo: str = "ctrl+shift+v",
        settle_ms: int = 40,
        timeout: float = 5.0,
    ) -> None:
        self.backend = backend
        self.combo = combo
        self.settle_ms = settle_ms
        self.timeout = timeout
        self._dotool: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn any long-running backend processes. Safe to call twice."""
        if self.backend == "dotool":
            with self._lock:
                self._spawn_dotool_locked()

    def close(self) -> None:
        with self._lock:
            if self._dotool is None:
                return
            proc = self._dotool
            self._dotool = None
        try:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            log.warning("dotool process didn't exit; killing")
            try:
                proc.kill()
            except Exception:
                pass

    # --- main API ---------------------------------------------------------

    def paste(self, text: str) -> None:
        """Clipboard + synthetic keystroke, with per-step timing logs."""
        if not text:
            return
        t0 = time.monotonic()
        copy_to_clipboard(text, timeout=self.timeout)
        t_copy = time.monotonic()
        if self.settle_ms > 0:
            time.sleep(self.settle_ms / 1000)
        t_settle = time.monotonic()
        self._send_key(self.combo)
        t_key = time.monotonic()
        log.info(
            "paste timings: copy=%.0fms settle=%.0fms key=%.0fms total=%.0fms",
            (t_copy - t0) * 1000,
            (t_settle - t_copy) * 1000,
            (t_key - t_settle) * 1000,
            (t_key - t0) * 1000,
        )

    # --- internals --------------------------------------------------------

    def _send_key(self, combo: str) -> None:
        if self.backend == "dotool":
            self._send_dotool(combo)
        elif self.backend == "wtype":
            # wtype is short-lived; no daemon to reuse.
            send_paste_shortcut(combo, backend="wtype", timeout=self.timeout)
        else:
            raise PasteError(f"unknown paste backend: {self.backend!r}")

    def _spawn_dotool_locked(self) -> None:
        path = _require("dotool")
        log.info("spawning persistent dotool process")
        self._dotool = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _send_dotool(self, combo: str) -> None:
        payload = f"key {combo}\n".encode("utf-8")
        with self._lock:
            if self._dotool is None or self._dotool.poll() is not None:
                log.warning("dotool process not running; respawning")
                self._spawn_dotool_locked()
            assert self._dotool is not None and self._dotool.stdin is not None
            try:
                self._dotool.stdin.write(payload)
                self._dotool.stdin.flush()
                return
            except (BrokenPipeError, OSError) as e:
                log.warning("dotool pipe broken (%s); respawning and retrying", e)
                try:
                    self._dotool.kill()
                except Exception:
                    pass
                self._spawn_dotool_locked()
                assert self._dotool is not None and self._dotool.stdin is not None
                self._dotool.stdin.write(payload)
                self._dotool.stdin.flush()
