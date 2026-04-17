"""Clipboard + auto-paste helpers for Wayland.

Uses ``dotool`` (uinput-based) for keystroke injection — works on KDE
Plasma / sway / Hyprland / niri. Requires the user to be in the
``input`` group.

``wl-copy`` is always used for the clipboard itself unless
``skip_clipboard_history`` is True, in which case :class:`Paster` uses
``dotool type`` to inject keystrokes directly — no clipboard involved.

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


def _run_wl_copy(cmd: list[str], text: str, timeout: float) -> None:
    try:
        proc = subprocess.run(
            cmd,
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


def copy_to_clipboard(
    text: str, *, timeout: float = 5.0, sensitive: bool = False
) -> None:
    """Put ``text`` on both the regular and primary Wayland clipboards via wl-copy.

    When ``sensitive`` is True, ``--sensitive`` is passed to wl-copy so that
    clipboard managers (e.g. KDE Klipper) skip recording this entry.  The text
    is still available for a manual Ctrl+V / Shift+Insert / middle-click paste.
    """
    if not text:
        return
    wl_copy = _require("wl-copy")
    base_cmd = [wl_copy]
    if sensitive:
        base_cmd.append("--sensitive")
    _run_wl_copy(base_cmd, text, timeout)
    _run_wl_copy(base_cmd + ["--primary"], text, timeout)


def read_clipboard(*, timeout: float = 2.0) -> str | None:
    """Return the current regular (Ctrl+V) clipboard text, or None if empty/unavailable."""
    wl_paste = shutil.which("wl-paste")
    if not wl_paste:
        return None
    try:
        proc = subprocess.run(
            [wl_paste, "--no-newline"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def restore_clipboard(text: str, *, timeout: float = 2.0, sensitive: bool = False) -> None:
    """Restore the regular clipboard only (not primary) — used after paste."""
    wl_copy = _require("wl-copy")
    cmd = [wl_copy]
    if sensitive:
        cmd.append("--sensitive")
    _run_wl_copy(cmd, text, timeout)


def _dotool_input(combo: str) -> bytes:
    # dotool reads commands from stdin; `key` accepts XKB-style
    # modifier+key joined by '+'.
    return f"key {combo}\n".encode("utf-8")


def send_paste_shortcut(
    combo: str = "ctrl+shift+v",
    *,
    backend: str = "dotool",
    timeout: float = 5.0,
) -> None:
    """Synthesise ``combo`` as a keystroke via dotool."""
    if backend != "dotool":
        raise PasteError(f"unknown paste backend: {backend!r}")
    dotool = _require("dotool")
    try:
        proc = subprocess.run(
            [dotool],
            input=_dotool_input(combo),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise PasteError(
            f"dotool timed out after {timeout}s — is the service running?"
        ) from e
    if proc.returncode != 0:
        raise PasteError(f"dotool exited {proc.returncode}")


def paste_text(
    text: str,
    *,
    combo: str = "ctrl+shift+v",
    backend: str = "dotool",
    settle_ms: int = 40,
    timeout: float = 5.0,
    sensitive: bool = False,
) -> None:
    """One-shot copy + paste. Prefer :class:`Paster` if you paste repeatedly —
    it keeps ``dotool`` warm and saves hundreds of ms per paste."""
    if not text:
        return
    copy_to_clipboard(text, timeout=timeout, sensitive=sensitive)
    if settle_ms > 0:
        time.sleep(settle_ms / 1000)
    send_paste_shortcut(combo, backend=backend, timeout=timeout)


def _build_type_payload(text: str) -> bytes:
    """Build the dotool stdin payload to type *text* directly.

    ``dotool type STRING`` types everything up to the newline.  Multi-line
    text is handled by interleaving ``key Return`` commands.
    """
    lines = text.split("\n")
    parts: list[bytes] = []
    for i, line in enumerate(lines):
        if line:
            # Escape backslashes so dotool doesn't misinterpret them.
            parts.append(f"type {line}\n".encode("utf-8"))
        if i < len(lines) - 1:
            parts.append(b"key Return\n")
    return b"".join(parts)


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
        skip_clipboard_history: bool = False,
        type_directly: bool = False,
        restore_clipboard: bool = True,
    ) -> None:
        self.backend = backend
        self.combo = combo
        self.settle_ms = settle_ms
        self.timeout = timeout
        # type_directly takes precedence: inject via ``dotool type``, no clipboard.
        # skip_clipboard_history: use wl-copy --sensitive so text lands in the
        # clipboard (available for Ctrl+V) but clipboard managers skip recording.
        self._type_directly = type_directly and backend == "dotool"
        self._sensitive = skip_clipboard_history and not self._type_directly
        # Restore the previous regular clipboard content after paste so dictation
        # doesn't clobber whatever the user had copied before.
        self._restore_clipboard = restore_clipboard and not self._type_directly
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
        """Inject text, with per-step timing logs."""
        if not text:
            return
        t0 = time.monotonic()
        if self._type_directly:
            # Inject via dotool type — clipboard never touched.
            self._send_dotool_type(text)
            t_key = time.monotonic()
            log.info(
                "paste (type-direct) timings: total=%.0fms",
                (t_key - t0) * 1000,
            )
        else:
            # Snapshot the current clipboard before overwriting it.
            old_clip: str | None = None
            if self._restore_clipboard:
                old_clip = read_clipboard(timeout=self.timeout)

            copy_to_clipboard(text, timeout=self.timeout, sensitive=self._sensitive)
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

            # Restore the previous clipboard after a brief pause so the target
            # app has time to read it before we overwrite it.
            if old_clip is not None:
                time.sleep(0.15)
                try:
                    restore_clipboard(old_clip, timeout=self.timeout, sensitive=self._sensitive)
                except PasteError as e:
                    log.warning("clipboard restore failed: %s", e)

    # --- internals --------------------------------------------------------

    def _send_key(self, combo: str) -> None:
        if self.backend != "dotool":
            raise PasteError(f"unknown paste backend: {self.backend!r}")
        self._send_dotool(combo)

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
        self._write_dotool(payload)

    def _send_dotool_type(self, text: str) -> None:
        payload = _build_type_payload(text)
        if not payload:
            return
        self._write_dotool(payload)

    def _write_dotool(self, payload: bytes) -> None:
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
