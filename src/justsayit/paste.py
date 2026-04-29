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

    Both ``wl-copy`` invocations run in parallel — wl-copy daemonises by
    default (per its man page), so each call returns as soon as the
    parent has handed the text off to the daemon. On well-behaved
    Wayland setups each call completes in ~10–20 ms; running them
    concurrently keeps the worst case at single-call latency instead of
    summing both. Some compositors / clipboard managers (KDE Klipper,
    GNOME with sluggish portal pipelines) can take 100 ms+ per call —
    parallel execution halves that wall-clock too.

    When ``sensitive`` is True, ``--sensitive`` is passed to wl-copy so that
    clipboard managers (e.g. KDE Klipper) skip recording this entry.  The text
    is still available for a manual Ctrl+V / Shift+Insert / middle-click paste.

    Both selections must be set BEFORE the paste keystroke fires:
    Shift+Insert and middle-click read different selections depending on
    the toolkit / app, so deferring either one risks pasting stale text.
    """
    if not text:
        return
    wl_copy = _require("wl-copy")
    # ``--type text/plain`` skips wl-copy's auto-MIME-inference, which
    # would otherwise fork ``xdg-mime`` per call (~14 ms on a fast box,
    # 100 ms+ on a busy KDE/GNOME session). The advertised MIME types
    # (``text/plain``, ``text/plain;charset=utf-8``, ``UTF8_STRING``,
    # ``TEXT``, ``STRING``) are identical either way — wl-copy always
    # offers the standard text variants — so this is a pure latency win.
    base_cmd = [wl_copy, "--type", "text/plain"]
    if sensitive:
        base_cmd.append("--sensitive")
    cmds = [base_cmd, base_cmd + ["--primary"]]
    results: list[tuple[float, BaseException | None]] = [(0.0, None)] * len(cmds)

    def _one(idx: int, cmd: list[str]) -> None:
        t0 = time.monotonic()
        try:
            _run_wl_copy(cmd, text, timeout)
            results[idx] = ((time.monotonic() - t0) * 1000, None)
        except BaseException as exc:
            results[idx] = ((time.monotonic() - t0) * 1000, exc)

    threads = [threading.Thread(target=_one, args=(i, c)) for i, c in enumerate(cmds)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.debug(
        "wl-copy parallel: regular=%.0fms primary=%.0fms",
        results[0][0],
        results[1][0],
    )
    for ms, exc in results:
        if exc is not None:
            raise exc


_TEXT_MIME_PREFERENCE = (
    "text/plain;charset=utf-8",
    "text/plain",
    "UTF8_STRING",
    "STRING",
    "TEXT",
)

_IMAGE_MIME_PREFERENCE = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
)


def _pick_text_mime(offered: list[str]) -> str | None:
    """Pick the best text MIME from what the clipboard advertises.
    Returns ``None`` when no text type is offered (e.g. an image-only
    clipboard) — signals the caller to skip instead of decoding binary
    bytes as UTF-8."""
    for pref in _TEXT_MIME_PREFERENCE:
        if pref in offered:
            return pref
    # Fallback: any ``text/*`` type we don't know by name.
    for t in offered:
        if t.startswith("text/"):
            return t
    return None


def read_clipboard(*, timeout: float = 2.0, text_only: bool = False) -> str | None:
    """Return the current regular (Ctrl+V) clipboard contents as a
    string, or ``None`` if empty / unavailable.

    With ``text_only=True`` the function first probes ``wl-paste
    --list-types`` and returns ``None`` when the clipboard only offers
    non-text MIME types (images, files, …). Without the guard an
    image-only clipboard would be decoded as UTF-8-with-replacement and
    produce kilobytes of ``\\ufffd`` noise — fine for a raw-bytes
    snapshot (paste-restore path keeps the default), catastrophic if
    the result is about to be fed to an LLM as context.
    """
    wl_paste = shutil.which("wl-paste")
    if not wl_paste:
        return None
    mime: str | None = None
    if text_only:
        try:
            probe = subprocess.run(
                [wl_paste, "--list-types"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if probe.returncode != 0:
            return None
        offered = probe.stdout.decode("ascii", "replace").split()
        mime = _pick_text_mime(offered)
        if mime is None:
            log.info(
                "clipboard has no text MIME type (offered: %s); skipping text read",
                ", ".join(offered) or "<none>",
            )
            return None
    cmd = [wl_paste, "--no-newline"]
    if mime is not None:
        cmd += ["--type", mime]
    try:
        proc = subprocess.run(
            cmd,
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


def read_clipboard_image(*, timeout: float = 2.0) -> tuple[bytes, str] | None:
    """Return ``(raw_bytes, mime_type)`` for an image on the clipboard, or ``None``.

    Probes ``wl-paste --list-types`` and picks the first supported image MIME
    type (png > jpeg > webp > gif). Returns ``None`` if no image is offered,
    ``wl-paste`` is unavailable, or reading fails.
    """
    wl_paste = shutil.which("wl-paste")
    if not wl_paste:
        return None
    try:
        probe = subprocess.run(
            [wl_paste, "--list-types"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if probe.returncode != 0:
        return None
    offered = probe.stdout.decode("ascii", "replace").split()
    mime = next((m for m in _IMAGE_MIME_PREFERENCE if m in offered), None)
    if mime is None:
        return None
    try:
        proc = subprocess.run(
            [wl_paste, "--type", mime],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout, mime


def restore_clipboard(text: str, *, timeout: float = 2.0, sensitive: bool = False) -> None:
    """Restore the regular clipboard only (not primary) — used after paste."""
    wl_copy = _require("wl-copy")
    # See ``copy_to_clipboard`` for why ``--type text/plain`` is set.
    cmd = [wl_copy, "--type", "text/plain"]
    if sensitive:
        cmd.append("--sensitive")
    _run_wl_copy(cmd, text, timeout)


def _restore_clipboard_image(data: bytes, mime_type: str, *, timeout: float = 2.0) -> None:
    """Restore a binary image to the regular clipboard via wl-copy."""
    wl_copy = _require("wl-copy")
    try:
        proc = subprocess.run(
            [wl_copy, "--type", mime_type],
            input=data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise PasteError(f"wl-copy timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise PasteError(f"wl-copy exited {proc.returncode}")


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
        restore_delay_ms: int = 250,
    ) -> None:
        self.backend = backend
        self.combo = combo
        self.settle_ms = settle_ms
        self.restore_delay_ms = restore_delay_ms
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
            log.debug(
                "paste (type-direct) timings: total=%.0fms",
                (t_key - t0) * 1000,
            )
        else:
            # Snapshot the current clipboard before overwriting it. wl-paste
            # blocks until the source app responds — usually fast but worth
            # surfacing in the timing line so we can spot stalls.
            # Try image first so we can restore it as binary (not as mangled text).
            old_clip: str | None = None
            old_clip_img: tuple[bytes, str] | None = None
            if self._restore_clipboard:
                old_clip_img = read_clipboard_image(timeout=self.timeout)
                if old_clip_img is None:
                    old_clip = read_clipboard(timeout=self.timeout)
            t_snap = time.monotonic()

            copy_to_clipboard(text, timeout=self.timeout, sensitive=self._sensitive)
            t_copy = time.monotonic()
            if self.settle_ms > 0:
                time.sleep(self.settle_ms / 1000)
            t_settle = time.monotonic()
            self._send_key(self.combo)
            t_key = time.monotonic()
            log.debug(
                "paste timings: snap=%.0fms copy=%.0fms settle=%.0fms key=%.0fms total=%.0fms",
                (t_snap - t0) * 1000,
                (t_copy - t_snap) * 1000,
                (t_settle - t_copy) * 1000,
                (t_key - t_settle) * 1000,
                (t_key - t0) * 1000,
            )

            # Restore the previous clipboard after a brief pause so the target
            # app has time to read it before we overwrite it.
            if old_clip_img is not None or old_clip is not None:
                if self.restore_delay_ms > 0:
                    time.sleep(self.restore_delay_ms / 1000)
                try:
                    if old_clip_img is not None:
                        _restore_clipboard_image(old_clip_img[0], old_clip_img[1], timeout=self.timeout)
                    else:
                        restore_clipboard(old_clip, timeout=self.timeout, sensitive=self._sensitive)  # type: ignore[arg-type]
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
