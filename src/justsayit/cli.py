"""CLI entry point and glue between audio, transcription, overlay, and paste."""

from __future__ import annotations

# --- gtk4-layer-shell preload ---------------------------------------------
# Layer-shell must be loaded *before* libwayland-client is pulled in. Once
# any `gi.require_version("Gtk", "4.0")` runs, it's too late — wayland is
# already in the process. So before we touch gi, re-exec ourselves with
# LD_PRELOAD set. This is the same fix the gtk4-layer-shell warning
# documents; doing it here means launching from a terminal or the .desktop
# file both just work.
import os as _os
import sys as _sys


def _app_id() -> str:
    """Portal application id used for D-Bus / systemd scoping. Can be
    overridden via ``JUSTSAYIT_APP_ID`` so a dev build can run in
    parallel with an installed build without fighting over the same
    shortcut binding."""
    return _os.environ.get("JUSTSAYIT_APP_ID", "dev.horo.justsayit")


def _reexec_under_systemd_scope() -> None:
    """Re-exec ourselves inside an ``app-<app_id>-*.scope`` cgroup so the
    XDG portal can identify us with a stable name no matter how we were
    launched. Without this, launching from a terminal lands us in the
    shell's scope (e.g. ``konsole.scope``) and the portal generates a
    fresh synthetic id every run — which is why the hotkey bind dialog
    pops on every terminal launch.

    Skipped inside Flatpak/Snap (already scoped by the sandbox), if
    ``systemd-run`` is missing, if we already re-execed once, or if the
    current cgroup already contains ``app-<app_id>``.
    """
    if _os.environ.get("_JUSTSAYIT_SCOPED") == "1":
        return
    if _os.environ.get("FLATPAK_ID") or _os.environ.get("SNAP"):
        return
    # Check current cgroup — if we're already in our app scope, nothing to do.
    app_id = _app_id()
    scope_marker = f"app-{app_id}"
    try:
        with open("/proc/self/cgroup", "r") as f:
            cg = f.read()
        if scope_marker in cg:
            return
    except OSError:
        # No cgroup info? Fall through and try to scope anyway.
        pass
    # systemd-run must exist on PATH.
    import shutil

    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        return
    env = _os.environ.copy()
    env["_JUSTSAYIT_SCOPED"] = "1"
    unit = f"app-{app_id}-{_os.getpid()}"
    argv = [
        systemd_run,
        "--user",
        "--scope",
        f"--unit={unit}",
        "--quiet",
        "--collect",
        _sys.executable,
        *_sys.argv,
    ]
    _os.execvpe(systemd_run, argv, env)


def _find_layer_shell_lib() -> str | None:
    candidates = [
        "/usr/lib/libgtk4-layer-shell.so",
        "/usr/lib64/libgtk4-layer-shell.so",
        "/usr/local/lib/libgtk4-layer-shell.so",
        "/usr/lib/x86_64-linux-gnu/libgtk4-layer-shell.so",
    ]
    for p in candidates:
        if _os.path.exists(p):
            return p
    # Fall back to ldconfig for odd layouts.
    try:
        import subprocess

        out = subprocess.check_output(["ldconfig", "-p"], text=True)
        for line in out.splitlines():
            if "libgtk4-layer-shell.so" in line and "=>" in line:
                return line.split("=>", 1)[1].strip()
    except Exception:
        pass
    return None


def _preload_layer_shell() -> None:
    if _os.environ.get("_JUSTSAYIT_PRELOADED") == "1":
        return
    lib = _find_layer_shell_lib()
    if lib is None:
        # Continue without preload; overlay will likely warn but the rest of
        # the app still works.
        return
    env = _os.environ.copy()
    existing = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{lib}:{existing}" if existing else lib
    env["_JUSTSAYIT_PRELOADED"] = "1"
    _os.execvpe(_sys.executable, [_sys.executable, *_sys.argv], env)


_reexec_under_systemd_scope()
_preload_layer_shell()

# --- regular imports below; safe now -------------------------------------
import argparse
import logging
import logging.handlers
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk  # noqa: E402

from justsayit import __version__
from justsayit.audio import AudioEngine, Segment, State
from justsayit.config import (
    Config,
    cache_dir,
    config_dir,
    default_config_toml,
    ensure_config_file,
    ensure_dirs,
    ensure_filters_file,
    load_config,
    save_config,
)
from justsayit.filters import apply_filters, load_filters
from justsayit.model import ensure_models
from justsayit.overlay import OverlayWindow
from justsayit.paste import PasteError, Paster
from justsayit.shortcuts import GlobalShortcutClient
from justsayit.sound import SoundPlayer
from justsayit.transcribe import Transcriber
from justsayit.tray import MenuItem, TrayIcon, open_with_xdg

ICON_ACTIVE = "audio-input-microphone"
ICON_PAUSED = "microphone-sensitivity-muted-symbolic"

# Stable menu item ids so we can update them in place.
MID_AUTO_LISTEN = 1
MID_SEP_1 = 2
MID_CONFIGURE_SHORTCUT = 3
MID_OPEN_CONFIG = 4
MID_RELOAD_CONFIG = 7
MID_SEP_2 = 5
MID_QUIT = 6

log = logging.getLogger("justsayit")

DEFAULT_FILTERS = [
    {
        "name": "trim whitespace",
        "pattern": r"^\s+|\s+$",
        "replacement": "",
    },
    {
        "name": "collapse whitespace",
        "pattern": r"\s{2,}",
        "replacement": " ",
    },
]


class App:
    def __init__(self, cfg: Config, *, no_overlay: bool, no_paste: bool) -> None:
        self.cfg = cfg
        self.no_overlay = no_overlay
        self.no_paste = no_paste

        self.model_paths = None  # set in setup_models
        self.transcriber: Transcriber | None = None
        self.engine: AudioEngine | None = None
        self.overlay: OverlayWindow | None = None
        self.shortcut_client: GlobalShortcutClient | None = None
        self.paster: Paster | None = None
        self.sound_player: SoundPlayer | None = None
        self.tray: TrayIcon | None = None
        self.gtk_app: Gtk.Application | None = None
        self.filters = []

        # Bounded queue so we can't run unbounded transcription work if the
        # user is trigger-happy with the hotkey.
        self._seg_q: queue.Queue[Segment | None] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._transcribe_thread: threading.Thread | None = None
        # Monotonic timestamp of the last successful transcription output,
        # used by the auto_space_timeout_ms feature.
        self._last_transcription_time: float | None = None
        self._restart_requested: bool = False

    # --- setup -------------------------------------------------------------

    def setup_models(self) -> None:
        # Always fetch the VAD model so the tray's auto-listen toggle
        # works regardless of whether it was enabled at startup.
        self.model_paths = ensure_models(self.cfg, want_vad=True)

    def setup_filters(self) -> None:
        self.filters = load_filters(self.cfg.filters_path)
        log.info(
            "loaded %d filter(s) from %s",
            len(self.filters),
            self.cfg.filters_path,
        )

    def setup_transcriber(self) -> None:
        assert self.model_paths is not None
        self.transcriber = Transcriber(self.cfg, self.model_paths)
        log.info("warming up Parakeet recognizer…")
        self.transcriber.warmup()

    def setup_sound(self) -> None:
        if not self.cfg.sound.enabled:
            log.info("sound effects disabled")
            return
        self.sound_player = SoundPlayer(volume=self.cfg.sound.volume)
        log.info("sound player ready (volume=%.2f)", self.cfg.sound.volume)

    def setup_audio(self) -> None:
        assert self.transcriber is not None
        assert self.model_paths is not None

        def validate(samples, sr):
            # Called from the audio thread. Parakeet on a 3s clip is fast.
            assert self.transcriber is not None
            try:
                return self.transcriber.has_words(samples, sr)
            except Exception:
                log.exception("validation transcription failed")
                return False

        def on_segment(seg: Segment) -> None:
            log.info(
                "queueing segment for transcription: %.2fs reason=%s",
                len(seg.samples) / seg.sample_rate,
                seg.reason,
            )
            try:
                self._seg_q.put_nowait(seg)
            except queue.Full:
                log.warning(
                    "transcription queue full; dropping %.2fs segment",
                    len(seg.samples) / seg.sample_rate,
                )

        _active = {State.RECORDING, State.MANUAL}
        prev_state: list[State] = [State.IDLE]  # mutable cell for closure

        def on_state(state: State) -> None:
            log.debug("engine state callback: %s", state.value)
            prev = prev_state[0]
            prev_state[0] = state
            if self.overlay is not None:
                self.overlay.push_state(state)
            if self.sound_player is not None:
                if state in _active and prev not in _active:
                    self.sound_player.play_start()
                elif state is State.IDLE and prev in _active:
                    self.sound_player.play_stop()

        def on_level(rms: float) -> None:
            if self.overlay is not None:
                self.overlay.push_level(rms)

        # Skip loading VAD entirely when disabled.
        vad_path = self.model_paths.vad if self.cfg.vad.enabled else None
        self.engine = AudioEngine(
            self.cfg,
            vad_path,
            validate_words=validate,
            on_segment=on_segment,
            on_state=on_state,
            on_level=on_level,
        )

    def setup_transcribe_thread(self) -> None:
        def run():
            while not self._stop.is_set():
                try:
                    seg = self._seg_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if seg is None:
                    break
                try:
                    self._handle_segment(seg)
                except Exception:
                    log.exception("segment handling failed")

        self._transcribe_thread = threading.Thread(
            target=run, name="justsayit-transcribe", daemon=True
        )
        self._transcribe_thread.start()

    def setup_paster(self) -> None:
        if self.no_paste or not self.cfg.paste.enabled:
            log.info("paste disabled; not starting paster")
            return
        self.paster = Paster(
            backend=self.cfg.paste.backend,
            combo=self.cfg.paste.paste_combo,
            settle_ms=self.cfg.paste.settle_ms,
            timeout=self.cfg.paste.subprocess_timeout,
        )
        try:
            self.paster.start()
            log.info(
                "paster ready (backend=%s combo=%s)",
                self.cfg.paste.backend,
                self.cfg.paste.paste_combo,
            )
        except PasteError as e:
            log.error("paster failed to start: %s", e)
            self.paster = None

    def setup_shortcut(self) -> None:
        def on_activated(shortcut_id: str) -> None:
            if self.engine is None:
                log.warning("hotkey fired before audio engine ready")
                return
            state = self.engine.state
            log.info("HOTKEY %s fired (state=%s)", shortcut_id, state.value)
            if self.engine.vad_enabled:
                # Auto-listen ON: hotkey pauses / resumes VAD triggering
                # (transient, not persisted). Stop any in-flight recording
                # first so the user lands in a quiet state with one press.
                if state in (State.VALIDATING, State.RECORDING, State.MANUAL):
                    self.engine.stop_manual()
                self.engine.set_vad_paused(not self.engine.vad_paused)
                self._sync_tray_and_icon()
            else:
                # Auto-listen OFF: hotkey starts / stops a manual recording.
                if state in (State.IDLE, State.VALIDATING):
                    self.engine.start_manual()
                else:
                    self.engine.stop_manual()

        def on_needs_rebinding(shortcut_id: str, reason: str) -> None:
            self._notify_shortcut_unbound(shortcut_id, reason)

        self.shortcut_client = GlobalShortcutClient(
            shortcut_id=self.cfg.shortcut.id,
            description=self.cfg.shortcut.description,
            preferred_trigger=self.cfg.shortcut.preferred,
            on_activated=on_activated,
            on_needs_rebinding=on_needs_rebinding,
        )

    def _notify_shortcut_unbound(self, shortcut_id: str, reason: str) -> None:
        """Surface a desktop notification when the portal can't open its
        own rebind dialog (old v1 GlobalShortcuts). We only hit this
        from the unbound-shortcut code path, so the user actually has
        something actionable to do."""
        if self.gtk_app is None:
            return
        title = "justsayit: shortcut not assigned"
        body = (
            f"The “{shortcut_id}” shortcut has no trigger. Open your "
            "desktop environment's shortcut settings to assign one "
            "(KDE: System Settings → Shortcuts; "
            "GNOME: Settings → Keyboard → View and Customize Shortcuts)."
        )
        try:
            note = Gio.Notification.new(title)
            note.set_body(body)
            note.set_priority(Gio.NotificationPriority.HIGH)
            # Stable id so repeated calls update a single notification
            # instead of spamming new ones.
            self.gtk_app.send_notification(
                f"justsayit-shortcut-unbound-{shortcut_id}", note
            )
            log.info("sent 'shortcut unbound' notification (reason=%s)", reason)
        except Exception:
            log.exception("could not send shortcut-unbound notification")

    def _notify_space_config_conflict(self) -> None:
        """Warn the user when both auto_space_timeout_ms and
        append_trailing_space are enabled — only the trailing-space
        behaviour will be active."""
        if self.gtk_app is None:
            return
        title = "justsayit: conflicting space settings"
        body = (
            "Both paste.auto_space_timeout_ms and paste.append_trailing_space "
            "are enabled. These settings conflict: the trailing space already "
            "acts as a separator, so the auto-prefix space has been disabled. "
            "Set auto_space_timeout_ms = 0 to suppress this warning."
        )
        try:
            note = Gio.Notification.new(title)
            note.set_body(body)
            note.set_priority(Gio.NotificationPriority.NORMAL)
            self.gtk_app.send_notification("justsayit-space-conflict", note)
            log.warning("space config conflict: auto_space_timeout_ms ignored")
        except Exception:
            log.exception("could not send space-conflict notification")

    def setup_tray(self) -> None:
        def on_toggle_auto_listen() -> None:
            if self.engine is None:
                return
            new_enabled = not self.engine.vad_enabled
            self.engine.set_vad_enabled(new_enabled)
            try:
                save_config(self.cfg)
                log.info(
                    "persisted auto-listen=%s to config.toml",
                    new_enabled,
                )
            except Exception:
                log.exception("failed to persist auto-listen state")
            self._sync_tray_and_icon()

        def on_open_config() -> None:
            cfg_path = ensure_config_file()
            log.info("opening config file: %s", cfg_path)
            open_with_xdg(str(cfg_path))

        def on_configure_shortcut() -> None:
            if self.shortcut_client is None:
                log.warning("no shortcut client to configure")
                return
            log.info("opening shortcut configuration dialog")
            self.shortcut_client.configure()

        def on_reload_config() -> None:
            log.info("reload-config requested from tray — restarting")
            self._restart_requested = True
            if self.gtk_app is not None:
                self.gtk_app.quit()

        def on_quit() -> None:
            log.info("quit requested from tray")
            if self.gtk_app is not None:
                self.gtk_app.quit()

        assert self.engine is not None
        items = [
            MenuItem(
                id=MID_AUTO_LISTEN,
                label="Auto-listen",
                toggle_type="checkmark",
                toggle_state=1 if self.engine.vad_enabled else 0,
                on_activate=on_toggle_auto_listen,
            ),
            MenuItem(id=MID_SEP_1, is_separator=True),
            MenuItem(
                id=MID_CONFIGURE_SHORTCUT,
                label="Configure shortcut…",
                on_activate=on_configure_shortcut,
            ),
            MenuItem(
                id=MID_OPEN_CONFIG,
                label="Open config file…",
                on_activate=on_open_config,
            ),
            MenuItem(
                id=MID_RELOAD_CONFIG,
                label="Reload config",
                on_activate=on_reload_config,
            ),
            MenuItem(id=MID_SEP_2, is_separator=True),
            MenuItem(id=MID_QUIT, label="Quit", on_activate=on_quit),
        ]
        self.tray = TrayIcon(
            icon_name=ICON_ACTIVE if self.engine.vad_active else ICON_PAUSED,
            tooltip=self._tray_tooltip(),
            items=items,
        )
        try:
            self.tray.start()
        except Exception:
            log.exception("tray icon failed to start")
            self.tray = None

    def _tray_tooltip(self) -> str:
        if self.engine is None:
            return "justsayit"
        if not self.engine.vad_enabled:
            return "Auto-listen off — hotkey records manually"
        if self.engine.vad_paused:
            return "Auto-listen paused — hotkey resumes"
        return "Auto-listen on — hotkey pauses"

    def _sync_tray_and_icon(self) -> None:
        """Reflect current engine state in the tray menu + icon. Safe to
        call even if the tray isn't running."""
        if self.tray is None or self.engine is None:
            return
        # Checkbox reflects the persisted master switch.
        self.tray.update_item(
            MID_AUTO_LISTEN,
            toggle_state=1 if self.engine.vad_enabled else 0,
        )
        # Icon reflects the live (enabled AND not paused) state so the
        # user can see at a glance whether VAD is actually listening.
        self.tray.set_icon(
            ICON_ACTIVE if self.engine.vad_active else ICON_PAUSED
        )
        self.tray.set_tooltip(self._tray_tooltip())

    # --- runtime -----------------------------------------------------------

    def _handle_segment(self, seg: Segment) -> None:
        assert self.transcriber is not None
        duration = len(seg.samples) / seg.sample_rate
        log.info("transcribing %.2fs (reason=%s)", duration, seg.reason)
        t0 = time.monotonic()
        raw = self.transcriber.transcribe(seg.samples, seg.sample_rate)
        dt = time.monotonic() - t0
        log.info("transcription done in %.2fs: raw=%r", dt, raw)
        if not raw:
            log.info("empty transcription; nothing to paste")
            return
        final = apply_filters(raw, self.filters)
        if final != raw:
            log.info("filters changed output: %r -> %r", raw, final)

        # Space prefix / suffix
        auto_space_ms = self.cfg.paste.auto_space_timeout_ms
        trailing_space = self.cfg.paste.append_trailing_space
        now = time.monotonic()

        if auto_space_ms > 0 and not trailing_space:
            if self._last_transcription_time is not None:
                seg_duration = len(seg.samples) / seg.sample_rate
                recording_started_at = now - seg_duration
                elapsed_ms = (recording_started_at - self._last_transcription_time) * 1000.0
                if elapsed_ms <= auto_space_ms:
                    log.debug(
                        "auto-space: elapsed=%.0fms ≤ timeout=%dms — prepending space",
                        elapsed_ms,
                        auto_space_ms,
                    )
                    final = " " + final

        if trailing_space:
            final = final + " "

        self._last_transcription_time = now

        print(final, flush=True)
        if self.no_paste or not self.cfg.paste.enabled:
            log.info("paste disabled — text only printed")
            return

        # Give the user a moment to let go of the stop-hotkey modifiers
        # before we synthesise ctrl+shift+v, otherwise the compositor may
        # see e.g. "Super+Ctrl+Shift+V" and not paste.
        if seg.stop_requested_at is not None:
            delay_target = self.cfg.paste.release_delay_ms / 1000.0
            elapsed = time.monotonic() - seg.stop_requested_at
            wait = delay_target - elapsed
            if wait > 0:
                log.info(
                    "waiting %.0fms for hotkey modifiers to release "
                    "(elapsed since stop=%.0fms, target=%.0fms)",
                    wait * 1000,
                    elapsed * 1000,
                    delay_target * 1000,
                )
                time.sleep(wait)
            else:
                log.info(
                    "processing already took %.0fms ≥ release target %.0fms; "
                    "pasting immediately",
                    elapsed * 1000,
                    delay_target * 1000,
                )

        if self.paster is None:
            log.warning("paster not ready; skipping paste")
            return
        try:
            log.info(
                "pasting %d chars via %s (backend=%s)",
                len(final),
                self.cfg.paste.paste_combo,
                self.cfg.paste.backend,
            )
            self.paster.paste(final)
        except PasteError as e:
            log.error("paste failed: %s", e)

    # --- GTK lifecycle -----------------------------------------------------

    def on_activate(self, app: Gtk.Application) -> None:
        # Keep the app alive even when the overlay is hidden (which is the
        # normal state now: overlay only shows while recording).
        app.hold()
        self.gtk_app = app
        if not self.no_overlay:
            self.overlay = OverlayWindow(app, self.cfg)
            # Explicitly hidden until the engine reports a non-idle state.
            self.overlay.set_visible(False)

        # Defer the heavy setup out of the activate handler so the UI loop
        # can keep ticking while we load models.
        def _later():
            try:
                self.setup_models()
                self.setup_filters()
                self.setup_transcriber()
                self.setup_sound()
                self.setup_audio()
                self.setup_transcribe_thread()
                self.setup_paster()
                assert self.engine is not None
                self.engine.start()
                self.setup_shortcut()
                assert self.shortcut_client is not None
                self.shortcut_client.start()
                self.setup_tray()
                if (
                    self.cfg.paste.auto_space_timeout_ms > 0
                    and self.cfg.paste.append_trailing_space
                ):
                    self._notify_space_config_conflict()
                log.info("justsayit ready")
            except Exception:
                log.exception("startup failed")
                app.quit()
            return False

        GLib.idle_add(_later)

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._seg_q.put_nowait(None)
        except queue.Full:
            pass
        if self.engine is not None:
            self.engine.stop()
        if self.shortcut_client is not None:
            self.shortcut_client.stop()
        if self._transcribe_thread is not None:
            self._transcribe_thread.join(timeout=2.0)
        if self.paster is not None:
            self.paster.close()
        if self.tray is not None:
            self.tray.stop()


# --- argparse --------------------------------------------------------------


def _write_default_config(force: bool = False) -> None:
    ensure_dirs()
    cfg_path = config_dir() / "config.toml"
    filters_path = config_dir() / "filters.json"

    if cfg_path.exists() and not force:
        print(f"config already exists: {cfg_path}", file=sys.stderr)
    else:
        cfg_path.write_text(default_config_toml(), encoding="utf-8")
        print(f"wrote {cfg_path}")

    if filters_path.exists() and not force:
        print(f"filters already exist: {filters_path}", file=sys.stderr)
    else:
        import json

        filters_path.write_text(
            json.dumps(DEFAULT_FILTERS, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {filters_path}")


def _download_models_only() -> int:
    ensure_dirs()
    cfg = load_config()
    p = ensure_models(cfg, force=False)
    print(f"models ready:\n  encoder: {p.encoder}\n  vad:     {p.vad}")
    return 0


def _setup_file_logging(cfg: Config, console_level: int) -> None:
    """Attach a rotating file handler to the root logger if the config
    has ``log.file_enabled=True``. The console handler (installed by
    ``logging.basicConfig``) is pinned to ``console_level`` so enabling a
    verbose file log doesn't also flood the terminal."""
    root = logging.getLogger()

    # Pin whatever the basicConfig handler installed to the console level
    # the user asked for, regardless of what we do with the root level.
    for h in root.handlers:
        if not isinstance(h, logging.handlers.RotatingFileHandler):
            h.setLevel(console_level)

    if not cfg.log.file_enabled:
        return

    path = cfg.log.file_path.strip() or str(cache_dir() / "justsayit.log")
    resolved = Path(path).expanduser()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            resolved,
            maxBytes=cfg.log.file_max_bytes,
            backupCount=cfg.log.file_backup_count,
            encoding="utf-8",
        )
    except OSError as e:
        log.error("could not open debug log file %s: %s", resolved, e)
        return

    try:
        file_level = getattr(logging, cfg.log.file_level.upper())
    except AttributeError:
        log.warning(
            "unknown log.file_level=%r — falling back to DEBUG",
            cfg.log.file_level,
        )
        file_level = logging.DEBUG
    handler.setLevel(file_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    # Root must be at least as verbose as the most verbose handler.
    root.setLevel(min(console_level, file_level))
    log.info(
        "debug log file enabled: %s (level=%s, max_bytes=%d, backups=%d)",
        resolved,
        logging.getLevelName(file_level),
        cfg.log.file_max_bytes,
        cfg.log.file_backup_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="justsayit")
    ap.add_argument("--version", action="version", version=f"justsayit {__version__}")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    ap.add_argument("--no-overlay", action="store_true", help="run without the overlay")
    ap.add_argument("--no-paste", action="store_true", help="don't auto-paste output")
    vad = ap.add_mutually_exclusive_group()
    vad.add_argument(
        "--vad",
        dest="vad",
        action="store_true",
        default=None,
        help="enable auto-start VAD (overrides config)",
    )
    vad.add_argument(
        "--no-vad",
        dest="vad",
        action="store_false",
        default=None,
        help="disable VAD; record only on hotkey (overrides config)",
    )
    ap.add_argument(
        "--config", type=Path, help="override path to config.toml"
    )
    sub = ap.add_subparsers(dest="subcommand")
    sub.add_parser("init", help="write default config + example filters")
    sub.add_parser("download-models", help="pre-download Parakeet + VAD models")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    console_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=console_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.subcommand == "init":
        _write_default_config(force=False)
        return 0
    if args.subcommand == "download-models":
        return _download_models_only()

    ensure_dirs()
    # Seed defaults on first run so `Open config file…` in the tray always
    # opens a useful, fully-documented file. Honors --config: if a custom
    # path is given, we only touch that path (not the XDG default).
    if args.config is None:
        ensure_config_file()
        ensure_filters_file()
    cfg = load_config(args.config)
    _setup_file_logging(cfg, console_level)
    if args.vad is not None:
        cfg.vad.enabled = args.vad
    log.info(
        "startup: vad=%s overlay=%s paste=%s",
        cfg.vad.enabled,
        (not args.no_overlay),
        (cfg.paste.enabled and not args.no_paste),
    )

    app = Gtk.Application.new(_app_id(), Gio.ApplicationFlags.FLAGS_NONE)
    ja = App(cfg, no_overlay=args.no_overlay, no_paste=args.no_paste)
    app.connect("activate", ja.on_activate)

    # Clean shutdown on Ctrl-C.
    def _handle_sigint(*_):
        log.info("SIGINT received; shutting down")
        ja.shutdown()
        app.quit()

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _handle_sigint)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _handle_sigint)

    rc = app.run([])
    ja.shutdown()
    if ja._restart_requested:
        log.info("restarting process to pick up new config")
        _os.execve(_sys.executable, [_sys.executable] + _sys.argv, _os.environ)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
