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


def _reexec_cmd() -> list[str]:
    """Return the argv to re-execute the current process.

    When running inside a Nix ``makeBinaryWrapper`` ELF wrapper, ``sys.argv[0]``
    is the ELF binary path, not a Python script. Passing it to
    ``[sys.executable, *sys.argv]`` would make Python try to execute a binary
    as source code. Detect this case and exec the ELF directly instead —
    the wrapper handles Python setup on its own.
    """
    argv0 = _sys.argv[0]
    try:
        with open(argv0, "rb") as f:
            if f.read(4) == b"\x7fELF":
                return [argv0] + _sys.argv[1:]
    except OSError:
        pass
    return [_sys.executable, *_sys.argv]


def _app_id() -> str:
    """Portal application id used for D-Bus / systemd scoping. Can be
    overridden via ``JUSTSAYIT_APP_ID`` so a dev build can run in
    parallel with an installed build without fighting over the same
    shortcut binding."""
    return _os.environ.get("JUSTSAYIT_APP_ID", "dev.horotw.justsayit")


def _relaunch_via_desktop() -> bool:
    """Spawn a fresh instance via the installed ``.desktop`` file so the
    desktop env (KDE/GNOME) places it in a portal-recognized systemd
    scope with a stable, well-formed app id. Returns ``True`` if a new
    instance was launched (caller should exit), ``False`` otherwise.

    Why this matters for the restart-from-tray flow: when launched from a
    terminal we self-create ``app-<app_id>-<pid>.scope`` via systemd-run.
    That scope name isn't always parsed back to our app id by the XDG
    portal's app-id resolver, so on the second D-Bus connection (after
    in-place execve) BindShortcuts can come back unassigned — the user
    sees the bind dialog pop again. Going through gio-launch-desktop
    (what ``Gio.DesktopAppInfo.launch`` does under the hood) makes the
    desktop env own the scope naming, which the portal then recognizes
    consistently across launches.
    """
    try:
        from gi.repository import Gio  # local import: avoids loading gi
        # before LD_PRELOAD reexec on cold start.

        info = Gio.DesktopAppInfo.new(_app_id() + ".desktop")
    except Exception:
        return False
    if info is None:
        return False
    try:
        return bool(info.launch([], None))
    except Exception:
        return False


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
    reexec = _reexec_cmd()
    argv = [
        systemd_run,
        "--user",
        "--scope",
        f"--unit={unit}",
        "--quiet",
        "--collect",
        *reexec,
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
    cmd = _reexec_cmd()
    _os.execvpe(cmd[0], cmd, env)


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
    ensure_config_file,
    ensure_dirs,
    ensure_filters_file,
    load_config,
    render_config_toml,
    save_config,
)
from justsayit.filters import apply_filters, load_filters
from justsayit.model import ensure_models, ensure_vad
from justsayit.postprocess import (
    KNOWN_LLM_MODELS,
    LLMPostprocessor,
    _CLEANUP_PROFILE_TOML,
    _CONTEXT_SIDECAR_TEMPLATE,
    _FUN_PROFILE_TOML,
    _OPENAI_PROFILE_TOML,
    download_llm_model,
    ensure_default_profile,
    ensure_default_profiles,
    ensure_dynamic_context_script,
    find_hf_q4_filename,
    load_profile,
    profiles_dir,
    update_profile_model,
)
from justsayit.overlay import OverlayWindow
from justsayit.paste import PasteError, Paster
from justsayit.shortcuts import GlobalShortcutClient
from justsayit.sound import SoundPlayer
from justsayit.transcribe import TranscriberBase, make_transcriber
from justsayit.tray import MenuItem, TrayIcon, open_with_xdg

ICON_ACTIVE = "audio-input-microphone"
ICON_PAUSED = "microphone-sensitivity-muted-symbolic"

# Stable menu item ids so we can update them in place.
MID_AUTO_LISTEN = 1
MID_SEP_1 = 2
MID_CONFIGURE_SHORTCUT = 3
MID_OPEN_CONFIG = 4
MID_RELOAD_CONFIG = 7
MID_LLM_SUBMENU = 8  # "LLM: <profile>" parent item
MID_LLM_OFF = 9  # "Off" radio item inside the LLM submenu
MID_LLM_SEP_INNER = 10  # separator before the "Off" item
MID_SEP_2 = 5
MID_QUIT = 6
# LLM profile radio items occupy IDs 100, 101, 102 … (one per profile file)
MID_LLM_PROFILE_BASE = 100

log = logging.getLogger("justsayit")


class App:
    def __init__(self, cfg: Config, *, no_overlay: bool, no_paste: bool) -> None:
        self.cfg = cfg
        self.no_overlay = no_overlay
        self.no_paste = no_paste

        self.model_paths = None  # set in setup_models (Parakeet only)
        self.vad_path = None  # set in setup_models (all backends)
        self.transcriber: TranscriberBase | None = None
        self.engine: AudioEngine | None = None
        self.overlay: OverlayWindow | None = None
        self.shortcut_client: GlobalShortcutClient | None = None
        self.paster: Paster | None = None
        self.sound_player: SoundPlayer | None = None
        self.postprocessor: LLMPostprocessor | None = None
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
        if self.cfg.model.backend == "openai":
            # Remote STT — no local ASR model to download. VAD is still
            # local (we don't want to stream audio over the network just
            # to detect silence) so fetch the tiny silero ONNX.
            self.vad_path = ensure_vad(self.cfg)
            self.model_paths = None
            log.info("openai backend: VAD ready, transcription via remote endpoint")
        elif self.cfg.model.backend == "whisper":
            self.vad_path = ensure_vad(self.cfg)
            self.model_paths = None
            log.info("whisper backend: VAD ready, Whisper model loads on first use")
        else:
            self.model_paths = ensure_models(self.cfg, want_vad=True)
            self.vad_path = self.model_paths.vad

    def setup_filters(self) -> None:
        self.filters = load_filters(self.cfg.filters_path)
        log.info(
            "loaded %d filter(s) from %s",
            len(self.filters),
            self.cfg.filters_path,
        )

    def setup_transcriber(self) -> None:
        self.transcriber = make_transcriber(self.cfg, self.model_paths)
        log.info("warming up %s recognizer…", self.cfg.model.backend)
        self.transcriber.warmup()

    def setup_postprocessor(self) -> None:
        if not self.cfg.postprocess.enabled:
            log.info("LLM postprocessor disabled")
            return
        try:
            profile = load_profile(self.cfg.postprocess.profile)
            self.postprocessor = LLMPostprocessor(
                profile,
                dynamic_context_script=self.cfg.postprocess.dynamic_context_script,
            )
            log.info(
                "warming up LLM postprocessor (profile=%s)…",
                self.cfg.postprocess.profile,
            )
            self.postprocessor.warmup()
            log.info("LLM postprocessor ready")
        except Exception:
            log.exception("LLM postprocessor failed to load; postprocessing disabled")
            self.postprocessor = None

    def setup_sound(self) -> None:
        if not self.cfg.sound.enabled:
            log.info("sound effects disabled")
            return
        self.sound_player = SoundPlayer(volume=self.cfg.sound.volume)
        log.info("sound player ready (volume=%.2f)", self.cfg.sound.volume)

    def setup_audio(self) -> None:
        assert self.transcriber is not None
        assert self.vad_path is not None

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

        prev_state: list[State] = [State.IDLE]  # mutable cell for closure

        def on_state(state: State) -> None:
            log.debug("engine state callback: %s", state.value)
            prev = prev_state[0]
            prev_state[0] = state
            if self.overlay is not None:
                self.overlay.push_state(state)
            if self.sound_player is not None:
                if state is State.VALIDATING and prev is State.IDLE:
                    # VAD heard something — overlay appears; soft chime so
                    # it doesn't startle while the result is still uncertain.
                    self.sound_player.play_start(self.cfg.sound.validating_volume_scale)
                elif state is State.MANUAL and prev is State.IDLE:
                    # Hotkey-triggered recording — full volume.
                    self.sound_player.play_start()
                elif state is State.IDLE and prev is not State.IDLE:
                    # Overlay disappears for any reason (recording done,
                    # validation failed, manual stop).
                    self.sound_player.play_stop()

        def on_level(rms: float) -> None:
            if self.overlay is not None:
                self.overlay.push_level(rms)

        # Always load the VAD model — it is always downloaded, and the
        # tray toggle needs it present to switch auto-listen on without
        # requiring a restart. Whether VAD actually auto-triggers is
        # controlled at runtime by vad_active (cfg.vad.enabled + not paused).
        self.engine = AudioEngine(
            self.cfg,
            self.vad_path,
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
        if self.cfg.paste.type_directly:
            log.info("type_directly enabled — text injected via dotool, no clipboard")
        elif self.cfg.paste.skip_clipboard_history:
            log.info("skip_clipboard_history enabled — using wl-copy --sensitive")
        self.paster = Paster(
            backend=self.cfg.paste.backend,
            combo=self.cfg.paste.paste_combo,
            settle_ms=self.cfg.paste.settle_ms,
            timeout=self.cfg.paste.subprocess_timeout,
            skip_clipboard_history=self.cfg.paste.skip_clipboard_history,
            type_directly=self.cfg.paste.type_directly,
            restore_clipboard=self.cfg.paste.restore_clipboard,
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
                if self.sound_player is not None:
                    if self.engine.vad_paused:
                        self.sound_player.play_mute()
                    else:
                        self.sound_player.play_unmute()
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

    def _kick_off_update_check(self) -> None:
        """Best-effort GitHub version check (24h cached). When a newer
        version exists, post a desktop notification AND show the small
        yellow "update available" badge in the overlay."""
        if self.gtk_app is None:
            return
        from justsayit.update_check import UpdateInfo, check_async, detect_install_dir

        install_dir = detect_install_dir()

        def _on_result(info: UpdateInfo | None) -> None:
            if info is None:
                return

            def _apply() -> bool:
                log.info("update available: v%s -> v%s", info.current, info.latest)
                if self.overlay is not None:
                    self.overlay.push_update_available(info.latest)
                self._notify_update_available(info, install_dir)
                return False

            GLib.idle_add(_apply)

        check_async(__version__, _on_result)

    def _notify_update_available(self, info, install_dir: Path | None) -> None:
        if self.gtk_app is None:
            return
        title = f"justsayit update available: v{info.latest}"
        if install_dir is not None:
            body = (
                f"You're running v{info.current}. v{info.latest} is on GitHub.\n"
                f"Update with:\n  cd {install_dir} && ./install.sh --update"
            )
        else:
            body = (
                f"You're running v{info.current}. v{info.latest} is on GitHub.\n"
                f"See {info.url}"
            )
        try:
            note = Gio.Notification.new(title)
            note.set_body(body)
            note.set_priority(Gio.NotificationPriority.LOW)
            # Stable id so re-checks within the day update one notification
            # rather than spawning a new one each launch.
            self.gtk_app.send_notification("justsayit-update-available", note)
        except Exception:
            log.exception("could not send update-available notification")

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
                    "persisted auto-listen=%s to state.toml",
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

        # --- LLM profile submenu -------------------------------------------

        def _llm_label() -> str:
            if self.cfg.postprocess.enabled:
                return f"LLM: {self.cfg.postprocess.profile}"
            return "LLM: off"

        # Discover profiles whose model file is actually on disk.
        pd = profiles_dir()
        profile_names: list[str] = []
        if pd.exists():
            for p in sorted(pd.glob("*.toml")):
                try:
                    prof = load_profile(str(p))
                    if Path(prof.model_path).expanduser().exists():
                        profile_names.append(p.stem)
                except Exception:
                    log.debug("skipping profile %s in tray setup", p.name)

        # Build radio items for each profile + an "Off" item.
        llm_profile_items: dict[str, MenuItem] = {}
        llm_submenu_item: MenuItem | None = None

        if profile_names:
            active = (
                self.cfg.postprocess.profile if self.cfg.postprocess.enabled else None
            )

            def _on_llm_profile(key: str | None) -> None:
                if key is None:
                    self.cfg.postprocess.enabled = False
                else:
                    self.cfg.postprocess.enabled = True
                    self.cfg.postprocess.profile = key
                try:
                    save_config(self.cfg)
                except Exception:
                    log.exception("failed to save config after LLM profile switch")
                # Swap the postprocessor (lazy — no warmup; loads on first use).
                self.postprocessor = None
                if self.cfg.postprocess.enabled:
                    try:
                        prof = load_profile(self.cfg.postprocess.profile)
                        self.postprocessor = LLMPostprocessor(prof)
                        log.info(
                            "LLM profile switched to: %s", self.cfg.postprocess.profile
                        )
                    except Exception:
                        log.exception(
                            "failed to load LLM profile %s",
                            self.cfg.postprocess.profile,
                        )
                # Update radio states and parent label via ItemsPropertiesUpdated
                # (not LayoutUpdated) so the client replaces its cached values
                # directly — LayoutUpdated goes through a merge path that leaves
                # the previously-selected radio still checked.
                cur = (
                    self.cfg.postprocess.profile
                    if self.cfg.postprocess.enabled
                    else None
                )
                prop_updates: list[tuple[int, dict]] = []
                for k, item in llm_profile_items.items():
                    item.toggle_state = 1 if k == cur else 0
                    prop_updates.append(
                        (
                            item.id,
                            {"toggle-state": GLib.Variant("i", item.toggle_state)},
                        )
                    )
                llm_off_item.toggle_state = 1 if cur is None else 0
                prop_updates.append(
                    (
                        llm_off_item.id,
                        {"toggle-state": GLib.Variant("i", llm_off_item.toggle_state)},
                    )
                )
                llm_submenu_item.label = _llm_label()  # type: ignore[union-attr]
                prop_updates.append(
                    (
                        llm_submenu_item.id,
                        {"label": GLib.Variant("s", llm_submenu_item.label)},
                    )  # type: ignore[union-attr]
                )
                if self.tray is not None:
                    self.tray.notify_properties_updated(prop_updates)

            children: list[MenuItem] = []
            for i, name in enumerate(profile_names):
                item = MenuItem(
                    id=MID_LLM_PROFILE_BASE + i,
                    label=name,
                    toggle_type="radio",
                    toggle_state=1 if name == active else 0,
                    on_activate=lambda n=name: _on_llm_profile(n),
                )
                llm_profile_items[name] = item
                children.append(item)

            llm_off_item = MenuItem(
                id=MID_LLM_OFF,
                label="Off",
                toggle_type="radio",
                toggle_state=1 if active is None else 0,
                on_activate=lambda: _on_llm_profile(None),
            )
            children += [
                MenuItem(id=MID_LLM_SEP_INNER, is_separator=True),
                llm_off_item,
            ]

            llm_submenu_item = MenuItem(
                id=MID_LLM_SUBMENU,
                label=_llm_label(),
                children=children,
            )

        # --- assemble main menu --------------------------------------------

        assert self.engine is not None
        items: list[MenuItem] = [
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
        ]
        if llm_submenu_item is not None:
            items.append(llm_submenu_item)
        items += [
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
        self.tray.set_icon(ICON_ACTIVE if self.engine.vad_active else ICON_PAUSED)
        self.tray.set_tooltip(self._tray_tooltip())

    # --- runtime -----------------------------------------------------------

    def _handle_segment(self, seg: Segment) -> None:
        assert self.transcriber is not None
        duration = len(seg.samples) / seg.sample_rate
        min_duration = self.cfg.audio.skip_segments_below_seconds
        if min_duration > 0 and duration < min_duration:
            log.info(
                "skipping short segment: %.2fs < %.2fs (reason=%s)",
                duration,
                min_duration,
                seg.reason,
            )
            if self.overlay is not None:
                self.overlay.push_hide()
            return
        log.info("transcribing %.2fs (reason=%s)", duration, seg.reason)
        t0 = time.monotonic()
        raw = self.transcriber.transcribe(seg.samples, seg.sample_rate)
        dt = time.monotonic() - t0
        log.info("transcription done in %.2fs: raw=%r", dt, raw)
        if not raw:
            log.info("empty transcription; nothing to paste")
            if self.overlay is not None:
                self.overlay.push_hide()
            return
        final = apply_filters(raw, self.filters)
        if final != raw:
            log.info("filters changed output: %r -> %r", raw, final)

        # Snapshot pp before the overlay update so we know whether to show the
        # LLM field immediately (as "Wait for LLM processing…").
        pp = self.postprocessor  # snapshot — avoids TOCTOU with tray thread

        # Show the filtered text in the top field.  The bottom (LLM) field is
        # shown as a waiting placeholder if the postprocessor is active.
        if self.overlay is not None:
            self.overlay.push_detected_text(final, llm_pending=(pp is not None))

        if pp is not None:
            try:
                cleaned = pp.process(final)
                if cleaned != final:
                    log.info("LLM cleaned: %r -> %r", final, cleaned)
                    final = cleaned
            except Exception:
                log.exception("LLM postprocessor failed; using unprocessed text")
            stripped = pp.strip_for_paste(final)
            # Surface the reasoning preamble (whatever paste_strip_regex
            # matched) above the body so the user sees the full LLM reply
            # but only the stripped body lands in the focused window.
            thought = ""
            if stripped != final:
                matches = [m.strip() for m in pp.find_strip_matches(final) if m.strip()]
                thought = "\n".join(matches)
                log.info(
                    "paste_strip_regex applied: %d -> %d chars",
                    len(final),
                    len(stripped),
                )
            # Always update the LLM field — clears "Wait…" even when text is unchanged.
            if self.overlay is not None:
                self.overlay.push_llm_text(stripped, thought=thought)
            final = stripped

        # Space prefix / suffix (applied to paste content only; not shown in overlay)
        auto_space_ms = self.cfg.paste.auto_space_timeout_ms
        trailing_space = self.cfg.paste.append_trailing_space
        now = time.monotonic()

        if auto_space_ms > 0 and not trailing_space:
            if self._last_transcription_time is not None:
                seg_duration = len(seg.samples) / seg.sample_rate
                recording_started_at = now - seg_duration
                elapsed_ms = (
                    recording_started_at - self._last_transcription_time
                ) * 1000.0
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
            if self.overlay is not None:
                self.overlay.push_linger_start()
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
            if self.overlay is not None:
                self.overlay.push_linger_start()
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
        finally:
            # Linger so the user can read the transcribed text regardless of
            # whether paste succeeded or failed.
            if self.overlay is not None:
                self.overlay.push_linger_start()

    # --- GTK lifecycle -----------------------------------------------------

    def on_activate(self, app: Gtk.Application) -> None:
        # Keep the app alive even when the overlay is hidden (which is the
        # normal state now: overlay only shows while recording).
        app.hold()
        self.gtk_app = app
        if not self.no_overlay:

            def _on_overlay_abort() -> None:
                if self.engine is not None:
                    self.engine.abort()

            self.overlay = OverlayWindow(app, self.cfg, on_abort=_on_overlay_abort)
            # Explicitly hidden until the engine reports a non-idle state.
            self.overlay.set_visible(False)

        # Defer the heavy setup out of the activate handler so the UI loop
        # can keep ticking while we load models.
        def _later():
            try:
                self.setup_models()
                self.setup_filters()
                self.setup_transcriber()
                self.setup_postprocessor()
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
                self._kick_off_update_check()
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


def _write_default_config(force: bool = False, backend: str | None = None) -> None:
    ensure_dirs()
    cfg_path = config_dir() / "config.toml"
    filters_path = config_dir() / "filters.json"

    cfg_pre_existed = cfg_path.exists()
    if cfg_pre_existed and force:
        cfg_path.unlink()
        cfg_pre_existed = False
    ensure_config_file(cfg_path)
    if cfg_pre_existed:
        print(f"config already exists: {cfg_path}", file=sys.stderr)
    else:
        if backend is not None and backend != Config().model.backend:
            # Append an uncommented [model] override so the user's
            # explicit --backend choice survives the commented-defaults
            # form. TOML allows reopening a section.
            with cfg_path.open("a", encoding="utf-8") as f:
                f.write(f'\n[model]\nbackend = "{backend}"\n')
        print(f"wrote {cfg_path}")

    cleanup_path, fun_path, openai_path = ensure_default_profiles()
    dynamic_context_path = ensure_dynamic_context_script()
    print(f"postprocess profile: {cleanup_path}  (recommended)")
    print(f"postprocess profile: {fun_path}      (emoji-heavy variant)")
    print(f"postprocess profile: {openai_path}   (OpenAI-compatible endpoint)")
    print(f"dynamic-context script: {dynamic_context_path}")

    filters_pre_existed = filters_path.exists()
    if filters_pre_existed and force:
        filters_path.unlink()
        filters_pre_existed = False
    ensure_filters_file(filters_path)
    if filters_pre_existed:
        print(f"filters already exist: {filters_path}", file=sys.stderr)
    else:
        print(f"wrote {filters_path}")


def _download_models_only() -> int:
    ensure_dirs()
    cfg = load_config()
    if cfg.model.backend == "openai":
        vad = ensure_vad(cfg, force=False)
        print(f"openai backend — VAD model ready: {vad}")
        print(
            f"  (transcription served by {cfg.model.openai_endpoint or '<unset endpoint>'})"
        )
    elif cfg.model.backend == "whisper":
        vad = ensure_vad(cfg, force=False)
        print(f"whisper backend — VAD model ready: {vad}")
        print("  (Whisper model downloads automatically on first transcription)")
    else:
        p = ensure_models(cfg, force=False)
        print(f"models ready:\n  encoder: {p.encoder}\n  vad:     {p.vad}")

    if cfg.postprocess.enabled:
        try:
            profile = load_profile(cfg.postprocess.profile)
            if profile.endpoint:
                print(f"LLM endpoint: {profile.endpoint} (model={profile.model!r})")
            elif profile.hf_repo and profile.hf_filename:
                from justsayit.postprocess import LLMPostprocessor

                pp = LLMPostprocessor(profile)
                model_path = pp._resolved_model_path()
                print(f"LLM model ready: {model_path}")
        except Exception as exc:
            print(f"postprocess download skipped: {exc}", file=sys.stderr)
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


def _ensure_llama_cpp(vulkan: bool = True) -> bool:
    """Ensure llama-cpp-python is importable, compiling it if needed.

    Returns True when llama_cpp is ready, False on unrecoverable error.
    Prints human-readable progress and error messages.
    """
    import shutil
    import subprocess

    # Quick check: already importable in the current process?
    # This covers Nix builds where the package is on sys.path via the wrapper
    # script (not as an env var) and therefore invisible to a fresh subprocess.
    try:
        import llama_cpp  # noqa: F401

        return True
    except ImportError:
        pass

    # Fallback subprocess check for uv / pip managed installs where the
    # package may be installed into a venv that sys.executable can reach.
    if (
        subprocess.run(
            [sys.executable, "-c", "import llama_cpp"],
            capture_output=True,
        ).returncode
        == 0
    ):
        return True

    print("\nllama-cpp-python is not installed — installing now.")

    if vulkan:
        if shutil.which("cmake") is None:
            print(
                "error: cmake is required to compile llama-cpp-python.\n"
                "  Install: pacman -S cmake  (or the equivalent for your distro)",
                file=sys.stderr,
            )
            return False
        vulkan_ok = (
            subprocess.run(
                ["pkg-config", "--exists", "vulkan"], capture_output=True
            ).returncode
            == 0
        )
        if not vulkan_ok:
            print(
                "warning: Vulkan headers not found — falling back to CPU-only build.\n"
                "  For GPU acceleration later: pacman -S vulkan-headers vulkan-icd-loader\n",
                file=sys.stderr,
            )
            vulkan = False

    if vulkan:
        print(
            "Compiling with Vulkan GPU support"
            " (CMAKE_ARGS=-DGGML_VULKAN=1) — this may take several minutes…\n"
        )
        env = {**_os.environ, "CMAKE_ARGS": "-DGGML_VULKAN=1"}
    else:
        print("Installing llama-cpp-python (CPU-only build)…\n")
        env = {k: v for k, v in _os.environ.items() if k != "CMAKE_ARGS"}

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3"],
        env=env,
    )
    if result.returncode != 0:
        print("error: llama-cpp-python installation failed.", file=sys.stderr)
        return False
    print("\nllama-cpp-python installed successfully.")
    return True


def _parse_selection(raw: str, max_n: int) -> list[int] | None:
    """Parse a selection string into sorted 1-based indices.

    Accepts a single number ("2"), comma-separated ("1,3"), ranges ("1-3"),
    mixed ("1,3-5"), or the keyword "all".  Returns None if invalid.
    """
    s = raw.strip().lower()
    if s in ("all", "a"):
        return list(range(1, max_n + 1))
    indices: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            sides = part.split("-", 1)
            try:
                lo, hi = int(sides[0].strip()), int(sides[1].strip())
            except ValueError:
                return None
            if not (1 <= lo <= hi <= max_n):
                return None
            indices.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError:
                return None
            if not (1 <= n <= max_n):
                return None
            indices.add(n)
    return sorted(indices) if indices else None


def _run_setup_llm(model_key: str | None = None, cpu: bool = False) -> int:
    """Interactively select, download, and configure one or more GGUF LLM models."""
    ensure_dirs()

    if not _ensure_llama_cpp(vulkan=not cpu):
        return 1

    keys = list(KNOWN_LLM_MODELS.keys())

    if model_key is not None:
        selected_keys = [model_key]
    else:
        print("\nAvailable LLM models for transcription cleanup:\n")
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {KNOWN_LLM_MODELS[key]['display']}")
        print()
        hint = f"1-{len(keys)}" if len(keys) > 2 else "1,2"
        while True:
            try:
                raw = input(
                    f"Select model(s) [number, range, or comma-separated, e.g. 1 or {hint}]"
                    " (Ctrl-C to skip): "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSkipped.")
                return 0
            indices = _parse_selection(raw, len(keys))
            if indices is not None:
                selected_keys = [keys[i - 1] for i in indices]
                break
            print(
                f"  Enter a number (1-{len(keys)}), a range (1-{len(keys)}),"
                " or comma-separated values."
            )

    downloaded_models: list[tuple[str, Path]] = []  # (key, model_path)
    activate_options: list[str] = []  # profile names to suggest in the hint
    failed: list[str] = []

    for key in selected_keys:
        info = KNOWN_LLM_MODELS[key]
        hf_repo = info["hf_repo"]
        print(f"\n{info['display']}")
        print("  Querying HuggingFace for Q4_K_M filename…", end="", flush=True)
        try:
            hf_filename = find_hf_q4_filename(hf_repo)
        except RuntimeError as exc:
            print(f"\n  error: {exc}", file=sys.stderr)
            failed.append(key)
            continue
        print(f" {hf_filename}")

        try:
            model_path = download_llm_model(hf_repo, hf_filename)
        except Exception as exc:
            print(f"  error: download failed: {exc}", file=sys.stderr)
            failed.append(key)
            continue

        downloaded_models.append((key, model_path))
        print(f"  Model:   {model_path}")

        if key == "gemma4":
            # gemma4 ships two profiles (-cleanup and -fun) that share the
            # same model file. Reuse them instead of creating a third
            # generic gemma4.toml that would only confuse users.
            cleanup_path, fun_path, _openai_path = ensure_default_profiles()
            update_profile_model(cleanup_path, model_path, hf_repo, hf_filename)
            update_profile_model(fun_path, model_path, hf_repo, hf_filename)
            activate_options.extend(["gemma4-cleanup", "gemma4-fun"])
            print(f"  Profile: {cleanup_path}")
            print(f"  Profile: {fun_path}")
        else:
            profile_path = profiles_dir() / f"{key}.toml"
            ensure_default_profile(profile_path)
            update_profile_model(profile_path, model_path, hf_repo, hf_filename)
            activate_options.append(key)
            print(f"  Profile: {profile_path}")

    if not downloaded_models:
        return 1 if failed else 0

    print(f"\n{'─' * 54}")
    if len(downloaded_models) == 1:
        print("\n1 model ready. Available profile(s):")
    else:
        print(f"\n{len(downloaded_models)} model(s) ready. Available profile(s):")
    for name in activate_options:
        print(f"  - {name}")
    print(
        "\nPick one from the tray menu (LLM submenu) — that's a runtime"
        " toggle and writes to ~/.config/justsayit/state.toml for you."
    )
    return 0


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
    ap.add_argument("--config", type=Path, help="override path to config.toml")
    sub = ap.add_subparsers(dest="subcommand")
    init_sub = sub.add_parser("init", help="write default config + example filters")
    init_sub.add_argument(
        "--backend",
        choices=["parakeet", "whisper"],
        default=None,
        help="transcription backend to set in config (default: parakeet)",
    )
    sub.add_parser(
        "download-models",
        help="pre-download models (Parakeet + VAD, or VAD only for whisper backend)",
    )
    setup_llm_sub = sub.add_parser(
        "setup-llm",
        help="interactively select and download a GGUF LLM for postprocessing",
    )
    setup_llm_sub.add_argument(
        "--model",
        choices=list(KNOWN_LLM_MODELS.keys()),
        default=None,
        help="model key to download without interactive prompt",
    )
    setup_llm_sub.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="install CPU-only llama-cpp-python (skip Vulkan GPU compilation)",
    )

    show_sub = sub.add_parser(
        "show-defaults",
        help="print the shipped default for a user-managed file to stdout "
        "(used by install.sh --update to diff against the user's file)",
    )
    show_sub.add_argument(
        "kind",
        choices=[
            "config",
            "filters",
            "context",
            "profile-cleanup",
            "profile-fun",
            "profile-openai",
        ],
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    console_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=console_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.subcommand == "init":
        _write_default_config(force=False, backend=getattr(args, "backend", None))
        return 0
    if args.subcommand == "download-models":
        return _download_models_only()
    if args.subcommand == "setup-llm":
        return _run_setup_llm(
            getattr(args, "model", None), cpu=getattr(args, "cpu", False)
        )
    if args.subcommand == "show-defaults":
        if args.kind == "config":
            sys.stdout.write(render_config_toml(commented=True))
        elif args.kind == "filters":
            import json

            sys.stdout.write(json.dumps(_default_filter_chain(), indent=2) + "\n")
        elif args.kind == "context":
            sys.stdout.write(_CONTEXT_SIDECAR_TEMPLATE)
        elif args.kind == "profile-cleanup":
            sys.stdout.write(_CLEANUP_PROFILE_TOML)
        elif args.kind == "profile-fun":
            sys.stdout.write(_FUN_PROFILE_TOML)
        elif args.kind == "profile-openai":
            sys.stdout.write(_OPENAI_PROFILE_TOML)
        return 0

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
        # Prefer relaunching via the installed .desktop file so the new
        # instance lands in a desktop-env-managed systemd scope. In-place
        # execve keeps the same scope/D-Bus connection lineage, which on
        # KDE causes the portal to lose track of our existing shortcut
        # binding (ConfigureShortcuts isn't supported on portal v1, so the
        # bind dialog re-pops). Falls back to execve in dev mode where no
        # .desktop file is installed.
        if _relaunch_via_desktop():
            return rc
        log.info("desktop relaunch unavailable, falling back to in-place execve")
        _os.execve(_sys.executable, [_sys.executable] + _sys.argv, _os.environ)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
