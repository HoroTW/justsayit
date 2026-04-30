"""CLI entry point and glue between audio, transcription, overlay, and paste."""

from __future__ import annotations

# --- gtk4-layer-shell preload ---------------------------------------------
# Layer-shell must be loaded *before* libwayland-client is pulled in. Once
# any `gi.require_version("Gtk", "4.0")` runs, it's too late. Re-exec with
# LD_PRELOAD set before we touch gi. Boot helpers live in _boot.py so this
# module stays lean; importing _boot.py only uses os/sys, no GTK.
from justsayit._boot import (
    _app_id,
    _find_layer_shell_lib,
    _is_remote_subcommand,
    _preload_layer_shell,
    _reexec_cmd,
    _reexec_under_systemd_scope,
    _relaunch_via_desktop,
    _set_process_name,
)

if not _is_remote_subcommand():
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
    _default_filter_chain,
    config_dir,
    ensure_config_file,
    ensure_dirs,
    ensure_filters_file,
    load_config,
    render_config_toml,
    save_config,
)
from justsayit.filters import load_filters
from justsayit.tools import load_tools
from justsayit.model import ensure_models, ensure_vad
from justsayit.postprocess import (
    KNOWN_LLM_MODELS,
    LLMPostprocessor,
    _CLEANUP_PROFILE_TOML,
    _CONTEXT_SIDECAR_TEMPLATE,
    _FUN_PROFILE_TOML,
    _OPENAI_PROFILE_TOML,
    load_profile,
    profiles_dir,
)
from justsayit.postprocess.backend_local import LocalBackend
from justsayit.overlay import OverlayWindow
from justsayit.paste import PasteError, Paster, read_clipboard, read_clipboard_image
from justsayit.shortcuts import GlobalShortcutClient
from justsayit.sound import SoundPlayer
from justsayit.transcribe import TranscriberBase, make_transcriber
from justsayit.tray import MenuItem, TrayIcon, open_with_xdg
from justsayit.pipeline import SegmentPipeline

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
MID_LLM_UNLOAD = 11  # "Unload local LLM" item inside the LLM submenu
MID_LLM_SEP_UNLOAD = 12  # separator before the "Unload" item
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

        self.vad_path = None  # set in setup_models (all backends)
        self.transcriber: TranscriberBase | None = None
        self.engine: AudioEngine | None = None
        self.overlay: OverlayWindow | None = None
        self.shortcut_client: GlobalShortcutClient | None = None
        self.paster: Paster | None = None
        self.sound_player: SoundPlayer | None = None
        self.postprocessor: LLMPostprocessor | None = None
        self.pipeline: SegmentPipeline | None = None
        self.tray: TrayIcon | None = None
        self.gtk_app: Gtk.Application | None = None
        self.filters = []
        self.after_llm_filters = []

        # Bounded queue so we can't run unbounded transcription work if the
        # user is trigger-happy with the hotkey.
        self._seg_q: queue.Queue[Segment | None] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._transcribe_thread: threading.Thread | None = None
        self._restart_requested: bool = False
        # One-time flag set by the overlay's 📋 button. Read+cleared in
        # _handle_segment; when True, clipboard contents are appended to
        # the LLM system prompt for that single transcription.
        self._clipboard_context_armed: bool = False
        # "Arm the NEXT recording" — set by the CLI toggle-ex action
        # (which runs on the main loop) BEFORE the audio worker thread
        # has transitioned state. Consumed by on_state at the IDLE →
        # VALIDATING / MANUAL edge: promotes to _clipboard_context_armed
        # instead of calling _disarm. Lets a single CLI invocation swap
        # profile + arm clipboard + trigger recording atomically without
        # losing the arm to the stale-defense disarm.
        self._clipboard_context_arm_next: bool = False
        # Continue-session state. _continue_window_active means the timer
        # is running; _continue_this_recording is snapshotted per segment
        # at the IDLE→MANUAL/VALIDATING edge so the transcribe thread can
        # read it without racing the timer callback.
        self._continue_window_active: bool = False
        self._continue_this_recording: bool = False
        self._continue_timer_id: int | None = None
        # Assistant mode: overlay stays open after results; every recording
        # continues the previous LLM session (is_continue always True).
        self._assistant_mode: bool = False
        # Tray LLM radio items — stored here (not captured in a closure)
        # so _apply_llm_profile can refresh their checked state from
        # outside the tray setup (e.g. from the toggle-ex D-Bus action).
        self._llm_profile_items: dict[str, MenuItem] = {}
        self._llm_off_item: MenuItem | None = None
        self._llm_submenu_item: MenuItem | None = None
        self._unload_llm_item: MenuItem | None = None
        self._unload_llm_sep: MenuItem | None = None
        # Local LLM stash: holds the last displaced LocalBackend so toggling
        # back to the same profile is instant.
        self._stashed_local: LLMPostprocessor | None = None
        self._stashed_local_profile: str | None = None
        # Profile name of the currently active self.postprocessor.
        self._active_profile_name: str | None = None
        # Last non-local profile; fallback target for _unload_local_llm.
        self._last_non_local_profile: str | None = None
        # Suppresses stash-on-displacement; set temporarily by _unload_local_llm.
        self._suppress_stash: bool = False

    # --- setup -------------------------------------------------------------

    def setup_models(self) -> None:
        # Always fetch the VAD model so the tray's auto-listen toggle
        # works regardless of whether it was enabled at startup.
        if self.cfg.model.backend == "openai":
            # Remote STT — no local ASR model to download. VAD is still
            # local (we don't want to stream audio over the network just
            # to detect silence) so fetch the tiny silero ONNX.
            self.vad_path = ensure_vad(self.cfg)
            log.info("openai backend: VAD ready, transcription via remote endpoint")
        elif self.cfg.model.backend == "whisper":
            self.vad_path = ensure_vad(self.cfg)
            log.info("whisper backend: VAD ready, Whisper model loads on first use")
        else:
            self.vad_path = ensure_models(self.cfg, want_vad=True).vad

    def setup_filters(self) -> None:
        self.filters = load_filters(self.cfg.filters_path)
        log.info(
            "loaded %d filter(s) from %s",
            len(self.filters),
            self.cfg.filters_path,
        )
        self.after_llm_filters = load_filters(self.cfg.after_llm_filters_path)
        if self.after_llm_filters:
            log.info(
                "loaded %d after-LLM filter(s) from %s",
                len(self.after_llm_filters),
                self.cfg.after_llm_filters_path,
            )

    def setup_tools(self) -> None:
        tool_defs = load_tools(self.cfg.tools_path)
        if self.pipeline is not None:
            self.pipeline.tool_definitions = tool_defs

    def setup_transcriber(self) -> None:
        self.transcriber = make_transcriber(self.cfg)
        log.info("warming up %s recognizer…", self.cfg.model.backend)
        self.transcriber.warmup()

        def _on_pipeline_error(stage: str, msg: str, retry_cb) -> None:
            if self.overlay is not None:
                self.overlay.push_error(stage, msg, retry_cb)

        def _enqueue(seg: Segment) -> None:
            try:
                self._seg_q.put_nowait(seg)
            except queue.Full:
                log.warning("retry: transcription queue full; dropping segment")

        self.pipeline = SegmentPipeline(
            self.cfg,
            self.transcriber,
            self.filters,
            self.paster,
            no_paste=self.no_paste,
            after_llm_filters=self.after_llm_filters,
            on_error=_on_pipeline_error,
            enqueue_segment=_enqueue,
        )

    def setup_postprocessor(self) -> None:
        if not self.cfg.postprocess.enabled:
            # Critical: clear the field, otherwise toggling LLM off via the
            # tray leaves the previous instance attached and _handle_segment
            # keeps routing through it ("LLM: off" but text still cleaned).
            if isinstance(self.postprocessor, LocalBackend) and not self._suppress_stash:
                self._stash_local(self.postprocessor, self._active_profile_name)
            self.postprocessor = None
            self._active_profile_name = None
            log.info("LLM postprocessor disabled")
            if self.pipeline is not None:
                self.pipeline.postprocessor = None
            return

        profile_name = self.cfg.postprocess.profile

        # Reuse stash when switching back to the same local profile.
        if self._stashed_local_profile == profile_name and self._stashed_local is not None:
            log.info("reusing stashed local LLM (profile=%s)", profile_name)
            if isinstance(self.postprocessor, LocalBackend) and not self._suppress_stash:
                self._stash_local(self.postprocessor, self._active_profile_name)
            self.postprocessor = self._stashed_local
            self._active_profile_name = profile_name
            self._stashed_local = None
            self._stashed_local_profile = None
            if self.pipeline is not None:
                self.pipeline.postprocessor = self.postprocessor
            return

        # Stash or drop the outgoing local postprocessor before replacing it.
        if isinstance(self.postprocessor, LocalBackend) and not self._suppress_stash:
            self._stash_local(self.postprocessor, self._active_profile_name)

        try:
            profile = load_profile(profile_name)
            self.postprocessor = LLMPostprocessor(
                profile,
                dynamic_context_script=self.cfg.postprocess.dynamic_context_script,
            )
            self._active_profile_name = profile_name
        except Exception:
            log.exception("LLM postprocessor failed to load; postprocessing disabled")
            self.postprocessor = None
            self._active_profile_name = None
            if self.pipeline is not None:
                self.pipeline.postprocessor = None
            return
        if self.pipeline is not None:
            self.pipeline.postprocessor = self.postprocessor
        log.info("warming up LLM postprocessor (profile=%s) in background…", profile_name)
        pp = self.postprocessor

        def _warmup() -> None:
            try:
                pp.warmup()
                log.info("LLM postprocessor ready (profile=%s)", profile_name)
            except Exception:
                log.exception("LLM postprocessor warmup failed; postprocessing disabled")

                def _clear() -> None:
                    if self.postprocessor is pp:
                        self.postprocessor = None
                        self._active_profile_name = None
                        if self.pipeline is not None:
                            self.pipeline.postprocessor = None

                GLib.idle_add(_clear)

        threading.Thread(target=_warmup, daemon=True).start()

    def _stash_local(self, pp: LLMPostprocessor, name: str | None) -> None:
        if name is None:
            return
        if self._stashed_local is not None and self._stashed_local_profile != name:
            log.info("evicting stashed local LLM (profile=%s)", self._stashed_local_profile)
        self._stashed_local = pp
        self._stashed_local_profile = name
        log.info("stashed local LLM (profile=%s)", name)

    def _push_llm_profile(self) -> None:
        if self.overlay is None:
            return
        pp = self.postprocessor
        if pp is None:
            self.overlay.push_llm_profile(None, None)
        else:
            base = pp.profile.base
            backend = "local" if base == "builtin" else base
            self.overlay.push_llm_profile(backend, self._active_profile_name or "?")

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
            if prev is State.IDLE and state in (State.VALIDATING, State.MANUAL):
                self._continue_this_recording = self._continue_window_active or self._assistant_mode
                if self._clipboard_context_arm_next:
                    # CLI asked to arm *this* recording. Promote the
                    # pending flag to _armed here, where we're
                    # guaranteed to be past the stale-defense disarm.
                    self._clipboard_context_arm_next = False
                    self._clipboard_context_armed = True
                    log.info("clipboard-context flag → armed (CLI, next recording)")
                    if self.overlay is not None:
                        self.overlay.push_clipboard_context_armed(True)
                else:
                    # Fresh recording — clipboard-context arming is strictly
                    # per-recording, so clear any leftover flag before it can
                    # feed a stale clipboard into this session's LLM call.
                    self._disarm_clipboard_context()
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
            restore_delay_ms=self.cfg.paste.restore_delay_ms,
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
        if self.pipeline is not None:
            self.pipeline.paster = self.paster

    def toggle(self) -> None:
        """Toggle recording. Called by the global shortcut hotkey and the DBus action."""
        if self.engine is None:
            log.warning("toggle fired before audio engine ready")
            return
        state = self.engine.state
        log.info("TOGGLE fired (state=%s)", state.value)
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

    def setup_shortcut(self) -> None:
        def on_activated(shortcut_id: str) -> None:
            self.toggle()

        def on_needs_rebinding(shortcut_id: str, reason: str) -> None:
            self._notify_shortcut_unbound(shortcut_id, reason)

        self.shortcut_client = GlobalShortcutClient(
            shortcut_id=self.cfg.shortcut.id,
            description=self.cfg.shortcut.description,
            preferred_trigger=self.cfg.shortcut.preferred,
            on_activated=on_activated,
            on_needs_rebinding=on_needs_rebinding,
        )

    def _send_notification(
        self,
        notification_id: str,
        title: str,
        body: str,
        priority: Gio.NotificationPriority = Gio.NotificationPriority.NORMAL,
    ) -> bool:
        """Post a desktop notification. Returns True on success."""
        if self.gtk_app is None:
            return False
        try:
            note = Gio.Notification.new(title)
            note.set_body(body)
            note.set_priority(priority)
            self.gtk_app.send_notification(notification_id, note)
            return True
        except Exception:
            log.exception("could not send notification %r", notification_id)
            return False

    def _notify_shortcut_unbound(self, shortcut_id: str, reason: str) -> None:
        body = (
            f'The "{shortcut_id}" shortcut has no trigger. Open your '
            "desktop environment's shortcut settings to assign one "
            "(KDE: System Settings → Shortcuts; "
            "GNOME: Settings → Keyboard → View and Customize Shortcuts)."
        )
        if self._send_notification(
            f"justsayit-shortcut-unbound-{shortcut_id}",
            "justsayit: shortcut not assigned",
            body,
            Gio.NotificationPriority.HIGH,
        ):
            log.info("sent 'shortcut unbound' notification (reason=%s)", reason)

    def _kick_off_update_check(self) -> None:
        """Best-effort GitHub version check (3h cached). When a newer
        version exists, post a desktop notification AND show the small
        yellow "update available" badge in the overlay."""
        if self.gtk_app is None:
            return
        from justsayit.update_check import UpdateInfo, check_async, detect_install_dir

        log.info("checking for updates on GitHub...")
        install_dir = detect_install_dir()

        def _on_result(info: UpdateInfo | None, checked: bool) -> None:
            if not checked:
                return
            if info is None:
                log.info("no update available on GitHub")
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
        self._send_notification(
            "justsayit-update-available", title, body, Gio.NotificationPriority.LOW
        )

    def _notify_space_config_conflict(self) -> None:
        body = (
            "Both paste.auto_space_timeout_ms and paste.append_trailing_space "
            "are enabled. These settings conflict: the trailing space already "
            "acts as a separator, so the auto-prefix space has been disabled. "
            "Set auto_space_timeout_ms = 0 to suppress this warning."
        )
        self._send_notification(
            "justsayit-space-conflict",
            "justsayit: conflicting space settings",
            body,
        )
        log.warning("space config conflict: auto_space_timeout_ms ignored")

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

        # List every profile whose TOML parses. We deliberately do NOT
        # gate on backend-specific readiness (builtin GGUF on disk,
        # remote endpoint set) — silent filtering hid remote profiles
        # for users without a local model installed. If a profile is
        # broken, ``setup_postprocessor`` catches the failure when the
        # user actually selects it: warmup raises, the exception is
        # logged, and postprocessing falls back to disabled.
        pd = profiles_dir()
        profile_names: list[str] = []
        if pd.exists():
            for p in sorted(pd.glob("*.toml")):
                try:
                    load_profile(str(p))
                except Exception:
                    log.debug("skipping unparseable profile %s in tray setup", p.name)
                    continue
                profile_names.append(p.stem)

        # Build radio items for each profile + an "Off" item.
        llm_submenu_item: MenuItem | None = None

        if profile_names:
            active = (
                self.cfg.postprocess.profile if self.cfg.postprocess.enabled else None
            )
            children: list[MenuItem] = []
            for i, name in enumerate(profile_names):
                item = MenuItem(
                    id=MID_LLM_PROFILE_BASE + i,
                    label=name,
                    toggle_type="radio",
                    toggle_state=1 if name == active else 0,
                    on_activate=lambda n=name: self._apply_llm_profile(n),
                )
                self._llm_profile_items[name] = item
                children.append(item)

            self._llm_off_item = MenuItem(
                id=MID_LLM_OFF,
                label="Off",
                toggle_type="radio",
                toggle_state=1 if active is None else 0,
                on_activate=lambda: self._apply_llm_profile(None),
            )
            children += [
                MenuItem(id=MID_LLM_SEP_INNER, is_separator=True),
                self._llm_off_item,
            ]

            llm_submenu_item = MenuItem(
                id=MID_LLM_SUBMENU,
                label=self._llm_tray_label(),
                children=children,
            )
            self._llm_submenu_item = llm_submenu_item

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
        has_local = self._has_local_llm_loaded()
        unload_sep = MenuItem(id=MID_LLM_SEP_UNLOAD, is_separator=True, visible=has_local)
        unload_item = MenuItem(
            id=MID_LLM_UNLOAD,
            label="Unload local LLM",
            visible=has_local,
            on_activate=self._unload_local_llm,
        )
        self._unload_llm_sep = unload_sep
        self._unload_llm_item = unload_item
        items += [
            unload_sep,
            unload_item,
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

    def _toggle_clipboard_context(self) -> None:
        """Flip the one-time clipboard-context flag from the overlay button.

        We don't read the clipboard here — armed → next ``_handle_segment``
        call reads it via ``read_clipboard()`` and feeds it to the LLM
        before clearing the flag. Disarming before recording cancels the
        pending read.
        """
        self._clipboard_context_armed = not self._clipboard_context_armed
        log.info(
            "clipboard-context flag → %s",
            "armed" if self._clipboard_context_armed else "disarmed",
        )
        if self.overlay is not None:
            self.overlay.push_clipboard_context_armed(
                self._clipboard_context_armed
            )

    def _arm_clipboard_next_recording(self) -> None:
        """Mark the NEXT recording as clipboard-armed. Used by the
        toggle-ex D-Bus action: we can't set ``_clipboard_context_armed``
        directly because the audio worker transitions IDLE → MANUAL/VALIDATING
        asynchronously and clears it via ``_disarm_clipboard_context``.
        Instead, set a pending flag that ``on_state`` consumes at the
        transition edge."""
        self._clipboard_context_arm_next = True
        log.info("clipboard-context → arm_next (CLI)")

    def _activate_continue_window(self) -> None:
        self._continue_window_active = True
        self._continue_this_recording = True
        self._reset_continue_timer()
        log.info("continue window → armed (%d min)", self.cfg.postprocess.continue_window_minutes or 5)
        if self.overlay is not None:
            self.overlay.push_continue_armed(True)

    def _deactivate_continue_window(self) -> None:
        self._continue_window_active = False
        self._continue_this_recording = False
        if self._continue_timer_id is not None:
            GLib.source_remove(self._continue_timer_id)
            self._continue_timer_id = None
        log.info("continue window → disarmed")
        if self.overlay is not None:
            self.overlay.push_continue_armed(False)

    def _toggle_continue_window(self) -> None:
        if self._continue_window_active:
            self._deactivate_continue_window()
        else:
            self._activate_continue_window()

    def _toggle_assistant_mode(self) -> None:
        self._assistant_mode = not self._assistant_mode
        log.info("assistant mode → %s", "on" if self._assistant_mode else "off")
        if self.pipeline is not None:
            self.pipeline.assistant_mode = self._assistant_mode
        if self.overlay is not None:
            self.overlay.push_assistant_mode(self._assistant_mode)

    def _reset_continue_timer(self) -> None:
        if self._continue_timer_id is not None:
            GLib.source_remove(self._continue_timer_id)
        mins = self.cfg.postprocess.continue_window_minutes or 5
        self._continue_timer_id = GLib.timeout_add_seconds(
            mins * 60, self._on_continue_timer_expired
        )

    def _on_continue_timer_expired(self) -> bool:
        self._continue_window_active = False
        self._continue_timer_id = None
        if self.overlay is not None:
            self.overlay.push_continue_armed(False)
        log.info("continue window expired")
        return False

    def _llm_tray_label(self) -> str:
        if self.cfg.postprocess.enabled:
            return f"LLM: {self.cfg.postprocess.profile}"
        return "LLM: off"

    def _profile_is_local(self, name: str) -> bool:
        try:
            return load_profile(name).base == "builtin"
        except Exception:
            return False

    def _has_local_llm_loaded(self) -> bool:
        return isinstance(self.postprocessor, LocalBackend) or self._stashed_local is not None

    def _apply_llm_profile(self, name: str | None) -> None:
        """Switch LLM profile (``name=None`` disables postprocessing).
        Persists to state.toml, rebuilds the postprocessor, and refreshes
        any tray radio items. Shared by the tray menu and the toggle-ex
        D-Bus action, so both paths stay in sync.
        """
        # Track the last non-local profile so _unload_local_llm can fall back to it.
        if self.cfg.postprocess.enabled and not self._profile_is_local(self.cfg.postprocess.profile):
            self._last_non_local_profile = self.cfg.postprocess.profile

        # No-op if already on this profile with a loaded postprocessor.
        if (
            name is not None
            and self.cfg.postprocess.enabled
            and self.cfg.postprocess.profile == name
            and self.postprocessor is not None
        ):
            self._refresh_llm_tray_state()
            return
        if name is None:
            self.cfg.postprocess.enabled = False
        else:
            self.cfg.postprocess.enabled = True
            self.cfg.postprocess.profile = name
        try:
            save_config(self.cfg)
        except Exception:
            log.exception("failed to save config after LLM profile switch")
        # Reinitialize exactly like startup so dynamic-context and
        # warmup behavior stay consistent after profile switches.
        self.setup_postprocessor()
        if self.cfg.postprocess.enabled and self.postprocessor is not None:
            log.info("LLM profile switched to: %s", self.cfg.postprocess.profile)
        self._refresh_llm_tray_state()
        self._push_llm_profile()

    def _unload_local_llm(self) -> None:
        """Drop all loaded local LLMs (active + stash) and fall back to
        the last non-local profile, or 'off' if none was used."""
        if self._stashed_local is not None:
            log.info("dropping stashed local LLM (profile=%s)", self._stashed_local_profile)
            self._stashed_local = None
            self._stashed_local_profile = None
        if isinstance(self.postprocessor, LocalBackend):
            self._suppress_stash = True
            try:
                self._apply_llm_profile(self._last_non_local_profile)
            finally:
                self._suppress_stash = False
        self._refresh_llm_tray_state()

    def _refresh_llm_tray_state(self) -> None:
        """Update the LLM submenu's radio items + parent label to match
        the current cfg. Uses ItemsPropertiesUpdated (not LayoutUpdated)
        so the client replaces its cached values directly — LayoutUpdated
        goes through a merge path that leaves the previously-selected
        radio still checked."""
        if self.tray is None or self._llm_submenu_item is None:
            return
        cur = (
            self.cfg.postprocess.profile
            if self.cfg.postprocess.enabled
            else None
        )
        prop_updates: list[tuple[int, dict]] = []
        for k, item in self._llm_profile_items.items():
            item.toggle_state = 1 if k == cur else 0
            in_memory = k == self._stashed_local_profile or (
                k == cur and isinstance(self.postprocessor, LocalBackend)
            )
            new_label = f"{k} *" if in_memory else k
            if item.label != new_label:
                item.label = new_label
            prop_updates.append(
                (item.id, {"toggle-state": GLib.Variant("i", item.toggle_state), "label": GLib.Variant("s", item.label)})
            )
        if self._llm_off_item is not None:
            self._llm_off_item.toggle_state = 1 if cur is None else 0
            prop_updates.append(
                (
                    self._llm_off_item.id,
                    {"toggle-state": GLib.Variant("i", self._llm_off_item.toggle_state)},
                )
            )
        self._llm_submenu_item.label = self._llm_tray_label()
        prop_updates.append(
            (
                self._llm_submenu_item.id,
                {"label": GLib.Variant("s", self._llm_submenu_item.label)},
            )
        )
        if self._unload_llm_item is not None and self._unload_llm_sep is not None:
            has_local = self._has_local_llm_loaded()
            if self._unload_llm_item.visible != has_local:
                self._unload_llm_item.visible = has_local
                self._unload_llm_sep.visible = has_local
                prop_updates.append(
                    (self._unload_llm_item.id, {"visible": GLib.Variant("b", has_local)})
                )
                prop_updates.append(
                    (self._unload_llm_sep.id, {"visible": GLib.Variant("b", has_local)})
                )
        self.tray.notify_properties_updated(prop_updates)

    def _disarm_clipboard_context(self) -> None:
        """Clear the armed flag without reading the clipboard. Called at
        the start of every new recording so arming is strictly
        per-recording and can never leak across sessions."""
        if not self._clipboard_context_armed:
            return
        self._clipboard_context_armed = False
        log.info("clipboard-context flag → disarmed (new recording starting)")
        if self.overlay is not None:
            self.overlay.push_clipboard_context_armed(False)

    def _consume_clipboard_context(self) -> tuple[str, bytes | None, str]:
        """If the flag is armed, read the clipboard (text or image), clear the
        flag, and update the overlay.
        Returns ``("", None, "")`` when not armed or clipboard empty."""
        if not self._clipboard_context_armed:
            return "", None, ""
        self._clipboard_context_armed = False
        if self.overlay is not None:
            self.overlay.push_clipboard_context_armed(False)
        clip = read_clipboard(text_only=True)
        if clip:
            log.info("injecting clipboard as one-time LLM context: %d chars", len(clip))
            return clip, None, ""
        img = read_clipboard_image()
        if img:
            img_bytes, img_mime = img
            log.info(
                "injecting clipboard image as one-time LLM context: %d bytes (%s)",
                len(img_bytes), img_mime,
            )
            return "", img_bytes, img_mime
        log.info("clipboard-context armed but clipboard is empty or unavailable — skipping injection")
        return "", None, ""

    def _handle_segment(self, seg: Segment) -> None:
        """Process one audio segment via the configured SegmentPipeline."""
        assert self.pipeline is not None, "_handle_segment called before setup_transcriber"
        is_continue = self._continue_this_recording
        self.pipeline.handle(seg, consume_clipboard_fn=self._consume_clipboard_context, is_continue=is_continue)
        if is_continue and self._continue_window_active:
            GLib.idle_add(self._reset_continue_timer)

    # --- GTK lifecycle -----------------------------------------------------

    def on_activate(self, app: Gtk.Application) -> None:
        # Keep the app alive even when the overlay is hidden (which is the
        # normal state now: overlay only shows while recording).
        app.hold()
        self.gtk_app = app

        toggle_action = Gio.SimpleAction.new("toggle", None)
        toggle_action.connect("activate", lambda _a, _p: self.toggle())
        app.add_action(toggle_action)

        # toggle-ex: composite action fired by `justsayit toggle [...]`
        # from a second process. Keeps profile-switch + arm-clipboard +
        # toggle atomic on the main loop so the arm survives the audio
        # worker's IDLE→MANUAL/VALIDATING state-transition disarm.
        #   a{sv} keys:
        #     "profile"       (s) — switch to this LLM profile first.
        #                           The sentinel "off" (case-insensitive)
        #                           disables postprocessing, mirroring the
        #                           tray submenu's "Off" radio item.
        #     "arm-clipboard" (b) — arm clipboard for this recording
        # Unknown keys are ignored so we can grow the surface later
        # without bumping the action name.
        toggle_ex_action = Gio.SimpleAction.new(
            "toggle-ex", GLib.VariantType.new("a{sv}")
        )

        def _on_toggle_ex(_a, param):
            try:
                opts = dict(param.unpack()) if param is not None else {}
            except Exception:
                log.exception("toggle-ex: could not unpack parameters")
                opts = {}
            profile = opts.get("profile")
            arm_clip = bool(opts.get("arm-clipboard", False))
            arm_continue = bool(opts.get("arm-continue", False))
            log.info(
                "toggle-ex received: profile=%r arm-clipboard=%s arm-continue=%s",
                profile,
                arm_clip,
                arm_continue,
            )
            if isinstance(profile, str) and profile:
                # "off" disables postprocessing — same effect as the
                # tray's "Off" radio item. Case-insensitive because
                # users type it from shortcut scripts.
                target = None if profile.lower() == "off" else profile
                self._apply_llm_profile(target)
            if arm_clip:
                self._arm_clipboard_next_recording()
            if arm_continue:
                self._activate_continue_window()
            self.toggle()

        toggle_ex_action.connect("activate", _on_toggle_ex)
        app.add_action(toggle_ex_action)

        unload_llm_action = Gio.SimpleAction.new("unload-local-llm", None)
        unload_llm_action.connect("activate", lambda _a, _p: self._unload_local_llm())
        app.add_action(unload_llm_action)
        if not self.no_overlay:

            def _on_overlay_abort() -> None:
                if self.engine is not None:
                    self.engine.abort()

            self.overlay = OverlayWindow(
                app,
                self.cfg,
                on_abort=_on_overlay_abort,
                on_toggle_clipboard_context=self._toggle_clipboard_context,
                on_toggle_continue_window=self._toggle_continue_window,
                on_toggle_assistant_mode=self._toggle_assistant_mode,
            )
            # Explicitly hidden until the engine reports a non-idle state.
            self.overlay.set_visible(False)

        # Defer the heavy setup out of the activate handler so the UI loop
        # can keep ticking while we load models.
        def _later():
            try:
                self.setup_models()
                self.setup_filters()
                self.setup_transcriber()
                if self.pipeline is not None:
                    self.pipeline.overlay = self.overlay
                self.setup_tools()
                self.setup_postprocessor()
                self._push_llm_profile()
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


from justsayit._subcommands import (
    _download_models_only,
    _ensure_llama_cpp,
    _parse_selection,
    _run_setup_llm,
    _send_toggle,
    _send_unload_llm,
    _setup_file_logging,
    _write_default_config,
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

    sub.add_parser(
        "unload-llm",
        help="unload all local LLMs from the running instance and fall back to the last remote profile or off",
    )

    toggle_sub = sub.add_parser(
        "toggle",
        help="toggle recording on the running instance (for custom shortcuts)",
        description=(
            "Send a toggle command to the already-running justsayit "
            "instance. Optional flags let one keyboard shortcut switch "
            "LLM profile and/or arm clipboard-context for the recording "
            "it starts — e.g. bind a 'privacy mode' shortcut to "
            "`justsayit toggle --profile my-local-llm`."
        ),
    )
    toggle_sub.add_argument(
        "--profile",
        default=None,
        help=(
            "switch LLM profile before toggling (persistent, like the "
            "tray menu); pass 'off' to disable postprocessing"
        ),
    )
    toggle_sub.add_argument(
        "--use-clipboard",
        action="store_true",
        default=False,
        help="arm clipboard-context for the recording this toggle starts",
    )
    toggle_sub.add_argument(
        "--continue",
        dest="continue_flag",
        action="store_true",
        default=False,
        help="start/extend continue window for LLM session continuation",
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
    # Set the kernel comm field so `killall justsayit` / `pgrep justsayit`
    # work. Has to happen here (not at module import) because the two
    # re-execs at the top of this module reset comm back to `python3`.
    _set_process_name("justsayit")

    args = _build_parser().parse_args(argv)

    console_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=console_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.subcommand == "toggle":
        return _send_toggle(
            profile=args.profile,
            use_clipboard=args.use_clipboard,
            continue_flag=getattr(args, "continue_flag", False),
        )
    if args.subcommand == "unload-llm":
        return _send_unload_llm()
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
