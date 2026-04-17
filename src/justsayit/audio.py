"""Microphone capture.

Two modes:

* **vad.enabled = False** (default) — pure hotkey mode. The mic is
  *closed* until ``start_manual()`` opens it; ``stop_manual()`` emits
  the segment and the mic is closed again. No Silero, no validation,
  no always-on stream.
* **vad.enabled = True** — Silero VAD watches for speech onset; the
  first ``validation_seconds`` get transcribed and discarded if they
  contain no words; otherwise recording continues until VAD silence.
  The mic stays open continuously while VAD is enabled.

Both modes share the same sounddevice input stream and the same
segment-emission callback. Consumers receive events via injected
callbacks so the ML-heavy work can live in another module/thread.
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from justsayit.config import Config

log = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    VALIDATING = "validating"
    RECORDING = "recording"
    MANUAL = "manual"


@dataclass
class Segment:
    samples: np.ndarray  # float32 mono @ cfg.audio.sample_rate
    sample_rate: int
    reason: str  # "vad", "manual", "max-length", "shutdown"
    # ``time.monotonic()`` timestamp of the hotkey-stop that caused this
    # segment to be emitted. ``None`` for non-manual segments. Consumers
    # use it to delay the paste until the user has plausibly released
    # the hotkey modifier keys.
    stop_requested_at: float | None = None


ValidateFn = Callable[[np.ndarray, int], bool]
SegmentFn = Callable[[Segment], None]
StateFn = Callable[[State], None]
LevelFn = Callable[[float], None]


class AudioEngine:
    def __init__(
        self,
        cfg: Config,
        vad_model_path: Path | None,
        *,
        validate_words: ValidateFn,
        on_segment: SegmentFn,
        on_state: StateFn | None = None,
        on_level: LevelFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.vad_model_path = Path(vad_model_path) if vad_model_path else None
        self.validate_words = validate_words
        self.on_segment = on_segment
        self.on_state = on_state or (lambda _: None)
        self.on_level = on_level or (lambda _: None)

        self._state = State.IDLE
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._validation_deadline = 0.0

        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=256)
        self._stop_ev = threading.Event()
        self._worker: threading.Thread | None = None
        self._stream: sd.InputStream | None = None
        # Guards stream open/close — set_vad_enabled (UI thread),
        # start_manual (hotkey thread), and the worker thread can all
        # race here.
        self._stream_lock = threading.Lock()
        self._external_stop = threading.Event()
        self._external_start = threading.Event()
        self._external_abort = threading.Event()
        self._stop_requested_at: float | None = None

        self._vad = None  # built on start when enabled
        self._block_count = 0  # for periodic debug logs

        # Runtime pause toggle for VAD auto-triggering. Distinct from
        # ``cfg.vad.enabled``: we keep the model loaded but silently
        # ignore it while paused. Toggled from the tray or the hotkey.
        self._vad_paused = False

        # Rolling lookback ring, populated while IDLE and consumed when
        # we transition into VALIDATING / MANUAL. Prevents a clipped
        # first word when VAD fires late or the user hits the hotkey a
        # beat after starting to speak.
        self._lookback_chunks: list[np.ndarray] = []
        self._lookback_samples = 0
        self._lookback_target = int(
            self.cfg.audio.sample_rate * self.cfg.audio.lookback_ms / 1000
        )

    # --- public control ----------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def vad_loaded(self) -> bool:
        return self._vad is not None

    @property
    def vad_enabled(self) -> bool:
        """Live read of the (possibly runtime-mutated) config flag,
        gated on the VAD model actually being available."""
        return bool(self.cfg.vad.enabled) and self.vad_loaded

    @property
    def vad_paused(self) -> bool:
        return self._vad_paused

    @property
    def vad_active(self) -> bool:
        """True iff VAD is enabled *and* not paused — the live check
        the audio loop uses to decide whether to auto-trigger."""
        return self.vad_enabled and not self._vad_paused

    def set_vad_enabled(self, enabled: bool) -> None:
        """Flip the master VAD toggle (tray-level). Resets any transient
        pause state. Safe to call from any thread."""
        if bool(self.cfg.vad.enabled) == enabled and not self._vad_paused:
            return
        log.info("auto-listen %s (was enabled=%s paused=%s)",
                 "ON" if enabled else "OFF",
                 self.cfg.vad.enabled,
                 self._vad_paused)
        self.cfg.vad.enabled = enabled
        self._vad_paused = False
        if self._vad is not None:
            try:
                self._vad.reset()
            except Exception:
                pass
        # Manual-only mode keeps the mic closed when idle; auto-listen
        # needs it always open. Toggle to match the new setting.
        if enabled:
            self._ensure_stream_open()
        elif self._state is State.IDLE:
            self._close_stream()

    def set_vad_paused(self, paused: bool) -> None:
        """Toggle the runtime pause (hotkey-level, not persisted).
        Safe to call from any thread."""
        if self._vad_paused == paused:
            return
        log.info("VAD %s", "paused" if paused else "resumed")
        self._vad_paused = paused
        if self._vad is not None:
            try:
                self._vad.reset()
            except Exception:
                pass

    def start(self) -> None:
        if self._worker is not None:
            return
        # Build VAD whenever the model file is available, regardless of
        # whether auto-listen is on in config right now — the user can
        # flip it at runtime via the tray, and we don't want to force an
        # app restart just to enable it.
        if self.vad_model_path is not None:
            self._build_vad()
            log.info(
                "audio engine starting (VAD loaded, auto-listen=%s)",
                "on" if self.cfg.vad.enabled else "off",
            )
        else:
            log.info(
                "audio engine starting (no VAD model — hotkey-only recording)"
            )
        self._stop_ev.clear()
        self._worker = threading.Thread(
            target=self._run, name="justsayit-audio", daemon=True
        )
        self._worker.start()
        # Manual-only mode (no VAD) keeps the mic closed until the user
        # presses the hotkey / activation button — no point holding the
        # microphone open for a stream we never look at.
        if self.cfg.vad.enabled:
            self._ensure_stream_open()
        else:
            log.info("auto-listen off — mic stays closed until activation")

    def stop(self) -> None:
        log.info("audio engine stopping")
        self._stop_ev.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._close_stream()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def start_manual(self) -> None:
        """External trigger: begin recording (bypass VAD)."""
        log.info("start_manual requested (state=%s)", self._state.value)
        # In manual-only mode the mic is closed until activation —
        # open it before flipping the external_start flag so the worker
        # has chunks to consume.
        self._ensure_stream_open()
        self._external_start.set()

    def stop_manual(self) -> None:
        """External trigger: stop current recording and emit the buffer."""
        self._stop_requested_at = time.monotonic()
        log.info(
            "stop_manual requested (state=%s, buffered=%.2fs)",
            self._state.value,
            self._buffered_seconds(),
        )
        self._external_stop.set()

    def abort(self) -> None:
        """Discard the current recording and return to IDLE without
        emitting a segment. No transcription, no paste. Used by the
        overlay's abort (×) button."""
        self._external_abort.set()
        log.info(
            "abort requested (state=%s, buffered=%.2fs)",
            self._state.value,
            self._buffered_seconds(),
        )

    # --- internals ---------------------------------------------------------

    def _build_vad(self):
        import sherpa_onnx

        assert self.vad_model_path is not None
        log.info("loading Silero VAD from %s", self.vad_model_path)
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(self.vad_model_path)
        vad_cfg.silero_vad.threshold = float(self.cfg.vad.silero_threshold)
        vad_cfg.silero_vad.min_silence_duration = float(
            self.cfg.vad.min_silence_seconds
        )
        vad_cfg.silero_vad.min_speech_duration = float(self.cfg.vad.min_speech_seconds)
        vad_cfg.sample_rate = int(self.cfg.audio.sample_rate)
        vad_cfg.num_threads = max(1, int(self.cfg.model.num_threads))
        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_cfg,
            buffer_size_in_seconds=int(self.cfg.vad.max_segment_seconds) + 5,
        )

    def _ensure_stream_open(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                return
            sr = self.cfg.audio.sample_rate
            block = max(32, int(sr * self.cfg.audio.block_ms / 1000))
            log.info(
                "opening audio stream: device=%s rate=%d channels=%d block=%d",
                self.cfg.audio.device,
                sr,
                self.cfg.audio.channels,
                block,
            )

            def _cb(indata, frames, _time, status):
                if status:
                    log.debug("sounddevice status: %s", status)
                try:
                    self._q.put_nowait(indata[:, 0].astype(np.float32, copy=True))
                except queue.Full:
                    log.warning("audio queue full; dropping %d frames", frames)

            self._stream = sd.InputStream(
                samplerate=sr,
                channels=self.cfg.audio.channels,
                dtype="float32",
                blocksize=block,
                device=self.cfg.audio.device,
                callback=_cb,
            )
            self._stream.start()
            log.info("audio stream open")

    def _close_stream(self) -> None:
        with self._stream_lock:
            if self._stream is None:
                return
            log.info("closing audio stream")
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover
                log.exception("error closing audio stream")
            self._stream = None
        # Reset transient audio state so we don't leak cached chunks /
        # level updates when the mic re-opens later.
        self._reset_lookback()
        try:
            self.on_level(0.0)
        except Exception:  # pragma: no cover
            log.exception("on_level callback raised")

    def _set_state(self, new: State) -> None:
        if new == self._state:
            return
        log.info("state %s -> %s", self._state.value, new.value)
        self._state = new
        try:
            self.on_state(new)
        except Exception:  # pragma: no cover
            log.exception("on_state callback raised")

    def _append(self, samples: np.ndarray) -> None:
        self._buffer.append(samples)
        self._buffer_samples += len(samples)

    def _flush(self) -> np.ndarray:
        if not self._buffer:
            return np.empty(0, dtype=np.float32)
        out = np.concatenate(self._buffer)
        self._buffer.clear()
        self._buffer_samples = 0
        return out

    def _update_lookback(self, chunk: np.ndarray) -> None:
        """Push ``chunk`` onto the lookback ring and drop old chunks
        while the total exceeds ``_lookback_target``."""
        if self._lookback_target <= 0:
            return
        self._lookback_chunks.append(chunk)
        self._lookback_samples += len(chunk)
        while (
            len(self._lookback_chunks) > 1
            and self._lookback_samples - len(self._lookback_chunks[0])
            >= self._lookback_target
        ):
            dropped = self._lookback_chunks.pop(0)
            self._lookback_samples -= len(dropped)

    def _consume_lookback(self) -> None:
        """Move the lookback ring into the segment buffer and clear it.
        Called on IDLE→VALIDATING and IDLE→MANUAL so the segment starts
        ``lookback_ms`` before the trigger."""
        if not self._lookback_chunks:
            return
        secs = self._lookback_samples / float(self.cfg.audio.sample_rate)
        log.debug("consuming %.0fms lookback into segment", secs * 1000)
        for prev in self._lookback_chunks:
            self._append(prev)
        self._lookback_chunks.clear()
        self._lookback_samples = 0

    def _reset_lookback(self) -> None:
        self._lookback_chunks.clear()
        self._lookback_samples = 0

    def _emit(self, reason: str) -> None:
        secs = self._buffered_seconds()
        if self._buffer_samples == 0:
            log.info("emit (%s) called with empty buffer — skipping", reason)
            return
        samples = self._flush()
        stop_at = self._stop_requested_at if reason == "manual" else None
        # Consume the stop timestamp once we've attached it to a segment.
        self._stop_requested_at = None
        seg = Segment(
            samples=samples,
            sample_rate=self.cfg.audio.sample_rate,
            reason=reason,
            stop_requested_at=stop_at,
        )
        log.info("emitting segment: %.2fs reason=%s", secs, reason)
        try:
            self.on_segment(seg)
        except Exception:  # pragma: no cover
            log.exception("on_segment callback raised")

    def _buffered_seconds(self) -> float:
        return self._buffer_samples / float(self.cfg.audio.sample_rate)

    def _run(self) -> None:
        sr = self.cfg.audio.sample_rate
        max_samples = int(self.cfg.vad.max_segment_seconds * sr)

        prev_speech = False

        while not self._stop_ev.is_set():
            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                chunk = None

            if chunk is None:
                if self._stop_ev.is_set():
                    break
                continue

            self._block_count += 1
            if self._block_count % 200 == 1:
                log.debug(
                    "audio loop alive (block %d, state=%s, buffered=%.2fs)",
                    self._block_count,
                    self._state.value,
                    self._buffered_seconds(),
                )

            # RMS for UI
            if len(chunk):
                rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
                try:
                    self.on_level(rms)
                except Exception:  # pragma: no cover
                    log.exception("on_level callback raised")
            else:
                rms = 0.0

            started = ended = False
            use_vad = self.vad_active  # live check so pause toggles take effect
            if use_vad:
                self._feed_vad(chunk)
                speech = self._vad_is_speech()
                started = speech and not prev_speech
                ended = (not speech) and prev_speech
                prev_speech = speech
            else:
                # Drop any stale speech state while paused/disabled so we
                # don't spuriously fire "ended" when re-enabled.
                prev_speech = False

            external_start = self._external_start.is_set()
            external_stop = self._external_stop.is_set()
            external_abort = self._external_abort.is_set()

            # Abort wins: discard everything, return to IDLE silently. The
            # overlay's × button uses this; we explicitly do not emit a
            # segment so nothing is transcribed or pasted.
            if external_abort:
                self._external_abort.clear()
                self._external_stop.clear()
                self._stop_requested_at = None
                if self._state is not State.IDLE:
                    log.info(
                        "abort: discarding %.2fs (state=%s)",
                        self._buffered_seconds(),
                        self._state.value,
                    )
                self._flush()
                if self._vad is not None:
                    try:
                        self._vad.reset()
                    except Exception:
                        pass
                prev_speech = False
                self._set_state(State.IDLE)
                continue

            if external_start:
                self._external_start.clear()
                if self._state in (State.IDLE, State.VALIDATING):
                    if self._state is State.IDLE:
                        self._consume_lookback()
                    self._set_state(State.MANUAL)

            if self._state is State.IDLE:
                if use_vad and started:
                    self._consume_lookback()
                    self._append(chunk)
                    self._validation_deadline = (
                        time.monotonic() + float(self.cfg.vad.validation_seconds)
                    )
                    self._set_state(State.VALIDATING)

            elif self._state is State.VALIDATING:
                self._append(chunk)
                # Validate early if speech ended inside the window —
                # otherwise short utterances end before the 3s deadline
                # and we'd hang in RECORDING waiting for a second
                # speech→silence edge that never comes.
                at_deadline = time.monotonic() >= self._validation_deadline
                speech_already_ended = ended or not prev_speech
                if ended or at_deadline:
                    trigger = "ended" if ended else "deadline"
                    snapshot = np.concatenate(self._buffer)
                    log.info(
                        "running validation transcription (%.2fs, trigger=%s)",
                        len(snapshot) / sr,
                        trigger,
                    )
                    try:
                        ok = bool(self.validate_words(snapshot, sr))
                    except Exception:
                        log.exception("validate_words raised; discarding segment")
                        ok = False
                    if not ok:
                        log.info(
                            "validation failed — discarding %.2fs",
                            self._buffered_seconds(),
                        )
                        self._flush()
                        if self._vad is not None:
                            try:
                                self._vad.reset()
                            except Exception:
                                pass
                        prev_speech = False
                        self._set_state(State.IDLE)
                    elif speech_already_ended:
                        # Short utterance: the whole sentence is in the
                        # buffer already. Emit now instead of stepping
                        # into RECORDING (which would hang waiting for
                        # another end-of-speech edge).
                        log.info(
                            "validation passed and speech ended; "
                            "emitting short segment"
                        )
                        self._emit("vad")
                        if self._vad is not None:
                            try:
                                self._vad.reset()
                            except Exception:
                                pass
                        prev_speech = False
                        self._set_state(State.IDLE)
                    else:
                        log.info(
                            "validation passed — continuing to record"
                        )
                        self._set_state(State.RECORDING)

            elif self._state is State.RECORDING:
                self._append(chunk)
                if ended:
                    self._emit("vad")
                    self._set_state(State.IDLE)
                elif self._buffer_samples >= max_samples:
                    log.info("max segment length reached; emitting partial")
                    self._emit("max-length")

            elif self._state is State.MANUAL:
                self._append(chunk)

            if external_stop:
                self._external_stop.clear()
                if self._state in (State.VALIDATING, State.RECORDING, State.MANUAL):
                    self._emit("manual")
                    if self._vad is not None:
                        try:
                            self._vad.reset()
                        except Exception:
                            pass
                    prev_speech = False
                    self._set_state(State.IDLE)
                else:
                    log.info("stop_manual ignored (already idle)")

            # In manual-only mode we close the mic between recordings —
            # the user explicitly opted out of always-on listening.
            # Lookback is also pointless in this mode (no audio survives
            # between sessions anyway), so skip maintaining it.
            if self._state is State.IDLE and not self.cfg.vad.enabled:
                if self._stream is not None:
                    self._close_stream()
                continue

            # Maintain the lookback ring while idle; drop it the moment
            # we start buffering so stale audio can't leak into a fresh
            # session on the next trigger.
            if self._state is State.IDLE:
                self._update_lookback(chunk)
            elif self._lookback_chunks:
                self._reset_lookback()

        if self._buffer_samples:
            self._emit("shutdown")
        log.info("audio worker exited")

    def _feed_vad(self, samples: np.ndarray) -> None:
        assert self._vad is not None
        self._vad.accept_waveform(samples)

    def _vad_is_speech(self) -> bool:
        assert self._vad is not None
        try:
            return bool(self._vad.is_speech_detected())
        except AttributeError:  # pragma: no cover
            return not self._vad.empty()
