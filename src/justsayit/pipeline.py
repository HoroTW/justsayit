"""Segment processing pipeline: transcribe → filter → LLM → paste."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from justsayit.filters import apply_filters

if TYPE_CHECKING:
    from justsayit.audio import Segment
    from justsayit.config import Config
    from justsayit.transcribe import TranscriberBase
    from justsayit.postprocess import PostprocessorBase
    from justsayit.overlay import OverlayWindow
    from justsayit.paste import Paster

log = logging.getLogger(__name__)


def _session_path() -> Path:
    from justsayit.config import cache_dir
    return cache_dir() / "session.json"


def _load_session() -> dict | None:
    try:
        return json.loads(_session_path().read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_session(data: dict) -> None:
    p = _session_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def _clear_session() -> None:
    try:
        _session_path().unlink(missing_ok=True)
    except Exception:
        pass


class SegmentPipeline:
    """Owns the transcribe→filter→LLM→paste flow for one segment."""

    def __init__(
        self,
        cfg: "Config",
        transcriber: "TranscriberBase",
        filters: list,
        paster: "Paster | None",
        *,
        no_paste: bool = False,
    ) -> None:
        self.cfg = cfg
        self.transcriber = transcriber
        self.filters = filters
        self.paster = paster
        self.no_paste = no_paste
        self.postprocessor: "PostprocessorBase | None" = None  # set externally
        self.overlay: "OverlayWindow | None" = None             # set externally
        self._last_transcription_time: float | None = None

    def handle(self, seg: "Segment", *, consume_clipboard_fn=None, is_continue: bool = False) -> None:
        """Process one audio segment end-to-end."""
        from justsayit.paste import PasteError

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
            llm_overlay_text = final
            llm_overlay_thought = ""
            paste_text = final
            if consume_clipboard_fn is not None:
                extra_context, extra_image, extra_image_mime = consume_clipboard_fn()
            else:
                extra_context, extra_image, extra_image_mime = "", None, ""
            previous_session = _load_session() if is_continue else None
            t_llm0 = time.monotonic()
            try:
                result = pp.process_with_reasoning(
                    final, extra_context=extra_context,
                    extra_image=extra_image, extra_image_mime=extra_image_mime,
                    previous_session=previous_session,
                )
                t_llm1 = time.monotonic()
                log.info("LLM call took %.0fms", (t_llm1 - t_llm0) * 1000)
                cleaned = result.text
                log.info("LLM: %r -> %r", final, cleaned)
                llm_overlay_text = cleaned
                paste_text = cleaned
            except Exception as exc:
                log.exception("LLM postprocessor failed; using unprocessed text")
                detail = str(exc).strip().splitlines()[0]
                llm_overlay_text = f"LLM error: {detail or exc.__class__.__name__}"
                if not is_continue:
                    _clear_session()
            else:
                stripped = pp.strip_for_paste(paste_text)
                # Surface the reasoning preamble (whatever paste_strip_regex
                # matched) above the body so the user sees the full LLM reply
                # but only the stripped body lands in the focused window.
                if stripped != paste_text:
                    matches = [
                        m.strip()
                        for m in pp.find_strip_matches(paste_text)
                        if m.strip()
                    ]
                    llm_overlay_thought = "\n".join(matches)
                    log.info(
                        "paste_strip_regex applied: %d -> %d chars",
                        len(paste_text),
                        len(stripped),
                    )
                # Remote backends (DeepSeek, Qwen via vLLM, OpenRouter) can
                # return structured reasoning in a separate field. When
                # present, prefer that — it's cleaner than regex-matched
                # inline blocks (and the local path can't populate it).
                if result.reasoning:
                    llm_overlay_thought = result.reasoning
                    log.info(
                        "remote returned reasoning field: %d chars",
                        len(result.reasoning),
                    )
                llm_overlay_text = stripped
                paste_text = stripped
                if is_continue and result.session_data:
                    _save_session(result.session_data)
                elif not is_continue:
                    _clear_session()
            # Always update the LLM field — clears "Wait…" even when text is unchanged.
            if self.overlay is not None:
                self.overlay.push_llm_text(
                    llm_overlay_text, thought=llm_overlay_thought
                )
            final = paste_text

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
                log.debug(
                    "waiting %.0fms for hotkey modifiers to release "
                    "(elapsed since stop=%.0fms, target=%.0fms)",
                    wait * 1000,
                    elapsed * 1000,
                    delay_target * 1000,
                )
                time.sleep(wait)
            else:
                log.debug(
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
            log.info("pasting %d chars", len(final))
            t_paste0 = time.monotonic()
            self.paster.paste(final)
            log.debug("paste call returned after %.0fms", (time.monotonic() - t_paste0) * 1000)
        except PasteError as e:
            log.error("paste failed: %s", e)
        finally:
            # Linger so the user can read the transcribed text regardless of
            # whether paste succeeded or failed.
            if self.overlay is not None:
                self.overlay.push_linger_start()
