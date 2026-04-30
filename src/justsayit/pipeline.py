"""Segment processing pipeline: transcribe → filter → LLM → paste."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from justsayit.filters import apply_filters

if TYPE_CHECKING:
    from justsayit.audio import Segment
    from justsayit.config import Config
    from justsayit.transcribe import TranscriberBase
    from justsayit.postprocess import PostprocessorBase
    from justsayit.overlay import OverlayWindow
    from justsayit.paste import Paster
    from justsayit.tools import ToolDefinition

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
        after_llm_filters: list | None = None,
        on_error: Callable[[str, str, Callable[[], None] | None], None] | None = None,
        enqueue_segment: Callable[["Segment"], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.transcriber = transcriber
        self.filters = filters
        self.paster = paster
        self.no_paste = no_paste
        self.after_llm_filters: list = after_llm_filters or []
        self.postprocessor: "PostprocessorBase | None" = None  # set externally
        self.overlay: "OverlayWindow | None" = None             # set externally
        self.tool_definitions: "list[ToolDefinition]" = []     # set externally
        self.assistant_mode: bool = False                       # set externally
        self.on_error = on_error
        self.enqueue_segment = enqueue_segment
        self._last_transcription_time: float | None = None
        self._last_detected_text: str | None = None
        self.last_was_assistant_mode: bool = False

    def _build_retry_cb(self, seg: "Segment") -> Callable[[], None] | None:
        if self.enqueue_segment is None:
            return None
        eq = self.enqueue_segment
        def _retry() -> None:
            try:
                eq(seg)
            except Exception:
                log.exception("retry re-enqueue failed")
        return _retry

    def _emit_error(self, stage: str, exc: BaseException, seg: "Segment") -> None:
        if self.on_error is None:
            return
        msg = (str(exc).strip().splitlines() or [exc.__class__.__name__])[0]
        try:
            self.on_error(stage, msg, self._build_retry_cb(seg))
        except Exception:
            log.exception("on_error callback raised")

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
        try:
            raw = self.transcriber.transcribe(seg.samples, seg.sample_rate)
        except Exception as exc:
            log.exception("transcription failed")
            self._emit_error("transcribe", exc, seg)
            return
        dt = time.monotonic() - t0
        log.info("transcription done in %.2fs: raw=%r", dt, raw)
        if not raw:
            log.info("empty transcription; nothing to paste")
            if self.overlay is not None:
                self.overlay.push_hide()
            return
        try:
            final = apply_filters(raw, self.filters)
        except Exception as e:
            log.exception("filters failed")
            self._emit_error("filters", str(e))
            return
        if final != raw:
            log.info("filters changed output: %r -> %r", raw, final)

        # Cache post-filter text and current mode for redo_with_override.
        self._last_detected_text = final
        self.last_was_assistant_mode = self.assistant_mode

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
            if is_continue:
                if previous_session:
                    prev_msgs = previous_session.get("prev_messages") or []
                    log.info(
                        "continue: loaded %d-turn history (backend=%s)",
                        len(prev_msgs) // 2,
                        previous_session.get("backend", "?"),
                    )
                else:
                    log.info("continue: no previous session found — starting fresh")
            tools = None
            tool_caller = None
            if self.tool_definitions and getattr(pp.profile, "use_tools", True) and self.assistant_mode:
                from justsayit.tools import execute_tool
                tools = [td.to_openai_format() for td in self.tool_definitions]
                _overlay = self.overlay
                _tools_by_name = {td.name: td for td in self.tool_definitions}
                def _call_tool(name: str, params: dict) -> str:
                    if _overlay is not None:
                        _overlay.push_tool_call(name, params)
                    td = _tools_by_name.get(name)
                    if td is None:
                        log.warning("tool %r called but not defined", name)
                        return f"Error: tool '{name}' is not defined."
                    return execute_tool(td, params)
                tool_caller = _call_tool
            t_llm0 = time.monotonic()
            try:
                result = pp.process_with_reasoning(
                    final, extra_context=extra_context,
                    extra_image=extra_image, extra_image_mime=extra_image_mime,
                    previous_session=previous_session,
                    tools=tools,
                    tool_caller=tool_caller,
                    assistant_mode=self.assistant_mode,
                )
                t_llm1 = time.monotonic()
                log.info("LLM call took %.0fms", (t_llm1 - t_llm0) * 1000)
                cleaned = result.text
                log.info("LLM: %r -> %r", final, cleaned)
                llm_overlay_text = cleaned
                paste_text = cleaned
            except Exception as exc:
                log.exception("LLM postprocessor failed; using unprocessed text")
                detail = (str(exc).strip().splitlines() or [exc.__class__.__name__])[0]
                llm_overlay_text = f"LLM error: {detail or exc.__class__.__name__}"
                # Also surface an amber error pill with retry so the
                # failure is visible even when the user wasn't watching
                # the LLM line.
                self._emit_error("llm", exc, seg)
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
                if self.after_llm_filters:
                    try:
                        normalized = apply_filters(stripped, self.after_llm_filters)
                    except Exception as e:
                        log.exception("after-LLM filters failed")
                        self._emit_error("filters", str(e))
                        normalized = stripped
                    if normalized != stripped:
                        log.info(
                            "after-LLM filters applied: %r -> %r",
                            stripped[:60], normalized[:60],
                        )
                    paste_text = normalized
                    llm_overlay_text = normalized
                if result.session_data:
                    prev_msgs = result.session_data.get("prev_messages") or []
                    log.info(
                        "session saved (%d turns, backend=%s)",
                        len(prev_msgs) // 2,
                        result.session_data.get("backend", "?"),
                    )
                    _save_session(result.session_data)
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

        if self.no_paste or not self.cfg.paste.enabled or self.assistant_mode:
            print(final, flush=True)
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
            self._emit_error("paste", str(e))
        finally:
            # Linger so the user can read the transcribed text regardless of
            # whether paste succeeded or failed.
            if self.overlay is not None:
                self.overlay.push_linger_start()

    def redo_with_override(self, *, assistant_mode_override: bool) -> None:
        """Re-run the last detected (post-filter) text through the LLM with
        an explicit mode override.  Skips ASR and filters — those already ran.
        No-op (with a warning) when no cached text is available."""
        from justsayit.paste import PasteError

        if self._last_detected_text is None:
            log.warning("redo_with_override: no cached detected text; ignoring")
            return

        pp = self.postprocessor
        if pp is None:
            log.warning("redo_with_override: no postprocessor active; ignoring")
            return

        text = self._last_detected_text

        # Switch overlay to LLM-thinking placeholder.
        if self.overlay is not None:
            self.overlay.push_detected_text(text, llm_pending=True)

        # Mirror the regular handle() path's tool wiring so the LLM gets
        # the exact same call shape it would for a normal recording with
        # the corresponding mode. Free-text "REDO" nudges aren't enough
        # for models like Gemma with strict static-prompt rules — only
        # the same prompt+tool structure as the regular path reliably
        # flips them between cleanup and assistant.
        tools = None
        tool_caller = None
        if (
            self.tool_definitions
            and getattr(pp.profile, "use_tools", True)
            and assistant_mode_override
        ):
            from justsayit.tools import execute_tool
            _overlay = self.overlay
            _tools_by_name = {td.name: td for td in self.tool_definitions}
            tools = [td.to_openai_format() for td in self.tool_definitions]

            def _call_tool(name: str, params: dict) -> str:
                if _overlay is not None:
                    _overlay.push_tool_call(name, params)
                td = _tools_by_name.get(name)
                if td is None:
                    log.warning("tool %r called but not defined", name)
                    return f"Error: tool '{name}' is not defined."
                return execute_tool(td, params)
            tool_caller = _call_tool

        llm_overlay_text = text
        llm_overlay_thought = ""
        paste_text = text
        try:
            result = pp.process_with_reasoning(
                text,
                tools=tools,
                tool_caller=tool_caller,
                assistant_mode=assistant_mode_override,
            )
            cleaned = result.text
            log.info("redo LLM: %r -> %r", text, cleaned)
            llm_overlay_text = cleaned
            paste_text = cleaned
        except Exception as exc:
            log.exception("redo LLM postprocessor failed; using unprocessed text")
            detail = (str(exc).strip().splitlines() or [exc.__class__.__name__])[0]
            llm_overlay_text = f"LLM error: {detail or exc.__class__.__name__}"
        else:
            stripped = pp.strip_for_paste(paste_text)
            if stripped != paste_text:
                matches = [
                    m.strip()
                    for m in pp.find_strip_matches(paste_text)
                    if m.strip()
                ]
                llm_overlay_thought = "\n".join(matches)
            if result.reasoning:
                llm_overlay_thought = result.reasoning
            llm_overlay_text = stripped
            paste_text = stripped

        if self.overlay is not None:
            self.overlay.push_llm_text(llm_overlay_text, thought=llm_overlay_thought)

        self.last_was_assistant_mode = assistant_mode_override

        if self.no_paste or not self.cfg.paste.enabled or assistant_mode_override:
            print(paste_text, flush=True)
            log.info("redo paste disabled — text only printed")
            if self.overlay is not None:
                self.overlay.push_linger_start()
            return

        if self.paster is None:
            log.warning("redo: paster not ready; skipping paste")
            if self.overlay is not None:
                self.overlay.push_linger_start()
            return
        try:
            log.info("redo pasting %d chars", len(paste_text))
            self.paster.paste(paste_text)
        except PasteError as e:
            log.error("redo paste failed: %s", e)
        finally:
            if self.overlay is not None:
                self.overlay.push_linger_start()
