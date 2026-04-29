"""PostprocessorBase and shared HTTP + logging helpers."""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from justsayit._http import request_with_retry
from justsayit.config import resolve_secret

from ._profile import PostprocessProfile, ProcessResult, _resolve_system_prompt_file

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared HTTP + logging helpers
# ---------------------------------------------------------------------------


def _json_post(
    url: str,
    body: dict,
    headers: dict,
    *,
    profile: PostprocessProfile,
    label: str = "LLM",
) -> dict:
    """POST JSON *body* to *url*, retrying on transient HTTP errors.

    Pulls timeout / retry knobs from *profile*; sets the standard
    JSON + User-Agent headers on top of the caller's *headers*.
    """
    encoded = json.dumps(body).encode("utf-8")
    all_headers = {
        "Content-Type": "application/json",
        "User-Agent": "justsayit",
        **headers,
    }
    req = urllib.request.Request(url, data=encoded, headers=all_headers, method="POST")
    raw = request_with_retry(
        req,
        timeout=profile.request_timeout,
        retries=profile.remote_retries,
        delay=profile.remote_retry_delay_seconds,
        label=label,
    )
    return json.loads(raw)


def _log_usage(profile: PostprocessProfile, usage: dict) -> None:
    """Log token counts and optional cost from a *usage* dict."""
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    token_summary = f"{prompt_tokens} prompt"
    if cached_tokens:
        token_summary += f" ({cached_tokens} cached)"
    token_summary += f" + {completion_tokens} completion = {prompt_tokens + completion_tokens} tokens"
    any_price = (
        profile.input_price_per_1m
        or profile.output_price_per_1m
        or profile.cached_input_price_per_1m
    )
    if any_price:
        # Cached tokens are charged at cached_input_price_per_1m; only the
        # non-cached portion pays the full input rate.
        input_cost = (prompt_tokens - cached_tokens) / 1_000_000 * profile.input_price_per_1m
        cached_cost = cached_tokens / 1_000_000 * profile.cached_input_price_per_1m
        output_cost = completion_tokens / 1_000_000 * profile.output_price_per_1m
        total_cost = input_cost + cached_cost + output_cost
        log.info(
            "LLM usage: %s | cost $%.6f (input $%.6f, cached $%.6f, output $%.6f)",
            token_summary, total_cost, input_cost, cached_cost, output_cost,
        )
    else:
        log.info("LLM usage: %s", token_summary)


# ---------------------------------------------------------------------------
# Base postprocessor
# ---------------------------------------------------------------------------


class PostprocessorBase:
    """Shared helpers for all postprocessor backends.

    Subclasses must implement :meth:`_run`.
    """

    def __init__(
        self,
        profile: PostprocessProfile,
        *,
        dynamic_context_script: str = "",
    ) -> None:
        self.profile = profile
        self.dynamic_context_script = dynamic_context_script
        self._llm = None  # only used by LocalBackend; kept here for back-compat
        self._paste_strip = self._compile_paste_strip(profile.paste_strip_regex)

    def _dynamic_context(self) -> str:
        script = self.dynamic_context_script.strip()
        if not script:
            return ""
        try:
            proc = subprocess.run(
                ["bash", str(Path(script).expanduser())],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
        except Exception:
            log.exception("dynamic context script failed to run: %s", script)
            return ""
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if stderr:
                log.warning(
                    "dynamic context script exited with %d: %s (%s)",
                    proc.returncode, script, stderr,
                )
            else:
                log.warning(
                    "dynamic context script exited with %d: %s",
                    proc.returncode, script,
                )
            return ""
        dynamic = proc.stdout.strip()
        if dynamic:
            log.info("using dynamic context from %s", Path(script).expanduser())
            log.debug("dynamic context:\n%s", dynamic)
        else:
            log.debug("dynamic context script returned empty output: %s", script)
        return dynamic

    def _require_api_key(self) -> str:
        """Resolve the profile API key or raise a clear RuntimeError."""
        api_key = resolve_secret(self.profile.api_key, self.profile.api_key_env)
        if not api_key:
            raise RuntimeError(
                "LLM endpoint is set but no API key was found.\n"
                f"  Set api_key in the profile, export {self.profile.api_key_env},\n"
                "  or put it in ~/.config/justsayit/.env."
            )
        return api_key

    @staticmethod
    def _compile_paste_strip(pattern: str) -> re.Pattern[str] | None:
        if not pattern.strip():
            return None
        try:
            return re.compile(pattern, re.DOTALL)
        except re.error as exc:
            log.error("invalid paste_strip_regex %r: %s — disabled", pattern, exc)
            return None

    def strip_for_paste(self, text: str) -> str:
        """Apply ``paste_strip_regex`` to *text*, returning *text* unchanged
        if no strip regex is configured."""
        if self._paste_strip is None:
            return text
        return self._paste_strip.sub("", text)

    def find_strip_matches(self, text: str) -> list[str]:
        """Return substrings of *text* matched by ``paste_strip_regex``.

        If the pattern has at least one capture group, returns group 1 of
        each match; otherwise returns the whole match. Empty list if no
        regex is configured.
        """
        if self._paste_strip is None:
            return []
        has_groups = self._paste_strip.groups > 0
        return [
            m.group(1) if has_groups else m.group(0)
            for m in self._paste_strip.finditer(text)
        ]

    def _build_system_prompt_parts(
        self,
        extra_context: str = "",
        extra_image_provided: bool = False,
        history_text: str = "",
        assistant_mode: bool = False,
    ) -> tuple[str, str]:
        """Return ``(static, dynamic)`` parts of the system prompt.

        *static*  — prompt file + ``append_to_system_prompt`` + user context.
                    Stable across calls; safe to cache.
        *dynamic* — history (if any) + ``dynamic-context.sh`` output + clipboard.
                    Changes every call; must not be cached.
        """
        prompt = self.profile.system_prompt.strip()
        if not prompt and self.profile.system_prompt_file.strip():
            prompt = _resolve_system_prompt_file(self.profile.system_prompt_file).strip()
        extra = self.profile.append_to_system_prompt.strip()
        if extra:
            prompt = f"{prompt}\n\n{extra}" if prompt else extra
        ctx = self.profile.context.strip()
        if ctx:
            prompt = f"{prompt}\n\n# User context\n{ctx}"

        dynamic_parts: list[str] = []
        if assistant_mode:
            dynamic_parts.append(
                "# ASSISTANT MODE\n"
                "Hey Computer — the user activated interactive assistant mode via the UI button. "
                "Every input from here on is addressed to you directly (treat it as if the user said "
                "'Hey Computer, …' before each message). "
                "Respond as a helpful assistant — answer questions, take actions using available tools. "
                "Do not treat the input as transcription text to clean up."
            )
        if history_text:
            dynamic_parts.append(history_text)
        dynamic = self._dynamic_context()
        if dynamic:
            dynamic_parts.append(f"# STATE (DYNAMIC CONTEXT):\n{dynamic}")
        clip = extra_context.strip()
        if clip:
            dynamic_parts.append(
                "# The user explicitly provided you with its current clipboard content as additional context "
                "(This always means you are in Assistant mode!)\n"
                "As the system assistant, you have access to the current clipboard content and need to use it as "
                "additional context for processing the user's request. "
                "## START clipboard content\n"
                f"{clip}\n"
                "## END clipboard content\n"
            )
        if extra_image_provided and not clip:
            dynamic_parts.append(
                "# The user has shared an image from their clipboard as additional context "
                "(This always means you are in Assistant mode!)\n"
                "Analyze or help with the image based on the spoken request."
            )
        return prompt, "\n\n".join(dynamic_parts)

    def _build_system_prompt(self, extra_context: str = "", history_text: str = "", extra_image_provided: bool = False, assistant_mode: bool = False) -> str:
        static, dynamic = self._build_system_prompt_parts(extra_context, extra_image_provided=extra_image_provided, history_text=history_text, assistant_mode=assistant_mode)
        return "\n\n".join(filter(None, [static, dynamic]))

    def _build_messages(
        self, text: str, extra_context: str = "", history_text: str = "", extra_image_provided: bool = False, assistant_mode: bool = False
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self._build_system_prompt(extra_context, history_text=history_text, extra_image_provided=extra_image_provided, assistant_mode=assistant_mode)},
            {"role": "user", "content": self.profile.user_template.format(text=text)},
        ]
        log.debug("assembled LLM system prompt:\n%s", messages[0]["content"])
        return messages

    def _build_messages_continued(
        self, text: str, extra_context: str, prev_messages: list[dict], extra_image_provided: bool = False, assistant_mode: bool = False
    ) -> list[dict]:
        messages = [
            {"role": "system", "content": self._build_system_prompt(extra_context, extra_image_provided=extra_image_provided, assistant_mode=assistant_mode)},
            *prev_messages,
            {"role": "user", "content": self.profile.user_template.format(text=text)},
        ]
        log.debug("assembled LLM system prompt (continued):\n%s", messages[0]["content"])
        return messages

    @staticmethod
    def _format_history_text(prev_messages: list[dict]) -> str:
        lines = ["## PREVIOUS SESSION HISTORY"]
        for msg in prev_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Content blocks (image + text): extract text parts only.
                # Images are not representable as plain text in the history string.
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if b.get("type") in ("text", "input_text") and b.get("text")
                ]
                content = " ".join(text_parts)
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)

    def _build_user_history_entry(
        self,
        user_text: str,
        extra_context: str = "",
        extra_image: bytes | None = None,
        extra_image_mime: str = "",
    ) -> dict:
        """Build the canonical user message dict for prev_messages storage.

        Includes clipboard text and image regardless of whether the current
        backend supports them, so cross-backend continuation preserves all input.
        Plain string content is kept when neither is present.
        """
        has_image = bool(extra_image and extra_image_mime)
        if not extra_context and not has_image:
            return {"role": "user", "content": user_text}
        content: list[dict] = [{"type": "text", "text": user_text}]
        if extra_context:
            content.append({"type": "text", "text": f"[Clipboard context]\n{extra_context.strip()}"})
        if has_image:
            img_detail = getattr(self.profile, "image_detail", "auto")
            if img_detail not in ("auto", "low", "high"):
                img_detail = "auto"
            img_b64 = base64.b64encode(extra_image).decode("ascii")  # type: ignore[arg-type]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{extra_image_mime};base64,{img_b64}", "detail": img_detail},
            })
        return {"role": "user", "content": content}

    def warmup(self) -> None:
        """No-op for remote backends. LocalBackend overrides this."""
        pass

    def _run(
        self,
        text: str,
        extra_context: str = "",
        extra_image: bytes | None = None,
        extra_image_mime: str = "",
        previous_session: dict | None = None,
        tools: list | None = None,
        tool_caller=None,
        assistant_mode: bool = False,
    ) -> ProcessResult:
        raise NotImplementedError

    def process_with_reasoning(
        self,
        text: str,
        *,
        extra_context: str = "",
        extra_image: bytes | None = None,
        extra_image_mime: str = "",
        previous_session: dict | None = None,
        tools: list | None = None,
        tool_caller=None,
        assistant_mode: bool = False,
    ) -> ProcessResult:
        """Run the LLM on *text* and return the result including any reasoning.

        Routes to the subclass ``_run`` method. ``extra_context`` is appended
        to the system prompt under a labeled "Clipboard as additional context"
        section — used by the overlay's clipboard-context button.
        ``extra_image`` / ``extra_image_mime`` carry a raw image captured from
        the clipboard; only ``ResponsesBackend`` uses them (other backends
        silently ignore). ``text`` falls back to the original input when the
        model returns an empty response.
        ``previous_session`` carries the session.json payload when continue
        mode is active; backends use it to prepend history or chain via
        ``previous_response_id``.
        ``tools`` is a list of OpenAI-format tool dicts; ``tool_caller`` is
        a callable(name, params) → str that executes tools and returns results.
        Both are optional — backends that don't support function calling ignore them.
        ``assistant_mode`` tells the model it is in interactive assistant mode
        rather than transcription-cleanup mode.
        """
        result = self._run(text, extra_context, extra_image, extra_image_mime, previous_session, tools, tool_caller, assistant_mode)
        if not result.text:
            result = ProcessResult(text=text, reasoning=result.reasoning, session_data=result.session_data)
        return result

    def process(self, text: str) -> str:
        """Backward-compatible thin wrapper: returns just the cleaned text."""
        return self.process_with_reasoning(text).text


# Back-compat alias — old code that instantiated LLMPostprocessor directly
# continues to work; ``make_postprocessor`` in __init__.py is the new API.
LLMPostprocessor = PostprocessorBase
