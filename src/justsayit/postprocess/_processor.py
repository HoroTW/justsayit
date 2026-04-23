"""PostprocessorBase and shared HTTP + logging helpers."""

from __future__ import annotations

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


def _http_post(
    url: str,
    body: dict,
    headers: dict,
    *,
    remote_retries: int,
    remote_retry_delay_seconds: float,
    request_timeout: float,
    label: str = "LLM",
) -> dict:
    """POST JSON *body* to *url*, retrying on transient HTTP errors."""
    encoded = json.dumps(body).encode("utf-8")
    all_headers = {
        "Content-Type": "application/json",
        "User-Agent": "justsayit",
        **headers,
    }
    req = urllib.request.Request(url, data=encoded, headers=all_headers, method="POST")
    raw = request_with_retry(req, timeout=request_timeout, retries=remote_retries, delay=remote_retry_delay_seconds, label=label)
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
            log.info(
                "dynamic context from %s:\n%s", Path(script).expanduser(), dynamic,
            )
        else:
            log.info("dynamic context script returned empty output: %s", script)
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

    def _build_system_prompt_parts(self, extra_context: str = "") -> tuple[str, str]:
        """Return ``(static, dynamic)`` parts of the system prompt.

        *static*  — prompt file + ``append_to_system_prompt`` + user context.
                    Stable across calls; safe to cache.
        *dynamic* — ``dynamic-context.sh`` output + clipboard content.
                    Changes every call; must not be cached.

        Used by :meth:`_responses_process`.
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
        return prompt, "\n\n".join(dynamic_parts)

    def _build_system_prompt(self, extra_context: str = "") -> str:
        static, dynamic = self._build_system_prompt_parts(extra_context)
        return "\n\n".join(filter(None, [static, dynamic]))

    def _build_messages(
        self, text: str, extra_context: str = ""
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self._build_system_prompt(extra_context)},
            {"role": "user", "content": self.profile.user_template.format(text=text)},
        ]
        log.debug("assembled LLM system prompt:\n%s", messages[0]["content"])
        return messages

    def warmup(self) -> None:
        """No-op for remote backends. LocalBackend overrides this."""
        pass

    def _run(self, text: str, extra_context: str = "") -> ProcessResult:
        raise NotImplementedError

    def process_with_reasoning(
        self, text: str, *, extra_context: str = ""
    ) -> ProcessResult:
        """Run the LLM on *text* and return the result including any reasoning.

        Routes to the subclass ``_run`` method. ``extra_context`` is appended
        to the system prompt under a labeled "Clipboard as additional context"
        section — used by the overlay's clipboard-context button.
        ``text`` falls back to the original input when the model returns
        an empty response.
        """
        result = self._run(text, extra_context)
        if not result.text:
            result = ProcessResult(text=text, reasoning=result.reasoning)
        return result

    def process(self, text: str) -> str:
        """Backward-compatible thin wrapper: returns just the cleaned text."""
        return self.process_with_reasoning(text).text


# Back-compat alias — old code that instantiated LLMPostprocessor directly
# continues to work; ``make_postprocessor`` in __init__.py is the new API.
LLMPostprocessor = PostprocessorBase
