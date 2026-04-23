"""OpenAI-compatible /chat/completions backend."""
from __future__ import annotations

import re
from typing import Any

from justsayit.config import resolve_secret
from ._processor import PostprocessorBase, _http_post, _log_usage, log
from ._profile import ProcessResult


class RemoteBackend(PostprocessorBase):
    def _run(self, text: str, extra_context: str = "") -> ProcessResult:
        """OpenAI-compatible /chat/completions POST."""
        api_key = resolve_secret(self.profile.api_key, self.profile.api_key_env)
        if not api_key:
            raise RuntimeError(
                "LLM endpoint is set but no API key was found.\n"
                f"  Set api_key in the profile, export {self.profile.api_key_env},\n"
                "  or put it in ~/.config/justsayit/.env."
            )
        if not self.profile.model:
            raise RuntimeError(
                "LLM endpoint is set but profile.model is empty — "
                "set 'model' in the profile (e.g. \"gpt-4o-mini\")."
            )
        url = self.profile.endpoint.rstrip("/") + "/chat/completions"
        # OpenAI reasoning models (o1/o3/o4/gpt-5.x …) reject most classic
        # sampling knobs and renamed ``max_tokens`` → ``max_completion_tokens``.
        is_reasoning = bool(
            re.match(r"^(o[1-9]|gpt-[5-9])", self.profile.model or "")
        )
        body: dict[str, Any] = {
            "model": self.profile.model,
            "messages": self._build_messages(text, extra_context),
        }
        if is_reasoning:
            body["max_completion_tokens"] = self.profile.max_tokens
            if self.profile.reasoning_effort:
                body["reasoning_effort"] = self.profile.reasoning_effort
        else:
            body["max_tokens"] = self.profile.max_tokens
            body["temperature"] = self.profile.temperature
            body["top_p"] = self.profile.top_p
            body["presence_penalty"] = self.profile.presence_penalty
            body["frequency_penalty"] = self.profile.frequency_penalty
            if self.profile.reasoning_effort:
                body["reasoning_effort"] = self.profile.reasoning_effort
        if self.profile.chat_template_kwargs:
            body["chat_template_kwargs"] = dict(self.profile.chat_template_kwargs)

        data = _http_post(
            url,
            body,
            {"Authorization": f"Bearer {api_key}"},
            remote_retries=self.profile.remote_retries,
            remote_retry_delay_seconds=self.profile.remote_retry_delay_seconds,
            request_timeout=self.profile.request_timeout,
            label="remote LLM",
        )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM endpoint returned no choices: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        # Some providers (DeepSeek, Qwen via vLLM, OpenRouter for reasoning
        # models) split hidden thinking into a separate field. Accept both
        # ``reasoning_content`` (DeepSeek/vLLM) and ``reasoning`` (OpenRouter).
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(reasoning, str):
            reasoning = ""
        else:
            reasoning = reasoning.strip()
        _log_usage(self.profile, data.get("usage") or {})
        return ProcessResult(text=content, reasoning=reasoning)
