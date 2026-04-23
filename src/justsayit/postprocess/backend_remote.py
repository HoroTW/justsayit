"""OpenAI-compatible /chat/completions backend."""
from __future__ import annotations

import re
import time
from typing import Any

from ._processor import PostprocessorBase, _http_post, _log_usage, log
from ._profile import ProcessResult


class RemoteBackend(PostprocessorBase):
    def _run(self, text: str, extra_context: str = "", extra_image: bytes | None = None, extra_image_mime: str = "", previous_session: dict | None = None) -> ProcessResult:
        """OpenAI-compatible /chat/completions POST."""
        api_key = self._require_api_key()
        if not self.profile.model:
            raise RuntimeError(
                "LLM endpoint is set but profile.model is empty — "
                "set 'model' in the profile (e.g. \"gpt-4o-mini\")."
            )
        same_backend = previous_session is not None and previous_session.get("backend") == "remote"
        prev_msgs: list[dict] = (previous_session.get("prev_messages") or []) if previous_session else []
        url = self.profile.endpoint.rstrip("/") + "/chat/completions"
        # OpenAI reasoning models (o1/o3/o4/gpt-5.x …) reject most classic
        # sampling knobs and renamed ``max_tokens`` → ``max_completion_tokens``.
        is_reasoning = bool(
            re.match(r"^(o[1-9]|gpt-[5-9])", self.profile.model or "")
        )
        if same_backend and prev_msgs:
            messages = self._build_messages_continued(text, extra_context, prev_msgs)
        elif prev_msgs:
            history_text = self._format_history_text(prev_msgs)
            messages = self._build_messages(text, extra_context, history_text=history_text)
        else:
            messages = self._build_messages(text, extra_context)
        body: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
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
        user_msg = {"role": "user", "content": self.profile.user_template.format(text=text)}
        new_prev_messages = prev_msgs + [user_msg, {"role": "assistant", "content": content}]
        session_data = {
            "backend": "remote",
            "prev_messages": new_prev_messages,
            "ts": time.time(),
        }
        return ProcessResult(text=content, reasoning=reasoning, session_data=session_data)
