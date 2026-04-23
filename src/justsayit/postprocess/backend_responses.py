"""OpenAI Responses API backend (/v1/responses)."""
from __future__ import annotations

import base64
import re
from typing import Any

from ._processor import PostprocessorBase, _http_post, _log_usage, log
from ._profile import ProcessResult


class ResponsesBackend(PostprocessorBase):
    def _run(self, text: str, extra_context: str = "", extra_image: bytes | None = None, extra_image_mime: str = "") -> ProcessResult:
        """OpenAI Responses API POST (/v1/responses).

        The static system prompt goes in ``instructions`` (cached prefix);
        dynamic context and clipboard go in a developer message inside
        ``input`` (uncached per-call).
        """
        api_key = self._require_api_key()
        if not self.profile.model:
            raise RuntimeError(
                "Responses API backend: profile.model is empty — "
                "set 'model' in the profile (e.g. \"gpt-5.4-mini\")."
            )

        has_image = bool(extra_image and extra_image_mime and self.profile.image_detail != "off")
        static_prompt, dynamic_prompt = self._build_system_prompt_parts(
            extra_context, extra_image_provided=has_image
        )
        log.debug("assembled Responses API instructions (static/cached):\n%s", static_prompt)
        if dynamic_prompt:
            log.debug("assembled Responses API dynamic context (uncached):\n%s", dynamic_prompt)

        user_text = self.profile.user_template.format(text=text)
        user_content: list[dict] = [{"type": "input_text", "text": user_text}]
        if has_image:
            b64 = base64.b64encode(extra_image).decode("ascii")  # type: ignore[arg-type]
            user_content.append({
                "type": "input_image",
                "image_url": f"data:{extra_image_mime};base64,{b64}",
                "detail": self.profile.image_detail,
            })
            log.debug(
                "attaching image (%s, %d bytes, detail=%s)",
                extra_image_mime, len(extra_image), self.profile.image_detail,
            )

        if dynamic_prompt or len(user_content) > 1:
            input_payload: Any = []
            if dynamic_prompt:
                input_payload.append({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": dynamic_prompt}],
                })
            input_payload.append({
                "role": "user",
                "content": user_content,
            })
        else:
            input_payload = user_text

        body: dict[str, Any] = {
            "model": self.profile.model,
            "instructions": static_prompt,
            "input": input_payload,
            "max_output_tokens": self.profile.max_tokens,
        }
        if self.profile.prompt_cache_retention:
            body["prompt_cache_retention"] = self.profile.prompt_cache_retention
        if self.profile.reasoning_effort:
            body["reasoning"] = {"effort": self.profile.reasoning_effort}
        if self.profile.responses_web_search:
            trigger = self.profile.responses_web_search_trigger
            if not trigger or extra_context or has_image or re.search(trigger, text):
                body["tools"] = [{"type": "web_search"}]

        url = self.profile.endpoint.rstrip("/") + "/responses"
        data = _http_post(
            url,
            body,
            {"Authorization": f"Bearer {api_key}"},
            remote_retries=self.profile.remote_retries,
            remote_retry_delay_seconds=self.profile.remote_retry_delay_seconds,
            request_timeout=self.profile.request_timeout,
            label="Responses API",
        )

        output_items = data.get("output") or []
        text_parts = []
        search_count = 0
        for item in output_items:
            if item.get("type") == "web_search_call" and item.get("status") == "completed":
                search_count += 1
            elif item.get("type") == "message":
                for block in item.get("content") or []:
                    if block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
        content = " ".join(text_parts).strip()
        if search_count:
            search_cost = search_count * self.profile.web_search_price_per_call
            if search_cost:
                log.info(
                    "web search: %d call(s) × $%.4f = $%.4f "
                    "(token cost for search results included in LLM usage below)",
                    search_count, self.profile.web_search_price_per_call, search_cost,
                )
            else:
                log.info(
                    "web search: %d call(s) "
                    "(set web_search_price_per_call in profile to log cost)",
                    search_count,
                )

        # Normalize Responses API usage to the shape _log_usage expects.
        raw = data.get("usage") or {}
        cache_details = (
            raw.get("prompt_tokens_details")
            or raw.get("input_tokens_details")
            or {}
        )
        _log_usage(self.profile, {
            "prompt_tokens": int(raw.get("input_tokens") or 0),
            "completion_tokens": int(raw.get("output_tokens") or 0),
            "prompt_tokens_details": {
                "cached_tokens": int(cache_details.get("cached_tokens") or 0)
            },
        })
        return ProcessResult(text=content)
