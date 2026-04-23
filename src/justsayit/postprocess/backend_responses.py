"""OpenAI Responses API backend (/v1/responses)."""
from __future__ import annotations

import base64
import re
import time
from typing import Any

from ._processor import PostprocessorBase, _http_post, _log_usage, log
from ._profile import ProcessResult


class ResponsesBackend(PostprocessorBase):
    def _run(self, text: str, extra_context: str = "", extra_image: bytes | None = None, extra_image_mime: str = "", previous_session: dict | None = None) -> ProcessResult:
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

        same_backend = previous_session is not None and previous_session.get("backend") == "responses"
        prev_msgs: list[dict] = (previous_session.get("prev_messages") or []) if previous_session else []
        prev_response_id: str = (previous_session.get("response_id") or "") if same_backend else ""
        history_text = "" if same_backend else (self._format_history_text(prev_msgs) if prev_msgs else "")

        has_image = bool(extra_image and extra_image_mime and self.profile.image_detail != "off")
        static_prompt, dynamic_prompt = self._build_system_prompt_parts(
            extra_context, extra_image_provided=has_image, history_text=history_text
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
        if prev_response_id:
            body["previous_response_id"] = prev_response_id
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
        open_page_count = 0
        for item in output_items:
            if item.get("type") == "web_search_call" and item.get("status") == "completed":
                action_type = (item.get("action") or {}).get("type", "")
                if action_type == "search":
                    search_count += 1
                elif action_type == "open_page":
                    open_page_count += 1
            elif item.get("type") == "message":
                for block in item.get("content") or []:
                    if block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
        content = " ".join(text_parts).strip()
        if search_count or open_page_count:
            any_price = (
                self.profile.web_search_price_per_call
                or self.profile.web_open_page_price_per_call
            )
            if any_price:
                search_cost = search_count * self.profile.web_search_price_per_call
                open_page_cost = open_page_count * self.profile.web_open_page_price_per_call
                total_ws_cost = search_cost + open_page_cost
                parts = []
                if search_count:
                    parts.append(f"{search_count} search × ${self.profile.web_search_price_per_call:.4f} = ${search_cost:.4f}")
                if open_page_count:
                    parts.append(f"{open_page_count} open_page × ${self.profile.web_open_page_price_per_call:.4f} = ${open_page_cost:.4f}")
                log.info("web search: %s | total $%.4f", ", ".join(parts), total_ws_cost)
            else:
                parts = []
                if search_count:
                    parts.append(f"{search_count} search")
                if open_page_count:
                    parts.append(f"{open_page_count} open_page")
                log.info(
                    "web search: %s (set web_search_price_per_call / web_open_page_price_per_call to log cost)",
                    ", ".join(parts),
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
        user_msg = {"role": "user", "content": self.profile.user_template.format(text=text)}
        new_prev_messages = prev_msgs + [user_msg, {"role": "assistant", "content": content}]
        session_data = {
            "backend": "responses",
            "prev_messages": new_prev_messages,
            "response_id": data.get("id", ""),
            "ts": time.time(),
        }
        return ProcessResult(text=content, session_data=session_data)
