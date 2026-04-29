"""OpenAI Responses API backend (/v1/responses)."""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

_MAX_TOOL_ROUNDS = 10

from ._processor import PostprocessorBase, _json_post, _log_usage, log
from ._profile import ProcessResult


class ResponsesBackend(PostprocessorBase):
    @staticmethod
    def _canonical_to_responses_input(prev_msgs: list[dict]) -> list[dict]:
        """Convert canonical chat-completions prev_messages to Responses API input format.

        Used for cross-backend continuation so images in session history are
        sent as actual input_image blocks rather than being stripped to text.
        """
        result = []
        for msg in prev_msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                if isinstance(content, str):
                    items: list[dict] = [{"type": "input_text", "text": content}]
                else:
                    items = []
                    for block in content:
                        if block.get("type") == "text":
                            items.append({"type": "input_text", "text": block.get("text", "")})
                        elif block.get("type") == "image_url":
                            img = block.get("image_url", {})
                            items.append({
                                "type": "input_image",
                                "image_url": img.get("url", ""),
                                "detail": img.get("detail", "auto"),
                            })
                result.append({"role": "user", "content": items})
            elif role == "assistant":
                if isinstance(content, str):
                    items = [{"type": "output_text", "text": content}]
                else:
                    items = [
                        {"type": "output_text", "text": b.get("text", "")}
                        for b in content
                        if b.get("type") in ("text", "output_text")
                    ]
                result.append({"role": "assistant", "content": items})
        return result

    def _run(self, text: str, extra_context: str = "", extra_image: bytes | None = None, extra_image_mime: str = "", previous_session: dict | None = None, tools: list | None = None, tool_caller=None, assistant_mode: bool = False) -> ProcessResult:
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
        # Cross-backend: prev_msgs present but no native response_id chain available.
        cross_backend_hist = not same_backend and bool(prev_msgs)

        has_image = bool(extra_image and extra_image_mime and self.profile.image_detail != "off")
        # Compute b64 once; reused for both the API payload and session history.
        img_b64 = base64.b64encode(extra_image).decode("ascii") if has_image else ""  # type: ignore[arg-type]

        static_prompt, dynamic_prompt = self._build_system_prompt_parts(
            extra_context, extra_image_provided=has_image, assistant_mode=assistant_mode
        )
        log.debug("assembled Responses API instructions (static/cached):\n%s", static_prompt)
        if dynamic_prompt:
            log.debug("assembled Responses API dynamic context (uncached):\n%s", dynamic_prompt)

        user_text = self.profile.user_template.format(text=text)
        user_content: list[dict] = [{"type": "input_text", "text": user_text}]
        if has_image:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:{extra_image_mime};base64,{img_b64}",
                "detail": self.profile.image_detail,
            })
            log.debug(
                "attaching image (%s, %d bytes, detail=%s)",
                extra_image_mime, len(extra_image), self.profile.image_detail,
            )

        if dynamic_prompt or len(user_content) > 1 or cross_backend_hist:
            input_payload: Any = []
            if dynamic_prompt:
                input_payload.append({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": dynamic_prompt}],
                })
            if cross_backend_hist:
                input_payload.extend(self._canonical_to_responses_input(prev_msgs))
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
        # Responses API uses flat tool format: {"type":"function","name":...}
        # (chat/completions nests under a "function" key — different shape).
        use_custom_tools = bool(tools and tool_caller and self.profile.use_tools)
        body_tools: list[dict] = []
        if self.profile.responses_web_search:
            trigger = self.profile.responses_web_search_trigger
            if not trigger or extra_context or has_image or assistant_mode or re.search(trigger, text):
                body_tools.append({"type": "web_search"})
        if use_custom_tools:
            for t in (tools or []):
                fn = t.get("function", {})
                body_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
        if body_tools:
            body["tools"] = body_tools

        url = self.profile.endpoint.rstrip("/") + "/responses"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = _json_post(
            url, body, headers,
            profile=self.profile,
            label="Responses API",
        )

        # Custom tool-call loop.  Responses API returns function_call items in
        # output[]; follow up with previous_response_id + function_call_result.
        if use_custom_tools:
            for _ in range(_MAX_TOOL_ROUNDS):
                fc_items = [i for i in (data.get("output") or []) if i.get("type") == "function_call"]
                if not fc_items:
                    break
                tool_results: list[dict] = []
                for fc in fc_items:
                    fn_name = fc.get("name", "")
                    call_id = fc.get("call_id", "")
                    try:
                        fn_args = json.loads(fc.get("arguments", "{}"))
                    except Exception:
                        fn_args = {}
                    result_str = tool_caller(fn_name, fn_args)
                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result_str,
                    })
                followup: dict[str, Any] = {
                    "model": self.profile.model,
                    "previous_response_id": data.get("id", ""),
                    "input": tool_results,
                    "max_output_tokens": self.profile.max_tokens,
                }
                if self.profile.reasoning_effort:
                    followup["reasoning"] = {"effort": self.profile.reasoning_effort}
                data = _json_post(
                    url, followup, headers,
                    profile=self.profile,
                    label="Responses API (tool follow-up)",
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
        user_msg = self._build_user_history_entry(
            self.profile.user_template.format(text=text), extra_context, extra_image, extra_image_mime
        )
        new_prev_messages = prev_msgs + [user_msg, {"role": "assistant", "content": content}]
        session_data = {
            "backend": "responses",
            "prev_messages": new_prev_messages,
            "response_id": data.get("id", ""),
            "ts": time.time(),
        }
        return ProcessResult(text=content, session_data=session_data)
