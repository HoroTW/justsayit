"""LLMPostprocessor and per-backend request logic."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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
    data = json.dumps(body).encode("utf-8")
    all_headers = {
        "Content-Type": "application/json",
        "User-Agent": "justsayit",
        **headers,
    }
    attempts = 1 + max(0, remote_retries)
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=data, headers=all_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = ""
            last_error = RuntimeError(
                f"{label} endpoint returned HTTP {exc.code}: {exc.reason}\n  {detail}"
            )
            if not retryable or attempt >= attempts:
                raise last_error from exc
            log.warning(
                "%s request failed with HTTP %d; retrying %d/%d in %.1fs",
                label, exc.code, attempt, attempts - 1, remote_retry_delay_seconds,
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            last_error = RuntimeError(f"{label} request failed: {reason}")
            if attempt >= attempts:
                raise last_error from exc
            log.warning(
                "%s request failed; retrying %d/%d in %.1fs: %s",
                label, attempt, attempts - 1, remote_retry_delay_seconds, reason,
            )
        time.sleep(max(0.0, remote_retry_delay_seconds))
    assert last_error is not None
    raise last_error


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
# LLM postprocessor
# ---------------------------------------------------------------------------


class LLMPostprocessor:
    """Synchronous LLM cleanup step.

    The model is loaded lazily on the first call to :meth:`process`
    (or eagerly via :meth:`warmup`). All calls are serialised by a
    threading lock so the same instance can safely be reused from the
    transcription worker thread.
    """

    def __init__(
        self,
        profile: PostprocessProfile,
        *,
        dynamic_context_script: str = "",
    ) -> None:
        self.profile = profile
        self.dynamic_context_script = dynamic_context_script
        self._llm = None
        self._lock = threading.Lock()
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

    def _resolved_model_path(self) -> Path:
        p = Path(self.profile.model_path).expanduser()
        if p.exists():
            return p
        if self.profile.hf_repo and self.profile.hf_filename:
            from justsayit.model import _download, models_dir

            dest = models_dir() / "llm" / self.profile.hf_filename
            if not dest.exists():
                url = (
                    f"https://huggingface.co/{self.profile.hf_repo}"
                    f"/resolve/main/{self.profile.hf_filename}"
                )
                log.info("downloading LLM model: %s", url)
                _download(url, dest)
            return dest
        raise RuntimeError(
            f"LLM model file not found: {p}\n"
            "Set 'model_path' in the profile, or configure 'hf_repo' + 'hf_filename' "
            "for automatic download."
        )

    def _build(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "  With Vulkan GPU:  CMAKE_ARGS='-DGGML_VULKAN=1' "
                "uv pip install llama-cpp-python\n"
                "  CPU only:         uv pip install llama-cpp-python"
            ) from exc

        model_path = self._resolved_model_path()
        log.info(
            "loading LLM %s  n_gpu_layers=%d  n_ctx=%d",
            model_path.name, self.profile.n_gpu_layers, self.profile.n_ctx,
        )
        return Llama(
            model_path=str(model_path),
            n_gpu_layers=self.profile.n_gpu_layers,
            n_ctx=self.profile.n_ctx,
            verbose=False,
        )

    def warmup(self) -> None:
        """Eagerly load the local model. No-op for remote-endpoint profiles."""
        if self.profile.base in {"remote", "responses", "anthropic"}:
            return
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
                self._install_chat_template_kwargs()

    def _build_system_prompt_parts(self, extra_context: str = "") -> tuple[str, str]:
        """Return ``(static, dynamic)`` parts of the system prompt.

        *static*  — prompt file + ``append_to_system_prompt`` + user context.
                    Stable across calls; safe to cache.
        *dynamic* — ``dynamic-context.sh`` output + clipboard content.
                    Changes every call; must not be cached.

        Used by :meth:`_responses_process` and :meth:`_anthropic_process`.
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
        # Inline ``system_prompt`` wins; otherwise resolve from the file.
        prompt = self.profile.system_prompt.strip()
        if not prompt and self.profile.system_prompt_file.strip():
            prompt = _resolve_system_prompt_file(self.profile.system_prompt_file).strip()
        extra = self.profile.append_to_system_prompt.strip()
        if extra:
            prompt = f"{prompt}\n\n{extra}" if prompt else extra
        dynamic = self._dynamic_context()
        if dynamic:
            prompt = f"# STATE (DYNAMIC CONTEXT):\n{dynamic}\n\n----\n\n{prompt}"
        ctx = self.profile.context.strip()
        if ctx:
            prompt = f"{prompt}\n\n# User context\n{ctx}"
        clip = extra_context.strip()
        if clip:
            prompt = (
                f"{prompt}\n\n# The user explicitly provided you with its current clipboard content as additional context "
                f"(This always means you are in Assistant mode!)\n"
                f"As the system assistant, you have access to the current clipboard content and need to use it as "
                f"additional context for processing the user's request. "
                f"## START clipboard content\n"
                f"{clip}\n"
                f"## END clipboard content\n"
            )
        return prompt

    def _build_messages(
        self, text: str, extra_context: str = ""
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self._build_system_prompt(extra_context)},
            {"role": "user", "content": self.profile.user_template.format(text=text)},
        ]
        log.info("assembled LLM system prompt:\n%s", messages[0]["content"])
        return messages

    def _install_chat_template_kwargs(self) -> None:
        # ``Llama.create_chat_completion()`` has a fixed keyword signature
        # (no ``**kwargs``), so passing ``chat_template_kwargs=`` raises
        # ``TypeError``. The chat handler underneath *does* accept
        # ``**kwargs`` and forwards them into the Jinja template, so we
        # wrap the handler to inject our profile's template kwargs at
        # call time (e.g. Qwen 3.5's ``enable_thinking``).
        if not self.profile.chat_template_kwargs:
            return
        template_kwargs = dict(self.profile.chat_template_kwargs)
        # Mirror the lookup order in ``Llama.create_chat_completion``:
        # (1) explicit chat_handler, (2) the per-instance ``_chat_handlers``
        # dict (where GGUF-embedded Jinja templates live), then (3) the
        # global static registry. Skipping (2) blows up on every modern GGUF
        # with a bundled template — Gemma, Qwen 3.5, Llama 3.x.
        base_handler = self._llm.chat_handler
        if base_handler is None:
            base_handler = self._llm._chat_handlers.get(self._llm.chat_format)
        if base_handler is None:
            from llama_cpp import llama_chat_format
            base_handler = llama_chat_format.get_chat_completion_handler(
                self._llm.chat_format
            )

        def _handler(**call_kwargs: Any):
            merged = dict(template_kwargs)
            merged.update(call_kwargs)
            return base_handler(**merged)

        self._llm.chat_handler = _handler

    # --- Backends -----------------------------------------------------------

    def _local_process(self, text: str, extra_context: str = "") -> ProcessResult:
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
                self._install_chat_template_kwargs()
            kwargs: dict[str, Any] = {
                "messages": self._build_messages(text, extra_context),
                "temperature": self.profile.temperature,
                "max_tokens": self.profile.max_tokens,
                "top_p": self.profile.top_p,
                "top_k": self.profile.top_k,
                "min_p": self.profile.min_p,
                "repeat_penalty": self.profile.repeat_penalty,
                "presence_penalty": self.profile.presence_penalty,
                "frequency_penalty": self.profile.frequency_penalty,
            }
            resp = self._llm.create_chat_completion(**kwargs)
        # llama-cpp-python keeps thinking inline in ``content``; the
        # display/paste split is done downstream via ``paste_strip_regex``.
        return ProcessResult(text=resp["choices"][0]["message"]["content"].strip())

    def _remote_process(self, text: str, extra_context: str = "") -> ProcessResult:
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

    def _responses_process(self, text: str, extra_context: str = "") -> ProcessResult:
        """OpenAI Responses API POST (/v1/responses).

        The static system prompt goes in ``instructions`` (cached prefix);
        dynamic context and clipboard go in a developer message inside
        ``input`` (uncached per-call).
        """
        api_key = resolve_secret(self.profile.api_key, self.profile.api_key_env)
        if not api_key:
            raise RuntimeError(
                "LLM endpoint is set but no API key was found.\n"
                f"  Set api_key in the profile, export {self.profile.api_key_env},\n"
                "  or put it in ~/.config/justsayit/.env."
            )
        if not self.profile.model:
            raise RuntimeError(
                "Responses API backend: profile.model is empty — "
                "set 'model' in the profile (e.g. \"gpt-5.4-mini\")."
            )

        static_prompt, dynamic_prompt = self._build_system_prompt_parts(extra_context)
        log.info("assembled Responses API instructions (static/cached):\n%s", static_prompt)
        if dynamic_prompt:
            log.info("assembled Responses API dynamic context (uncached):\n%s", dynamic_prompt)

        user_text = self.profile.user_template.format(text=text)
        if dynamic_prompt:
            input_payload: Any = [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": dynamic_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                },
            ]
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
            if not trigger or re.search(trigger, text):
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

    def _anthropic_process(self, text: str, extra_context: str = "") -> ProcessResult:
        """Native Anthropic /v1/messages POST with prompt caching.

        The static part of the system prompt is sent as a cached block;
        the dynamic part is sent uncached so the cache point stays stable.
        """
        api_key = resolve_secret(self.profile.api_key, self.profile.api_key_env)
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not found.\n"
                f"  Set api_key in the profile, export {self.profile.api_key_env},\n"
                "  or put it in ~/.config/justsayit/.env."
            )
        if not self.profile.model:
            raise RuntimeError(
                "Anthropic backend: profile.model is empty — "
                "set 'model' in the profile (e.g. \"claude-sonnet-4-6\")."
            )

        static_prompt, dynamic_prompt = self._build_system_prompt_parts(extra_context)
        log.info("assembled Anthropic system prompt (static/cached):\n%s", static_prompt)
        if dynamic_prompt:
            log.info("assembled Anthropic system prompt (dynamic/uncached):\n%s", dynamic_prompt)

        system_blocks: list[dict[str, Any]] = []
        if static_prompt:
            system_blocks.append({
                "type": "text",
                "text": static_prompt,
                "cache_control": {"type": "ephemeral"},
            })
        if dynamic_prompt:
            system_blocks.append({"type": "text", "text": dynamic_prompt})

        body: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": self.profile.max_tokens,
            "messages": [
                {"role": "user", "content": self.profile.user_template.format(text=text)}
            ],
        }
        if system_blocks:
            body["system"] = system_blocks
        if self.profile.anthropic_web_search:
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

        betas = ["prompt-caching-2024-07-31"]
        if self.profile.anthropic_extended_cache:
            betas.append("extended-cache-ttl-2025-02-19")

        url = self.profile.endpoint.rstrip("/") + "/messages"
        data = _http_post(
            url,
            body,
            {
                "x-api-key": api_key,
                "anthropic-version": self.profile.anthropic_version,
                "anthropic-beta": ",".join(betas),
            },
            remote_retries=self.profile.remote_retries,
            remote_retry_delay_seconds=self.profile.remote_retry_delay_seconds,
            request_timeout=self.profile.request_timeout,
            label="Anthropic",
        )

        # Collect text blocks; ignore tool_use / tool_result blocks from
        # web search (the final answer always comes in a text block).
        content_blocks = data.get("content") or []
        content = " ".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        ).strip()

        # Normalize Anthropic usage fields to the shape _log_usage expects.
        raw = data.get("usage") or {}
        cache_read = int(raw.get("cache_read_input_tokens") or 0)
        cache_write = int(raw.get("cache_creation_input_tokens") or 0)
        if cache_read or cache_write:
            log.info(
                "Anthropic cache: %d tokens read from cache, %d tokens written to cache",
                cache_read, cache_write,
            )
        _log_usage(self.profile, {
            "prompt_tokens": int(raw.get("input_tokens") or 0) + cache_write,
            "completion_tokens": int(raw.get("output_tokens") or 0),
            "prompt_tokens_details": {"cached_tokens": cache_read},
        })
        return ProcessResult(text=content)

    # --- Dispatch -----------------------------------------------------------

    def process_with_reasoning(
        self, text: str, *, extra_context: str = ""
    ) -> ProcessResult:
        """Run the LLM on *text* and return the result including any reasoning.

        Routes by ``profile.base``. ``extra_context`` is appended to the
        system prompt under a labeled "Clipboard as additional context"
        section — used by the overlay's clipboard-context button.
        ``text`` falls back to the original input when the model returns
        an empty response.
        """
        if self.profile.base == "remote":
            result = self._remote_process(text, extra_context)
        elif self.profile.base == "responses":
            result = self._responses_process(text, extra_context)
        elif self.profile.base == "anthropic":
            result = self._anthropic_process(text, extra_context)
        else:
            result = self._local_process(text, extra_context)
        if not result.text:
            result = ProcessResult(text=text, reasoning=result.reasoning)
        return result

    def process(self, text: str) -> str:
        """Backward-compatible thin wrapper: returns just the cleaned text."""
        return self.process_with_reasoning(text).text
