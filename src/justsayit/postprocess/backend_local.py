"""Local GGUF inference backend (llama-cpp-python)."""
from __future__ import annotations

import threading
from typing import Any

from ._processor import PostprocessorBase, log
from ._profile import ProcessResult


class LocalBackend(PostprocessorBase):
    def __init__(self, profile, *, dynamic_context_script=""):
        super().__init__(profile, dynamic_context_script=dynamic_context_script)
        self._llm = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
                self._install_chat_template_kwargs()

    def _resolved_model_path(self):
        from pathlib import Path
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

    def _run(self, text: str, extra_context: str = "", extra_image: bytes | None = None, extra_image_mime: str = "", previous_session: dict | None = None) -> ProcessResult:
        import time
        prev_msgs: list[dict] = (previous_session.get("prev_messages") or []) if previous_session else []
        # Local models are text-only for inference; always use formatted history text.
        # Images in prev_msgs are preserved in session storage for cross-backend switches.
        if prev_msgs:
            history_text = self._format_history_text(prev_msgs)
            messages = self._build_messages(text, extra_context, history_text=history_text)
        else:
            messages = self._build_messages(text, extra_context)
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
                self._install_chat_template_kwargs()
            kwargs: dict[str, Any] = {
                "messages": messages,
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
        content = resp["choices"][0]["message"]["content"].strip()
        user_msg = self._build_user_history_entry(
            self.profile.user_template.format(text=text), extra_context, extra_image, extra_image_mime
        )
        new_prev_messages = prev_msgs + [user_msg, {"role": "assistant", "content": content}]
        session_data = {
            "backend": "local",
            "prev_messages": new_prev_messages,
            "ts": time.time(),
        }
        return ProcessResult(text=content, session_data=session_data)
