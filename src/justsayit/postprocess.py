"""LLM-based post-processing of raw transcription output.

Each post-processor is driven by a *profile* — a small TOML file that
lives in ``$XDG_CONFIG_HOME/justsayit/postprocess/<name>.toml`` and
controls the GGUF model path, GPU offloading, inference parameters, and
the system prompt sent to the model.

Inference is handled by ``llama-cpp-python``.  For Vulkan GPU support on
AMD (or any non-NVIDIA card) the package must be compiled with::

    CMAKE_ARGS="-DGGML_VULKAN=1" pip install llama-cpp-python

For CPU-only use the regular wheel is sufficient.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from justsayit.config import config_dir, ensure_commented_form_file, resolve_secret

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile dataclass + loader
# ---------------------------------------------------------------------------

# Shipped prompt + config templates live as plain text files alongside
# this module so they can be edited in a content-aware editor (Markdown
# for prompts, TOML for profile templates, shell for the dynamic-context
# helper) without Python-string escaping. See ``src/justsayit/prompts/``
# and ``src/justsayit/templates/``.
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# Canonical defaults for each inference backend. These TOML files are
# the single source of truth: the dataclass defaults below are derived
# from them at module-import time, AND the user-facing profile
# templates document them by reference. Editing one of these files is
# all it takes to change a shipped default.
_BASE_DEFAULTS: dict[str, dict[str, Any]] = {
    "builtin": tomllib.loads(_load_template("builtin-defaults.toml")),
    "remote": tomllib.loads(_load_template("remote-defaults.toml")),
}


def _builtin_default(name: str, fallback: Any) -> Any:
    """Look up a default value for *name* from ``builtin-defaults.toml``.

    Used to populate dataclass field defaults so the TOML stays the
    single source of truth. The *fallback* is what the dataclass uses
    if the key happens to be absent from the builtin defaults file —
    i.e. for fields that only make sense on the ``remote`` base
    (``endpoint``, ``model``, ``api_key``, …)."""
    return _BASE_DEFAULTS["builtin"].get(name, fallback)


# Distinctive header line embedded in profile files. Used by the
# migration check in ``ensure_commented_form_file`` to recognise files
# we've already written (so we don't keep backing them up on each
# install). The literal must match the first line of every shipped
# profile template under ``templates/``.
_PROFILE_COMMENTED_FORM_MARKER = (
    "# justsayit postprocess profile (commented-defaults form)."
)


def _load_profile_template(name: str) -> str:
    """Read a packaged user-facing profile template by name.

    Profile templates are static — they document the canonical defaults
    files by reference (``base = "builtin"``, ``system_prompt_file =
    "cleanup_local.md"``) rather than embedding values, so no
    substitution is needed."""
    return _load_template(name)


_CLEANUP_PROFILE_TOML = _load_profile_template("profile-gemma4-cleanup.toml")
_FUN_PROFILE_TOML = _load_profile_template("profile-gemma4-fun.toml")
_OPENAI_PROFILE_TOML = _load_profile_template("profile-openai-cleanup.toml")
_OLLAMA_GEMMA_PROFILE_TOML = _load_profile_template("profile-ollama-gemma.toml")


@dataclass(frozen=True)
class ProcessResult:
    """Return value of :meth:`LLMPostprocessor.process_with_reasoning`.

    ``text`` is the model's visible reply (the part that goes into the
    paste pipeline). ``reasoning`` is the model's hidden thinking, when
    the backend exposes it as a structured field — currently only the
    remote backend, which reads OpenAI-style ``message.reasoning_content``
    or ``message.reasoning``. Local llama-cpp-python output keeps
    reasoning inline in ``content`` (handled via ``paste_strip_regex``),
    so for the local path ``reasoning`` is always ``""``.
    """

    text: str
    reasoning: str = ""


@dataclass
class PostprocessProfile:
    # Which backend defaults file to overlay user values onto.
    # "builtin" → llama-cpp-python loads a local GGUF.
    # "remote"  → HTTP POST to an OpenAI-compatible /chat/completions.
    base: str = _builtin_default("base", "builtin")

    # --- Inference backend (built-in via llama-cpp-python + GGUF) -------
    model_path: str = _builtin_default("model_path", "")
    hf_repo: str = _builtin_default("hf_repo", "")
    hf_filename: str = _builtin_default("hf_filename", "")
    n_gpu_layers: int = _builtin_default("n_gpu_layers", -1)
    n_ctx: int = _builtin_default("n_ctx", 4096)

    # --- Cleanup tuning -------------------------------------------------
    temperature: float = _builtin_default("temperature", 0.08)
    max_tokens: int = _builtin_default("max_tokens", 4096)
    # Sampling knobs — defaults match llama-cpp-python's
    # ``create_chat_completion`` defaults, so profiles that don't set
    # them see the same behaviour as before. Raise ``presence_penalty``
    # (e.g. 1.5) to break loops on small models; per Qwen's own docs
    # this is the most effective single lever against their 0.6B/0.8B
    # thinking-loop tendency.
    top_p: float = _builtin_default("top_p", 0.95)
    top_k: int = _builtin_default("top_k", 40)
    min_p: float = _builtin_default("min_p", 0.05)
    repeat_penalty: float = _builtin_default("repeat_penalty", 1.0)
    presence_penalty: float = _builtin_default("presence_penalty", 0.0)
    frequency_penalty: float = _builtin_default("frequency_penalty", 0.0)
    user_template: str = _builtin_default("user_template", "{text}")
    paste_strip_regex: str = _builtin_default(
        "paste_strip_regex", r"<\|channel>thought(.*?)<channel\|>"
    )

    # --- System prompt (orthogonal to backend) --------------------------
    # Path to a .md prompt file. Bare names resolve against the packaged
    # ``prompts/`` dir; paths with a slash (or ~) are loaded as-is.
    system_prompt_file: str = _builtin_default(
        "system_prompt_file", "cleanup_local.md"
    )
    # Inline override. When non-empty, takes precedence over the file.
    system_prompt: str = _builtin_default("system_prompt", "")
    # Extra text appended to the resolved system prompt — convenient for
    # tweaking the shipped default without forking it into a custom file.
    # Joined with "\n\n" so it reads as its own paragraph.
    append_to_system_prompt: str = _builtin_default(
        "append_to_system_prompt", ""
    )

    # Passthrough dict forwarded to the chat template. On llama-cpp-python
    # it reaches the Jinja renderer via ``chat_template_kwargs=``; on the
    # remote OpenAI-compatible path it's included in the JSON body under
    # the same key. Empty → not forwarded at all (keeps requests clean for
    # providers that don't understand it). Typical use: toggling
    # per-model features like Qwen 3.5's ``enable_thinking``.
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(_builtin_default("chat_template_kwargs", {}))
    )

    # --- User context (also see context.toml sidecar) -------------------
    context: str = _builtin_default("context", "")

    # --- HTTP / OpenAI-compatible backend (used when base = "remote") ---
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout: float = 60.0
    remote_retries: int = 3
    remote_retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        # Auto-infer remote backend when endpoint is set and base wasn't
        # explicitly bumped off the default. Mirrors the load_profile()
        # inference for direct dataclass construction (notably tests and
        # programmatic users who instantiate PostprocessProfile directly
        # with an endpoint).
        if self.base == "builtin" and self.endpoint:
            self.base = "remote"


def _resolve_system_prompt_file(value: str) -> str:
    """Load a prompt file from *value* (a path or bare name).

    - Bare name (no slash) → packaged ``src/justsayit/prompts/`` dir.
    - Anything else → expanded path (``~`` resolved).

    Returns the file contents as a string. Raises ``FileNotFoundError``
    with a hint if the file is missing."""
    p = Path(value).expanduser()
    if "/" not in value and "\\" not in value and not p.is_absolute():
        p = _PROMPTS_DIR / value
    if not p.exists():
        raise FileNotFoundError(
            f"system_prompt_file not found: {p}\n"
            "Bare names resolve against the packaged prompts/ dir; "
            "use a full path (e.g. ~/my-prompt.md) for files outside it."
        )
    return p.read_text(encoding="utf-8")


def profiles_dir() -> Path:
    return config_dir() / "postprocess"


# Personal context lives in its own sidecar file rather than per-profile
# so updates to shipped profile TOMLs (system prompt, model paths,
# regexes) can be replaced without clobbering anything the user wrote.
# Profile-level context still works (load_profile honors a non-empty
# `context` field in the profile) and takes precedence over the sidecar.
_CONTEXT_SIDECAR_TEMPLATE = _load_template("context-sidecar.toml")


# Default dynamic-context helper script written to ``<config_dir>/
# dynamic-context.sh`` on first run. The script's stdout is captured
# and prepended to every postprocess request as a STATE block.
_DYNAMIC_CONTEXT_SCRIPT = _load_template("dynamic-context.sh")


def context_file_path() -> Path:
    return config_dir() / "context.toml"


def dynamic_context_script_path() -> Path:
    return config_dir() / "dynamic-context.sh"


def ensure_context_file(path: Path | None = None) -> Path:
    """Write the personal-context sidecar with a documented empty template
    if it doesn't exist.  This file is purely user-data — it is never
    overwritten by ``install.sh --update`` and has no defaults baseline."""
    if path is None:
        path = context_file_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_CONTEXT_SIDECAR_TEMPLATE, encoding="utf-8")
    return path


def ensure_dynamic_context_script(path: Path | None = None) -> Path:
    """Write the default dynamic-context helper script if missing."""
    if path is None:
        path = dynamic_context_script_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DYNAMIC_CONTEXT_SCRIPT, encoding="utf-8")
        path.chmod(0o755)
    return path


def load_context_sidecar(path: Path | None = None) -> str:
    """Return the ``context`` string from ``context.toml`` (or "" if the
    file is missing or unreadable)."""
    if path is None:
        path = context_file_path()
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log.warning("could not read context sidecar %s: %s", path, exc)
        return ""
    val = raw.get("context", "")
    return val if isinstance(val, str) else ""


def _toml_validator(text: str) -> None:
    tomllib.loads(text)


def ensure_default_profile(path: Path | None = None) -> Path:
    """Write the recommended ``gemma4-cleanup.toml`` profile if it's missing.

    Writes the cleanup-style template in commented-defaults form. Used
    both by ``justsayit init`` and by ``setup-llm`` when seeding a
    per-model profile. Pre-existing legacy fully-populated profile
    files get backed up + rewritten once (see ``ensure_commented_form_file``).
    Files that carry the marker but fail TOML parsing (i.e. were written
    by an earlier buggy template) are also re-migrated.
    """
    if path is None:
        path = profiles_dir() / "gemma4-cleanup.toml"
    ensure_commented_form_file(
        path,
        _CLEANUP_PROFILE_TOML,
        _PROFILE_COMMENTED_FORM_MARKER,
        validator=_toml_validator,
    )
    return path


def ensure_fun_profile(path: Path | None = None) -> Path:
    """Write the ``gemma4-fun.toml`` companion profile if it's missing.

    A tiny emoji-heavy variant of the cleanup profile, written alongside
    it on first ``init`` so users discover the schema and have an obvious
    second profile to switch to via the ``profile`` config field.
    Same migration treatment as ``ensure_default_profile``.
    """
    if path is None:
        path = profiles_dir() / "gemma4-fun.toml"
    ensure_commented_form_file(
        path,
        _FUN_PROFILE_TOML,
        _PROFILE_COMMENTED_FORM_MARKER,
        validator=_toml_validator,
    )
    return path


def ensure_openai_profile(path: Path | None = None) -> Path:
    """Write the ``openai-cleanup.toml`` profile if it's missing.

    Same commented-defaults convention as ``gemma4-cleanup.toml`` but
    with ``base = "remote"`` and ``endpoint`` / ``model`` uncommented
    as the keys that DEFINE the OpenAI-compatible variant. The system
    prompt is selected via ``system_prompt_file = "cleanup_remote.md"``
    in ``remote-defaults.toml``.
    """
    if path is None:
        path = profiles_dir() / "openai-cleanup.toml"
    ensure_commented_form_file(
        path,
        _OPENAI_PROFILE_TOML,
        _PROFILE_COMMENTED_FORM_MARKER,
        validator=_toml_validator,
    )
    return path


def ensure_ollama_gemma_profile(path: Path | None = None) -> Path:
    """Write the ``ollama-gemma.toml`` profile if it's missing.

    Demonstrates that backend (``base = "remote"``) and prompt
    (``system_prompt_file = "cleanup_local.md"``, the Gemma
    ``<|think|>`` channel variant) are independent. Useful when running
    Gemma through a local Ollama install over the OpenAI-compatible
    /v1 endpoint.
    """
    if path is None:
        path = profiles_dir() / "ollama-gemma.toml"
    ensure_commented_form_file(
        path,
        _OLLAMA_GEMMA_PROFILE_TOML,
        _PROFILE_COMMENTED_FORM_MARKER,
        validator=_toml_validator,
    )
    return path


def ensure_default_profiles() -> tuple[Path, Path, Path, Path]:
    """Write the cleanup, fun, openai, and ollama-gemma default profiles.

    Returns ``(cleanup, fun, openai, ollama_gemma)``.
    """
    ensure_context_file()
    ensure_dynamic_context_script()
    return (
        ensure_default_profile(),
        ensure_fun_profile(),
        ensure_openai_profile(),
        ensure_ollama_gemma_profile(),
    )


def load_profile(name_or_path: str) -> PostprocessProfile:
    """Load a :class:`PostprocessProfile` from *name_or_path*.

    If the argument looks like a file path (contains a separator or ends
    with ``.toml``) it is used directly; otherwise it is resolved to
    ``config_dir()/postprocess/<name>.toml``.

    Resolution order: ``<base>-defaults.toml`` (where *base* comes from
    the user file's ``base`` field, defaulting to ``"builtin"``) is the
    starting point; the user file's keys are then overlaid on top. Any
    field not in the merged result falls through to the dataclass
    default. Legacy profiles without a ``base`` field but with
    ``endpoint`` set are auto-treated as ``base = "remote"`` so existing
    setups keep working.

    If the loaded profile's ``context`` field is empty, the personal-context
    sidecar (``~/.config/justsayit/context.toml``) is consulted so updates
    to shipped profile TOMLs don't wipe user-written context.
    """
    p = Path(name_or_path).expanduser()
    is_explicit = p.suffix == ".toml" or "/" in name_or_path or "\\" in name_or_path
    if not is_explicit:
        p = profiles_dir() / f"{name_or_path}.toml"
    if not p.exists():
        raise FileNotFoundError(
            f"Postprocess profile not found: {p}\n"
            "Run 'justsayit init' to generate the default profile, or create it manually."
        )
    with p.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    base = raw.get("base")
    if base is None:
        # Legacy profiles: infer from ``endpoint`` so existing files
        # don't silently regress to the wrong defaults after upgrade.
        base = "remote" if raw.get("endpoint") else "builtin"
    if base not in _BASE_DEFAULTS:
        log.warning(
            "profile %s: unknown base %r, falling back to 'builtin'", p, base
        )
        base = "builtin"

    merged: dict[str, Any] = {**_BASE_DEFAULTS[base], **raw, "base": base}
    valid = {fld.name for fld in fields(PostprocessProfile)}
    kwargs = {k: v for k, v in merged.items() if k in valid}
    profile = PostprocessProfile(**kwargs)
    # Profile-level `context` (if non-empty) wins over the sidecar so
    # users with per-profile overrides keep working unchanged.
    if not profile.context.strip():
        profile.context = load_context_sidecar()
    return profile


# ---------------------------------------------------------------------------
# LLM postprocessor
# ---------------------------------------------------------------------------


class LLMPostprocessor:
    """Synchronous LLM cleanup step using llama-cpp-python.

    The model is loaded lazily on the first call to :meth:`process`
    (or eagerly via :meth:`warmup`).  All calls are serialised by a
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
                    proc.returncode,
                    script,
                    stderr,
                )
            else:
                log.warning(
                    "dynamic context script exited with %d: %s",
                    proc.returncode,
                    script,
                )
            return ""
        dynamic = proc.stdout.strip()
        if dynamic:
            log.info(
                "dynamic context from %s:\n%s",
                Path(script).expanduser(),
                dynamic,
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
        """Apply ``paste_strip_regex`` to *text*. Returns *text* unchanged
        if the profile has no strip regex (or it was invalid)."""
        if self._paste_strip is None:
            return text
        return self._paste_strip.sub("", text)

    def find_strip_matches(self, text: str) -> list[str]:
        """Return the substrings of *text* that ``paste_strip_regex`` matches.

        Used by the overlay to display the stripped "thought" / reasoning
        preamble alongside the pasted body so the user can see the full
        model reply.

        If the pattern has at least one capture group, the value of group 1
        is returned for each match — letting users wrap parens around just
        the thought *content* (e.g. ``<\\|channel>(.*?)<channel\\|>``) so
        the framing tokens are stripped from the overlay too. Without a
        capture group, the whole match is returned (legacy behaviour).
        Empty list if no regex is configured.
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
            model_path.name,
            self.profile.n_gpu_layers,
            self.profile.n_ctx,
        )
        return Llama(
            model_path=str(model_path),
            n_gpu_layers=self.profile.n_gpu_layers,
            n_ctx=self.profile.n_ctx,
            verbose=False,
        )

    def warmup(self) -> None:
        """Eagerly load the local model so the first transcription is not
        slow.  No-op for the remote endpoint path — there is nothing to
        load locally and a probe request would cost real money / latency."""
        if self.profile.base == "remote":
            return
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
                self._install_chat_template_kwargs()

    def _system_prompt(self) -> str:
        # Inline ``system_prompt`` wins; otherwise resolve from the
        # ``system_prompt_file`` (the canonical mechanism — Gemma's
        # ``<|think|>`` prompt and the channel-free OpenAI variant are
        # both just .md files on disk, picked per-profile rather than
        # auto-swapped based on backend).
        prompt = self.profile.system_prompt.strip()
        if not prompt and self.profile.system_prompt_file.strip():
            prompt = _resolve_system_prompt_file(
                self.profile.system_prompt_file
            ).strip()
        extra = self.profile.append_to_system_prompt.strip()
        if extra:
            prompt = f"{prompt}\n\n{extra}" if prompt else extra
        dynamic = self._dynamic_context()
        if dynamic:
            prompt = f"# STATE (DYNAMIC CONTEXT):\n{dynamic}\n\n----\n\n{prompt}"
        ctx = self.profile.context.strip()
        if ctx:
            prompt = f"{prompt}\n\n# User context\n{ctx}"
        return prompt

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self.profile.user_template.format(text=text)},
        ]
        log.info("assembled LLM system prompt:\n%s", messages[0]["content"])
        return messages

    def _remote_process(self, text: str) -> ProcessResult:
        """OpenAI-compatible /chat/completions POST.  Pure stdlib (no
        ``openai`` dep) — same response shape as ``llama_cpp`` so the
        extraction below mirrors the local path."""
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
        body: dict[str, Any] = {
            "model": self.profile.model,
            "messages": self._build_messages(text),
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "top_p": self.profile.top_p,
            "presence_penalty": self.profile.presence_penalty,
            "frequency_penalty": self.profile.frequency_penalty,
        }
        if self.profile.chat_template_kwargs:
            # Forwarded to the server's template renderer. Supported by
            # Ollama, vLLM, SGLang, LM Studio, llama.cpp-server. OpenAI
            # proper rejects unknown fields with HTTP 400, so the remote
            # default is {} — opt in only when pointing at a server that
            # understands the field.
            body["chat_template_kwargs"] = dict(self.profile.chat_template_kwargs)
        attempts = 1 + max(0, self.profile.remote_retries)
        last_error: RuntimeError | None = None
        for attempt in range(1, attempts + 1):
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "justsayit",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    req, timeout=self.profile.request_timeout
                ) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    detail = ""
                last_error = RuntimeError(
                    f"LLM endpoint returned HTTP {exc.code}: {exc.reason}\n  {detail}"
                )
                if not retryable or attempt >= attempts:
                    raise last_error from exc
                log.warning(
                    "remote LLM request failed with HTTP %d; retrying %d/%d in %.1fs",
                    exc.code,
                    attempt,
                    attempts - 1,
                    self.profile.remote_retry_delay_seconds,
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = getattr(exc, "reason", exc)
                last_error = RuntimeError(f"LLM endpoint request failed: {reason}")
                if attempt >= attempts:
                    raise last_error from exc
                log.warning(
                    "remote LLM request failed; retrying %d/%d in %.1fs: %s",
                    attempt,
                    attempts - 1,
                    self.profile.remote_retry_delay_seconds,
                    reason,
                )
            time.sleep(max(0.0, self.profile.remote_retry_delay_seconds))
        else:
            assert last_error is not None
            raise last_error
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM endpoint returned no choices: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        # Some OpenAI-compatible providers (DeepSeek, Qwen via vLLM, OpenRouter
        # for reasoning models, …) split the model's hidden thinking into a
        # separate field. Naming isn't standardised — DeepSeek + vLLM use
        # ``reasoning_content``, OpenRouter uses ``reasoning`` — so accept
        # both. Empty string when the field is absent.
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        if isinstance(reasoning, str):
            reasoning = reasoning.strip()
        else:
            reasoning = ""
        return ProcessResult(text=content, reasoning=reasoning)

    def _install_chat_template_kwargs(self) -> None:
        # ``Llama.create_chat_completion()`` has a fixed keyword signature
        # (no ``**kwargs``), so passing ``chat_template_kwargs=`` raises
        # ``TypeError``. The chat handler underneath *does* accept
        # ``**kwargs`` and forwards them into the Jinja template, so we
        # wrap the handler to inject our profile's template kwargs at
        # call time (e.g. Qwen 3.5's ``enable_thinking``).
        if not self.profile.chat_template_kwargs:
            return
        from llama_cpp import llama_chat_format

        template_kwargs = dict(self.profile.chat_template_kwargs)
        # Mirror the lookup order in ``Llama.create_chat_completion``:
        # (1) explicit chat_handler, (2) the per-instance
        # ``_chat_handlers`` dict (where GGUF-embedded Jinja templates
        # live under the magic name ``chat_template.default``), then
        # (3) the global static registry. Skipping (2) blows up on
        # every modern GGUF with a bundled template — Gemma, Qwen 3.5,
        # Llama 3.x — because their ``chat_format`` isn't in the
        # static registry.
        base_handler = self._llm.chat_handler
        if base_handler is None:
            base_handler = self._llm._chat_handlers.get(self._llm.chat_format)
        if base_handler is None:
            base_handler = llama_chat_format.get_chat_completion_handler(
                self._llm.chat_format
            )

        def _handler(**call_kwargs: Any):
            merged = dict(template_kwargs)
            merged.update(call_kwargs)
            return base_handler(**merged)

        self._llm.chat_handler = _handler

    def _local_process(self, text: str) -> ProcessResult:
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
                self._install_chat_template_kwargs()
            kwargs: dict[str, Any] = {
                "messages": self._build_messages(text),
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
        # display/paste split is done downstream via ``paste_strip_regex``,
        # so there's no separate reasoning field to surface here.
        return ProcessResult(
            text=resp["choices"][0]["message"]["content"].strip()
        )

    def process_with_reasoning(self, text: str) -> ProcessResult:
        """Run the LLM on *text* and return both the cleaned text and the
        backend's reasoning field (when exposed).

        Routes by ``profile.base``: ``"remote"`` POSTs to the
        OpenAI-compatible endpoint and reads structured reasoning from
        ``message.reasoning_content`` / ``message.reasoning`` if present;
        ``"builtin"`` loads a GGUF via llama-cpp-python and never returns
        a separate reasoning field (any thinking is inline in the text,
        handled via ``paste_strip_regex`` downstream).

        ``text`` falls back to the original input when the model returns
        an empty response, matching ``process()``.
        """
        if self.profile.base == "remote":
            result = self._remote_process(text)
        else:
            result = self._local_process(text)
        if not result.text:
            result = ProcessResult(text=text, reasoning=result.reasoning)
        return result

    def process(self, text: str) -> str:
        """Backward-compatible thin wrapper: returns just the cleaned text.

        Use :meth:`process_with_reasoning` to also receive the model's
        structured reasoning field (when the backend exposes one)."""
        return self.process_with_reasoning(text).text


# ---------------------------------------------------------------------------
# Model catalogue + install helpers
# ---------------------------------------------------------------------------

#: Built-in LLM choices offered by ``justsayit setup-llm``.
#: Each entry: display label + HuggingFace repo, plus an optional
#: ``profile_overrides`` dict of TOML values to bake into the seeded
#: profile (sampling knobs, temperature, …) so users don't have to
#: discover model-specific tuning on their own.
KNOWN_LLM_MODELS: dict[str, dict[str, Any]] = {
    "gemma4": {
        "display": "gemma-4-E4B-it      (4B, ~3 GB)   — Google Gemma 4, highest quality  (recommended — tuned for best results)",
        "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
    },
    "qwen3-4b": {
        "display": "Qwen3-4B-Instruct   (4B, ~3 GB)   — Alibaba Qwen3, strong multilingual",
        "hf_repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    },
    "qwen3-0.8b": {
        "display": "Qwen3.5-0.8B        (0.8B, ~600 MB) — fastest, smallest footprint",
        "hf_repo": "unsloth/Qwen3.5-0.8B-GGUF",
        # Thinking mode is OFF at 0.8B — the model reliably enters
        # thinking loops on the complex cleanup+assistant-mode prompt.
        # cleanup_qwen_simple.md is a short cleanup-only prompt that
        # the model handles well at near-greedy temperature.
        # paste_strip_regex cleared (no <think> blocks to strip).
        "profile_overrides": {
            "system_prompt_file": "cleanup_qwen_simple.md",
            "chat_template_kwargs": {},
            "paste_strip_regex": "",
            "temperature": 0.08,
        },
    },
}


def find_hf_q4_filename(hf_repo: str) -> str:
    """Query the HuggingFace API and return the Q4_K_M GGUF filename in *hf_repo*.

    Raises ``RuntimeError`` if no matching file is found or the request fails.
    """
    url = f"https://huggingface.co/api/models/{hf_repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "justsayit/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data: dict = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(
            f"Could not query HuggingFace API for {hf_repo!r}: {exc}"
        ) from exc

    matches = [
        s["rfilename"]
        for s in data.get("siblings", [])
        if "Q4_K_M" in s.get("rfilename", "") and s["rfilename"].endswith(".gguf")
    ]
    if not matches:
        all_files = [s.get("rfilename", "") for s in data.get("siblings", [])]
        raise RuntimeError(
            f"No Q4_K_M .gguf file found in {hf_repo}.\nFiles present: {all_files}"
        )
    return matches[0]


def download_llm_model(hf_repo: str, hf_filename: str) -> Path:
    """Download *hf_filename* from *hf_repo* into the llm models directory.

    Returns the local ``Path``.  Skips the download if the file already exists.
    """
    from justsayit.model import _download, models_dir

    dest = models_dir() / "llm" / hf_filename
    if dest.exists():
        log.info("LLM model already cached: %s", dest)
        return dest
    url = f"https://huggingface.co/{hf_repo}/resolve/main/{hf_filename}"
    _download(url, dest)
    return dest


def _format_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ", ".join(
            f"{k} = {_format_toml_scalar(v)}" for k, v in value.items()
        )
        return f"{{ {items} }}"
    return f'"{value}"'


def _set_toml_key(src: str, key: str, value: Any) -> str:
    # Upsert *key* = *value* in *src*, matching active (``key = …``) and
    # commented-default (``# key = …``) lines alike. Any existing
    # occurrences — whether commented or active — beyond the first are
    # DELETED. That's load-bearing: pre-0.13.6 ``update_profile_model``
    # appended a fresh override line at the bottom when the regex
    # couldn't see the commented example, so legacy profiles now
    # contain the commented example *and* an appended active line.
    # Without the dedupe, a re-run seeds a second active line and the
    # profile parses as a duplicate-key TOML error — which the tray
    # silently swallows, making the whole profile disappear.
    formatted = f"{key} = {_format_toml_scalar(value)}"
    pattern = re.compile(rf"^(?:#\s*)?{re.escape(key)}\s*=\s*.*$")
    lines = src.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if pattern.match(line):
            if not replaced:
                out.append(formatted)
                replaced = True
            # else: drop duplicate
        else:
            out.append(line)
    if not replaced:
        if out and out[-1].strip():
            out.append("")  # blank line before an appended key
        out.append(formatted)
    result = "\n".join(out)
    if src.endswith("\n"):
        result += "\n"
    return result


def update_profile_model(
    profile_path: Path, model_path: Path, hf_repo: str, hf_filename: str
) -> None:
    """Patch *profile_path* in-place to point at the downloaded model.

    Uses regex substitution so comments and all other settings are preserved.
    """
    text = profile_path.read_text(encoding="utf-8")
    text = _set_toml_key(text, "model_path", str(model_path))
    text = _set_toml_key(text, "hf_repo", hf_repo)
    text = _set_toml_key(text, "hf_filename", hf_filename)
    profile_path.write_text(text, encoding="utf-8")


def apply_profile_overrides(profile_path: Path, overrides: dict[str, Any]) -> None:
    """Write model-specific tuning into *profile_path*.

    Each key/value in *overrides* replaces the matching (possibly
    commented-out) line in the seeded template. Used by ``setup-llm``
    to bake Qwen-recommended sampling knobs into a fresh Qwen 3.5 0.8B
    profile so users aren't dropped into looping-prone defaults.
    """
    if not overrides:
        return
    text = profile_path.read_text(encoding="utf-8")
    for key, value in overrides.items():
        text = _set_toml_key(text, key, value)
    profile_path.write_text(text, encoding="utf-8")
