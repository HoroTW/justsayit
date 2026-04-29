"""Profile dataclass, loader, and all profile-management helpers."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from justsayit.config import config_dir, resolve_secret  # noqa: F401

log = logging.getLogger(__name__)

# Shipped prompt + config templates live as plain text files alongside
# the package so they can be edited in a content-aware editor without
# Python-string escaping.
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# Canonical defaults for each inference backend. These TOML files are
# the single source of truth: the dataclass defaults are derived from
# them at module-import time, AND the user-facing profile templates
# document them by reference.
_BASE_DEFAULTS: dict[str, dict[str, Any]] = {
    "builtin": tomllib.loads(_load_template("builtin-defaults.toml")),
    "remote": tomllib.loads(_load_template("remote-defaults.toml")),
    "responses": tomllib.loads(_load_template("responses-defaults.toml")),
}


def _builtin_default(name: str, fallback: Any) -> Any:
    return _BASE_DEFAULTS["builtin"].get(name, fallback)


_PROFILE_COMMENTED_FORM_MARKER = (
    "# justsayit postprocess profile (commented-defaults form)."
)


def _load_profile_template(name: str) -> str:
    return _load_template(name)


_CLEANUP_PROFILE_TOML = _load_profile_template("profile-gemma4-cleanup.toml")
_FUN_PROFILE_TOML = _load_profile_template("profile-gemma4-fun.toml")
_OPENAI_PROFILE_TOML = _load_profile_template("profile-openai-cleanup.toml")
_RESPONSES_PROFILE_TOML = _load_profile_template("profile-openai-responses.toml")
_OLLAMA_GEMMA_PROFILE_TOML = _load_profile_template("profile-ollama-gemma.toml")


@dataclass(frozen=True)
class ProcessResult:
    """Return value of :meth:`LLMPostprocessor.process_with_reasoning`.

    ``text`` is the model's visible reply. ``reasoning`` is the model's
    hidden thinking, when the backend exposes it as a structured field
    (currently only the remote backend). Local llama-cpp-python output
    keeps reasoning inline in ``content`` (handled via
    ``paste_strip_regex``), so for the local path ``reasoning`` is always
    ``""``.
    ``session_data`` is populated by backends when they successfully
    process a request; ``pipeline.py`` writes it to session.json.
    """

    text: str
    reasoning: str = ""
    session_data: dict | None = None


@dataclass
class PostprocessProfile:
    # Which backend defaults file to overlay user values onto.
    base: str = _builtin_default("base", "builtin")

    # --- Inference backend (built-in via llama-cpp-python + GGUF) -------
    model_path: str = _builtin_default("model_path", "")
    hf_repo: str = _builtin_default("hf_repo", "")
    hf_filename: str = _builtin_default("hf_filename", "")
    n_gpu_layers: int = _builtin_default("n_gpu_layers", -1)
    n_ctx: int = _builtin_default("n_ctx", 20480)

    # --- Cleanup tuning -------------------------------------------------
    temperature: float = _builtin_default("temperature", 0.08)
    max_tokens: int = _builtin_default("max_tokens", 4096)
    # Sampling knobs — defaults match llama-cpp-python's
    # ``create_chat_completion`` defaults. Raise ``presence_penalty``
    # (e.g. 1.5) to break loops on small models.
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
    system_prompt_file: str = _builtin_default(
        "system_prompt_file", "cleanup_gemma.md"
    )
    system_prompt: str = _builtin_default("system_prompt", "")
    append_to_system_prompt: str = _builtin_default("append_to_system_prompt", "")

    # Passthrough dict forwarded to the chat template. On llama-cpp-python
    # it reaches the Jinja renderer via ``chat_template_kwargs=``; on the
    # remote OpenAI-compatible path it's included in the JSON body. Empty
    # → not forwarded (keeps requests clean for providers that reject it).
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(_builtin_default("chat_template_kwargs", {}))
    )

    # --- User context ---------------------------------------------------
    context: str = _builtin_default("context", "")

    # --- HTTP / OpenAI-compatible backend (base = "remote") -------------
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout: float = 60.0
    remote_retries: int = 3
    remote_retry_delay_seconds: float = 1.0
    # "low" | "medium" | "high" | "" (= don't send the field).
    reasoning_effort: str = ""
    # Token pricing (per 1 million tokens). 0.0 = don't log cost.
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0
    cached_input_price_per_1m: float = 0.0

    # --- OpenAI Responses API backend (base = "responses") --------------
    # "24h" = keep cached system-prompt prefix alive for 24 hours (free).
    prompt_cache_retention: str = "24h"
    responses_web_search: bool = False
    # If non-empty, web_search is only added when the raw transcription
    # matches this regex (re.search). Keeps the ~4k token tool-schema
    # out of plain cleanup calls.
    responses_web_search_trigger: str = ""
    # All models: $0.010/call flat fee per `search` action.
    web_search_price_per_call: float = 0.0
    # Flat fee per `open_page` action (URL fetch). Billed separately from
    # input tokens — not included in the LLM usage field.
    web_open_page_price_per_call: float = 0.0
    # Image detail level when an image is provided (e.g. from the clipboard).
    # "off" = never send images. "auto" = model decides low vs. high (default).
    # "low" | "high" = force detail tier. "original" = full resolution (5.4+).
    image_detail: str = "auto"

    # --- Tools (function calling) -------------------------------------------
    # When True (default) and tools are defined in tools.json, the LLM is
    # given the tool definitions and may call them. Set to false in a profile
    # to suppress tool use even when tools are loaded.
    use_tools: bool = True

    def __post_init__(self) -> None:
        # Auto-infer remote backend when endpoint is set and base wasn't
        # explicitly bumped off the default.
        if self.base == "builtin" and self.endpoint:
            self.base = "remote"


# Old prompt names → new prompt names. Profiles seeded by older versions
# reference these; we transparently redirect with a warning.
_PROMPT_LEGACY_ALIASES = {
    "cleanup_local.md": "cleanup_gemma.md",
    "cleanup_remote.md": "cleanup_openai.md",
}


def _resolve_system_prompt_file(value: str) -> str:
    """Load a prompt file from *value* (path or bare name).

    Bare name → packaged ``prompts/`` dir. Otherwise expand and load."""
    if value in _PROMPT_LEGACY_ALIASES:
        new_name = _PROMPT_LEGACY_ALIASES[value]
        log.warning(
            "system_prompt_file %r was renamed to %r — using the new file. "
            "Update your profile to silence this warning.",
            value,
            new_name,
        )
        value = new_name
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
# so updates to shipped profile TOMLs don't clobber anything the user wrote.
_CONTEXT_SIDECAR_TEMPLATE = _load_template("context-sidecar.toml")
_DYNAMIC_CONTEXT_SCRIPT = _load_template("dynamic-context.sh")


def context_file_path() -> Path:
    return config_dir() / "context.toml"


def dynamic_context_script_path() -> Path:
    return config_dir() / "dynamic-context.sh"


def ensure_context_file(path: Path | None = None) -> Path:
    """Write the personal-context sidecar if it doesn't exist."""
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
    """Return the ``context`` string from ``context.toml`` (or "" if missing)."""
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


def _has_uncommented_assignment(text: str) -> bool:
    """True if *text* contains at least one ``key = value`` line that
    isn't commented out — the heuristic for "still on legacy fully-
    populated form"."""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        if "=" in stripped:
            return True
    return False


def _ensure_commented_form_file(
    path: Path,
    commented: str,
    marker: str,
    *,
    suffix: str = ".bak-pre-commented-form",
    validator: Callable[[str], None] | None = None,
) -> bool:
    """Ensure *path* exists in commented-defaults form, migrating from
    legacy fully-populated form once if necessary.

    The marker is a stable header line embedded in *commented*; finding
    it in the user file means migration already happened, so we leave
    the file alone (the user may have uncommented overrides). For files
    that lack the marker AND contain uncommented ``key = value`` lines
    (legacy form), the existing file is backed up to
    ``<path><suffix>`` (if no backup exists yet) and overwritten with
    *commented*. Pure-comment / empty files get the commented template
    written without backup.

    If *validator* is given, it is called on the existing file content
    even when the marker is present; raising any exception means "this
    file is corrupt despite the marker" and triggers re-migration. This
    rescues files written by an earlier buggy template that happened to
    embed the marker.

    Returns ``True`` if the file was just written / migrated, ``False``
    if it was found already in commented form.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commented, encoding="utf-8")
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    was_marked_but_corrupt = False
    if marker in head[:8192]:
        if validator is None:
            return False
        try:
            validator(head)
            return False
        except Exception:
            was_marked_but_corrupt = True
    if was_marked_but_corrupt or _has_uncommented_assignment(head):
        backup = path.with_name(path.name + suffix)
        if not backup.exists():
            try:
                backup.write_bytes(path.read_bytes())
            except OSError:
                pass
    try:
        path.write_text(commented, encoding="utf-8")
    except OSError:
        pass
    return True


def ensure_profile(content: str, path: Path) -> Path:
    """Write a profile in commented-defaults form to *path* if it is missing or stale."""
    _ensure_commented_form_file(
        path, content, _PROFILE_COMMENTED_FORM_MARKER, validator=_toml_validator
    )
    return path


def ensure_default_profiles() -> tuple[Path, Path, Path, Path, Path]:
    """Write the cleanup, fun, openai, responses, and ollama-gemma default profiles."""
    ensure_context_file()
    ensure_dynamic_context_script()
    pd = profiles_dir()
    return (
        ensure_profile(_CLEANUP_PROFILE_TOML, pd / "gemma4-cleanup.toml"),
        ensure_profile(_FUN_PROFILE_TOML, pd / "gemma4-fun.toml"),
        ensure_profile(_OPENAI_PROFILE_TOML, pd / "openai-cleanup.toml"),
        ensure_profile(_RESPONSES_PROFILE_TOML, pd / "openai-responses.toml"),
        ensure_profile(_OLLAMA_GEMMA_PROFILE_TOML, pd / "ollama-gemma.toml"),
    )


def load_profile(name_or_path: str) -> PostprocessProfile:
    """Load a :class:`PostprocessProfile` from *name_or_path*.

    If the argument looks like a file path (contains a separator or ends
    with ``.toml``) it is used directly; otherwise it is resolved to
    ``config_dir()/postprocess/<name>.toml``.

    Resolution order: ``<base>-defaults.toml`` (where *base* comes from
    the user file's ``base`` field, defaulting to ``"builtin"``) is the
    starting point; the user file's keys are then overlaid on top.
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
    if not profile.context.strip():
        profile.context = load_context_sidecar()
    return profile
