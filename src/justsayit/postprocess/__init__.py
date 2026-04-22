"""LLM-based post-processing of raw transcription output.

Each post-processor is driven by a *profile* — a small TOML file in
``$XDG_CONFIG_HOME/justsayit/postprocess/<name>.toml``.

Public API re-exported from the sub-modules:

- Profile management: :func:`load_profile`, :func:`profiles_dir`,
  :func:`ensure_default_profile`, :func:`ensure_default_profiles`, …
- Data types: :class:`PostprocessProfile`, :class:`ProcessResult`
- Processor: :class:`LLMPostprocessor`
- Model catalogue: :data:`KNOWN_LLM_MODELS`, :func:`find_hf_q4_filename`,
  :func:`download_llm_model`, :func:`update_profile_model`,
  :func:`apply_profile_overrides`
"""

from ._models import (
    KNOWN_LLM_MODELS,
    apply_profile_overrides,
    download_llm_model,
    find_hf_q4_filename,
    update_profile_model,
)
from ._processor import LLMPostprocessor
from ._profile import (
    PostprocessProfile,
    ProcessResult,
    _CLEANUP_PROFILE_TOML,
    _CONTEXT_SIDECAR_TEMPLATE,
    _DYNAMIC_CONTEXT_SCRIPT,
    _FUN_PROFILE_TOML,
    _OLLAMA_GEMMA_PROFILE_TOML,
    _OPENAI_PROFILE_TOML,
    _PROFILE_COMMENTED_FORM_MARKER,
    _load_prompt,
    _resolve_system_prompt_file,
    context_file_path,
    dynamic_context_script_path,
    ensure_context_file,
    ensure_default_profile,
    ensure_default_profiles,
    ensure_dynamic_context_script,
    ensure_fun_profile,
    ensure_ollama_gemma_profile,
    ensure_openai_profile,
    load_context_sidecar,
    load_profile,
    profiles_dir,
)

__all__ = [
    # Profile management
    "PostprocessProfile",
    "ProcessResult",
    "load_profile",
    "profiles_dir",
    "context_file_path",
    "dynamic_context_script_path",
    "ensure_context_file",
    "ensure_default_profile",
    "ensure_default_profiles",
    "ensure_dynamic_context_script",
    "ensure_fun_profile",
    "ensure_ollama_gemma_profile",
    "ensure_openai_profile",
    "load_context_sidecar",
    # Processor
    "LLMPostprocessor",
    # Model catalogue
    "KNOWN_LLM_MODELS",
    "apply_profile_overrides",
    "download_llm_model",
    "find_hf_q4_filename",
    "update_profile_model",
    # Private but exported for legacy imports in tests / cli
    "_CLEANUP_PROFILE_TOML",
    "_CONTEXT_SIDECAR_TEMPLATE",
    "_DYNAMIC_CONTEXT_SCRIPT",
    "_FUN_PROFILE_TOML",
    "_OLLAMA_GEMMA_PROFILE_TOML",
    "_OPENAI_PROFILE_TOML",
    "_PROFILE_COMMENTED_FORM_MARKER",
    "_load_prompt",
    "_resolve_system_prompt_file",
]
