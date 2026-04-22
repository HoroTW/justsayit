"""Model catalogue, HuggingFace helpers, and TOML profile patching utilities."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Built-in LLM choices offered by ``justsayit setup-llm``.
#: Each entry: display label + HuggingFace repo, plus an optional
#: ``profile_overrides`` dict of TOML values to bake into the seeded profile.
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
        # Thinking mode is OFF at 0.8B — loops badly on the complex prompt.
        # cleanup_qwen_simple.md + near-greedy temperature is what works.
        "profile_overrides": {
            "system_prompt_file": "cleanup_qwen_simple.md",
            "chat_template_kwargs": {},
            "paste_strip_regex": "",
            "temperature": 0.08,
        },
    },
}


def find_hf_q4_filename(hf_repo: str) -> str:
    """Query the HuggingFace API and return the Q4_K_M GGUF filename."""
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
    """Download *hf_filename* from *hf_repo* into the llm models directory."""
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
    # DELETED. This deduplication heals legacy profiles where pre-0.13.6
    # ``update_profile_model`` appended a fresh override line at the bottom
    # when the regex couldn't see the commented example, leaving both a
    # commented line and an appended active line — a TOML duplicate-key error.
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
            out.append("")
        out.append(formatted)
    result = "\n".join(out)
    if src.endswith("\n"):
        result += "\n"
    return result


def update_profile_model(
    profile_path: Path, model_path: Path, hf_repo: str, hf_filename: str
) -> None:
    """Patch *profile_path* in-place to point at the downloaded model."""
    text = profile_path.read_text(encoding="utf-8")
    text = _set_toml_key(text, "model_path", str(model_path))
    text = _set_toml_key(text, "hf_repo", hf_repo)
    text = _set_toml_key(text, "hf_filename", hf_filename)
    profile_path.write_text(text, encoding="utf-8")


def apply_profile_overrides(profile_path: Path, overrides: dict[str, Any]) -> None:
    """Write model-specific tuning into *profile_path*."""
    if not overrides:
        return
    text = profile_path.read_text(encoding="utf-8")
    for key, value in overrides.items():
        text = _set_toml_key(text, key, value)
    profile_path.write_text(text, encoding="utf-8")
