"""Model catalogue, HuggingFace helpers, and TOML profile patching utilities."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import tomlkit

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


def update_profile_model(
    profile_path: Path, model_path: Path, hf_repo: str, hf_filename: str
) -> None:
    """Patch *profile_path* in-place to point at the downloaded model."""
    doc = tomlkit.parse(profile_path.read_text(encoding="utf-8"))
    doc["model_path"] = str(model_path)
    doc["hf_repo"] = hf_repo
    doc["hf_filename"] = hf_filename
    profile_path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def apply_profile_overrides(profile_path: Path, overrides: dict[str, Any]) -> None:
    """Write model-specific tuning into *profile_path*."""
    if not overrides:
        return
    doc = tomlkit.parse(profile_path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        doc[key] = value
    profile_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
