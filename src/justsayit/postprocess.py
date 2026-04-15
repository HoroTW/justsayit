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
import threading
import tomllib
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from justsayit.config import config_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile dataclass + loader
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_DE = (
    "Bereinige das folgende Transkript: entferne Füllwörter (ähm, öhm, halt, also),"
    " korrigiere Grammatik und Rechtschreibung sowie falsch verstandene Wörter,"
    " behalte den Sinn und den Sprachstil bei."
    " Gib NUR den bereinigten Text zurück, keine Erklärungen."
)

# Written to disk on first ``justsayit init`` so the user can inspect and
# customise it without having to know the TOML schema.
_DEFAULT_PROFILE_TOML = f"""\
# justsayit postprocessing profile — gemma-cleanup
#
# Enable this profile in config.toml:
#   [postprocess]
#   enabled = true
#   profile = "gemma-cleanup"
#
# Then install the inference backend (with Vulkan GPU support):
#   CMAKE_ARGS="-DGGML_VULKAN=1" uv pip install llama-cpp-python
#
# And download a GGUF model.  Example (adjust to your preferred quant):
#   wget -P ~/.cache/justsayit/models/llm/ \\
#     https://huggingface.co/<repo>/resolve/main/<model>.gguf

# Path to the GGUF model file.  ~ is expanded.
model_path = "~/.cache/justsayit/models/llm/gemma-4-e4b-it-Q4_K_M.gguf"

# Optional: HuggingFace repo + filename for auto-download via
#   justsayit download-models
# Set both to enable; leave empty to manage the file yourself.
hf_repo = ""
hf_filename = ""

# GPU layer offloading.  -1 = all layers on GPU (fastest).  0 = CPU only.
n_gpu_layers = -1

# Context window size in tokens.  2048 is plenty for short transcriptions.
n_ctx = 2048

# Temperature.  Keep very low (≤ 0.1) for deterministic cleanup.
temperature = 0.08

# Hard cap on generated tokens.
max_tokens = 512

# System prompt.  Edit freely — the model reads this before every request.
system_prompt = "{_DEFAULT_SYSTEM_DE}"

# User message template.  {{text}} is replaced with the raw transcription.
user_template = "{{text}}"
"""


@dataclass
class PostprocessProfile:
    model_path: str = "~/.cache/justsayit/models/llm/gemma-4-e4b-it-Q4_K_M.gguf"
    hf_repo: str = ""
    hf_filename: str = ""
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    temperature: float = 0.08
    max_tokens: int = 512
    system_prompt: str = _DEFAULT_SYSTEM_DE
    user_template: str = "{text}"


def profiles_dir() -> Path:
    return config_dir() / "postprocess"


def ensure_default_profile(path: Path | None = None) -> Path:
    """Write the default ``gemma-cleanup.toml`` profile if it doesn't exist yet."""
    if path is None:
        path = profiles_dir() / "gemma-cleanup.toml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_PROFILE_TOML, encoding="utf-8")
    return path


def load_profile(name_or_path: str) -> PostprocessProfile:
    """Load a :class:`PostprocessProfile` from *name_or_path*.

    If the argument looks like a file path (contains a separator or ends
    with ``.toml``) it is used directly; otherwise it is resolved to
    ``config_dir()/postprocess/<name>.toml``.
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
    valid = {fld.name for fld in fields(PostprocessProfile)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return PostprocessProfile(**kwargs)


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

    def __init__(self, profile: PostprocessProfile) -> None:
        self.profile = profile
        self._llm = None
        self._lock = threading.Lock()

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
        """Eagerly load the model so the first transcription is not slow."""
        with self._lock:
            if self._llm is None:
                self._llm = self._build()

    def process(self, text: str) -> str:
        """Run the LLM on *text* and return the cleaned result.

        Returns the original *text* unchanged if the model produces an
        empty response.
        """
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
            user_msg = self.profile.user_template.format(text=text)
            resp = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.profile.system_prompt.strip()},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.profile.temperature,
                max_tokens=self.profile.max_tokens,
            )
        result: str = resp["choices"][0]["message"]["content"].strip()
        return result if result else text


# ---------------------------------------------------------------------------
# Model catalogue + install helpers
# ---------------------------------------------------------------------------

#: Built-in LLM choices offered by ``justsayit setup-llm``.
#: Each entry maps a short key → display label + HuggingFace repo.
KNOWN_LLM_MODELS: dict[str, dict[str, str]] = {
    "gemma4": {
        "display": "gemma-4-E4B-it      (4B, ~3 GB)   — Google Gemma 4, highest quality",
        "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
    },
    "qwen3-4b": {
        "display": "Qwen3-4B-Instruct   (4B, ~3 GB)   — Alibaba Qwen3, strong multilingual",
        "hf_repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    },
    "qwen3-0.8b": {
        "display": "Qwen3.5-0.8B        (0.8B, ~600 MB) — fastest, smallest footprint",
        "hf_repo": "unsloth/Qwen3.5-0.8B-GGUF",
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
        raise RuntimeError(f"Could not query HuggingFace API for {hf_repo!r}: {exc}") from exc

    matches = [
        s["rfilename"]
        for s in data.get("siblings", [])
        if "Q4_K_M" in s.get("rfilename", "") and s["rfilename"].endswith(".gguf")
    ]
    if not matches:
        all_files = [s.get("rfilename", "") for s in data.get("siblings", [])]
        raise RuntimeError(
            f"No Q4_K_M .gguf file found in {hf_repo}.\n"
            f"Files present: {all_files}"
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


def update_profile_model(
    profile_path: Path, model_path: Path, hf_repo: str, hf_filename: str
) -> None:
    """Patch *profile_path* in-place to point at the downloaded model.

    Uses regex substitution so comments and all other settings are preserved.
    """
    text = profile_path.read_text(encoding="utf-8")

    def _set(src: str, key: str, value: str) -> str:
        result, n = re.subn(
            rf"^{re.escape(key)}\s*=\s*.*$",
            f'{key} = "{value}"',
            src,
            flags=re.MULTILINE,
        )
        if n == 0:
            result = src.rstrip() + f'\n{key} = "{value}"\n'
        return result

    text = _set(text, "model_path", str(model_path))
    text = _set(text, "hf_repo", hf_repo)
    text = _set(text, "hf_filename", hf_filename)
    profile_path.write_text(text, encoding="utf-8")
