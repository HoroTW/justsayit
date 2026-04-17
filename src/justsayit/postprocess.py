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

_DEFAULT_SYSTEM_PROMPT = """\
You are `Computer`, a voice-transcript (STT) cleaner and assistant.
<|think|> INTERNAL reasoning ONLY — at most ONE short sentence (≤ 15 words). NEVER echo the input, list filler/mishear/formatting checks, enumerate corrections, or show step-by-step work. If nothing needs changing, just write `No changes.` and stop.

# Default mode — CONSERVATIVE CLEANUP
You are NOT a copy editor. Output the transcript verbatim except for these specific edits:
- remove obvious filler words: `ähm`, `öhm`, `halt`, `also`, `um`, `uh`, `like`, `so`
- fix words the STT clearly misheard
- replace spoken punctuation / line-break words with the actual character (see below)
- apply formatting only when explicitly dictated

DO NOT:
- rephrase, restructure, or reorder words
- "improve" valid colloquial grammar (especially German modal particles like `denn`, `doch`, `mal`, `ja`, `eben`, `schon` — keep them as-is, they carry meaning)
- change `?` ↔ `.` or drop punctuation that wasn't a spoken word
- normalise mixed German + English — keep the mix
- translate (unless `Computer` mode, see below)

When in doubt: leave it exactly as the user said it.

# Spoken punctuation / line-break words
These dictated words become the actual character. CRITICAL: if the STT already produced the corresponding character (or inserting it would leave a stray symbol on its own line), DROP the spoken word silently.
- `Punkt` / `period`               -> `.`
- `Komma` / `comma`                -> `,`
- `Fragezeichen` / `question mark` -> `?`
- `Ausrufezeichen` / `exclamation mark` -> `!`
- `Doppelpunkt` / `colon`          -> `:`
- `Semikolon` / `semicolon`        -> `;`
- `neue Zeile` / `new line`        -> a real newline
- `neuer Absatz` / `new paragraph` -> a blank line

Examples:
- `Hallo, neue Zeile. Ich komme nicht. Punkt. Neue Zeile, euer Pete.` ->
  `Hallo,
Ich komme nicht.
euer Pete`
  (STT already wrote `.` after `nicht`; the spoken `Punkt` is redundant — drop it. NEVER leave a stray `.` on its own line.)
- `Hello comma new line greetings` -> `Hello,
greetings`
- `... new line dash some point new line dash another point` -> `...
 - some point
 - another point`
- `laughing emoji` -> `🤣`
- code-y words in backticks: 'The cat command is helpful.' -> 'The `cat` command is helpful.'

# Examples of what NOT to change
- `Ich weiß nicht, was denkst du denn?` -> `Ich weiß nicht, was denkst du denn?`  (valid German; `denn` is a modal particle, keep it; do NOT restructure to "was du denkst")
- `I don't know, what do you think?` -> `I don't know, what do you think?`  (already clean)
- `Das war halt so` — `halt` is slang (colloquial language) here -> `Das war halt so`

# Assistant mode — ONLY when explicitly addressed
Switch to assistant mode ONLY IF the literal word `Computer` appears in the transcript (anywhere — start, middle, end). Without `Computer`, the transcript is dictated content for some other app (chat, editor, email, …), NEVER for you. This holds EVEN IF the text is phrased as a question, a request, or an instruction. No exceptions, no "but it sounded like a request".

Examples:
- `Can you tell me how many things you can see?`               -> CLEANUP only (no `Computer`)
- `Ich weiß nicht, was denkst du denn?`                        -> CLEANUP only (no `Computer`)
- `Translate this to German: hello world`                       -> CLEANUP only (no `Computer`)
- `Hey Computer, can you tell me how many things you can see?` -> ANSWER (explicit `Computer`)
- `Computer, translate this to German: hello world`             -> ACT on the request
- `… and then I told him, hey computer remind me tomorrow.`     -> ACT (anywhere counts, lowercase counts)

When addressed:
- follow the request directly; do NOT echo the source first
- if asked to translate, output ONLY the translation
- short, on-point reply — no preamble like "Sure, here you go:"

# Output
Return ONLY the cleaned text (default) OR the assistant reply (assistant mode). No meta explanations.
"""


# Minimal "fun" profile prompt — written to disk as gemma4-fun.toml so users
# can flip to a playful, emoji-heavy variant without having to compose a
# prompt themselves.  Intentionally tiny: the recommended everyday default
# is gemma4-cleanup, this one is the silly sibling.
_FUN_SYSTEM_PROMPT = """\
Emojify the transcript as much as possible. Keep the original wording and order, just sprinkle in plenty of fitting emojis (between words, at the end of sentences, wherever they fit). Reply with the emojified text only — no explanations, no preamble.
"""


# Written to disk on first ``justsayit init`` so the user can inspect and
# customise it without having to know the TOML schema.  Embed the system
# prompt in a TOML triple-quoted basic string ("""…""") so newlines stay
# real and the file is readable; this requires the Python literal itself
# to use single-quoted '''…''' delimiters so the embedded """ aren't
# parsed as the closing delimiter.
_CLEANUP_PROFILE_TOML = f'''\
# justsayit postprocessing profile — gemma4-cleanup (recommended default)
#
# Enable this profile in config.toml:
#   [postprocess]
#   enabled = true
#   profile = "gemma4-cleanup"
#
# Then install the inference backend (with Vulkan GPU support):
#   CMAKE_ARGS="-DGGML_VULKAN=1" uv pip install llama-cpp-python
#
# And download a GGUF model.  Example (adjust to your preferred quant):
#   wget -P ~/.cache/justsayit/models/llm/ \\
#     https://huggingface.co/<repo>/resolve/main/<model>.gguf

# Path to the GGUF model file.  ~ is expanded.
model_path = "~/.cache/justsayit/models/llm/gemma-4-E4B-it-Q4_K_M.gguf"

# Optional: HuggingFace repo + filename for auto-download via
#   justsayit download-models
# Set both to enable; leave empty to manage the file yourself.
hf_repo = "unsloth/gemma-4-E4B-it-GGUF"
hf_filename = "gemma-4-E4B-it-Q4_K_M.gguf"

# GPU layer offloading.  -1 = all layers on GPU (fastest).  0 = CPU only.
n_gpu_layers = -1

# Context window size in tokens.
n_ctx = 4096

# Temperature.  Keep very low (≤ 0.1) for deterministic cleanup.
temperature = 0.08

# Hard cap on generated tokens.
max_tokens = 4096

# System prompt.  Edit freely — the model reads this before every request.
#
# The default prompt enables Gemma's "thinking" channel (the `<|think|>`
# token makes the model emit a `<|channel>...<channel|>` reasoning block
# before its real reply).  This usually improves quality on ambiguous
# "Hey Computer" requests but adds latency and produces extra text that
# has to be stripped before pasting (see `paste_strip_regex` below).
#
# To disable thinking entirely: remove the `<|think|>` marker from the
# prompt.  The model will then reply directly with the cleaned text and
# you can also clear `paste_strip_regex` since there is no channel block
# to strip.
system_prompt = """
{_DEFAULT_SYSTEM_PROMPT}"""

# User message template.  {{text}} is replaced with the raw transcription.
user_template = "{{text}}"

# Optional regex (re.DOTALL) applied to the LLM output before it is pasted
# but NOT before it is shown in the overlay. Useful for "thinking" models
# whose output contains a reasoning preamble that should not land in the
# focused window. Empty = no stripping.
#
# The default below matches Gemma's channel block — note the tags are
# ASYMMETRIC: the opening is `<|channel>` (one pipe, before `channel`)
# and the closing is `<channel|>` (one pipe, after).  Don't add a second
# pipe to the opening — the model never emits `<|channel|>`.
#
# Wrap the parts you want SHOWN in the overlay in a capture group `(…)` —
# the whole match is still stripped from the paste, but only the captured
# content is displayed as the "thought". Without a capture group, the
# entire match (including the framing tokens) is shown.
#
# Examples:
#   # Gemma thinking channel — drops the literal `thought` label too:
#   paste_strip_regex = '<\\|channel>thought(.*?)<channel\\|>'
#   # Generic <think>…</think> — show only inner content:
#   paste_strip_regex = '<think>(.*?)</think>'
#   # Strip everything before the final answer — nothing to show:
#   paste_strip_regex = '(?s)^.*?</think>'
paste_strip_regex = '<\\|channel>thought(.*?)<channel\\|>'

# Optional free-form context about the user — appended to the system prompt
# under a "User context" heading, so the model can correctly spell your name,
# pick the right register, etc. Use a TOML multi-line string ("""…""").
# Leave empty to send no context. Example:
#
#   context = """
#   Name: Jane Doe
#   Country: Germany
#   Languages: German (native), English (fluent)
#   Notes: works in software, often dictates code-related text
#   """
context = ""
'''


# Companion "fun" profile — a tiny stub that emojifies the transcript.
# Written alongside gemma4-cleanup so users can flip via the `profile`
# config field without composing a prompt themselves. The header
# explicitly points back at the recommended cleanup profile so anyone
# who lands here by accident knows where to go for serious dictation.
_FUN_PROFILE_TOML = f'''\
# justsayit postprocessing profile — gemma4-fun
#
# Playful sibling of `gemma4-cleanup`: keeps the wording but sprinkles
# in plenty of emojis. Use it when you want a chatty, expressive tone
# in messages or social posts.
#
# For everyday cleanup, switch back to the recommended default:
#   profile = "gemma4-cleanup"
#
# Activate this one in config.toml:
#   [postprocess]
#   enabled = true
#   profile = "gemma4-fun"

# Same model file as gemma4-cleanup — if you ran `setup-llm` once you
# already have it on disk and no extra download happens.
model_path = "~/.cache/justsayit/models/llm/gemma-4-E4B-it-Q4_K_M.gguf"
hf_repo = "unsloth/gemma-4-E4B-it-GGUF"
hf_filename = "gemma-4-E4B-it-Q4_K_M.gguf"

n_gpu_layers = -1
n_ctx = 4096

# Slightly higher temperature so emoji choice has some variety.
temperature = 0.4
max_tokens = 4096

system_prompt = """
{_FUN_SYSTEM_PROMPT}"""

user_template = "{{text}}"

# No `<|think|>` in the prompt → no channel block to strip.
paste_strip_regex = ""

context = ""
'''


@dataclass
class PostprocessProfile:
    model_path: str = "~/.cache/justsayit/models/llm/gemma-4-E4B-it-Q4_K_M.gguf"
    hf_repo: str = "unsloth/gemma-4-E4B-it-GGUF"
    hf_filename: str = "gemma-4-E4B-it-Q4_K_M.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    temperature: float = 0.08
    max_tokens: int = 4096
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    user_template: str = "{text}"
    # Regex applied (re.DOTALL) to the LLM output before it is pasted
    # but NOT before it is shown in the overlay. Useful to strip the
    # reasoning preamble produced by "thinking" models (e.g. Gemma's
    # asymmetric `<|channel>...<channel|>` block) so the user sees the
    # full reply in the overlay but only the final message lands in the
    # focused window.
    #
    # If the pattern includes a capture group, only the captured content
    # is shown as the "thought" in the overlay (the full match — tags
    # included — is still stripped from paste). Without a group, the
    # whole match is shown.
    #
    # Default matches Gemma 4 with the `<|think|>` markers in the prompt
    # and captures the inner content so the framing tags AND the literal
    # `thought` channel label don't appear in the overlay; set to "" if
    # you remove `<|think|>` from system_prompt.
    paste_strip_regex: str = r"<\|channel>thought(.*?)<channel\|>"
    # Free-form text appended to the system prompt under a "User context"
    # heading so the model knows who's dictating (name, language, country,
    # technical interests, etc.). Empty by default; users can fill in via
    # a multi-line TOML string. Only sent if non-empty.
    context: str = ""


def profiles_dir() -> Path:
    return config_dir() / "postprocess"


def ensure_default_profile(path: Path | None = None) -> Path:
    """Write the recommended ``gemma4-cleanup.toml`` profile if it's missing.

    Writes the cleanup-style template (the conservative everyday default).
    Used both by ``justsayit init`` and by ``setup-llm`` when seeding a
    per-model profile.
    """
    if path is None:
        path = profiles_dir() / "gemma4-cleanup.toml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_CLEANUP_PROFILE_TOML, encoding="utf-8")
    return path


def ensure_fun_profile(path: Path | None = None) -> Path:
    """Write the ``gemma4-fun.toml`` companion profile if it's missing.

    A tiny emoji-heavy variant of the cleanup profile, written alongside
    it on first ``init`` so users discover the schema and have an obvious
    second profile to switch to via the ``profile`` config field.
    """
    if path is None:
        path = profiles_dir() / "gemma4-fun.toml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_FUN_PROFILE_TOML, encoding="utf-8")
    return path


def ensure_default_profiles() -> tuple[Path, Path]:
    """Write both the cleanup and fun default profiles. Returns (cleanup, fun)."""
    return ensure_default_profile(), ensure_fun_profile()


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
        self._paste_strip = self._compile_paste_strip(profile.paste_strip_regex)

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
        """Eagerly load the model so the first transcription is not slow."""
        with self._lock:
            if self._llm is None:
                self._llm = self._build()

    def _system_prompt(self) -> str:
        prompt = self.profile.system_prompt.strip()
        ctx = self.profile.context.strip()
        if ctx:
            prompt = f"{prompt}\n\n# User context\n{ctx}"
        return prompt

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
                    {"role": "system", "content": self._system_prompt()},
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
