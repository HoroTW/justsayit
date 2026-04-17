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
import urllib.error
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from justsayit.config import config_dir, ensure_commented_form_file, resolve_secret

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

KEEP every newline and blank line from the input exactly where it is — line breaks and paragraph spacing are part of the user's intended structure and must round-trip 1:1 into the output.

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
Switch to assistant mode ONLY IF the transcript STARTS with `Hey Computer` (case-insensitive, and tolerate obvious STT mishears like `Hi Computer`, `Hey Computa`). Anything else — including a bare `Computer`, a mid-sentence `hey computer`, or a quoted/reported `hey computer` — is CLEANUP only. Without a leading `Hey Computer`, the transcript is dictated content for some other app (chat, editor, email, …), NEVER for you. This holds EVEN IF the text is phrased as a question, a request, or an instruction. No exceptions, no "but it sounded like a request".

Examples:
- `Can you tell me how many things you can see?`               -> CLEANUP only (no trigger)
- `Ich weiß nicht, was denkst du denn?`                        -> CLEANUP only (no trigger)
- `Translate this to German: hello world`                       -> CLEANUP only (no trigger)
- `Computer, translate this to German: hello world`             -> CLEANUP only (bare `Computer` is NOT the trigger)
- `… and then I told him, hey computer remind me tomorrow.`     -> CLEANUP only (mid-sentence / quoted, not a leading address)
- `Hey Computer, can you tell me how many things you can see?` -> ANSWER (leading `Hey Computer`)
- `hey computer translate this to German: hello world`          -> ACT (case-insensitive leading trigger)

When addressed:
- follow the request directly; do NOT echo the source first
- if asked to translate, output ONLY the translation
- short, on-point reply — no preamble like "Sure, here you go:"

# Output
Return ONLY the cleaned text (default) OR the assistant reply (assistant mode). No meta explanations.
"""


# Variant of the cleanup prompt for OpenAI-compatible endpoints.  The
# default prompt above relies on Gemma's `<|think|>` channel to hide
# reasoning from the final reply; a generic LLM has no such channel and
# happily echoes the literal `No changes.` instruction or dumps reasoning
# into its visible output.  This version drops the channel directives and
# tells the model to silently emit the input verbatim instead — same
# cleanup rules, no Gemma-specific scaffolding.
_REMOTE_CLEANUP_SYSTEM_PROMPT = """\
You are `Computer`, a voice-transcript (STT) cleaner and assistant.

# Default mode — CONSERVATIVE CLEANUP
You are NOT a copy editor. Output the transcript verbatim except for these specific edits:
- remove obvious filler words: `ähm`, `öhm`, `halt`, `also`, `um`, `uh`, `like`, `so`
- fix words the STT clearly misheard
- replace spoken punctuation / line-break words with the actual character (see below)
- apply formatting only when explicitly dictated

KEEP every newline and blank line from the input exactly where it is — line breaks and paragraph spacing are part of the user's intended structure and must round-trip 1:1 into the output.

DO NOT:
- rephrase, restructure, or reorder words
- "improve" valid colloquial grammar (especially German modal particles like `denn`, `doch`, `mal`, `ja`, `eben`, `schon` — keep them as-is, they carry meaning)
- change `?` ↔ `.` or drop punctuation that wasn't a spoken word
- normalise mixed German + English — keep the mix
- translate (unless `Computer` mode, see below)

When in doubt: leave it exactly as the user said it.

If nothing needs changing, return the input verbatim — do NOT write `No changes.`, do NOT add commentary, do NOT explain that the text is already clean. Just echo the input.

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
Switch to assistant mode ONLY IF the transcript STARTS with `Hey Computer` (case-insensitive, and tolerate obvious STT mishears like `Hi Computer`, `Hey Computa`). Anything else — including a bare `Computer`, a mid-sentence `hey computer`, or a quoted/reported `hey computer` — is CLEANUP only. Without a leading `Hey Computer`, the transcript is dictated content for some other app (chat, editor, email, …), NEVER for you. This holds EVEN IF the text is phrased as a question, a request, or an instruction. No exceptions, no "but it sounded like a request".

Examples:
- `Can you tell me how many things you can see?`               -> CLEANUP only (no trigger)
- `Ich weiß nicht, was denkst du denn?`                        -> CLEANUP only (no trigger)
- `Translate this to German: hello world`                       -> CLEANUP only (no trigger)
- `Computer, translate this to German: hello world`             -> CLEANUP only (bare `Computer` is NOT the trigger)
- `… and then I told him, hey computer remind me tomorrow.`     -> CLEANUP only (mid-sentence / quoted, not a leading address)
- `Hey Computer, can you tell me how many things you can see?` -> ANSWER (leading `Hey Computer`)
- `hey computer translate this to German: hello world`          -> ACT (case-insensitive leading trigger)

When addressed:
- follow the request directly; do NOT echo the source first
- if asked to translate, output ONLY the translation
- short, on-point reply — no preamble like "Sure, here you go:"

# Output
Return ONLY the cleaned text (default) OR the assistant reply (assistant mode). No meta explanations, no status lines like `No changes.`, no reasoning preamble.
"""


# Minimal "fun" profile prompt — written to disk as gemma4-fun.toml so users
# can flip to a playful, emoji-heavy variant without having to compose a
# prompt themselves.  Intentionally tiny: the recommended everyday default
# is gemma4-cleanup, this one is the silly sibling.
_FUN_SYSTEM_PROMPT = """\
Emojify the transcript as much as possible. Keep the original wording and order, just sprinkle in plenty of fitting emojis (between words, at the end of sentences, wherever they fit). Reply with the emojified text only — no explanations, no preamble.
"""


# Distinctive header line embedded in commented-defaults profile files.
# Used by ``ensure_default_profile`` / ``ensure_fun_profile`` to recognise
# files we've already migrated to the commented form (so we don't keep
# backing them up on each install). Mirrors the marker in config.py.
_PROFILE_COMMENTED_FORM_MARKER = (
    "# justsayit postprocess profile (commented-defaults form)."
)


def _comment_block(text: str) -> str:
    """Return *text* with every line prefixed by ``# `` (or ``#`` for
    empties).  Used to safely embed multi-line defaults (system prompts,
    code samples) inside a commented-defaults TOML file without leaking
    raw lines that would otherwise break TOML parsing."""
    return "\n".join(f"# {line}" if line else "#" for line in text.splitlines())


# Written to disk on first ``justsayit init``. Uses the "commented
# defaults" convention: every value line is commented out so the file
# acts as in-place documentation. Users uncomment + edit only the keys
# they actually want to override; everything else tracks the dataclass
# default. Embedded with single-quoted '''…''' so embedded """ aren't
# parsed as the closing delimiter.
_CLEANUP_PROFILE_TOML = f'''\
{_PROFILE_COMMENTED_FORM_MARKER}
# Profile: gemma4-cleanup (recommended everyday default).
#
# Every key below is commented out — that means "use the shipped default".
# Uncomment a line and change the value to override it for this profile.
# Lines you don't touch keep tracking the shipped defaults, so future
# updates that tweak a default just work.
#
# Activate this profile from the tray's LLM submenu, or set in
# ~/.config/justsayit/state.toml:
#   [postprocess]
#   enabled = true
#   profile = "gemma4-cleanup"
#
# (See gemma4-fun.toml for an example with overrides uncommented.)
#
# Inference backend setup (with Vulkan GPU support):
#   CMAKE_ARGS="-DGGML_VULKAN=1" uv pip install llama-cpp-python
# Then run `justsayit setup-llm` to download a GGUF model.

# Path to the GGUF model file. ~ is expanded.
# model_path = "~/.cache/justsayit/models/llm/gemma-4-E4B-it-Q4_K_M.gguf"

# HuggingFace source for auto-download via `justsayit setup-llm`.
# hf_repo = "unsloth/gemma-4-E4B-it-GGUF"
# hf_filename = "gemma-4-E4B-it-Q4_K_M.gguf"

# GPU layer offloading. -1 = all layers on GPU (fastest). 0 = CPU only.
# n_gpu_layers = -1

# Context window size in tokens.
# n_ctx = 4096

# Temperature. Keep very low (≤ 0.1) for deterministic cleanup.
# temperature = 0.08

# Hard cap on generated tokens.
# max_tokens = 4096

# User message template. {{text}} is replaced with the raw transcription.
# user_template = "{{text}}"

# Regex (re.DOTALL) applied to the LLM output before paste, but NOT
# before display in the overlay. Default strips Gemma's thinking-channel
# block. Set to "" if you remove `<|think|>` from system_prompt.
# paste_strip_regex = '<\\|channel>thought(.*?)<channel\\|>'

# System prompt — the cleanup prompt is the dataclass default. Uncomment
# the block below to override it for this profile. The default enables
# Gemma's "thinking" channel (the `<|think|>` token makes the model emit
# a `<|channel>...<channel|>` reasoning block); pair changes here with
# `paste_strip_regex` above.
# system_prompt = """
{_comment_block(_DEFAULT_SYSTEM_PROMPT.rstrip())}
# """

# User-context lives in a shared sidecar so it's preserved across
# profile updates: ~/.config/justsayit/context.toml. Uncomment to set
# per-profile context that overrides the sidecar:
# context = """
# Name: Jane Doe
# ...
# """

# --- OpenAI-compatible /chat/completions endpoint --------------------
# When `endpoint` is set, the LLM call goes over HTTP instead of loading
# a local GGUF.  The local GGUF fields above (model_path, hf_repo,
# n_gpu_layers, n_ctx) are ignored on this path — no llama-cpp-python
# is needed.  Works with OpenAI, OpenRouter, Groq, Together, vLLM,
# Ollama (`/v1`), LM Studio, llama.cpp's bundled server, etc.
#
# API keys come from one of three places, in priority order:
#   1. `api_key = "sk-..."` below (explicit, lowest-friction for tests)
#   2. the env var named by `api_key_env` (default OPENAI_API_KEY)
#   3. `~/.config/justsayit/.env` — same KEY=VALUE format as
#      python-dotenv; lines you've already exported in your shell win.
#
# When `endpoint` is set AND `system_prompt` is left at the dataclass
# default, justsayit auto-swaps the Gemma `<|think|>`-channel prompt for
# a channel-free variant — generic OpenAI-compatible models don't have
# that channel and would otherwise reply literally `No changes.` or leak
# reasoning into the output. Override `system_prompt` to opt out.
#
# endpoint = "https://api.openai.com/v1"
# model = "gpt-4o-mini"
# api_key = ""
# api_key_env = "OPENAI_API_KEY"
# request_timeout = 60.0
'''


# Companion "fun" profile — emojifies the transcript. Demonstrates what
# the commented-defaults form looks like with actual overrides: the
# three keys that DEFINE the fun flavor (system_prompt, temperature,
# paste_strip_regex) stay uncommented, everything else falls through to
# the dataclass default.
_FUN_PROFILE_TOML = f'''\
{_PROFILE_COMMENTED_FORM_MARKER}
# Profile: gemma4-fun (playful sibling of gemma4-cleanup).
#
# Same commented-defaults convention as gemma4-cleanup.toml: comment =
# uses default, uncommented line = override. The three uncommented keys
# below (system_prompt, temperature, paste_strip_regex) are what makes
# this the "fun" profile — leave them as-is unless you want to customise
# the playful flavor.
#
# For everyday cleanup, switch back to the recommended default:
#   profile = "gemma4-cleanup"
#
# Activate from the tray's LLM submenu, or set in
# ~/.config/justsayit/state.toml:
#   [postprocess]
#   enabled = true
#   profile = "gemma4-fun"

# Same model file as gemma4-cleanup — if you ran `setup-llm` once you
# already have it on disk and no extra download happens.
# model_path = "~/.cache/justsayit/models/llm/gemma-4-E4B-it-Q4_K_M.gguf"
# hf_repo = "unsloth/gemma-4-E4B-it-GGUF"
# hf_filename = "gemma-4-E4B-it-Q4_K_M.gguf"

# n_gpu_layers = -1
# n_ctx = 4096
# max_tokens = 4096
# user_template = "{{text}}"

# Slightly higher temperature so emoji choice has some variety.
temperature = 0.4

# No `<|think|>` in the prompt → no channel block to strip.
paste_strip_regex = ""

# Override: this is the "fun" prompt. Edit freely.
system_prompt = """
{_FUN_SYSTEM_PROMPT}"""

# User-context lives in ~/.config/justsayit/context.toml (shared across
# profiles). Uncomment to override the sidecar for this profile only:
# context = ""
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
    # --- OpenAI-compatible /chat/completions endpoint --------------------
    # When ``endpoint`` is set, the LLM call goes over HTTP instead of
    # loading a local GGUF via llama-cpp-python. Works with any provider
    # that speaks the OpenAI chat-completions schema: OpenAI, OpenRouter,
    # Groq, Together, vLLM, Ollama (/v1), LM Studio, llama.cpp's server …
    # Local GGUF fields above (model_path, hf_repo, n_gpu_layers, n_ctx)
    # are ignored on the remote path.
    endpoint: str = ""
    # Model name passed in the JSON body (e.g. "gpt-4o-mini",
    # "openai/gpt-4o", "qwen2.5-7b-instruct"). Required when endpoint is set.
    model: str = ""
    # Inline API key. Empty by default — prefer api_key_env / .env.
    api_key: str = ""
    # Process env var to read the key from when api_key is empty.
    # Falls through to ``<config_dir>/.env`` (loaded once into os.environ).
    api_key_env: str = "OPENAI_API_KEY"
    # HTTP timeout (seconds) for the chat-completions request.
    request_timeout: float = 60.0


def profiles_dir() -> Path:
    return config_dir() / "postprocess"


# Personal context lives in its own sidecar file rather than per-profile
# so updates to shipped profile TOMLs (system prompt, model paths,
# regexes) can be replaced without clobbering anything the user wrote.
# Profile-level context still works (load_profile honors a non-empty
# `context` field in the profile) and takes precedence over the sidecar.
_CONTEXT_SIDECAR_TEMPLATE = '''\
# Personal context for the LLM postprocessor.
#
# The string assigned to `context` below is appended to every postprocess
# profile's system prompt under a "User context" heading on every
# transcription.  Comments (lines starting with `#`) are NOT sent to the
# model — only the value of `context`.
#
# Tips:
# - Be concise; this is sent on every dictation.
# - Spell out your name (so the model gets it right), country/languages,
#   and any project / proper-noun spellings the model gets wrong.
# - Don't put secrets here.
#
# Example:
#   context = """
#   Name: Jane Doe
#   Country: Germany
#   Languages: German (native), English (fluent), Python
#   Notes: software engineer; often dictates code-related text.
#   """

context = ""
'''


def context_file_path() -> Path:
    return config_dir() / "context.toml"


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


def ensure_default_profiles() -> tuple[Path, Path]:
    """Write both the cleanup and fun default profiles. Returns (cleanup, fun)."""
    ensure_context_file()
    return ensure_default_profile(), ensure_fun_profile()


def load_profile(name_or_path: str) -> PostprocessProfile:
    """Load a :class:`PostprocessProfile` from *name_or_path*.

    If the argument looks like a file path (contains a separator or ends
    with ``.toml``) it is used directly; otherwise it is resolved to
    ``config_dir()/postprocess/<name>.toml``.

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
    valid = {fld.name for fld in fields(PostprocessProfile)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
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
        """Eagerly load the local model so the first transcription is not
        slow.  No-op for the remote endpoint path — there is nothing to
        load locally and a probe request would cost real money / latency."""
        if self.profile.endpoint:
            return
        with self._lock:
            if self._llm is None:
                self._llm = self._build()

    def _system_prompt(self) -> str:
        prompt = self.profile.system_prompt.strip()
        # The shipped default leans on Gemma's `<|think|>` channel to hide
        # reasoning; generic OpenAI-compatible models don't have it and end
        # up replying literally `No changes.` or leaking reasoning. Swap in
        # a channel-free variant when the user hasn't customised the prompt.
        if self.profile.endpoint and prompt == _DEFAULT_SYSTEM_PROMPT.strip():
            prompt = _REMOTE_CLEANUP_SYSTEM_PROMPT.strip()
        ctx = self.profile.context.strip()
        if ctx:
            prompt = f"{prompt}\n\n# User context\n{ctx}"
        return prompt

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self.profile.user_template.format(text=text)},
        ]

    def _remote_process(self, text: str) -> str:
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
        body = {
            "model": self.profile.model,
            "messages": self._build_messages(text),
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
        }
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
            with urllib.request.urlopen(req, timeout=self.profile.request_timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = ""
            raise RuntimeError(
                f"LLM endpoint returned HTTP {exc.code}: {exc.reason}\n  {detail}"
            ) from exc
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"LLM endpoint returned no choices: {str(data)[:300]}"
            )
        return (choices[0].get("message") or {}).get("content", "").strip()

    def _local_process(self, text: str) -> str:
        with self._lock:
            if self._llm is None:
                self._llm = self._build()
            resp = self._llm.create_chat_completion(
                messages=self._build_messages(text),
                temperature=self.profile.temperature,
                max_tokens=self.profile.max_tokens,
            )
        return resp["choices"][0]["message"]["content"].strip()

    def process(self, text: str) -> str:
        """Run the LLM on *text* and return the cleaned result.

        Routes to the remote OpenAI-compatible endpoint when
        ``profile.endpoint`` is set; otherwise loads a local GGUF via
        llama-cpp-python.  Returns the original *text* unchanged if the
        model produces an empty response.
        """
        if self.profile.endpoint:
            result = self._remote_process(text)
        else:
            result = self._local_process(text)
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
