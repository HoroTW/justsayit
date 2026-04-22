# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: justsayit

Local Parakeet-powered voice dictation for Wayland. Single-process GTK4
app with a layer-shell overlay; ASR backends (Parakeet / Whisper /
OpenAI-compatible) feed an optional LLM cleanup pass (local
llama-cpp-python or any OpenAI-compatible endpoint).

## Common commands

```sh
# Run the app (after install.sh has set up the venv)
uv run justsayit              # overlay + portal hotkey
uv run justsayit --no-overlay # headless (e.g. GNOME Mutter — no layer-shell)
uv run justsayit --no-paste   # print transcripts to stdout instead of pasting
uv run justsayit init         # write default config / profiles / filters
uv run justsayit show-defaults config   # diff your config vs current defaults

# Tests (default config in pyproject.toml deselects `burn` marker)
uv run pytest                                # full suite, ~1.5s when HF is reachable
uv run pytest --no-network                   # skip HF-reachability tests (defined in conftest.py)
uv run pytest -m burn                        # slow GGUF-loading integration tests
uv run pytest tests/test_postprocess.py::test_name -v   # single test

# Version bump (three files must stay in sync; the script enforces it
# and refreshes uv.lock — CHANGELOG.md is still your job)
./scripts/bump-version.py 0.13.21
```

Project memory expects every commit to bump the version + add a CHANGELOG
entry; do this even for tiny patch fixes.

## Architecture

**Threading model (cli.py / `App`):** GLib mainloop owns the UI thread.
A `sounddevice` callback feeds raw audio into `audio.AudioEngine`, which
runs its own worker thread to assemble `Segment` objects. Segments go on
a bounded queue consumed by a transcribe thread, which calls the active
`Transcriber` and then `_handle_segment` (filters → optional LLM →
paste). All cross-thread UI updates go through `GLib.idle_add` (mainly
via `OverlayWindow.push_*` helpers).

**State machine (audio.py):** four states — `IDLE`, `VALIDATING`,
`RECORDING`, `MANUAL`. Two activation modes coexist:
- `vad.enabled = false`: hotkey-only. Mic is closed until
  `start_manual()` opens it; `stop_manual()` emits the segment.
- `vad.enabled = true`: Silero VAD watches a continuously-open stream;
  the first `validation_seconds` are transcribed in a "did we hear
  words?" check before transitioning to `RECORDING`.
The `on_state` callback drives the overlay, sound chimes, and any
recording-edge work (e.g. clipboard-context disarm fires here on
`IDLE → VALIDATING/MANUAL`).

**LLM postprocessing (postprocess.py):** profiles live in TOML files
under `~/.config/justsayit/postprocess/`. Each profile sets
`base = "builtin" | "remote"` which pulls defaults from
`src/justsayit/templates/{builtin,remote}-defaults.toml`; only
explicit overrides go in the user file. The system prompt is composed
at call time:
```
[# STATE (DYNAMIC CONTEXT)]   # from dynamic-context.sh stdout if any
{system_prompt or system_prompt_file}
{append_to_system_prompt}
[# User context]              # from context.toml
[# Clipboard as additional context]   # from one-shot 📋 button
```
System prompts are markdown files in `src/justsayit/prompts/`
(`cleanup_gemma.md`, `cleanup_openai.md`, `cleanup_qwen_simple.md`,
`fun.md`); the profile's `system_prompt_file` field references one by
bare name (resolved against that directory).

**Re-exec dance (cli.py):** at startup the process re-execs itself
twice — once with `LD_PRELOAD=libgtk4-layer-shell.so` (so layer-shell
is initialised before GTK), once under a systemd user scope (so the
XDG GlobalShortcuts portal can resolve the app-id from the cgroup
unit name). Tests bypass both via env vars set in `tests/conftest.py`
(`_JUSTSAYIT_SCOPED=1`, `_JUSTSAYIT_PRELOADED=1`).

**Version pinned in 3 places:** `pyproject.toml`,
`src/justsayit/__init__.py`, and `uv.lock`. Use `scripts/bump-version.py`
— manual edits desync them.

## Editing prompts

When fixing an LLM-prompt regression in `src/justsayit/prompts/*.md`,
add a general rule, NOT the failing input as a literal example. Examples
should illustrate the *category* generically; mirroring the user's test
sentence patches one phrase and bloats the prompt.

---

## Behavioral guidelines (general)

These reduce common LLM coding mistakes. Bias toward caution over speed;
for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
