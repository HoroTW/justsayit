# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.16.4] - 2026-04-24

### Fixed
- Remote backend: clipboard image now activates assistant mode in the system prompt (`extra_image_provided` was not threaded through `_build_messages` / `_build_messages_continued`).

## [0.16.3] - 2026-04-24

### Fixed
- Remote backend: clipboard image is now stored in session history so continue-mode turn 2 re-sends it and turn 3+ benefits from prompt caching (same prefix → image tokens are cached rather than billed again).

## [0.16.2] - 2026-04-24

### Fixed
- Remote (`/chat/completions`) backend now sends clipboard images to vision-capable models (gpt-4o-mini, gpt-4o, …). The `image_detail` profile field works the same way; `"original"` falls back to `"auto"` since that tier is Responses-API-only.

## [0.16.1] - 2026-04-23

### Fixed
- Continue session: session.json was cleared instead of saved on every non-continue LLM call, so there was never anything to load when continue mode was armed. Now every successful call saves its exchange (overwriting old history on a non-continue call, appending on a continue call).
- Continue button (↩) clicks were silent — added INFO-level log on arm/disarm.
- History load and save are now logged at INFO (`continue: loaded N-turn history` / `session saved (N turns)`).

## [0.16.0] - 2026-04-23

### Added
- **Continue session** — arm the ↩ button in the overlay (or `justsayit toggle --continue`) to start a 5-minute window in which each recording continues the previous LLM conversation. Session history grows with each turn and persists in `~/.cache/justsayit/session.json`. The Responses API uses `previous_response_id` natively; other backends prepend the full message history.
- `postprocess.continue_window_minutes` config option (default 5).

## [0.15.1] - 2026-04-23

### Fixed
- Web search cost: only `search` actions count for the flat fee; `open_page` billed separately via `web_open_page_price_per_call` (default 0.0).

## [0.15.0] - 2026-04-23

### Added
- **Clipboard image input** (Responses API only) — arm the 📋 button with an image on the clipboard to send it alongside your dictation. Profile field `image_detail`: `"auto"` (default), `"low"`, `"high"` (OCR/screenshots), `"off"`.

### Changed
- LLM input→output now always logged at INFO, even when unchanged.
- Dynamic context content moved to DEBUG; paste/release timing moved to DEBUG.

## [0.14.1] - 2026-04-23

- README update only.

## [0.14.0] - 2026-04-23

### Added
- **Responses API backend** (`base = "responses"`) — OpenAI `/v1/responses` with cached instructions, optional web search tool, and `reasoning_effort` support.
- Web search auto-enabled when clipboard context is armed (no regex trigger needed).
- Retry logic for OpenAI Whisper backend (`openai_retries`, `openai_retry_delay`).
- `install.sh --update` now re-installs llama-cpp-python if dropped by `uv sync`.

### Changed
- Codebase restructured: `postprocess.py` → package, `config.py` → package, `cli.py` shed pipeline/boot/subcommand code into separate modules.

## [0.13.35] - 2026-04-23

- Internal: extracted `_http.py` retry helper — no user-visible change.

## [0.13.34] - 2026-04-22

- Internal: consolidated `ensure_profile` wrappers — no user-visible change.

## [0.13.33] - 2026-04-22

### Fixed
- System prompt ordering unified across all backends: dynamic context (clipboard, dynamic-context.sh) always appended after static instructions.

## [0.13.32] - 2026-04-22

- CLAUDE.md update only.

## [0.13.31] - 2026-04-22

- Internal refactor (precursor to 0.14.0 backend split) — no user-visible change.

## [0.13.30] - 2026-04-22

### Removed
- Anthropic native backend (`base = "anthropic"`) — too expensive for routine cleanup.

## [0.13.29] - 2026-04-22

- Internal: postprocess split into package; cli.py shrunk — precursor to 0.14.0.

## [0.13.28] - 2026-04-22

### Added
- Token usage logged after every remote LLM call.
- Optional cost logging: set `input_price_per_1m`, `output_price_per_1m`, `cached_input_price_per_1m` in a profile (USD per 1M tokens, default 0.0).

## [0.13.27] - 2026-04-22

### Changed
- `openai-cleanup` profile now defaults to `gpt-5.4-mini` with `reasoning_effort = "low"` (was `gpt-4o-mini`). Hits 100% on the eval suite with lower p90 latency.

## [0.13.26] - 2026-04-22

### Added
- `reasoning_effort` profile field — forward `"low"` / `"medium"` / `"high"` to OpenAI reasoning models.
- `evals/IMPROVING.md` — guide for future prompt-iteration sessions.

### Fixed
- Remote backend now auto-strips unsupported sampling params (`temperature`, `top_p`, etc.) for reasoning models.

## [0.13.25] - 2026-04-22

### Changed
- `Hey Computer` trigger tightened: only that exact phrase (case-insensitive, with close STT mishears). Other greetings + "Computer" no longer enter assistant mode.

## [0.13.24] - 2026-04-22

### Changed
- `cleanup_openai.md`: improved clipboard-present assistant mode accuracy (+10.5 pp overall).

## [0.13.23] - 2026-04-22

### Added
- Prompt-evaluation harness: `evals/` + `scripts/eval-cleanup-prompt.py` (19 cases, confusion matrix, `--runs`, `--json`, `--no-judge`).

### Changed
- Clipboard context now automatically triggers assistant mode.
- `cleanup_openai.md` rewritten: −39% size, +5.3 pp accuracy.

## [0.13.22] - 2026-04-22

### Fixed
- Clipboard-context injection skips non-text clipboards (images no longer fed as garbage bytes to the LLM).

## [0.13.21] - 2026-04-22

### Added
- `justsayit toggle` subcommand — sends commands to the running app over DBus (~80 ms cold start). Flags: `--profile NAME` (switches active LLM profile), `--use-clipboard` (arms clipboard context for the next recording). Useful for special-mode keyboard shortcuts.

## [0.13.20] - 2026-04-21

### Fixed
- Clipboard-context arming now disarms at the start of each recording (no stale-arm leakage).
- `cleanup_openai.md`: fixed example so the model uses provided clipboard content instead of refusing.

## [0.13.19] - 2026-04-19

### Changed
- `wl-copy` now passes `--type text/plain`, skipping `xdg-mime` lookup. Per-call latency: ~165 ms → ~64 ms.

## [0.13.18] - 2026-04-19

### Changed
- Regular + primary `wl-copy` calls now run in parallel — halves clipboard copy time on slow portal setups.

## [0.13.17] - 2026-04-19

### Fixed
- Switching LLM profile to "Off" in the tray now actually disables postprocessing.

### Changed
- Timing logs added around the LLM call and paste call.

## [0.13.16] - 2026-04-19

### Changed
- Tray LLM profile submenu now lists all parseable profiles regardless of backend readiness. Failures surface at selection time.

## [0.13.15] - 2026-04-19

### Fixed
- `base = "remote"` profiles no longer disappear from the tray on machines without the local GGUF.

## [0.13.14] - 2026-04-19

### Changed
- Built-in backend default `n_ctx` raised from 4096 → 20480 (avoids silent truncation on longer prompts / clipboard context).

## [0.13.13] - 2026-04-19

### Fixed
- "processing…" pill could stick after aborting during RECORDING/MANUAL — now clears correctly.

### Changed
- `cleanup_gemma.md`: bare imperative sentences without "Computer" no longer trigger assistant mode.

## [0.13.12] - 2026-04-19

### Added
- Overlay 📋 button arms one-time clipboard context for the current manual recording. Click again to cancel; auto-disarms after use.
- `process_with_reasoning(text, *, extra_context="")` on the postprocessor — exposes both pasted text and reasoning.

### Changed
- Prompts renamed: `cleanup_local.md` → `cleanup_gemma.md`, `cleanup_remote.md` → `cleanup_openai.md`. Legacy names still work with a deprecation warning.

## [0.13.11] - 2026-04-19

### Added
- Overlay reasoning slot now shows DeepSeek/vLLM/OpenRouter reasoning (`reasoning_content` / `reasoning` fields), not just local `<think>` blocks.

## [0.13.10] - 2026-04-18

### Changed
- `ollama-gemma.toml` default endpoint changed to LM Studio (`http://localhost:1234/v1`).
- `cleanup_qwen_simple.md`: removed assistant-mode logic (0.8B can't reliably follow it; cleanup only).

## [0.13.9] - 2026-04-18

### Fixed
- Qwen 3.5 0.8B looping "Thinking Process:" output fixed with a dedicated `cleanup_qwen_simple.md` prompt and corrected `setup-llm qwen3-0.8b` seed.

## [0.13.8] - 2026-04-18

### Fixed
- Remote backend crashed with HTTP 400 when `chat_template_kwargs` was sent to the real OpenAI API. Remote default is now `{}`.

## [0.13.7] - 2026-04-18

### Fixed
- Profiles silently disappeared from the tray after re-running `setup-llm` (TOML duplicate-key parse error). `_set_toml_key` now upserts cleanly.

## [0.13.6] - 2026-04-18

### Changed
- `setup-llm qwen3-0.8b` now seeds Qwen-recommended sampling params (`temperature = 0.6`, `presence_penalty = 1.5`, etc.) to avoid thinking loops.

## [0.13.5] - 2026-04-18

### Added
- Six new sampling profile fields: `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`.

## [0.13.4] - 2026-04-18

### Fixed
- Local LLM crashed (`LlamaChatCompletionHandlerNotFoundException`) for GGUFs with bundled Jinja templates (Gemma 4, Qwen 3.5, Llama 3.x).

## [0.13.3] - 2026-04-18

### Fixed
- Local backend crashed with `unexpected keyword argument 'chat_template_kwargs'` — now routes through `chat_handler`.

## [0.13.2] - 2026-04-18

- Profile template comment fix — no user-visible change.

## [0.13.1] - 2026-04-18

### Changed
- Profile templates now document `chat_template_kwargs` and `append_to_system_prompt` as commented examples.

## [0.13.0] - 2026-04-18

### Added
- `chat_template_kwargs` profile field (default `{ enable_thinking = true }`) — enables Qwen 3.5 thinking mode; ignored by Gemma and OpenAI.
- `append_to_system_prompt` profile field — append a short addition to any prompt without forking the `.md` file.

## [0.12.1] - 2026-04-18

### Added
- `paste.restore_delay_ms` (default 250) — tunable delay before clipboard is restored after paste (was hardcoded at 150 ms).

## [0.12.0] - 2026-04-18

### Changed (BREAKING)
- Profile `base` field now explicit: `"builtin"` | `"remote"` (was inferred from `endpoint` presence). Legacy profiles without `base` auto-infer.
- Defaults now come from `builtin-defaults.toml` / `remote-defaults.toml` rather than Python dataclass defaults.
- System prompt decoupled from backend: use `system_prompt_file = "filename.md"`. Old `endpoint`-based auto-swap removed.
- New `ollama-gemma.toml` shipped profile (`base = "remote"` + Gemma's `<|think|>` prompt, for Ollama/LM Studio).

### Removed
- Baseline reconciliation machinery (replaced by commented-defaults form introduced in 0.9.0).

## [0.11.16] - 2026-04-18

- Internal cleanup — no user-visible change.

## [0.11.15] - 2026-04-18

- Internal cleanup — no user-visible change.

## [0.11.14] - 2026-04-17

### Changed
- `Hey Computer` now triggers assistant mode anywhere in a transcript (not just leading). Profile templates extracted to `src/justsayit/templates/`.

## [0.11.13] - 2026-04-18

### Added
- `killall justsayit` and `pgrep justsayit` now find the running process (`prctl(PR_SET_NAME)` at startup).

## [0.11.12] - 2026-04-18

### Fixed
- Gemma local prompt no longer pastes literal `No changes.` when nothing needed cleanup.

### Changed
- Relaxed `<|think|>` reasoning constraint — multi-sentence thinking is fine for tricky inputs.

## [0.11.11] - 2026-04-18

### Fixed
- Prompts hardened against bare questions triggering assistant mode without "Computer".

### Changed
- Prompts extracted to standalone Markdown files under `src/justsayit/prompts/`.

## [0.11.10] - 2026-04-17

### Fixed
- Prompts hardened against bare `Hey`/`Hi`/`Hallo` misfires — the literal word `Computer` must be present.

## [0.11.3] - 2026-04-17

### Changed
- Trailing `Hey Computer` rewrite reverted — prompt-guided best-effort only.

## [0.11.2] - 2026-04-17

- Doc update only — surfaced "Hey Computer" inline assistant mode in README.

## [0.11.1] - 2026-04-17

- Doc update only — README feature list and docs expanded.

## [0.11.0] - 2026-04-17

### Added
- `openai-cleanup.toml` shipped as a third default profile alongside `gemma4-cleanup` and `gemma4-fun`.

## [0.10.3] - 2026-04-17

### Changed
- Blank-line preservation rule in prompts reframed as a positive instruction (more reliably followed by models).

## [0.10.2] - 2026-04-17

### Changed
- Cleanup prompts now explicitly forbid removing blank lines or collapsing whitespace.

## [0.10.1] - 2026-04-17

### Fixed
- Remote LLM no longer replies with literal `No changes.` or leaks `<|channel>thought…` — channel-free prompt now used automatically when `endpoint` is set.

## [0.10.0] - 2026-04-17

### Added
- **OpenAI-compatible LLM endpoint** for postprocessing — set `endpoint`, `model`, and API key in a profile; no new dependencies (stdlib `urllib`).
- **OpenAI-compatible Whisper STT backend** (`model.backend = "openai"`) — posts audio to any `/audio/transcriptions` endpoint.
- **Shared `.env` file** at `~/.config/justsayit/.env` for API keys.

## [0.9.1] - 2026-04-17

### Fixed
- Profiles silently missing from the tray after `init` — multi-line system prompt in the template generated invalid TOML.

## [0.9.0] - 2026-04-17

### Changed
- Config and profile TOMLs now use a **commented-defaults form**: every default is commented out; user uncomments only overrides. Existing files migrated automatically (backed up to `.bak-pre-commented-form`).

### Removed
- Defaults-baseline reconciliation machinery (no longer needed with commented-defaults form).

## [0.8.13] - 2026-04-17

### Changed
- `setup-llm` now directs users to the tray menu instead of `config.toml` after model download.

## [0.8.12] - 2026-04-17

### Fixed
- `setup-llm gemma4` no longer creates a redundant `gemma4.toml`; patches the two shipped profiles in place.

## [0.8.11] - 2026-04-17

### Fixed
- Hotkey rebind dialog after tray-triggered restart (CLI launches only) — restart now uses `DesktopAppInfo.launch` so the portal recognises the app-id consistently.

## [0.8.10] - 2026-04-17

### Fixed
- `install.sh --update` no longer destroys the venv (and with it `llama-cpp-python`) — reuses existing venv.

### Added
- `--update` warns if `llama_cpp` isn't importable but postprocess is enabled in state.

## [0.8.9] - 2026-04-17

### Fixed
- Postprocess profile TOMLs now get a baseline snapshot on first write, so `install.sh --update` can distinguish stale defaults from user customisations.

## [0.8.8] - 2026-04-17

### Changed
- Runtime state (`vad.enabled`, `postprocess.enabled`, `postprocess.profile`) moved to `state.toml`. `config.toml` is never rewritten by the app — user comments and customisations survive.

## [0.8.7] - 2026-04-17

### Changed
- Baseline snapshots moved to hidden `.baseline/` subdirectories (was: sibling files). Migrated automatically on next launch or `--update`.

## [0.8.6] - 2026-04-17

### Changed
- `install.sh --update` no longer reconciles `config.toml` (it's user settings, not a shipped template). Use `justsayit show-defaults config` to diff manually.

## [0.8.5] - 2026-04-17

### Fixed
- `install.sh --update` was silently looking in the wrong config directory and skipping every reconcile prompt.

## [0.8.4] - 2026-04-17

### Changed
- `install.sh --update` now defaults to "yes" when offering to replace stale user config files.

## [0.8.3] - 2026-04-17

### Changed
- Assistant-mode trigger tightened: only `Hey Computer` at the START of a transcript triggers it. Mid-sentence and trailing uses stay as plain cleanup.

## [0.8.2] - 2026-04-17

### Added
- **Personal-context sidecar** `~/.config/justsayit/context.toml` — free-form text appended to every postprocess prompt without touching profiles.
- `install.sh --update` now reconciles postprocess profile TOMLs too.
- `show-defaults context|profile-cleanup|profile-fun` subcommand variants.

## [0.8.1] - 2026-04-17

### Changed
- `install.sh --update` now uses a defaults-baseline to distinguish stale defaults from user customisations — no more scary "overwrite everything?" diff for users who never edited a file.

## [0.8.0] - 2026-04-17

### Added
- **GitHub update check on startup** — background fetch; overlay badge + desktop notification when a newer version is available (cached 3h).
- **`install.sh --update`** — pull latest commits, refresh venv, reconcile `config.toml` / `filters.json`.
- `justsayit show-defaults config|filters` subcommand.

## [0.7.2] - 2026-04-17

### Added
- **Spoken-punctuation regex filters** shipped as `filters.json` defaults — `Punkt`, `Komma`, `Fragezeichen`, `neue Zeile`, etc. (German + English). Clean output without the LLM postprocess step.

## [0.7.1] - 2026-04-17

### Changed
- Cleanup prompt: added explicit spoken-punctuation mapping section; `<|think|>` reasoning constrained to a single sentence.

## [0.7.0] - 2026-04-17

Milestone rolling up LLM postprocessing UX improvements (0.6.8–0.6.15).

### Highlights
- `gemma4-cleanup` profile: conservative, no rephrasing, German modal particles preserved, assistant mode only on literal "Computer".
- `gemma4-fun` companion profile written on `init`.
- Overlay renders Gemma's reasoning italic blue-green; result linger halved to 5 s.
- Abort × pinned to overlay top-right in expanded result view.

## [0.6.15] - 2026-04-17

### Changed
- `paste_strip_regex` now strips the `thought` label Gemma emits with `<|channel>`. Overlay linger halved to 5 s. Thought separated from body with blank line.

## [0.6.14] - 2026-04-17

### Changed
- Default profile renamed `gemma-cleanup` → `gemma4-cleanup`. Conservative tuned prompt made the default.

## [0.6.13] - 2026-04-17

### Changed
- `paste_strip_regex` capture group: group 1 shown in the overlay, full match stripped from the paste. Thought rendered italic blue-green (`#5ed1c4`).

## [0.6.12] - 2026-04-17

### Changed
- Default profile system prompt stored as TOML triple-quoted string. Overlay shows thought preamble separately from the pasted body.

## [0.6.10] - 2026-04-17

### Fixed
- Default `paste_strip_regex` now matches Gemma's actual asymmetric channel tags (`<|channel>` … `<channel|>`). Previous pattern never matched, leaking reasoning into the focused window.

## [0.6.9] - 2026-04-17

### Fixed
- Abort × stays pinned to overlay top-right in result view (layout collapse fix).

## [0.6.8] - 2026-04-17

### Changed
- Default `gemma-cleanup` prompt rewritten: `Hey Computer` assistant mode, German/English mixed, formatting examples. `max_tokens` raised to 4096.

## [0.6.7] - 2026-04-17

### Added
- `context` field on postprocess profiles — free-form text appended to the system prompt under a `# User context` heading.

## [0.6.6] - 2026-04-17

### Added
- **Abort button (×)** in overlay — discards the audio buffer and returns to IDLE without transcribing.

## [0.6.5] - 2026-04-17

### Added
- `paste_strip_regex` profile field — strip a regex from LLM output before pasting (e.g. Gemma's reasoning block) while still showing the full output in the overlay.

## [0.6.4] - 2026-04-17

### Changed
- Manual mode (VAD off) now closes the microphone between recordings.

## [0.6.3] - 2026-04-17

### Changed
- Install instructions: `dotool` correctly fetched via AUR helper (not `pacman`).

## [0.6.2] - 2026-04-17

### Removed
- `wtype` paste backend — `dotool` covers all supported compositors uniformly.

## [0.6.1] - 2026-04-17

### Fixed
- `with-llm-vulkan` (Nix) now works on non-NixOS hosts — bundled nixpkgs Mesa Vulkan ICDs with absolute store paths.

## [0.6.0] - 2026-04-16

### Changed
- README restructured with copy-paste quick-start paths; detailed docs moved to `docs/`.

## [0.5.4] - 2026-04-16

### Added
- Nix LLM packages: `nix build .#with-llm` (CPU) and `.#with-llm-vulkan` (Vulkan GPU).
- `install.sh --nix [BINARY]` — install desktop integration for a Nix-built binary.

### Changed
- App ID renamed `dev.horo.justsayit` → `dev.horotw.justsayit`.

## [0.5.3] - 2026-04-16

### Added
- `paste.restore_clipboard` (default true) — restores the clipboard to its previous content after paste.
- Nix flake (`flake.nix`) — packages justsayit with GTK4 layer-shell, PipeWire, and runtime tools.

## [0.5.2] - 2026-04-15

### Added
- `paste.skip_clipboard_history` (default true) — pass `--sensitive` to `wl-copy` so clipboard managers skip recording dictated text.
- `paste.type_directly` — inject text via `dotool type` without touching the clipboard.

## [0.5.1] - 2026-04-15

### Added
- **Overlay result linger** — after transcription, overlay expands to show detected text (top) and LLM-cleaned result (bottom, italic green). Duration: `overlay.result_linger_ms` (default 10 s).
- `overlay.max_width` / `overlay.max_height` config fields.

## [0.5.0] - 2026-04-15

### Added
- **LLM postprocessor** — optional cleanup step via local GGUF model (`llama-cpp-python`). Removes filler words, fixes grammar/spelling, corrects misheards.
- `[postprocess]` config section, per-model profile TOMLs, `gemma-cleanup` default profile.
- `justsayit setup-llm` wizard — lists models, downloads GGUF, patches profile.
- `install.sh --postprocess` — builds and installs `llama-cpp-python` with Vulkan.

## [0.4.0] - 2026-04-15

### Added
- **Multi-backend transcription**: `model.backend = "parakeet"` (default) or `"whisper"` (faster-whisper).
- `install.sh --model parakeet|whisper` and `[whisper]` pip extra.

## [0.3.3] - 2026-04-15

### Fixed
- Auto-listen tray toggle now works correctly on first click after starting with VAD disabled.

## [0.3.2] - 2026-04-15

### Added
- Mute/unmute sounds for VAD auto-listen mode (two-tone chimes).

## [0.3.1] - 2026-04-15

### Changed
- Start chime now plays on `VALIDATING` entry (early auditory feedback) at reduced volume.

## [0.3.0] - 2026-04-15

### Added
- **Notification sounds** — chimes on recording start/stop. Config: `sound.enabled`, `sound.volume`.

## [0.2.2] - 2026-04-15

### Added
- **Reload config** tray item — restarts via `execve` to apply all config changes.

## [0.2.1] - 2026-04-15

### Changed
- Default overlay width reduced (260 → 174 px).

## [0.2.0] - 2026-04-15

### Added
- `paste.auto_space_timeout_ms` — auto-prepend a space between consecutive dictations.
- `paste.append_trailing_space` — always append a trailing space.
- `overlay.visualizer_sensitivity`, `overlay.opacity` config fields.

### Changed
- Overlay layout: status dot left + level meter right; visualizer grows from center.

## [0.1.0] - 2026-04-13

### Added
- Initial release.
