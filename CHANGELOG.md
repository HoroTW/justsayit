# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.23.1] - 2026-04-30

### Fixed
- 🤖 redo button now actually flips strict-rule models (Gemma) into assistant mode. The 0.23.0 implementation passed the right `assistant_mode=True` flag but ALSO injected a free-text "REDO: respond as an assistant…" nudge — Gemma's static cleanup prompt has hard rules ("MUST have 'Computer' in transcript") that override loose nudges, and the redo path also lacked the tool-wiring the regular 💬 button path sets up. `redo_with_override` now mirrors `handle()`'s LLM call shape exactly: no custom nudge, same tool/tool_caller setup; the mode flag alone does the work, just like the regular button.

## [0.23.0] - 2026-04-30

### Added
- Overlay redo buttons (🧹 cleanup / 🤖 assistant): after an LLM result is shown, the opposite-mode button appears so the user can re-run the cached detected text without re-recording. Clicking 🧹 re-runs with cleanup mode; 🤖 re-runs with assistant mode. A strong nudge is injected into the system prompt so the model honours the override.
- "LLM thinking" placeholder is animated (cycling dots) so the wait feels alive instead of frozen. Updates every 300 ms while the LLM call is in flight.

## [0.22.5] - 2026-04-30

### Fixed
- LLM output containing a markdown link no longer falls back to plain text. The Pango validator added in 0.22.1 rejects `<a>` tags (a GtkLabel extension that pure Pango doesn't recognise), so any response with even one link lost ALL its markdown styling — bold, code, tables, italics — and rendered as raw text. The validator now strips `<a>` open/close tags for the validation pass only; the original markup with links intact still reaches `set_markup`.

## [0.22.4] - 2026-04-30

### Fixed
- A single click on the result pill no longer select-alls the text, and drag-to-select within a label works again. Both were caused by the explicit `grab_focus()` call added in 0.22.3 — `grab_focus()` on a selectable GtkLabel triggers GTK's select-all-on-focus-gain and also interrupts the in-flight press→drag sequence. Removed the explicit grab; GTK's natural click-to-focus handles it correctly because `can_focus=True` is set in CAPTURE phase before the label sees the event.

## [0.22.3] - 2026-04-30

### Fixed
- Overlay no longer steals keyboard focus from the focused window when a result is shown — paste was breaking because layer-shell `ON_DEMAND` + focusable labels let the compositor grant the overlay focus before the user clicked. Labels now stay non-focusable until the user explicitly clicks the result pill, at which point focus is enabled and granted to the clicked label so Ctrl+C works.

### Changed
- Markdown tables now wrap cells longer than 40 chars onto multiple display rows instead of letting one wide cell stretch the overlay across the screen. The whole table is rendered with `allow_breaks="false"` so Pango doesn't re-break the carefully-aligned rows on whitespace.

## [0.22.2] - 2026-04-30

### Fixed
- Ctrl+C in the result overlay actually works now: the labels were marked `can_focus=False` (a leftover from 0.20.0), so the layer-shell ON_DEMAND keyboard mode had nothing to deliver focus to. Removed `can_focus=False` so clicking the label grabs focus and routes Ctrl+C to GtkLabel's clipboard handler.
- Clicking on the result text actually cancels the auto-dismiss now: switched the root gesture controller to CAPTURE phase so it fires before selectable-label gestures swallow the click. Removed the per-label CAPTURE-phase gestures (no longer needed). The handler also logs at INFO so the cancellation is visible in the logs.
- Markdown table separator row no longer has an extra `─` at each end; the `┼` columns now align exactly with the `│` columns in the data rows.

## [0.22.1] - 2026-04-30

### Fixed
- Overlay no longer hangs at "Wait for LLM processing…" when the LLM returns a fenced code block containing a markdown table. The fence pass now runs before the table pass so `|`-lines inside fences are never misdetected as tables.
- `_set_label_markup_safe` now validates markup with `Pango.parse_markup` before calling `set_markup`; invalid markup falls back to `set_text` instead of silently leaving the label unchanged.

## [0.22.0] - 2026-04-30

### Fixed
- Ctrl+C in the result overlay now copies selected text. The layer-shell window switches to `KeyboardMode.ON_DEMAND` while a result is shown so the compositor routes key events to GTK's native clipboard handler; switches back to `NONE` when the overlay collapses so it never intercepts keys while you're typing into the focused app.
- Clicking directly on the result text now cancels the auto-dismiss timer. Selectable GtkLabels capture clicks so the parent gesture wasn't firing — added a CAPTURE-phase gesture on each label that runs before the label's own click handling.

### Added
- Markdown tables in the LLM result render as monospace aligned blocks with `─┼─` dividers instead of raw `| col | col |` text.

## [0.21.2] - 2026-04-30

### Fixed
- `setup-llm gemma4`: unpacking `ensure_default_profiles()` into 4 variables crashed with `ValueError`; corrected to 5.
- `pipeline.py`: `apply_filters` exceptions (pre-LLM and post-LLM) are now caught and surfaced as error pills instead of crashing the transcribe thread.
- `pipeline.py`: paste failures now emit an error pill so the user sees the failure in the overlay.
- `pipeline.py`: empty exception message no longer causes `IndexError` in the LLM error detail line.
- Overlay clipboard-context tooltip had a dangling `"recording"` word; corrected to `"just once for the next recording"`.
- `_StubOverlay` test fixture lacked `push_tool_call`; added no-op to prevent future `AttributeError`.
- Clipboard-context tests no longer spawn a real `wl-paste` process; `read_clipboard_image` is now monkeypatched to `None` in the relevant fixtures.

## [0.21.1] - 2026-04-30

### Fixed
- Overlay crashed at runtime as soon as the meter ticked: `_tick` accepted 3 args but `add_tick_callback` calls it with 4 (widget, frame_clock, user_data). The mic-level indicator stopped updating because the first tick raised `TypeError` and silently killed the animation. Added regression tests that assert callback arity for `_tick`, the draw funcs, gesture handlers, and the selection-notify handler.

## [0.21.0] - 2026-04-30

### Added
- Soft-failure error pill in the overlay — ASR/LLM exceptions now surface as an amber pill with a 🔁 retry button instead of vanishing silently.
- Window-class clipboard policy — `[window_clipboard_policy]` config section auto-arms or blocks clipboard-context based on the focused window's class (e.g. block in password managers, auto-arm in chat apps).

### Changed
- Overlay tick loop uses `add_tick_callback` and auto-pauses while hidden — saves CPU when no recording is active.
- Overlay button CSS deduped via a single `.justsayit-overlay-btn` base class.
- Postprocess profile defaults: dropped the parallel `*-defaults.toml` overlay; the dataclass is the single source of truth (TOML files remain as user-facing reference).
- `update_profile_model` / `apply_profile_overrides` now use `tomlkit` for round-trip-safe profile edits.
- `render_config_toml` uses `tomlkit.dumps` (proper string escaping).
- Profile system-prompt assembly: `_build_messages` now handles fresh and continuation cases via `prev_messages=`.

### Removed
- Dead one-shot helpers `paste_text` and `send_paste_shortcut` from `paste.py`.
- `_handle_segment` "transient pipeline" fallback — production path always uses the constructed `SegmentPipeline`.
- Legacy fully-populated-form config migration (`ensure_commented_form_file`).
- Dual JSON+TOML parsing in `update_check.py` — only the production TOML path remains.
- Redundant post-download existence checks in `model.py`.
- `make_transcriber`'s `model_paths` parameter — Parakeet resolves its own paths.

## [0.20.0] - 2026-04-29

### Added
- Selectable text in the overlay — both the regex-filtered transcript and the LLM result can now be selected and copied with the mouse. Starting a selection cancels the auto-dismiss timer.
- Markdown rendering in the LLM result field — bold, italic, inline/fenced code, headings, bullet lists, links, blockquotes, and strikethrough render as Pango markup instead of showing raw `**stars**`.

## [0.19.2] - 2026-04-29

### Fixed
- Manual recording stuck in `manual` state after abort: stream opened before `external_start` flag was set, so the first audio chunk could close the stream again before the flag was seen; abort/stop events are only processed on chunks, leaving the engine frozen.

## [0.19.1] - 2026-04-29

### Fixed
- Pasting broken when stdout is closed (headless/autostart): `print(final)` was unconditional and crashed the segment handler with `OSError: [Errno 5]`; moved inside the no-paste branch where it belongs.

## [0.19.0] - 2026-04-29

### Fixed
- `responses_web_search_trigger` is now bypassed in assistant mode — web search was silently omitted when assistant mode was activated via the UI button (no "Hey Computer" in the transcription, no clipboard shared).

## [0.18.0] - 2026-04-27

### Added
- **Local LLM stash** — switching away from a local profile stashes the loaded model; switching back reattaches instantly. Stashed and active local profiles show a `*` suffix in the tray submenu.
- **"Unload local LLM"** tray item (main menu, only when a local model is in memory) — drops the stash and falls back to the last-used remote profile or off.
- `justsayit unload-llm` subcommand — same as the tray item, for scripting.
- **Overlay LLM profile label** — tiny top-left line shows the active backend and profile (`local/gemma4-cleanup`, `responses/gpt-5.4-mini`, …) in both the recording pill and the result view; shows `direct (no LLM)` when postprocessing is off.

### Fixed
- LLM warmup runs on a background thread; recording starts immediately.
- `justsayit toggle --profile <same>` no longer rebuilds the postprocessor when already active.
- Gemma 4 native tool call tokens (`<|tool_call>…<tool_call|>`) now parsed and executed correctly.
- Tool injection gated to button assistant mode only; "Hey Computer" inline calls are unaffected.
- Clicking the result pill now cancels auto-dismiss only; assistant mode must be activated explicitly via 💬.

## [0.17.0] - 2026-04-25

### Added
- **Assistant mode** (`💬` button in the overlay) — toggle to keep the overlay open after each result instead of auto-dismissing; the response is displayed but **not pasted** into the focused window; session continuation is automatically armed so every recording builds on the previous reply; clicking the result pill activates assistant mode on the fly. A 📄 copy-to-clipboard button appears alongside the result.
- **Custom function tools** (`~/.config/justsayit/tools.json`) — define shell-backed tools in OpenAI function-calling format; the LLM can invoke them during a request (max 10 rounds). The exec string supports `{param}` substitution (shell-quoted). The overlay shows `⚙ tool_name(params)` during execution. Supported by all three backends. Profile field `use_tools = true` (default); set to `false` to opt out per profile.
- `justsayit init` creates a commented example `tools.json`.

### Fixed
- Assistant mode injects a `# ASSISTANT MODE` section into the dynamic system prompt so the model responds conversationally rather than treating input as transcription to clean up.
- Responses API tool follow-up used wrong item type (`function_call_result` → `function_call_output`); caused HTTP 400 on every tool call.

## [0.16.5] - 2026-04-24

### Fixed
- `install.sh --update` now calls `justsayit init` at the start of the update block so any config files added in a new release (e.g. `after_LLM_filters.json`, `openai-responses.toml`) are created automatically on update.

## [0.16.4] - 2026-04-24

### Added
- **`after_LLM_filters.json`** — new JSON filter file applied after the LLM response, before paste. Same format as `filters.json`. Default rules: em dash → ` - `, en dash → `-`, curly/German quotes → straight quotes, curly single quotes → straight (all enabled); ellipsis → `...` (disabled). `justsayit init` writes the file alongside `filters.json`.

## [0.16.3] - 2026-04-24

### Fixed
- **Clipboard image restore corruption** — after pasting, `Paster` now snapshots image clipboards as raw bytes + MIME type and restores them via `wl-copy --type image/png`. Previously the snapshot decoded PNG binary as UTF-8 and restored it as `text/plain`, causing the next clipboard-context arm to feed raw PNG bytes into the LLM as text context instead of as an image.

### Added
- `openai-responses.toml` deployed by `justsayit init` as the recommended cloud profile (OpenAI Responses API backend, `gpt-5.4-mini`, `reasoning_effort = "low"`).

## [0.16.2] - 2026-04-24

### Added
- **Clipboard images in remote backend** — `/chat/completions` backend now forwards clipboard images to vision-capable models (gpt-4o-mini, gpt-4o, …); `image_detail` profile field works the same as for the Responses backend (`"original"` falls back to `"auto"`).

### Fixed
- **Cross-backend session continuation** — switching backends mid-conversation now preserves the full image history. All backends write `prev_messages` in canonical chat-completions `image_url` format. Remote backend passes history directly to `_build_messages_continued()`; Responses backend converts via `_canonical_to_responses_input()` (maps `image_url`→`input_image`/`output_text`). 2-turn and 3-turn alternating-backend scenarios verified with burn tests.
- **Canonical session storage** — `PostprocessorBase._build_user_history_entry()` is the single source of truth for `prev_messages` entries: includes spoken text, clipboard text, and image for all backends (including local, which stores images for future cross-backend switches even though it can't use them for inference).
- Assistant mode system-prompt note now fires correctly when a clipboard image is sent via the remote backend (`extra_image_provided` was not threaded through `_build_messages`/`_build_messages_continued`).
- Images stored in session history so turn 3+ benefits from prompt caching (same prefix → image tokens cached rather than billed again).

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
