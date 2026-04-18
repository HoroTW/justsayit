# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.9] - 2026-04-18

### Fixed

- Qwen 3.5 0.8B produced multi-page looping "Thinking Process:" output
  instead of a clean transcript. Three compounding causes:
  1. Profile was using `cleanup_local.md`, which opens with Gemma's
     `<|think|>` channel instruction — Qinja doesn't know that
     syntax, so the model saw it as literal system-prompt text and
     got confused by the complex conditional logic.
  2. `paste_strip_regex` was the builtin default (Gemma's
     `<\|channel>thought…<channel\|>`) — doesn't match Qwen's
     `<think>…</think>` format, so thinking blocks bled into the paste.
  3. With thinking enabled, the model looped endlessly on the
     assistant-mode trigger-detection rules in the full prompt.

  Fix: new `cleanup_qwen_simple.md` — a short, cleanup-only prompt
  (no "Hey Computer" assistant mode, no channel instructions) that the
  0.8B model handles reliably. Thinking disabled (`chat_template_kwargs
  = {}`); temperature back to 0.08 (near-greedy is fine without
  thinking). The anti-loop sampling knobs from 0.13.6 (presence_penalty
  1.5, temperature 0.6, etc.) are removed — they were fighting the
  symptom, not the cause.

- `setup-llm qwen3-0.8b` now seeds the correct profile: points at
  `cleanup_qwen_simple.md`, disables thinking, and clears the Gemma
  `paste_strip_regex`. `profile_overrides` in `KNOWN_LLM_MODELS` drives
  this; required fixing `_format_toml_scalar` to handle dict values
  (empty `{}` → `{}` TOML inline table) so dict overrides like
  `chat_template_kwargs = {}` could be upserted into the seeded file.

### Added

- `src/justsayit/prompts/cleanup_qwen_simple.md` — concise cleanup
  prompt for small models that can't reliably follow complex
  conditional logic. Core cleanup rules only: filler words, misheards,
  spoken punctuation. No reasoning channel, no assistant mode.

## [0.13.8] - 2026-04-18

### Fixed

- Remote (OpenAI-compatible) backend crashed with HTTP 400
  (``Unrecognized request argument supplied: chat_template_kwargs``)
  on every request to the real OpenAI API. 0.13.5 introduced
  ``chat_template_kwargs = { enable_thinking = true }`` as the default
  in *both* ``builtin-defaults.toml`` and ``remote-defaults.toml`` on
  the assumption OpenAI would silently drop unknown body fields — it
  doesn't, the API strictly validates the body and 400s. Most other
  hosted OpenAI-compatible providers do the same. The remote default
  is now ``{}``; the builtin default keeps ``{enable_thinking = true}``
  so local Qwen 3.5 still has thinking on out of the box. Self-hosted
  template-aware servers (Ollama, vLLM, SGLang, LM Studio,
  llama.cpp-server) can opt back in per profile.

### Added

- Real-OpenAI burn smoke test (``test_burn_openai_default_profile_round_trips``)
  that goes through ``load_profile`` end-to-end and POSTs to
  ``https://api.openai.com/v1/chat/completions`` with ``gpt-4o-mini``.
  Pairs with ``test_burn_openai_explicit_chat_template_kwargs_400s``,
  which confirms the 400 still happens when the field is opted back
  in — so we'll know if OpenAI ever loosens the validation. Both
  skip cleanly when ``OPENAI_API_KEY`` is unset. This is the test
  that would have caught the 0.13.5 regression on the first run.
- Unit guardrail (``test_remote_default_chat_template_kwargs_is_empty``)
  that loads a minimal ``base = "remote"`` profile via
  ``load_profile`` and asserts the remote default stays empty —
  catches the regression without needing an API key.

## [0.13.7] - 2026-04-18

### Fixed

- Built-in profiles silently disappeared from the tray's LLM submenu
  after running ``setup-llm`` under 0.13.6. Pre-0.13.6
  ``update_profile_model`` couldn't see the commented
  ``# model_path = …`` example in the seeded template, so it
  *appended* a fresh active line at the bottom of every legacy
  profile. 0.13.6's regex finally matched the commented example and
  replaced it — but ``count=1`` left the appended duplicate intact,
  yielding two ``model_path = …`` lines and a TOML duplicate-key
  parse error. The tray's ``except Exception: log.debug`` around
  ``load_profile`` then swallowed the error silently, and the whole
  profile vanished from the menu. ``_set_toml_key`` now upserts with
  full de-duplication (replace first match, drop the rest), so
  re-seeding heals legacy files in place.
- Ships with a regression test that constructs a legacy-shape file
  and asserts both ``tomllib.loads`` and ``load_profile`` succeed
  after ``update_profile_model`` runs again. (The earlier mock-style
  tests only checked the happy-path template, missing the
  legacy-state path that real users actually had on disk.)

## [0.13.6] - 2026-04-18

### Added

- ``justsayit setup-llm`` now bakes Qwen-recommended sampling into
  the Qwen 3.5 0.8B profile at creation time, so users don't land in
  near-greedy defaults that drive the model into thinking loops. The
  seeded ``qwen3-0.8b.toml`` gets ``temperature = 0.6``,
  ``top_p = 0.95``, ``top_k = 20``, ``min_p = 0.0``, and
  ``presence_penalty = 1.5`` — the exact thinking-mode combo from
  Qwen's model card (https://huggingface.co/Qwen/Qwen3.5-0.8B).
  Machinery is generic (``profile_overrides`` dict on each
  ``KNOWN_LLM_MODELS`` entry), so per-model tuning for Qwen 3 4B or
  any future addition is a one-line edit. Existing on-disk profiles
  are not rewritten — back up and delete
  ``~/.config/justsayit/postprocess/qwen3-0.8b.toml`` and re-run
  ``setup-llm qwen3-0.8b`` to pick up the new defaults, or edit the
  file by hand.

## [0.13.5] - 2026-04-18

### Added

- Six new profile fields expose llama-cpp-python's sampling knobs:
  `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`,
  `frequency_penalty`. Motivated by Qwen 3.5 0.8B's documented
  tendency to enter thinking loops — Qwen's own guidance is to raise
  `presence_penalty` to 1.5 (thinking) or 2.0 (non-thinking) as the
  single most effective lever, so it was missing and you couldn't
  work around it short of editing the source. Defaults match
  llama-cpp-python's `create_chat_completion` defaults (no behaviour
  change for existing profiles). `top_p`, `presence_penalty`,
  `frequency_penalty` go over HTTP for the remote backend; `top_k` /
  `min_p` / `repeat_penalty` are llama.cpp-specific and runtime-only
  for the built-in backend.
- New "Looping / repetition on small models" section in
  `docs/postprocessing.md` with the Qwen-recommended knob order
  (temperature away from greedy, then `presence_penalty`) and a
  ready-to-paste Qwen 3.5 thinking-mode override block.

## [0.13.4] - 2026-04-18

### Fixed

- Local LLM load crashed with
  ``LlamaChatCompletionHandlerNotFoundException: Invalid chat handler:
  chat_template.default`` on every GGUF that ships with a bundled
  Jinja chat template (Gemma 4, Qwen 3.5, Llama 3.x, …). The wrapper
  installed in 0.13.3 was skipping the per-Llama ``_chat_handlers``
  dict — where llama-cpp-python stores the GGUF-embedded template
  under the magic name ``chat_template.default`` — and falling
  straight through to the static registry. Now mirrors the same
  three-tier lookup ``Llama.create_chat_completion()`` uses
  internally.

### Added

- New ``burn`` pytest marker for end-to-end tests that load a real
  GGUF via llama-cpp-python. Skipped by default
  (``addopts = "-m 'not burn'"``); run explicitly with
  ``pytest -m burn``. First suite at ``tests/test_burn_postprocess.py``
  exercises ``process()`` against locally cached Gemma 4 and
  Qwen 3.5 0.8B models — both the 0.13.3 crash *and* the earlier
  0.13.0 ``chat_template_kwargs=`` kwarg crash would have tripped on
  the first burn run. The previous mock-only tests accepted arbitrary
  kwargs and resolved no real handlers, so integration bugs kept
  slipping through.

## [0.13.3] - 2026-04-18

### Fixed

- Local (built-in llama-cpp-python) backend crashed with
  ``TypeError: Llama.create_chat_completion() got an unexpected
  keyword argument 'chat_template_kwargs'`` because that method has a
  fixed keyword signature with no ``**kwargs`` passthrough. Switched
  to wrapping the underlying ``chat_handler`` (which *does* accept
  ``**kwargs`` and forwards them into the Jinja chat template), so
  Qwen 3.5's ``enable_thinking`` flag — and any other template
  kwargs — actually reach the template rather than blowing up the
  request. Added a regression test using a strict mock that mirrors
  the real fixed signature; the previous mock accepted arbitrary
  kwargs and let the bug ship.

## [0.13.2] - 2026-04-18

### Fixed

- The commented example lines for `append_to_system_prompt` in
  `profile-gemma4-cleanup.toml` and `profile-openai-cleanup.toml`
  showed `"Always reply in English."` as the assignment value —
  uncommenting that line literally would have silently activated that
  instruction instead of the intended empty default. Moved the example
  into the comment text and made the commented assignment line show
  the actual default (`""`), matching the convention used elsewhere in
  these templates.

## [0.13.1] - 2026-04-18

### Fixed

- The new `chat_template_kwargs` and `append_to_system_prompt` keys
  introduced in 0.13.0 weren't documented in any of the four shipped
  user-facing profile templates (`gemma4-cleanup`, `gemma4-fun`,
  `openai-cleanup`, `ollama-gemma`), so users letting a profile
  re-create from the template never saw the new options. Added
  commented-out example lines for both keys to all four templates.
  (Existing on-disk profiles need to be backed up + deleted +
  re-initialised to pick up the documentation; the runtime defaults
  themselves apply unchanged either way.)

## [0.13.0] - 2026-04-18

### Added

- **`chat_template_kwargs`** (new profile field, default
  `{ enable_thinking = true }`) — inline TOML table forwarded into the
  chat template on both backends. On the built-in backend it reaches
  llama-cpp-python as `chat_template_kwargs=`; on the remote backend
  it's included in the JSON body under the same key. Motivated by
  Qwen 3.5 (all sizes, including the 0.8B) which ships with thinking
  **disabled** by default and needs this flag to turn it on at all.
  The soft switch (`/think` / `/no_think` in the user prompt) was too
  fragile across llama.cpp versions — the chat-template kwarg is the
  reliable route. Default is safe for Gemma (its template ignores the
  flag) and for OpenAI / most hosted providers (they drop unknown body
  fields). Set to `{}` or `{ enable_thinking = false }` to opt out
  per-profile. Documented in `docs/postprocessing.md` under
  *"Thinking mode (Qwen 3.5, Gemma, …)"*.

- **`append_to_system_prompt`** (new profile field, default `""`) —
  extra text glued onto the end of the resolved system prompt
  (separated by a blank line). Makes it trivial to extend a shipped
  prompt with a small addition (e.g. `"Always reply in English."`)
  without forking the whole `.md` file. Sits between the base prompt
  and the `context` block.

## [0.12.1] - 2026-04-18

### Added

- **`paste.restore_delay_ms`** (default `250`) — delay in milliseconds
  between the synthetic paste keystroke firing and the previous
  clipboard being restored. Previously hardcoded at 150 ms; the new
  default of 250 ms is more forgiving of slow Electron / web-based
  apps that read the clipboard a beat late and would otherwise paste
  the restored content. Drop it to `0` if your target app is fast and
  you want the clipboard back sooner. No-op when
  `restore_clipboard = false` or `type_directly = true`.

### Changed

- README and `docs/configuration.md` now have a dedicated
  paste-timing section that calls out the three knobs together
  (`release_delay_ms`, `settle_ms`, `restore_delay_ms`) with
  symptom-by-symptom guidance for unreliable paste. All three stages
  of the paste pipeline are now tunable; previously the third was
  hardcoded.

## [0.12.0] - 2026-04-18

### Changed (BREAKING — see "Migration" below for legacy behaviour)

- **Postprocess defaults are now sourced from canonical TOML files**,
  not duplicated as Python dataclass fields. Two new files under
  `src/justsayit/templates/`:
  - `builtin-defaults.toml` — defaults for the built-in
    `llama-cpp-python` + GGUF backend.
  - `remote-defaults.toml` — defaults for the HTTP / OpenAI-compatible
    backend (works with OpenAI, OpenRouter, Groq, Together, vLLM,
    Ollama, LM Studio, llama.cpp's bundled server, etc.).
  These are THE source of truth: `PostprocessProfile` field defaults
  are derived from `builtin-defaults.toml` at module import time, and
  the user-facing profile templates document them by reference rather
  than embedding values. Editing one of these files is all it takes to
  change a shipped default.

- **Backend selection is now explicit via `base = "builtin" |
  "remote"`** in the user's profile TOML. The loader reads `base`,
  loads the matching defaults file, and overlays the user's overrides
  on top. Routing in `LLMPostprocessor` is now driven by `base` rather
  than an implicit "is `endpoint` set?" check.

- **System prompt selection is decoupled from backend selection**. The
  old auto-swap (`endpoint set + system_prompt unchanged → swap to
  channel-free variant`) is gone — it conflated two orthogonal axes
  and broke the case where a user wanted Gemma's `<|think|>`-channel
  prompt over an HTTP backend (e.g. Ollama serving Gemma). Replaced by
  an explicit `system_prompt_file = "..."` field on the profile. Bare
  filenames resolve against the packaged `src/justsayit/prompts/`
  directory; paths with a slash or `~` are loaded as-is. The inline
  `system_prompt = "..."` field still wins when non-empty.

- **New `ollama-gemma.toml` profile** demonstrates the orthogonality:
  `base = "remote"` + `system_prompt_file = "cleanup_local.md"` +
  `paste_strip_regex = '<\|channel>thought(.*?)<channel\|>'` runs
  Gemma's `<|think|>` cleanup prompt over an HTTP backend talking to a
  local Ollama install. `justsayit init` now writes four shipped
  profiles (`gemma4-cleanup`, `gemma4-fun`, `openai-cleanup`,
  `ollama-gemma`); `ensure_default_profiles()` returns a 4-tuple.

- The shipped profile templates no longer embed the default system
  prompt as a 50-line commented block. They reference it by file name
  (`system_prompt_file = "cleanup_local.md"` etc.) and document the
  override pattern in two short comment lines instead.

### Removed

- `_DEFAULT_SYSTEM_PROMPT`, `_REMOTE_CLEANUP_SYSTEM_PROMPT`,
  `_FUN_SYSTEM_PROMPT` module constants — the prompts now live only on
  disk under `src/justsayit/prompts/` and are loaded on demand by
  `_resolve_system_prompt_file()`.
- `_comment_block` helper — no longer needed; nothing embeds the
  prompt as commented documentation.
- The `endpoint`-triggered system-prompt auto-swap in
  `LLMPostprocessor._system_prompt`.

### Migration

- Existing user profiles without an explicit `base` field but with
  `endpoint` set are auto-treated as `base = "remote"`, so legacy
  setups keep working without intervention. `load_profile()` logs
  nothing in this case — it's silent backward-compat.
- Legacy fully-populated profile files (no commented-defaults marker)
  get backed up + rewritten on the next `init` / `setup-llm`, same as
  before. Once the backup is taken, the new shape is in place.

## [0.11.16] - 2026-04-18

### Changed

- Inlined the `commented-defaults form` marker literal directly into
  the three profile templates (`profile-gemma4-cleanup.toml`,
  `profile-gemma4-fun.toml`, `profile-openai-cleanup.toml`) instead of
  injecting it via a `{{COMMENTED_FORM_MARKER}}` placeholder. The
  marker never varies, so the substitution machinery added no value.
  No behavioral change — generated files remain byte-identical.

## [0.11.15] - 2026-04-18

### Changed

- Simplified the postprocess template loader: `_load_template` is now a
  thin `read_text` wrapper, and the three profile-TOML templates do
  their `{{NAME}}` substitution inline via plain `.replace()` chains.
  The two static templates (`context-sidecar.toml`,
  `dynamic-context.sh`) no longer pretend to take substitutions they
  never had. No behavioral change — generated files remain
  byte-identical.

## [0.11.14] - 2026-04-17

### Changed

- Loosened the shipped local and OpenAI-compatible cleanup prompts so
  `Hey Computer` anywhere in a transcript is generally treated as an
  assistant cue, while clearly quoted, reported, incidental, or
  otherwise nonsensical uses should still stay cleanup-only.

- Refactored `postprocess.py`: the three TOML profile templates
  (`gemma4-cleanup`, `gemma4-fun`, `openai-cleanup`), the
  context-sidecar template, and the `dynamic-context.sh` script now
  live as standalone files under `src/justsayit/templates/` instead of
  inline Python f-strings. A new `_load_template(name, **subst)` helper
  reads them from disk and substitutes `{{NAME}}` markers with literal
  `str.replace`, so braces in the template body (e.g. `{text}` in
  commented-out `user_template` examples) pass through unchanged.
  No behavioral change — generated profile files are byte-identical.

### Documentation

- Updated README, `docs/postprocessing.md`, and
  `docs/configuration.md` to describe the shipped `Hey Computer`
  behavior as best-effort prompt semantics rather than deterministic app
  logic.

## [0.11.13] - 2026-04-18

### Added

- `killall justsayit` (and `pgrep justsayit` without `-f`, `htop`,
  `top`, etc.) now actually find the running process. The installed
  `justsayit` entry-point is a tiny Python shim, which means the
  kernel's `comm` field for the running process was `python3` — so any
  tool that matches by short name (`killall`, `pgrep` without `-f`)
  saw nothing. `cli.main()` now calls `prctl(PR_SET_NAME, "justsayit")`
  via `ctypes` at startup so the comm field is correct. Linux-only,
  no new dependencies. Has to run from `main()` (not module-import
  time) because the two re-execs at the top of `cli.py`
  (systemd-scope wrapping + LD_PRELOAD for gtk4-layer-shell) reset
  comm back to `python3`; setting it in `main()` runs after both.

### Fixed

- Stale assertion in `test_default_overlay_fields` (commit `89c4f65`
  changed `OverlayConfig.visualizer_sensitivity` 1.0 → 2.5 and
  `opacity` 0.78 → 0.7 but didn't update the test). Test now matches
  the shipped defaults.

## [0.11.12] - 2026-04-18

### Fixed

- The shipped local Gemma cleanup prompt previously instructed the
  model: *"If nothing needs changing, just write `No changes.` and
  stop."* (commit 984c78b). That instruction lived inside the
  `<|think|>` channel block, but Gemma sometimes treated `No changes.`
  as the visible reply — so the user pasted a literal `No changes.`
  into their document instead of getting their text back. Removed the
  shortcut entirely. The prompt now states explicitly (mirroring the
  remote/OpenAI variant) that the visible reply is ALWAYS the
  transcript itself — cleaned where edits apply, otherwise verbatim —
  and lists `No changes.` / `Already clean.` / `OK.` as forbidden
  meta-strings. The `# Output` section repeats the same backstop.

### Changed

- Loosened the `<|think|>` reasoning constraint in the local Gemma
  prompt. Commit 984c78b had tightened it to *"at most ONE short
  sentence (≤ 15 words)"* to fight bloated chain-of-thought, but the
  cap was too aggressive — local Gemma cleanup quality benefits from
  the model thinking through tricky inputs (mishears, modal particles,
  punctuation collisions, trigger-or-not decisions). The channel is
  now described as a working space where focused, multi-sentence
  thinking is fine for tricky input, with the still-firm rules being
  "no whole-input echo, no per-word enumeration." Bloat is bounded by
  intent, not a hard word count.

## [0.11.11] - 2026-04-18

### Fixed

- Hardened both shipped cleanup prompts against Gemma 3 answering bare
  questions like `Wie viel Uhr ist es gerade?` instead of cleaning them
  up. The Assistant-mode block now states explicitly that a bare
  question (without the literal word `Computer` somewhere in the
  transcript) is NEVER a trigger, and the examples list adds German
  + English question counter-examples (`Wie viel Uhr ist es gerade?`,
  `Was meinst du dazu?`, `What time is it?`, `Kannst du mir das Salz
  reichen?`). The local-Gemma prompt also tells the model not to use
  the `<|think|>` channel to deliberate "is the user asking me?" — that
  deliberation is what was talking the model into responding.

### Changed

- Extracted the three shipped system prompts (local Gemma cleanup,
  OpenAI-compatible cleanup, fun) out of inline Python triple-quoted
  strings in `postprocess.py` and into standalone Markdown files under
  `src/justsayit/prompts/`. The Python module now loads them via a
  small `_load_prompt()` helper at import time. Same prompt text, just
  much easier to read, diff, and edit. Hatchling already includes the
  whole package directory in the wheel, so no packaging changes are
  needed.

### Tests

- Added a parametrized regression test pinning the bare-question rule
  and the German + English question counter-examples in both shipped
  prompts.

## [0.11.10] - 2026-04-17

### Fixed

- Hardened both shipped cleanup prompts (local Gemma 3 and the
  OpenAI-compatible variant) against bare-`Hey` misfires. Gemma was
  fuzzy-matching leading interjections like `Hey, ich habe gesehen, …`
  to `Hey Computer` and switching into assistant mode on plain
  dictation. The prompts now state a HARD REQUIREMENT that the literal
  word `Computer` must be present, explicitly name common bare
  greetings (`Hey` / `Hi` / `Hallo` / `Hej` / `Hallöchen` / `Yo` /
  `Servus`) as never being triggers on their own, and surface the
  in-the-wild German failure case as the most prominent counter-example.

### Tests

- Added a parametrized regression test covering both shipped prompts to
  pin the HARD REQUIREMENT wording and the bare-greeting
  counter-examples in place.

## [0.11.3] - 2026-04-17

### Changed

- Reverted the code-level trailing `Hey Computer` input rewrite in
  `postprocess.py`. Leading `Hey Computer` requests still work unchanged.
- The shipped local and OpenAI-compatible cleanup prompts now describe a
  conservative trailing rewrite/edit convention as prompt-guided
  best-effort behavior instead.

### Documentation

- README, `docs/postprocessing.md`, and `docs/configuration.md` now show
  both rewrite-style and composition-style leading `Hey Computer` usage,
  and explain the conservative trailing convention as prompt/docs
  behavior rather than deterministic app logic.

### Tests

- Removed postprocess tests for the reverted code-level trailing `Hey
  Computer` normalisation path.

## [0.11.2] - 2026-04-17

### Documentation

- Surface "Hey Computer" inline assistant mode as a top-level feature
  in the README (it had been demoted to a sub-bullet of "LLM cleanup"
  and was buried — turns out it's one of the most useful tricks).
- New `## "Hey Computer" — inline assistant mode` section in
  postprocessing.md with the trigger rules and worked examples
  (math, translation, code one-liners, polite decline …).
- Cross-reference from the LLM section of configuration.md.

## [0.11.1] - 2026-04-17

### Documentation

- README feature list expanded to surface previously undocumented
  capabilities: alternative ASR backends (faster-whisper, OpenAI-
  compatible Whisper endpoint), OpenAI-compatible LLM endpoint,
  customisable system prompts (emojify / translate / summarise / your
  own style), system tray with profile switcher, personal-context
  sidecar, notification sounds, privacy paste options.
- `docs/configuration.md` rewritten as an index of all config files
  with new sections for ASR backends, `.env` / API keys, system tray,
  sounds, paste privacy options, and logging. The LLM section is now a
  pointer.
- New `docs/postprocessing.md` covering the three shipped profiles
  (gemma4-cleanup, gemma4-fun, openai-cleanup), OpenAI-compatible
  endpoint setup, personal-context sidecar, and worked examples for
  custom profiles (emojify, translate, summarise, formal-email tone).

## [0.11.0] - 2026-04-17

### Added

- Ship a third default postprocess profile, `openai-cleanup.toml`,
  alongside `gemma4-cleanup.toml` and `gemma4-fun.toml`. Same
  commented-defaults convention, but with `endpoint`
  (`https://api.openai.com/v1`) and `model` (`gpt-4o-mini`) uncommented
  as the keys that DEFINE the OpenAI-compatible variant. The cleanup
  prompt stays commented so it tracks the dataclass default — which
  auto-swaps to the channel-free `_REMOTE_CLEANUP_SYSTEM_PROMPT` when
  `endpoint` is set. Discoverable from the tray's LLM submenu after
  `init`; users only need to drop their key into
  `~/.config/justsayit/.env` (or export `OPENAI_API_KEY`) to use it.
- New helper `ensure_openai_profile()` and a third entry in the
  `ensure_default_profiles()` tuple (now `(cleanup, fun, openai)`).

## [0.10.3] - 2026-04-17

### Changed

- Reframed the blank-line preservation rule as a positive `KEEP every
  newline and blank line` instruction (own paragraph, prominent), since
  the previous negative bullet in the DO NOT list was being ignored —
  models followed positive directives more reliably.

## [0.10.2] - 2026-04-17

### Changed

- Default cleanup prompts (local Gemma + remote variant) now explicitly
  forbid removing existing blank lines or collapsing whitespace —
  models were occasionally flattening multi-paragraph dictations.

## [0.10.1] - 2026-04-17

### Fixed

- Remote OpenAI-compatible LLM endpoints no longer reply with the
  literal string `No changes.` (or leak `<|channel>thought…` reasoning).
  The shipped default cleanup prompt relies on Gemma's `<|think|>`
  channel to hide reasoning from the visible reply; generic models
  (OpenAI / OpenRouter / Groq / vLLM / …) have no such channel and
  interpreted "If nothing needs changing, just write `No changes.` and
  stop" as a literal output instruction. When `profile.endpoint` is set
  AND the user hasn't customised `system_prompt`, justsayit now
  auto-swaps in `_REMOTE_CLEANUP_SYSTEM_PROMPT` — same cleanup rules
  and spoken-punctuation table, no Gemma-specific channel scaffolding,
  and an explicit "echo the input verbatim if nothing needs changing"
  rule. Custom `system_prompt` values are passed through untouched.

## [0.10.0] - 2026-04-17

### Added

- **OpenAI-compatible LLM endpoint** for postprocessing. Set
  ``endpoint``, ``model``, and an API key on a `PostprocessProfile`
  and the cleanup call goes over HTTP instead of loading a local GGUF
  via llama-cpp-python. Compatible with OpenAI, OpenRouter, Groq,
  Together, vLLM, Ollama (`/v1`), LM Studio, llama.cpp's bundled
  server, and anything else that speaks the chat-completions schema.
  No new dependencies — pure stdlib `urllib`.
- **OpenAI-compatible Whisper STT backend**. New
  ``model.backend = "openai"`` value plus ``model.openai_endpoint /
  openai_model / openai_api_key / openai_api_key_env / openai_language
  / openai_timeout``. Captured audio is encoded as 16-bit PCM WAV
  in-memory and posted as multipart form to ``/audio/transcriptions``.
  Local Parakeet/Whisper downloads are skipped when this backend is
  selected (only the tiny Silero VAD ONNX is fetched, since we don't
  want to stream audio to the network just to detect silence).
- **Shared `.env` file** at ``$XDG_CONFIG_HOME/justsayit/.env`` for
  API keys. Same `KEY=VALUE` format as `python-dotenv`, with optional
  matched quotes and a leading `export ` for shell parity. Process
  env wins when a value is defined in both places. Loaded lazily on
  first ``resolve_secret`` call so test isolation stays clean.
- Three places to source secrets, in priority order: explicit literal
  in the config / profile → process env (``api_key_env``) → `.env`
  file. Both the LLM profile and the OpenAI Whisper backend use the
  same resolver.
- Inline documentation block at the bottom of `gemma4-cleanup.toml`
  showing the new endpoint fields and explaining the `.env`
  precedence — discoverable without leaving the file.

### Changed

- ``justsayit init`` (and the post-install model fetch) now
  short-circuits the Parakeet/Whisper download path when
  ``model.backend == "openai"``, printing the configured endpoint
  instead of pretending there is something local to fetch.
- ``LLMPostprocessor.warmup()`` is a no-op on the remote path — there
  is no local model to load and a probe request would burn quota.

### Internal

- New ``justsayit/transcribe_openai.py`` with
  ``OpenAIWhisperTranscriber`` plus reusable ``_encode_wav`` and
  ``_build_multipart`` helpers.
- 26 new tests cover the `.env` loader (precedence, quote stripping,
  process-env wins), the LLM remote-process path (request shape,
  auth header, missing-key error, empty-response fallback,
  warmup no-op), and the OpenAI Whisper backend (WAV round-trip,
  multipart structure, language hint, JSON + plain-text response
  parsing, empty-buffer short-circuit, missing-key error,
  ``make_transcriber`` dispatch).

## [0.9.1] - 2026-04-17

### Fixed

- **Tray LLM submenu silently dropped most profiles** because the
  shipped `gemma4-cleanup.toml` template generated invalid TOML.
  The commented-defaults template embedded the multi-line default
  system prompt via an f-string ``# {_DEFAULT_SYSTEM_PROMPT}"""``,
  which only commented out the *first* line — every subsequent
  line of the prompt landed at column 1 and broke `tomllib.loads`.
  The tray's `load_profile` swallowed the parse error and skipped
  the profile, so users only saw the one profile (`gemma4-fun`)
  whose template happened to be well-formed. The same template is
  reused for `qwen3-*` profiles, so `setup-llm` users with multiple
  models saw only their `gemma4-fun` entry.
  - Added `_comment_block` helper in `postprocess.py` that prefixes
    every embedded line with `# ` so multi-line defaults can be
    safely included in a commented-form file.
  - `ensure_commented_form_file` now accepts an optional `validator`
    callable. `ensure_default_profile` / `ensure_fun_profile` pass
    `tomllib.loads`, so any profile that bears the commented-form
    marker but fails TOML parsing (i.e. was written by the buggy
    template) is automatically backed up to `.bak-pre-commented-form`
    and rewritten with the fixed template on next `init` /
    `setup-llm` run. No manual intervention required.
  - Regression test: `test_shipped_profile_templates_parse_as_valid_toml`
    asserts `tomllib.loads(_CLEANUP_PROFILE_TOML)` and the fun
    template both succeed, so this can never re-ship.

## [0.9.0] - 2026-04-17

### Changed

- **`config.toml` and the shipped postprocess profile TOMLs now use a
  "commented-defaults" form**. Every key in the shipped file is the
  default value, commented out. The user uncomments + edits only the
  knobs they actually want to override; everything else tracks the
  shipped default automatically. Defaults can drift in future releases
  without ever colliding with user overrides — the only thing that
  changes between versions is the value embedded in the comment.
- **One-shot migration**: pre-existing `config.toml` and profile TOMLs
  in the legacy fully-populated form (every key uncommented) are
  backed up to `<file>.bak-pre-commented-form` and rewritten in the
  new commented form on next app start / install. A header marker line
  embedded in the new templates makes the migration idempotent — once
  a file carries the marker, subsequent runs leave it alone (so the
  user's later overrides are preserved).
- **`gemma4-fun.toml` template trimmed to its actual overrides**. Only
  the three keys that define the "fun" flavor (`system_prompt`,
  `temperature`, `paste_strip_regex`) stay uncommented; the rest fall
  through to the dataclass defaults. Demonstrates the commented-form
  convention with a real override pattern.

### Removed

- **Defaults-baseline machinery is gone**. The old `.baseline/<name>`
  sidecar (and its pre-0.8.7 `<stem>.defaults-baseline.<ext>` fallback)
  earned its keep when `config.toml` was fully-populated and we needed
  3-way reconciliation to tell "you never customised" apart from "you
  customised". With commented-defaults form, comments and overrides
  live in different layers of the file and never collide, so the
  baseline became vestigial. Removed: `defaults_baseline_path`,
  `_legacy_baseline_path`, `_migrate_legacy_baseline`,
  `_write_baseline`, `_heal_baseline`, `write_or_heal_baseline` from
  `config.py`; `baseline_path_for` and the 5-case reconcile from
  `install.sh`. Existing `.baseline/` directories under
  `~/.config/justsayit/` are now orphan and can be deleted by hand
  (they are not referenced by anything anymore).
- **`maybe_update_user_file` reduced to a simple "diff & prompt"**.
  Previously a 5-case state machine (in-sync / never-customised /
  defaults-static / both-diverged / no-baseline). Now: read user file,
  compare to shipped, if different show the diff and prompt with
  default `Y`. Used only for `filters.json` (JSON has no commented-
  defaults form, so it's the only file left that needs reconciling).

## [0.8.13] - 2026-04-17

### Changed

- **`setup-llm` activation hint no longer points at `config.toml`**.
  After downloading models, setup-llm used to print
  `To activate, set in config.toml: [postprocess] enabled = true,
  profile = "<name>"`. Both halves were wrong: those keys are runtime
  state and live in `state.toml` (since 0.8.8), and steering users at
  hand-editing a state file is the wrong default — the tray's LLM
  submenu toggles `enabled` and selects `profile` for you, writing to
  `state.toml` correctly. The hint now lists the available profile
  names and tells users to pick one from the tray menu.

## [0.8.12] - 2026-04-17

### Fixed

- **`setup-llm gemma4` no longer creates a redundant `gemma4.toml`**.
  Previously, selecting the gemma4 model wrote a third profile file
  alongside the two we ship (`gemma4-cleanup.toml`,
  `gemma4-fun.toml`), copying the cleanup template and patching in the
  model path. The third file then had to be activated via
  `profile = "gemma4"` (which the suggestion line printed), even though
  the shipped profiles already bind the same model and have proper
  filenames. Now `setup-llm gemma4` ensures the two shipped profiles
  exist, patches their `model_path` to the actual download location,
  and prints `profile = "gemma4-cleanup"` / `"gemma4-fun"` as the
  activation hints. Other model keys (`qwen3-4b`, `qwen3-0.8b`)
  continue to get a single per-key profile file as before, since no
  shipped variants exist for them.

## [0.8.11] - 2026-04-17

### Fixed

- **Hotkey re-bind dialog after restarting from the tray (CLI launches
  only)**. Symptom: launching from a terminal bound the global shortcut
  fine, but selecting "Reload config" from the tray triggered a restart
  that came back with `shortcut toggle-dictation -> (unassigned)` and
  popped the portal bind dialog again. Launches via the installed
  `.desktop` file were unaffected. Diagnosis: the restart used in-place
  `os.execve`, preserving the self-managed `app-<app_id>-<pid>.scope`
  cgroup and starting a fresh D-Bus connection inside it. KDE's portal
  v1 (which doesn't support `ConfigureShortcuts`) couldn't match the new
  connection back to the prior binding under that scope name. Fix: the
  restart path now first tries `Gio.DesktopAppInfo.launch` against the
  installed `dev.horotw.justsayit.desktop`, so the desktop env owns the
  scope naming and the portal recognizes the app id consistently across
  launches. Falls back to in-place `execve` in dev mode where no
  `.desktop` is installed.

## [0.8.10] - 2026-04-17

### Fixed

- **`install.sh --update` no longer destroys the venv** (and with it
  manually-installed `llama-cpp-python`). Previously every `--update`
  unconditionally ran `uv venv --system-site-packages "$VENV_DIR"`,
  which prompted "replace existing venv? [y/N]" — and if the user
  said yes, the venv (including `llama-cpp-python` built locally with
  `CMAKE_ARGS=-DGGML_VULKAN=1` for Vulkan GPU) was nuked. The
  reinstall step then only restored `pyproject.toml` extras, so
  `llama_cpp` was missing on next launch and the first dictation
  crashed with `ModuleNotFoundError`. install.sh now reuses an
  existing valid venv (the `uv pip install -e` upgrade still pulls
  in dep refreshes) and only creates fresh when none exists.

### Added

- **`--update` defense-in-depth check**: at the end of an update,
  if `llama_cpp` isn't importable from the venv but the user's
  `state.toml` or `config.toml` has `[postprocess] enabled = true`,
  print a clear warning telling them to run
  `./install.sh --update --postprocess`. Catches users who already
  hit the 0.8.9-and-earlier venv-nuke bug and need a one-line
  recovery hint instead of debugging a `ModuleNotFoundError` from
  the app log. Awk-based check (no Python TOML parser needed) —
  triggers only on the `[postprocess]` section's `enabled` line, not
  on `[vad] enabled = true`.

## [0.8.9] - 2026-04-17

### Fixed

- **Postprocess profile TOMLs (`gemma4-cleanup.toml`,
  `gemma4-fun.toml`) now get a `.baseline/` snapshot on first write
  and auto-heal on app start**, matching the behaviour
  `config.toml` and `filters.json` already had. Previously
  `ensure_default_profile()` / `ensure_fun_profile()` (postprocess.py)
  just wrote the file and never touched the baseline — so install.sh
  `--update` always landed in Case 5 (pre-baseline diff prompt) for
  profile TOMLs even on fresh installs, instead of Case 1 (in sync,
  no-op) or Case 2 (stale defaults, friendly `[Y/n]`). The path was
  always correct (`postprocess/.baseline/<name>.toml` after 0.8.7);
  the file just never got written from the Python side.
- Refactored the write-and-snapshot pattern into a single public
  helper `write_or_heal_baseline(user_path, current_defaults, *,
  just_written)` in `config.py` so the four `ensure_*` sites
  (config, filters, cleanup-profile, fun-profile) share one
  implementation.

### Added

- Four new tests covering profile baseline write-on-fresh-install,
  heal-on-sync, no-heal-on-customised, and the parallel for the fun
  profile. 165 tests total (was 161).

## [0.8.8] - 2026-04-17

### Changed

- **Runtime state split out into `state.toml`.** Previously the app
  toggling auto-listen, postprocess on/off, or the active profile
  caused `save_config()` to re-load `config.toml`, overlay the three
  runtime-mutable fields (`vad.enabled`, `postprocess.enabled`,
  `postprocess.profile`), and rewrite the **whole** file via
  `render_config_toml()` — which nuked any inline comments the user
  had written in `config.toml` and dropped any unknown fields.

  Now those three fields persist to `~/.config/justsayit/state.toml`
  instead, and `load_config()` overlays state on top after reading
  `config.toml`. State wins. The user's `config.toml` is never
  touched by the app — comments and customisations there survive
  forever. Same separation pattern we already used for `context.toml`
  (pure user data) and the `.baseline/` snapshots (internal
  bookkeeping).

  Migration is implicit: existing `vad.enabled = true` / `profile =
  "gemma4"` entries in your authored `config.toml` keep working as
  the "initial state". The first time you toggle anything, a
  `state.toml` appears and overlays from then on. Delete `state.toml`
  to reset to whatever your `config.toml` says. `save_config()` is
  now a thin back-compat wrapper around the new `save_state()`; the
  CLI log message changed from "persisted auto-listen=… to
  config.toml" to "to state.toml" to reflect reality.

  `install.sh --update` already left `config.toml` alone (since
  0.8.6); the notice now also mentions that `state.toml` is
  app-managed runtime state and not reconciled either.

## [0.8.7] - 2026-04-17

### Changed

- **Defaults-baseline snapshots moved to a hidden `.baseline/` subdir
  per directory.** Previously each baseline lived next to its user
  file as `<stem>.defaults-baseline.<ext>`, cluttering the visible
  config tree (a user with config + filters + two profiles ended up
  with four sidecars next to four real files). Now:
  - `~/.config/justsayit/filters.json` → `~/.config/justsayit/.baseline/filters.json`
  - `~/.config/justsayit/postprocess/gemma4-cleanup.toml` → `~/.config/justsayit/postprocess/.baseline/gemma4-cleanup.toml`
  - (config.toml is no longer reconciled, so its baseline is just
    catch-up state from earlier `init` runs.)

  Migration is automatic and lazy: both the Python helpers
  (`_write_baseline`, `_heal_baseline`) and the shell
  `baseline_path_for()` move legacy sidecars into the new layout on
  first encounter (next app start OR next `install.sh --update`,
  whichever runs first). Users who never re-run install.sh get
  migrated on app launch; users who never launch the app get
  migrated on `--update`. Idempotent — collisions remove the legacy
  copy. Best-effort: if `mkdir`/`mv` fails, install.sh degrades to
  the existing "no baseline → plain diff prompt" path.

## [0.8.6] - 2026-04-17

### Changed

- **`install.sh --update` no longer reconciles `config.toml`.** It's
  a settings file (per-user choices like `postprocess.enabled` and
  `postprocess.profile`), not a shipped template. Treating it as a
  template meant every update produced an "always" conflict prompt —
  e.g. a client running `enabled = true, profile = "gemma4"` was
  prompted to overwrite with the shipped `enabled = false, profile =
  "gemma4-cleanup"`, which would have silently disabled their LLM
  postprocess. New config keys we add ship with sensible dataclass
  defaults in `config.py`, so old user configs silently pick them up
  on next load — no overwrite needed. Power users can diff against
  shipped defaults with `justsayit show-defaults config`. The
  `filters.json` and profile TOML reconciles are unchanged (those
  are templates the user is meant to either accept or fork).

## [0.8.5] - 2026-04-17

### Fixed

- **`install.sh --update` was silently looking in the wrong config
  directory** and skipping every reconcile prompt. Python's
  `config_dir()` uses platformdirs with `APP_NAME = "justsayit"`
  (→ `~/.config/justsayit/`), but the shell script was building paths
  from the `.desktop` `APP_ID = "dev.horotw.justsayit"`
  (→ `~/.config/dev.horotw.justsayit/`, which doesn't exist). Result:
  `maybe_update_user_file` hit `[ -f "$_USER_FILE" ] || return 0` for
  every file and returned silently — clients running `--update` never
  saw the diff prompt for `filters.json`, `config.toml`, or the
  profile TOMLs, no matter how stale they were. The same bug also
  made `init` re-run on every `--update` (the wrong-path existence
  check was always true), but `init` is internally idempotent so it
  just printed "config already exists" and moved on. Added a
  `CONFIG_DIR_NAME="justsayit"` variable and use it for both the
  init-gate check (line ~196) and `_CFG_HOME` (line ~395). The
  reverse-DNS `APP_ID` is still used (correctly) for the `.desktop`
  filename and `StartupWMClass`.

## [0.8.4] - 2026-04-17

### Changed

- **`install.sh --update` now defaults to "yes" when offering to
  replace a stale user config file** (Cases 4 and 5 — the customised
  3-way drift, and the pre-baseline migration where we can't tell new
  defaults from user edits). Both prompts flip from `[y/N]` to
  `[Y/n]`, and the non-interactive default flips from `n` to `y`.
  Rationale: shipped defaults exist to be used (the assistant-trigger
  tightening, filter-chain improvements, profile-prompt updates), and
  the previous file is always saved as `.bak.<ts>` so users can
  re-apply any customisations from there. Stale-defaults users were
  silently keeping outdated files because Enter at the prompt meant
  "no". Case 2 (never customised, just stale defaults) was already
  `[Y/n]` and is unchanged.

## [0.8.3] - 2026-04-17

### Changed

- **Tightened the assistant-mode trigger** in the default
  `gemma4-cleanup` system prompt. Assistant mode now only activates when
  the transcript STARTS with `Hey Computer` (case-insensitive, with
  tolerance for obvious STT mishears like `Hi Computer` /
  `Hey Computa`). A bare leading `Computer`, mid-sentence `hey
  computer`, or quoted/reported uses no longer trigger a reply — they
  are passed through as ordinary cleanup. This stops the assistant from
  jumping in on dictated text that merely contains the word "Computer".
  Existing user-customised profile TOMLs are not modified;
  `install.sh --update` will offer the new prompt as a diff over the
  previous shipped default.

## [0.8.2] - 2026-04-17

### Added

- **Personal-context sidecar:** `~/.config/justsayit/context.toml`, a
  TOML file with a single `context = "..."` field appended to every
  postprocess profile's system prompt under a "User context" heading on
  every dictation. Lives separately from profile TOMLs so updates to
  shipped profiles (system prompt, model paths, regexes) never wipe
  user-written personal context. Comments in the file are not sent to
  the LLM — only the string value of `context` is. Created on first
  `init` (or first profile setup) with a documented empty template; the
  install update flow never touches it (it's pure user data, not
  shipped defaults).
- **`install.sh --update` now reconciles postprocess profile TOMLs**
  too (`gemma4-cleanup.toml` and `gemma4-fun.toml`), using the same
  baseline-aware diff/.bak machinery as `config.toml` and
  `filters.json`. Shipped-default changes (system prompt tweaks, model
  bumps, paste-strip regex updates) are surfaced cleanly without
  clobbering user customisations.
- **`show-defaults` subcommand** gained three new kinds: `context`
  (sidecar template), `profile-cleanup`, and `profile-fun` (the two
  shipped profile TOML templates). Used by `install.sh --update` to
  diff against the user's on-disk copies.

### Changed

- **Profile-level `context = "..."` is now optional** in profile TOMLs.
  When missing or empty, `load_profile()` falls back to the shared
  `context.toml` sidecar. A non-empty profile-level `context` still
  wins, preserving backward compatibility for users who already have
  context inline in their profile.
- New shipped profile templates (`gemma4-cleanup.toml`,
  `gemma4-fun.toml`) drop the `context = ""` field and instead carry a
  short comment pointing at the sidecar. Existing user profiles with
  `context = "..."` are not auto-migrated — they keep working as-is.

## [0.8.1] - 2026-04-17

### Changed

- **`install.sh --update` now uses a defaults-baseline sidecar** to tell
  three previously-indistinguishable cases apart on every update:
  *stale shipped defaults the user never customised* (safe to replace
  with `[Y/n]` default-yes), *user customisations against unchanged
  defaults* (silent no-op), and *true 3-way drift* (shows two diffs —
  what changed in shipped defaults, what the user customised — and
  prompts `[y/N]` default-no since accept is destructive). Without the
  baseline, the previous flow always showed the same scary "do you want
  to lose all this?" diff, which made users decline even when they
  hadn't customised anything.
- **Sidecar file convention:** `filters.json` →
  `filters.defaults-baseline.json` (and same for `config.toml`), written
  by `ensure_filters_file()` / `ensure_config_file()` on first run and
  refreshed by `install.sh` whenever the user accepts an overwrite.
  `defaults_baseline_path()` in `config.py` is the source of truth;
  `install.sh`'s `baseline_path_for()` derives the same path with shell
  parameter expansion.
- **Self-healing migration for pre-baseline installs:** on app startup,
  if the user file matches current shipped defaults verbatim and no
  baseline exists yet, one is snapshotted silently. Customised
  pre-baseline installs degrade to a plain diff prompt for one update
  cycle, then the baseline is established on accept/decline.
- Best-effort everywhere: missing or unreadable baselines fall back to
  the plain diff prompt — install.sh never refuses to run because
  baseline state is bad.

## [0.8.0] - 2026-04-17

### Added

- **GitHub update check on startup.** A best-effort background fetch
  reads the latest GitHub release metadata, parses the release tag
  (`v0.11.6` or `0.11.6`), and compares against the running `__version__`.
  When a newer
  version is available:
  - the overlay shows a small yellow `update available` badge to the
    left of the × button (tooltip carries the new version number);
  - a low-priority desktop notification fires once per launch (stable
    notification id, so re-checks update one entry rather than spamming);
  - if the running install is a git checkout (detected by `install.sh`
    + `.git` next to the package source), the notification body tells
    the user how to update: `cd <install dir> && ./install.sh --update`.
  Result is cached in `~/.cache/justsayit/update_check.json` for 3h so
  repeated launches don't hammer the API. Network errors / malformed
  responses are silently ignored — startup never blocks.
- **`install.sh --update`** mode: pulls latest commits (fast-forward
  only), refreshes the venv + dependencies, refreshes the `.desktop`
  entry, and interactively offers to replace `config.toml` /
  `filters.json` with the freshly-rendered shipped defaults — current
  files are backed up to `*.bak.<timestamp>` before being overwritten.
  Implies `--skip-models` and skips the postprocess prompt (already
  configured on the original install).
- **`justsayit show-defaults config|filters`** subcommand prints the
  current shipped defaults to stdout. Used by `install.sh --update` to
  diff against the user's file and offer the update prompt.

### Fixed

- `justsayit init` now writes the new spoken-punctuation filter chain
  introduced in 0.7.2. Was still writing the stale two-rule starter
  list because of a separate hardcoded copy in `cli.py`. The full
  chain is now sourced from `_default_filter_chain()` in `config.py`,
  so `init` and `ensure_filters_file` can never drift again.

## [0.7.2] - 2026-04-17

### Added

- **Spoken-punctuation regex filters shipped as `filters.json`
  defaults**, so the dictation flow handles `Punkt` / `Komma` /
  `Fragezeichen` / `Ausrufezeichen` / `Doppelpunkt` / `Semikolon` /
  `neue Zeile` / `neuer Absatz` (and English equivalents incl.
  `full stop`) without needing the LLM postprocess step at all. The
  LLM is now a backup, not the primary line of defence — `justsayit`
  with `postprocess.enabled = false` produces clean text on its own.
  Each spoken word ships as a pair: a "drop redundant" rule that
  removes the word silently when the STT already wrote the matching
  character, plus a "replace" rule for the standalone case. Cleanup
  rules drop punctuation-only lines, leading punctuation after a
  forced newline, and trailing whitespace; `collapse spaces` now uses
  `[ \t]{2,}` so newlines from `neue Zeile` survive. The headline
  failure case (`Hallo, neue Zeile. Ich komme nicht. Punkt. Neue
  Zeile, eure Katja.`) is covered by an end-to-end test.

### Changed

- **Default `filters.json` chain replaced.** The old two-rule starter
  (trim + collapse-whitespace) is preserved (with collapse fixed to
  preserve newlines) but now sits at the end of a much richer chain.
  Existing users keep their `filters.json` untouched — delete it to
  regenerate with the new defaults, or copy individual rules in by
  hand.

## [0.7.1] - 2026-04-17

### Changed

- **Default cleanup prompt now has a dedicated "Spoken punctuation /
  line-break words" section** with explicit mappings (`Punkt` → `.`,
  `Komma` → `,`, `Fragezeichen` → `?`, `Ausrufezeichen` → `!`,
  `Doppelpunkt` → `:`, `Semikolon` → `;`, `neue Zeile` → newline,
  `neuer Absatz` → blank line) and a CRITICAL rule: drop the spoken
  word silently if the STT already produced the character or inserting
  it would leave a stray symbol on its own line. The exact failure
  case that motivated the fix (`Hallo, neue Zeile. Ich komme nicht.
  Punkt. Neue Zeile, eure Katja.` rendering with a stray `.` on its
  own line between sentences) is baked in as the headline example.
- **`<|think|>` constraint tightened hard** — was "very brief (under
  3 sentences)", now "INTERNAL reasoning ONLY — at most ONE short
  sentence (≤ 15 words). NEVER echo the input, list filler/mishear/
  formatting checks, enumerate corrections, or show step-by-step
  work. If nothing needs changing, just write `No changes.` and stop."
  Stops Gemma from emitting multi-paragraph chain-of-thought blocks
  that bloated the overlay and added latency.

## [0.7.0] - 2026-04-17

Milestone release rolling up the 0.6.8 – 0.6.15 push around LLM
post-processing UX. The default cleanup pipeline is now usable out of
the box: a tuned conservative prompt, a sibling fun profile, asymmetric
Gemma channel-tag stripping done right, a separate "thought" view in
the overlay, and quality-of-life fixes throughout.

### Highlights

- **Recommended `gemma4-cleanup` default profile** with a conservative
  tuned prompt: no rephrasing, no restructuring, no `?` ↔ `.` flips,
  German modal particles preserved (`denn`, `doch`, `mal`, `ja`,
  `eben`, `schon`). Assistant mode triggers only on the literal word
  `Computer` — questions and instructions without it stay pure
  dictation. The exact failure cases that motivated the rewrite are
  baked in as counter-examples.
- **`gemma4-fun` companion profile** auto-written on `init` — a tiny
  emojify-the-transcript stub that points users back at cleanup.
- **`setup-llm` model picker** tags `gemma4` as
  `(recommended — tuned for best results)`.
- **Overlay "thought" rendering**: `paste_strip_regex` now supports an
  optional capture group; matched content is shown italicised in
  blue-green (`#5ed1c4`) with a blank line separating it from the
  pasted body. The default regex strips Gemma's literal
  `<|channel>thought…<channel|>` framing so only the reasoning text
  appears in the overlay.
- **Default `result_linger_ms` halved** from 10 s to 5 s — long enough
  to glance at the result, short enough to clear before the next take.
- **Abort × pinned to overlay top-right** in the expanded result view
  via `Gtk.CenterBox`, fixing the layout collapse when the state label
  was hidden.
- **Default profile written as a TOML triple-quoted multi-line string**
  so the on-disk prompt is readable and editable, plus a new
  `context = ""` field for per-user notes.
- **`KNOWN_LLM_MODELS` model picker** lists the qwen alternatives but
  no longer presents them as co-equal to the tuned gemma4 path.

### Notes

- Existing installs whose `config.toml` still references
  `profile = "gemma-cleanup"` keep working — only fresh configs default
  to the renamed `gemma4-cleanup`.
- See entries 0.6.8 – 0.6.15 below for the full chronological log.

## [0.6.15] - 2026-04-17

### Changed

- **Default `paste_strip_regex` now consumes the literal `thought`
  channel label** that Gemma emits right after `<|channel>`:
  `<\|channel>thought(.*?)<channel\|>`. Previously the label leaked into
  the displayed thought as a stray "thought" word at the start of the
  italic preamble.
- **Overlay default linger halved** from 10 s to 5 s
  (`overlay.result_linger_ms = 5_000`). The old value lingered long
  enough to feel like a stuck overlay; 5 s is enough to glance at the
  thought and the cleaned text without getting in the way of the next
  dictation.
- **Overlay LLM markup now puts a blank line between the thought and
  the body**, not just a single newline. Visually separates the italic
  blue-green reasoning from the green reply you actually pasted.

## [0.6.14] - 2026-04-17

### Changed

- **Default profile renamed `gemma-cleanup` → `gemma4-cleanup`** and its
  system prompt promoted to the conservative tuned version that has been
  giving the best results in the wild. The new prompt is explicit about
  cleanup being non-destructive: no rephrasing, no restructuring, no
  punctuation flips, German modal particles preserved (`denn`, `doch`,
  `mal`, `ja`, `eben`, `schon`). Assistant mode only fires on the literal
  trigger word `Computer` — questions and instructions without it stay
  pure dictation. The exact failure cases that motivated the rewrite
  (over-correction of `was denkst du denn?`, treating "Can you tell me
  …?" as a request) are baked into the prompt as counter-examples.
- **Default `[postprocess].profile` in fresh configs now reads
  `gemma4-cleanup`** (was `gemma-cleanup`). Existing installs continue
  working with whatever filename their config already references.

### Added

- **`gemma4-fun` companion profile** written alongside cleanup on first
  `init`. A tiny stub with system prompt "emojify the transcript as much
  as possible" and a header that points users back at the recommended
  `gemma4-cleanup` profile. Useful for chat / social messages where you
  want a playful tone without composing a custom prompt.
- **`(recommended — tuned for best results)`** tag on the `gemma4` entry
  in the `setup-llm` model picker so the qwen alternatives don't look
  like equally-supported options.
- New helpers `ensure_fun_profile()` and `ensure_default_profiles()`
  (returns both paths). `init` now writes both profile files.

## [0.6.13] - 2026-04-17

### Changed

- **`paste_strip_regex` now honours an optional capture group** to control
  what is shown as the "thought" in the overlay. If the pattern contains
  one or more capture groups, group 1 is rendered (so users can wrap the
  inner content with `(…)` and hide the framing tokens). Without a group,
  the entire match is shown — same as before. The strip-from-paste
  behaviour is unchanged in both cases: the full match is always removed
  from the text that goes to the focused window.
- **Default `paste_strip_regex` now wraps the channel content in a
  capture group**: `<\|channel>(.*?)<channel\|>`. Fresh installs see the
  reasoning text without the surrounding `<|channel>` / `<channel|>`
  tags cluttering the overlay.
- **Overlay "thought" line is now blue-green** (`#5ed1c4`) and italic,
  visually separating it from the green LLM reply body. Implemented via
  Pango span markup so no extra CSS class was needed.

### Added

- Tests covering `find_strip_matches` with no group, with one group, and
  with multiple matches in the same text.

## [0.6.12] - 2026-04-17

### Changed

- **Default profile now writes `system_prompt` as a TOML triple-quoted
  multi-line string** instead of an escaped single-line basic string.
  The prompt is long enough that the previous form was unreadable on
  disk and discouraged customisation. The Python literal switched from
  `f"""…"""` to `f'''…'''` so the embedded `"""…"""` TOML delimiters are
  not parsed as the closing of the Python string.
- **Default `system_prompt` synced with the in-the-wild tuned version**:
  single `<|think|>` placement (was repeated), refined formatting
  examples for emojis / newlines / bullet lists / backticks, clearer
  `Hey Computer` instructions about not parroting the cleaned source
  back when the request was a translation/answer.
- Removed the now-unused `_toml_basic_escape()` helper — TOML
  triple-quoted basic strings need no escaping for the prompt content.

### Added

- New `context = ""` field with a commented example block in the
  default profile, mirroring the live `gemma4.toml` schema.

### Changed

- **Overlay LLM field now formats the "thought" preamble in italic** on
  its own line above the pasted body. Previously the entire LLM-cleaned
  output (including any `<|channel>…<channel|>` reasoning block) was
  shown in italic green and the user couldn't visually distinguish what
  the model "thought" from what it actually replied. Now whatever
  `paste_strip_regex` matches is rendered italicised, then a newline,
  then the stripped body in normal weight — same body that is pasted
  into the focused window.
- New `LLMPostprocessor.find_strip_matches()` helper returns the
  substrings the regex matched, so the cli can hand them to the overlay
  for display without depending on the private compiled pattern.

## [0.6.10] - 2026-04-17

### Changed

- **Default `paste_strip_regex` now matches the actual Gemma channel
  tags** the model emits with the default `<|think|>` prompt:
  `<\|channel>.*?<channel\|>`. The tags are asymmetric — the opening is
  `<|channel>` (one pipe, before `channel`) and the closing is
  `<channel|>` (one pipe, after). Earlier docs and examples used
  `<|channel|>` for the opening, which the model never emits, so the
  strip silently no-op'd and the entire reasoning preamble landed in
  the focused window.
- **Default profile comments** explain how to disable thinking entirely:
  remove BOTH `<|think|>` markers from `system_prompt` and clear
  `paste_strip_regex`. Useful for users who'd rather trade reply quality
  on ambiguous "Hey Computer" prompts for lower latency.

## [0.6.9] - 2026-04-17

### Fixed

- **Abort × is now pinned to the overlay's top-right** in the expanded
  result view. Previously it was packed in a horizontal Box next to the
  state label and only had `halign=END`; once the state label was hidden
  in result mode its allocation collapsed and the × ended up rendering
  at the start of the row. Switched the top row to `Gtk.CenterBox` with
  start/end widgets, which anchor independently regardless of which is
  visible.

## [0.6.8] - 2026-04-17

### Changed

- **Default `gemma-cleanup` profile rewritten** to match the prompt that has
  been giving the best results in practice — a "Computer" persona with a brief
  reasoning preamble, German/English mixed-language handling, formatting
  examples (emoji, multiline list rendering), and an explicit `Hey Computer`
  trigger phrase that switches the model from cleanup-only mode into
  follow-the-instruction mode (translate, compose, answer, chat, …).
  `max_tokens` raised to `4096` so longer "Hey Computer" replies aren't
  truncated. The default profile is generated on `justsayit init`; existing
  profiles are not overwritten.

### Fixed

- `_DEFAULT_PROFILE_TOML` now escapes the embedded system prompt before
  interpolation, so newlines and quotes in the default prompt produce valid
  TOML on disk (previously a multi-line default would render an unparseable
  basic string).

## [0.6.7] - 2026-04-17

### Added

- **`context`** field on the postprocess profile — free-form text
  (TOML multi-line string) appended to the system prompt under a
  `# User context` heading so the LLM knows who's dictating (name,
  country, languages, area of work, …). Empty by default.

  ```toml
  context = '''
  Name: Jane Doe
  Country: Germany
  Languages: German (native), English (fluent)
  '''
  ```

## [0.6.6] - 2026-04-17

### Added

- **Abort button (×) in the overlay's top-right.** Click it during a
  recording (validating / recording / manual) to discard the audio
  buffer and return to IDLE without transcribing or pasting. During
  the post-result linger phase it just dismisses the overlay early.
  Backed by a new `AudioEngine.abort()` that flushes the buffer and
  resets VAD without emitting a segment.

## [0.6.5] - 2026-04-17

### Added

- **`paste_strip_regex`** field on the postprocess profile. A regex
  (compiled with `re.DOTALL`) applied to the LLM output before pasting
  but **not** before the overlay shows it — so you can see the model's
  full reasoning while only the final message lands in the focused
  window. Designed for "thinking" models like Gemma's harmony format
  (`<|channel|>analysis…<|message|>final`). Default is empty (no
  stripping) for backwards compatibility.

  Examples:
  ```toml
  paste_strip_regex = '<\|channel\|>.*?<\|message\|>'  # one channel block
  paste_strip_regex = '(?s).*<\|message\|>'            # everything before last <|message|>
  ```

## [0.6.4] - 2026-04-17

### Changed

- **Manual mode (auto-listen off) now closes the microphone between
  recordings.** Previously the audio stream stayed open continuously and
  buffered into a lookback ring even when no recording was happening —
  fine for VAD mode but unnecessary (and a minor privacy / power cost)
  when the user is opting in per-press. The mic is opened on
  `start_manual()` and closed again as soon as the worker returns to
  IDLE. Auto-listen mode (`vad.enabled = true`) keeps the always-on
  stream and lookback. Toggling auto-listen from the tray opens / closes
  the stream live.

  Tradeoff: lookback (`audio.lookback_ms`) does nothing in manual mode
  now — the stream is closed before the user presses the hotkey, so
  there's nothing to look back at.

## [0.6.3] - 2026-04-17

### Changed

- **Install instructions:** `dotool` is now fetched separately via an AUR
  helper (`sudo yay -S dotool`) instead of being listed in the `pacman -S`
  command — `dotool` is in the AUR, not the official repos, so the previous
  instruction quietly failed for users without an AUR-aware wrapper.
- The `usermod` one-liner now prints "Please log out and back in for changes
  to take effect" when it actually adds the user, instead of silently
  succeeding.

## [0.6.2] - 2026-04-17

### Removed

- **`wtype` paste backend.** Was an alternative to `dotool` (virtual-keyboard
  protocol) but unused — `dotool` covers KDE Plasma / sway / Hyprland / niri
  uniformly and the wtype path was never selected in practice. `paste.backend`
  in `config.toml` now only accepts `"dotool"`. Drop `wtype` from your install
  command (`pacman -S … wtype` no longer needed); already-installed `wtype`
  packages can be uninstalled.

## [0.6.1] - 2026-04-17

### Fixed

- **`with-llm-vulkan` now works on non-NixOS hosts.** Previously the nixpkgs
  `vulkan-loader` found the system ICD JSON (e.g. `/usr/share/vulkan/icd.d/radeon_icd.json`)
  but couldn't resolve its relative `library_path` against the host's `/usr/lib`,
  so GPU init failed silently and llama.cpp fell back to CPU. The flake now
  bundles nixpkgs `mesa`'s Vulkan ICDs (absolute store paths) and the wrapper
  appends them to `VK_ADD_DRIVER_FILES`. Covers AMD (radv), Intel (anv),
  Nouveau, lavapipe, virtio. NixOS + NVIDIA users keep their system ICD from
  `/run/opengl-driver/…` (appends, doesn't replace). Non-NixOS NVIDIA
  proprietary users still need nixGL as a wrapper.
- Added `llama-cpp-python-vulkan` as an exposed flake package for debugging
  (`nm` / `readelf` on the compiled `libggml-vulkan.so`).

## [0.6.0] - 2026-04-16

### Added

- **Quick Start** — README restructured with copy-paste Arch and Nix quick
  start paths (Vulkan + LLM in two commands). Detailed install and
  configuration reference moved to `docs/install.md` and
  `docs/configuration.md`.
- Input group check one-liner (`id -nG | grep -qw input`) in quick start and
  install docs so users skip the `usermod` if already set up.

### Fixed

- `dotoold` service incorrectly listed as a requirement — justsayit spawns
  its own persistent `dotool` process; `input` group membership is sufficient.

## [0.5.4] - 2026-04-16

### Added

- **Nix LLM support** — `nix build .#with-llm` (CPU) and `nix build .#with-llm-vulkan`
  (Vulkan GPU) package outputs. Overrides `llama-cpp-python` to 0.3.20 so all current
  model architectures (Qwen3.5, Gemma 4, …) are supported.
- **`install.sh --nix [BINARY]`** — installs desktop integration and downloads models
  for a Nix-built binary; skips venv/pip setup. Resolves the `result` symlink to the
  real Nix store path so the `.desktop` entry survives rebuilds.
- **Updated default postprocess profile** — multilingual system prompt (DE/EN/mixed)
  with formatting and MetaRequest support; `n_ctx` and `max_tokens` raised to 4096;
  `hf_repo`/`hf_filename` pre-filled for auto-download.

### Fixed

- **`setup-llm` under Nix** — `_ensure_llama_cpp` now tries importing in the current
  process first; the Nix wrapper injects `sys.path` inline (not via env) so a fresh
  subprocess could never see the package.

### Changed

- **App ID renamed** `dev.horo.justsayit` → `dev.horotw.justsayit`. Users with an
  existing `.desktop` file should re-run `install.sh` (or `install.sh --nix`) to
  update it; the old entry is cleaned up automatically.

## [0.5.3] - 2026-04-16

### Added

- **`paste.restore_clipboard`** (default `true`) — the regular (Ctrl+V)
  clipboard is restored to its previous content after the synthetic paste
  keystroke, so dictation no longer clobbers whatever the user had copied.
  Primary/selection clipboard is not restored.  No-op when `type_directly`
  is enabled.
- **Nix flake** — `flake.nix` packages justsayit for Nix on Arch Linux.
  `nix build` produces a working binary with GTK4 layer-shell, PipeWire audio,
  and all runtime tools (`wl-clipboard`, `dotool`, `wtype`) on PATH.
  `nix build .#with-llm` adds `llama-cpp-python` (CPU) for LLM postprocessing.

## [0.5.2] - 2026-04-15

### Added

- **`paste.skip_clipboard_history`** (default `true`) — pass `--sensitive`
  to `wl-copy` so clipboard managers (e.g. KDE Klipper) skip recording the
  dictated text.  The text IS still available for a manual Ctrl+V paste
  immediately after dictation.
- **`paste.type_directly`** — inject text via `dotool type` directly (no
  clipboard involved at all; text is NOT available for re-paste).  Requires
  `backend = "dotool"`.  Takes precedence over `skip_clipboard_history` when
  both are set.

## [0.5.1] - 2026-04-15

### Added

- **Overlay result linger with two-field result view** — after a successful
  transcription the compact pill expands into two multi-line text fields:
  - **Top field** — the regex-filtered detected text, shown as soon as
    transcription finishes.
  - **Bottom field** — the LLM-cleaned result (light green, italic).  Shows
    "Wait for LLM processing…" while the model runs; hidden when LLM is off.
  The overlay stays visible after paste for `overlay.result_linger_ms`
  (default 10 s).  A pulsing green dot indicates the result phase.
  Setting `result_linger_ms = 0` hides immediately after paste.
- `overlay.max_width` (default 600 px) and `overlay.max_height` (default
  400 px) — cap the expanded overlay size.  Height is pre-estimated as
  `text_height × 2 + static_height` when detected text arrives.

### Changed

- Overlay transitions through "processing…" between recording stop and the
  first text result so the user always sees what the engine is doing.

## [0.5.0] - 2026-04-15

### Added

- **LLM postprocessor** — optional cleanup step that runs after ASR and
  regex filters. Uses a local GGUF model via `llama-cpp-python` to
  remove filler words, fix grammar/spelling, and correct misheard words
  while preserving meaning and style.
- **`[postprocess]` config section** — `enabled` (default `false`) and
  `profile` (name of the profile file to load, default `"gemma-cleanup"`).
- **Per-model profile files** — each LLM is configured in its own TOML
  file at `~/.config/justsayit/postprocess/<name>.toml`. Settings:
  `model_path`, `hf_repo` + `hf_filename` (for auto-download),
  `n_gpu_layers` (`-1` = all on GPU), `n_ctx`, `temperature`,
  `max_tokens`, `system_prompt`, `user_template`.
- **Default profile `gemma-cleanup.toml`** written by `justsayit init`
  with a German system prompt and `temperature = 0.08` (deterministic).
- **Auto-download** — if `hf_repo` + `hf_filename` are set in the
  profile and `model_path` doesn't exist, `justsayit download-models`
  fetches the GGUF from HuggingFace.
- **`[llm]` install extra** — `pip install 'justsayit[llm]'` for CPU.
  For Vulkan GPU (AMD/Intel): `CMAKE_ARGS="-DGGML_VULKAN=1" pip install
  llama-cpp-python`.
- **`install.sh --postprocess`** — compiles and installs
  `llama-cpp-python` with `GGML_VULKAN=1`; validates that cmake and
  Vulkan headers are present before starting; then launches the
  interactive `setup-llm` wizard.
- **`justsayit setup-llm`** — interactive wizard that lists the built-in
  model catalogue (gemma4, qwen3-4b, qwen3-0.8b), queries the
  HuggingFace API for the Q4_K_M GGUF filename, downloads it to the
  local cache, and patches the profile to point at the downloaded file.
  Pass `--model KEY` to skip the interactive prompt.

## [0.4.0] - 2026-04-15

### Added

- **Multi-backend transcription** — `model.backend` can now be set to
  `"parakeet"` (default, sherpa-onnx, bundled dep) or `"whisper"`
  (faster-whisper / distil-whisper, optional dep).
- **`model.whisper_model`** — HuggingFace model ID or local path for the
  Whisper backend (default: `"Systran/faster-distil-whisper-large-v3"`).
- **`model.whisper_device`** — inference device for Whisper (`"cpu"` or
  `"cuda"`, default `"cpu"`).
- **`model.whisper_compute_type`** — CTranslate2 quantisation for Whisper
  (`"int8"`, `"float16"`, `"float32"`, default `"int8"`).
- **`[whisper]` install extra** — `uv pip install 'justsayit[whisper]'`
  (or `install.sh --model whisper`) pulls in `faster-whisper`.
- **`install.sh --model parakeet|whisper`** — select backend at install
  time; writes `model.backend` into config.toml and installs required extras.
- **`justsayit init --backend parakeet|whisper`** — set backend in the
  generated config.toml without editing it by hand.
- Whisper model downloads lazily from HuggingFace on first transcription
  into `<cache>/justsayit/models/whisper/`; no extra download step needed.

### Changed

- `install.sh`: `gtk4-layer-shell` is now a **hard install blocker** (was
  a warning). The Wayland layer-shell overlay cannot work without it, so
  aborting early gives a clearer error message.
- `justsayit download-models` now prints a tailored message for the Whisper
  backend (only downloads the tiny VAD ONNX; Whisper model is deferred).

## [0.3.3] - 2026-04-15

### Fixed

- Auto-listen tray toggle now works immediately after the first reload.
  Previously, starting with `vad.enabled = false` meant the VAD model was
  never loaded (`vad_loaded = false`), so `vad_enabled` stayed `false`
  regardless of tray clicks — the checkbox appeared stuck and only a second
  reload (which picked up the silently-saved `vad.enabled = true`) fixed it.
  The VAD model is now always loaded on startup since it is always downloaded.

## [0.3.2] - 2026-04-15

### Added

- **Mute / unmute sounds** for VAD auto-listen mode — a descending two-tone
  "dub-di" (G4 → D4) plays when VAD is paused via the hotkey, and an
  ascending "dub-do" (D4 → G4) plays when it is resumed.

## [0.3.1] - 2026-04-15

### Changed

- In VAD auto-listen mode the start chime now plays as soon as the overlay
  appears (entering `VALIDATING`) at a reduced volume, giving early auditory
  feedback while the result is still uncertain. The scale is configurable via
  `sound.validating_volume_scale` (default `0.4`).
- The stop chime now plays whenever the overlay disappears (any → `IDLE`),
  including validation failures and manual stops.
- Hotkey-triggered (manual) recordings still play the start chime at full
  volume.

## [0.3.0] - 2026-04-15

### Added

- **Notification sounds** — a short chime plays when recording starts (A4,
  380 ms) and a lower, longer chime when it stops (E4, 530 ms). Sounds are
  generated from first-principles using numpy and bundled as WAV files; no
  external assets required.
- **`sound.enabled`** — master switch to disable sounds entirely.
- **`sound.volume`** — playback volume (0.0–1.0, default 1.0).
- `sounds/generate_sounds.py` — developer script to regenerate the bundled
  WAV files (not needed by end-users).

## [0.2.2] - 2026-04-15

### Added

- **Reload config** tray menu item — restarts the process via `execve` so
  all config changes (including overlay, audio, and model settings that
  cannot be hot-reloaded) take effect immediately.

## [0.2.1] - 2026-04-15

### Changed

- Default overlay width reduced from 260 to 174 (⅔ of previous).

## [0.2.0] - 2026-04-15

### Added

- **`paste.auto_space_timeout_ms`** — prepend a space before a transcription
  when the previous one finished within this many milliseconds, so continuous
  dictation works without manually inserting spaces between phrases. The
  timeout is checked against when the new recording *started* (derived from
  segment duration), so long recordings never incorrectly skip the prefix.
- **`paste.append_trailing_space`** — always append a trailing space after
  every transcription so the cursor is ready for the next word. Takes
  precedence over `auto_space_timeout_ms` when both are set; a desktop
  notification warns about the conflict.
- **`overlay.visualizer_sensitivity`** — scale factor for the mic-level bar
  (default `1.0`). Increase if your microphone records quietly; decrease if
  the bar clips on every word.
- **`overlay.opacity`** — background opacity of the overlay pill (`0.0`–`1.0`,
  default `0.78`). Applied uniformly to the entire widget (background, text,
  dot, and meter) via GTK `set_opacity`.

### Changed

- Overlay layout: status dot is now on the left and vertically centered;
  state label sits above the level meter in a vertical stack on the right.
- Visualizer bar grows symmetrically from the center outward instead of
  filling left-to-right.

## [0.1.0] - 2026-04-13

### Added

- Initial release.
