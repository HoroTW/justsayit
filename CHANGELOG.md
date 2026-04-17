# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  reads `pyproject.toml` from `main` on GitHub, parses the `version`
  field, and compares against the running `__version__`. When a newer
  version is available:
  - the overlay shows a small yellow `update available` badge to the
    left of the × button (tooltip carries the new version number);
  - a low-priority desktop notification fires once per launch (stable
    notification id, so re-checks update one entry rather than spamming);
  - if the running install is a git checkout (detected by `install.sh`
    + `.git` next to the package source), the notification body tells
    the user how to update: `cd <install dir> && ./install.sh --update`.
  Result is cached in `~/.cache/justsayit/update_check.json` for 24h so
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
