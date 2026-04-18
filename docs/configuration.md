# Configuration

Config files live under `~/.config/justsayit/`. Run `justsayit init` once to
write the defaults — every key is shipped commented out, so future updates
that change a default flow through automatically; you only override what
you actually want.

| File | Purpose |
|------|---------|
| `config.toml` | Audio, VAD, shortcut, paste, ASR backend, overlay, sounds, logging — the user-authored settings. The app never rewrites this file. |
| `state.toml` | Tray-toggleable runtime state (auto-VAD on/off, postprocess on/off + active profile). Auto-written when you flip a tray switch. |
| `filters.json` | Ordered list of regex post-processing rules. |
| `.env` | API keys for the OpenAI-compatible LLM and Whisper backends. Same `KEY=VALUE` format as python-dotenv. Process env wins on collision. |
| `context.toml` | Personal context (name, languages, project spellings) appended to every LLM cleanup prompt. |
| `dynamic-context.sh` | Bash script run on every LLM request. Its stdout is prepended before the normal system prompt as a dynamic state block when non-empty. |
| `postprocess/<profile>.toml` | LLM profile (model, prompt, temperature, …). Three are shipped — see [postprocessing.md](postprocessing.md). |

Want to see every available knob? `justsayit show-defaults config` prints
the shipped `config.toml` in the same commented-defaults form that `init`
writes.

## ASR backends

Pick the speech-to-text engine in `config.toml`:

```toml
[model]
backend = "parakeet"   # default — local, offline, GPU-friendly
# backend = "whisper"  # local faster-whisper, optional [whisper] extra
# backend = "openai"   # remote OpenAI-compatible /audio/transcriptions
```

| Backend | Where it runs | Setup |
|---------|---------------|-------|
| `parakeet` | Local sherpa-onnx | `justsayit download-models` (default — ships with everything) |
| `whisper` | Local faster-whisper | `uv pip install -e ".[whisper]"`, then set `model.whisper_model` (default `Systran/faster-distil-whisper-large-v3`) |
| `openai` | Any OpenAI-compatible HTTP endpoint (OpenAI, Groq, vLLM, faster-whisper-server, whisper.cpp …) | Set `model.openai_endpoint`, `model.openai_model`, drop your key into `~/.config/justsayit/.env` |

Local model downloads are skipped when `backend = "openai"` — only the
tiny Silero VAD ONNX is fetched (we never stream audio to the network
just to detect silence).

`justsayit show-defaults config` lists every backend-specific knob
(`whisper_device`, `whisper_compute_type`, `openai_language`,
`openai_timeout`, …).

## API keys: `.env`

Both the OpenAI Whisper backend and the OpenAI-compatible LLM endpoint
read keys with the same three-tier resolver:

1. Inline literal in the relevant config field (`openai_api_key` /
   `api_key`) — easiest, but gets committed if you check in your dotfiles.
2. Process env var named by `*_api_key_env` (default `OPENAI_API_KEY`).
3. `~/.config/justsayit/.env` — `KEY=VALUE` per line, optional `export `
   prefix, optional matched single/double quotes around the value.

```env
# ~/.config/justsayit/.env
OPENAI_API_KEY=sk-...
GROQ_API_KEY="gsk_..."
```

The `.env` is loaded into `os.environ` on first secret resolution, but
anything you've already exported in the shell wins — same precedence as
python-dotenv.

## Activation modes

## Short-segment Skip

Very short clips can be dropped before any text processing happens:

```toml
[audio]
skip_segments_below_seconds = 1.0   # default; 0 disables the skip
```

Segments below this threshold are ignored before transcription, regex
filters, and LLM cleanup. The app just logs the skip and clears/hides
the overlay instead of producing text.

### Global hotkey (default)

On first run the portal pops up a dialog asking you to confirm / rebind the
requested shortcut (default `Super+\`). First press starts recording, second
press stops it and the buffer gets transcribed. No VAD, no validation, nothing
auto-opens.

### Auto-VAD (opt-in)

Set `vad.enabled = true` in `config.toml`, pass `--vad` on the command
line, or toggle it from the tray menu. Silero VAD opens a recording when
it detects speech. The first `validation_seconds` (default 3 s) are
transcribed immediately; if no words come out the segment is discarded
and we go back to idle. The hotkey still works alongside VAD.

## Overlay

A small rounded bar at the bottom of your screen. The dot colour:

| colour | state |
|--------|-------|
| grey   | idle |
| amber  | listening (first 3 s validation) |
| red    | recording (auto / VAD) |
| blue   | recording (manual / hotkey) |

The bar fills as your mic input gets louder. The overlay only appears
while a recording is active and lingers for `overlay.result_linger_ms`
(default 5 s) after a successful paste so you can read the result.
When the LLM emits a "thinking" preamble (e.g. Gemma's `<|channel>...`
block), it's shown above the cleaned text — only the cleaned text gets
pasted.

## System tray

A StatusNotifier tray icon (KDE / GNOME extension / waybar — anything
that speaks SNI) gives you a menu without leaving the keyboard:

- Toggle dictation on/off
- Toggle auto-VAD
- Toggle LLM postprocess + switch active profile (cleanup / fun /
  openai-cleanup / any custom one in `postprocess/`)
- Open `config.toml`, `filters.json`, the active profile, the personal
  context sidecar, or the log file in `xdg-open`
- Quit

State changes (toggles, profile switches) are persisted to `state.toml`
so they survive a restart. Your hand-edited `config.toml` is never
rewritten.

## Sounds

Optional notification chimes for start / stop / mute. Disable in
`config.toml`:

```toml
[sound]
enabled = true
volume = 1.0
validating_volume_scale = 0.4   # quieter chime while VAD is still validating
```

## Paste

Defaults route the text through `wl-copy` (with `--sensitive` so KDE
Klipper et al. don't record it) and trigger `Shift+Insert` via `dotool`.
Privacy-conscious knobs:

```toml
[paste]
type_directly = false           # bypass the clipboard entirely (dotool type)
skip_clipboard_history = true   # wl-copy --sensitive
restore_clipboard = true        # restore your previous clipboard after paste
```

### Paste timing — when pasting drops characters or pastes the wrong text

If pasting is unreliable in slow apps (Electron, web-based IDEs, remote
desktops), raise the relevant delay. Each setting is the wait time
*from* the first event *until* the second:

| Setting                  | Wait starts when…              | …and ends when                   | Default |
| ------------------------ | ------------------------------ | -------------------------------- | ------- |
| `paste.release_delay_ms` | the stop-hotkey is released    | the synthetic paste keystroke fires | 250 ms  |
| `paste.settle_ms`        | `wl-copy` finishes writing the clipboard | the synthetic paste keystroke fires | 40 ms   |
| `paste.restore_delay_ms` | the synthetic paste keystroke has fired | the previous clipboard is restored  | 250 ms  |

Symptoms vs. knob:

- **Paste produces stray modifier characters** (Super/Ctrl/Shift leaking
  into the keystroke) → raise `release_delay_ms`.
- **Paste pastes the wrong / previous clipboard** → raise `settle_ms`.
- **Paste pastes your old clipboard instead of the dictation** → raise
  `restore_delay_ms` (the app read the clipboard *after* we restored
  it).

`release_delay_ms` is a target, not an additive wait — if transcription
+ LLM cleanup already took longer than that, paste fires immediately.
`restore_delay_ms` is a no-op when `restore_clipboard = false` or
`type_directly = true`.

For continuous dictation:

```toml
auto_space_timeout_ms = 1500    # prepend " " if last paste was <1.5 s ago
append_trailing_space = false   # alternative: always end with a space
```

## Regex filters

`filters.json` is a JSON array. Each entry has `name`, `pattern`,
`replacement`, and optionally `flags` (a list of `IGNORECASE`, `MULTILINE`,
`DOTALL`, etc.) and `enabled` (bool). `re.sub`-style backreferences work,
including numbered (`\1`) and named (`\g<name>`) groups.

```json
[
  { "name": "trim",         "pattern": "^\\s+|\\s+$",        "replacement": "" },
  { "name": "collapse ws",  "pattern": "\\s{2,}",             "replacement": " " },
  {
    "name": "spoken email",
    "pattern": "(\\w+)\\s+at\\s+(\\w+)\\s+dot\\s+(\\w+)",
    "replacement": "\\1@\\2.\\3",
    "flags": ["IGNORECASE"]
  }
]
```

Filters run top-to-bottom, so later rules can operate on earlier results.
The default chain shipped on `init` already handles dictated punctuation
and line-break words in DE+EN, so you can use spoken `Komma` /
`new line` / `Punkt` / etc. without an LLM.

## LLM postprocessing

Optional cleanup pass with shipped profiles for local Gemma 4, an
emoji-heavy "fun" variant, and an OpenAI-compatible endpoint variant —
plus full customisation for your own style profiles (translate,
summarise, format-as-Markdown, you name it). Doubles as an inline
assistant when [`Hey Computer …`](postprocessing.md#hey-computer--inline-assistant-mode)
appears as an actual cue to the model. The shipped prompts treat that as
a best-effort signal anywhere in the transcript, while clearly quoted,
reported, incidental, or nonsensical uses should remain cleanup-only.
This is prompt-guided behavior, not a separate hard-coded parser.

```toml
[postprocess]
dynamic_context_script = "~/.config/justsayit/dynamic-context.sh"
```

`justsayit init` writes that script if missing. It is executed on every
LLM request with a small timeout; when it prints text, justsayit
prepends it before the normal system prompt in a `# STATE (DYNAMIC CONTEXT):`
block. The shipped script uses only local heuristics and prints local
time, date, timezone, and a locale hint when available.

Set `postprocess.dynamic_context_script = ""` to disable dynamic
context entirely.

See [docs/postprocessing.md](postprocessing.md).

## Logging

Console logging is always on (level via `--log-level`). Optional
rotating file log:

```toml
[log]
file_enabled = true
file_path = ""                  # "" → <cache_dir>/justsayit.log
file_level = "DEBUG"
file_max_bytes = 5_000_000
file_backup_count = 3
```
