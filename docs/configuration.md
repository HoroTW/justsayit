# Configuration

Config files live under `~/.config/justsayit/`. Run `justsayit init` once to
write the defaults.

- `config.toml` — sample rate, VAD thresholds, shortcut preference, overlay
  geometry, paste combo, model URLs, postprocessing toggle.
- `filters.json` — ordered list of regex post-processing rules.
- `postprocess/<profile>.toml` — LLM model path, context size, system prompt.

## Activation modes

### Global hotkey (default)

On first run the portal pops up a dialog asking you to confirm / rebind the
requested shortcut (default `Super+\`). First press starts recording, second
press stops it and the buffer gets transcribed. No VAD, no validation, nothing
auto-opens.

### Auto-VAD (opt-in)

Set `vad.enabled = true` in `config.toml` or pass `--vad`. Silero VAD opens a
recording when it detects speech. The first `validation_seconds` (default 3 s)
are transcribed immediately; if no words come out the segment is discarded and
we go back to idle. The hotkey still works alongside VAD.

## Overlay

A small rounded bar at the bottom of your screen. The dot colour:

| colour | state |
|--------|-------|
| grey   | idle |
| amber  | listening (first 3 s validation) |
| red    | recording (auto / VAD) |
| blue   | recording (manual / hotkey) |

The bar fills as your mic input gets louder. The overlay only appears while a
recording is active — it stays hidden when idle.

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

## LLM postprocessing

An optional LLM cleanup pass runs after transcription to fix grammar, remove
filler words, apply formatting, and handle MetaRequests spoken into the mic.

Driven by a profile TOML under `~/.config/justsayit/postprocess/`. `init`
writes two profiles:

- **`gemma4-cleanup`** (recommended) — conservative DE/EN cleanup tuned for
  Gemma 4 E4B. Removes filler words, fixes obvious mishears, applies dictated
  formatting, and switches into assistant mode only when the literal trigger
  word `Computer` appears in the transcript.
- **`gemma4-fun`** — a tiny emoji-heavy variant of cleanup. Keeps the original
  wording but sprinkles emojis throughout. Useful for chat / social messages.

To set up a profile interactively:

```sh
justsayit setup-llm
```

Enable in `config.toml`:

```toml
[postprocess]
enabled = true
profile = "gemma4-cleanup"   # filename stem under postprocess/
```
