# JustSayIt

Local Parakeet v3 voice dictation for Wayland.

> Heads up: I wrote this for myself because none of the existing
> solutions I tried were quite what I wanted. It's published in case it's
> useful to someone else, but it's shaped around my machine and my
> habits — no promises it fits yours.
>
> And yes, it's mostly vibe-coded. Take that however you want 😉

- Offline ASR via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) +
  [Parakeet TDT 0.6B v3 INT8](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Transparent layer-shell overlay (GTK4 + `gtk4-layer-shell`)
- Global toggle shortcut via the XDG Desktop Portal
- Auto-start on speech with a 3-second "did we actually hear words?" validation
- `wl-copy` + `dotool` to paste the result into the focused window
- JSON regex post-processing (with capture groups)

## Install

Requirements (Arch package names in parentheses):

- Wayland compositor with `zwlr_layer_shell_v1` — KDE Plasma, Hyprland, sway,
  niri, river…
- `uv` (`uv`)
- `gtk4`, `gtk4-layer-shell`, `python-gobject`
- `wl-clipboard`, `dotool`, `wtype` (`wl-clipboard`, `dotool`, `wtype`)
- `portaudio` for `sounddevice` (`portaudio`)

Then:

```sh
./install.sh
```

This creates `.venv/` with `--system-site-packages` so PyGObject can pick up
the system GTK typelibs, installs the project, downloads the Parakeet and
Silero VAD models into `~/.cache/justsayit/models/`, and drops a `.desktop`
file into `~/.local/share/applications/`.

Flags: `--autostart` (install `~/.config/autostart/justsayit.desktop`),
`--skip-models`, `--uninstall`.

You also need to be in the `input` group and have the `dotoold` service
running for paste to work:

```sh
sudo usermod -aG input $USER
sudo systemctl enable --now dotoold
# log out / log in once so the new group membership takes effect
```

## Install via Nix flake

Requires Nix with flakes enabled. The flake targets `x86_64-linux`.

### Basic install

```sh
nix build
./result/bin/justsayit download-models
```

Run it directly from the result symlink or add it to your `PATH`:

```sh
./result/bin/justsayit
```

Or run without building first:

```sh
nix run github:HoroTW/justsayit
```

### Desktop launcher integration

The Nix build doesn't install a `.desktop` file. Use `install.sh --nix` to
handle that (and model download) in one step:

```sh
nix build           # or .#with-llm / .#with-llm-vulkan
./install.sh --nix  # reads ./result/bin/justsayit by default
```

The script resolves the `result` symlink to the real Nix store path, so the
launcher entry stays valid after rebuilds. Pass a custom path if needed:

```sh
./install.sh --nix /path/to/justsayit
```

### Paste (same requirement as regular install)

```sh
sudo usermod -aG input $USER
# log out and back in so the group takes effect
```

### With LLM postprocessing (CPU)

The `with-llm` package includes `llama-cpp-python` (CPU-only).
After building you still need to download a GGUF model:

```sh
nix build .#with-llm
./result/bin/justsayit setup-llm --cpu   # interactive: picks and downloads a model
```

`setup-llm` writes a profile under `~/.config/justsayit/postprocess/` and
updates `config.toml` to point at it. Enable postprocessing in
`~/.config/justsayit/config.toml`:

```toml
[postprocess]
enabled = true
```

### With LLM postprocessing (Vulkan GPU)

The `with-llm-vulkan` package rebuilds `llama-cpp-python` with
`-DGGML_VULKAN=1` so inference runs on your GPU (takes a few minutes to
compile):

```sh
nix build .#with-llm-vulkan
./result/bin/justsayit setup-llm --cpu   # interactive: picks and downloads a model
```

Then enable postprocessing as above. `--cpu` only filters the interactive
model list to smaller sizes — inference still uses the GPU at runtime.
The build pulls in `vulkan-loader`, `vulkan-headers`, and `shaderc`
automatically.

## Usage

```sh
justsayit                 # normal run — overlay + portal shortcut
justsayit --no-overlay    # headless
justsayit --no-paste      # print to stdout only, don't simulate paste
justsayit init            # (re)write default config + example filters
justsayit download-models # pre-download the models
```

### Activation

Two modes — VAD is **off by default** while the app stabilises.

1. **Global hotkey (default)** — on first run the portal pops up a
   dialog asking you to confirm / rebind the requested shortcut
   (default `Super+\`). First press starts recording, second press
   stops it, and the buffer gets transcribed. No VAD, no validation,
   nothing auto-opens.

2. **Auto-VAD (opt-in)** — set `vad.enabled = true` in `config.toml`
   or pass `--vad`. Silero VAD opens when it detects speech. The first
   `validation_seconds` (default 3s) are transcribed immediately; if
   no words come out, the segment is discarded and we go back to idle.
   The hotkey still works alongside it.

The overlay only appears while a recording is active — it stays hidden
when idle, so you won't see it sitting on screen doing nothing.

### Overlay

A small rounded bar at the bottom of your screen. The dot is:

| colour | state |
|--------|-------|
| grey   | idle |
| amber  | listening (first 3s validation) |
| red    | recording (auto) |
| blue   | recording (manual / hotkey) |

The bar fills as your mic input gets louder.

## Configuration

Config files live under `~/.config/justsayit/`:

- `config.toml` — sample rate, VAD thresholds, shortcut preference, overlay
  geometry, paste combo, model URLs.
- `filters.json` — ordered list of regex post-processing rules.

Run `justsayit init` once to drop the defaults there.

### Regex filters

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

## Tests

```sh
uv run pytest
```

Covers filter parsing, group-reference replacement, flag handling, chain
ordering, and file-loading edge cases.

## Layout

```
src/justsayit/
    audio.py         mic capture + Silero VAD state machine
    cli.py           argparse + GLib glue
    config.py        TOML loader with dataclass defaults
    filters.py       regex post-processor
    model.py         download Parakeet + VAD to ~/.cache/justsayit
    overlay.py       gtk4-layer-shell bar with mic meter
    paste.py         wl-copy + dotool helpers
    shortcuts.py     XDG Desktop Portal GlobalShortcuts client
    transcribe.py    sherpa-onnx Parakeet TDT recognizer
tests/test_filters.py
install.sh
```

## Known gotchas

- **GNOME Mutter** doesn't implement `zwlr_layer_shell_v1`. Run with
  `--no-overlay` there.
- The XDG GlobalShortcuts portal is KDE Plasma 6 / GNOME 45+. On
  compositors without it (sway, niri, Hyprland) you'd want a compositor
  keybind calling a tiny IPC command instead — not currently shipped.
- The default model URL is the sherpa-onnx release tarball for
  `parakeet-tdt-0.6b-v3-int8`. If that URL has moved, override
  `model.parakeet_archive_url` and `model.parakeet_archive_dir` in
  `config.toml`.
