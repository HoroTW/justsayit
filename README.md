# JustSayIt

![demo](docs/demo.gif)

Local Parakeet v3 voice dictation for Wayland.

> Heads up: I wrote this for myself because none of the existing
> solutions I tried were quite what I wanted. It's published in case it's
> useful to someone else, but it's shaped around my machine and my
> habits — no promises it fits yours.
>
> And yes, it's mostly vibe-coded. Take that however you want 😉

- **Offline ASR** via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) +
  [Parakeet TDT 0.6B v3 INT8](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  — also supports local [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  or any **OpenAI-compatible `/audio/transcriptions` endpoint** (Groq,
  vLLM, whisper.cpp server, …). See [docs/configuration.md](docs/configuration.md#asr-backends).
- **Transparent layer-shell overlay** (GTK4 + `gtk4-layer-shell`) with
  mic visualizer, status colours, and a result preview after paste
- **Global toggle shortcut** via the XDG Desktop Portal
- **Auto-start on speech** with a 3-second "did we actually hear words?" validation
- **`wl-copy` + `dotool` paste** into the focused window — with a privacy
  mode that bypasses the clipboard entirely (`paste.type_directly`)
- **System tray** to toggle dictation / auto-VAD, switch postprocess
  profile, and open config files
- **"Hey Computer" inline assistant** — `Hey Computer` anywhere in a
  dictation is generally treated as a cue that the text is for the LLM,
  so it may answer directly into the focused window instead of doing
  cleanup only. This is prompt-guided best effort, not a hard-coded
  parser rule: clearly quoted, reported, incidental, or otherwise
  non-addressed uses should stay cleanup-only, and so should cases where
  treating it as an instruction clearly does not make sense. Use it for
  rewrite-style requests like `Hey Computer, make this sound more
  formal`, composition/help requests like `Hey Computer, there is an
  offering, please write a humble decline with the wording 'deeply sorry
  ...'`, or already-dictated text such as `... Hey Computer, please
  clean this up`. Ask for a translation, a quick rewrite, a calculation,
  a snippet of code, … without leaving the keyboard. See
  [docs/postprocessing.md#hey-computer--inline-assistant-mode](docs/postprocessing.md#hey-computer--inline-assistant-mode).
- **Optional LLM cleanup pass** with shipped profiles (cleanup, emoji,
  OpenAI-compatible endpoint), a per-request dynamic-context script, and
  **fully customisable system prompts** for emojification,
  translation, summarisation, or your own style. Runs locally via
  `llama-cpp-python` or remotely against any **OpenAI-compatible
  `/chat/completions` endpoint** (OpenAI, OpenRouter, Groq, vLLM,
  Ollama, LM Studio, …). API keys can live in a shared
  `~/.config/justsayit/.env`. See [docs/postprocessing.md](docs/postprocessing.md).
- **JSON regex post-processing** with capture groups (default chain
  handles dictated punctuation in DE+EN — works without an LLM)
- **Personal context sidecar** (`~/.config/justsayit/context.toml`) so
  the LLM knows your name, languages, and project-specific spellings
- **Notification sounds** for start / stop / mute (configurable, fully
  optional)

## Quick Start

### Arch Linux (Vulkan GPU + LLM)

```sh
# 1. Dependencies + input group
sudo pacman -S uv gtk4 gtk4-layer-shell python-gobject portaudio wl-clipboard
yay -S dotool   # AUR — swap yay for your favourite AUR helper
id -nG | grep -qw input && echo "already in input group" \
    || (sudo usermod -aG input $USER && echo "Please log out and back in for changes to take effect.")

# 2. Clone, install, download models, set up LLM
git clone https://github.com/HoroTW/justsayit && cd justsayit
./install.sh --postprocess
```

`install.sh` handles the venv, model downloads, `.desktop` file, and the
interactive LLM model selection (Gemma 4 for best quality).

### Nix flake (Vulkan GPU + LLM)

```sh
# 1. Input group
id -nG | grep -qw input && echo "already in input group" \
    || (sudo usermod -aG input $USER && echo "Please log out and back in for changes to take effect.")

# 2. Download models + set up LLM
nix run github:HoroTW/justsayit#with-llm-vulkan -- download-models
nix run github:HoroTW/justsayit#with-llm-vulkan -- setup-llm

# 3. Run
nix run github:HoroTW/justsayit#with-llm-vulkan
```

Non-NixOS hosts with the **NVIDIA proprietary driver** need a nixGL wrapper
(bundled mesa covers AMD / Intel / Nouveau only):

```sh
nix run --impure github:nix-community/nixGL -- nix run github:HoroTW/justsayit#with-llm-vulkan
```

For a persistent install with a desktop launcher, see [docs/install.md](docs/install.md).

## Usage

```sh
justsayit                 # overlay + portal shortcut
justsayit --no-overlay    # headless
justsayit --no-paste      # print to stdout only
justsayit init            # write default config + example filters
justsayit download-models # pre-download models
justsayit setup-llm       # interactive LLM model setup
```

See [docs/configuration.md](docs/configuration.md) for activation modes,
including short-segment skipping, ASR backends (Parakeet / Whisper /
OpenAI), overlay, sounds, tray, and regex filters. LLM cleanup — shipped
profiles, custom prompts (emoji / translate / summarise / your own
style), OpenAI-compatible endpoints, dynamic context, personal-context
sidecar — is in [docs/postprocessing.md](docs/postprocessing.md).


## Update

You can check the [CHANGELOG.md](CHANGELOG.md) for new features and fixes.
To update, pull the latest changes and run the update command:

```sh
justsayit --update

# To check for the new feature config flags in the config use:
diff -u --color <(justsayit show-defaults config) ~/.config/justsayit/config.toml
# or your favorite diff tool, e.g. meld
```


# for a harder reset + all new config options (but loosing your current):
justsayit init

```

## Known gotchas

- **GNOME Mutter** doesn't implement `zwlr_layer_shell_v1`. Run with
  `--no-overlay` there.
- The XDG GlobalShortcuts portal requires KDE Plasma 6 / GNOME 45+. On
  compositors without it (sway, niri, Hyprland) use a compositor keybind
  to toggle recording via DBus instead:
  ```sh
  busctl --user call dev.horotw.justsayit /dev/horotw/justsayit org.gtk.Actions Activate "sava{sv}" toggle 0 0
  ```
  For example, in a niri config:
  ```
  Super+T { spawn "busctl" "--user" "call" "dev.horotw.justsayit" "/dev/horotw/justsayit" "org.gtk.Actions" "Activate" "sava{sv}" "toggle" "0" "0"; }
  ```
- If the Parakeet model URL has moved, override `model.parakeet_archive_url`
  and `model.parakeet_archive_dir` in `config.toml`.
- **Paste sometimes drops characters / pastes the wrong text?** Slow
  apps (Electron-based ones are the usual suspect) need a bit more
  breathing room. Raise the relevant delay in `[paste]` — each value
  is the wait *from* the first event *until* the second:

  | Setting                  | From                                     | Until                               | Default |
  | ------------------------ | ---------------------------------------- | ----------------------------------- | ------- |
  | `paste.release_delay_ms` | stop-hotkey released                     | synthetic paste keystroke fires     | 250 ms  |
  | `paste.settle_ms`        | `wl-copy` finishes writing the clipboard | synthetic paste keystroke fires     | 40 ms   |
  | `paste.restore_delay_ms` | synthetic paste keystroke has fired      | previous clipboard is restored      | 250 ms  |

  See [docs/configuration.md → Paste timing](docs/configuration.md#paste-timing--when-pasting-drops-characters-or-pastes-the-wrong-text)
  for symptom-by-symptom guidance.

## Layout

```
src/justsayit/
    audio.py             mic capture + Silero VAD state machine
    cli.py               argparse + GLib glue
    config.py            TOML loader with dataclass defaults + .env
    filters.py           regex post-processor
    model.py             download Parakeet + VAD to ~/.cache/justsayit
    overlay.py           gtk4-layer-shell bar with mic meter
    paste.py             wl-copy + dotool helpers
    postprocess.py       LLM cleanup (local llama-cpp + remote OpenAI)
    shortcuts.py         XDG Desktop Portal GlobalShortcuts client
    sound.py             notification chimes
    transcribe.py        backend dispatcher
    transcribe_parakeet.py
    transcribe_whisper.py    faster-whisper backend
    transcribe_openai.py     OpenAI-compatible /audio/transcriptions
    tray.py              StatusNotifier tray icon + menu
docs/
    install.md           detailed install (Arch + Nix)
    configuration.md     backends, activation, overlay, sounds, tray, filters
    postprocessing.md    LLM profiles, custom prompts, OpenAI endpoint, context
```
