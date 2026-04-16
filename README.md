# JustSayIt

![demo](docs/demo.gif)

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
- Optional LLM cleanup pass (grammar, filler words, formatting, MetaRequests)
- JSON regex post-processing (with capture groups)

## Quick Start

### Arch Linux (Vulkan GPU + LLM)

```sh
# 1. Dependencies + input group
sudo pacman -S uv gtk4 gtk4-layer-shell python-gobject portaudio wl-clipboard dotool wtype
id -nG | grep -qw input && echo "already in input group" || sudo usermod -aG input $USER
# if you just added yourself: log out and back in

# 2. Clone, install, download models, set up LLM
git clone https://github.com/HoroTW/justsayit && cd justsayit
./install.sh --postprocess
```

`install.sh` handles the venv, model downloads, `.desktop` file, and the
interactive LLM model selection (Gemma 4 for best quality).

### Nix flake (Vulkan GPU + LLM)

```sh
# 1. Input group (log out and back in if you just added yourself)
id -nG | grep -qw input && echo "already in input group" || sudo usermod -aG input $USER

# 2. Download models + set up LLM
nix run github:HoroTW/justsayit#with-llm-vulkan -- download-models
nix run github:HoroTW/justsayit#with-llm-vulkan -- setup-llm

# 3. Run
nix run github:HoroTW/justsayit#with-llm-vulkan
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
overlay states, regex filters, and LLM postprocessing.

## Known gotchas

- **GNOME Mutter** doesn't implement `zwlr_layer_shell_v1`. Run with
  `--no-overlay` there.
- The XDG GlobalShortcuts portal requires KDE Plasma 6 / GNOME 45+. On
  compositors without it (sway, niri, Hyprland) use a compositor keybind
  instead — not currently shipped.
- If the Parakeet model URL has moved, override `model.parakeet_archive_url`
  and `model.parakeet_archive_dir` in `config.toml`.

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
docs/
    install.md       detailed install (Arch + Nix)
    configuration.md activation, overlay, filters, LLM postprocessing
```
