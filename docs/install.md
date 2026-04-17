# Installation

## Arch Linux

### Dependencies

```sh
sudo pacman -S uv gtk4 gtk4-layer-shell python-gobject portaudio wl-clipboard dotool wtype
```

### Input group (required for paste)

justsayit spawns its own persistent `dotool` process on demand — the
`dotoold` daemon is not required. Being in the `input` group is sufficient:

```sh
# Check if you're already in the group
id -nG | grep -qw input && echo "already in input group" || echo "run: sudo usermod -aG input $USER"
```

If you need to add yourself, log out and back in after:

```sh
sudo usermod -aG input $USER
```

### Install

```sh
git clone https://github.com/HoroTW/justsayit && cd justsayit
./install.sh
```

This creates `.venv/` with `--system-site-packages` so PyGObject picks up the
system GTK typelibs, installs the project, downloads the Parakeet and Silero
VAD models into `~/.cache/justsayit/models/`, and drops a `.desktop` file into
`~/.local/share/applications/`.

`install.sh` flags:

| Flag | Effect |
|------|--------|
| `--postprocess` | Set up LLM cleanup with Vulkan GPU (recommended) |
| `--autostart` | Install `~/.config/autostart/` entry |
| `--skip-models` | Skip model download (fetched on first run) |
| `--uninstall` | Remove `.desktop` file (venv and models left intact) |

---

## Nix flake

Requires Nix with flakes enabled. The flake targets `x86_64-linux`.

### Input group

Same requirement as Arch — see above.

### Package variants

| Command | llama-cpp-python | GPU |
|---------|-----------------|-----|
| `nix build` | — | — |
| `nix build .#with-llm` | CPU | — |
| `nix build .#with-llm-vulkan` | Vulkan | ✓ |

The Vulkan build compiles `llama-cpp-python` from source — takes a few minutes
the first time, then cached. It also fetches nixpkgs `mesa` (~800 MB, binary
cache) so the bundled Vulkan ICDs (RADV, ANV, Nouveau, lavapipe, …) work on
non-NixOS hosts without [nixGL](https://github.com/nix-community/nixGL).

> **NVIDIA proprietary driver:** not bundled (mesa only ships open-source
> drivers). On NixOS, your system's NVIDIA ICD at `/run/opengl-driver/…` is
> picked up automatically (the flake uses `VK_ADD_DRIVER_FILES`, which
> appends rather than overrides). On non-NixOS with the NVIDIA proprietary
> driver, wrap the command: `nix run --impure github:nix-community/nixGL --
> nix run github:HoroTW/justsayit#with-llm-vulkan`.

### Desktop launcher integration

`install.sh --nix` handles model download and the `.desktop` file. It resolves
the `result` symlink to the real Nix store path so the entry stays valid after
rebuilds.

```sh
nix build .#with-llm-vulkan
./install.sh --nix             # reads ./result/bin/justsayit automatically
# or combine with LLM model setup:
./install.sh --nix --postprocess
```

Pass a custom binary path if needed:

```sh
./install.sh --nix /path/to/justsayit
```

### LLM model setup

After building a `with-llm*` variant, download a GGUF model interactively:

```sh
./result/bin/justsayit setup-llm
```

Then enable postprocessing in `~/.config/justsayit/config.toml`:

```toml
[postprocess]
enabled = true
```
