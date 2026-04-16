#!/usr/bin/env sh
# install.sh — set up (or update) the justsayit venv and desktop integration.
#
# Idempotent: safe to run repeatedly to pull new dependencies or refresh the
# .desktop file.

set -eu

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd -P)
PROJECT_DIR=$SCRIPT_DIR
VENV_DIR=${VENV_DIR:-$PROJECT_DIR/.venv}
# APP_ID is both the .desktop filename stem AND the systemd scope / Gtk
# application id. They must stay in sync so the XDG portal can resolve
# our app-id (derived from the cgroup unit name) to real metadata
# (friendly name + icon) in the installed .desktop file. If they
# diverge you'll see the bare app-id show up under "System Services"
# with no icon in KDE's shortcut settings.
APP_ID="dev.horotw.justsayit"
APP_NAME="Just Say It"
# Legacy name from earlier versions; we clean it up on (re)install so
# people upgrading don't end up with two launcher entries.
LEGACY_APP_ID="justsayit"
DESKTOP_DIR=${XDG_DATA_HOME:-$HOME/.local/share}/applications
DESKTOP_FILE="$DESKTOP_DIR/$APP_ID.desktop"
LEGACY_DESKTOP_FILE="$DESKTOP_DIR/$LEGACY_APP_ID.desktop"
AUTOSTART_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/autostart
AUTOSTART_FILE="$AUTOSTART_DIR/$APP_ID.desktop"
LEGACY_AUTOSTART_FILE="$AUTOSTART_DIR/$LEGACY_APP_ID.desktop"

UNINSTALL=0
AUTOSTART=0
SKIP_MODELS=0
MODEL=""        # "" = parakeet (default), "whisper" = faster-whisper
POSTPROCESS=0   # 1 = install llama-cpp-python with Vulkan for LLM cleanup
NIX=0           # 1 = Nix-built binary; skip venv/pip, just do desktop + models
NIX_BIN=""      # path to the Nix-built binary (default: ./result/bin/justsayit)

usage() {
    cat <<'EOF'
Usage: install.sh [--uninstall] [--autostart] [--skip-models]
                  [--model parakeet|whisper] [--postprocess]
                  [--nix [BINARY]]

  (default)              Create/update .venv, install deps, install .desktop
                         file, download models (Parakeet + VAD).
  --uninstall            Remove the .desktop file (and autostart entry if
                         present). The .venv and ~/.cache/justsayit are left
                         in place — delete them yourself for a clean wipe.
  --autostart            Also install a user-autostart .desktop so justsayit
                         runs on login.
  --skip-models          Don't pre-download models (fetched on first run).
  --model parakeet       Use the bundled Parakeet TDT v3 backend (default).
  --model whisper        Use faster-whisper / distil-whisper (installs the
                         [whisper] extra; model downloads on first use).
  --postprocess          Skip the postprocessing prompt and always set up
                         LLM cleanup (Vulkan GPU, interactive model select).
                         Useful for non-interactive / scripted installs.
  --nix [BINARY]         Install desktop integration for a Nix-built binary.
                         Skips venv/pip setup. BINARY defaults to
                         ./result/bin/justsayit. Pairs with:
                           nix build .#with-llm-vulkan
                           ./install.sh --nix --postprocess
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        --autostart) AUTOSTART=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        --postprocess) POSTPROCESS=1 ;;
        --nix)
            NIX=1
            # Optional next argument: path to the binary (not a flag)
            if [ $# -gt 1 ]; then
                case "$2" in
                    -*) : ;;  # next arg is a flag, not a path
                    *)  NIX_BIN="$2"; shift ;;
                esac
            fi
            ;;
        --model)
            shift
            case "$1" in
                parakeet|whisper) MODEL="$1" ;;
                *) echo "unknown model: $1 (choose parakeet or whisper)" >&2; usage; exit 2 ;;
            esac
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

if [ "$UNINSTALL" -eq 1 ]; then
    [ -f "$DESKTOP_FILE" ] && rm -v "$DESKTOP_FILE"
    [ -f "$AUTOSTART_FILE" ] && rm -v "$AUTOSTART_FILE"
    [ -f "$LEGACY_DESKTOP_FILE" ] && rm -v "$LEGACY_DESKTOP_FILE"
    [ -f "$LEGACY_AUTOSTART_FILE" ] && rm -v "$LEGACY_AUTOSTART_FILE"
    echo "uninstalled desktop integration. Venv at $VENV_DIR was NOT removed."
    exit 0
fi

if [ "$NIX" -eq 1 ]; then
    # --- Nix mode: skip venv/pip, just wire up desktop + models ---------------

    if [ -z "$NIX_BIN" ]; then
        NIX_BIN="$PROJECT_DIR/result/bin/justsayit"
    fi
    # Resolve symlink so the .desktop Exec= points at the real store path,
    # which remains valid even after the result symlink is updated by a rebuild.
    if command -v readlink >/dev/null 2>&1; then
        NIX_BIN_REAL=$(readlink -f "$NIX_BIN" 2>/dev/null || echo "$NIX_BIN")
    else
        NIX_BIN_REAL="$NIX_BIN"
    fi
    if [ ! -x "$NIX_BIN_REAL" ]; then
        echo "error: Nix binary not found or not executable: $NIX_BIN_REAL" >&2
        echo "  Run 'nix build' first, or pass the path: ./install.sh --nix /path/to/justsayit" >&2
        exit 1
    fi
    BIN="$NIX_BIN_REAL"
    echo "==> using Nix binary: $BIN"
else
    # --- tool checks -----------------------------------------------------------

    need() {
        if ! command -v "$1" >/dev/null 2>&1; then
            echo "missing required tool: $1" >&2
            echo "$2" >&2
            exit 1
        fi
    }

    need uv "install from https://docs.astral.sh/uv/"
    need wl-copy "install the 'wl-clipboard' package"
    need dotool "install 'dotool' (AUR: dotool, yay -S dotool)"
    need wtype "install 'wtype' (optional fallback; Arch: pacman -S wtype)"

    if ! pkg-config --exists gtk4 2>/dev/null; then
        echo "warning: gtk4 pkg-config not found. PyGObject may still work via system gi," >&2
        echo "         but if build fails install 'gtk4' and headers for your distro." >&2
    fi
    if ! pkg-config --exists gtk4-layer-shell-0 2>/dev/null; then
        echo "error: gtk4-layer-shell not found." >&2
        echo "  The Wayland layer-shell overlay requires this library." >&2
        echo "  Install: pacman -S gtk4-layer-shell  (Arch / Manjaro)" >&2
        echo "           or equivalent for your distro, then re-run install.sh." >&2
        exit 1
    fi

    # --- venv ------------------------------------------------------------------

    echo "==> creating venv at $VENV_DIR (using --system-site-packages for gi bindings)"
    uv venv --system-site-packages "$VENV_DIR" >/dev/null

    echo "==> installing project into venv"
    EXTRAS="dev"
    [ "$MODEL" = "whisper" ] && EXTRAS="dev,whisper"
    UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv pip install --python "$VENV_DIR/bin/python" \
        -e "$PROJECT_DIR[$EXTRAS]"

    BIN="$VENV_DIR/bin/justsayit"
    if [ ! -x "$BIN" ]; then
        echo "installation failed: $BIN is missing" >&2
        exit 1
    fi
fi

# --- default config --------------------------------------------------------

if [ ! -f "${XDG_CONFIG_HOME:-$HOME/.config}/$APP_ID/config.toml" ]; then
    echo "==> writing default config and example filters"
    INIT_ARGS=""
    [ -n "$MODEL" ] && INIT_ARGS="--backend $MODEL"
    # shellcheck disable=SC2086
    "$BIN" init $INIT_ARGS || true
fi

# --- models ---------------------------------------------------------------

if [ "$SKIP_MODELS" -eq 0 ]; then
    if [ "$MODEL" = "whisper" ]; then
        echo "==> downloading VAD model (Whisper model downloads on first use)"
    else
        echo "==> downloading Parakeet + VAD models (first run only)"
    fi
    "$BIN" download-models
else
    echo "skipping model download (--skip-models)"
fi

# --- .desktop entry --------------------------------------------------------

mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Local Parakeet voice dictation with Wayland overlay
Exec=$BIN
Icon=audio-input-microphone
Terminal=false
Categories=Utility;AudioVideo;
StartupNotify=false
StartupWMClass=$APP_ID
X-GNOME-UsesNotifications=true
EOF
chmod 0644 "$DESKTOP_FILE"
echo "installed $DESKTOP_FILE"

# Remove the pre-rename .desktop if it's still lying around, so KDE's
# launcher doesn't show two "Just Say It" entries.
if [ -f "$LEGACY_DESKTOP_FILE" ]; then
    rm -v "$LEGACY_DESKTOP_FILE"
fi

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
fi

if [ "$AUTOSTART" -eq 1 ]; then
    mkdir -p "$AUTOSTART_DIR"
    cp "$DESKTOP_FILE" "$AUTOSTART_FILE"
    echo "installed autostart entry $AUTOSTART_FILE"
fi
# Always sweep the legacy autostart file too; if the user previously
# opted into autostart under the old id they'd otherwise get both.
if [ -f "$LEGACY_AUTOSTART_FILE" ]; then
    rm -v "$LEGACY_AUTOSTART_FILE"
fi

# --- LLM postprocessing (optional) ----------------------------------------

if [ "$POSTPROCESS" -eq 0 ] && [ -t 0 ]; then
    printf "\nSet up LLM postprocessing (fixes grammar/filler words after dictation)? [y/N] "
    read -r _REPLY
    case "$_REPLY" in
        [Yy]*) POSTPROCESS=1 ;;
    esac
fi

if [ "$POSTPROCESS" -eq 1 ]; then
    echo ""
    echo "==> setting up LLM postprocessing"
    "$BIN" setup-llm || true
fi

cat <<EOF

Done.

Next steps:
  * Make sure you're in the 'input' group so dotool can send keystrokes:
      sudo usermod -aG input "$USER"
    (log out / log in for group membership to take effect)
  * Launch from your app launcher, or run: $BIN
  * Accept the KDE permission dialog the first time justsayit asks for
    a global shortcut; rebind it under System Settings → Shortcuts.
  * Edit ~/.config/justsayit/config.toml and filters.json to taste.
EOF
