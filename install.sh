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
# Config directory name — must match APP_NAME in src/justsayit/config.py
# (Python's platformdirs uses that, NOT the .desktop APP_ID). Keep this
# in sync or --update will silently look in the wrong place and skip
# every reconcile prompt.
CONFIG_DIR_NAME="justsayit"
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
UPDATE=0        # 1 = git pull + refresh deps + interactively update user config files

usage() {
    cat <<'EOF'
Usage: install.sh [--uninstall] [--autostart] [--skip-models] [--update]
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
  --update               Pull latest commits (git pull), refresh the venv +
                         dependencies, refresh the .desktop entry, and
                         interactively offer to replace your filters.json
                         and shipped postprocess profile TOMLs with the
                         new shipped defaults. config.toml is left alone
                         (settings file, not a template — new keys inherit
                         dataclass defaults). Files that are replaced are
                         backed up to *.bak.<ts> first. Implies
                         --skip-models and skips the postprocess prompt.
                         Automatically restores llama-cpp-python (with
                         Vulkan if available) when postprocess is enabled.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        --autostart) AUTOSTART=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        --postprocess) POSTPROCESS=1 ;;
        --update) UPDATE=1; SKIP_MODELS=1 ;;
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

if [ "$UPDATE" -eq 1 ]; then
    if [ ! -d "$PROJECT_DIR/.git" ]; then
        echo "error: --update requires a git checkout, but $PROJECT_DIR is not one." >&2
        echo "       (Re)clone with: git clone https://github.com/HoroTW/justsayit" >&2
        exit 1
    fi
    echo "==> pulling latest commits in $PROJECT_DIR"
    (cd "$PROJECT_DIR" && git pull --ff-only) || {
        echo "error: git pull failed (uncommitted changes? non-fast-forward?). Resolve and re-run." >&2
        exit 1
    }
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

    # Reuse an existing venv rather than recreating it. Recreation prompts
    # ("replace? [y/N]") and, if the user agrees, NUKES manually-installed
    # extras like llama-cpp-python (built locally with CMAKE_ARGS=-DGGML_VULKAN=1
    # — not in pyproject extras since it needs custom CMake flags). Reusing
    # preserves them across `--update` runs.
    if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
        echo "==> reusing existing venv at $VENV_DIR (preserves llama-cpp-python etc.)"
    else
        echo "==> creating venv at $VENV_DIR (using --system-site-packages for gi bindings)"
        uv venv --system-site-packages "$VENV_DIR" >/dev/null
    fi

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

if [ ! -f "${XDG_CONFIG_HOME:-$HOME/.config}/$CONFIG_DIR_NAME/config.toml" ]; then
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

# --- update mode: prompt to refresh user config files ---------------------

# Simple "diff and prompt" reconcile for files that have no commented-
# defaults form (filters.json, since JSON has no comment syntax). For
# config.toml + the two postprocess profile TOMLs we ship the commented-
# defaults form: every key is a commented default, only the user's
# uncommented overrides matter, and shipped-default drift never
# collides with their settings — so those files don't need reconciling
# at all.
maybe_update_user_file() {
    _USER_FILE=$1
    _KIND=$2
    [ -f "$_USER_FILE" ] || return 0
    _NEW=$(mktemp -t justsayit-defaults.XXXXXX)
    if ! "$BIN" show-defaults "$_KIND" >"$_NEW" 2>/dev/null; then
        echo "  could not render defaults for $_KIND — skipping." >&2
        rm -f "$_NEW"
        return 0
    fi
    if cmp -s "$_USER_FILE" "$_NEW"; then
        rm -f "$_NEW"
        return 0
    fi
    echo
    echo "==> $_KIND differs from the latest shipped defaults."
    echo "    ($_USER_FILE)"
    if command -v diff >/dev/null 2>&1; then
        echo "    diff (your current file -> shipped defaults), first 60 lines:"
        diff -u "$_USER_FILE" "$_NEW" | sed 's/^/      /' | head -60
    fi
    echo
    echo "    Replacing will overwrite the file (kept as .bak — re-apply"
    echo "    any local edits from there)."
    if [ -t 0 ]; then
        printf "Replace with new shipped defaults? Current file will be backed up. [Y/n] "
        read -r _REPLY
    else
        _REPLY="y"
    fi
    case "$_REPLY" in
        [Nn]*)
            rm -f "$_NEW"
            echo "  kept current $_USER_FILE."
            ;;
        *)
            _TS=$(date +%Y%m%d-%H%M%S)
            cp -v "$_USER_FILE" "$_USER_FILE.bak.$_TS"
            mv "$_NEW" "$_USER_FILE"
            echo "  updated. Old file kept at $_USER_FILE.bak.$_TS"
            ;;
    esac
}

if [ "$UPDATE" -eq 1 ]; then
    _CFG_HOME=${XDG_CONFIG_HOME:-$HOME/.config}/$CONFIG_DIR_NAME
    # Create any config files added since the last install (idempotent —
    # justsayit init only writes files that don't already exist).
    echo "==> ensuring new config files exist"
    "$BIN" init || true
    # config.toml + postprocess profile TOMLs ship in commented-defaults
    # form (every key is a commented default; uncommented lines are the
    # user's overrides). New shipped defaults change only the comments,
    # never colliding with overrides, so no reconcile is needed for
    # those. context.toml is pure user data and never reconciled.
    if [ -f "$_CFG_HOME/config.toml" ]; then
        echo
        echo "==> config.toml left untouched (commented-defaults form —"
        echo "    your uncommented overrides keep working as defaults"
        echo "    drift). Run \`justsayit show-defaults config\` to see"
        echo "    the current shipped values."
    fi
    maybe_update_user_file "$_CFG_HOME/filters.json" "filters"

    # Defense-in-depth: catch the case where the venv was rebuilt by an
    # older install.sh (or by the user manually) and llama-cpp-python
    # got dropped, but the user's config/state still has postprocess
    # enabled. The app would otherwise crash with ModuleNotFoundError
    # on the first dictation.
    if [ "$NIX" -eq 0 ] && [ -x "$VENV_DIR/bin/python" ]; then
        if ! "$VENV_DIR/bin/python" -c "import llama_cpp" 2>/dev/null; then
            # Cheap grep instead of TOML parsing — `enabled = true` (any
            # whitespace) under [postprocess] in either file is enough to
            # warn. Both files are tiny so reading them is fine.
            _PP_ON=0
            for _f in "$_CFG_HOME/state.toml" "$_CFG_HOME/config.toml"; do
                [ -f "$_f" ] || continue
                if awk '
                    /^\[postprocess\]/ { in_pp=1; next }
                    /^\[/              { in_pp=0 }
                    in_pp && /^[[:space:]]*enabled[[:space:]]*=[[:space:]]*true/ { found=1 }
                    END { exit (found ? 0 : 1) }
                ' "$_f"; then
                    _PP_ON=1
                    break
                fi
            done
            if [ "$_PP_ON" -eq 1 ]; then
                echo
                echo "==> postprocess is enabled but llama-cpp-python is missing — restoring..."
                _VULKAN_OK=0
                pkg-config --exists vulkan 2>/dev/null && command -v cmake >/dev/null 2>&1 && _VULKAN_OK=1
                if [ "$_VULKAN_OK" -eq 1 ]; then
                    echo "    (compiling with Vulkan GPU support — this may take a few minutes)"
                    CMAKE_ARGS="-DGGML_VULKAN=1" UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
                        uv pip install --python "$VENV_DIR/bin/python" "llama-cpp-python>=0.3" || {
                        echo "WARNING: Vulkan build failed — retrying CPU-only build" >&2
                        UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
                            uv pip install --python "$VENV_DIR/bin/python" "llama-cpp-python>=0.3" || {
                            echo "ERROR: could not restore llama-cpp-python. Run manually:" >&2
                            echo "  CMAKE_ARGS=\"-DGGML_VULKAN=1\" uv pip install llama-cpp-python>=0.3" >&2
                        }
                    }
                else
                    echo "    (Vulkan/cmake not found — installing CPU-only build)"
                    UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
                        uv pip install --python "$VENV_DIR/bin/python" "llama-cpp-python>=0.3" || {
                        echo "ERROR: could not restore llama-cpp-python. Run manually:" >&2
                        echo "  uv pip install llama-cpp-python>=0.3" >&2
                    }
                fi
            fi
        fi
    fi
fi

# --- LLM postprocessing (optional) ----------------------------------------

# In --update mode the user already chose their LLM setup on the original
# install; don't pester them again — they can re-run setup-llm by hand.
if [ "$UPDATE" -eq 0 ] && [ "$POSTPROCESS" -eq 0 ] && [ -t 0 ]; then
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
