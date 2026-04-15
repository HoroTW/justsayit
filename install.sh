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
APP_ID="dev.horo.justsayit"
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

usage() {
    cat <<'EOF'
Usage: install.sh [--uninstall] [--autostart] [--skip-models]

  (default)       Create/update .venv, install deps, install .desktop file,
                  download models.
  --uninstall     Remove the .desktop file (and autostart entry if present).
                  The .venv and ~/.cache/justsayit are left in place — delete
                  them yourself if you want a clean wipe.
  --autostart     Also install a user-autostart .desktop so justsayit runs
                  on login.
  --skip-models   Don't pre-download models (they'll be fetched on first run).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        --autostart) AUTOSTART=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
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
need dotool "install 'dotool' (AUR: dotool, yay -S dotool) and enable the dotoold service"
need wtype "install 'wtype' (optional fallback; Arch: pacman -S wtype)"

if ! pkg-config --exists gtk4 2>/dev/null; then
    echo "warning: gtk4 pkg-config not found. PyGObject may still work via system gi," >&2
    echo "         but if build fails install 'gtk4' and headers for your distro." >&2
fi
if ! pkg-config --exists gtk4-layer-shell-0 2>/dev/null; then
    echo "warning: gtk4-layer-shell pkg-config not found. Install 'gtk4-layer-shell'" >&2
    echo "         (Arch: pacman -S gtk4-layer-shell)." >&2
fi

# --- venv ------------------------------------------------------------------

echo "==> creating venv at $VENV_DIR (using --system-site-packages for gi bindings)"
uv venv --system-site-packages "$VENV_DIR" >/dev/null

echo "==> installing project into venv"
# shellcheck disable=SC2046
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv pip install --python "$VENV_DIR/bin/python" \
    -e "$PROJECT_DIR[dev]"

BIN="$VENV_DIR/bin/justsayit"
if [ ! -x "$BIN" ]; then
    echo "installation failed: $BIN is missing" >&2
    exit 1
fi

# --- default config --------------------------------------------------------

if [ ! -f "${XDG_CONFIG_HOME:-$HOME/.config}/$APP_ID/config.toml" ]; then
    echo "==> writing default config and example filters"
    "$BIN" init || true
fi

# --- models ---------------------------------------------------------------

if [ "$SKIP_MODELS" -eq 0 ]; then
    echo "==> downloading Parakeet + VAD models (first run only)"
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

cat <<EOF

Done.

Next steps:
  * Make sure you're in the 'input' group so dotool can send keystrokes:
      sudo usermod -aG input "$USER" && sudo systemctl start dotoold
    (log out / log in for group membership to take effect)
  * Launch from your app launcher, or run: $BIN
  * Accept the KDE permission dialog the first time justsayit asks for
    a global shortcut; rebind it under System Settings → Shortcuts.
  * Edit ~/.config/justsayit/config.toml and filters.json to taste.
EOF
