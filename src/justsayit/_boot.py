"""Pre-GTK boot helpers: process name, systemd scoping, layer-shell preload.

All functions here use only ``os`` and ``sys`` (plus lazy stdlib imports)
so they can be imported before any GTK or GLib code touches the process.
"""

from __future__ import annotations

import os
import sys

# Subcommands that only talk to the already-running primary over D-Bus
# don't need layer-shell preload or systemd scoping.
_REMOTE_SUBCOMMANDS = {"toggle"}


def _reexec_cmd() -> list[str]:
    """Return the argv to re-execute the current process.

    When running inside a Nix ``makeBinaryWrapper`` ELF wrapper,
    ``sys.argv[0]`` is the ELF binary path. Detect this case and exec the
    ELF directly instead — the wrapper handles Python setup on its own.
    """
    argv0 = sys.argv[0]
    try:
        with open(argv0, "rb") as f:
            if f.read(4) == b"\x7fELF":
                return [argv0] + sys.argv[1:]
    except OSError:
        pass
    return [sys.executable, *sys.argv]


def _set_process_name(name: str) -> None:
    """Set the kernel-visible ``comm`` field via ``prctl(PR_SET_NAME)``
    so ``killall`` / ``pgrep`` (without ``-f``) show *name* instead of
    ``python3``. Linux-only; quietly no-ops on failure."""
    if sys.platform != "linux":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(15, name.encode("ascii", "replace")[:15], 0, 0, 0)
    except Exception:
        pass


def _app_id() -> str:
    """Portal application id used for D-Bus / systemd scoping."""
    return os.environ.get("JUSTSAYIT_APP_ID", "dev.horotw.justsayit")


def _relaunch_via_desktop() -> bool:
    """Spawn a fresh instance via the installed ``.desktop`` file so the
    desktop env places it in a portal-recognized systemd scope with a
    stable app id. Returns ``True`` if a new instance was launched."""
    try:
        from gi.repository import Gio  # local import: avoids loading gi before LD_PRELOAD reexec
        info = Gio.DesktopAppInfo.new(_app_id() + ".desktop")
    except Exception:
        return False
    if info is None:
        return False
    try:
        return bool(info.launch([], None))
    except Exception:
        return False


def _reexec_under_systemd_scope() -> None:
    """Re-exec inside an ``app-<app_id>-*.scope`` cgroup so the XDG portal
    can identify us with a stable name no matter how we were launched.

    Skipped inside Flatpak/Snap (already scoped by the sandbox), if
    ``systemd-run`` is missing, if we already re-execed once, or if the
    current cgroup already contains our app-id scope marker.
    """
    if os.environ.get("_JUSTSAYIT_SCOPED") == "1":
        return
    if os.environ.get("FLATPAK_ID") or os.environ.get("SNAP"):
        return
    app_id = _app_id()
    scope_marker = f"app-{app_id}"
    try:
        with open("/proc/self/cgroup", "r") as f:
            if scope_marker in f.read():
                return
    except OSError:
        pass
    import shutil
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        return
    env = os.environ.copy()
    env["_JUSTSAYIT_SCOPED"] = "1"
    unit = f"app-{app_id}-{os.getpid()}"
    reexec = _reexec_cmd()
    os.execvpe(systemd_run, [
        systemd_run, "--user", "--scope",
        f"--unit={unit}", "--quiet", "--collect",
        *reexec,
    ], env)


def _find_layer_shell_lib() -> str | None:
    candidates = [
        "/usr/lib/libgtk4-layer-shell.so",
        "/usr/lib64/libgtk4-layer-shell.so",
        "/usr/local/lib/libgtk4-layer-shell.so",
        "/usr/lib/x86_64-linux-gnu/libgtk4-layer-shell.so",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    try:
        import subprocess
        out = subprocess.check_output(["ldconfig", "-p"], text=True)
        for line in out.splitlines():
            if "libgtk4-layer-shell.so" in line and "=>" in line:
                return line.split("=>", 1)[1].strip()
    except Exception:
        pass
    return None


def _preload_layer_shell() -> None:
    if os.environ.get("_JUSTSAYIT_PRELOADED") == "1":
        return
    lib = _find_layer_shell_lib()
    if lib is None:
        return
    env = os.environ.copy()
    existing = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{lib}:{existing}" if existing else lib
    env["_JUSTSAYIT_PRELOADED"] = "1"
    cmd = _reexec_cmd()
    os.execvpe(cmd[0], cmd, env)


def _is_remote_subcommand() -> bool:
    for tok in sys.argv[1:]:
        if tok.startswith("-"):
            continue
        return tok in _REMOTE_SUBCOMMANDS
    return False
