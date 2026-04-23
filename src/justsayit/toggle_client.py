"""Fast-path client for ``justsayit toggle``.

Imports only ``gi`` + ``Gio`` / ``GLib`` — not Gtk, not numpy, not
sherpa-onnx, not llama-cpp. Sends the D-Bus message to the running
primary directly (same path ``busctl`` uses) and returns.

Why this exists: the full ``justsayit.cli`` module pulls in the audio
stack and the whole GTK world at import time (~500ms on a warm disk).
A keyboard-shortcut-bound ``justsayit toggle`` doesn't need any of it
— it just needs to poke the primary. ``_entry.py`` dispatches the
``toggle`` subcommand here to skip that cost. Raw ``busctl`` is still
faster (~15ms vs ~150ms) and is documented as the alternative for users
who want absolute minimum latency.
"""

from __future__ import annotations

import argparse
import os
import sys

import gi
from gi.repository import Gio, GLib  # noqa: E402


def _app_id() -> str:
    return os.environ.get("JUSTSAYIT_APP_ID", "dev.horotw.justsayit")


def _bus_path(app_id: str) -> str:
    """Map an app-id to the default GApplication object path.

    GApplication publishes its action group at
    ``/`` + app-id-with-dots-to-slashes. Stable since GLib forever.
    Hyphens are also replaced with underscores — the GLib rule is
    "non-``[A-Za-z0-9_]`` → ``_``".
    """
    return "/" + "".join(
        "/" if c == "." else (c if c.isalnum() or c == "_" else "_")
        for c in app_id
    )


def send_toggle(*, profile: str | None, use_clipboard: bool, continue_flag: bool = False) -> int:
    """Send either ``toggle`` (cheap) or ``toggle-ex`` (with options) to
    the running primary. Returns a shell-style exit code."""
    app_id = _app_id()
    path = _bus_path(app_id)

    opts: dict[str, GLib.Variant] = {}
    if profile:
        opts["profile"] = GLib.Variant("s", profile)
    if use_clipboard:
        opts["arm-clipboard"] = GLib.Variant("b", True)
    if continue_flag:
        opts["arm-continue"] = GLib.Variant("b", True)

    if opts:
        action = "toggle-ex"
        # org.gtk.Actions.Activate expects parameter as `av` (array of
        # variants). Wrap our a{sv} dict as a single variant inside it.
        params = [GLib.Variant("a{sv}", opts)]
    else:
        action = "toggle"
        params = []
    payload = GLib.Variant("(sava{sv})", (action, params, {}))

    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error as e:
        print(f"error: session bus unavailable: {e.message}", file=sys.stderr)
        return 1
    try:
        bus.call_sync(
            app_id,
            path,
            "org.gtk.Actions",
            "Activate",
            payload,
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )
    except GLib.Error as e:
        if "ServiceUnknown" in e.message or "NameHasNoOwner" in e.message:
            print(
                "error: justsayit is not running — start it first "
                "(`uv run justsayit` or via your .desktop launcher).",
                file=sys.stderr,
            )
            return 1
        print(f"error: D-Bus call failed: {e.message}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry invoked by ``_entry.main`` when the subcommand is ``toggle``.

    ``argv`` is the full post-binary argument list (so its first token
    is the subcommand name, e.g. ``["toggle", "--profile", "x"]``).
    """
    argv = sys.argv[1:] if argv is None else argv
    # Drop the subcommand token so argparse sees only the flags.
    if argv and argv[0] == "toggle":
        argv = argv[1:]

    ap = argparse.ArgumentParser(
        prog="justsayit toggle",
        description=(
            "Toggle recording on the running justsayit instance. "
            "Bind this to a compositor keybind (sway, niri, Hyprland, …) "
            "as a portal-free alternative to the XDG GlobalShortcuts path."
        ),
    )
    ap.add_argument(
        "--profile",
        default=None,
        help=(
            "switch LLM profile before toggling (persistent, like the "
            "tray menu); pass 'off' to disable postprocessing"
        ),
    )
    ap.add_argument(
        "--use-clipboard",
        action="store_true",
        default=False,
        help="arm clipboard-context for the recording this toggle starts",
    )
    ap.add_argument(
        "--continue",
        dest="continue_flag",
        action="store_true",
        default=False,
        help="start/extend continue window for LLM session continuation",
    )
    args = ap.parse_args(argv)
    return send_toggle(profile=args.profile, use_clipboard=args.use_clipboard, continue_flag=args.continue_flag)
