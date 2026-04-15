"""XDG Desktop Portal GlobalShortcuts client.

Registers a single toggle shortcut with the portal and emits a Python
callback whenever the user triggers it. Uses Gio's D-Bus binding so
everything runs on the GLib main loop that GTK already drives.

Spec: https://flatpak.github.io/xdg-desktop-portal/docs/#gdbus-org.freedesktop.portal.GlobalShortcuts
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable

import gi

gi.require_version("Gio", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gio  # noqa: E402

log = logging.getLogger(__name__)

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
GS_IFACE = "org.freedesktop.portal.GlobalShortcuts"
REQ_IFACE = "org.freedesktop.portal.Request"


ShortcutCallback = Callable[[str], None]


class PortalShortcutError(RuntimeError):
    pass


class GlobalShortcutClient:
    def __init__(
        self,
        shortcut_id: str,
        description: str,
        preferred_trigger: str,
        on_activated: ShortcutCallback,
        on_needs_rebinding: Callable[[str, str], None] | None = None,
    ) -> None:
        self.shortcut_id = shortcut_id
        self.description = description
        self.preferred_trigger = preferred_trigger
        self.on_activated = on_activated
        # Called with (shortcut_id, reason) when the portal is too old to
        # open the rebind dialog itself — so the app can surface a
        # desktop notification or similar user-visible hint.
        self.on_needs_rebinding = on_needs_rebinding

        self._bus: Gio.DBusConnection | None = None
        self._sender: str = ""  # our unique bus name with '.' -> '_'
        self._session_handle: str = ""
        self._activated_sub_id: int = 0
        self._req_subs: dict[str, int] = {}
        # Portal GlobalShortcuts interface version. 0 means "not yet
        # queried"; ConfigureShortcuts requires >= 2 (added after v1).
        self._gs_version: int = 0

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        unique = self._bus.get_unique_name()
        if not unique or not unique.startswith(":"):
            raise PortalShortcutError(f"unexpected unique name {unique!r}")
        self._sender = unique[1:].replace(".", "_")
        # Version probe happens first; _create_session runs after it
        # completes (success OR failure) so we always know whether
        # ConfigureShortcuts is available before BindShortcuts returns.
        self._query_version()

    def _query_version(self) -> None:
        assert self._bus is not None

        def _cb(bus, res):
            try:
                reply = bus.call_finish(res)
                (val,) = reply.unpack()
                self._gs_version = int(val)
                log.info("portal GlobalShortcuts version=%s", self._gs_version)
            except GLib.Error as e:
                log.warning(
                    "could not read portal GlobalShortcuts version (%s); "
                    "assuming v1 — ConfigureShortcuts will be unavailable",
                    e.message,
                )
                self._gs_version = 1
            self._create_session()

        body = GLib.Variant("(ss)", (GS_IFACE, "version"))
        self._bus.call(
            PORTAL_BUS,
            PORTAL_PATH,
            "org.freedesktop.DBus.Properties",
            "Get",
            body,
            GLib.VariantType.new("(v)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            _cb,
        )

    def stop(self) -> None:
        if self._bus is None:
            return
        if self._activated_sub_id:
            self._bus.signal_unsubscribe(self._activated_sub_id)
            self._activated_sub_id = 0
        for sid in self._req_subs.values():
            try:
                self._bus.signal_unsubscribe(sid)
            except Exception:
                pass
        self._req_subs.clear()
        self._bus = None

    # --- portal flow -------------------------------------------------------

    def _request_path(self, token: str) -> str:
        return f"/org/freedesktop/portal/desktop/request/{self._sender}/{token}"

    def _session_path(self, token: str) -> str:
        return f"/org/freedesktop/portal/desktop/session/{self._sender}/{token}"

    def _subscribe_request(
        self,
        req_path: str,
        callback: Callable[[int, dict], None],
    ) -> None:
        assert self._bus is not None

        def _on_response(
            _conn, _sender, _path, _iface, _signal, params: GLib.Variant
        ):
            response, results = params.unpack()
            sid = self._req_subs.pop(req_path, 0)
            if sid:
                self._bus.signal_unsubscribe(sid)
            callback(int(response), dict(results))

        sid = self._bus.signal_subscribe(
            PORTAL_BUS,
            REQ_IFACE,
            "Response",
            req_path,
            None,
            Gio.DBusSignalFlags.NONE,
            _on_response,
        )
        self._req_subs[req_path] = sid

    def _create_session(self) -> None:
        assert self._bus is not None
        handle_token = uuid.uuid4().hex
        session_token = uuid.uuid4().hex
        req_path = self._request_path(handle_token)

        def _on_created(response: int, results: dict) -> None:
            if response != 0:
                log.error("CreateSession failed: response=%s results=%s", response, results)
                return
            self._session_handle = results.get("session_handle") or self._session_path(
                session_token
            )
            log.info("portal session established: %s", self._session_handle)
            self._subscribe_activated()
            self._bind_shortcuts()

        self._subscribe_request(req_path, _on_created)

        options_v = GLib.Variant(
            "a{sv}",
            {
                "handle_token": GLib.Variant("s", handle_token),
                "session_handle_token": GLib.Variant("s", session_token),
            },
        )
        body = GLib.Variant.new_tuple(options_v)
        self._bus.call(
            PORTAL_BUS,
            PORTAL_PATH,
            GS_IFACE,
            "CreateSession",
            body,
            GLib.VariantType.new("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._default_async_error("CreateSession"),
        )

    def _bind_shortcuts(self) -> None:
        assert self._bus is not None
        handle_token = uuid.uuid4().hex
        req_path = self._request_path(handle_token)

        def _on_bound(response: int, results: dict) -> None:
            if response != 0:
                log.warning(
                    "BindShortcuts not accepted (response=%s). "
                    "Opening the configuration dialog so the user can "
                    "pick a trigger.",
                    response,
                )
                self._try_configure_shortcuts("bind-refused")
                return
            shortcuts = results.get("shortcuts") or []
            self._log_shortcuts(shortcuts)
            # If any shortcut came back with no trigger (e.g. the user
            # reassigned it to another app, or they've never bound one)
            # open the portal's Configure dialog so they can fix it —
            # otherwise the app looks silently broken.
            unbound = [
                sid
                for sid, props in shortcuts
                if not (props.get("trigger_description") or "").strip()
            ]
            if unbound:
                log.warning(
                    "shortcut(s) %s have no trigger assigned",
                    unbound,
                )
                self._try_configure_shortcuts("unbound-shortcut")

        self._subscribe_request(req_path, _on_bound)

        # Build each argument as its own Variant so we never embed a Variant
        # as a Python value in another Variant constructor (which would make
        # the GI override iterate the inner Variant as a dict).
        session_v = GLib.Variant("o", self._session_handle)
        shortcuts_v = GLib.Variant(
            "a(sa{sv})",
            [
                (
                    self.shortcut_id,
                    {
                        "description": GLib.Variant("s", self.description),
                        "preferred_trigger": GLib.Variant(
                            "s", self.preferred_trigger
                        ),
                    },
                )
            ],
        )
        parent_v = GLib.Variant("s", "")
        options_v = GLib.Variant(
            "a{sv}", {"handle_token": GLib.Variant("s", handle_token)}
        )
        body = GLib.Variant.new_tuple(session_v, shortcuts_v, parent_v, options_v)

        self._bus.call(
            PORTAL_BUS,
            PORTAL_PATH,
            GS_IFACE,
            "BindShortcuts",
            body,
            GLib.VariantType.new("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._default_async_error("BindShortcuts"),
        )

    def _subscribe_activated(self) -> None:
        assert self._bus is not None

        def _on_activated(
            _conn, _sender, _path, _iface, _signal, params: GLib.Variant
        ):
            session_handle, shortcut_id, _timestamp, _opts = params.unpack()
            if session_handle != self._session_handle:
                return
            log.debug("shortcut activated: %s", shortcut_id)
            try:
                self.on_activated(shortcut_id)
            except Exception:  # pragma: no cover
                log.exception("on_activated callback raised")

        self._activated_sub_id = self._bus.signal_subscribe(
            PORTAL_BUS,
            GS_IFACE,
            "Activated",
            PORTAL_PATH,
            None,
            Gio.DBusSignalFlags.NONE,
            _on_activated,
        )

    # --- configure / inspect -----------------------------------------------

    def configure(self) -> None:
        """Ask the portal to show its shortcut-configuration dialog so
        the user can (re)bind this app's shortcut. Safe to call after
        ``start()`` once the portal session has been established.
        No-ops (with a warning) if called before the session is ready,
        or if the portal is too old to support ConfigureShortcuts."""
        if self._bus is None or not self._session_handle:
            log.warning(
                "configure() called before portal session was ready; "
                "shortcut dialog cannot be opened yet"
            )
            return
        self._try_configure_shortcuts("user-request")

    def _try_configure_shortcuts(self, reason: str) -> None:
        """Call ConfigureShortcuts if the portal supports it (v2+),
        otherwise log a clear message telling the user to rebind the
        shortcut via their desktop environment's system settings."""
        if self._gs_version >= 2:
            log.info("opening portal shortcut configuration dialog (%s)", reason)
            self._configure_shortcuts()
            return
        log.warning(
            "portal GlobalShortcuts v%s does not support ConfigureShortcuts "
            "(need v2+); cannot open the rebind dialog programmatically. "
            "Please assign a trigger for '%s' in your desktop environment's "
            "shortcut settings (KDE: System Settings → Shortcuts; "
            "GNOME: Settings → Keyboard → View and Customize Shortcuts).",
            self._gs_version,
            self.shortcut_id,
        )
        if self.on_needs_rebinding is not None:
            try:
                self.on_needs_rebinding(self.shortcut_id, reason)
            except Exception:  # pragma: no cover
                log.exception("on_needs_rebinding callback raised")

    def _configure_shortcuts(self) -> None:
        assert self._bus is not None
        handle_token = uuid.uuid4().hex
        req_path = self._request_path(handle_token)

        def _on_configured(response: int, results: dict) -> None:
            if response == 0:
                log.info("shortcut configuration dialog closed (accepted)")
                # Re-query so we log the freshly-chosen trigger.
                self._list_shortcuts()
            else:
                log.info(
                    "shortcut configuration dialog closed "
                    "(response=%s — likely cancelled)",
                    response,
                )

        self._subscribe_request(req_path, _on_configured)

        session_v = GLib.Variant("o", self._session_handle)
        parent_v = GLib.Variant("s", "")
        options_v = GLib.Variant(
            "a{sv}", {"handle_token": GLib.Variant("s", handle_token)}
        )
        body = GLib.Variant.new_tuple(session_v, parent_v, options_v)
        self._bus.call(
            PORTAL_BUS,
            PORTAL_PATH,
            GS_IFACE,
            "ConfigureShortcuts",
            body,
            GLib.VariantType.new("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._default_async_error("ConfigureShortcuts"),
        )

    def _list_shortcuts(self) -> None:
        assert self._bus is not None
        handle_token = uuid.uuid4().hex
        req_path = self._request_path(handle_token)

        def _on_listed(response: int, results: dict) -> None:
            if response != 0:
                log.warning("ListShortcuts failed (response=%s)", response)
                return
            self._log_shortcuts(results.get("shortcuts") or [])

        self._subscribe_request(req_path, _on_listed)

        session_v = GLib.Variant("o", self._session_handle)
        options_v = GLib.Variant(
            "a{sv}", {"handle_token": GLib.Variant("s", handle_token)}
        )
        body = GLib.Variant.new_tuple(session_v, options_v)
        self._bus.call(
            PORTAL_BUS,
            PORTAL_PATH,
            GS_IFACE,
            "ListShortcuts",
            body,
            GLib.VariantType.new("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._default_async_error("ListShortcuts"),
        )

    @staticmethod
    def _log_shortcuts(shortcuts) -> None:
        if not shortcuts:
            log.info("portal returned zero shortcuts")
            return
        for entry in shortcuts:
            try:
                sid, props = entry
            except (TypeError, ValueError):
                log.info("portal shortcut (unparsed): %r", entry)
                continue
            trigger = (props or {}).get("trigger_description") or "(unassigned)"
            desc = (props or {}).get("description") or ""
            log.info("shortcut %s -> %s  (%s)", sid, trigger, desc)

    # --- helpers -----------------------------------------------------------

    def _default_async_error(self, op: str):
        def _cb(bus, res):
            try:
                bus.call_finish(res)
            except GLib.Error as e:
                log.error("portal call %s failed: %s", op, e.message)

        return _cb
