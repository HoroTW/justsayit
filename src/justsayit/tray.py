"""System tray icon via org.kde.StatusNotifierItem + com.canonical.dbusmenu.

We talk the SNI protocol directly over GDBus instead of going through
AppIndicator, because AppIndicator's Python binding pulls in Gtk 3 and
our overlay already pinned Gtk 4 with ``gi.require_version``. GI's
version lock is process-global, so the two cannot coexist in one
interpreter.

Works on KDE Plasma out of the box. GNOME users need the
"AppIndicator and KStatusNotifierItem Support" shell extension.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Callable

from gi.repository import Gio, GLib

log = logging.getLogger(__name__)

SNI_IFACE = "org.kde.StatusNotifierItem"
SNI_ITEM_PATH = "/StatusNotifierItem"
SNI_WATCHER_NAME = "org.kde.StatusNotifierWatcher"
SNI_WATCHER_PATH = "/StatusNotifierWatcher"
SNI_WATCHER_IFACE = "org.kde.StatusNotifierWatcher"

DBUSMENU_IFACE = "com.canonical.dbusmenu"
DBUSMENU_PATH = "/MenuBar"

SNI_XML = """<node>
  <interface name="org.kde.StatusNotifierItem">
    <property name="Category" type="s" access="read"/>
    <property name="Id" type="s" access="read"/>
    <property name="Title" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconName" type="s" access="read"/>
    <property name="IconPixmap" type="a(iiay)" access="read"/>
    <property name="OverlayIconName" type="s" access="read"/>
    <property name="AttentionIconName" type="s" access="read"/>
    <property name="ToolTip" type="(sa(iiay)ss)" access="read"/>
    <property name="Menu" type="o" access="read"/>
    <property name="ItemIsMenu" type="b" access="read"/>
    <method name="Activate">
      <arg type="i" direction="in"/>
      <arg type="i" direction="in"/>
    </method>
    <method name="SecondaryActivate">
      <arg type="i" direction="in"/>
      <arg type="i" direction="in"/>
    </method>
    <method name="ContextMenu">
      <arg type="i" direction="in"/>
      <arg type="i" direction="in"/>
    </method>
    <method name="Scroll">
      <arg type="i" direction="in"/>
      <arg type="s" direction="in"/>
    </method>
    <signal name="NewTitle"/>
    <signal name="NewIcon"/>
    <signal name="NewAttentionIcon"/>
    <signal name="NewOverlayIcon"/>
    <signal name="NewStatus">
      <arg type="s"/>
    </signal>
    <signal name="NewToolTip"/>
  </interface>
</node>"""

DBUSMENU_XML = """<node>
  <interface name="com.canonical.dbusmenu">
    <property name="Version" type="u" access="read"/>
    <property name="TextDirection" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconThemePath" type="as" access="read"/>
    <method name="GetLayout">
      <arg type="i" name="parentId" direction="in"/>
      <arg type="i" name="recursionDepth" direction="in"/>
      <arg type="as" name="propertyNames" direction="in"/>
      <arg type="u" name="revision" direction="out"/>
      <arg type="(ia{sv}av)" name="layout" direction="out"/>
    </method>
    <method name="GetGroupProperties">
      <arg type="ai" name="ids" direction="in"/>
      <arg type="as" name="propertyNames" direction="in"/>
      <arg type="a(ia{sv})" name="properties" direction="out"/>
    </method>
    <method name="GetProperty">
      <arg type="i" name="id" direction="in"/>
      <arg type="s" name="name" direction="in"/>
      <arg type="v" name="value" direction="out"/>
    </method>
    <method name="Event">
      <arg type="i" name="id" direction="in"/>
      <arg type="s" name="eventId" direction="in"/>
      <arg type="v" name="data" direction="in"/>
      <arg type="u" name="timestamp" direction="in"/>
    </method>
    <method name="EventGroup">
      <arg type="a(isvu)" name="events" direction="in"/>
      <arg type="ai" name="idErrors" direction="out"/>
    </method>
    <method name="AboutToShow">
      <arg type="i" name="id" direction="in"/>
      <arg type="b" name="needUpdate" direction="out"/>
    </method>
    <method name="AboutToShowGroup">
      <arg type="ai" name="ids" direction="in"/>
      <arg type="ai" name="updatesNeeded" direction="out"/>
      <arg type="ai" name="idErrors" direction="out"/>
    </method>
    <signal name="ItemsPropertiesUpdated">
      <arg type="a(ia{sv})"/>
      <arg type="a(ias)"/>
    </signal>
    <signal name="LayoutUpdated">
      <arg type="u"/>
      <arg type="i"/>
    </signal>
    <signal name="ItemActivationRequested">
      <arg type="i"/>
      <arg type="u"/>
    </signal>
  </interface>
</node>"""


@dataclass
class MenuItem:
    """One entry in the tray menu. ``id`` must be > 0 and unique."""

    id: int
    label: str = ""
    is_separator: bool = False
    toggle_type: str | None = None  # "checkmark" or None
    toggle_state: int = -1  # 0 off, 1 on, -1 indeterminate
    enabled: bool = True
    visible: bool = True
    on_activate: Callable[[], None] | None = None

    def to_props(self) -> dict[str, GLib.Variant]:
        if self.is_separator:
            return {"type": GLib.Variant("s", "separator")}
        props: dict[str, GLib.Variant] = {}
        if self.label:
            props["label"] = GLib.Variant("s", self.label)
        if not self.enabled:
            props["enabled"] = GLib.Variant("b", False)
        if not self.visible:
            props["visible"] = GLib.Variant("b", False)
        if self.toggle_type:
            props["toggle-type"] = GLib.Variant("s", self.toggle_type)
            props["toggle-state"] = GLib.Variant("i", self.toggle_state)
        return props


class TrayIcon:
    """Minimal StatusNotifierItem + dbusmenu implementation.

    Call ``start()`` once the GLib main loop is running. Menu item
    callbacks are invoked on the main GLib context, so they can touch
    GTK / other GLib state directly.
    """

    def __init__(
        self,
        *,
        icon_name: str = "audio-input-microphone",
        title: str = "justsayit",
        item_id: str = "justsayit",
        tooltip: str = "justsayit dictation",
        items: list[MenuItem] | None = None,
    ) -> None:
        self.icon_name = icon_name
        self.title = title
        self.item_id = item_id
        self.tooltip = tooltip
        self.items = items or []
        self._item_by_id: dict[int, MenuItem] = {it.id: it for it in self.items}

        self._conn: Gio.DBusConnection | None = None
        self._sni_reg_id = 0
        self._menu_reg_id = 0
        self._own_name_id = 0
        self._bus_name = ""
        self._menu_revision = 1

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self._bus_name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"

        sni_info = Gio.DBusNodeInfo.new_for_xml(SNI_XML).interfaces[0]
        menu_info = Gio.DBusNodeInfo.new_for_xml(DBUSMENU_XML).interfaces[0]

        self._sni_reg_id = self._conn.register_object(
            SNI_ITEM_PATH,
            sni_info,
            self._sni_method_call,
            self._sni_get_property,
            None,
        )
        self._menu_reg_id = self._conn.register_object(
            DBUSMENU_PATH,
            menu_info,
            self._menu_method_call,
            self._menu_get_property,
            None,
        )
        log.info("tray: exported SNI + dbusmenu objects")

        self._own_name_id = Gio.bus_own_name_on_connection(
            self._conn,
            self._bus_name,
            Gio.BusNameOwnerFlags.NONE,
            self._on_name_acquired,
            self._on_name_lost,
        )

    def stop(self) -> None:
        if self._conn is None:
            return
        if self._sni_reg_id:
            self._conn.unregister_object(self._sni_reg_id)
            self._sni_reg_id = 0
        if self._menu_reg_id:
            self._conn.unregister_object(self._menu_reg_id)
            self._menu_reg_id = 0
        if self._own_name_id:
            Gio.bus_unown_name(self._own_name_id)
            self._own_name_id = 0

    # --- public surface ----------------------------------------------------

    def set_icon(self, icon_name: str) -> None:
        if icon_name == self.icon_name:
            return
        self.icon_name = icon_name
        if self._conn is not None:
            self._conn.emit_signal(
                None, SNI_ITEM_PATH, SNI_IFACE, "NewIcon", None
            )

    def set_tooltip(self, tooltip: str) -> None:
        if tooltip == self.tooltip:
            return
        self.tooltip = tooltip
        if self._conn is not None:
            self._conn.emit_signal(
                None, SNI_ITEM_PATH, SNI_IFACE, "NewToolTip", None
            )

    def update_item(self, item_id: int, **fields) -> None:
        """Mutate a menu item's fields (``label``, ``toggle_state`` …) and
        tell the client to re-query the layout."""
        it = self._item_by_id.get(item_id)
        if it is None:
            log.warning("tray: update_item for unknown id %d", item_id)
            return
        for k, v in fields.items():
            setattr(it, k, v)
        if self._conn is not None:
            self._menu_revision += 1
            self._conn.emit_signal(
                None,
                DBUSMENU_PATH,
                DBUSMENU_IFACE,
                "LayoutUpdated",
                GLib.Variant("(ui)", (self._menu_revision, 0)),
            )

    # --- name lifecycle ----------------------------------------------------

    def _on_name_acquired(self, conn: Gio.DBusConnection, name: str) -> None:
        log.info("tray: acquired bus name %s", name)
        try:
            conn.call_sync(
                SNI_WATCHER_NAME,
                SNI_WATCHER_PATH,
                SNI_WATCHER_IFACE,
                "RegisterStatusNotifierItem",
                GLib.Variant("(s)", (name,)),
                None,
                Gio.DBusCallFlags.NONE,
                5000,
                None,
            )
            log.info("tray: registered with StatusNotifierWatcher")
        except GLib.Error as e:
            log.warning(
                "tray: could not register with watcher (%s). On GNOME, "
                "enable the AppIndicator/KStatusNotifierItem extension.",
                e.message,
            )

    def _on_name_lost(self, conn: Gio.DBusConnection, name: str) -> None:
        log.warning("tray: lost bus name %s", name)

    # --- StatusNotifierItem handlers --------------------------------------

    def _sni_method_call(
        self,
        conn: Gio.DBusConnection,
        sender: str,
        path: str,
        iface: str,
        method: str,
        params: GLib.Variant,
        invocation: Gio.DBusMethodInvocation,
    ) -> None:
        if method == "Activate":
            # Left-click: same as the "Auto listen" menu item (toggle first
            # checkable menu item if any; otherwise no-op).
            self._fire_primary_toggle()
        elif method in ("SecondaryActivate", "ContextMenu", "Scroll"):
            pass
        invocation.return_value(None)

    def _sni_get_property(
        self,
        conn: Gio.DBusConnection,
        sender: str,
        path: str,
        iface: str,
        prop: str,
    ) -> GLib.Variant | None:
        if prop == "Category":
            return GLib.Variant("s", "ApplicationStatus")
        if prop == "Id":
            return GLib.Variant("s", self.item_id)
        if prop == "Title":
            return GLib.Variant("s", self.title)
        if prop == "Status":
            return GLib.Variant("s", "Active")
        if prop == "IconName":
            return GLib.Variant("s", self.icon_name)
        if prop == "IconPixmap":
            return GLib.Variant("a(iiay)", [])
        if prop == "OverlayIconName":
            return GLib.Variant("s", "")
        if prop == "AttentionIconName":
            return GLib.Variant("s", "")
        if prop == "ToolTip":
            return GLib.Variant(
                "(sa(iiay)ss)", ("", [], self.title, self.tooltip)
            )
        if prop == "Menu":
            return GLib.Variant("o", DBUSMENU_PATH)
        if prop == "ItemIsMenu":
            return GLib.Variant("b", False)
        return None

    def _fire_primary_toggle(self) -> None:
        for it in self.items:
            if it.toggle_type == "checkmark" and it.on_activate is not None:
                try:
                    it.on_activate()
                except Exception:
                    log.exception("primary toggle callback failed")
                return

    # --- dbusmenu handlers -------------------------------------------------

    def _menu_get_property(
        self,
        conn: Gio.DBusConnection,
        sender: str,
        path: str,
        iface: str,
        prop: str,
    ) -> GLib.Variant | None:
        if prop == "Version":
            return GLib.Variant("u", 3)
        if prop == "TextDirection":
            return GLib.Variant("s", "ltr")
        if prop == "Status":
            return GLib.Variant("s", "normal")
        if prop == "IconThemePath":
            return GLib.Variant("as", [])
        return None

    def _menu_method_call(
        self,
        conn: Gio.DBusConnection,
        sender: str,
        path: str,
        iface: str,
        method: str,
        params: GLib.Variant,
        invocation: Gio.DBusMethodInvocation,
    ) -> None:
        try:
            if method == "GetLayout":
                parent_id, _depth, _prop_names = params.unpack()
                layout = self._build_layout(int(parent_id))
                invocation.return_value(
                    GLib.Variant("(u(ia{sv}av))", (self._menu_revision, layout))
                )
            elif method == "GetGroupProperties":
                ids, _prop_names = params.unpack()
                result = []
                for raw_id in ids:
                    id_ = int(raw_id)
                    if id_ == 0:
                        result.append(
                            (0, {"children-display": GLib.Variant("s", "submenu")})
                        )
                        continue
                    it = self._item_by_id.get(id_)
                    if it is None:
                        continue
                    result.append((id_, it.to_props()))
                invocation.return_value(
                    GLib.Variant("(a(ia{sv}))", (result,))
                )
            elif method == "GetProperty":
                id_, name = params.unpack()
                id_ = int(id_)
                if id_ == 0:
                    if name == "children-display":
                        invocation.return_value(
                            GLib.Variant("(v)", (GLib.Variant("s", "submenu"),))
                        )
                        return
                    invocation.return_error_literal(
                        Gio.dbus_error_quark(),
                        Gio.DBusError.INVALID_ARGS,
                        f"no property {name} on root",
                    )
                    return
                it = self._item_by_id.get(id_)
                if it is None:
                    invocation.return_error_literal(
                        Gio.dbus_error_quark(),
                        Gio.DBusError.INVALID_ARGS,
                        f"unknown id {id_}",
                    )
                    return
                props = it.to_props()
                if name in props:
                    invocation.return_value(
                        GLib.Variant("(v)", (props[name],))
                    )
                else:
                    invocation.return_error_literal(
                        Gio.dbus_error_quark(),
                        Gio.DBusError.INVALID_ARGS,
                        f"no property {name} on id {id_}",
                    )
            elif method == "Event":
                id_, event_id, _data, _ts = params.unpack()
                if str(event_id) == "clicked":
                    self._fire_activate(int(id_))
                invocation.return_value(None)
            elif method == "EventGroup":
                (events,) = params.unpack()
                errors: list[int] = []
                for raw_id, event_id, _data, _ts in events:
                    id_ = int(raw_id)
                    if str(event_id) != "clicked":
                        continue
                    try:
                        self._fire_activate(id_)
                    except Exception:
                        log.exception("menu activate failed for id %d", id_)
                        errors.append(id_)
                invocation.return_value(GLib.Variant("(ai)", (errors,)))
            elif method == "AboutToShow":
                invocation.return_value(GLib.Variant("(b)", (False,)))
            elif method == "AboutToShowGroup":
                invocation.return_value(GLib.Variant("(aiai)", ([], [])))
            else:
                invocation.return_value(None)
        except Exception:
            log.exception("dbusmenu method %s raised", method)
            invocation.return_error_literal(
                Gio.dbus_error_quark(),
                Gio.DBusError.FAILED,
                f"{method} failed",
            )

    def _build_layout(self, parent_id: int) -> tuple:
        """Return ``(id, props_dict, children_av)`` — a{sv}-style."""
        if parent_id == 0:
            root_props = {"children-display": GLib.Variant("s", "submenu")}
            children: list[GLib.Variant] = [
                GLib.Variant("(ia{sv}av)", (it.id, it.to_props(), []))
                for it in self.items
            ]
            return (0, root_props, children)
        it = self._item_by_id.get(parent_id)
        if it is None:
            return (parent_id, {}, [])
        return (parent_id, it.to_props(), [])

    def _fire_activate(self, item_id: int) -> None:
        it = self._item_by_id.get(item_id)
        if it is None:
            log.warning("tray: activate for unknown id %d", item_id)
            return
        if it.on_activate is None:
            return
        try:
            it.on_activate()
        except Exception:
            log.exception("menu item activate raised")


def open_with_xdg(path: str) -> None:
    """Fire-and-forget ``xdg-open`` — used from menu callbacks."""
    try:
        subprocess.Popen(
            ["xdg-open", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        log.error("xdg-open not found on PATH")
    except Exception:
        log.exception("xdg-open failed for %s", path)
