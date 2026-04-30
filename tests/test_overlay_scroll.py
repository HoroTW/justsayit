"""Structural tests for the scrolled result content area."""

from __future__ import annotations

import inspect


def test_overlay_max_height_default_is_1000():
    """OverlayConfig.max_height default must be 1000 so the scrolled content
    cap is generous enough to show long LLM responses."""
    from justsayit.config import Config
    cfg = Config()
    assert cfg.overlay.max_height == 1000


def test_overlay_init_creates_scrolled_window():
    """OverlayWindow.__init__ must construct a Gtk.ScrolledWindow and assign it
    to self._content_scroll with the max_content_height set."""
    from justsayit.overlay import OverlayWindow
    src = inspect.getsource(OverlayWindow.__init__)
    assert "Gtk.ScrolledWindow" in src
    assert "_content_scroll" in src
    assert "set_max_content_height" in src
    assert "set_propagate_natural_height" in src


def test_overlay_init_uses_content_box():
    """The labels must be appended to a content box inside the scroll, not
    directly to root."""
    from justsayit.overlay import OverlayWindow
    src = inspect.getsource(OverlayWindow.__init__)
    assert "_content_box" in src
    assert "set_child(_content_box)" in src


def test_overlay_hide_text_areas_hides_scroll():
    """_hide_text_areas must also hide the _content_scroll container."""
    from justsayit.overlay import OverlayWindow
    src = inspect.getsource(OverlayWindow._hide_text_areas)
    assert "_content_scroll" in src


def test_expand_window_simplified():
    """_expand_window must no longer contain manual pixel arithmetic (char
    width calculations were removed); it should just call set_default_size."""
    from justsayit.overlay import OverlayWindow
    src = inspect.getsource(OverlayWindow._expand_window)
    assert "set_default_size" in src
    # The old char-width-based arithmetic is gone.
    assert "_CHAR_WIDTH_PX" not in src
    assert "chars_per_line" not in src
