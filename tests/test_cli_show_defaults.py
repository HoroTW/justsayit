from __future__ import annotations

import sys
from pathlib import Path

import pytest

from justsayit.cli import _set_process_name, main


def test_show_defaults_config_prints_commented_defaults(capsys):
    assert main(["show-defaults", "config"]) == 0

    out = capsys.readouterr().out
    assert "# justsayit configuration (commented-defaults form)." in out
    assert "[audio]" in out
    assert '# device = ""' in out
    assert "# enabled = true" in out
    assert "# filters_path = " in out
    assert "\nfilters_path = " not in out


def test_show_defaults_profile_openai_prints_shipped_openai_profile(capsys):
    assert main(["show-defaults", "profile-openai"]) == 0

    out = capsys.readouterr().out
    assert "# Profile: openai-cleanup" in out
    assert 'endpoint = "https://api.openai.com/v1"' in out
    assert 'model = "gpt-4o-mini"' in out
    assert "# remote_retries = 3" in out
    assert "# remote_retry_delay_seconds = 1.0" in out


def test_set_process_name_changes_kernel_comm():
    """End-to-end: `_set_process_name` must change `/proc/self/comm` so
    `killall justsayit` / `pgrep justsayit` (without `-f`) actually find
    the running process. The entry-point shim leaves us as `python3`
    otherwise; this is the whole point of the helper."""
    if sys.platform != "linux" or not Path("/proc/self/comm").exists():
        pytest.skip("requires Linux /proc")

    original = Path("/proc/self/comm").read_text().strip()
    try:
        _set_process_name("justsayit-test")
        actual = Path("/proc/self/comm").read_text().strip()
        # Comm is truncated to 15 bytes by the kernel; "justsayit-test" is
        # 14 chars and fits in full.
        assert actual == "justsayit-test"
    finally:
        # Restore so subsequent tests don't see a weird comm.
        _set_process_name(original)
