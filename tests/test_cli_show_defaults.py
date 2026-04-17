from __future__ import annotations

from justsayit.cli import main


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
