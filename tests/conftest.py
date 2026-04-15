"""Shared pytest configuration.

Sets environment flags before any test module is imported so that
cli.py's module-level re-exec guards (systemd scope, LD_PRELOAD) are
skipped.  Without these, importing ``justsayit.cli`` in tests would
attempt to re-exec the test process under a systemd scope.
"""

import os

import pytest

os.environ.setdefault("_JUSTSAYIT_SCOPED", "1")
os.environ.setdefault("_JUSTSAYIT_PRELOADED", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: mark test as requiring live network access (HuggingFace API)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--no-network",
        action="store_true",
        default=False,
        help="skip tests that require network access",
    )


@pytest.fixture(autouse=False)
def network(request):
    if request.config.getoption("--no-network"):
        pytest.skip("network tests disabled (--no-network)")
