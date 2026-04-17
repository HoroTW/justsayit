#!/usr/bin/env python3
"""Bump justsayit's version string across every file that hardcodes it.

The version lives in three places that have to stay in lock-step
(otherwise `pyproject` says one thing, the running app reports another,
and `nix build` produces a third). This script edits all of them
atomically: each file is read, the exact `version = "<old>"` (or
`__version__ = "<old>"`) line is rewritten, and the file is written
back. If any file's pattern fails to match (e.g. a line was renamed
upstream), the script aborts before touching anything else, so you
don't end up with half a bumped repo.

Run from the repo root:

    ./scripts/bump-version.py 0.8.1
    ./scripts/bump-version.py 0.8.1 --no-uv-sync   # skip uv.lock refresh

`uv.lock` is refreshed via `uv sync` at the end (skipped with the flag);
CHANGELOG.md is left to you — drafting an entry is the human's job.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Strict semver — script targets X.Y.Z releases (no pre-release tags).
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class VersionFile:
    """One file that hardcodes the project version.

    *path* is relative to the repo root. *pattern* must contain a single
    capture group around the version string (so we can both read the
    current value and surgically replace it).
    """
    path: Path
    pattern: re.Pattern[str]
    label: str  # what to call this in error messages


# Add a new VersionFile here whenever a fresh place starts hardcoding
# the version. Order is irrelevant — every file is processed atomically.
FILES: list[VersionFile] = [
    VersionFile(
        REPO_ROOT / "pyproject.toml",
        # Anchored to start-of-line so we don't accidentally match e.g.
        # `requires-python = ">=3.11"` or some dependency's version line.
        re.compile(r'^(version\s*=\s*")(\d+\.\d+\.\d+)(")', re.MULTILINE),
        "pyproject.toml [project] version",
    ),
    VersionFile(
        REPO_ROOT / "src" / "justsayit" / "__init__.py",
        re.compile(r'^(__version__\s*=\s*")(\d+\.\d+\.\d+)(")', re.MULTILINE),
        "src/justsayit/__init__.py __version__",
    ),
    VersionFile(
        REPO_ROOT / "flake.nix",
        # The flake also overrides llama-cpp-python's version on a nearby
        # line with the same 8-space indent, so indent alone can't pick
        # out justsayit's. Anchor on the preceding `pname = "justsayit";`
        # line — that pair is unique to the buildPythonApplication block.
        re.compile(
            r'(pname\s*=\s*"justsayit";\s*\n\s*version\s*=\s*")(\d+\.\d+\.\d+)(")',
        ),
        "flake.nix mkJustsayit version",
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="new version, in X.Y.Z form")
    ap.add_argument(
        "--no-uv-sync",
        action="store_true",
        help="skip the trailing `uv sync` that refreshes uv.lock",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change but don't write anything",
    )
    return ap.parse_args(argv)


def read_current_version(file: VersionFile) -> str:
    text = file.path.read_text(encoding="utf-8")
    m = file.pattern.search(text)
    if m is None:
        raise SystemExit(
            f"error: {file.label} ({file.path}) — version pattern "
            f"didn't match. Has the file been restructured? Update "
            f"FILES in scripts/bump-version.py."
        )
    return m.group(2)


def rewrite(file: VersionFile, new: str, *, dry_run: bool) -> tuple[str, str]:
    """Replace the version in *file*. Returns (old, new). Atomic per file."""
    text = file.path.read_text(encoding="utf-8")
    new_text, n = file.pattern.subn(rf"\g<1>{new}\g<3>", text)
    if n == 0:
        raise SystemExit(f"error: {file.label} — pattern matched on read but not on subn?!")
    if n > 1:
        raise SystemExit(
            f"error: {file.label} — pattern matched {n} lines, expected exactly 1. "
            f"Tighten the regex in FILES."
        )
    old = file.pattern.search(text).group(2)
    if not dry_run:
        file.path.write_text(new_text, encoding="utf-8")
    return old, new


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    new_version: str = args.version

    if not SEMVER_RE.match(new_version):
        print(
            f"error: {new_version!r} is not X.Y.Z semver "
            "(no pre-release/build suffixes supported here).",
            file=sys.stderr,
        )
        return 2

    # Pre-flight: make sure every file's pattern matches BEFORE writing
    # anything. Catches "you renamed flake.nix's version line" cleanly,
    # without leaving the repo half-bumped.
    current_versions = []
    for f in FILES:
        if not f.path.exists():
            print(f"error: {f.path} not found", file=sys.stderr)
            return 1
        current_versions.append((f, read_current_version(f)))

    # Surface inconsistencies in the current state — they happen when a
    # previous bump missed a file. Don't bail; just warn loudly so the
    # bump itself realigns everything.
    distinct = {v for _, v in current_versions}
    if len(distinct) > 1:
        print(
            f"warning: files are currently OUT OF SYNC: {sorted(distinct)} "
            "— this bump will realign all of them.",
            file=sys.stderr,
        )

    print(f"bumping to {new_version}:")
    for f in FILES:
        old, new = rewrite(f, new_version, dry_run=args.dry_run)
        marker = "would update" if args.dry_run else "updated"
        if old == new:
            print(f"  {marker} (no-op): {f.label} already at {new}")
        else:
            print(f"  {marker}: {f.label}: {old} -> {new}")

    if args.dry_run:
        print("\ndry run — no files written.")
        return 0

    if not args.no_uv_sync:
        # uv sync refreshes the project's own row in uv.lock to the new
        # version. The dev extra includes pytest etc; matches what the
        # devShell uses so we don't churn unrelated lock entries.
        print("\nrefreshing uv.lock via `uv sync --extra dev`…")
        try:
            subprocess.run(
                ["uv", "sync", "--extra", "dev"],
                check=True,
                cwd=REPO_ROOT,
            )
        except FileNotFoundError:
            print(
                "warning: `uv` not on PATH — skipping uv.lock refresh. "
                "Run it yourself before committing.",
                file=sys.stderr,
            )
        except subprocess.CalledProcessError as e:
            print(f"error: uv sync failed (exit {e.returncode}).", file=sys.stderr)
            return e.returncode

    print(
        "\nNext steps:\n"
        f"  1. Add a [{new_version}] section to CHANGELOG.md\n"
        "  2. git add -p && git commit\n"
        "  (Reminder: don't add Co-Authored-By trailers in this repo.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
