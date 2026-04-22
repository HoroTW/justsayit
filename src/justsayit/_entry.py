"""Thin argv dispatcher.

The only module-level imports are ``sys`` — everything else is loaded
lazily based on the subcommand. Lets ``justsayit toggle`` skip the
full ``cli`` import chain (Gtk, numpy, sherpa-onnx, llama-cpp …),
shaving several hundred ms off keyboard-shortcut-bound invocations.
"""

from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    for tok in argv:
        if tok.startswith("-"):
            continue
        if tok == "toggle":
            from justsayit.toggle_client import main as toggle_main

            return toggle_main(argv)
        break
    from justsayit.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
