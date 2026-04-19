#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the callback-based METIS placement example.

This keeps ``combined_launch.py`` as the original mixed launcher that still
supports:

- tracking only
- JSON-driven placement
- callback-driven placement

This wrapper exists only to provide a clean METIS-specific entry point. It
forwards all arguments to ``combined_launch.py`` while forcing
``--callback-placement`` and rejecting the JSON placement path.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys


def _ensure_pymetis_installed() -> None:
    """Install pymetis into the active repo environment when it is missing."""
    if importlib.util.find_spec("pymetis") is not None:
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = ["uv", "pip", "install", "pymetis"]
    print("[METIS] pymetis not found; installing with:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Failed to install pymetis with `uv pip install pymetis`."
        ) from exc

    if importlib.util.find_spec("pymetis") is None:
        raise SystemExit(
            "pymetis still unavailable after installation attempt."
        )


def main() -> None:
    _ensure_pymetis_installed()
    argv = sys.argv[1:]
    if "--expert-placement-config" in argv:
        raise SystemExit(
            "--expert-placement-config is not supported "
        )
    if "--callback-placement" not in argv:
        argv = [*argv, "--callback-placement"]

    # Import after argv is prepared so combined_launch parses the forced flag.
    import combined_launch

    sys.argv = [sys.argv[0], *argv]
    combined_launch.main()


if __name__ == "__main__":
    main()
