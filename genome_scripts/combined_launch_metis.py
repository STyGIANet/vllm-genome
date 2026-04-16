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

import sys


def main() -> None:
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
