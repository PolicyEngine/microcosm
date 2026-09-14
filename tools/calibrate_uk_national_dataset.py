"""Compatibility command for the canonical UK full build.

This command has the same all-geography default as build_uk_full.
Use --target-geographies country only when explicitly requesting that filter.
The old national-only solver and its separate output controls are retired.
"""

from __future__ import annotations

import sys

_RETIRED_OPTIONS = frozenset(
    {
        "--staging-h5",
        "--diagnostics-json",
        "--build-record-json",
        "--terminal-gate-json",
        "--allow-unpinned-feed",
    }
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    retired = sorted(
        {argument.split("=", 1)[0] for argument in arguments} & _RETIRED_OPTIONS
    )
    if retired:
        raise SystemExit(
            "The independent UK national calibration driver is retired; "
            f"unsupported old option(s): {', '.join(retired)}. "
            "Use tools/build_uk_full.py --help for the canonical "
            "full-build options. Both command names default to all geographies; "
            "--target-geographies country is an explicit target filter."
        )
    from microcosm.build.uk_runtime.full_build_cli import main as full_build_main

    return full_build_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
