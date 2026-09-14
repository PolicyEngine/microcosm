"""Compatibility command for the canonical UK full build, all targets by default.

Geography-only cloning/export orchestration is retired. Runtime cloning and
geography helpers remain available for analysis and graph-owned population work.
"""

from __future__ import annotations

import sys

_RETIRED_OPTIONS = frozenset(
    {
        "--crosswalk",
        "--constituency-codes",
        "--la-codes",
        "--candidate-clone-counts",
        "--dataset-filename",
        "--allow-missing-country",
        "--allow-blank-constituency",
        "--allow-cross-region-assignment",
        "--allow-constituency-collisions",
    }
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    retired = sorted(
        {argument.split("=", 1)[0] for argument in arguments} & _RETIRED_OPTIONS
    )
    if retired:
        raise SystemExit(
            "The independent UK geography-only build driver is retired; "
            f"unsupported old controls: {', '.join(retired)}. "
            "Use tools/build_uk_full.py with a bound --input-h5 or --spine-request, "
            "--ladder, --ledger-facts and --out. --n-clones controls geography K; "
            "--dry-run describes the full graph. All target geographies remain the default."
        )
    from microcosm.build.uk_runtime.full_build_cli import main as full_main

    return full_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
