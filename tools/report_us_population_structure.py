#!/usr/bin/env python3
"""Write the held-out population-structure report for any US release H5.

The fiscal-refresh release writes ``us_population_structure_holdout.json``
for new builds. This tool runs the same function on an already published or
local H5, so a release built before the check existed can be audited without
a rebuild. It reads the H5's person and household tables only; no engine.

    uv run python tools/report_us_population_structure.py \\
        populace_us_2024.h5 --release-id populace-us-2024-spm-20260909 \\
        --out us_population_structure_holdout.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from microcosm.build.us_runtime.population_structure_holdout import (
    population_structure_holdout_from_h5,
    write_population_structure_holdout,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="US release H5")
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("us_population_structure_holdout.json"),
    )
    args = parser.parse_args(argv)
    payload = population_structure_holdout_from_h5(
        args.dataset, period=args.period, release_id=args.release_id
    )
    path = write_population_structure_holdout(payload, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
