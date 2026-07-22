"""Regenerate or check the UK local-target census artifact (populace#495).

The census inventories the UK local calibration surface: every household-side
metric ``local_targets`` computes today, the official UK local statistics that
could supply target values for each family, and the reviewed fences that
constrain binding. The committed JSON lives next to the other UK evidence
artifacts in ``populace.build.uk`` and is drift-gated by the test suite.

Usage:
    uv run python tools/census_uk_local_targets.py            # rewrite committed
    uv run python tools/census_uk_local_targets.py --check    # verify, no write
    uv run python tools/census_uk_local_targets.py --out x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from populace.build.uk_runtime.local_target_census import (
    assert_uk_local_target_census_current,
    build_uk_local_target_census,
    committed_uk_local_target_census_path,
    write_uk_local_target_census,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output JSON path. Defaults to the committed artifact in populace.build.uk."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the artifact matches the live surface; write nothing.",
    )
    args = parser.parse_args(argv)
    path = args.out if args.out is not None else committed_uk_local_target_census_path()

    if args.check:
        try:
            assert_uk_local_target_census_current(path)
        except (ValueError, OSError) as exc:
            print(f"stale: {exc}", file=sys.stderr)
            return 1
        print(f"current: {path}")
        return 0

    written = write_uk_local_target_census(path)
    census = build_uk_local_target_census()
    summary = {
        "path": str(written),
        "metrics": len(census["metrics"]),
        "families": len(census["families"]),
        "sources": len(census["sources"]),
        "binding_fences": len(census["binding_fences"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
