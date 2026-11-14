"""Score a sized UK local run's per-area representativeness (microcosm#1124 item G).

Reads a finished run directory: its evidence index, the problem, the dense and
size-refit solutions, the size receipt, and the pool frame in the graph store.
It writes two files under ``--out``, which must lie outside the repository:

- ``representativeness_by_area.csv``: for each grain, area and variable, S's
  total variation distance from D, its noise null and S's support there. It
  names areas, so it is private: never committed or published.
- ``representativeness_summary.json``: cross-area statistics only, with no area
  code, maximum or minimum, beside the digests of the artifacts it read.

A run without a size refit ships D itself. The summary then says
``not_applicable`` and no table is written.

Usage:

    uv run python tools/score_uk_local_representativeness.py \\
      --run-dir <finished run directory> --out <private directory>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microcosm.build.uk_runtime.representativeness import (
    UK_REPRESENTATIVENESS_MINIMUM_AREAS,
    UK_REPRESENTATIVENESS_MINIMUM_SOURCES,
    UK_REPRESENTATIVENESS_RATIO_THRESHOLD,
    read_uk_local_run,
    uk_area_representativeness,
    uk_representativeness_summary,
)

REPOSITORY = Path(__file__).resolve().parents[1]
TABLE_FILE = "representativeness_by_area.csv"
SUMMARY_FILE = "representativeness_summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a sized UK local run's per-area representativeness."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--graph-store",
        type=Path,
        help="The run's content store, when it is not the run's .graph-store.",
    )
    parser.add_argument(
        "--minimum-sources", type=int, default=UK_REPRESENTATIVENESS_MINIMUM_SOURCES
    )
    parser.add_argument(
        "--ratio-threshold", type=float, default=UK_REPRESENTATIVENESS_RATIO_THRESHOLD
    )
    parser.add_argument(
        "--minimum-areas", type=int, default=UK_REPRESENTATIVENESS_MINIMUM_AREAS
    )
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out == REPOSITORY or REPOSITORY in out.parents:
        parser.error(
            "--out must lie outside the repository: the per-area table names areas."
        )
    run = read_uk_local_run(args.run_dir, graph_store=args.graph_store)
    out.mkdir(parents=True, exist_ok=True)
    if run is None:
        summary = {
            "status": "not_applicable",
            "reason": "The run has no size refit: it ships the dense solution itself.",
        }
    else:
        table = uk_area_representativeness(
            run.profile,
            dense_weights=run.dense_weights,
            support=run.support,
            weights=run.weights,
        )
        table.to_csv(out / TABLE_FILE, index=False)
        summary = {
            "status": "scored",
            "artifacts_sha256": dict(run.artifacts),
            **uk_representativeness_summary(
                table,
                minimum_sources=args.minimum_sources,
                ratio_threshold=args.ratio_threshold,
                minimum_areas=args.minimum_areas,
            ),
        }
    (out / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{summary['status']}: {out / SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
