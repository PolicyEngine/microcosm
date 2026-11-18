#!/usr/bin/env python3
"""Compare two tax years at the same IRS filing-season processing week."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime.filing_season import filing_season_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "facts", type=Path, help="Chronicle consumer artifact or JSONL feed"
    )
    parser.add_argument("--first-year", type=int, required=True)
    parser.add_argument("--last-year", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = load_ledger_consumer_artifact(args.facts)
    report = filing_season_comparison(
        artifact.facts,
        first_year=args.first_year,
        last_year=args.last_year,
        week=args.week,
    )
    report["facts_sha256"] = artifact.facts_sha256
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
