"""Audit IRS PUF aggregate-row disaggregation with Microcosm-native code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from microcosm.build.us_runtime.puf_aggregate_records import (
    audit_puf_aggregate_disaggregation,
    disaggregate_puf_aggregate_records,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw PUF totals against the old aggregate-row drop path and "
            "the Microcosm aggregate-row disaggregation path."
        )
    )
    parser.add_argument(
        "--puf-csv",
        type=Path,
        required=True,
        help="Path to the IRS PUF CSV containing RECID, S006, and PUF amount columns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path for the JSON audit report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260529,
        help="Seed for synthetic donor selection.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help=(
            "Optional PUF amount columns to include in the source-loss surface. "
            "Defaults to every PUF amount column."
        ),
    )
    parser.add_argument(
        "--write-disaggregated-csv",
        type=Path,
        default=None,
        help="Optional path for the transformed PUF CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    puf = pd.read_csv(args.puf_csv, low_memory=False)
    audit = audit_puf_aggregate_disaggregation(
        puf,
        seed=args.seed,
        columns=args.columns,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if args.write_disaggregated_csv is not None:
        args.write_disaggregated_csv.parent.mkdir(parents=True, exist_ok=True)
        disaggregate_puf_aggregate_records(puf, seed=args.seed).to_csv(
            args.write_disaggregated_csv,
            index=False,
        )

    summary = {
        "audit": str(args.out),
        "old_drop_aggregate_loss": audit["source_reconstruction_loss"][
            "old_drop_aggregate"
        ]["loss"],
        "disaggregated_loss": audit["source_reconstruction_loss"]["disaggregated"][
            "loss"
        ],
        "raw_aggregate_rows": audit["raw_aggregate_rows"],
        "synthetic_rows": audit["synthetic_rows"],
    }
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
