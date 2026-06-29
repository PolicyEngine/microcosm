"""Build the pooled CPS ASEC source-side US unit frame.

This is a pre-calibration support construction step. It pools raw ASEC years
before unit assignment and writes diagnostics proving source-year shares. It
does not run donor imputations, fiscal calibration, or release certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.us_runtime import AsecSource, build_pooled_asec_unit_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asec-h5",
        action="append",
        required=True,
        help="ASEC source as YEAR=PATH. Pass once per source year.",
    )
    parser.add_argument("--target-year", type=int, default=2024)
    parser.add_argument(
        "--max-households",
        type=int,
        help="Optional smoke limit applied to every ASEC source.",
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--out-h5",
        type=Path,
        help=(
            "Optional raw pooled-unit H5. This is not a calibrated release H5; "
            "it is a source-side support artifact for downstream stages."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sources = tuple(
        _parse_source(value, max_households=args.max_households)
        for value in args.asec_h5
    )
    frame, metadata = build_pooled_asec_unit_frame(
        sources,
        target_year=args.target_year,
    )
    summary = _summary(frame, metadata)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.out_h5 is not None:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

        args.out_h5.parent.mkdir(parents=True, exist_ok=True)
        PolicyEngineUSEngine().write_dataset(
            frame, args.out_h5, period=args.target_year
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_source(value: str, *, max_households: int | None) -> AsecSource:
    if "=" not in value:
        raise ValueError(f"ASEC source must be YEAR=PATH, got {value!r}.")
    raw_year, raw_path = value.split("=", 1)
    year = int(raw_year)
    path = Path(raw_path)
    return AsecSource(year=year, path=path, max_households=max_households)


def _summary(frame, metadata: dict) -> dict[str, object]:
    person = frame.table("person")
    household = frame.table("household")
    weights = frame.weights_for("household").values
    household_pos = pd.Series(
        np.arange(len(household)), index=household["household_id"]
    )
    person_household_pos = household_pos.reindex(
        person["person_household_id"]
    ).to_numpy(dtype=np.int64)
    person_weight = weights[person_household_pos]
    by_year = (
        pd.DataFrame(
            {
                "source_year": person["source_year"].to_numpy(),
                "person_weight": person_weight,
            }
        )
        .groupby("source_year")["person_weight"]
        .sum()
    )
    return {
        "metadata": metadata,
        "rows": {entity: frame.n(entity) for entity in frame.entities},
        "household_weight_total": float(weights.sum()),
        "weighted_person_population_by_year": {
            str(int(year)): float(value) for year, value in by_year.items()
        },
        "tax_unit_filing_status_counts": {
            str(status): int(count)
            for status, count in frame.table("tax_unit")["filing_status_input"]
            .value_counts()
            .items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
