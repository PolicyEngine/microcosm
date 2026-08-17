"""Emit disclosure-safe BRMA distribution diagnostics for a UK frame."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.frs_brma import load_brma_count_resource
from microcosm.build.uk_runtime.national_build import load_uk_national_frame


def brma_distribution(
    household: pd.DataFrame,
    *,
    count_resource: Mapping[str, object],
    minimum_count: int = 3,
) -> dict[str, object]:
    """Compare built household BRMA shares with count-table region priors."""

    required = {"region", "brma"}
    missing = required - set(household.columns)
    if missing:
        raise ValueError(f"household table is missing column(s): {sorted(missing)}.")
    rows: list[dict[str, object]] = []
    cells = count_resource["cells"]
    for region, region_table in sorted(cells.items()):
        expected_counts: dict[str, int] = {}
        for category_counts in region_table.values():
            for brma, count in category_counts.items():
                expected_counts[brma] = expected_counts.get(brma, 0) + int(count)
        built = household.loc[household["region"].astype(str) == str(region), "brma"]
        built_counts = built.astype(str).value_counts().to_dict()
        built_total = int(sum(built_counts.values()))
        expected_total = int(sum(expected_counts.values()))
        for brma, expected_count in sorted(expected_counts.items()):
            built_count = int(built_counts.get(brma, 0))
            if 0 < built_count < minimum_count:
                built_count_out: int | str = f"<{minimum_count}"
                built_share: float | None = None
            else:
                built_count_out = built_count
                built_share = built_count / built_total if built_total else 0.0
            rows.append(
                {
                    "region": region,
                    "brma": brma,
                    "built_count": built_count_out,
                    "built_share": built_share,
                    "count_table_share": expected_count / expected_total,
                }
            )
    return {
        "check": "uk_brma_distribution",
        "minimum_count": minimum_count,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=3)
    args = parser.parse_args()
    frame, _provenance = load_uk_national_frame(args.input_h5)
    payload = brma_distribution(
        frame.table("household"),
        count_resource=load_brma_count_resource(),
        minimum_count=args.minimum_count,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
