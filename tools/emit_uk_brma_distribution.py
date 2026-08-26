"""Emit disclosure-safe BRMA distribution diagnostics for a UK frame.

The comparison runs at the sampler's native margin — (region, LHA category)
cells — because that is where the assignment is defined: within a cell the
built benunit-level BRMA shares are multinomial around the count table's
conditional distribution. A region-level margin would mix our benunit
category composition with the rents table's lettings-weighted composition
and mislead.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.frs_brma import (
    UK_BRMA_DECLARED_SEEDS,
    _benunit_regions,
    _enum_name,
    assign_brma_by_cell,
    load_brma_count_resource,
)
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)


def brma_cell_distribution(
    benunit: pd.DataFrame,
    *,
    count_resource: Mapping[str, object],
    minimum_count: int = 3,
) -> dict[str, object]:
    """Compare built benunit BRMA shares per (region, LHA category) cell."""

    required = {"region", "LHA_category", "brma"}
    missing = required - set(benunit.columns)
    if missing:
        raise ValueError(f"benunit table is missing column(s): {sorted(missing)}.")
    cells = count_resource["cells"]
    rows: list[dict[str, object]] = []
    max_abs_z = 0.0
    for (region, category), group in benunit.groupby(
        ["region", "LHA_category"], sort=True
    ):
        expected_counts = cells.get(str(region), {}).get(str(category))
        if expected_counts is None:
            raise KeyError(
                f"missing count-table cell for region={region!r}, "
                f"LHA_category={category!r}."
            )
        expected_total = float(sum(expected_counts.values()))
        built_counts = group["brma"].astype(str).value_counts().to_dict()
        cell_n = int(len(group))
        for brma, expected_count in sorted(expected_counts.items()):
            expected_share = float(expected_count) / expected_total
            built_count = int(built_counts.get(brma, 0))
            row: dict[str, object] = {
                "region": str(region),
                "lha_category": str(category),
                "brma": str(brma),
                "cell_n": cell_n,
                "expected_share": expected_share,
            }
            if 0 < built_count < minimum_count:
                row["built_count"] = f"<{minimum_count}"
                row["built_share"] = None
                row["z"] = None
            else:
                built_share = built_count / cell_n if cell_n else 0.0
                sigma = (
                    math.sqrt(expected_share * (1.0 - expected_share) / cell_n)
                    if cell_n and 0.0 < expected_share < 1.0
                    else None
                )
                z = (built_share - expected_share) / sigma if sigma else None
                row["built_count"] = built_count
                row["built_share"] = built_share
                row["z"] = z
                if z is not None:
                    max_abs_z = max(max_abs_z, abs(z))
            rows.append(row)
    return {
        "check": "uk_brma_cell_distribution",
        "margin": "benunit BRMA within (region, LHA category) cells",
        "minimum_count": minimum_count,
        "max_abs_z": max_abs_z,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=3)
    args = parser.parse_args()

    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    frame, _provenance = load_uk_national_frame(args.input_h5)
    engine = PolicyEngineUKEngine()
    lha_category = engine.materialize(frame, ("LHA_category",), uk_time_period(frame))[
        "LHA_category"
    ]
    benunit = frame.table("benunit").copy()
    benunit["LHA_category"] = [_enum_name(value) for value in lha_category]
    benunit["region"] = _benunit_regions(
        frame.table("person"), frame.table("household"), benunit
    )
    count_resource = load_brma_count_resource()
    # Deterministic re-derivation of the benunit-level assignment that fed the
    # stored household collapse (proven identical by the identity receipt).
    benunit["brma"] = assign_brma_by_cell(
        benunit,
        count_resource=count_resource,
        seed=UK_BRMA_DECLARED_SEEDS["brma"],
    )
    payload = brma_cell_distribution(
        benunit,
        count_resource=count_resource,
        minimum_count=args.minimum_count,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("brma cell distribution: max |z| =", round(payload["max_abs_z"], 2))


if __name__ == "__main__":
    main()
