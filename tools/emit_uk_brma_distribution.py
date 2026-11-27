"""Emit private BRMA distribution diagnostics for a UK frame's count-table draws.

Since area_brma took frs_brma's place (microcosm#931), only Northern Ireland's
BRMAs are drawn from the census count table: Great Britain's are read off each
household's drawn area through the area-to-BRMA layer, so they hold no draw
to compare. The comparison runs over Northern Ireland's benefit units at the
sampler's native margin, (region, LHA category) cells, because that is where
the assignment is defined: within a cell the built benefit-unit BRMA shares
are multinomial around the count table's conditional distribution. A
region-level margin would mix our benefit-unit category composition with the
census table's private-rented-household composition and mislead.

Disclosure (UKDS CD137 clause 8). Every count is of distinct true source
households, a household's clones and copies counted once (``receipt_counts``):
a cell's ``cell_n`` and each BRMA's ``built_count``, through the shared
rounding (``rounded_count``). A BRMA that one or two source households carry
withholds every figure of its cell, its counts, shares and z, since the
others' would give it back by difference. The shares and z read benefit
units, so as exact ratios they would give a cell's unit counts back: a share
is floored to the hundredth (``floored_share``) and z rounded to one decimal,
and both are withheld in a cell resting on fewer than ten source households.
The output stays private: ``--output`` must lie outside every git work tree.
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
from microcosm.build.uk_runtime.geographic_clones import geographic_clone_count
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    distinct_carriers,
    floored_share,
    outside_work_trees,
    record_source_households,
    rounded_count,
)

#: Each benefit unit's true source household, which every count counts by
#: (receipt_counts), so a household's clones and copies count once.
SOURCE_COLUMN = "source_household"
#: The one region whose BRMAs the count table draws since area_brma
#: (microcosm#931).
NORTHERN_IRELAND = "NORTHERN_IRELAND"
#: The fewest distinct source households a cell rests on for its shares and
#: z to print: below it its counts read "3-9", and its benefit-unit shares,
#: even floored, would narrow its unit counts.
SHARE_MINIMUM_SOURCES = 10
#: The decimals z is rounded to.
Z_DECIMALS = 1


def _sources(count: int, minimum_count: int) -> int | str:
    """A count of source households: withheld below ``minimum_count``, else
    rounded by the shared rules."""

    if 0 < count < minimum_count:
        return f"<{minimum_count}"
    return rounded_count(count)


def brma_cell_distribution(
    benunit: pd.DataFrame,
    *,
    count_resource: Mapping[str, object],
    minimum_count: int = UK_DISCLOSURE_FLOOR,
) -> dict[str, object]:
    """Compare built benefit-unit BRMA shares per (region, LHA category) cell.

    ``benunit`` holds Northern Ireland's benefit units alone, each with its
    ``region``, ``LHA_category``, ``brma`` and true source household
    (``SOURCE_COLUMN``). ``cell_n`` and ``built_count`` count source
    households, rounded; a BRMA carried by fewer than ``minimum_count`` of
    them withholds every figure of its cell, each count, share and z reading
    ``<minimum_count`` or ``None``. The shares and z read benefit units, the
    sampler's own unit: a share is floored to the hundredth and z rounded to
    one decimal, and both are withheld (``None``) in a cell resting on fewer
    than ``SHARE_MINIMUM_SOURCES`` source households. ``max_abs_z`` reads the
    z that print.

    Refused: a missing column, a row outside Northern Ireland, a
    ``minimum_count`` below the UK disclosure floor, and a cell the count
    table lacks.
    """

    required = {"region", "LHA_category", "brma", SOURCE_COLUMN}
    missing = required - set(benunit.columns)
    if missing:
        raise ValueError(f"benunit table is missing column(s): {sorted(missing)}.")
    if int(minimum_count) < UK_DISCLOSURE_FLOOR:
        raise ValueError(
            f"minimum_count cannot be below the UK disclosure floor, "
            f"{UK_DISCLOSURE_FLOOR}."
        )
    if not benunit["region"].astype(str).eq(NORTHERN_IRELAND).all():
        raise ValueError(
            "Only Northern Ireland's BRMAs are drawn from the count table since "
            "area_brma (microcosm#931); Great Britain's are read off their drawn "
            "areas, so the comparison takes Northern Ireland's benefit units "
            "alone."
        )
    cells = count_resource["cells"]
    rows: list[dict[str, object]] = []
    max_abs_z = 0.0
    withheld_cells = 0
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
        brmas = group["brma"].astype(str)
        built_units = brmas.value_counts().to_dict()
        carriers = {
            str(brma): distinct_carriers(sources)
            for brma, sources in group.groupby(brmas)[SOURCE_COLUMN]
        }
        # One withheld BRMA withholds every figure of its cell: the others'
        # counts and shares would give it back by difference.
        withheld = any(0 < carried < minimum_count for carried in carriers.values())
        withheld_cells += int(withheld)
        marker = f"<{int(minimum_count)}"
        units = int(len(group))
        sources = distinct_carriers(group[SOURCE_COLUMN])
        # A thin cell prints its rounded counts but no share or z.
        shares_withheld = withheld or sources < SHARE_MINIMUM_SOURCES
        cell_n = marker if withheld else _sources(sources, minimum_count)
        for brma, expected_count in sorted(expected_counts.items()):
            expected_share = float(expected_count) / expected_total
            carried = int(carriers.get(str(brma), 0))
            row: dict[str, object] = {
                "region": str(region),
                "lha_category": str(category),
                "brma": str(brma),
                "cell_n": cell_n,
                "expected_share": expected_share,
                "built_count": marker if withheld else _sources(carried, minimum_count),
            }
            if shares_withheld:
                row["built_share"] = None
                row["z"] = None
            else:
                built_share = int(built_units.get(brma, 0)) / units if units else 0.0
                sigma = (
                    math.sqrt(expected_share * (1.0 - expected_share) / units)
                    if units and 0.0 < expected_share < 1.0
                    else None
                )
                z = (built_share - expected_share) / sigma if sigma else None
                row["built_share"] = floored_share(built_share)
                row["z"] = None if z is None else round(z, Z_DECIMALS)
                if row["z"] is not None:
                    max_abs_z = max(max_abs_z, abs(row["z"]))
            rows.append(row)
    return {
        "check": "uk_brma_cell_distribution",
        "private": True,
        "margin": (
            "Northern Ireland's benefit-unit BRMA within (region, LHA category) cells"
        ),
        "counts": (
            "cell_n and built_count are distinct true source households, rounded; "
            f"a BRMA below {int(minimum_count)} withholds every figure of its "
            "cell; built_share, a benefit-unit share, is floored to the "
            f"hundredth and z rounded to {Z_DECIMALS} decimal, both withheld in "
            f"a cell of fewer than {SHARE_MINIMUM_SOURCES} source households"
        ),
        "minimum_count": int(minimum_count),
        "cells_withheld": withheld_cells,
        "max_abs_z": max_abs_z,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="The private diagnostics file, outside every git work tree.",
    )
    parser.add_argument("--minimum-count", type=int, default=UK_DISCLOSURE_FLOOR)
    args = parser.parse_args()
    if args.minimum_count < UK_DISCLOSURE_FLOOR:
        parser.error(
            f"--minimum-count cannot be below the UK disclosure floor, "
            f"{UK_DISCLOSURE_FLOOR}."
        )
    output = outside_work_trees(args.output, what="the BRMA distribution diagnostics")

    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    frame, _provenance = load_uk_national_frame(args.input_h5)
    # The engine runs one simulation per geographic clone and refuses a frame
    # of another clone count (microcosm#931).
    engine = PolicyEngineUKEngine(
        clones=geographic_clone_count(frame.table("household"))
    )
    lha_category = engine.materialize(frame, ("LHA_category",), uk_time_period(frame))[
        "LHA_category"
    ]
    benunit = frame.table("benunit").copy()
    benunit["LHA_category"] = [_enum_name(value) for value in lha_category]
    benunit["region"] = _benunit_regions(
        frame.table("person"), frame.table("household"), benunit
    )
    benunit[SOURCE_COLUMN] = record_source_households(
        frame, "benunit", stage="The BRMA distribution diagnostics"
    )
    benunit = benunit.loc[benunit["region"].eq(NORTHERN_IRELAND)].reset_index(drop=True)
    count_resource = load_brma_count_resource()
    # Deterministic re-derivation of the benefit-unit draw that fed each
    # Northern Ireland household's pick (area_brma's, and spi_copy_areas' on
    # the SPI copies), keyed on each unit's own id.
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
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("brma cell distribution: max |z| =", payload["max_abs_z"])


if __name__ == "__main__":
    main()
