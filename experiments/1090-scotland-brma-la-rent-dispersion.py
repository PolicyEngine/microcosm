#!/usr/bin/env python
"""How far a BRMA's mean rent sits from its councils' mean rents (#1090).

ONS's Price Index of Private Rents gives one rent level per Scottish Broad
Rental Market Area. This script asks what a council-grain target built from
those levels would get wrong, using the only Scottish rent records that carry
both a BRMA and a finer location: Rent Service Scotland's market evidence for
the year to September 2021 (FOI 202200303624), one row per letting with its
BRMA, weekly rent and postcode district.

For each council it compares

- the candidate rule: the mean of the BRMA means, weighted by the council's
  private-rented households in each BRMA (``scotland_brma_la_overlap.json``);
- the council's own mean rent in the same records.

A record is placed in councils by the household shares of its (postcode
district, BRMA) pair among 2022 output areas; a pair no output area carries
falls back to the district's shares. That placement is approximate, so the
receipt reports magnitudes, not council rent statistics.

    uvx --with geopandas --with pandas --with openpyxl --with pyogrio \\
        python experiments/1090-scotland-brma-la-rent-dispersion.py --cache <dir>

``--cache`` holds the originals the overlap tool pins, plus the two below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.build_uk_scotland_brma_la_overlap import (  # noqa: E402
    PIPR_BRMAS,
    RESOURCE,
    _brma_key,
    output_area_assignments,
    verify_cache,
)

RECEIPT = Path(__file__).with_suffix(".json")
EVIDENCE = {
    "filename": "scot_foi_brma_2016_2021.xlsx",
    "sha256": "dba86fa1cce920c8f646460bb13e1d99e16927d57fdf290e874f9631e804ed06",
    "landing_page": "https://www.gov.scot/publications/foi-202200303624/",
    "sheet": "2021",
}
PIPR = {
    "filename": "pipr_monthly_price_statistics_july2026_edition.xlsx",
    "sha256": "d429e553ade454903b731f6c257609f14d0c2171822791e09c21a9d53d5e24a9",
    "landing_page": (
        "https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/"
        "priceindexofprivaterentsukmonthlypricestatistics"
    ),
    "sheet": "Table 1",
}
#: Councils with fewer placed records than this are left out of the gap table.
MIN_RECORDS = 30
WEEKS_PER_MONTH = 52 / 12


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RECEIPT)
    args = parser.parse_args(argv)

    verify_cache(args.cache)
    for extra in (EVIDENCE, PIPR):
        path = args.cache / extra["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != extra["sha256"]:
            raise SystemExit(f"{path} does not match its pin")

    overlap = json.loads((REPO / RESOURCE).read_text(encoding="utf-8"))
    cells = pd.DataFrame(overlap["cells"])
    council_total = cells.groupby("local_authority")[
        "private_rented_households"
    ].transform("sum")
    cells["share_of_council"] = cells["private_rented_households"] / council_total

    records = _market_evidence(args.cache)
    placed, district_only = _place_records(records, args.cache)
    brma_mean = records.groupby("brma")["monthly_rent"].mean()
    council = (
        placed.assign(weighted=placed["monthly_rent"] * placed["weight"])
        .groupby("council")[["weighted", "weight"]]
        .sum()
    )
    council["records_mean"] = council["weighted"] / council["weight"]

    cells["brma_records_mean"] = cells["brma"].map(brma_mean)
    rule = (
        (cells["share_of_council"] * cells["brma_records_mean"])
        .groupby(cells["local_authority"])
        .sum()
    )
    pipr = _pipr_2025_means(args.cache)
    cells["brma_pipr_2025"] = cells["brma"].map(pipr)
    rule_pipr = (
        (cells["share_of_council"] * cells["brma_pipr_2025"])
        .groupby(cells["local_authority"])
        .sum()
    )
    main_brma = cells.loc[
        cells.groupby("local_authority")["share_of_council"].idxmax()
    ].set_index("local_authority")

    rows = []
    for code in sorted(rule.index):
        n = float(council["weight"].get(code, 0.0))
        own = float(council["records_mean"].get(code, float("nan")))
        row = {
            "local_authority": code,
            "main_brma": str(main_brma.at[code, "brma"]),
            "share_of_council_in_main_brma": round(
                float(main_brma.at[code, "share_of_council"]), 4
            ),
            "placed_records": round(n, 1),
            "rule_value_pipr_2025_monthly": round(float(rule_pipr[code]), 2),
        }
        if n >= MIN_RECORDS:
            row["records_mean_monthly"] = round(own, 2)
            row["rule_value_records_monthly"] = round(float(rule[code]), 2)
            row["rule_over_records_mean_pct"] = round(
                100 * (float(rule[code]) / own - 1), 1
            )
        rows.append(row)

    gaps = sorted(
        abs(row["rule_over_records_mean_pct"])
        for row in rows
        if "rule_over_records_mean_pct" in row
    )
    receipt = {
        "issue": "PolicyEngine/microcosm#1090",
        "question": (
            "What would a council target built as the household-weighted mean "
            "of BRMA rents get wrong?"
        ),
        "evidence": EVIDENCE,
        "pipr": PIPR,
        "overlap_resource": str(RESOURCE),
        "records": int(len(records)),
        "records_placed_by_district_only": district_only,
        "min_records_per_council": MIN_RECORDS,
        "brma_means": {
            brma: {
                "records_mean_monthly": round(float(brma_mean[brma]), 2),
                "records": int((records["brma"] == brma).sum()),
                "pipr_2025_calendar_mean_monthly": round(float(pipr[brma]), 2),
            }
            for brma in sorted(PIPR_BRMAS)
        },
        "summary": {
            "councils_compared": len(gaps),
            "median_abs_gap_pct": gaps[len(gaps) // 2],
            "max_abs_gap_pct": gaps[-1],
            "councils_over_5_pct": sum(gap > 5 for gap in gaps),
            "councils_over_10_pct": sum(gap > 10 for gap in gaps),
        },
        "councils": rows,
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["summary"], indent=2))
    return 0


def _market_evidence(cache: Path):
    import pandas as pd

    frame = pd.read_excel(
        cache / EVIDENCE["filename"], sheet_name=EVIDENCE["sheet"], dtype=str
    )
    if set(frame["FREQUENCY"]) != {"Weekly"}:
        raise SystemExit("expected weekly rents only")
    frame["monthly_rent"] = frame["NET RENT"].astype(float) * WEEKS_PER_MONTH
    frame["brma"] = frame["BRMA"].map(_brma_key)
    if set(frame["brma"]) != set(PIPR_BRMAS):
        raise SystemExit("unexpected BRMA names in the market evidence")
    frame["district"] = frame["POST CODE"].str.strip().str.upper()
    return frame[["brma", "district", "monthly_rent"]]


def _place_records(records, cache: Path):
    import geopandas as gpd
    import pandas as pd

    areas = output_area_assignments(cache)
    centroids = cache / "scot_oa2022_population_weighted_centroids.zip"
    postcodes = gpd.read_file(
        f"zip://{centroids}!OutputArea2022_PWC/OutputArea2022_PWC.shp",
        ignore_geometry=True,
    )[["code", "masterpc"]]
    areas = areas.merge(
        postcodes, left_on="output_area", right_on="code", validate="one_to_one"
    )
    areas["district"] = areas["masterpc"].str.split().str[0].str.upper()

    def shares(keys: list[str]):
        grouped = areas.groupby([*keys, "council"])["households"].sum()
        total = grouped.groupby(level=list(range(len(keys)))).transform("sum")
        return (grouped / total).rename("weight").reset_index()

    exact = records.reset_index().merge(
        shares(["district", "brma"]), on=["district", "brma"], how="left"
    )
    missing = exact["weight"].isna()
    fallback = exact.loc[missing, ["index", "brma", "district", "monthly_rent"]].merge(
        shares(["district"]), on="district", how="left"
    )
    if fallback["weight"].isna().any():
        raise SystemExit("a record's postcode district has no output area")
    placed = pd.concat([exact.loc[~missing], fallback], ignore_index=True)
    if abs(placed["weight"].sum() - len(records)) > 1e-6:
        raise SystemExit("record placement lost or duplicated records")
    return placed, int(missing.sum())


def _pipr_2025_means(cache: Path):
    import pandas as pd

    frame = pd.read_excel(
        cache / PIPR["filename"],
        sheet_name=PIPR["sheet"],
        header=2,
        usecols=["Time period", "Area code", "Rental price"],
    )
    codes = {code: brma for brma, (code, _name) in PIPR_BRMAS.items()}
    frame = frame[frame["Area code"].isin(codes)]
    frame = frame[pd.to_datetime(frame["Time period"]).dt.year == 2025]
    counts = frame.groupby("Area code")["Rental price"].count()
    if len(counts) != len(codes) or not counts.eq(12).all():
        raise SystemExit("expected twelve 2025 months for all 18 BRMAs")
    means = frame.groupby("Area code")["Rental price"].mean()
    return means.rename(index=codes)


if __name__ == "__main__":
    raise SystemExit(main())
