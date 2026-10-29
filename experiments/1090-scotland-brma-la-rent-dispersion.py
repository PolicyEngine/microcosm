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
falls back to the district's shares. Each output area's BRMA comes from the
committed area-to-BRMA layer, its council from the committed Scotland support,
its households from Census 2022 UV404 and its postcode district from its
population-weighted centroid's master postcode. That placement is
approximate, so the receipt reports magnitudes, not council rent statistics.
PIPR's calendar-2025 BRMA means are the feed's twelve monthly facts.

    uvx --with geopandas --with pandas --with numpy --with openpyxl --with pyogrio \\
        python experiments/1090-scotland-brma-la-rent-dispersion.py \\
        --chronicle-data <chronicle>/db/data \\
        --ledger-facts <feed artifact>/consumer_facts.jsonl \\
        --supports <directory of the atomic support NPZs>
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
    LAYER,
    PIPR_NAMES,
    RESOURCE,
    layer_module,
    output_area_councils,
    uv404_counts,
)

RECEIPT = Path(__file__).with_suffix(".json")
EVIDENCE = {
    "directory": "scotgov/brma_market_evidence_2016_2021",
    "filename": "scot_foi_brma_2016_2021.xlsx",
    "sha256": "dba86fa1cce920c8f646460bb13e1d99e16927d57fdf290e874f9631e804ed06",
    "landing_page": "https://www.gov.scot/publications/foi-202200303624/",
    "sheet": "2021",
}
CENTROIDS = {
    "directory": "nrs/oa2022_population_weighted_centroids",
    "filename": "output-area-2022-pwc.zip",
    "sha256": "509d7d937c8cc1477259dcd8e4e0e61240f5cc782721a79e9c279bd589e0ec0c",
    "layer": "OutputArea2022_PWC/OutputArea2022_PWC.shp",
}
#: PIPR's monthly BRMA rents for 2025 in the pinned feed (September 2026 edition).
PIPR = {
    "record_sets": "ons.pipr_september2026.2025-{01..12}.rental_price",
    "geography": {"level": "statistical_scope", "vintage": "brma_scotland"},
}
#: Councils with fewer placed records than this are left out of the gap table.
MIN_RECORDS = 30
WEEKS_PER_MONTH = 52 / 12


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chronicle-data", type=Path, required=True)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--supports", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RECEIPT)
    args = parser.parse_args(argv)

    for pin in (EVIDENCE, CENTROIDS):
        path = args.chronicle_data / pin["directory"] / pin["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            raise SystemExit(f"{path} does not match its pin")

    overlap = json.loads(RESOURCE.read_text(encoding="utf-8"))
    cells = pd.DataFrame(overlap["cells"])
    council_total = cells.groupby("local_authority")[
        "private_rented_households"
    ].transform("sum")
    cells["share_of_council"] = cells["private_rented_households"] / council_total

    records = _market_evidence(args.chronicle_data)
    areas = _output_areas(args)
    placed, district_only = _place_records(records, areas)
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
    codes = {brma: entry["pipr_area_code"] for brma, entry in overlap["brmas"].items()}
    pipr = _pipr_2025_means(args.ledger_facts, codes)
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
        "centroids": CENTROIDS,
        "pipr": PIPR,
        "overlap_resource": str(RESOURCE.relative_to(REPO)),
        "layer_sha256": hashlib.sha256(LAYER.read_bytes()).hexdigest(),
        "records": int(len(records)),
        "records_placed_by_district_only": district_only,
        "min_records_per_council": MIN_RECORDS,
        "brma_means": {
            brma: {
                "records_mean_monthly": round(float(brma_mean[brma]), 2),
                "records": int((records["brma"] == brma).sum()),
                "pipr_2025_calendar_mean_monthly": round(float(pipr[brma]), 2),
            }
            for brma in sorted(PIPR_NAMES)
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


def _market_evidence(chronicle_data: Path):
    import pandas as pd

    frame = pd.read_excel(
        chronicle_data / EVIDENCE["directory"] / EVIDENCE["filename"],
        sheet_name=EVIDENCE["sheet"],
        dtype=str,
    )
    if set(frame["FREQUENCY"].str.strip()) != {"Weekly"}:
        raise SystemExit("expected weekly rents only")
    frame["monthly_rent"] = frame["NET RENT"].astype(float) * WEEKS_PER_MONTH
    frame["brma"] = frame["BRMA"].map(layer_module.list_brma_key)
    if set(frame["brma"]) != set(PIPR_NAMES):
        raise SystemExit("unexpected BRMA names in the market evidence")
    frame["district"] = frame["POST CODE"].str.strip().str.upper()
    return frame[["brma", "district", "monthly_rent"]]


def _output_areas(args: argparse.Namespace):
    """Each Scottish output area's BRMA, council, households and district."""

    import geopandas as gpd
    import pandas as pd

    layer = layer_module.load_brma_area_layer(LAYER.read_bytes())
    brma_by_area = layer.brma_by_area["scotland_oa22"]
    councils = output_area_councils(args.supports)
    households = uv404_counts(args.ledger_facts)["all_households"]
    path = args.chronicle_data / CENTROIDS["directory"] / CENTROIDS["filename"]
    postcodes = gpd.read_file(
        f"zip://{path}!{CENTROIDS['layer']}", ignore_geometry=True
    )[["code", "masterpc"]]
    areas = pd.DataFrame(
        {
            "output_area": list(brma_by_area),
            "brma": list(brma_by_area.values()),
        }
    )
    areas["council"] = areas["output_area"].map(councils)
    areas["households"] = areas["output_area"].map(households)
    if areas[["council", "households"]].isna().any().any():
        raise SystemExit("an output area has no council or household count")
    areas = areas.merge(
        postcodes, left_on="output_area", right_on="code", validate="one_to_one"
    )
    if len(areas) != len(brma_by_area):
        raise SystemExit("an output area has no centroid")
    areas["district"] = areas["masterpc"].str.split().str[0].str.upper()
    return areas


def _place_records(records, areas):
    import pandas as pd

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


def _pipr_2025_means(path: Path, codes: dict[str, str]):
    import pandas as pd

    months = {f"ons.pipr_september2026.2025-{m:02d}.rental_price" for m in range(1, 13)}
    brma_by_code = {code: brma for brma, code in codes.items()}
    values: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"brma_scotland"' not in line:
                continue
            fact = json.loads(line)
            if (fact.get("layout") or {}).get("record_set_id") not in months:
                continue
            geography = fact["geography"]
            if (
                geography.get("level") != PIPR["geography"]["level"]
                or geography.get("vintage") != PIPR["geography"]["vintage"]
                or fact.get("dimensions")
            ):
                continue
            brma = brma_by_code.get(str(geography["id"]))
            if brma is None:
                raise SystemExit(f"PIPR names an unknown BRMA {geography['id']}")
            values.setdefault(brma, []).append(float(fact["value"]))
    if set(values) != set(codes) or any(len(v) != 12 for v in values.values()):
        raise SystemExit("expected twelve 2025 months for all 18 BRMAs")
    return pd.Series({brma: sum(v) / 12 for brma, v in values.items()})


if __name__ == "__main__":
    raise SystemExit(main())
