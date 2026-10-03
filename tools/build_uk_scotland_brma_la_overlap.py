#!/usr/bin/env python
"""Rebuild the Scottish BRMA x council-area overlap from official originals.

ONS's Price Index of Private Rents publishes Scottish rent levels for the 18
Broad Rental Market Areas, not for the 32 council areas. This tool measures
how the two geographies overlap, in households, so the question "can a BRMA
rent become a council target?" is answered from sources rather than assumed
(microcosm#1090).

Every 2022 census output area is placed in one BRMA by its population-weighted
centroid in the Scottish Government's BRMA polygons, checked against Rent
Service Scotland's postcode lookup. Output areas nest in 2022 electoral wards,
and wards nest in council areas. Two weights come out per (BRMA, council):

- ``private_rented_households``: Census 2022 ward private-rented households
  (private landlord or letting agency, plus other private rented), each ward
  split across BRMAs by the share of its output-area households in each. The
  ward is the finest level at which the census releases this tenure split, so
  inside a split ward private renters are assumed to be spread like all
  households.
- ``all_households``: output-area household counts summed directly, with no
  within-ward assumption.

The five originals are pinned by sha256 and read from ``--cache``; nothing is
downloaded. Four have stable URLs (``SOURCES``); the ward table is a manual
export from the census table builder. The geography step needs geopandas,
which is not a Microcosm dependency:

    uvx --with geopandas --with pandas --with openpyxl --with pyogrio \\
        python tools/build_uk_scotland_brma_la_overlap.py --cache <dir>

``--check`` compares the rebuild with the committed resource instead of
writing it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

RESOURCE = Path(
    "packages/microcosm-build/src/microcosm/build/uk/scotland_brma_la_overlap.json"
)
SCHEMA_VERSION = 1

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "scot_brma_boundaries_2009",
        "publisher": "Scottish Government",
        "title": "Broad Rental Market Areas (2009 boundaries, revised 28 July 2015)",
        "url": "https://maps.gov.scot/ATOM/shapefiles/SG_BroadRentalMarketAreas_2009.zip",
        "landing_page": (
            "https://www.data.gov.uk/dataset/39d08296-874a-4604-9ca4-6fe807aee4a8/"
            "broad-rental-market-areas"
        ),
        "sha256": "4d121236a35ce896edeb663a7c2c390053c0deebe0cc149c84696bbfd4bddec0",
        "licence": "Open Government Licence v3.0",
        "filename": "scot_brma_boundaries_2009.zip",
    },
    {
        "id": "scot_oa2022_population_weighted_centroids",
        "publisher": "National Records of Scotland",
        "title": "2022 Census Output Area Population Weighted Centroids",
        "url": "https://www.nrscotland.gov.uk/media/gm4dvdsv/output-area-2022-pwc.zip",
        "landing_page": (
            "https://www.nrscotland.gov.uk/publications/2022-census-geography-products/"
        ),
        "sha256": "509d7d937c8cc1477259dcd8e4e0e61240f5cc782721a79e9c279bd589e0ec0c",
        "licence": "Open Government Licence v3.0",
        "filename": "scot_oa2022_population_weighted_centroids.zip",
    },
    {
        "id": "scot_census2022_geography_index",
        "publisher": "National Records of Scotland",
        "title": "2022 Census Index: Postcode to OA and OA to higher areas",
        "url": "https://www.nrscotland.gov.uk/media/utrbt5ze/census_2022_index.zip",
        "landing_page": (
            "https://www.nrscotland.gov.uk/publications/2022-census-geography-products/"
        ),
        "sha256": "0e4096a321cba2318dc5d2e35c197bf815ac333aebdf4cb0ddce82112b57a261",
        "licence": "Open Government Licence v3.0",
        "filename": "scot_census2022_geography_index.zip",
    },
    {
        "id": "scot_foi_postcodes_brma_2023",
        "publisher": "Scottish Government / Rent Service Scotland",
        "title": "FOI 202300368850: postcode to BRMA list held in LHA Direct",
        "url": (
            "https://www.gov.scot/binaries/content/documents/govscot/publications/"
            "foi-eir-release/2023/11/foi-202300368850/documents/"
            "foi-202300368850---information-released---postcodes/"
            "foi-202300368850---information-released---postcodes/"
            "govscot%3Adocument/"
            "FOI%2B202300368850%2B-%2BInformation%2BReleased%2B-%2BPostcodes.xlsx"
        ),
        "landing_page": "https://www.gov.scot/publications/foi-202300368850/",
        "sha256": "50366fea2b87bd6b697e7aa8736fbc870838b5e327458213294f31c3de7ede5e",
        "licence": "Open Government Licence v3.0",
        "filename": "scot_foi_postcodes_brma_2023.xlsx",
    },
    {
        "id": "nrs_census_2022_ward2022_tenure_bedrooms",
        "publisher": "National Records of Scotland",
        "title": (
            "Scotland's Census 2022: household tenure by electoral ward 2022 by "
            "number of bedrooms (dataset household2022ftb)"
        ),
        "url": "https://www.scotlandscensus.gov.uk/webapi/downloadTable?type=csv",
        "landing_page": "https://www.scotlandscensus.gov.uk/webapi/home",
        "sha256": "03d6157d34bd3df44e79ded1c03510954dee55e9e05d3c276b26ad463a565677",
        "licence": "Open Government Licence v3.0",
        "filename": "census_2022_ward2022_tenure_bedrooms.csv",
        "manual": (
            "Guest session in the census table builder: all 355 Electoral Ward "
            "2022 codes as rows, Number of bedrooms (One to Five or more, and "
            "Total) as columns, Household tenure (all eight values and Total) "
            "as wafers, Counting: Households; download as CSV unmodified."
        ),
    },
)

# The Price Index of Private Rents' own codes and names for the 18 BRMAs
# (ONS monthly price statistics, July 2026 edition, Table 1).
PIPR_BRMAS: dict[str, tuple[str, str]] = {
    "ABERDEEN_AND_SHIRE": ("S33000001", "Aberdeen and Shire"),
    "ARGYLL_AND_BUTE": ("S33000002", "Argyll and Bute"),
    "AYRSHIRES": ("S33000003", "Ayrshires"),
    "DUMFRIES_AND_GALLOWAY": ("S33000004", "Dumfries and Galloway"),
    "DUNDEE_AND_ANGUS": ("S33000005", "Dundee and Angus"),
    "EAST_DUNBARTONSHIRE": ("S33000006", "East Dunbartonshire"),
    "FIFE": ("S33000007", "Fife"),
    "FORTH_VALLEY": ("S33000008", "Forth Valley"),
    "GREATER_GLASGOW": ("S33000009", "Greater Glasgow"),
    "HIGHLAND_AND_ISLANDS": ("S33000010", "Highland and Islands"),
    "LOTHIAN": ("S33000011", "Lothian"),
    "NORTH_LANARKSHIRE": ("S33000012", "North Lanarkshire"),
    "PERTH_AND_KINROSS": ("S33000013", "Perth and Kinross"),
    "RENFREWSHIRE_AND_INVERCLYDE": ("S33000014", "Renfrewshire/Inverclyde"),
    "SCOTTISH_BORDERS": ("S33000015", "Scottish Borders"),
    "SOUTH_LANARKSHIRE": ("S33000016", "South Lanarkshire"),
    "WEST_DUNBARTONSHIRE": ("S33000017", "West Dunbartonshire"),
    "WEST_LOTHIAN": ("S33000018", "West Lothian"),
}

# Two Balloch output areas whose master postcodes the lookup lists under two
# BRMAs, neither of them the older polygon's answer. LHA Direct's search for
# West Dunbartonshire council returned the Argyll and Bute and West
# Dunbartonshire BRMAs on 2026-10-02, and each postcode's candidates meet that
# pair in West Dunbartonshire only. The reply page is evidence, not an input.
OUTPUT_AREA_OVERRIDES = {
    "S00179382": "WEST_DUNBARTONSHIRE",  # G83 8NQ
    "S00179383": "WEST_DUNBARTONSHIRE",  # G83 8SD
}
OVERRIDE_COUNCIL = "S12000039"

PRIVATE_RENTED_TENURES = (
    "Private rented: Private landlord or letting agency",
    "Private rented: Other",
)
BEDROOM_BANDS = (
    "One bedroom",
    "Two bedrooms",
    "Three bedrooms",
    "Four bedrooms",
    "Five or more bedrooms",
)
EXPECTED_OUTPUT_AREAS = 46_363
EXPECTED_WARDS = 355
EXPECTED_COUNCILS = 32
#: Decimal places kept for the fractional private-rented cells: enough to
#: reconcile to the census total, few enough to be stable across platforms.
PRIVATE_RENTED_DECIMALS = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESOURCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    verify_cache(args.cache)
    payload = build_resource(args.cache)
    text = json.dumps(payload, indent=2) + "\n"
    if args.check:
        if args.out.read_text(encoding="utf-8") != text:
            print(f"stale: {args.out} differs from the rebuild", file=sys.stderr)
            return 1
        print(f"current: {args.out}")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}: {len(payload['cells'])} cells")
    return 0


def verify_cache(cache: Path) -> None:
    for source in SOURCES:
        path = cache / source["filename"]
        if not path.is_file():
            raise SystemExit(f"missing original {path} ({source['url']})")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise SystemExit(
                f"{path} has sha256 {digest}; the pin is {source['sha256']}."
            )


def build_resource(cache: Path) -> dict[str, Any]:
    output_areas = output_area_assignments(cache)
    wards = ward_private_rented_households(cache)
    cells, totals = overlap_cells(output_areas, wards)
    return {
        "schema_version": SCHEMA_VERSION,
        "resource": "scotland_brma_la_overlap",
        "description": (
            "Households in each Scottish Broad Rental Market Area x council "
            "area intersection, from Scotland's Census 2022 and the official "
            "BRMA geography. Rebuilt by tools/build_uk_scotland_brma_la_overlap.py."
        ),
        "weights": {
            "private_rented_households": (
                "Census 2022 ward households renting from a private landlord "
                "or letting agency or other private renting, summed over the "
                "published bedroom bands and split across BRMAs by each "
                "ward's share of output-area households. The ward is the "
                "finest level with this tenure split, so within a ward that "
                "crosses a BRMA boundary private renters are assumed to be "
                "spread like all households."
            ),
            "all_households": (
                "NRS 2022 output-area household counts summed per "
                "intersection; each output area has one council and one BRMA."
            ),
        },
        "method": {
            "output_area_to_brma": (
                "Population-weighted centroid within the Scottish Government "
                "BRMA polygon. A centroid outside every polygon takes its "
                "master postcode's single BRMA in Rent Service Scotland's "
                "postcode lookup. Where the lookup contradicts the polygon "
                "the newer lookup wins when it names one BRMA; the two output "
                "areas it leaves ambiguous are listed under overrides."
            ),
            "output_area_to_council": (
                "NRS Census 2022 index (OA_TO_HIGHER_AREAS.csv): 2022 "
                "electoral ward and 2019 council area of every output area."
            ),
            "overrides": dict(sorted(OUTPUT_AREA_OVERRIDES.items())),
            "private_rented_decimals": PRIVATE_RENTED_DECIMALS,
        },
        "sources": [dict(source) for source in SOURCES],
        "brmas": {
            brma: {"pipr_area_code": code, "pipr_area_name": name}
            for brma, (code, name) in sorted(PIPR_BRMAS.items())
        },
        "totals": totals,
        "cells": cells,
    }


def output_area_assignments(cache: Path):
    """One row per 2022 output area: households, ward, council and BRMA."""

    import geopandas as gpd
    import pandas as pd

    brmas = gpd.read_file(f"zip://{cache / 'scot_brma_boundaries_2009.zip'}")
    if len(brmas) != len(PIPR_BRMAS) or not brmas.geometry.is_valid.all():
        raise SystemExit("expected 18 valid BRMA polygons")
    brmas["brma"] = brmas["BRMAName"].map(_brma_key)
    if set(brmas["brma"]) != set(PIPR_BRMAS):
        raise SystemExit(f"unexpected BRMA names: {sorted(set(brmas['brma']))}")

    centroids = cache / "scot_oa2022_population_weighted_centroids.zip"
    oas = gpd.read_file(
        f"zip://{centroids}!OutputArea2022_PWC/OutputArea2022_PWC.shp"
    ).rename(columns={"code": "output_area", "HHcount": "households"})
    oas = oas.to_crs(brmas.crs)
    if len(oas) != EXPECTED_OUTPUT_AREAS or oas["output_area"].duplicated().any():
        raise SystemExit("expected 46,363 unique output areas")
    if oas["households"].isna().any() or (oas["households"] < 0).any():
        raise SystemExit("invalid output-area household counts")

    polygons = brmas[["brma", "geometry"]]
    matches = gpd.sjoin(oas, polygons, how="left", predicate="within")
    outside = matches.loc[matches["brma"].isna(), "output_area"]
    boundary = gpd.sjoin(
        oas[oas["output_area"].isin(outside)],
        polygons,
        how="left",
        predicate="intersects",
    )
    matches = pd.concat(
        [matches[~matches["output_area"].isin(outside)], boundary], ignore_index=True
    )
    hits = matches.groupby("output_area")["brma"].count()
    single = matches[matches["output_area"].map(hits).eq(1)]
    oas = oas.drop(columns="geometry").merge(
        single[["output_area", "brma"]],
        on="output_area",
        how="left",
        validate="one_to_one",
    )
    polygon_brma = oas["brma"].copy()

    lookup = pd.read_excel(
        cache / "scot_foi_postcodes_brma_2023.xlsx", usecols=[0, 1], dtype=str
    )
    lookup.columns = ["brma", "postcode"]
    lookup["postcode"] = lookup["postcode"].map(_postcode_key)
    lookup["brma"] = lookup["brma"].map(_brma_key)
    candidates = lookup.groupby("postcode")["brma"].agg(set)
    oas["candidates"] = oas["masterpc"].map(_postcode_key).map(candidates)
    unique = oas["candidates"].map(
        lambda found: (
            next(iter(found)) if isinstance(found, set) and len(found) == 1 else None
        )
    )
    oas["brma"] = oas["brma"].fillna(unique)
    if oas["brma"].isna().any():
        raise SystemExit("an output area outside the polygons has no single BRMA")

    with zipfile.ZipFile(cache / "scot_census2022_geography_index.zip") as archive:
        with archive.open("Census_2022_Index/OA_TO_HIGHER_AREAS.csv") as stream:
            higher = pd.read_csv(
                stream,
                encoding="utf-8-sig",
                usecols=["OA2022", "EW2022", "CA2019"],
            )
    oas = oas.merge(
        higher, left_on="output_area", right_on="OA2022", validate="one_to_one"
    )
    if len(oas) != EXPECTED_OUTPUT_AREAS:
        raise SystemExit("the census index does not cover every output area")

    conflict = pd.Series(
        [
            isinstance(found, set) and pd.notna(polygon) and polygon not in found
            for polygon, found in zip(polygon_brma, oas["candidates"], strict=True)
        ],
        index=oas.index,
    )
    corrected = conflict & unique.notna()
    oas.loc[corrected, "brma"] = unique[corrected]
    unresolved = conflict & unique.isna()
    if set(oas.loc[unresolved, "output_area"]) != set(OUTPUT_AREA_OVERRIDES):
        raise SystemExit("the set of ambiguous output areas changed")
    if not oas.loc[unresolved, "CA2019"].eq(OVERRIDE_COUNCIL).all():
        raise SystemExit("an overridden output area left West Dunbartonshire")
    oas.loc[unresolved, "brma"] = oas.loc[unresolved, "output_area"].map(
        OUTPUT_AREA_OVERRIDES
    )
    disagree = [
        area
        for area, brma, found in zip(
            oas["output_area"], oas["brma"], oas["candidates"], strict=True
        )
        if isinstance(found, set) and brma not in found
    ]
    if disagree:
        raise SystemExit(f"assignments contradict the postcode lookup: {disagree}")
    return oas[["output_area", "households", "EW2022", "CA2019", "brma"]].rename(
        columns={"EW2022": "ward", "CA2019": "council"}
    )


def ward_private_rented_households(cache: Path) -> dict[str, int]:
    """Ward private-rented households from the unmodified census export."""

    path = cache / "census_2022_ward2022_tenure_bedrooms.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if ["Counting: Households"] not in rows:
        raise SystemExit("the ward export does not count households")
    if any(row and row[0].strip() == "ERROR" for row in rows):
        raise SystemExit("the ward export carries suppression errors")
    tenure = None
    headers: list[str] | None = None
    wards: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    for row in rows:
        if not row:
            continue
        label = row[0].strip()
        if len(row) == 1:
            # Each wafer opens with a one-cell tenure label; resetting on
            # every one keeps other tenures out of the private-rented sum.
            tenure = label if label in PRIVATE_RENTED_TENURES else None
        elif label == "Number of bedrooms":
            headers = [header.strip() for header in row[1:]]
        elif tenure is not None and re.fullmatch(r"S13\d{6}", label):
            if headers is None:
                raise SystemExit("ward rows precede the bedroom header")
            values = dict(zip(headers, row[1:], strict=True))
            if label in seen.setdefault(tenure, set()):
                raise SystemExit(f"ward {label} repeats under {tenure!r}")
            seen[tenure].add(label)
            wards[label] = wards.get(label, 0) + sum(
                int(values[band]) for band in BEDROOM_BANDS
            )
    if set(seen) != set(PRIVATE_RENTED_TENURES) or any(
        len(found) != EXPECTED_WARDS for found in seen.values()
    ):
        raise SystemExit("expected both private-rented wafers for all 355 wards")
    return wards


def overlap_cells(
    output_areas,
    wards: dict[str, int],
    *,
    expected_councils: int = EXPECTED_COUNCILS,
):
    """Split each ward's private renters across BRMAs and sum by council."""

    councils_per_ward = output_areas.groupby("ward")["council"].nunique()
    if not councils_per_ward.eq(1).all():
        raise SystemExit("a 2022 ward spans two council areas")
    if set(output_areas["ward"]) != set(wards):
        raise SystemExit("the census wards and the geography wards differ")
    if output_areas["council"].nunique() != expected_councils:
        raise SystemExit(f"expected {expected_councils} council areas")

    ward_households = output_areas.groupby("ward")["households"].sum()
    if not ward_households.gt(0).all():
        raise SystemExit("a ward has no output-area households")
    by_ward = output_areas.groupby(["ward", "council", "brma"], as_index=False)[
        "households"
    ].sum()
    by_ward["share"] = by_ward["households"] / by_ward["ward"].map(ward_households)
    by_ward["private_rented"] = by_ward["share"] * by_ward["ward"].map(wards)
    private = by_ward.groupby(["council", "brma"])["private_rented"].sum()
    everyone = output_areas.groupby(["council", "brma"])["households"].sum()

    private_total = int(sum(wards.values()))
    if abs(float(private.sum()) - private_total) > 1e-6:
        raise SystemExit("the BRMA split changed the census private-rented total")

    cells = [
        {
            "brma": brma,
            "local_authority": council,
            "private_rented_households": round(
                float(private[(council, brma)]), PRIVATE_RENTED_DECIMALS
            ),
            "all_households": int(everyone[(council, brma)]),
        }
        for council, brma in sorted(everyone.index)
    ]
    totals = {
        "cells": len(cells),
        "output_areas": int(len(output_areas)),
        "wards": int(len(wards)),
        "wards_split_across_brmas": int(
            by_ward.groupby("ward")["brma"].nunique().gt(1).sum()
        ),
        "private_rented_households": private_total,
        "all_households": int(output_areas["households"].sum()),
    }
    return cells, totals


def _brma_key(name: str) -> str:
    key = re.sub(r"\s*/\s*", " AND ", str(name).strip()).upper().replace(" ", "_")
    return "AYRSHIRES" if key == "AYRSHIRE" else key


def _postcode_key(value: str) -> str:
    # NRS suffixes a split postcode's records with A or B; the lookup does not.
    key = re.sub(r"\s+", "", str(value).upper())
    match = re.fullmatch(r"([A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2})[A-Z]?", key)
    if not match:
        raise SystemExit(f"unrecognised postcode {value!r}")
    return match.group(1)


if __name__ == "__main__":
    raise SystemExit(main())
