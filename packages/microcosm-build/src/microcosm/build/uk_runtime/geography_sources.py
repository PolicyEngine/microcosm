"""Official-source helpers for complete UK row-wise geography crosswalks.

The row-wise assignment module stays source-agnostic. This module composes
public geography source tables into the explicit crosswalk frame consumed by
those primitives, without importing the incumbent UK data package.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from microcosm.build.uk_runtime.rowwise_geography import (
    CROSSWALK_COLUMNS,
    prepare_geography_crosswalk,
)

EW_OA_HIERARCHY_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "b9ca90c10aaa4b8d9791e9859a38ca67/csv?layers=0"
)
EW_OA_LAD23_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "83982ff4a8144038be52be65dd2b8fa0/csv?layers=0"
)
EW_OA_CONSTITUENCY_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "5968b5b2c0f14dd29ba277beaae6dec3/csv?layers=0"
)
EW_OA_POPULATION_URL = (
    "https://www.nomisweb.co.uk/output/census/2021/census2021-ts001.zip"
)
ENGLAND_LAD_REGION_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "78b348cd8fb04037ada3c862aa054428/csv?layers=0"
)

SCOTLAND_OA_DZ_IZ_URL = (
    "https://www.nrscotland.gov.uk/media/iz3evrqt/oa22_dz22_iz22.zip"
)
SCOTLAND_OA_LAU_ITL_URL = (
    "https://www.nrscotland.gov.uk/media/qsxon3dm/oa22_lau25_itl25.zip"
)
SCOTLAND_OA_CONSTITUENCY_URL = (
    "https://www.nrscotland.gov.uk/media/njkmhppf/oa22_ukpc24.zip"
)
SCOTLAND_OA_POPULATION_URL = (
    "https://www.nrscotland.gov.uk/media/owpknvgk/"
    "outputarea2022_usualresidentpopulation.csv"
)

NI_DZ_GEOJSON_ZIP_URL = (
    "https://www.nisra.gov.uk/files/nisra/publications/geography-dz2021-geojson.zip"
)
NI_DZ_POPULATION_CSV_URL = (
    "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=DZ21"
)
NI_DZ_HOUSEHOLDS_CSV_URL = (
    "https://build.nisra.gov.uk/en/custom/table.csv?d=HOUSEHOLD&v=DZ21"
)

#: NRS Census 2022 index (2022 Census Geography Products register). One zip
#: carries both Scotland OA-ladder layers: OA_TO_HIGHER_AREAS.csv maps every
#: OA2022 to its 2022 Electoral Ward (EW2022, point-in-polygon centroid per
#: the in-zip index file specification), and Postcode_To_OA.csv carries the
#: spec-defined 2022 Census occupied household count per postcode (cell-key
#: perturbed), summed to the OA for the ladder's stage-one draw weight.
SCOTLAND_CENSUS_INDEX_ZIP_URL = (
    "https://www.nrscotland.gov.uk/media/utrbt5ze/census_2022_index.zip"
)

UK_POSTCODE_OA_MAY25_ZIP_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "7fc55d71a09d4dcfa1fd6473138aacc3/data"
)
UK_POSTCODE_PCON_MAY24_ZIP_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "6f2f35a9a0b94e7e949eeba7785911d4/data"
)

# OA-ladder-only sources (microcosm #349). The stage-one constituency draw is
# weighted by OA household counts (Nomis Census 2021 TS041 "Number of
# households"); ward and ITL are the ONS Open Geography Portal best-fit lookups
# the ladder carries per its LAD (April 2023) anchor.
EW_OA_HOUSEHOLDS_URL = (
    "https://www.nomisweb.co.uk/output/census/2021/census2021-ts041.zip"
)
# "Output Area (2021) to Ward to LAD to CTYUA to Region to Country (2022) Best
# Fit Lookup in EW (V2)" — ONS Open Geography Portal item 7207b517....
EW_OA_WARD_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "7207b51700f7472e88460f3a2e1eb5f9/csv?layers=0"
)
# "Local Authority District (April 2023) to LAU1 to ITL3 to ITL2 to ITL1
# (January 2021) Lookup in the UK" — ONS Open Geography Portal item
# 02b49429.... Keyed on LAD April 2023, matching the ladder's LAD anchor, so
# the join is exact rather than a cross-vintage best-fit (the #205 rule).
LAD23_ITL_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "02b4942973374f039ec1d2e7d35c16a9/csv?layers=0"
)

ENGLAND_WALES_OA2021_COUNT = 188_880
SCOTLAND_OA2022_COUNT = 46_363
NI_DZ2021_COUNT = 3_780
MAX_UNMATCHED_ACTIVE_NI_POSTCODE_SHARE = 0.01


def load_england_wales_oa_hierarchy(
    url: str = EW_OA_HIERARCHY_URL,
) -> pd.DataFrame:
    """Load ONS OA2021 -> LSOA/MSOA/LAD22 hierarchy for England and Wales."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_ew_hierarchy(
        frame,
        expected_count=ENGLAND_WALES_OA2021_COUNT,
    )


def load_ew_oa_lad23_lookup(url: str = EW_OA_LAD23_URL) -> pd.DataFrame:
    """Load the ONS OA2021 -> LAD April 2023 best-fit lookup."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_ew_lad_lookup(frame)


def load_england_wales_oa_population(
    url: str = EW_OA_POPULATION_URL,
) -> pd.DataFrame:
    """Load Nomis Census 2021 OA population counts for England and Wales."""

    frame = _read_zip_csv_url(
        url,
        filename_contains="census2021-ts001-oa",
        dtype=str,
    )
    return _normalise_ew_population(
        frame,
        expected_count=ENGLAND_WALES_OA2021_COUNT,
    )


def load_england_wales_oa_constituencies(
    url: str = EW_OA_CONSTITUENCY_URL,
) -> pd.DataFrame:
    """Load ONS OA2021 -> 2024 Westminster constituency lookup for E/W."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_ew_constituencies(
        frame,
        expected_count=ENGLAND_WALES_OA2021_COUNT,
    )


def load_england_lad_region_lookup(
    url: str = ENGLAND_LAD_REGION_URL,
) -> pd.DataFrame:
    """Load ONS English LAD22 -> region lookup."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_england_lad_region_lookup(frame)


def load_england_wales_oa_households(
    url: str = EW_OA_HOUSEHOLDS_URL,
) -> pd.DataFrame:
    """Load Nomis Census 2021 OA household counts (TS041) for E/W.

    The household count is the stage-one weight for the OA-ladder's
    within-region constituency draw (microcosm #349). Cross-check: the 188,880
    E/W output areas sum to 24,783,304 households, matching the published ONS
    Census 2021 England-&-Wales total of ~24.8 million.
    """

    frame = _read_zip_csv_url(
        url,
        filename_contains="census2021-ts041-oa",
        dtype=str,
    )
    return _normalise_ew_households(
        frame,
        expected_count=ENGLAND_WALES_OA2021_COUNT,
    )


def load_england_wales_oa_ward_lookup(
    url: str = EW_OA_WARD_URL,
) -> pd.DataFrame:
    """Load ONS OA2021 -> electoral ward best-fit lookup for E/W."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_ew_ward_lookup(
        frame,
        expected_count=ENGLAND_WALES_OA2021_COUNT,
    )


def load_lad_itl_lookup(url: str = LAD23_ITL_URL) -> pd.DataFrame:
    """Load ONS LAD (April 2023) -> ITL3 lookup for the UK."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_lad_itl_lookup(frame)


def load_scotland_oa_dz_iz_lookup(
    url: str = SCOTLAND_OA_DZ_IZ_URL,
) -> pd.DataFrame:
    """Load NRS OA2022 -> Data Zone 2022 -> Intermediate Zone 2022 lookup."""

    frame = _read_zip_csv_url(url, filename_contains="OA22_DZ22_IZ22", dtype=str)
    return _normalise_scotland_oa_dz_iz(
        frame,
        expected_count=SCOTLAND_OA2022_COUNT,
    )


def load_scotland_oa_lau_lookup(
    url: str = SCOTLAND_OA_LAU_ITL_URL,
) -> pd.DataFrame:
    """Load NRS OA2022 -> council-area lookup via LAU 2025 Level 1."""

    data = _read_url_bytes(url)
    oa_lau = _read_zip_csv_bytes(
        data,
        filename_contains="OA22_LAU25_L1",
        dtype=str,
    )
    council_lau = _read_zip_csv_bytes(
        data,
        filename_contains="CA19 - LAU25L1",
        dtype=str,
    )
    return _normalise_scotland_oa_lau(
        oa_lau,
        council_lau,
        expected_count=SCOTLAND_OA2022_COUNT,
    )


def load_scotland_oa_constituencies(
    url: str = SCOTLAND_OA_CONSTITUENCY_URL,
) -> pd.DataFrame:
    """Load NRS OA2022 -> 2024 Westminster constituency lookup."""

    frame = _read_zip_csv_url(url, filename_contains="OA22_UKPC24", dtype=str)
    return _normalise_scotland_constituencies(
        frame,
        expected_count=SCOTLAND_OA2022_COUNT,
    )


def load_scotland_oa_population(
    url: str = SCOTLAND_OA_POPULATION_URL,
) -> pd.DataFrame:
    """Load NRS Scotland Census 2022 OA resident population counts."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_scotland_population(
        frame,
        expected_count=SCOTLAND_OA2022_COUNT,
    )


def load_ni_dz_hierarchy(url: str = NI_DZ_GEOJSON_ZIP_URL) -> pd.DataFrame:
    """Load NISRA Data Zone 2021 hierarchy from the GeoJSON ZIP."""

    data = _read_geojson_zip_url(url)
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        rows.append(
            {
                "oa_code": props.get("DZ2021_cd"),
                "lsoa_code": props.get("DZ2021_cd"),
                "msoa_code": props.get("SDZ2021_cd"),
                "la_code": props.get("LGD2014_cd"),
            }
        )
    return _normalise_ni_hierarchy(
        pd.DataFrame(rows),
        expected_count=NI_DZ2021_COUNT,
    )


def load_scotland_oa_ward_lookup(
    url: str = SCOTLAND_CENSUS_INDEX_ZIP_URL,
) -> pd.DataFrame:
    """Load the OA2022 -> Electoral Ward 2022 lookup from the NRS census index."""

    frame = _read_zip_csv_url(
        url,
        filename_contains="oa_to_higher_areas",
        dtype=str,
        encoding="utf-8-sig",
    )
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key == "oa2022":
            rename[column] = "oa_code"
        elif key == "ew2022":
            rename[column] = "ward_code"
    frame = frame.rename(columns=rename)
    missing = sorted({"oa_code", "ward_code"} - set(frame.columns))
    if missing:
        raise ValueError(f"Scotland OA ward lookup is missing column(s): {missing}.")
    lookup = frame.loc[:, ["oa_code", "ward_code"]].copy()
    for column in ("oa_code", "ward_code"):
        lookup[column] = lookup[column].fillna("").astype(str).str.strip()
        blank = lookup[column] == ""
        if blank.any():
            raise ValueError(
                f"Scotland OA ward lookup has {int(blank.sum())} blank "
                f"{column} value(s)."
            )
    if lookup["oa_code"].duplicated().any():
        duplicates = lookup.loc[lookup["oa_code"].duplicated(), "oa_code"]
        raise ValueError(
            "Scotland OA ward lookup OA codes must be unique; duplicate "
            f"value(s): {list(map(str, duplicates.unique()[:5]))}."
        )
    _validate_expected_count(
        lookup,
        expected_count=SCOTLAND_OA2022_COUNT,
        label="Scotland OA ward lookup",
        unit_label="OA2022",
    )
    return lookup.sort_values("oa_code", kind="mergesort").reset_index(drop=True)


def load_scotland_oa_households(
    url: str = SCOTLAND_CENSUS_INDEX_ZIP_URL,
) -> pd.DataFrame:
    """Sum census occupied households per OA2022 from the NRS postcode index."""

    frame = _read_zip_csv_url(
        url,
        filename_contains="postcode_to_oa",
        dtype=str,
        encoding="utf-8-sig",
    )
    missing = sorted({"OutputArea2022Code", "HouseholdCount"} - set(frame.columns))
    if missing:
        raise ValueError(f"Scotland postcode index is missing column(s): {missing}.")
    postcode = frame.loc[:, ["OutputArea2022Code", "HouseholdCount"]].copy()
    postcode["OutputArea2022Code"] = (
        postcode["OutputArea2022Code"].fillna("").astype(str).str.strip()
    )
    if (postcode["OutputArea2022Code"] == "").any():
        raise ValueError("Scotland postcode index has blank OA codes.")
    if postcode["HouseholdCount"].isna().any():
        raise ValueError(
            "Scotland postcode index has missing HouseholdCount value(s); a "
            "blank count would silently sum as zero."
        )
    postcode["HouseholdCount"] = pd.to_numeric(
        postcode["HouseholdCount"],
        errors="raise",
    )
    if (postcode["HouseholdCount"] < 0).any():
        raise ValueError(
            "Scotland postcode index household counts must be non-negative."
        )
    households = (
        postcode.groupby("OutputArea2022Code", sort=True)["HouseholdCount"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "OutputArea2022Code": "oa_code",
                "HouseholdCount": "households",
            }
        )
    )
    _validate_expected_count(
        households,
        expected_count=SCOTLAND_OA2022_COUNT,
        label="Scotland OA households",
        unit_label="OA2022",
    )
    return households.reset_index(drop=True)


def load_ni_dz_ward_lookup(url: str = NI_DZ_GEOJSON_ZIP_URL) -> pd.DataFrame:
    """Load the DZ2021 -> DEA2014 (ward analogue) lookup from the NI GeoJSON."""

    data = _read_geojson_zip_url(url)
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        rows.append(
            {
                "oa_code": props.get("DZ2021_cd"),
                "ward_code": props.get("DEA2014_cd"),
            }
        )
    lookup = pd.DataFrame(rows)
    if lookup.empty:
        raise ValueError("NI DZ ward lookup has no features.")
    for column in ("oa_code", "ward_code"):
        lookup[column] = lookup[column].fillna("").astype(str).str.strip()
        blank = lookup[column] == ""
        if blank.any():
            raise ValueError(
                f"NI DZ ward lookup has {int(blank.sum())} blank {column} value(s)."
            )
    if lookup["oa_code"].duplicated().any():
        duplicates = lookup.loc[lookup["oa_code"].duplicated(), "oa_code"]
        raise ValueError(
            "NI DZ ward lookup DZ codes must be unique; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    _validate_expected_count(
        lookup,
        expected_count=NI_DZ2021_COUNT,
        label="NI DZ ward lookup",
    )
    return lookup.sort_values("oa_code", kind="mergesort").reset_index(drop=True)


def load_ni_dz_households(url: str = NI_DZ_HOUSEHOLDS_CSV_URL) -> pd.DataFrame:
    """Load NISRA Census 2021 Data Zone occupied household counts."""

    frame = _read_csv_url(url, dtype=str)
    rename = {}
    for column in frame.columns:
        lower = str(column).lower()
        if "data zone code" in lower or lower.strip() == "oa_code":
            rename[column] = "oa_code"
        elif lower.strip() in {"count", "households"}:
            rename[column] = "households"
    households = frame.rename(columns=rename)
    missing = sorted({"oa_code", "households"} - set(households.columns))
    if missing:
        raise ValueError(f"NI households is missing column(s): {missing}.")
    households = households[["oa_code", "households"]].copy()
    households["oa_code"] = households["oa_code"].astype(str).str.strip()
    households = households[households["oa_code"].str.startswith("N2")]
    if households["oa_code"].duplicated().any():
        duplicates = households.loc[households["oa_code"].duplicated(), "oa_code"]
        raise ValueError(
            "NI household DZ codes must be unique; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    households["households"] = pd.to_numeric(
        households["households"],
        errors="raise",
    )
    if (households["households"] < 0).any():
        raise ValueError("NI household counts must be non-negative.")
    _validate_expected_count(
        households,
        expected_count=NI_DZ2021_COUNT,
        label="NI DZ households",
    )
    return households.sort_values("oa_code", kind="mergesort").reset_index(drop=True)


def load_ni_dz_population(url: str = NI_DZ_POPULATION_CSV_URL) -> pd.DataFrame:
    """Load NISRA Census 2021 Data Zone population counts."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_ni_population(frame, expected_count=NI_DZ2021_COUNT)


def load_uk_postcode_oa_lookup(url: str = UK_POSTCODE_OA_MAY25_ZIP_URL) -> pd.DataFrame:
    """Load ONS UK postcode -> OA2021 lookup from a zipped CSV."""

    return _read_zip_csv_url(
        url,
        usecols=["pcds", "doterm", "oa21cd"],
        dtype=str,
    )


def load_uk_postcode_constituency_lookup(
    url: str = UK_POSTCODE_PCON_MAY24_ZIP_URL,
) -> pd.DataFrame:
    """Load ONS UK postcode -> Westminster constituency lookup."""

    return _read_zip_csv_url(
        url,
        usecols=["pcd", "pconcd"],
        dtype=str,
        encoding="latin-1",
    )


def build_england_wales_crosswalk(
    oa_hierarchy: pd.DataFrame,
    oa_population: pd.DataFrame,
    oa_constituencies: pd.DataFrame,
    oa_lad23_lookup: pd.DataFrame,
    england_lad_region_lookup: pd.DataFrame,
    *,
    expected_oa_count: int | None = ENGLAND_WALES_OA2021_COUNT,
) -> pd.DataFrame:
    """Build E/W OA rows for the Microcosm row-wise crosswalk."""

    hierarchy = _normalise_ew_hierarchy(
        oa_hierarchy,
        expected_count=expected_oa_count,
    )
    population = _normalise_ew_population(
        oa_population,
        expected_count=expected_oa_count,
    )
    constituencies = _normalise_ew_constituencies(
        oa_constituencies,
        expected_count=expected_oa_count,
    )
    lad23 = _normalise_ew_lad_lookup(oa_lad23_lookup)
    regions = _normalise_england_lad_region_lookup(england_lad_region_lookup)
    _validate_matching_source_codes(
        "E/W OA",
        "codes",
        {
            "hierarchy": hierarchy["oa_code"],
            "population": population["oa_code"],
            "constituency": constituencies["oa_code"],
            "lad23": lad23["oa_code"],
        },
    )

    rows = (
        hierarchy.merge(population, on="oa_code", how="left")
        .merge(constituencies, on="oa_code", how="left")
        .merge(lad23, on="oa_code", how="left")
        .merge(regions, on="la_code", how="left")
    )
    rows["country"] = rows["oa_code"].map(
        lambda code: "Wales" if str(code).startswith("W") else "England"
    )
    rows.loc[rows["country"] == "Wales", "region_code"] = "W99999999"
    missing_region = rows[(rows["country"] == "England") & rows["region_code"].isna()][
        "la_code"
    ].drop_duplicates()
    if not missing_region.empty:
        raise ValueError(
            "English LAD region lookup is missing LAD22 code(s): "
            f"{missing_region.astype(str).tolist()[:5]}."
        )
    missing = rows[
        rows[["population", "constituency_code", "lad23_code"]].isna().any(axis=1)
    ]["oa_code"].tolist()
    if missing:
        raise ValueError(f"E/W rows are missing source values: {missing[:5]}.")
    rows["la_code"] = rows["lad23_code"]
    return prepare_geography_crosswalk(rows.loc[:, CROSSWALK_COLUMNS])


def build_scotland_crosswalk(
    oa_dz_iz: pd.DataFrame,
    oa_lau: pd.DataFrame,
    oa_constituencies: pd.DataFrame,
    oa_population: pd.DataFrame,
    *,
    expected_oa_count: int | None = SCOTLAND_OA2022_COUNT,
) -> pd.DataFrame:
    """Build Scotland OA rows for the Microcosm row-wise crosswalk."""

    hierarchy = _normalise_scotland_oa_dz_iz(
        oa_dz_iz,
        expected_count=expected_oa_count,
    )
    las = _normalise_scotland_oa_lau(oa_lau, expected_count=expected_oa_count)
    constituencies = _normalise_scotland_constituencies(
        oa_constituencies,
        expected_count=expected_oa_count,
    )
    population = _normalise_scotland_population(
        oa_population,
        expected_count=expected_oa_count,
    )
    _validate_matching_source_codes(
        "Scotland OA",
        "codes",
        {
            "hierarchy": hierarchy["oa_code"],
            "la": las["oa_code"],
            "constituency": constituencies["oa_code"],
            "population": population["oa_code"],
        },
    )

    rows = (
        hierarchy.merge(las, on="oa_code", how="left")
        .merge(constituencies, on="oa_code", how="left")
        .merge(population, on="oa_code", how="left")
    )
    rows["region_code"] = "S99999999"
    rows["country"] = "Scotland"
    missing = rows[
        rows[["la_code", "constituency_code", "population"]].isna().any(axis=1)
    ]["oa_code"].tolist()
    if missing:
        raise ValueError(f"Scotland rows are missing source values: {missing[:5]}.")
    return prepare_geography_crosswalk(rows.loc[:, CROSSWALK_COLUMNS])


def build_great_britain_crosswalk(
    *,
    ew_oa_hierarchy: pd.DataFrame,
    ew_oa_population: pd.DataFrame,
    ew_oa_constituencies: pd.DataFrame,
    ew_oa_lad23_lookup: pd.DataFrame,
    england_lad_region_lookup: pd.DataFrame,
    scotland_oa_dz_iz: pd.DataFrame,
    scotland_oa_lau: pd.DataFrame,
    scotland_oa_constituencies: pd.DataFrame,
    scotland_oa_population: pd.DataFrame,
    expected_england_wales_oa_count: int | None = ENGLAND_WALES_OA2021_COUNT,
    expected_scotland_oa_count: int | None = SCOTLAND_OA2022_COUNT,
) -> pd.DataFrame:
    """Build GB OA rows from official E/W and Scotland sources."""

    england_wales = build_england_wales_crosswalk(
        ew_oa_hierarchy,
        ew_oa_population,
        ew_oa_constituencies,
        ew_oa_lad23_lookup,
        england_lad_region_lookup,
        expected_oa_count=expected_england_wales_oa_count,
    )
    scotland = build_scotland_crosswalk(
        scotland_oa_dz_iz,
        scotland_oa_lau,
        scotland_oa_constituencies,
        scotland_oa_population,
        expected_oa_count=expected_scotland_oa_count,
    )
    return prepare_geography_crosswalk(
        pd.concat([england_wales, scotland], ignore_index=True)
    )


def build_official_uk_geography_crosswalk(
    *,
    ew_oa_hierarchy: pd.DataFrame | None = None,
    ew_oa_population: pd.DataFrame | None = None,
    ew_oa_constituencies: pd.DataFrame | None = None,
    ew_oa_lad23_lookup: pd.DataFrame | None = None,
    england_lad_region_lookup: pd.DataFrame | None = None,
    scotland_oa_dz_iz: pd.DataFrame | None = None,
    scotland_oa_lau: pd.DataFrame | None = None,
    scotland_oa_constituencies: pd.DataFrame | None = None,
    scotland_oa_population: pd.DataFrame | None = None,
    ni_dz_hierarchy: pd.DataFrame | None = None,
    ni_dz_population: pd.DataFrame | None = None,
    ni_dz_constituencies: pd.DataFrame | None = None,
    postcode_oa: pd.DataFrame | None = None,
    postcode_constituency: pd.DataFrame | None = None,
    expected_england_wales_oa_count: int | None = ENGLAND_WALES_OA2021_COUNT,
    expected_scotland_oa_count: int | None = SCOTLAND_OA2022_COUNT,
    expected_ni_dz_count: int | None = NI_DZ2021_COUNT,
) -> pd.DataFrame:
    """Build the complete UK crosswalk from public ONS, NRS, and NISRA sources."""

    if ew_oa_hierarchy is None:
        ew_oa_hierarchy = load_england_wales_oa_hierarchy()
    if ew_oa_population is None:
        ew_oa_population = load_england_wales_oa_population()
    if ew_oa_constituencies is None:
        ew_oa_constituencies = load_england_wales_oa_constituencies()
    if ew_oa_lad23_lookup is None:
        ew_oa_lad23_lookup = load_ew_oa_lad23_lookup()
    if england_lad_region_lookup is None:
        england_lad_region_lookup = load_england_lad_region_lookup()
    if scotland_oa_dz_iz is None:
        scotland_oa_dz_iz = load_scotland_oa_dz_iz_lookup()
    if scotland_oa_lau is None:
        scotland_oa_lau = load_scotland_oa_lau_lookup()
    if scotland_oa_constituencies is None:
        scotland_oa_constituencies = load_scotland_oa_constituencies()
    if scotland_oa_population is None:
        scotland_oa_population = load_scotland_oa_population()
    if ni_dz_hierarchy is None:
        ni_dz_hierarchy = load_ni_dz_hierarchy()
    if ni_dz_population is None:
        ni_dz_population = load_ni_dz_population()
    if ni_dz_constituencies is None:
        if postcode_oa is None:
            postcode_oa = load_uk_postcode_oa_lookup()
        if postcode_constituency is None:
            postcode_constituency = load_uk_postcode_constituency_lookup()
        ni_dz_constituencies = infer_ni_dz_constituencies_from_postcodes(
            postcode_oa,
            postcode_constituency,
        )

    gb = build_great_britain_crosswalk(
        ew_oa_hierarchy=ew_oa_hierarchy,
        ew_oa_population=ew_oa_population,
        ew_oa_constituencies=ew_oa_constituencies,
        ew_oa_lad23_lookup=ew_oa_lad23_lookup,
        england_lad_region_lookup=england_lad_region_lookup,
        scotland_oa_dz_iz=scotland_oa_dz_iz,
        scotland_oa_lau=scotland_oa_lau,
        scotland_oa_constituencies=scotland_oa_constituencies,
        scotland_oa_population=scotland_oa_population,
        expected_england_wales_oa_count=expected_england_wales_oa_count,
        expected_scotland_oa_count=expected_scotland_oa_count,
    )
    ni = build_northern_ireland_crosswalk(
        ni_dz_hierarchy,
        ni_dz_population,
        ni_dz_constituencies,
        expected_dz_count=expected_ni_dz_count,
    )
    return prepare_geography_crosswalk(pd.concat([gb, ni], ignore_index=True))


def update_england_wales_lad_codes(
    crosswalk: pd.DataFrame,
    oa_lad_lookup: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Update an OA crosswalk to April 2023 E/W LAD codes."""

    base = prepare_geography_crosswalk(crosswalk, require_constituency=False)
    lookup = _normalise_ew_lad_lookup(oa_lad_lookup)
    if strict:
        ew_rows = base["country"].isin(["England", "Wales"])
        missing = sorted(set(base.loc[ew_rows, "oa_code"]) - set(lookup["oa_code"]))
        if missing:
            raise ValueError(f"E/W LAD23 lookup is missing OA code(s): {missing[:5]}.")
    repaired = base.merge(lookup, on="oa_code", how="left")
    mask = repaired["lad23_code"].notna()
    repaired.loc[mask, "la_code"] = repaired.loc[mask, "lad23_code"]
    return repaired.drop(columns=["lad23_code"])


def infer_ni_dz_constituencies_from_postcodes(
    postcode_oa: pd.DataFrame,
    postcode_constituency: pd.DataFrame,
    *,
    max_unmatched_active_postcode_share: float = (
        MAX_UNMATCHED_ACTIVE_NI_POSTCODE_SHARE
    ),
) -> pd.DataFrame:
    """Infer NI Data Zone -> PCON24 by active-postcode modal constituency."""

    if not 0 <= max_unmatched_active_postcode_share <= 1:
        raise ValueError("max_unmatched_active_postcode_share must be in [0, 1].")

    oa = postcode_oa.copy()
    if "pcds" not in oa.columns or "oa21cd" not in oa.columns:
        raise ValueError("postcode_oa must include 'pcds' and 'oa21cd'.")
    oa["pcd_key"] = _normalise_postcode(oa["pcds"])
    oa = oa[oa["oa21cd"].astype(str).str.startswith("N", na=False)]
    if "doterm" in oa.columns:
        active = oa["doterm"].isna() | oa["doterm"].astype(str).str.strip().eq("")
        oa = oa[active]
    if oa.empty:
        raise ValueError("postcode_oa did not include active NI postcodes.")

    pcon = postcode_constituency.copy()
    if "pcd" not in pcon.columns or "pconcd" not in pcon.columns:
        raise ValueError("postcode_constituency must include 'pcd' and 'pconcd'.")
    pcon["pcd_key"] = _normalise_postcode(pcon["pcd"])
    pcon = pcon[pcon["pconcd"].astype(str).str.startswith("N", na=False)]
    if pcon.empty:
        raise ValueError("postcode_constituency did not include NI postcodes.")

    # Duplicate normalized keys would Cartesian-expand the merge below and
    # could flip a Data Zone's modal constituency without ever tripping the
    # unmatched-postcode fence — refuse them in either source.
    for label, frame in (("postcode_oa", oa), ("postcode_constituency", pcon)):
        duplicated = frame["pcd_key"].duplicated()
        if duplicated.any():
            examples = sorted(frame.loc[duplicated, "pcd_key"].unique()[:5])
            raise ValueError(
                f"{label} contains {int(duplicated.sum())} duplicate "
                f"normalized postcode key(s); examples {examples}."
            )

    pcon_keys = set(pcon["pcd_key"])
    matched_mask = oa["pcd_key"].isin(pcon_keys)
    active_postcode_count = len(oa)
    unmatched_postcode_count = int((~matched_mask).sum())
    unmatched_share = unmatched_postcode_count / active_postcode_count
    if unmatched_share > max_unmatched_active_postcode_share:
        raise ValueError(
            "postcode constituency source is missing too many active NI "
            f"postcodes: {unmatched_postcode_count}/{active_postcode_count} "
            f"({unmatched_share:.2%})."
        )

    joined = oa.loc[matched_mask, ["pcd_key", "oa21cd"]].merge(
        pcon[["pcd_key", "pconcd"]],
        on="pcd_key",
        how="inner",
    )
    if joined.empty:
        raise ValueError("postcode sources did not produce any NI DZ-PCON matches.")

    counts = (
        joined.groupby(["oa21cd", "pconcd"], sort=True)
        .size()
        .rename("postcode_count")
        .reset_index()
        .sort_values(
            ["oa21cd", "postcode_count", "pconcd"],
            ascending=[True, False, True],
        )
    )
    mode = counts.drop_duplicates("oa21cd")
    result = mode.rename(
        columns={
            "oa21cd": "oa_code",
            "pconcd": "constituency_code",
        }
    )[["oa_code", "constituency_code", "postcode_count"]].reset_index(drop=True)
    missing_dz = sorted(set(oa["oa21cd"]) - set(result["oa_code"]))
    if missing_dz:
        raise ValueError(
            "postcode sources left active NI DZ code(s) without PCON matches: "
            f"{missing_dz[:5]}."
        )
    result.attrs["active_ni_postcode_count"] = active_postcode_count
    result.attrs["unmatched_active_ni_postcode_count"] = unmatched_postcode_count
    result.attrs["unmatched_active_ni_postcode_share"] = unmatched_share
    return result


def build_northern_ireland_crosswalk(
    dz_hierarchy: pd.DataFrame,
    dz_population: pd.DataFrame,
    dz_constituencies: pd.DataFrame,
    *,
    expected_dz_count: int | None = NI_DZ2021_COUNT,
) -> pd.DataFrame:
    """Build NI Data Zone rows for the Microcosm row-wise crosswalk."""

    hierarchy = _normalise_ni_hierarchy(
        dz_hierarchy,
        expected_count=expected_dz_count,
    )
    population = _normalise_ni_population(
        dz_population,
        expected_count=expected_dz_count,
    )
    constituencies = dz_constituencies.copy()
    if "oa_code" not in constituencies or "constituency_code" not in constituencies:
        raise ValueError(
            "dz_constituencies must include 'oa_code' and 'constituency_code'."
        )
    constituencies = constituencies[["oa_code", "constituency_code"]].copy()
    constituencies["oa_code"] = (
        constituencies["oa_code"].fillna("").astype(str).str.strip()
    )
    constituencies["constituency_code"] = (
        constituencies["constituency_code"].fillna("").astype(str).str.strip()
    )
    blank_oa = constituencies["oa_code"] == ""
    if blank_oa.any():
        raise ValueError("NI constituency rows must not include blank OA codes.")
    blank_pcon = constituencies["constituency_code"] == ""
    if blank_pcon.any():
        missing_codes = constituencies.loc[blank_pcon, "oa_code"].tolist()
        raise ValueError(
            "NI constituency rows must not include blank PCON codes: "
            f"{missing_codes[:5]}."
        )
    if constituencies["oa_code"].duplicated().any():
        duplicates = constituencies.loc[
            constituencies["oa_code"].duplicated(),
            "oa_code",
        ]
        raise ValueError(
            "NI constituency rows must have unique OA codes; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    _validate_matching_ni_codes(
        {
            "hierarchy": hierarchy["oa_code"],
            "population": population["oa_code"],
            "constituency": constituencies["oa_code"],
        }
    )

    rows = hierarchy.merge(population, on="oa_code", how="left").merge(
        constituencies,
        on="oa_code",
        how="left",
    )
    rows["region_code"] = "N99999999"
    rows["country"] = "Northern Ireland"
    missing = rows[rows[["population", "constituency_code"]].isna().any(axis=1)][
        "oa_code"
    ].tolist()
    if missing:
        raise ValueError(f"NI rows are missing population or PCON: {missing[:5]}.")
    return prepare_geography_crosswalk(rows.loc[:, CROSSWALK_COLUMNS])


def build_complete_uk_geography_crosswalk(
    base_crosswalk: pd.DataFrame,
    *,
    ew_oa_lad23_lookup: pd.DataFrame,
    ni_dz_hierarchy: pd.DataFrame,
    ni_dz_population: pd.DataFrame,
    ni_dz_constituencies: pd.DataFrame,
    expected_ni_dz_count: int | None = NI_DZ2021_COUNT,
) -> pd.DataFrame:
    """Repair GB LAD codes and append NI rows to a base OA crosswalk."""

    repaired = update_england_wales_lad_codes(
        base_crosswalk,
        ew_oa_lad23_lookup,
    )
    repaired = repaired[repaired["country"].astype(str) != "Northern Ireland"]
    ni = build_northern_ireland_crosswalk(
        ni_dz_hierarchy,
        ni_dz_population,
        ni_dz_constituencies,
        expected_dz_count=expected_ni_dz_count,
    )
    combined = pd.concat([repaired, ni], ignore_index=True)
    return prepare_geography_crosswalk(combined)


def _normalise_ew_hierarchy(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    upper_to_column = {str(column).strip().upper(): column for column in source}
    lad_column = upper_to_column.get("LAD22CD")
    if lad_column is None:
        lad_column = next(
            (
                column
                for upper, column in upper_to_column.items()
                if upper.startswith("LAD") and upper.endswith("CD")
            ),
            None,
        )
    column_map = {
        upper_to_column.get("OA21CD"): "oa_code",
        upper_to_column.get("LSOA21CD"): "lsoa_code",
        upper_to_column.get("MSOA21CD"): "msoa_code",
        lad_column: "la_code",
    }
    source = source.rename(
        columns={column: name for column, name in column_map.items() if column}
    )
    missing = sorted({"oa_code", "lsoa_code", "msoa_code", "la_code"} - set(source))
    if missing:
        raise ValueError(f"E/W hierarchy is missing column(s): {missing}.")
    hierarchy = _normalise_code_rows(
        source[["oa_code", "lsoa_code", "msoa_code", "la_code"]],
        label="E/W hierarchy",
        unique_column="oa_code",
        expected_count=expected_count,
        unit_label="OA2021",
        prefixes=("E", "W"),
    )
    return hierarchy


def _normalise_ew_population(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    geo_column = _find_column(source, exact=("geography code", "oa_code"))
    population_column = _find_column(
        source,
        exact=("population", "count"),
        contains_all=(("total", "measures"),),
    )
    if geo_column is None or population_column is None:
        raise ValueError("E/W population is missing OA code or population columns.")
    population = pd.DataFrame(
        {
            "oa_code": source[geo_column],
            "population": source[population_column],
        }
    )
    population["oa_code"] = population["oa_code"].fillna("").astype(str).str.strip()
    population = population[population["oa_code"].str.match(r"^[EW]00")].copy()
    _validate_unique_nonblank(
        population,
        "oa_code",
        label="E/W population",
        expected_count=expected_count,
        unit_label="OA2021",
    )
    population["population"] = pd.to_numeric(
        population["population"],
        errors="raise",
    )
    return population.reset_index(drop=True)


def _normalise_ew_constituencies(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    upper_to_column = {str(column).strip().upper(): column for column in source}
    pcon_column = next(
        (
            column
            for upper, column in upper_to_column.items()
            if upper.startswith("PCON") and upper.endswith("CD")
        ),
        None,
    )
    column_map = {
        upper_to_column.get("OA21CD"): "oa_code",
        pcon_column: "constituency_code",
    }
    source = source.rename(
        columns={column: name for column, name in column_map.items() if column}
    )
    missing = sorted({"oa_code", "constituency_code"} - set(source))
    if missing:
        raise ValueError(f"E/W constituency lookup is missing column(s): {missing}.")
    return _normalise_code_rows(
        source[["oa_code", "constituency_code"]],
        label="E/W constituency lookup",
        unique_column="oa_code",
        expected_count=expected_count,
        unit_label="OA2021",
        prefixes=("E", "W"),
    )


def _normalise_england_lad_region_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy()
    if {"la_code", "region_code"}.issubset(source.columns):
        lookup = source[["la_code", "region_code"]].copy()
        return _normalise_code_rows(
            lookup,
            label="English LAD region lookup",
            unique_column="la_code",
            prefixes=("E",),
        )
    upper_to_column = {str(column).strip().upper(): column for column in source}
    lad_column = upper_to_column.get("LAD22CD") or next(
        (
            column
            for upper, column in upper_to_column.items()
            if upper.startswith("LAD") and upper.endswith("CD")
        ),
        None,
    )
    region_column = upper_to_column.get("RGN22CD") or next(
        (
            column
            for upper, column in upper_to_column.items()
            if upper.startswith("RGN") and upper.endswith("CD")
        ),
        None,
    )
    if lad_column is None or region_column is None:
        raise ValueError("English LAD region lookup is missing LAD or region columns.")
    lookup = pd.DataFrame(
        {
            "la_code": source[lad_column],
            "region_code": source[region_column],
        }
    )
    return _normalise_code_rows(
        lookup,
        label="English LAD region lookup",
        unique_column="la_code",
        prefixes=("E",),
    )


def _normalise_ew_households(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    geo_column = _find_column(source, exact=("geography code", "oa_code"))
    if geo_column is None:
        raise ValueError("E/W households is missing an OA code column.")
    if "households" in source.columns and geo_column != "households":
        households_values = pd.to_numeric(source["households"], errors="raise")
    else:
        households_values = _household_count_values(source, geo_column=geo_column)
    households = pd.DataFrame(
        {
            "oa_code": source[geo_column],
            "households": households_values.to_numpy(),
        }
    )
    households["oa_code"] = households["oa_code"].fillna("").astype(str).str.strip()
    households = households[households["oa_code"].str.match(r"^[EW]00")].copy()
    _validate_unique_nonblank(
        households,
        "oa_code",
        label="E/W households",
        expected_count=expected_count,
        unit_label="OA2021",
    )
    households["households"] = pd.to_numeric(
        households["households"],
        errors="raise",
    )
    return households.reset_index(drop=True)


def _normalise_ew_ward_lookup(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    upper_to_column = {str(column).strip().upper(): column for column in source}
    oa_column = upper_to_column.get("OA21CD")
    ward_column = next(
        (
            column
            for upper, column in upper_to_column.items()
            if upper.startswith("WD") and upper.endswith("CD")
        ),
        None,
    )
    if oa_column is None or ward_column is None:
        raise ValueError("E/W ward lookup is missing OA or ward columns.")
    lookup = pd.DataFrame(
        {
            "oa_code": source[oa_column],
            "ward_code": source[ward_column],
        }
    )
    return _normalise_code_rows(
        lookup,
        label="E/W ward lookup",
        unique_column="oa_code",
        expected_count=expected_count,
        unit_label="OA2021",
        prefixes=("E", "W"),
    )


def _normalise_lad_itl_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy()
    upper_to_column = {str(column).strip().upper(): column for column in source}
    lad_column = next(
        (
            column
            for upper, column in upper_to_column.items()
            if upper.startswith("LAD") and upper.endswith("CD")
        ),
        None,
    )
    itl3_column = next(
        (
            column
            for upper, column in upper_to_column.items()
            if "ITL3" in upper and upper.endswith("CD")
        ),
        None,
    )
    if lad_column is None or itl3_column is None:
        raise ValueError("LAD-ITL lookup is missing LAD or ITL3 columns.")
    lookup = pd.DataFrame(
        {
            "local_authority_code": source[lad_column],
            "itl3_code": source[itl3_column],
        }
    )
    lookup["local_authority_code"] = (
        lookup["local_authority_code"].fillna("").astype(str).str.strip()
    )
    lookup["itl3_code"] = lookup["itl3_code"].fillna("").astype(str).str.strip()
    lookup = lookup[lookup["local_authority_code"] != ""].copy()
    blank_itl = lookup["itl3_code"] == ""
    if blank_itl.any():
        missing_codes = lookup.loc[blank_itl, "local_authority_code"].tolist()
        raise ValueError(
            f"LAD-ITL lookup must not include blank ITL3 codes: {missing_codes[:5]}."
        )
    lookup = lookup.drop_duplicates(subset=["local_authority_code"])
    if lookup["local_authority_code"].duplicated().any():
        duplicates = lookup.loc[
            lookup["local_authority_code"].duplicated(), "local_authority_code"
        ].unique()
        raise ValueError(
            "LAD-ITL lookup maps a LAD to more than one ITL3; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return lookup.reset_index(drop=True)


def _normalise_scotland_oa_dz_iz(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    upper_to_column = {str(column).strip().upper(): column for column in source}
    column_map = {
        upper_to_column.get("OA22"): "oa_code",
        upper_to_column.get("DZ22"): "lsoa_code",
        upper_to_column.get("IZ22"): "msoa_code",
    }
    source = source.rename(
        columns={column: name for column, name in column_map.items() if column}
    )
    missing = sorted({"oa_code", "lsoa_code", "msoa_code"} - set(source))
    if missing:
        raise ValueError(f"Scotland OA-DZ-IZ lookup is missing column(s): {missing}.")
    return _normalise_code_rows(
        source[["oa_code", "lsoa_code", "msoa_code"]],
        label="Scotland OA-DZ-IZ lookup",
        unique_column="oa_code",
        expected_count=expected_count,
        unit_label="OA2022",
        prefixes=("S",),
    )


def _normalise_scotland_oa_lau(
    oa_lau: pd.DataFrame,
    council_lau: pd.DataFrame | None = None,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = oa_lau.copy()
    if council_lau is None and {"oa_code", "la_code"}.issubset(source.columns):
        lookup = source[["oa_code", "la_code"]].copy()
    else:
        source_columns = {str(column).strip().upper(): column for column in source}
        oa_column = source_columns.get("OUTPUTAREA2022CODE") or source_columns.get(
            "OA22"
        )
        lau_column = source_columns.get("LAU2025LEVEL1CODE")
        if oa_column is None or lau_column is None:
            raise ValueError("Scotland OA-LAU lookup is missing OA or LAU columns.")
        if council_lau is None:
            raise ValueError("Scotland OA-LAU lookup needs a council-area bridge.")
        bridge_columns = {
            str(column).strip().upper(): column for column in council_lau.columns
        }
        bridge_lau_column = bridge_columns.get("LAU2025LEVEL1CODE")
        council_column = bridge_columns.get("COUNCILAREA2019CODE")
        if bridge_lau_column is None or council_column is None:
            raise ValueError(
                "Scotland LAU council bridge is missing LAU or council columns."
            )
        oa_lau_lookup = pd.DataFrame(
            {
                "oa_code": source[oa_column],
                "lau25_code": source[lau_column],
            }
        )
        council_lookup = pd.DataFrame(
            {
                "lau25_code": council_lau[bridge_lau_column],
                "la_code": council_lau[council_column],
            }
        )
        council_lookup = _normalise_code_rows(
            council_lookup,
            label="Scotland LAU council bridge",
            unique_column="lau25_code",
            prefixes=("S",),
        )
        lookup = oa_lau_lookup.merge(council_lookup, on="lau25_code", how="left")
        missing_la = lookup[lookup["la_code"].isna()]["lau25_code"].drop_duplicates()
        if not missing_la.empty:
            raise ValueError(
                "Scotland LAU council bridge is missing LAU code(s): "
                f"{missing_la.astype(str).tolist()[:5]}."
            )
        lookup = lookup[["oa_code", "la_code"]]
    return _normalise_code_rows(
        lookup,
        label="Scotland OA-LAU lookup",
        unique_column="oa_code",
        expected_count=expected_count,
        unit_label="OA2022",
        prefixes=("S",),
    )


def _normalise_scotland_constituencies(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    if {"oa_code", "constituency_code"}.issubset(source.columns):
        lookup = source[["oa_code", "constituency_code"]].copy()
        return _normalise_code_rows(
            lookup,
            label="Scotland constituency lookup",
            unique_column="oa_code",
            expected_count=expected_count,
            unit_label="OA2022",
            prefixes=("S",),
        )
    upper_to_column = {str(column).strip().upper(): column for column in source}
    oa_column = upper_to_column.get("OA22") or upper_to_column.get("OA_CODE")
    pcon_column = upper_to_column.get("UKPC24") or upper_to_column.get(
        "CONSTITUENCY_CODE"
    )
    if oa_column is None or pcon_column is None:
        raise ValueError("Scotland constituency lookup is missing OA or UKPC columns.")
    lookup = pd.DataFrame(
        {
            "oa_code": source[oa_column],
            "constituency_code": source[pcon_column],
        }
    )
    return _normalise_code_rows(
        lookup,
        label="Scotland constituency lookup",
        unique_column="oa_code",
        expected_count=expected_count,
        unit_label="OA2022",
        prefixes=("S",),
    )


def _normalise_scotland_population(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    source = frame.copy()
    if {"oa_code", "population"}.issubset(source.columns):
        population = source[["oa_code", "population"]].copy()
        population["oa_code"] = population["oa_code"].fillna("").astype(str).str.strip()
        population = population[population["oa_code"].str.startswith("S")].copy()
        _validate_unique_nonblank(
            population,
            "oa_code",
            label="Scotland population",
            expected_count=expected_count,
            unit_label="OA2022",
        )
        population["population"] = pd.to_numeric(
            population["population"],
            errors="raise",
        )
        return population.reset_index(drop=True)
    upper_to_column = {str(column).strip().upper(): column for column in source}
    oa_column = upper_to_column.get("OUTPUTAREA2022") or upper_to_column.get("OA_CODE")
    population_column = upper_to_column.get("USUALRESIDENTPOPULATION")
    if oa_column is None or population_column is None:
        raise ValueError("Scotland population is missing OA or population columns.")
    population = pd.DataFrame(
        {
            "oa_code": source[oa_column],
            "population": source[population_column],
        }
    )
    population["oa_code"] = population["oa_code"].fillna("").astype(str).str.strip()
    population = population[population["oa_code"].str.startswith("S")].copy()
    _validate_unique_nonblank(
        population,
        "oa_code",
        label="Scotland population",
        expected_count=expected_count,
        unit_label="OA2022",
    )
    population["population"] = pd.to_numeric(
        population["population"],
        errors="raise",
    )
    return population.reset_index(drop=True)


def _normalise_ew_lad_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {}
    for column in frame.columns:
        upper = str(column).upper()
        if upper == "OA21CD":
            column_map[column] = "oa_code"
        elif upper == "LAD23CD":
            column_map[column] = "lad23_code"
    lookup = frame.rename(columns=column_map)
    missing = sorted({"oa_code", "lad23_code"} - set(lookup.columns))
    if missing:
        raise ValueError(f"E/W LAD lookup is missing column(s): {missing}.")
    lookup = lookup[["oa_code", "lad23_code"]].copy()
    lookup["oa_code"] = lookup["oa_code"].fillna("").astype(str).str.strip()
    lookup["lad23_code"] = lookup["lad23_code"].fillna("").astype(str).str.strip()
    blank_oa = lookup["oa_code"] == ""
    if blank_oa.any():
        raise ValueError("E/W LAD lookup must not include blank OA codes.")
    blank_lad = lookup["lad23_code"] == ""
    if blank_lad.any():
        missing_codes = lookup.loc[blank_lad, "oa_code"].tolist()
        raise ValueError(
            f"E/W LAD lookup must not include blank LAD23 codes: {missing_codes[:5]}."
        )
    if lookup["oa_code"].duplicated().any():
        duplicates = lookup.loc[lookup["oa_code"].duplicated(), "oa_code"].unique()
        raise ValueError(
            "E/W LAD lookup OA codes must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return lookup


def _normalise_ni_hierarchy(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    hierarchy = frame.copy()
    missing = sorted({"oa_code", "lsoa_code", "msoa_code", "la_code"} - set(hierarchy))
    if missing:
        raise ValueError(f"NI hierarchy is missing column(s): {missing}.")
    hierarchy = hierarchy[["oa_code", "lsoa_code", "msoa_code", "la_code"]].copy()
    for column in hierarchy.columns:
        hierarchy[column] = hierarchy[column].astype(str).str.strip()
    hierarchy = hierarchy[hierarchy["oa_code"].str.startswith("N2")]
    if hierarchy["oa_code"].duplicated().any():
        duplicates = hierarchy.loc[hierarchy["oa_code"].duplicated(), "oa_code"]
        raise ValueError(
            "NI hierarchy OA codes must be unique; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    _validate_expected_count(
        hierarchy,
        expected_count=expected_count,
        label="NI hierarchy",
    )
    return hierarchy


def _normalise_ni_population(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    population = frame.copy()
    rename = {}
    for column in population.columns:
        lower = str(column).lower()
        if "data zone code" in lower or lower.strip() == "oa_code":
            rename[column] = "oa_code"
        elif lower.strip() in {"count", "population"}:
            rename[column] = "population"
    population = population.rename(columns=rename)
    missing = sorted({"oa_code", "population"} - set(population.columns))
    if missing:
        raise ValueError(f"NI population is missing column(s): {missing}.")
    population = population[["oa_code", "population"]].copy()
    population["oa_code"] = population["oa_code"].astype(str).str.strip()
    population = population[population["oa_code"].str.startswith("N2")]
    if population["oa_code"].duplicated().any():
        duplicates = population.loc[population["oa_code"].duplicated(), "oa_code"]
        raise ValueError(
            "NI population OA codes must be unique; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    population["population"] = pd.to_numeric(
        population["population"],
        errors="raise",
    )
    _validate_expected_count(
        population,
        expected_count=expected_count,
        label="NI population",
    )
    return population


def _normalise_code_rows(
    frame: pd.DataFrame,
    *,
    label: str,
    unique_column: str,
    expected_count: int | None = None,
    unit_label: str = "row",
    prefixes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].fillna("").astype(str).str.strip()
    if prefixes is not None:
        result = result[result[unique_column].str.startswith(prefixes)].copy()
    for column in result.columns:
        blank = result[column] == ""
        if blank.any():
            missing_codes = result.loc[blank, unique_column].tolist()
            raise ValueError(
                f"{label} must not include blank {column} values: {missing_codes[:5]}."
            )
    _validate_unique_nonblank(
        result,
        unique_column,
        label=label,
        expected_count=expected_count,
        unit_label=unit_label,
    )
    return result.reset_index(drop=True)


def _validate_unique_nonblank(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
    expected_count: int | None = None,
    unit_label: str = "row",
) -> None:
    blank = frame[column] == ""
    if blank.any():
        raise ValueError(f"{label} must not include blank {column} values.")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column]
        raise ValueError(
            f"{label} {column} values must be unique; duplicate value(s): "
            f"{list(map(str, duplicates.unique()[:5]))}."
        )
    _validate_expected_count(
        frame,
        expected_count=expected_count,
        label=label,
        unit_label=unit_label,
    )


def _household_count_values(
    frame: pd.DataFrame,
    *,
    geo_column: str,
) -> pd.Series:
    """Return per-row household counts from a Nomis TS041 (households) frame.

    The Nomis bulk export names the count column with the table label (e.g.
    "Number of households: Value"). A single count column is used directly; if
    the table is broken into a "Total" plus household-size bands, the "Total"
    column is used; percentage columns are never counted.
    """

    candidates: list[str] = []
    for column in frame.columns:
        if column == geo_column:
            continue
        lowered = str(column).strip().lower()
        if "household" not in lowered:
            continue
        if any(token in lowered for token in ("%", "percent", "proportion")):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            candidates.append(column)
    if not candidates:
        raise ValueError("E/W households frame has no household count column.")
    totals = [column for column in candidates if "total" in str(column).lower()]
    chosen = totals or candidates
    values = sum(pd.to_numeric(frame[column], errors="raise") for column in chosen)
    return pd.Series(values, index=frame.index)


def _find_column(
    frame: pd.DataFrame,
    *,
    exact: tuple[str, ...] = (),
    contains_all: tuple[tuple[str, ...], ...] = (),
) -> str | None:
    lower_to_column = {str(column).strip().lower(): column for column in frame.columns}
    for name in exact:
        column = lower_to_column.get(name.lower())
        if column is not None:
            return column
    for terms in contains_all:
        lowered_terms = tuple(term.lower() for term in terms)
        for column in frame.columns:
            lowered = str(column).strip().lower()
            if all(term in lowered for term in lowered_terms):
                return column
    return None


def _validate_expected_count(
    frame: pd.DataFrame,
    *,
    expected_count: int | None,
    label: str,
    unit_label: str = "DZ2021",
) -> None:
    if expected_count is None:
        return
    if len(frame) != expected_count:
        raise ValueError(
            f"{label} expected {expected_count} {unit_label} row(s), "
            f"found {len(frame)}."
        )


def _validate_matching_ni_codes(codes_by_source: dict[str, pd.Series]) -> None:
    _validate_matching_source_codes("NI DZ", "codes", codes_by_source)


def _validate_matching_source_codes(
    label: str,
    code_label: str,
    codes_by_source: dict[str, pd.Series],
) -> None:
    reference_name = next(iter(codes_by_source))
    reference_codes = set(codes_by_source[reference_name].astype(str))
    failures: list[str] = []
    for source_name, codes in codes_by_source.items():
        if source_name == reference_name:
            continue
        source_codes = set(codes.astype(str))
        missing = sorted(reference_codes - source_codes)
        extra = sorted(source_codes - reference_codes)
        if missing:
            failures.append(
                f"{source_name} missing {len(missing)} code(s): {missing[:5]}"
            )
        if extra:
            failures.append(f"{source_name} extra {len(extra)} code(s): {extra[:5]}")
    if failures:
        raise ValueError(f"{label} source {code_label} differ; " + "; ".join(failures))


def _normalise_postcode(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(" ", "", regex=False).str.upper()


def _read_url_bytes(
    url: str,
    *,
    timeout: int = 300,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PolicyEngine-Microcosm/0.1"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            should_retry = error.code in {429, 500, 502, 503, 504}
            if not should_retry or attempt == retries - 1:
                raise
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(retry_delay * (2**attempt))
    raise RuntimeError(f"Could not download {url}.")


def _read_csv_url(url: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_read_url_bytes(url)), **kwargs)


def _read_zip_csv_url(
    url: str,
    *,
    filename_contains: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    data = _read_url_bytes(url)
    return _read_zip_csv_bytes(
        data,
        filename_contains=filename_contains,
        **kwargs,
    )


def _read_zip_csv_bytes(
    data: bytes,
    *,
    filename_contains: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        csv_files = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]
        if filename_contains is not None:
            needle = filename_contains.lower()
            csv_files = [name for name in csv_files if needle in name.lower()]
        if not csv_files:
            descriptor = (
                "CSV"
                if filename_contains is None
                else f"CSV matching {filename_contains!r}"
            )
            raise FileNotFoundError(
                f"No {descriptor} found in ZIP. Contents: {archive.namelist()}."
            )
        with archive.open(csv_files[0]) as csv_file:
            return pd.read_csv(csv_file, **kwargs)


def _read_geojson_zip_url(url: str) -> dict[str, Any]:
    data = _read_url_bytes(url)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        geojson_files = [
            name for name in archive.namelist() if name.endswith(".geojson")
        ]
        if not geojson_files:
            raise FileNotFoundError(f"No GeoJSON found in ZIP from {url}.")
        with archive.open(geojson_files[0]) as geojson_file:
            return json.load(geojson_file)


def write_geography_crosswalk(crosswalk: pd.DataFrame, path: str | Path) -> None:
    """Write a validated geography crosswalk to CSV or CSV.GZ."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_geography_crosswalk(crosswalk).to_csv(output, index=False)
