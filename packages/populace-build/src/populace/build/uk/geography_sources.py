"""Official-source helpers for complete UK row-wise geography crosswalks.

The row-wise assignment module stays source-agnostic. This module composes
public geography source tables into the explicit crosswalk frame consumed by
those primitives, without importing the incumbent UK data package.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from populace.build.uk.rowwise_geography import (
    CROSSWALK_COLUMNS,
    prepare_geography_crosswalk,
)

EW_OA_LAD23_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "83982ff4a8144038be52be65dd2b8fa0/csv?layers=0"
)

NI_DZ_GEOJSON_ZIP_URL = (
    "https://www.nisra.gov.uk/files/nisra/publications/"
    "geography-dz2021-geojson.zip"
)
NI_DZ_POPULATION_CSV_URL = (
    "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=DZ21"
)

UK_POSTCODE_OA_MAY25_ZIP_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "7fc55d71a09d4dcfa1fd6473138aacc3/data"
)
UK_POSTCODE_PCON_MAY24_ZIP_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "6f2f35a9a0b94e7e949eeba7785911d4/data"
)

NI_DZ2021_COUNT = 3_780
MAX_UNMATCHED_ACTIVE_NI_POSTCODE_SHARE = 0.01


def load_ew_oa_lad23_lookup(url: str = EW_OA_LAD23_URL) -> pd.DataFrame:
    """Load the ONS OA2021 -> LAD April 2023 best-fit lookup."""

    frame = _read_csv_url(url, dtype=str)
    return _normalise_ew_lad_lookup(frame)


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
            raise ValueError(
                "E/W LAD23 lookup is missing OA code(s): "
                f"{missing[:5]}."
            )
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
        oa = oa[oa["doterm"].isna()]
    if oa.empty:
        raise ValueError("postcode_oa did not include active NI postcodes.")

    pcon = postcode_constituency.copy()
    if "pcd" not in pcon.columns or "pconcd" not in pcon.columns:
        raise ValueError("postcode_constituency must include 'pcd' and 'pconcd'.")
    pcon["pcd_key"] = _normalise_postcode(pcon["pcd"])
    pcon = pcon[pcon["pconcd"].astype(str).str.startswith("N", na=False)]
    if pcon.empty:
        raise ValueError("postcode_constituency did not include NI postcodes.")

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
    """Build NI Data Zone rows for the Populace row-wise crosswalk."""

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
    missing = rows[
        rows[["population", "constituency_code"]].isna().any(axis=1)
    ]["oa_code"].tolist()
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
            "E/W LAD lookup must not include blank LAD23 codes: "
            f"{missing_codes[:5]}."
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


def _validate_expected_count(
    frame: pd.DataFrame,
    *,
    expected_count: int | None,
    label: str,
) -> None:
    if expected_count is None:
        return
    if len(frame) != expected_count:
        raise ValueError(
            f"{label} expected {expected_count} DZ2021 row(s), found {len(frame)}."
        )


def _validate_matching_ni_codes(codes_by_source: dict[str, pd.Series]) -> None:
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
        raise ValueError("NI DZ source codes differ; " + "; ".join(failures))


def _normalise_postcode(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(" ", "", regex=False).str.upper()


def _read_url_bytes(url: str, *, timeout: int = 300) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PolicyEngine-Populace/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _read_csv_url(url: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_read_url_bytes(url)), **kwargs)


def _read_zip_csv_url(url: str, **kwargs: Any) -> pd.DataFrame:
    data = _read_url_bytes(url)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in ZIP from {url}.")
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
