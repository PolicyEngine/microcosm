#!/usr/bin/env python3
"""Build the three UK atomic-area support artifacts from primary publisher sources.

One NPZ per nation system (`atomic_area_support.SYSTEMS`): England & Wales at
2021 Census Output Area grain, Scotland at 2022 Census Output Area grain and
Northern Ireland at 2021 Data Zone grain, each in the shared atomic-geography
support format (`microcosm.build.atomic_geography.encode_atomic_support`) with a
`{kind, source, vintage, relation}` description per mapping column and a
`{kind, source, basis}` description per count column. The publisher files, the
per-nation joins and every code/count fence are the same ones the OA ladder
tool uses; this tool re-targets them at the UK adapter
(`assemble_uk_atomic_area_support`) instead of the single national ladder NPZ.

FY2024-25 products (María's rulings, 2026-09-15): LAD April 2023 code set via
the ONS OA -> Parish/NCP -> LAD -> Region (December 2024) V2 file, Westminster
constituencies July 2024, Ward 2024, ITL 2025 for every nation; Scotland and
Northern Ireland products unchanged from the ladder. Every input's URL, sha256,
size, vintage and Chronicle package id is written to the committed provenance
JSON; each mapping column's `source` string carries the Chronicle package id
and the publisher sha256, so the assignment definition (and therefore every
keyed draw) moves iff the publisher bytes move.

Example:
    uv run python tools/build_uk_atomic_area_supports.py \
        --out-dir build/uk/supports \
        --ladder build/uk/uk_oa_ladder_2021.npz \
        --provenance-json packages/microcosm-build/src/microcosm/build/uk/uk_atomic_area_supports.provenance.json

`--source-manifest` runs from exact local files only (path/url/sha256/bytes per
source, no downloads), in the same envelope as `build_us_puma_ladder_artifact`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.uk_runtime import (
    EW_OA_CONSTITUENCY_URL,
    EW_OA_HIERARCHY_URL,
    EW_OA_HOUSEHOLDS_URL,
    EW_OA_POPULATION_URL,
    NI_DZ_GEOJSON_ZIP_URL,
    NI_DZ_HOUSEHOLDS_CSV_URL,
    NI_DZ_PARLCON24_LOOKUP_XLSX_URL,
    NI_DZ_POPULATION_CSV_URL,
    SCOTLAND_CENSUS_INDEX_ZIP_URL,
    SCOTLAND_OA_CONSTITUENCY_URL,
    SCOTLAND_OA_DZ_IZ_URL,
    SCOTLAND_OA_LAU_ITL_URL,
    SCOTLAND_OA_POPULATION_URL,
    build_england_wales_crosswalk,
    build_northern_ireland_crosswalk,
    build_scotland_crosswalk,
    geography_sources,
    join_uk_oa_ladder_layers,
    load_england_wales_oa_constituencies,
    load_england_wales_oa_hierarchy,
    load_england_wales_oa_households,
    load_england_wales_oa_population,
    load_england_wales_oa_ward_lookup,
    load_lad_itl_lookup,
    load_ni_dz_hierarchy,
    load_ni_dz_households,
    load_ni_dz_parlcon24_lookup,
    load_ni_dz_population,
    load_ni_dz_ward_lookup,
    load_scotland_oa_constituencies,
    load_scotland_oa_dz_iz_lookup,
    load_scotland_oa_households,
    load_scotland_oa_lau_lookup,
    load_scotland_oa_population,
    load_scotland_oa_ward_lookup,
    load_uk_oa_ladder,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    SOURCES as SUPPORT_SOURCES,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    SYSTEMS,
    assemble_uk_atomic_area_support,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.geography_sources import (
    EW_OA_PARNCP_LAD_REGION_URL,
    EW_OA_WARD24_URL,
    LAD24_ITL25_URL,
    load_ew_oa_lad_region_lookup,
)
from microcosm.build.uk_runtime.oa_ladder_sources import LADDER_OA_COLUMNS
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

_TOOLS = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _TOOLS.parent
_UK_PACKAGE = _REPOSITORY_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
_DEFAULT_PROVENANCE = _UK_PACKAGE / "uk_atomic_area_supports.provenance.json"
_CROSSWALK_RESOURCE = _UK_PACKAGE / "local_area_crosswalk.json"

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_KIND = "uk_atomic_area_supports_provenance"


def _load_tool(name: str):
    """Import a sibling tool module by path (the tools directory is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ladder_tool = _load_tool("build_uk_oa_ladder_artifact")
_us_tool = _load_tool("build_us_puma_ladder_artifact")
_download = _ladder_tool._download
_sha256 = _ladder_tool._sha256
_pinned_sources = _us_tool._pinned_sources
_verify_pinned_source = _us_tool._verify_pinned_source

#: Every publisher file, keyed as in the provenance JSON. ``chronicle_package_id``
#: names the raw-only Chronicle package that registers the same bytes.
SOURCES: dict[str, dict[str, str | int]] = {
    "oa_hierarchy": {
        "url": EW_OA_HIERARCHY_URL,
        "name": "ew_oa21_lsoa_msoa_lad22_hierarchy.csv",
        "vintage": "2021_census",
        "publisher": "ONS",
        "product": (
            "Output Area (2021) to LSOA to MSOA to LAD (December 2021) Exact Fit "
            "Lookup in EW"
        ),
        "item_id": "b9ca90c10aaa4b8d9791e9859a38ca67",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-oa21-lsoa-msoa-lad22-lookup",
        "chronicle_year": 2021,
    },
    "oa_parncp_lad_region": {
        "url": EW_OA_PARNCP_LAD_REGION_URL,
        "name": "ew_oa21_parncp_lad24_rgn24_dec2024.csv",
        "vintage": "2023_april_lad",
        "publisher": "ONS",
        "product": (
            "Output Area (2021) to Parish/NCP to LAD to Region to Country "
            "(December 2024) Lookup in EW V2"
        ),
        "item_id": "7507c0292db546ed83e4ba60f1115b1d",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-oa21-parncp-lad-rgn-ctry-dec2024-lookup",
        "chronicle_year": 2024,
    },
    "oa_constituency": {
        "url": EW_OA_CONSTITUENCY_URL,
        "name": "ew_oa21_pcon24_lookup.csv",
        "vintage": "2024_pcon",
        "publisher": "ONS",
        "product": (
            "Output Area (2021) to Westminster Parliamentary Constituency "
            "(July 2024) Best Fit Lookup in EW"
        ),
        "item_id": "5968b5b2c0f14dd29ba277beaae6dec3",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-oa21-pcon-jul2024-lookup",
        "chronicle_year": 2024,
    },
    "oa_ward": {
        "url": EW_OA_WARD24_URL,
        "name": "ew_oa21_ward24_lad24_bestfit.csv",
        "vintage": "2024_wd",
        "publisher": "ONS",
        "product": (
            "Output Area (2021) to Ward (2024) to LAD (May 2024) Best Fit "
            "Lookup in EW V2"
        ),
        "item_id": "b21c632804094c149ccdfea1ae59c8c8",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-oa21-wd24-lad24-lookup",
        "chronicle_year": 2024,
    },
    "lad_itl": {
        "url": LAD24_ITL25_URL,
        "name": "uk_lad24_itl25_lookup.csv",
        "vintage": "2025_itl",
        "publisher": "ONS",
        "product": (
            "Local Authority District (December 2024) to LAU1 to ITL3 to ITL2 "
            "to ITL1 (January 2025) Lookup in the UK"
        ),
        "item_id": "15bdb6b0ff1c4a64b34a64e2e39f8caf",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-lad-dec2024-itl-jan2025-lookup",
        "chronicle_year": 2025,
    },
    "oa_population": {
        "url": EW_OA_POPULATION_URL,
        "name": "census2021-ts001.zip",
        "vintage": "2021_census",
        "publisher": "ONS (Nomis)",
        "product": "Census 2021 TS001 usual resident population by Output Area",
        "item_id": "",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-census2021-ts001-oa",
        "chronicle_year": 2021,
    },
    "oa_households": {
        "url": EW_OA_HOUSEHOLDS_URL,
        "name": "census2021-ts041.zip",
        "vintage": "2021_census",
        "publisher": "ONS (Nomis)",
        "product": "Census 2021 TS041 number of households by Output Area",
        "item_id": "",
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-census2021-ts041-oa",
        "chronicle_year": 2021,
    },
    "scotland_dz_iz": {
        "url": SCOTLAND_OA_DZ_IZ_URL,
        "name": "scotland_oa22_dz22_iz22.zip",
        "vintage": "2022_census",
        "publisher": "NRS",
        "product": "OA2022 to Data Zone 2022 to Intermediate Zone 2022 lookup",
        "item_id": "",
        "chronicle_source_id": "nrs",
        "chronicle_package_id": "nrs-oa22-dz22-iz22-lookup",
        "chronicle_year": 2022,
    },
    "scotland_lau": {
        "url": SCOTLAND_OA_LAU_ITL_URL,
        "name": "scotland_oa22_lau25_itl25.zip",
        "vintage": "2019_council_area",
        "publisher": "NRS",
        "product": "OA2022 to LAU 2025 to ITL 2025 lookup (council area via CA19)",
        "item_id": "",
        "chronicle_source_id": "nrs",
        "chronicle_package_id": "nrs-oa22-lau25-itl25-lookup",
        "chronicle_year": 2025,
    },
    "scotland_constituency": {
        "url": SCOTLAND_OA_CONSTITUENCY_URL,
        "name": "scotland_oa22_ukpc24.zip",
        "vintage": "2024_pcon",
        "publisher": "NRS",
        "product": "OA2022 to UK Parliamentary Constituency 2024 lookup",
        "item_id": "",
        "chronicle_source_id": "nrs",
        "chronicle_package_id": "nrs-oa22-ukpc24-lookup",
        "chronicle_year": 2024,
    },
    "scotland_population": {
        "url": SCOTLAND_OA_POPULATION_URL,
        "name": "scotland_outputarea2022_population.csv",
        "vintage": "2022_census",
        "publisher": "NRS",
        "product": "Output Area 2022 usual resident population",
        "item_id": "",
        "chronicle_source_id": "nrs",
        "chronicle_package_id": "nrs-census2022-oa22-usual-resident-population",
        "chronicle_year": 2022,
    },
    "scotland_census_index": {
        "url": SCOTLAND_CENSUS_INDEX_ZIP_URL,
        "name": "scotland_census_2022_index.zip",
        "vintage": "2022_census",
        "publisher": "NRS",
        "product": (
            "Census 2022 index: OA_TO_HIGHER_AREAS (EW2022 electoral ward) and "
            "Postcode_To_OA (occupied household counts, cell-key perturbed)"
        ),
        "item_id": "",
        "chronicle_source_id": "nrs",
        "chronicle_package_id": "nrs-census2022-index",
        "chronicle_year": 2022,
    },
    "ni_geojson": {
        "url": NI_DZ_GEOJSON_ZIP_URL,
        "name": "ni_dz2021_geojson.zip",
        "vintage": "2021_census",
        "publisher": "NISRA",
        "product": "Data Zone 2021 GeoJSON (SDZ2021, LGD2014 and DEA2014 properties)",
        "item_id": "",
        "chronicle_source_id": "nisra",
        "chronicle_package_id": "nisra-dz2021-geojson",
        "chronicle_year": 2021,
    },
    "ni_population": {
        "url": NI_DZ_POPULATION_CSV_URL,
        "name": "ni_dz21_population.csv",
        "vintage": "2021_census",
        "publisher": "NISRA",
        "product": "Census 2021 table builder, usual residents by DZ21",
        "item_id": "",
        "chronicle_source_id": "nisra",
        "chronicle_package_id": "nisra-census2021-dz21-people",
        "chronicle_year": 2021,
    },
    "ni_households": {
        "url": NI_DZ_HOUSEHOLDS_CSV_URL,
        "name": "ni_dz21_households.csv",
        "vintage": "2021_census",
        "publisher": "NISRA",
        "product": "Census 2021 table builder, households by DZ21",
        "item_id": "",
        "chronicle_source_id": "nisra",
        "chronicle_package_id": "nisra-census2021-dz21-households",
        "chronicle_year": 2021,
    },
    "ni_parlcon24_lookup": {
        "url": NI_DZ_PARLCON24_LOOKUP_XLSX_URL,
        "name": "ni_dz21_sdz21_lookups_v3.xlsx",
        "vintage": "2024_pcon",
        "publisher": "NISRA",
        "product": (
            "Geography Data Zone and Super Data Zone Lookups V3, "
            "DZ2021_Admin_geog_lookup (PARLCON2024, SDZ2021, LGD2014, DEA2014)"
        ),
        "item_id": "",
        "chronicle_source_id": "nisra",
        "chronicle_package_id": "nisra-dz-sdz-lookups-v3",
        "chronicle_year": 2025,
    },
}

#: Which publisher file supplies each support column, per system. The
#: relation is the publisher's own classification of the mapping.
_EW, _SCOT, _NI = SYSTEMS
COLUMN_SOURCES: dict[str, dict[str, tuple[str, str, str]]] = {
    # column: (source key, vintage string, relation)
    _EW: {
        "oa_code": ("oa_hierarchy", "2021_census", "exact"),
        "lsoa_code": ("oa_hierarchy", "2021_census", "exact"),
        "msoa_code": ("oa_hierarchy", "2021_census", "exact"),
        "local_authority_code": ("oa_parncp_lad_region", "2023_april_lad", "best_fit"),
        "region_code": ("oa_parncp_lad_region", "2024_rgn", "best_fit"),
        "constituency_code": ("oa_constituency", "2024_pcon", "best_fit"),
        "ward_code": ("oa_ward", "2024_wd", "best_fit"),
        "itl3_code": ("lad_itl", "2025_itl", "exact"),
    },
    _SCOT: {
        "oa_code": ("scotland_dz_iz", "2022_census", "exact"),
        "lsoa_code": ("scotland_dz_iz", "2022_census", "exact"),
        "msoa_code": ("scotland_dz_iz", "2022_census", "exact"),
        "local_authority_code": ("scotland_lau", "2019_council_area", "exact"),
        "region_code": ("scotland_dz_iz", "frs_region_sentinel", "exact"),
        "constituency_code": ("scotland_constituency", "2024_pcon", "best_fit"),
        "ward_code": ("scotland_census_index", "2022_ew", "best_fit"),
        "itl3_code": ("lad_itl", "2025_itl", "exact"),
    },
    _NI: {
        "oa_code": ("ni_parlcon24_lookup", "2021_census", "exact"),
        "lsoa_code": ("ni_parlcon24_lookup", "2021_census", "exact"),
        "msoa_code": ("ni_parlcon24_lookup", "2021_census", "exact"),
        "local_authority_code": ("ni_parlcon24_lookup", "2014_lgd", "exact"),
        "region_code": ("ni_geojson", "frs_region_sentinel", "exact"),
        "constituency_code": (
            "ni_parlcon24_lookup",
            "2024_pcon",
            "official_tabulation",
        ),
        "ward_code": ("ni_parlcon24_lookup", "2014_dea", "exact"),
        "itl3_code": ("lad_itl", "2025_itl", "exact"),
    },
}
COUNT_SOURCES: dict[str, dict[str, tuple[str, str]]] = {
    # column: (source key, basis)
    _EW: {
        "population": ("oa_population", "census_2021_usual_residents"),
        "households": ("oa_households", "census_2021_households"),
    },
    _SCOT: {
        "population": ("scotland_population", "census_2022_usual_residents"),
        "households": (
            "scotland_census_index",
            "census_2022_occupied_households_postcode_index_cell_key_perturbed",
        ),
    },
    _NI: {
        "population": ("ni_population", "census_2021_usual_residents"),
        "households": ("ni_households", "census_2021_households"),
    },
}
NATION_PREFIXES = {_EW: ("E", "W"), _SCOT: ("S",), _NI: ("N",)}

#: Reviewed conventions carried into the provenance register.
NOTES = (
    "E&W local_authority_code is the April 2023 LAD code set as published in the "
    "December 2024 OA->PARNCP->LAD->Region file (field LAD24CD); the 1 April 2025 "
    "Barnsley/Sheffield recode is a vintage-translation follow-up.",
    "E&W region_code is read per OA from the same December 2024 file (RGN24CD); "
    "Wales carries the W99999999 sentinel, Scotland S99999999, NI N99999999.",
    "The Ward 2024 file's LAD column is never read; the ITL 2025 file is joined "
    "on the April 2023 LAD codes (exact key join, #205 rule).",
    "The ONS LAD->ITL file lists North Ayrshire (S12000021) under two ITL3 rows "
    "(TLM20 Arran and Cumbrae first, TLM93 North Ayrshire mainland second); the "
    "shared loader keeps the first row, so North Ayrshire carries TLM20, as the "
    "ladder's loader did with the January 2021 file. Follow-up: prefer the "
    "mainland LAU or an explicit override.",
    "Scotland households are census 2022 occupied households summed from the NRS "
    "postcode index (cell-key perturbed).",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the three UK atomic-area support NPZ artifacts."
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "populace-uk-geography",
        help="Download cache; re-runs reuse verified files.",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        help="Pinned OA ladder NPZ to diff against (LA/constituency/region must not move).",
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=_DEFAULT_PROVENANCE,
        help="Committed provenance register to write.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Path for the build summary. Defaults to <out-dir>/summary.json.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Use only exact local inputs from a schema_version=1 JSON manifest "
            "with sources keyed exactly as this tool's SOURCES. Each row requires "
            "path, url, sha256 and bytes. Disables downloads."
        ),
    )
    parser.add_argument(
        "--crosswalk-resource",
        type=Path,
        default=_CROSSWALK_RESOURCE,
        help="Committed local_area_crosswalk.json whose rosters the supports must match.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Self-check definition seed."
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _fetch(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Resolve every source to a local path plus its identity."""
    fetched: dict[str, dict[str, object]] = {}
    if args.source_manifest is not None:
        rows, manifest_sha = _pinned_sources(
            args.source_manifest, {key: spec["url"] for key, spec in SOURCES.items()}
        )
        for key, spec in SOURCES.items():
            path = _verify_pinned_source(key, rows[key])
            fetched[key] = {**spec, "path": str(path), "sha256": rows[key]["sha256"]}
        fetched["__manifest_sha256__"] = {"sha256": manifest_sha}
        return fetched
    for key, spec in SOURCES.items():
        path = _download(str(spec["url"]), args.cache_dir, str(spec["name"]))
        fetched[key] = {**spec, "path": str(path), "sha256": _sha256(path)}
    return fetched


def _file_url(fetched: dict[str, dict[str, object]], key: str) -> str:
    return Path(str(fetched[key]["path"])).resolve().as_uri()


def _ew_region_lookup(
    hierarchy: pd.DataFrame, oa_lad_region: pd.DataFrame
) -> pd.DataFrame:
    """Derive the English LAD22 -> region lookup the E&W builder joins on.

    The builder keys regions on the hierarchy's LAD (December 2021) code
    (``la_code`` after normalisation). The December 2024 file assigns each OA
    its region directly, so the LAD22 -> region map is read off the OA rows and
    must be single-valued per LAD22; the retired LAD22 -> RGN22 product is not
    fetched.
    """
    rows = hierarchy[["oa_code", "la_code"]].merge(
        oa_lad_region[["oa_code", "region_code"]], on="oa_code", how="inner"
    )
    rows = rows[rows["oa_code"].str.startswith("E")]
    rows = (
        rows.loc[:, ["la_code", "region_code"]]
        .drop_duplicates()
        .sort_values("la_code")
        .reset_index(drop=True)
    )
    if (rows["region_code"] == "").any():
        raise ValueError("English OA rows without a region in the December 2024 file.")
    if rows["la_code"].duplicated().any():
        ambiguous = rows.loc[rows["la_code"].duplicated(), "la_code"].tolist()
        raise ValueError(f"LAD22 maps to more than one region: {ambiguous[:5]}.")
    return rows


def _build_ew_frame(fetched: dict[str, dict[str, object]]) -> pd.DataFrame:
    hierarchy = load_england_wales_oa_hierarchy(_file_url(fetched, "oa_hierarchy"))
    oa_lad_region = load_ew_oa_lad_region_lookup(
        _file_url(fetched, "oa_parncp_lad_region")
    )
    base = build_england_wales_crosswalk(
        hierarchy,
        load_england_wales_oa_population(_file_url(fetched, "oa_population")),
        load_england_wales_oa_constituencies(_file_url(fetched, "oa_constituency")),
        oa_lad_region[["oa_code", "la_code"]].rename(columns={"la_code": "lad23_code"}),
        _ew_region_lookup(hierarchy, oa_lad_region),
        expected_oa_count=geography_sources.ENGLAND_WALES_OA2021_COUNT,
    )
    return join_uk_oa_ladder_layers(
        base,
        oa_households=load_england_wales_oa_households(
            _file_url(fetched, "oa_households")
        ),
        oa_ward=load_england_wales_oa_ward_lookup(_file_url(fetched, "oa_ward")),
        lad_itl=load_lad_itl_lookup(_file_url(fetched, "lad_itl")),
    )


def _build_scotland_frame(fetched: dict[str, dict[str, object]]) -> pd.DataFrame:
    base = build_scotland_crosswalk(
        load_scotland_oa_dz_iz_lookup(_file_url(fetched, "scotland_dz_iz")),
        load_scotland_oa_lau_lookup(_file_url(fetched, "scotland_lau")),
        load_scotland_oa_constituencies(_file_url(fetched, "scotland_constituency")),
        load_scotland_oa_population(_file_url(fetched, "scotland_population")),
        expected_oa_count=geography_sources.SCOTLAND_OA2022_COUNT,
    )
    return join_uk_oa_ladder_layers(
        base,
        oa_households=load_scotland_oa_households(
            _file_url(fetched, "scotland_census_index")
        ),
        oa_ward=load_scotland_oa_ward_lookup(
            _file_url(fetched, "scotland_census_index")
        ),
        lad_itl=load_lad_itl_lookup(_file_url(fetched, "lad_itl")),
    )


def _build_ni_frame(fetched: dict[str, dict[str, object]]) -> pd.DataFrame:
    hierarchy = load_ni_dz_hierarchy(_file_url(fetched, "ni_geojson"))
    lookup = load_ni_dz_parlcon24_lookup(_file_url(fetched, "ni_parlcon24_lookup"))
    ward_lookup = load_ni_dz_ward_lookup(_file_url(fetched, "ni_geojson"))
    comparison = ward_lookup.merge(
        lookup[["oa_code", "ward_code"]],
        on="oa_code",
        suffixes=("_geojson", "_lookup"),
        validate="one_to_one",
    )
    mismatch = comparison[
        comparison["ward_code_geojson"] != comparison["ward_code_lookup"]
    ]
    if not mismatch.empty:
        raise ValueError(
            "NI published DZ lookup DEA codes disagree with the GeoJSON; "
            f"DZ code(s): {mismatch['oa_code'].tolist()[:5]}."
        )
    base = build_northern_ireland_crosswalk(
        hierarchy,
        load_ni_dz_population(_file_url(fetched, "ni_population")),
        lookup,
        expected_dz_count=geography_sources.NI_DZ2021_COUNT,
    )
    return join_uk_oa_ladder_layers(
        base,
        oa_households=load_ni_dz_households(_file_url(fetched, "ni_households")),
        oa_ward=ward_lookup,
        lad_itl=load_lad_itl_lookup(_file_url(fetched, "lad_itl")),
    )


def _roster(resource: Path) -> dict[str, set[str]]:
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return {
        level: set(payload["levels"][level]["area_ids"])
        for level in ("constituency", "local_authority")
    }


def _expected_areas() -> dict[str, int]:
    """National area counts, read at call time from the shared loader module."""
    return {
        _EW: geography_sources.ENGLAND_WALES_OA2021_COUNT,
        _SCOT: geography_sources.SCOTLAND_OA2022_COUNT,
        _NI: geography_sources.NI_DZ2021_COUNT,
    }


def _fence(system: str, frame: pd.DataFrame, roster: dict[str, set[str]]) -> None:
    """Refuse a nation frame whose codes, rosters or counts differ from the committed ones."""
    prefixes = NATION_PREFIXES[system]
    expected = _expected_areas()[system]
    if len(frame) != expected:
        raise ValueError(f"{system}: {len(frame):,} areas, expected {expected:,}.")
    if (frame["population"] <= 0).any():
        raise ValueError(f"{system}: an area has zero population.")
    if system == _EW:
        las = set(frame["local_authority_code"])
        if not {"E08000016", "E08000019"} <= las or las & {"E08000038", "E08000039"}:
            raise ValueError("E&W LAD codes are not the April 2023 code set.")
    if (
        system == _NI
        and frame["constituency_code"].nunique() != geography_sources.NI_PARLCON24_COUNT
    ):
        raise ValueError("NI must carry exactly 18 PARLCON2024 constituencies.")
    for level, column in (
        ("constituency", "constituency_code"),
        ("local_authority", "local_authority_code"),
    ):
        nation = {code for code in roster[level] if code[0] in prefixes}
        actual = set(frame[column].astype(str))
        if actual != nation:
            raise ValueError(
                f"{system}: {level} roster differs from local_area_crosswalk.json "
                f"(+{sorted(actual - nation)[:5]} -{sorted(nation - actual)[:5]})."
            )


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in LADDER_OA_COLUMNS:
        if column in {"population", "households"}:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy()
            arrays[column] = np.asarray(values, dtype=np.float64)
        else:
            arrays[column] = np.asarray(frame[column].astype(str).to_numpy(), dtype="U")
    return arrays


def _column_metadata(
    system: str, fetched: dict[str, dict[str, object]]
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for column, (key, vintage, relation) in COLUMN_SOURCES[system].items():
        spec = fetched[key]
        metadata[column] = {
            "kind": "code",
            "source": f"{spec['chronicle_package_id']}@sha256:{spec['sha256']}",
            "vintage": vintage,
            "relation": relation,
        }
    for column, (key, basis) in COUNT_SOURCES[system].items():
        spec = fetched[key]
        metadata[column] = {
            "kind": "weight",
            "source": f"{spec['chronicle_package_id']}@sha256:{spec['sha256']}",
            "basis": basis,
        }
    return metadata


def _ladder_diff(
    ladder_path: Path, frames: dict[str, pd.DataFrame]
) -> dict[str, object]:
    """Report which arrays moved against the pinned ladder, aligned by area code."""
    ladder = load_uk_oa_ladder(ladder_path)
    reference = pd.DataFrame(
        {column: getattr(ladder, column) for column in LADDER_OA_COLUMNS}
    ).set_index("oa_code")
    combined = pd.concat(frames.values(), ignore_index=True).set_index("oa_code")
    if set(reference.index) != set(combined.index):
        raise ValueError("support area roster differs from the pinned ladder.")
    combined = combined.loc[reference.index]
    changed: dict[str, int] = {}
    moved: dict[str, list[str]] = {}
    for column in LADDER_OA_COLUMNS:
        if column == "oa_code":
            continue
        left = (
            reference[column].astype(str)
            if column not in {"population", "households"}
            else reference[column].astype(float)
        )
        right = (
            combined[column].astype(str)
            if column not in {"population", "households"}
            else combined[column].astype(float)
        )
        differs = left.to_numpy() != right.to_numpy()
        changed[column] = int(differs.sum())
        if (
            column
            in {
                "local_authority_code",
                "constituency_code",
                "region_code",
                "population",
                "households",
                "lsoa_code",
                "msoa_code",
            }
            and differs.any()
        ):
            moved[column] = [str(code) for code in reference.index[differs][:10]]
    return {
        "ladder_sha256": _sha256(ladder_path),
        "areas": int(len(reference)),
        "changed_areas_by_column": changed,
        "unexpected_moves": moved,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _log("Building UK atomic-area supports")
    fetched = _fetch(args)
    roster = _roster(args.crosswalk_resource)
    builders = {
        _EW: _build_ew_frame,
        _SCOT: _build_scotland_frame,
        _NI: _build_ni_frame,
    }
    frames: dict[str, pd.DataFrame] = {}
    payloads: dict[str, bytes] = {}
    for system, build in builders.items():
        _log(f"  joining {system}")
        frame = build(fetched)
        _fence(system, frame, roster)
        frames[system] = frame
        payloads[system] = assemble_uk_atomic_area_support(
            system=system,
            arrays=_arrays(frame),
            column_metadata=_column_metadata(system, fetched),
        )
        if len(payloads[system]) > RAW_BYTES_MAX_BYTES:
            raise ValueError(f"{system}: support exceeds the raw-bytes-v1 cap.")
    # Self-check only: the definition (and its sha) is a run-time binding of the
    # full build (seed, law), not a property of the support artifacts.
    uk_atomic_assignment_definition(dict(payloads), seed=args.seed)
    diff = _ladder_diff(args.ladder, frames) if args.ladder is not None else None
    if diff is not None and diff["unexpected_moves"]:
        raise ValueError(
            f"unexpected array moves against the pinned ladder: {diff['unexpected_moves']}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    supports: dict[str, dict[str, object]] = {}
    for system, payload in payloads.items():
        path = args.out_dir / f"{SUPPORT_SOURCES[system]}.npz"
        path.write_bytes(payload)
        support = decode_atomic_support(path.read_bytes())
        frame = frames[system]
        supports[system] = {
            "filename": path.name,
            "sha256": support.sha256,
            "size_bytes": len(payload),
            "areas": int(len(frame)),
            "population_total": int(frame["population"].sum()),
            "households_total": int(frame["households"].sum()),
            "constituencies": int(frame["constituency_code"].nunique()),
            "local_authorities": int(frame["local_authority_code"].nunique()),
            "wards": int(frame["ward_code"].nunique()),
            "column_metadata": {
                k: dict(v) for k, v in support.metadata["columns"].items()
            },
        }
    publisher_files = {
        key: {
            "url": spec["url"],
            "sha256": spec["sha256"],
            "size_bytes": Path(str(spec["path"])).stat().st_size,
            "vintage": spec["vintage"],
            "publisher": spec["publisher"],
            "product": spec["product"],
            "item_id": spec["item_id"],
            "chronicle_source_id": spec["chronicle_source_id"],
            "chronicle_package_id": spec["chronicle_package_id"],
            "chronicle_year": spec["chronicle_year"],
            "retrieved_at": dt.datetime.fromtimestamp(
                Path(str(spec["path"])).stat().st_mtime, tz=dt.UTC
            )
            .date()
            .isoformat(),
        }
        for key, spec in fetched.items()
        if not key.startswith("__")
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": PROVENANCE_KIND,
        "built_on": dt.date.today().isoformat(),
        "tool": "tools/build_uk_atomic_area_supports.py",
        "ladder_reference": None
        if args.ladder is None
        else {"path": args.ladder.name, "sha256": diff["ladder_sha256"]},
        "publisher_files": publisher_files,
        "supports": supports,
        "ladder_diff": None
        if diff is None
        else {k: v for k, v in diff.items() if k != "ladder_sha256"},
        "reviewed_differences": [],
        "notes": NOTES,
    }
    args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_json.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = args.summary_json or (args.out_dir / "summary.json")
    summary_path.write_text(
        json.dumps(
            {"provenance": str(args.provenance_json), **provenance},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for system, entry in supports.items():
        _log(
            f"  {system}: {entry['areas']:,} areas, {entry['size_bytes']:,} bytes, sha256 {entry['sha256'][:16]}"
        )
    _log(f"  provenance written to {args.provenance_json}")


if __name__ == "__main__":
    main()
