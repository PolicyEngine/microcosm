"""Build the UK OA-ladder artifact from primary ONS/Nomis sources.

Builds the full-UK ladder by default: England & Wales at 2021 Census OA
grain, Scotland at 2022 Census OA grain, and Northern Ireland at DZ2021
grain (--coverage england_and_wales preserves the original milestone).
Downloads (with a local cache) the England-&-Wales 2021 Census output-area
sources — the OA -> LSOA -> MSOA -> LAD structural hierarchy, OA -> LAD (April
2023), OA -> PCON24 constituency best-fit, OA usual-resident population (Nomis
TS001), OA household counts (Nomis TS041), the OA -> ward best-fit, and the LAD
(April 2023) -> ITL lookup — joins them at 2021-OA grain, and writes one
national NPZ artifact whose embedded metadata records a vintage, source, URL,
and sha256 per derived layer (``vintage_policy: error`` — the loader refuses an
artifact missing any of them). No per-area files, the standing rule (#275).

The artifact is self-checked by loading it back through
``microcosm.build.uk_runtime.load_uk_oa_ladder`` before the summary is written,
so a published ladder is by construction a loadable ladder.

Scotland and Northern Ireland vintages were pinned on microcosm#495
(increment 3): both Scotland ladder-only layers come from the NRS Census
2022 index zip, NI households from the NISRA table builder, and the NI ward
analogue (DEA2014) from the pinned DZ2021 GeoJSON. The assignment still
refuses a household whose region is absent from the ladder, so a partial
build cannot silently ship.

Example:
    uv run python tools/build_uk_oa_ladder_artifact.py \
        --out build/uk/uk_oa_ladder_2021.npz \
        --cache-dir ~/.cache/populace-uk-geography
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import (
    ENGLAND_LAD_REGION_URL,
    EW_OA_CONSTITUENCY_URL,
    EW_OA_HIERARCHY_URL,
    EW_OA_HOUSEHOLDS_URL,
    EW_OA_LAD23_URL,
    EW_OA_POPULATION_URL,
    EW_OA_WARD_URL,
    LAD23_ITL_URL,
    NI_DZ_GEOJSON_ZIP_URL,
    NI_DZ_HOUSEHOLDS_CSV_URL,
    NI_DZ_POPULATION_CSV_URL,
    SCOTLAND_CENSUS_INDEX_ZIP_URL,
    SCOTLAND_OA_CONSTITUENCY_URL,
    SCOTLAND_OA_DZ_IZ_URL,
    SCOTLAND_OA_LAU_ITL_URL,
    SCOTLAND_OA_POPULATION_URL,
    UK_POSTCODE_OA_MAY25_ZIP_URL,
    UK_POSTCODE_PCON_MAY24_ZIP_URL,
    assemble_uk_oa_ladder,
    build_england_wales_crosswalk,
    build_northern_ireland_crosswalk,
    build_scotland_crosswalk,
    concat_uk_ladder_frames,
    infer_ni_dz_constituencies_from_postcodes,
    join_uk_oa_ladder_layers,
    load_england_lad_region_lookup,
    load_england_wales_oa_constituencies,
    load_england_wales_oa_hierarchy,
    load_england_wales_oa_households,
    load_england_wales_oa_population,
    load_england_wales_oa_ward_lookup,
    load_ew_oa_lad23_lookup,
    load_lad_itl_lookup,
    load_ni_dz_hierarchy,
    load_ni_dz_households,
    load_ni_dz_population,
    load_ni_dz_ward_lookup,
    load_scotland_oa_constituencies,
    load_scotland_oa_dz_iz_lookup,
    load_scotland_oa_households,
    load_scotland_oa_lau_lookup,
    load_scotland_oa_population,
    load_scotland_oa_ward_lookup,
    load_uk_oa_ladder,
    load_uk_postcode_constituency_lookup,
    load_uk_postcode_oa_lookup,
)

#: Per-source provenance: cache filename, download URL, and the layer(s) it
#: feeds. Vintages are recorded in the artifact metadata below. Each URL is the
#: ONS Open Geography Portal / Nomis endpoint cited in ``geography_sources.py``.
SOURCES = {
    "oa_hierarchy": {
        "url": EW_OA_HIERARCHY_URL,
        "name": "ew_oa21_lsoa_msoa_lad22_hierarchy.csv",
        "vintage": "2021_census",
        "citation": (
            "ONS Open Geography Portal, Output Area (2021) to LSOA to MSOA to "
            "LAD (2022) best fit lookup in EW (item b9ca90c1...)"
        ),
    },
    "oa_lad23": {
        "url": EW_OA_LAD23_URL,
        "name": "ew_oa21_lad23_lookup.csv",
        "vintage": "2023_april_lad",
        "citation": (
            "ONS Open Geography Portal, OA (2021) to LAD (April 2023) best fit "
            "lookup in EW (item 83982ff4...)"
        ),
    },
    "oa_constituency": {
        "url": EW_OA_CONSTITUENCY_URL,
        "name": "ew_oa21_pcon24_lookup.csv",
        "vintage": "2024_pcon",
        "citation": (
            "ONS Open Geography Portal, OA (2021) to Westminster Parliamentary "
            "Constituency (July 2024) best fit lookup in EW (item 5968b5b2...)"
        ),
    },
    "lad_region": {
        "url": ENGLAND_LAD_REGION_URL,
        "name": "england_lad22_region22_lookup.csv",
        "vintage": "2022_rgn",
        "citation": (
            "ONS Open Geography Portal, LAD (2022) to Region (2022) lookup in "
            "England (item 78b348cd...)"
        ),
    },
    "oa_population": {
        "url": EW_OA_POPULATION_URL,
        "name": "census2021-ts001.zip",
        "vintage": "2021_census",
        "citation": "Nomis Census 2021 TS001 (usual resident population), OA grain",
    },
    "oa_households": {
        "url": EW_OA_HOUSEHOLDS_URL,
        "name": "census2021-ts041.zip",
        "vintage": "2021_census",
        "citation": "Nomis Census 2021 TS041 (number of households), OA grain",
    },
    "oa_ward": {
        "url": EW_OA_WARD_URL,
        "name": "ew_oa21_ward22_bestfit.csv",
        "vintage": "2022_wd",
        "citation": (
            "ONS Open Geography Portal, Output Area (2021) to Ward to LAD to "
            "CTYUA to Region to Country (2022) best fit lookup in EW V2 "
            "(item 7207b517...)"
        ),
    },
    "lad_itl": {
        "url": LAD23_ITL_URL,
        "name": "uk_lad23_itl21_lookup.csv",
        "vintage": "2021_itl",
        "citation": (
            "ONS Open Geography Portal, LAD (April 2023) to LAU1 to ITL3 to "
            "ITL2 to ITL1 (January 2021) lookup in the UK (item 02b49429...)"
        ),
    },
}

#: Scotland and Northern Ireland sources (microcosm#495 increment 3). Only
#: fetched for --coverage uk.
UK_EXTRA_SOURCES = {
    "scotland_dz_iz": {
        "url": SCOTLAND_OA_DZ_IZ_URL,
        "name": "scotland_oa22_dz22_iz22.zip",
        "vintage": "2022_census",
        "citation": "NRS, OA2022 to Data Zone 2022 to Intermediate Zone 2022 lookup",
    },
    "scotland_lau": {
        "url": SCOTLAND_OA_LAU_ITL_URL,
        "name": "scotland_oa22_lau25_itl25.zip",
        "vintage": "2019_council_area",
        "citation": "NRS, OA2022 to LAU 2025 lookup (council area via CA19 join)",
    },
    "scotland_constituency": {
        "url": SCOTLAND_OA_CONSTITUENCY_URL,
        "name": "scotland_oa22_ukpc24.zip",
        "vintage": "2024_pcon",
        "citation": "NRS, OA2022 to UK Parliamentary Constituency 2024 lookup",
    },
    "scotland_population": {
        "url": SCOTLAND_OA_POPULATION_URL,
        "name": "scotland_outputarea2022_population.csv",
        "vintage": "2022_census",
        "citation": "NRS, Output Area 2022 usual resident population",
    },
    "scotland_census_index": {
        "url": SCOTLAND_CENSUS_INDEX_ZIP_URL,
        "name": "scotland_census_2022_index.zip",
        "vintage": "2022_census",
        "citation": (
            "NRS Census 2022 index: OA_TO_HIGHER_AREAS (EW2022 electoral "
            "ward, PiP centroid) and Postcode_To_OA (census occupied "
            "household counts, cell-key perturbed), per the in-zip index "
            "file specification"
        ),
    },
    "ni_geojson": {
        "url": NI_DZ_GEOJSON_ZIP_URL,
        "name": "ni_dz2021_geojson.zip",
        "vintage": "2021_census",
        "citation": (
            "NISRA DZ2021 GeoJSON (SDZ2021, LGD2014, and DEA2014 feature properties)"
        ),
    },
    "ni_population": {
        "url": NI_DZ_POPULATION_CSV_URL,
        "name": "ni_dz21_population.csv",
        "vintage": "2021_census",
        "citation": "NISRA Census 2021 table builder, usual residents by DZ21",
    },
    "ni_households": {
        "url": NI_DZ_HOUSEHOLDS_CSV_URL,
        "name": "ni_dz21_households.csv",
        "vintage": "2021_census",
        "citation": "NISRA Census 2021 table builder, households by DZ21",
    },
    "postcode_oa": {
        "url": UK_POSTCODE_OA_MAY25_ZIP_URL,
        "name": "uk_postcode_oa21_may25.zip",
        "vintage": "2025_may_postcodes",
        "citation": "ONS, UK postcode to OA (2021) lookup (May 2025)",
    },
    "postcode_pcon": {
        "url": UK_POSTCODE_PCON_MAY24_ZIP_URL,
        "name": "uk_postcode_pcon24_may24.zip",
        "vintage": "2024_pcon",
        "citation": "ONS, UK postcode to Westminster constituency (May 2024)",
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the national UK OA-ladder NPZ artifact."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--coverage",
        choices=("uk", "england_and_wales"),
        default="uk",
        help=(
            "Geographic coverage: full UK (England & Wales OA21 + Scotland "
            "OA22 + Northern Ireland DZ21) or the original England & Wales "
            "milestone."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "populace-uk-geography",
        help="Download cache; re-runs reuse verified files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Path for the build summary. Defaults beside --out.",
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / name
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    _log(f"  downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "microcosm-build (UK OA ladder artifact)"}
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310 (trusted ONS URLs)
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty download from {url}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


def _fetch(
    cache_dir: Path,
    sources: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Download every source, returning the cache path and sha256 per source."""
    fetched: dict[str, dict[str, str]] = {}
    for key, spec in (SOURCES if sources is None else sources).items():
        path = _download(spec["url"], cache_dir, spec["name"])
        fetched[key] = {
            "path": str(path),
            "sha256": _sha256(path),
            "url": spec["url"],
            "vintage": spec["vintage"],
            "citation": spec["citation"],
        }
    return fetched


def _file_url(fetched: dict[str, dict[str, str]], key: str) -> str:
    return Path(fetched[key]["path"]).resolve().as_uri()


def _layer_metadata(fetched: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    def entry(key: str) -> dict[str, str]:
        spec = fetched[key]
        return {
            "vintage": spec["vintage"],
            "source": spec["citation"],
            "url": spec["url"],
            "sha256": spec["sha256"],
        }

    return {
        "constituency": entry("oa_constituency"),
        "lsoa": entry("oa_hierarchy"),
        "msoa": entry("oa_hierarchy"),
        "local_authority": entry("oa_lad23"),
        "ward": entry("oa_ward"),
        "itl": entry("lad_itl"),
        "region": entry("lad_region"),
    }


def _build_scotland_frame(fetched: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Join Scotland's OA22-grain ladder frame from the pinned NRS sources."""

    base = build_scotland_crosswalk(
        load_scotland_oa_dz_iz_lookup(_file_url(fetched, "scotland_dz_iz")),
        load_scotland_oa_lau_lookup(_file_url(fetched, "scotland_lau")),
        load_scotland_oa_constituencies(_file_url(fetched, "scotland_constituency")),
        load_scotland_oa_population(_file_url(fetched, "scotland_population")),
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


def _build_ni_frame(fetched: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Join Northern Ireland's DZ21-grain ladder frame from NISRA sources."""

    base = build_northern_ireland_crosswalk(
        load_ni_dz_hierarchy(_file_url(fetched, "ni_geojson")),
        load_ni_dz_population(_file_url(fetched, "ni_population")),
        infer_ni_dz_constituencies_from_postcodes(
            load_uk_postcode_oa_lookup(_file_url(fetched, "postcode_oa")),
            load_uk_postcode_constituency_lookup(_file_url(fetched, "postcode_pcon")),
        ),
    )
    return join_uk_oa_ladder_layers(
        base,
        oa_households=load_ni_dz_households(_file_url(fetched, "ni_households")),
        oa_ward=load_ni_dz_ward_lookup(_file_url(fetched, "ni_geojson")),
        lad_itl=load_lad_itl_lookup(_file_url(fetched, "lad_itl")),
    )


def _uk_layer_metadata(
    fetched: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Per-layer metadata for full-UK coverage: one composite vintage per
    derived layer (vintage_policy: error), with per-country identities in the
    countries submap and every source sha-pinned in source_files."""

    ew = _layer_metadata(fetched)

    def country_entry(key: str) -> dict[str, object]:
        return {
            "vintage": fetched[key]["vintage"],
            "source": fetched[key]["citation"],
            "url": fetched[key]["url"],
            "sha256": fetched[key]["sha256"],
        }

    def composite(
        layer: str,
        scotland_key: str,
        ni_key: str,
        vintage: str,
        *,
        ni_entry: dict[str, object] | None = None,
    ) -> dict[str, object]:
        # No top-level url/sha256 on composite layers: a single conventional
        # identity would go stale when only Scotland/NI inputs change. The
        # per-country identities below and source_files carry every sha.
        entry = {
            "vintage": vintage,
            "source": (
                f"{ew[layer]['source']}; Scotland: "
                f"{fetched[scotland_key]['citation']}; NI: "
                f"{fetched[ni_key]['citation']}"
            ),
            "countries": {
                "england_and_wales": ew[layer],
                "scotland": country_entry(scotland_key),
                "northern_ireland": (
                    country_entry(ni_key) if ni_entry is None else ni_entry
                ),
            },
        }
        return entry

    ni_constituency_entry = {
        "vintage": "2024_pcon",
        "source": (
            "Reviewed active-postcode modal inference "
            "(infer_ni_dz_constituencies_from_postcodes) joining the ONS "
            "postcode->OA (May 2025) and postcode->PCON24 (May 2024) lookups; "
            "fenced by the max-unmatched-active-postcode share."
        ),
        "sources": {
            "postcode_oa": country_entry("postcode_oa"),
            "postcode_constituency": country_entry("postcode_pcon"),
        },
    }
    return {
        "constituency": composite(
            "constituency",
            "scotland_constituency",
            "postcode_pcon",
            "2024_pcon",
            ni_entry=ni_constituency_entry,
        ),
        "lsoa": composite(
            "lsoa",
            "scotland_dz_iz",
            "ni_geojson",
            "ew:2021_census;scotland:2022_census;ni:2021_census",
        ),
        "msoa": composite(
            "msoa",
            "scotland_dz_iz",
            "ni_geojson",
            "ew:2021_census;scotland:2022_census;ni:2021_census",
        ),
        "local_authority": composite(
            "local_authority",
            "scotland_lau",
            "ni_geojson",
            "ew:2023_april_lad;scotland:2019_council_area;ni:2014_lgd",
        ),
        "ward": composite(
            "ward",
            "scotland_census_index",
            "ni_geojson",
            "ew:2022_wd;scotland:2022_ew;ni:2014_dea",
        ),
        "itl": ew["itl"],
        "region": composite(
            "region",
            "scotland_dz_iz",
            "ni_geojson",
            "ew:2022_rgn;scotland:sentinel_s99999999;ni:sentinel_n99999999",
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    full_uk = args.coverage == "uk"
    _log(f"Building UK OA ladder ({args.coverage})")
    sources = dict(SOURCES)
    if full_uk:
        sources.update(UK_EXTRA_SOURCES)
    fetched = _fetch(args.cache_dir, sources)

    _log("  parsing base England & Wales OA crosswalk")
    base = build_england_wales_crosswalk(
        load_england_wales_oa_hierarchy(_file_url(fetched, "oa_hierarchy")),
        load_england_wales_oa_population(_file_url(fetched, "oa_population")),
        load_england_wales_oa_constituencies(_file_url(fetched, "oa_constituency")),
        load_ew_oa_lad23_lookup(_file_url(fetched, "oa_lad23")),
        load_england_lad_region_lookup(_file_url(fetched, "lad_region")),
    )

    _log("  joining ladder-only layers (households, ward, ITL)")
    joined = join_uk_oa_ladder_layers(
        base,
        oa_households=load_england_wales_oa_households(
            _file_url(fetched, "oa_households")
        ),
        oa_ward=load_england_wales_oa_ward_lookup(_file_url(fetched, "oa_ward")),
        lad_itl=load_lad_itl_lookup(_file_url(fetched, "lad_itl")),
    )

    if full_uk:
        _log("  joining Scotland OA22 ladder frame")
        scotland = _build_scotland_frame(fetched)
        _log("  joining Northern Ireland DZ21 ladder frame")
        northern_ireland = _build_ni_frame(fetched)
        _log(
            f"  concatenating {len(joined):,} EW + {len(scotland):,} S + "
            f"{len(northern_ireland):,} NI output areas"
        )
        joined = concat_uk_ladder_frames(joined, scotland, northern_ireland)

    metadata = {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": args.coverage,
        "oa_vintage": (
            "ew:2021_census;scotland:2022_census;ni:dz2021"
            if full_uk
            else "2021_census"
        ),
        "constituency_sampling_basis": (
            "census household counts (EW TS041 2021; Scotland census 2022 "
            "postcode index; NI census 2021 table builder)"
            if full_uk
            else "census_2021_household_counts"
        ),
        "oa_sampling_basis": (
            "census usual-resident population (EW TS001 2021; Scotland NRS "
            "OA22; NI census 2021)"
            if full_uk
            else "census_2021_usual_resident_population"
        ),
        "layers": _uk_layer_metadata(fetched) if full_uk else _layer_metadata(fetched),
        "source_files": {
            key: {"sha256": spec["sha256"], "url": spec["url"]}
            for key, spec in fetched.items()
        },
    }
    _log(f"  assembling {len(joined):,} joined output areas")
    payload = assemble_uk_oa_ladder(joined, metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    ladder = load_uk_oa_ladder(args.out)  # self-check: must load cleanly
    summary = {
        "output": str(args.out.resolve()),
        "output_sha256": _sha256(args.out),
        "output_areas": int(len(ladder)),
        "population": float(ladder.population.sum()),
        "households": float(ladder.households.sum()),
        "constituencies": int(np.unique(ladder.constituency_code).size),
        "local_authorities": int(np.unique(ladder.local_authority_code).size),
        "wards": int(np.unique(ladder.ward_code).size),
        "regions": int(np.unique(ladder.region_code).size),
        "layer_vintages": ladder.layer_vintages,
        "source_files": {
            key: {"sha256": spec["sha256"], "url": spec["url"]}
            for key, spec in fetched.items()
        },
    }
    summary_path = (
        args.summary_json
        if args.summary_json is not None
        else args.out.with_suffix(".summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
