"""Build the UK OA-ladder artifact from primary ONS/Nomis sources.

Downloads (with a local cache) the England-&-Wales 2021 Census output-area
sources — the OA -> LSOA -> MSOA -> LAD structural hierarchy, OA -> LAD (April
2023), OA -> PCON24 constituency best-fit, OA usual-resident population (Nomis
TS001), OA household counts (Nomis TS041), the OA -> ward best-fit, and the LAD
(April 2023) -> ITL lookup — joins them at 2021-OA grain, and writes one
national NPZ artifact whose embedded metadata records a vintage, source, URL,
and sha256 per derived layer (``vintage_policy: error`` — the loader refuses an
artifact missing any of them). No per-area files, the standing rule (#275).

The artifact is self-checked by loading it back through
``populace.build.uk_runtime.load_uk_oa_ladder`` before the summary is written,
so a published ladder is by construction a loadable ladder.

England & Wales is the first milestone (populace #349). Scotland and Northern
Ireland follow once their lookup vintages are pinned — the assignment refuses a
household whose region is absent from the ladder, so a GB/UK build cannot
silently ship partial coverage.

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

from populace.build.uk_runtime import (
    ENGLAND_LAD_REGION_URL,
    EW_OA_CONSTITUENCY_URL,
    EW_OA_HIERARCHY_URL,
    EW_OA_HOUSEHOLDS_URL,
    EW_OA_LAD23_URL,
    EW_OA_POPULATION_URL,
    EW_OA_WARD_URL,
    LAD23_ITL_URL,
    assemble_uk_oa_ladder,
    build_england_wales_crosswalk,
    join_uk_oa_ladder_layers,
    load_england_lad_region_lookup,
    load_england_wales_oa_constituencies,
    load_england_wales_oa_hierarchy,
    load_england_wales_oa_households,
    load_england_wales_oa_population,
    load_england_wales_oa_ward_lookup,
    load_ew_oa_lad23_lookup,
    load_lad_itl_lookup,
    load_uk_oa_ladder,
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the national UK OA-ladder NPZ artifact (England & Wales)."
    )
    parser.add_argument("--out", required=True, type=Path)
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
        url, headers={"User-Agent": "populace-build (UK OA ladder artifact)"}
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310 (trusted ONS URLs)
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty download from {url}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


def _fetch(cache_dir: Path) -> dict[str, dict[str, str]]:
    """Download every source, returning the cache path and sha256 per source."""
    fetched: dict[str, dict[str, str]] = {}
    for key, spec in SOURCES.items():
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


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _log("Building UK OA ladder for England & Wales")
    fetched = _fetch(args.cache_dir)

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

    metadata = {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "england_and_wales",
        "oa_vintage": "2021_census",
        "constituency_sampling_basis": "census_2021_household_counts",
        "oa_sampling_basis": "census_2021_usual_resident_population",
        "layers": _layer_metadata(fetched),
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
