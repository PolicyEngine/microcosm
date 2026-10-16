"""Toy UK atomic-area supports whose codes match the toy OA ladder.

The toy ladder (``uk_ladder_rowwise_clone._ladder_frame``) carries two
London output areas, one Welsh, one Scottish and one Northern Irish. The UK
adapter requires every FRS region of a nation system to carry household mass,
so the England & Wales support adds one filler area per remaining English
region; no toy household lives there. Every code the toy ladder's rosters,
target fixtures and gates know is therefore also an atomic-area code here.
No publisher file or country engine is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microcosm.build.uk_runtime.atomic_area_support import (
    SOURCES,
    SYSTEMS,
    assemble_uk_atomic_area_support,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from test_support.microcosm_build.uk_ladder_rowwise_clone import _ladder_frame

EW, SCOT, NI = SYSTEMS
_COLUMNS = (
    "oa_code",
    "population",
    "households",
    "constituency_code",
    "region_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "itl3_code",
)


def _filler_rows() -> list[dict]:
    """One populated area per English region absent from the toy ladder."""
    present = set(_ladder_frame()["region_code"])
    rows = []
    letters = "CDEFGHJK"
    for offset, (_name, code) in enumerate(
        (n, c)
        for n, c in FRS_REGION_TO_REGION_CODE.items()
        if c.startswith("E") and c not in present
    ):
        serial = 100 + offset
        rows.append(
            {
                "oa_code": f"E00{serial:06d}",
                "population": 1.0,
                "households": 1.0,
                "constituency_code": f"E14{serial:06d}",
                "region_code": code,
                "lsoa_code": f"E01{serial:06d}",
                "msoa_code": f"E02{serial:06d}",
                "local_authority_code": f"E06{serial:06d}",
                "ward_code": f"E05{serial:06d}",
                "itl3_code": f"TL{letters[offset]}11",
            }
        )
    return rows


def toy_support_frames() -> dict[str, list[dict]]:
    ladder = _ladder_frame().to_dict("records")
    return {
        EW: [r for r in ladder if r["oa_code"][0] in "EW"] + _filler_rows(),
        SCOT: [r for r in ladder if r["oa_code"][0] == "S"],
        NI: [r for r in ladder if r["oa_code"][0] == "N"],
    }


def _metadata(system: str) -> dict:
    vintage = "2022_census" if system == SCOT else "2021_census"
    descriptions = {}
    for column in _COLUMNS:
        if column in {"population", "households"}:
            descriptions[column] = {
                "kind": "weight",
                "source": f"toy-{system}-{column}",
                "basis": "toy persons" if column == "population" else "toy households",
            }
        else:
            descriptions[column] = {
                "kind": "code",
                "source": f"toy-{system}-{column}",
                "vintage": vintage,
                "relation": "exact"
                if column in {"oa_code", "lsoa_code", "msoa_code", "region_code"}
                else "best_fit",
            }
    descriptions["constituency_code"]["vintage"] = "2024_pcon"
    if system == NI:
        descriptions["constituency_code"]["relation"] = "official_tabulation"
    return descriptions


def toy_support_payloads() -> dict[str, bytes]:
    """The three toy support artifacts, keyed by system."""
    payloads = {}
    for system, rows in toy_support_frames().items():
        arrays = {}
        for column in _COLUMNS:
            values = [row[column] for row in rows]
            arrays[column] = (
                np.asarray(values, dtype=np.float64)
                if column in {"population", "households"}
                else np.asarray([str(v) for v in values], dtype="U")
            )
        payloads[system] = assemble_uk_atomic_area_support(
            system=system, arrays=arrays, column_metadata=_metadata(system)
        )
    return payloads


def write_toy_supports(directory: Path) -> tuple[dict[str, bytes], dict[str, Path]]:
    """Write the toy supports as raw-bytes sources named by ``SOURCES``."""
    directory.mkdir(parents=True, exist_ok=True)
    payloads = toy_support_payloads()
    paths = {}
    for system, payload in payloads.items():
        path = directory / f"{SOURCES[system]}.npz"
        path.write_bytes(payload)
        paths[system] = path
    return payloads, paths


def toy_support_sources(paths: dict[str, Path]) -> dict[str, Path]:
    """Graph source bindings for the three toy supports."""
    return {SOURCES[system]: path for system, path in paths.items()}


def toy_property_resources() -> dict[str, dict]:
    """The microcosm#1240 knots and band-mix resources over the toy areas.

    Every toy constituency carries one straight-line distribution from
    100,000 to 2,000,000 (so a uniform ``u`` reads ``100000 + 1900000 u``),
    winsorised at 2,000,000; every toy LSOA splits bands A to H equally.
    """
    from microcosm.build.uk_runtime.property_localisation import COUNCIL_TAX_BANDS

    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    seats = {}
    lsoas = {}
    for rows in toy_support_frames().values():
        for row in rows:
            seat = str(row["constituency_code"])
            if seat[0] in "EW":
                seats[seat] = {
                    "count": 100,
                    "weight": 100.0,
                    "quantiles": [100_000.0 + 1_900_000.0 * p for p in grid],
                }
            lsoa = str(row["lsoa_code"])
            if lsoa[0] in "EW":
                cumulative = [(i + 1) / 8 for i in range(8)] + [1.0]
                lsoas[lsoa] = {"all_properties": 800, "cumulative": cumulative}
    knots = {
        "schema_version": 1,
        "kind": "uk_high_value_home_value_knots",
        "resource": "toy-knots.json",
        "built_on": "2026-10-13",
        "inputs_sha256": "toy",
        "method": "toy",
        "basis_year": 2024,
        "cap": 2_000_000,
        "minimum_count": 1,
        "weight": "toy",
        "grid": grid,
        "levels": {"constituency": seats},
        "suppressed": {"constituency": []},
    }
    band_mix = {
        "schema_version": 1,
        "kind": "uk_voa_council_tax_band_mix_lsoa",
        "resource": "toy-band-mix.json",
        "built_on": "2026-10-13",
        "method": "toy",
        "description": "toy",
        "source": {"sha256": "toy", "as_at": "2025-03-31"},
        "bands": list(COUNCIL_TAX_BANDS),
        "receipts": {},
        "lsoa": lsoas,
    }
    return {"knots": knots, "band_mix": band_mix}


def write_toy_property_resources(directory: Path) -> dict[str, Path]:
    """Write the toy knots and band mix as graph sources named by the nodes."""
    from microcosm.build.uk_runtime.property_localisation import (
        BAND_MIX_SOURCE,
        QUANTILES_SOURCE,
    )

    directory.mkdir(parents=True, exist_ok=True)
    resources = toy_property_resources()
    paths = {
        QUANTILES_SOURCE: directory / "toy-knots.json",
        BAND_MIX_SOURCE: directory / "toy-band-mix.json",
    }
    paths[QUANTILES_SOURCE].write_text(json.dumps(resources["knots"]), encoding="utf-8")
    paths[BAND_MIX_SOURCE].write_text(
        json.dumps(resources["band_mix"]), encoding="utf-8"
    )
    return paths


#: The BRMA the toy layer gives each toy area (microcosm#1262); every other
#: England-and-Wales filler area sits in MAIDSTONE.
TOY_AREA_BRMAS = {
    "E00000001": "CENTRAL_LONDON",
    "E00000002": "INNER_EAST_LONDON",
    "W00000001": "CARDIFF",
    "S00000001": "LOTHIAN",
}


def toy_brma_area_layer() -> dict:
    """The microcosm#1262 area-to-BRMA layer over the toy supports."""
    from microcosm.build.uk_runtime import brma_area_layer as bal

    regions = {code: name for name, code in FRS_REGION_TO_REGION_CODE.items()}
    frames = toy_support_frames()
    layout: dict[str, dict] = {"ew_lsoa21": {}, "scotland_oa22": {}}
    rosters: dict[str, list[str]] = {"ew_lsoa21": [], "scotland_oa22": []}
    for grain, system, key in (
        ("ew_lsoa21", EW, "lsoa_code"),
        ("scotland_oa22", SCOT, "oa_code"),
    ):
        for row in frames[system]:
            code = str(row[key])
            brma = TOY_AREA_BRMAS.get(code, "MAIDSTONE")
            region = regions[str(row["region_code"])]
            layout[grain].setdefault(region, {}).setdefault(brma, []).append(code)
            rosters[grain].append(code)
    return {
        "schema_version": bal.SCHEMA_VERSION,
        "kind": bal.KIND,
        "method": bal.METHOD_VERSION,
        "grains": {
            grain: {
                "area_system": bal.GRAINS[grain].system,
                "key": bal.GRAINS[grain].key,
                "areas": len(codes),
                "roster_sha256": bal.roster_sha256(codes),
            }
            for grain, codes in rosters.items()
        },
        "scotland_brma_codes": {"LOTHIAN": "S33000011"},
        "brmas": layout,
    }


def write_toy_brma_area_layer(directory: Path) -> dict[str, Path]:
    """Write the toy layer as the graph source the area-BRMA node reads."""
    from microcosm.build.uk_runtime.brma_area_layer import BRMA_AREA_LAYER_SOURCE

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "toy-brma-area-layer.json"
    path.write_text(json.dumps(toy_brma_area_layer()), encoding="utf-8")
    return {BRMA_AREA_LAYER_SOURCE: path}


#: Toy weekly rent levels by BRMA; each category scales its BRMA's level.
TOY_BRMA_RENT_LEVELS = {
    "CENTRAL_LONDON": 400.0,
    "INNER_EAST_LONDON": 250.0,
    "CARDIFF": 180.0,
    "LOTHIAN": 170.0,
    "MAIDSTONE": 200.0,
}
_TOY_CATEGORY_FACTORS = {"B": 0.8, "C": 1.0, "D": 1.2, "E": 1.5}


def toy_private_rent_resources() -> dict[str, dict]:
    """The microcosm#1262 rent quantiles and census counts over the toy BRMAs.

    Each toy BRMA's list runs linearly from half to one and a half times its
    level, so a rank ``u`` reads ``level * (0.5 + u)`` inside the knots; the
    counts put every toy region's private renters in the toy region's BRMAs.
    """
    from microcosm.build.uk_runtime import private_rent_localisation as prl

    grid = list(prl.GRID)
    brmas = {}
    for brma, level in TOY_BRMA_RENT_LEVELS.items():
        cells = {"nation": "toy"}
        for category, factor in _TOY_CATEGORY_FACTORS.items():
            knots = [level * factor * (0.5 + p) for p in grid]
            cells[category] = {
                "lets": 100,
                "basis": "own_lets",
                "quantiles": knots,
                "mean": prl.knot_mean(grid, knots),
            }
        brmas[brma] = cells
    quantiles = {
        "schema_version": prl.QUANTILES_SCHEMA_VERSION,
        "kind": prl.QUANTILES_KIND,
        "method": prl.QUANTILES_METHOD,
        "unit": "gbp_per_week",
        "grid": grid,
        "categories": list(prl.CATEGORIES),
        "brmas": brmas,
    }
    regions = {
        "LONDON": {"CENTRAL_LONDON": 50.0, "INNER_EAST_LONDON": 50.0},
        "WALES": {"CARDIFF": 100.0},
        "SCOTLAND": {"LOTHIAN": 100.0},
    }
    counts = {
        "cells": {
            region: {category: dict(cells) for category in ("A", *prl.CATEGORIES)}
            for region, cells in regions.items()
        }
    }
    return {"quantiles": quantiles, "counts": counts}


def write_toy_private_rent_resources(directory: Path) -> dict[str, Path]:
    """Write the toy rent quantiles and counts as the rent node's graph sources."""
    from microcosm.build.uk_runtime.private_rent_localisation import (
        COUNTS_SOURCE,
        QUANTILES_SOURCE,
    )

    directory.mkdir(parents=True, exist_ok=True)
    resources = toy_private_rent_resources()
    paths = {
        QUANTILES_SOURCE: directory / "toy-brma-rent-quantiles.json",
        COUNTS_SOURCE: directory / "toy-brma-private-renters.json",
    }
    paths[QUANTILES_SOURCE].write_text(
        json.dumps(resources["quantiles"]), encoding="utf-8"
    )
    paths[COUNTS_SOURCE].write_text(json.dumps(resources["counts"]), encoding="utf-8")
    return paths


__all__ = [name for name in globals() if not name.startswith("__")]
