"""Parsers and assembly for the primary Census sources behind the US PUMA ladder.

The PUMA ladder anchors every household at a 2020 Public Use Microdata Area —
the finest geography the ACS PUMS publishes — and derives every coarser layer
(congressional district, county, tract) from that anchor by a
population-weighted draw over the PUMA's overlap with each layer. One national
dataset, filterable at any grain (microcosm #275; no per-area files).

Because 2020 tabulation blocks nest in 2020 census tracts, and 2020 tracts nest
in 2020 PUMAs, every overlap distribution the ladder needs comes from a single
block pass joined to one small tract-to-PUMA relationship file:

- ``tract = block_geoid // 10**4`` and ``county = block_geoid // 10**10`` are
  structural prefixes of the 15-digit block geoid.
- ``tract -> PUMA`` comes from the Census 2020 Census Tract to 2020 PUMA
  relationship file (each 2020 tract belongs to exactly one 2020 PUMA).
- ``block -> 119th CD`` and ``block -> POP100`` reuse the block-ladder parsers
  :func:`parse_national_cd_bef` and :func:`parse_pl_geo_blocks` unchanged.

Summing block populations by ``(PUMA, CD)``, ``(PUMA, county)`` and
``(PUMA, tract)`` yields the three overlap tables; summing by PUMA yields the
anchor population used for the ASEC state -> PUMA draw. Every populated block
contributes to exactly one PUMA, one CD, one county and one tract, so each
overlap table conserves its PUMA's population exactly — a defect (a populated
block the CD file does not cover, or a tract the relationship file omits) fails
loudly here rather than shipping silent partial weights.

Download orchestration lives in ``tools/build_us_puma_ladder_artifact.py``;
everything here is pure and unit-testable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

#: PUMA geoids are ``state_fips * 10**5 + PUMA5CE`` (a 2-digit state prefix and
#: the 5-digit 2020 PUMA code), matching how block/tract geoids drop leading
#: zeros to an integer. Tract geoids are the 11-digit ``state+county+tract``
#: integer; county fips the 5-digit ``state+county`` integer.
PUMA_GEOID_STATE_DIVISOR = 10**5
TRACT_FROM_BLOCK_DIVISOR = 10**4
COUNTY_FROM_BLOCK_DIVISOR = 10**10
COUNTY_FROM_TRACT_DIVISOR = 10**6

_TRACT_TO_PUMA_HEADER = ("STATEFP", "COUNTYFP", "TRACTCE", "PUMA5CE")


def parse_tract_to_puma_relationship(
    lines: Iterable[str],
    *,
    allowed_state_fips: frozenset[str] | None = None,
) -> dict[int, int]:
    """Parse the 2020 tract-to-PUMA relationship file into tract geoid -> PUMA.

    The file is comma-delimited with header ``STATEFP,COUNTYFP,TRACTCE,PUMA5CE``
    (a UTF-8 BOM may prefix the first field). Rows for states outside
    ``allowed_state_fips`` (Puerto Rico and the island territories, when the
    caller passes the 50-states-plus-DC set) are skipped so the returned map
    matches the block spine. Every populated tract maps to exactly one PUMA;
    a tract that appears twice with conflicting PUMAs is a source defect.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source="tract-to-PUMA relationship")
    cleaned = header.lstrip("﻿")
    if tuple(part.strip().upper() for part in cleaned.split(",")) != (
        _TRACT_TO_PUMA_HEADER
    ):
        raise ValueError(
            "tract-to-PUMA relationship header must be "
            f"'STATEFP,COUNTYFP,TRACTCE,PUMA5CE', got {header!r}."
        )
    result: dict[int, int] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        if len(parts) != 4:
            raise ValueError(
                f"tract-to-PUMA line {line_number} must have four fields, "
                f"got {stripped!r}."
            )
        state_fips, county_fips, tract_ce, puma_ce = (part.strip() for part in parts)
        _require_digits(state_fips, width=2, source=f"tract-to-PUMA line {line_number}")
        if allowed_state_fips is not None and state_fips not in allowed_state_fips:
            continue
        _require_digits(
            county_fips, width=3, source=f"tract-to-PUMA line {line_number}"
        )
        _require_digits(tract_ce, width=6, source=f"tract-to-PUMA line {line_number}")
        _require_digits(puma_ce, width=5, source=f"tract-to-PUMA line {line_number}")
        tract = int(f"{state_fips}{county_fips}{tract_ce}")
        puma = int(f"{state_fips}{puma_ce}")
        existing = result.get(tract)
        if existing is not None and existing != puma:
            raise ValueError(
                f"tract-to-PUMA relationship assigns tract "
                f"{state_fips}{county_fips}{tract_ce} to both PUMA "
                f"{existing} and {puma}."
            )
        result[tract] = puma
    if not result:
        raise ValueError("tract-to-PUMA relationship contained no rows.")
    return result


def assemble_us_puma_ladder(
    *,
    block_population: Mapping[int, int],
    cd_by_block: Mapping[int, int],
    tract_to_puma: Mapping[int, int],
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Aggregate the block pass into the PUMA ladder artifact's NPZ payload.

    For every populated block the block's tract (a structural prefix) must map
    to a PUMA and the block must carry a congressional district; either gap is
    a source defect, not a skippable row, so the three overlap tables each
    conserve their PUMA's population exactly. Returns arrays for the anchor
    (``puma`` / ``puma_population``) and the three overlap tables, each sorted
    by ``(puma, layer_value)`` for a stable, searchsorted-friendly artifact.
    """

    puma_population: dict[int, int] = {}
    puma_cd_population: dict[tuple[int, int], int] = {}
    puma_county_population: dict[tuple[int, int], int] = {}
    puma_tract_population: dict[tuple[int, int], int] = {}
    missing_puma: list[int] = []
    missing_cd: list[int] = []

    for block, population in block_population.items():
        if population <= 0:
            continue
        tract = block // TRACT_FROM_BLOCK_DIVISOR
        puma = tract_to_puma.get(tract)
        if puma is None:
            missing_puma.append(block)
            continue
        cd = cd_by_block.get(block)
        if cd is None:
            missing_cd.append(block)
            continue
        county = block // COUNTY_FROM_BLOCK_DIVISOR
        puma_population[puma] = puma_population.get(puma, 0) + population
        cd_key = (puma, cd)
        puma_cd_population[cd_key] = puma_cd_population.get(cd_key, 0) + population
        county_key = (puma, county)
        puma_county_population[county_key] = (
            puma_county_population.get(county_key, 0) + population
        )
        tract_key = (puma, tract)
        puma_tract_population[tract_key] = (
            puma_tract_population.get(tract_key, 0) + population
        )

    if missing_puma:
        examples = [f"{block:015d}" for block in sorted(missing_puma)[:5]]
        raise ValueError(
            f"{len(missing_puma)} populated block(s) have a tract absent from "
            f"the tract-to-PUMA relationship file; examples: {examples}."
        )
    if missing_cd:
        examples = [f"{block:015d}" for block in sorted(missing_cd)[:5]]
        raise ValueError(
            f"{len(missing_cd)} populated block(s) have no congressional "
            f"district in the CD BEF; examples: {examples}."
        )
    if not puma_population:
        raise ValueError("PUMA ladder assembly produced no populated PUMAs.")

    pumas = np.asarray(sorted(puma_population), dtype=np.int64)
    anchor_population = np.asarray(
        [puma_population[puma] for puma in pumas.tolist()], dtype=np.int64
    )
    cd_puma, cd_value, cd_pop = _overlap_arrays(puma_cd_population)
    county_puma, county_value, county_pop = _overlap_arrays(puma_county_population)
    tract_puma, tract_value, tract_pop = _overlap_arrays(puma_tract_population)

    _assert_conserves(
        pumas, anchor_population, cd_puma, cd_pop, layer="congressional_district"
    )
    _assert_conserves(pumas, anchor_population, county_puma, county_pop, layer="county")
    _assert_conserves(pumas, anchor_population, tract_puma, tract_pop, layer="tract")

    return {
        "puma": pumas,
        "puma_population": anchor_population,
        "cd_overlap_puma": cd_puma,
        "cd_overlap_cd": cd_value,
        "cd_overlap_population": cd_pop,
        "county_overlap_puma": county_puma,
        "county_overlap_county": county_value.astype(np.int32),
        "county_overlap_population": county_pop,
        "tract_overlap_puma": tract_puma,
        "tract_overlap_tract": tract_value,
        "tract_overlap_population": tract_pop,
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }


def _overlap_arrays(
    overlap: Mapping[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(puma, value, population)`` arrays sorted by ``(puma, value)``."""

    keys = sorted(overlap)
    puma = np.asarray([key[0] for key in keys], dtype=np.int64)
    value = np.asarray([key[1] for key in keys], dtype=np.int64)
    population = np.asarray([overlap[key] for key in keys], dtype=np.int64)
    return puma, value, population


def _assert_conserves(
    pumas: np.ndarray,
    anchor_population: np.ndarray,
    overlap_puma: np.ndarray,
    overlap_population: np.ndarray,
    *,
    layer: str,
) -> None:
    """Every PUMA's overlap population must sum to its anchor population."""

    totals = {int(puma): 0 for puma in pumas.tolist()}
    for puma, population in zip(
        overlap_puma.tolist(), overlap_population.tolist(), strict=True
    ):
        if puma not in totals:
            raise ValueError(
                f"{layer} overlap references PUMA {puma} absent from the anchor."
            )
        totals[puma] += population
    for puma, anchor in zip(pumas.tolist(), anchor_population.tolist(), strict=True):
        if totals[puma] != anchor:
            raise ValueError(
                f"{layer} overlap for PUMA {puma} sums to {totals[puma]:,} but "
                f"the anchor population is {anchor:,}."
            )


def _required_header(iterator: Iterable[str], *, source: str) -> str:
    for line in iterator:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError(f"{source} is empty.")


def _require_digits(value: str, *, width: int, source: str) -> None:
    if not (value.isdigit() and len(value) == width):
        raise ValueError(f"{source}: expected a {width}-digit code, got {value!r}.")


__all__ = [
    "COUNTY_FROM_BLOCK_DIVISOR",
    "COUNTY_FROM_TRACT_DIVISOR",
    "PUMA_GEOID_STATE_DIVISOR",
    "TRACT_FROM_BLOCK_DIVISOR",
    "assemble_us_puma_ladder",
    "parse_tract_to_puma_relationship",
]
