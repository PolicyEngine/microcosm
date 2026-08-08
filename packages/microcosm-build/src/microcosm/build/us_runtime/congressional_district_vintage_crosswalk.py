"""Build the 117th->119th congressional-district vintage crosswalk from Census.

Microcosm calibrates congressional-district (CD) fiscal targets from the IRS SOI
CD table, whose district geography is the **117th Congress** (the pre-2020-
apportionment plan, drawn on 2010 geography). The current Microcosm CD surface
that ``policyengine.py`` consumes is the **119th Congress** (the post-2020-
apportionment plan). ``congressional_district_vintage`` translates SOI CD facts
from the old vintage onto the current one through a proportional crosswalk; this
module *builds that crosswalk* from primary Census sources, so the weights are
reproducible and cited rather than an opaque artifact.

Method — a single-vintage block overlay, so no 2010<->2020 block bridge is
needed:

- **Old (117th) district of each 2020 block**: the 2020 Block Assignment File
  ``CD`` layer (``BlockAssign_ST{fips}_{usps}_CD.txt``, ``BLOCKID|DISTRICT``).
  The 2020 BAFs were published with the 2020 P.L. 94-171 release and carry the
  116th-Congress plan (identical district geography to the 117th) expressed on
  **2020** tabulation blocks.
- **Current (119th) district of each 2020 block**: the 119th Congressional
  District Block Equivalency File (``NationalCD119.txt``, ``GEOID,CDFP``) — the
  same source ``build_us_block_ladder_artifact`` already uses for the current
  CD surface.
- **Weight**: 2020 P.L. 94-171 ``POP100`` per block (the block ladder's
  :func:`~microcosm.build.us_runtime.block_ladder_sources.parse_pl_geo_blocks`
  convention), because congressional apportionment and equal-population
  redistricting are population operations. Population is the correct default
  basis; ACS income/tax proxy weights for fiscal targets are a later refinement
  (PolicyEngine/microcosm#205).

Because both district assignments are read on the *same* 2020 blocks and weighted
by the *same* 2020 block populations, each old district's population is
redistributed across current districts — it is never invented. Per-state
residuals (populated blocks the BEF/BAF fail to cover) are reported, not hidden.

Geoid conventions match ``congressional_district_vintage`` and the block ladder:
old geoids use the ``5001700US`` prefix, current geoids ``5001900US``; the last
four characters are ``state_fips + district`` (``SSDD``); at-large states and the
DC non-voting delegate normalize to district ``00`` (the repo-wide convention;
see ``block_ladder_sources``).

This module is pure and unit-testable. Download orchestration lives in
``tools/build_us_congressional_district_vintage_crosswalk.py``; the derived
crosswalk is a regenerable build artifact, not a Ledger fact (the fact-vs-
computed boundary of PolicyEngine/ledger#71).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from microcosm.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
)

#: Census "assigned to no district" markers (large water bodies).
_CD_UNASSIGNED = frozenset({"ZZ", "ZZZ", ""})
#: Non-voting delegate districts (DC) — the district-00 rung of the Microcosm
#: at-large convention ``state_fips * 100 + 00`` (matches block_ladder_sources).
_CD_DELEGATE = "98"

CROSSWALK_BASIS_BLOCK_POPULATION = "block_population"


def normalize_district_code(district_raw: str) -> str | None:
    """Normalize a 2-character Census CD code to ``DD``; ``None`` if unassigned.

    Delegate districts (``98``) and at-large states normalize to ``00``, matching
    the repo-wide at-large/delegate convention. Water-only ``ZZ``/``ZZZ`` blocks
    and blanks are unassigned.
    """

    district = district_raw.strip()
    if district in _CD_UNASSIGNED:
        return None
    if district == _CD_DELEGATE:
        return "00"
    if district.isdigit() and len(district) <= 2:
        return f"{int(district):02d}"
    raise ValueError(
        f"congressional-district code {district_raw!r}: expected a two-digit "
        "code, '98', 'ZZ'/'ZZZ', or blank."
    )


def parse_baf_cd_layer(lines: Iterable[str], *, label: str) -> dict[int, str]:
    """Parse a 2020 BAF ``CD`` layer into 2020 block geoid -> district ``DD``.

    Header ``BLOCKID|DISTRICT``; unassigned (water) blocks are dropped, delegate
    ``98`` and at-large normalize to ``00``.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source=label)
    if [part.strip().upper() for part in header.split("|")] != ["BLOCKID", "DISTRICT"]:
        raise ValueError(f"{label} header must be 'BLOCKID|DISTRICT', got {header!r}.")
    result: dict[int, str] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        parts = stripped.split("|")
        if len(parts) != 2:
            raise ValueError(
                f"{label} line {line_number} must have two fields, got {stripped!r}."
            )
        district = normalize_district_code(parts[1])
        if district is None:
            continue
        block = _block_geoid(parts[0].strip(), source=f"{label} line {line_number}")
        if block in result:
            raise ValueError(
                f"{label} assigns block {parts[0].strip()} more than once."
            )
        result[block] = district
    return result


def parse_national_cd_bef_districts(lines: Iterable[str]) -> dict[int, str]:
    """Parse a national CD BEF (``GEOID,CDFP``) into block geoid -> ``DD``.

    Same normalization as :func:`parse_baf_cd_layer` (delegate/at-large ``00``,
    water dropped). Unlike ``block_ladder_sources.parse_national_cd_bef`` this
    keeps the district code as a two-character string rather than folding in the
    state FIPS, so the crosswalk builder can key by state itself.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source="national CD BEF")
    if [part.strip().upper() for part in header.split(",")] != ["GEOID", "CDFP"]:
        raise ValueError(
            f"National CD BEF header must be 'GEOID,CDFP', got {header!r}."
        )
    result: dict[int, str] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"National CD BEF line {line_number} must have two fields, "
                f"got {stripped!r}."
            )
        district = normalize_district_code(parts[1])
        if district is None:
            continue
        block = _block_geoid(
            parts[0].strip(), source=f"national CD BEF line {line_number}"
        )
        if block in result:
            raise ValueError(
                f"National CD BEF assigns block {parts[0].strip()} more than once."
            )
        result[block] = district
    if not result:
        raise ValueError("National CD BEF contained no district assignments.")
    return result


def build_cd_vintage_crosswalk_rows(
    *,
    old_cd_by_block: Mapping[int, str],
    current_cd_by_block: Mapping[int, str],
    block_population: Mapping[int, int],
    source_prefix: str = SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
    target_prefix: str = CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join old + current CD block assignments by 2020 block population.

    Returns ``(rows, diagnostics)`` where each row is
    ``{"source_geography_id", "target_geography_id", "pair_population",
    "weight"}``: ``pair_population`` is the summed 2020 population of the
    blocks shared by that (old, current) district pair, ``weight`` is that
    mass normalized by the old district's total assigned population — so
    weights sum to 1.0 per source district — and ``diagnostics`` reports
    per-state population conservation.

    A block contributes only when it has an old district, a current district,
    and a positive population. Blocks the assignments do not cover (or that fall
    in a different state on each side — a source defect) are counted in the
    diagnostics as unmatched, never silently mapped.
    """

    pair_population: dict[tuple[str, str], int] = {}
    state_assigned: dict[str, int] = {}
    state_population: dict[str, int] = {}
    unmatched_no_old = 0
    unmatched_no_current = 0
    cross_state = 0

    for block, population in block_population.items():
        if population <= 0:
            continue
        state_fips = f"{block:015d}"[:2]
        state_population[state_fips] = state_population.get(state_fips, 0) + population
        old_district = old_cd_by_block.get(block)
        current_district = current_cd_by_block.get(block)
        if old_district is None:
            unmatched_no_old += population
            continue
        if current_district is None:
            unmatched_no_current += population
            continue
        source_geoid = f"{state_fips}{old_district}"
        target_geoid = f"{state_fips}{current_district}"
        if source_geoid[:2] != target_geoid[:2]:  # defensive; blocks are single-state
            cross_state += population
            continue
        key = (source_geoid, target_geoid)
        pair_population[key] = pair_population.get(key, 0) + population
        state_assigned[state_fips] = state_assigned.get(state_fips, 0) + population

    source_totals: dict[str, int] = {}
    for (source_geoid, _), population in pair_population.items():
        source_totals[source_geoid] = source_totals.get(source_geoid, 0) + population
    rows = [
        {
            "source_geography_id": f"{source_prefix}{source_geoid}",
            "target_geography_id": f"{target_prefix}{target_geoid}",
            "pair_population": population,
            "weight": population / source_totals[source_geoid],
        }
        for (source_geoid, target_geoid), population in sorted(pair_population.items())
    ]

    state_conservation = {
        state_fips: {
            "state_population": state_population.get(state_fips, 0),
            "assigned_population": state_assigned.get(state_fips, 0),
            "unmatched_population": (
                state_population.get(state_fips, 0) - state_assigned.get(state_fips, 0)
            ),
        }
        for state_fips in sorted(state_population)
    }
    diagnostics = {
        "basis": CROSSWALK_BASIS_BLOCK_POPULATION,
        "row_count": len(rows),
        "source_district_count": len({row["source_geography_id"] for row in rows}),
        "target_district_count": len({row["target_geography_id"] for row in rows}),
        "unmatched_population_no_old_district": unmatched_no_old,
        "unmatched_population_no_current_district": unmatched_no_current,
        "cross_state_population": cross_state,
        "state_conservation": state_conservation,
    }
    return rows, diagnostics


def _required_header(iterator: Iterable[str], *, source: str) -> str:
    for line in iterator:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError(f"{source} is empty.")


def _block_geoid(value: str, *, source: str) -> int:
    if not (value.isdigit() and len(value) == 15):
        raise ValueError(f"{source}: block geoid must be 15 digits, got {value!r}.")
    return int(value)


__all__ = [
    "CROSSWALK_BASIS_BLOCK_POPULATION",
    "build_cd_vintage_crosswalk_rows",
    "normalize_district_code",
    "parse_baf_cd_layer",
    "parse_national_cd_bef_districts",
]
