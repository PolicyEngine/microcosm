"""US block -> congressional-district plan registry.

A household's location is one 2020 census tabulation block, and every other
geography is a lookup from that block. Congressional districts are the one
layer that has several live versions on the same 2020 blocks, so this module
keeps them in one versioned artifact: for every populated 2020 block, the
district the block belongs to under each registered plan.

- ``117th_congress``: the plan the IRS SOI congressional-district tables use
  (see ``congressional_district_vintage``).
- ``118th_congress``: the first post-2020-apportionment plan.
- ``119th_congress``: the current plan and the block ladder's primary one.
- ``120th_congress``: the plans for the 2026 elections (120th Congress),
  including the 2025-26 mid-decade redraws.

Each plan is one ``int16`` array aligned to the registry's sorted
``block_geoid``, holding district geoids ``state_fips * 100 + district``
(at-large states and the DC delegate are district ``00``, the repo-wide
convention). A new plan is one more array plus one metadata entry.

The artifact is an NPZ built by ``tools/build_us_cd_plan_registry_artifact.py``
from primary Census sources. Its embedded metadata records each plan's source
files (URL + SHA-256), the apportionment it must match, and any known
deviations. The build is byte-reproducible, and its SHA-256 is pinned in the
packaged ``us_cd_plan_registry.provenance.json``.

Structural invariants, checked by :func:`assemble_us_cd_plan_registry` when
the artifact is built and again by :func:`load_us_cd_plan_registry` on every
load:

- ``block_geoid``, ``population`` and every plan array are one-dimensional
  and aligned; the blocks are 15-digit geoids, sorted and unique, each with
  positive population.
- Every block has exactly one district under every plan. A populated block a
  plan's source leaves unassigned is a build error, never a gap.
- Every district lies in its block's state.
- For every state present, the districts under a plan are exactly its House
  apportionment for that plan's census: ``00`` for an at-large state or the
  DC delegate, otherwise ``01`` through ``n``. So every apportioned district
  contains at least one populated block.

Summing block population by any plan's district therefore reproduces every
state's population exactly: the plans are partitions of the same blocks.

These checks are structural: a registry for a subset of states passes them.
That the published artifact covers exactly the national universe (every 2020
P.L. 94-171 block with positive population in the 50 states and DC) is
established by the build, by its exact agreement with the block ladder, and
by pinning the artifact's SHA-256 (:func:`load_pinned_us_cd_plan_registry`).

This module is pure apart from the NPZ read. Nothing here assigns households;
the block draw derives every plan's district from the chosen block.
"""

from __future__ import annotations

import hashlib
import json
import re
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)
from microcosm.build.us_runtime.congressional_district_vintage_crosswalk import (
    normalize_district_code,
)

US_CD_PLAN_REGISTRY_KIND = "us_cd_plan_registry"
US_CD_PLAN_REGISTRY_SCHEMA_VERSION = 1
US_CD_PLAN_REGISTRY_BLOCK_VINTAGE = "2020_tabulation_blocks"
#: NPZ key prefix of each plan's district array (``cd__119th_congress``).
CD_PLAN_ARRAY_PREFIX = "cd__"
#: The plan the block ladder's own ``congressional_district_geoid`` carries.
PRIMARY_CD_PLAN = CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
#: Keys every plan's metadata must carry (``block_location`` requires
#: ``vintage`` and ``source``; the rest are this registry's provenance).
REQUIRED_PLAN_SOURCE_KEYS = (
    "vintage",
    "source",
    "url",
    "apportionment",
    "source_files",
)

US_CD_PLAN_REGISTRY_PROVENANCE_RESOURCE = "us_cd_plan_registry.provenance.json"

#: Census "Apportionment Data" historical table, the source of
#: :data:`US_HOUSE_APPORTIONMENT`. The registry build re-derives both columns
#: from this file and refuses to run if they disagree with the constant.
US_HOUSE_APPORTIONMENT_SOURCE = {
    "url": (
        "https://www2.census.gov/programs-surveys/decennial/2020/data/"
        "apportionment/apportionment.csv"
    ),
    "sha256": "d9be9c833ad936ee309a83fa6428a3ae2f82d5d8e5b2c69c4d6ff1e79ba57fcd",
}

#: Voting House seats by state FIPS for each decennial apportionment
#: (``Number of Representatives`` in the Census table; DC has none).
# fmt: off
US_HOUSE_APPORTIONMENT: Mapping[str, Mapping[str, int]] = {
    "2010_census": {
        "01": 7, "02": 1, "04": 9, "05": 4, "06": 53, "08": 7, "09": 5,
        "10": 1, "11": 0, "12": 27, "13": 14, "15": 2, "16": 2, "17": 18,
        "18": 9, "19": 4, "20": 4, "21": 6, "22": 6, "23": 2, "24": 8,
        "25": 9, "26": 14, "27": 8, "28": 4, "29": 8, "30": 1, "31": 3,
        "32": 4, "33": 2, "34": 12, "35": 3, "36": 27, "37": 13, "38": 1,
        "39": 16, "40": 5, "41": 5, "42": 18, "44": 2, "45": 7, "46": 1,
        "47": 9, "48": 36, "49": 4, "50": 1, "51": 11, "53": 10, "54": 3,
        "55": 8, "56": 1,
    },
    "2020_census": {
        "01": 7, "02": 1, "04": 9, "05": 4, "06": 52, "08": 8, "09": 5,
        "10": 1, "11": 0, "12": 28, "13": 14, "15": 2, "16": 2, "17": 17,
        "18": 9, "19": 4, "20": 4, "21": 6, "22": 6, "23": 2, "24": 8,
        "25": 9, "26": 13, "27": 8, "28": 4, "29": 8, "30": 2, "31": 3,
        "32": 4, "33": 2, "34": 12, "35": 3, "36": 26, "37": 14, "38": 1,
        "39": 15, "40": 5, "41": 6, "42": 17, "44": 2, "45": 7, "46": 1,
        "47": 9, "48": 38, "49": 4, "50": 1, "51": 11, "53": 10, "54": 2,
        "55": 8, "56": 1,
    },
}
# fmt: on

_PLAN_ID = re.compile(r"^[0-9a-z][0-9a-z_]*$")
_BLOCK_COLUMNS = ("GEOID", "BLOCKID", "BLOCK", "BLOCK_KEY")
_DISTRICT_COLUMNS = ("CDFP", "DISTRICT")
_BLOCK_STATE_DIVISOR = 10**13
_MIN_BLOCK_GEOID = 10**13  # state FIPS 01
_MAX_BLOCK_GEOID = 10**15


def expected_district_counts(apportionment: str) -> dict[str, int]:
    """District count per state FIPS under ``apportionment``.

    A state with one seat has one at-large district; DC, with no voting seat,
    has one delegate district. Both carry district code ``00``.
    """

    try:
        seats = US_HOUSE_APPORTIONMENT[apportionment]
    except KeyError:
        raise ValueError(
            f"Unknown apportionment {apportionment!r}; registered: "
            f"{sorted(US_HOUSE_APPORTIONMENT)}."
        ) from None
    return {state: max(count, 1) for state, count in seats.items()}


@dataclass(frozen=True)
class CdBlockAssignment:
    """One plan's block -> district assignment, as sorted parallel arrays.

    ``block_geoid`` is sorted and unique (``int64``); ``district_geoid`` holds
    ``state_fips * 100 + district`` aligned to it. Arrays rather than a dict
    because a national plan has about 8.2 million blocks.
    """

    block_geoid: np.ndarray
    district_geoid: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.block_geoid.ndim != 1
            or self.block_geoid.shape != self.district_geoid.shape
        ):
            raise ValueError(
                "block_geoid and district_geoid must be aligned 1-D arrays."
            )
        if len(self.block_geoid) and not bool(np.all(np.diff(self.block_geoid) > 0)):
            raise ValueError("CdBlockAssignment block_geoid must be sorted and unique.")

    def __len__(self) -> int:
        return len(self.block_geoid)

    def states(self) -> set[str]:
        return {
            f"{state:02d}"
            for state in np.unique(self.block_geoid // _BLOCK_STATE_DIVISOR).tolist()
        }

    def for_state(self, state_fips: str) -> CdBlockAssignment:
        mask = self.block_geoid // _BLOCK_STATE_DIVISOR == int(state_fips)
        return CdBlockAssignment(self.block_geoid[mask], self.district_geoid[mask])


def cd_block_assignment(
    pairs: Mapping[int, int] | Iterable[tuple[int, int]], *, label: str
) -> CdBlockAssignment:
    """Build an assignment from ``{block: district}`` pairs; duplicates refused."""

    items = list(pairs.items()) if isinstance(pairs, Mapping) else list(pairs)
    blocks = np.fromiter(
        (block for block, _ in items), dtype=np.int64, count=len(items)
    )
    districts = np.fromiter(
        (district for _, district in items), dtype=np.int64, count=len(items)
    )
    return _sorted_assignment(blocks, districts, label=label)


def parse_cd_block_assignment(lines: Iterable[str], *, label: str) -> CdBlockAssignment:
    """Parse a Census CD block file into a block -> district geoid assignment.

    Reads every Census block-to-district format the registry uses, choosing
    columns by header name:

    - 2020 Block Assignment File ``CD`` layer: ``BLOCKID|DISTRICT``;
    - 118th/119th BEF: ``GEOID,CDFP`` (the 118th state files write
      ``GEOID, CDFP``);
    - 120th BEF: ``GEOID,STATEFP,COUNTYFP,TRACTCE,BLOCKCE,CDFP``, where
      ``STATEFP`` is cross-checked against the geoid;
    - a state legislature's plan block file, quoted CSV such as
      ``"Block","District"`` or ``"Block_Key","District"`` (these are often on
      2010 blocks; the parser does not care which vintage the geoids are).

    ``ZZ``/``ZZZ``/blank (no district, usually water) rows are dropped; the
    delegate code ``98`` and at-large ``00`` both become district ``00``.
    A block listed twice is an error. ``lines`` is consumed as a stream.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source=label)
    delimiter = "|" if "|" in header else ","
    columns = [part.strip().strip('"').upper() for part in header.split(delimiter)]
    block_index = _column_index(columns, _BLOCK_COLUMNS, label=label)
    district_index = _column_index(columns, _DISTRICT_COLUMNS, label=label)
    state_index = columns.index("STATEFP") if "STATEFP" in columns else None
    width = len(columns)

    blocks = array("q")
    districts = array("q")
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip().strip('"') for part in stripped.split(delimiter)]
        if len(parts) != width:
            raise ValueError(
                f"{label} line {line_number} has {len(parts)} fields, "
                f"expected {width}: {stripped!r}."
            )
        district = normalize_district_code(parts[district_index])
        if district is None:
            continue
        geoid = parts[block_index]
        block = _block_geoid(geoid, source=f"{label} line {line_number}")
        if state_index is not None and parts[state_index] != geoid[:2]:
            raise ValueError(
                f"{label} line {line_number}: STATEFP "
                f"{parts[state_index]!r} disagrees with GEOID {geoid}."
            )
        blocks.append(block)
        districts.append((block // _BLOCK_STATE_DIVISOR) * 100 + int(district))
    if not blocks:
        raise ValueError(f"{label} contained no district assignments.")
    return _sorted_assignment(
        np.frombuffer(blocks, dtype=np.int64),
        np.frombuffer(districts, dtype=np.int64),
        label=label,
    )


def concat_cd_block_assignments(
    parts: Iterable[CdBlockAssignment], *, label: str
) -> CdBlockAssignment:
    """Union per-state assignments into one; a block in two parts is an error."""

    parts = list(parts)
    if not parts:
        raise ValueError(f"{label}: no assignments to combine.")
    return _sorted_assignment(
        np.concatenate([part.block_geoid for part in parts]),
        np.concatenate([part.district_geoid for part in parts]),
        label=label,
    )


def replace_state_assignments(
    base: CdBlockAssignment,
    replacement: CdBlockAssignment,
    *,
    state_fips: str,
) -> CdBlockAssignment:
    """Return ``base`` with every block of ``state_fips`` taken from ``replacement``.

    Used when one state's plan comes from a different source than the rest of
    the plan. ``replacement`` must hold only that state's blocks and cover
    every block ``base`` assigns in that state.
    """

    state = int(state_fips)
    if (replacement.block_geoid // _BLOCK_STATE_DIVISOR != state).any():
        raise ValueError(
            f"Replacement for state {state_fips} contains blocks from other states."
        )
    base_in_state = base.block_geoid // _BLOCK_STATE_DIVISOR == state
    uncovered = np.setdiff1d(base.block_geoid[base_in_state], replacement.block_geoid)
    if len(uncovered):
        raise ValueError(
            f"Replacement for state {state_fips} leaves {len(uncovered)} block(s) "
            f"unassigned, e.g. {int(uncovered[0]):015d}."
        )
    return concat_cd_block_assignments(
        [
            CdBlockAssignment(
                base.block_geoid[~base_in_state], base.district_geoid[~base_in_state]
            ),
            replacement,
        ],
        label=f"state {state_fips} replacement",
    )


@dataclass(frozen=True)
class BlockRelationship:
    """Census 2010 -> 2020 tabulation-block intersections (``rel2020/t10t20``).

    One row per intersecting (2010 block, 2020 block) pair: both 15-digit
    geoids and the land and water area of the intersection in square meters.
    """

    block_2010: np.ndarray
    block_2020: np.ndarray
    area_land: np.ndarray
    area_water: np.ndarray


_RELATIONSHIP_COLUMNS = (
    "STATE_2010",
    "COUNTY_2010",
    "TRACT_2010",
    "BLK_2010",
    "STATE_2020",
    "COUNTY_2020",
    "TRACT_2020",
    "BLK_2020",
    "AREALAND_INT",
    "AREAWATER_INT",
)


def parse_block_relationship_2010_2020(
    lines: Iterable[str], *, label: str
) -> BlockRelationship:
    """Parse a Census 2010-to-2020 tabulation block relationship file.

    The state files (``TAB2010_TAB2020_ST{fips}.zip``) are pipe-delimited with
    a BOM on the header; the block geoid on each side is its state, county,
    tract and block columns concatenated.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source=label)
    columns = [part.strip().upper() for part in header.split("|")]
    missing = [name for name in _RELATIONSHIP_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{label} header is missing columns {missing}.")
    index = {name: columns.index(name) for name in _RELATIONSHIP_COLUMNS}
    width = len(columns)
    block_2010 = array("q")
    block_2020 = array("q")
    area_land = array("q")
    area_water = array("q")
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.rstrip("\r\n")
        if not stripped.strip():
            continue
        parts = stripped.split("|")
        if len(parts) != width:
            raise ValueError(
                f"{label} line {line_number} has {len(parts)} fields, expected {width}."
            )
        source = f"{label} line {line_number}"
        block_2010.append(
            _block_geoid(
                "".join(parts[index[name]] for name in _RELATIONSHIP_COLUMNS[:4]),
                source=source,
            )
        )
        block_2020.append(
            _block_geoid(
                "".join(parts[index[name]] for name in _RELATIONSHIP_COLUMNS[4:8]),
                source=source,
            )
        )
        area_land.append(int(parts[index["AREALAND_INT"]] or 0))
        area_water.append(int(parts[index["AREAWATER_INT"]] or 0))
    if not block_2020:
        raise ValueError(f"{label} contained no relationships.")
    return BlockRelationship(
        block_2010=np.frombuffer(block_2010, dtype=np.int64).copy(),
        block_2020=np.frombuffer(block_2020, dtype=np.int64).copy(),
        area_land=np.frombuffer(area_land, dtype=np.int64).copy(),
        area_water=np.frombuffer(area_water, dtype=np.int64).copy(),
    )


def crosswalk_plan_to_2020_blocks(
    plan_2010: CdBlockAssignment,
    relationship: BlockRelationship,
    *,
    state_fips: str,
) -> tuple[CdBlockAssignment, dict[str, np.ndarray]]:
    """Carry a plan drawn on 2010 blocks onto the 2020 blocks of ``state_fips``.

    Each 2020 block takes the district whose 2010 blocks cover the most of its
    land area; ties go to more water area, then the lower district code. A
    2020 block none of whose 2010 counterparts the plan assigns (for example
    one whose 2010 side lies in another state) is left out and reported.

    Returns the assignment and diagnostics: ``blocks_2020`` (every 2020 block
    of the state in the relationship file), ``straddling`` (2020 blocks whose
    2010 counterparts lie in more than one district) and ``unassigned``.
    """

    state = int(state_fips)
    rows = relationship.block_2020 // _BLOCK_STATE_DIVISOR == state
    block_2020 = relationship.block_2020[rows]
    if len(block_2020) == 0:
        raise ValueError(
            f"The relationship file has no 2020 blocks in state {state_fips}."
        )
    district, covered = _align(plan_2010, relationship.block_2010[rows])
    block = block_2020[covered]
    district = district[covered]
    land = relationship.area_land[rows][covered]
    water = relationship.area_water[rows][covered]

    all_2020 = np.unique(block_2020)
    if len(block) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return CdBlockAssignment(empty, empty.copy()), {
            "blocks_2020": all_2020,
            "straddling": empty.copy(),
            "unassigned": all_2020,
        }

    # Sum intersection area per (2020 block, district).
    order = np.lexsort((district, block))
    block, district = block[order], district[order]
    land, water = land[order], water[order]
    starts = np.flatnonzero(
        np.r_[True, (np.diff(block) != 0) | (np.diff(district) != 0)]
    )
    pair_block = block[starts]
    pair_district = district[starts]
    pair_land = np.add.reduceat(land, starts)
    pair_water = np.add.reduceat(water, starts)

    # Largest land, then water, then the lowest district code, per block.
    choice = np.lexsort((pair_district, -pair_water, -pair_land, pair_block))
    ranked_block = pair_block[choice]
    first = np.flatnonzero(np.r_[True, np.diff(ranked_block) != 0])
    assigned_block = ranked_block[first]
    assigned_district = pair_district[choice][first]
    candidates = np.diff(np.r_[first, len(ranked_block)])

    diagnostics = {
        "blocks_2020": all_2020,
        "straddling": assigned_block[candidates > 1],
        "unassigned": np.setdiff1d(all_2020, assigned_block),
    }
    return CdBlockAssignment(assigned_block, assigned_district), diagnostics


def fill_unassigned_blocks(
    assignment: CdBlockAssignment,
    unassigned: np.ndarray,
    *,
    block_geoid: np.ndarray,
    population: np.ndarray,
) -> tuple[CdBlockAssignment, list[dict[str, Any]]]:
    """Give each unassigned block its neighbourhood's majority district.

    The neighbourhood is the smallest of the block's block group, tract and
    county that contains assigned blocks; the district holding most of their
    population (``block_geoid``/``population``; blocks absent there count as
    zero) wins, then the one with more blocks, then the lower code. Returns
    the completed assignment and one record per filled block.
    """

    weights = dict(
        zip(
            np.asarray(block_geoid).tolist(),
            np.asarray(population).tolist(),
            strict=True,
        )
    )
    fills: list[dict[str, Any]] = []
    new_blocks: list[int] = []
    new_districts: list[int] = []
    for block in sorted(np.asarray(unassigned).tolist()):
        for rule, divisor in (
            ("block_group", 10**3),
            ("tract", 10**4),
            ("county", 10**10),
        ):
            prefix = block // divisor
            lo = int(np.searchsorted(assignment.block_geoid, prefix * divisor))
            hi = int(np.searchsorted(assignment.block_geoid, (prefix + 1) * divisor))
            if hi <= lo:
                continue
            tally: dict[int, list[int]] = {}
            for neighbour, district in zip(
                assignment.block_geoid[lo:hi].tolist(),
                assignment.district_geoid[lo:hi].tolist(),
                strict=True,
            ):
                entry = tally.setdefault(district, [0, 0])
                entry[0] += weights.get(neighbour, 0)
                entry[1] += 1
            district = min(tally, key=lambda d: (-tally[d][0], -tally[d][1], d))
            fills.append(
                {
                    "block": f"{block:015d}",
                    "district": f"{district % 100:02d}",
                    "rule": rule,
                    "population": int(weights.get(block, 0)),
                }
            )
            new_blocks.append(block)
            new_districts.append(district)
            break
        else:
            raise ValueError(
                f"No assigned block shares a county with block {block:015d}."
            )
    filled = concat_cd_block_assignments(
        [
            assignment,
            CdBlockAssignment(
                np.asarray(new_blocks, dtype=np.int64),
                np.asarray(new_districts, dtype=np.int64),
            ),
        ],
        label="filled assignment",
    )
    return filled, fills


def assignment_agreement(
    left: CdBlockAssignment,
    right: CdBlockAssignment,
    *,
    block_geoid: np.ndarray,
    population: np.ndarray,
) -> dict[str, Any]:
    """Share of the given blocks, and of their population, placed alike.

    Blocks either assignment leaves out count as disagreements.
    """

    blocks = np.asarray(block_geoid, dtype=np.int64)
    weights = np.asarray(population, dtype=np.int64)
    left_district, left_covered = _align(left, blocks)
    right_district, right_covered = _align(right, blocks)
    same = left_covered & right_covered & (left_district == right_district)
    total = int(weights.sum())
    return {
        "blocks": int(len(blocks)),
        "blocks_agreeing": int(same.sum()),
        "population": total,
        "population_agreeing": int(weights[same].sum()),
        "population_share_agreeing": (
            float(weights[same].sum() / total) if total else 1.0
        ),
    }


def assemble_us_cd_plan_registry(
    *,
    block_geoid: np.ndarray,
    population: np.ndarray,
    plan_assignments: Mapping[str, CdBlockAssignment],
    plan_sources: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Join per-plan block assignments into the registry's NPZ payload.

    ``block_geoid``/``population`` define the universe: unique blocks, every
    population positive, in any order. Each ``plan_assignments[plan]`` may
    also cover blocks outside the universe (unpopulated blocks, Puerto Rico);
    those are counted in the plan's metadata and otherwise ignored. Refuses a
    populated block a plan leaves unassigned, a district outside its block's
    state, and a state whose district count differs from its apportionment.
    The payload depends only on the inputs' contents, not their order.
    """

    if not plan_assignments:
        raise ValueError("CD plan registry needs at least one plan.")
    if set(plan_assignments) != set(plan_sources):
        raise ValueError(
            "plan_assignments and plan_sources must name the same plans; got "
            f"{sorted(plan_assignments)} and {sorted(plan_sources)}."
        )
    universe = _int64_array(np.asarray(block_geoid), label="block_geoid")
    weights = _int64_array(np.asarray(population), label="population")
    if universe.shape != weights.shape or universe.ndim != 1:
        raise ValueError("block_geoid and population must be aligned 1-D arrays.")
    if len(universe) == 0:
        raise ValueError("CD plan registry needs at least one populated block.")
    order = np.argsort(universe, kind="stable")
    blocks = universe[order]
    population = weights[order]
    _validate_block_geoids(blocks, label="block_geoid")
    if not bool(np.all(np.diff(blocks) > 0)):
        raise ValueError("block_geoid values must be unique.")
    if (population <= 0).any():
        bad = blocks[population <= 0][:5].tolist()
        raise ValueError(
            "population must be strictly positive (the universe is populated "
            f"blocks); non-positive at {[f'{b:015d}' for b in bad]}."
        )

    payload: dict[str, np.ndarray] = {"block_geoid": blocks, "population": population}
    plans_metadata: dict[str, Any] = {}
    for plan in sorted(plan_assignments):
        spec = _validated_plan_spec(plan, plan_sources[plan])
        districts, covered = _align(plan_assignments[plan], blocks)
        if not covered.all():
            missing = blocks[~covered]
            by_state: dict[str, int] = {}
            for state in (missing // _BLOCK_STATE_DIVISOR).tolist():
                by_state[f"{state:02d}"] = by_state.get(f"{state:02d}", 0) + 1
            raise ValueError(
                f"CD plan {plan!r} leaves {len(missing)} populated block(s) "
                f"unassigned (by state: {by_state}); examples: "
                f"{[f'{b:015d}' for b in missing[:5].tolist()]}."
            )
        district_counts = _validate_plan_districts(
            blocks, districts, plan=plan, apportionment=spec["apportionment"]
        )
        payload[f"{CD_PLAN_ARRAY_PREFIX}{plan}"] = districts.astype(np.int16)
        plans_metadata[plan] = {
            **spec,
            "districts": int(sum(district_counts.values())),
            "blocks_outside_universe": int(len(plan_assignments[plan]) - len(blocks)),
        }

    body = {
        **(dict(metadata) if metadata else {}),
        "kind": US_CD_PLAN_REGISTRY_KIND,
        "schema_version": US_CD_PLAN_REGISTRY_SCHEMA_VERSION,
        "block_vintage": US_CD_PLAN_REGISTRY_BLOCK_VINTAGE,
        "primary_plan": PRIMARY_CD_PLAN,
        "plans": plans_metadata,
    }
    payload["metadata_json"] = np.asarray(json.dumps(body, sort_keys=True))
    return payload


@dataclass(frozen=True)
class UsCdPlanRegistry:
    """Every populated 2020 block's district under each registered plan.

    Attributes:
        block_geoid: sorted, unique 15-digit 2020 block geoids (``int64``).
        population: 2020 P.L. 94-171 population per block (``int64``, > 0).
        plans: district geoid (``state_fips * 100 + district``, ``int64``)
            per plan id, aligned to ``block_geoid``.
        plan_sources: per-plan provenance (vintage, source, url, source files,
            apportionment, known deviations), as recorded in the artifact.
        metadata: the artifact's full embedded metadata.
        sha256: SHA-256 of the artifact file as loaded.
    """

    block_geoid: np.ndarray
    population: np.ndarray
    plans: Mapping[str, np.ndarray]
    plan_sources: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    sha256: str

    def __len__(self) -> int:
        return len(self.block_geoid)

    @property
    def plan_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.plans))

    def state_population(self) -> dict[str, int]:
        """Population per state FIPS."""

        return _sum_by(
            self.block_geoid // _BLOCK_STATE_DIVISOR, self.population, width=2
        )

    def district_population(self, plan: str) -> dict[str, int]:
        """Population per district geoid (``SSDD``) under ``plan``."""

        if plan not in self.plans:
            raise KeyError(
                f"CD plan {plan!r} is not in the registry ({self.plan_ids})."
            )
        return _sum_by(self.plans[plan], self.population, width=4)


def load_us_cd_plan_registry(
    path: str | Path, *, expected_sha256: str | None = None
) -> UsCdPlanRegistry:
    """Load and validate a CD plan registry artifact (NPZ).

    Re-checks every build invariant (see the module docstring), so a file
    edited after the build fails here. With ``expected_sha256`` the file must
    also match that digest; :func:`load_pinned_us_cd_plan_registry` pins to
    the packaged provenance.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"US CD plan registry artifact not found: {source}")
    digest = _sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"US CD plan registry {source} has SHA-256 {digest}, expected "
            f"{expected_sha256}."
        )
    with np.load(source, allow_pickle=False) as arrays:
        keys = set(arrays.files)
        for required in ("block_geoid", "population", "metadata_json"):
            if required not in keys:
                raise ValueError(f"US CD plan registry is missing array {required!r}.")
        metadata = json.loads(str(arrays["metadata_json"]))
        block_geoid = _int64_array(arrays["block_geoid"], label="block_geoid")
        population = _int64_array(arrays["population"], label="population")
        plan_arrays = {
            key[len(CD_PLAN_ARRAY_PREFIX) :]: _int64_array(arrays[key], label=key)
            for key in keys
            if key.startswith(CD_PLAN_ARRAY_PREFIX)
        }
        unexpected = {
            key
            for key in keys - {"block_geoid", "population", "metadata_json"}
            if not key.startswith(CD_PLAN_ARRAY_PREFIX)
        }
    if unexpected:
        raise ValueError(
            f"US CD plan registry has unexpected arrays {sorted(unexpected)}."
        )

    for key, expected in (
        ("kind", US_CD_PLAN_REGISTRY_KIND),
        ("schema_version", US_CD_PLAN_REGISTRY_SCHEMA_VERSION),
        ("block_vintage", US_CD_PLAN_REGISTRY_BLOCK_VINTAGE),
    ):
        if metadata.get(key) != expected:
            raise ValueError(
                f"US CD plan registry metadata {key}={metadata.get(key)!r}, "
                f"expected {expected!r}."
            )
    plans_metadata = metadata.get("plans")
    if not isinstance(plans_metadata, Mapping) or not plans_metadata:
        raise ValueError("US CD plan registry metadata must list its plans.")
    if set(plans_metadata) != set(plan_arrays):
        raise ValueError(
            "US CD plan registry metadata plans "
            f"{sorted(plans_metadata)} do not match its arrays {sorted(plan_arrays)}."
        )

    _require_aligned_1d(
        {"block_geoid": block_geoid, "population": population}
        | {
            f"{CD_PLAN_ARRAY_PREFIX}{plan}": values
            for plan, values in plan_arrays.items()
        },
        label="US CD plan registry",
    )
    _validate_block_geoids(block_geoid, label="block_geoid")
    if len(block_geoid) == 0:
        raise ValueError("US CD plan registry has zero blocks.")
    if not bool(np.all(np.diff(block_geoid) > 0)):
        raise ValueError("US CD plan registry block_geoid must be sorted and unique.")
    if (population <= 0).any():
        raise ValueError(
            "US CD plan registry population must be positive for every block."
        )

    plan_sources: dict[str, dict[str, Any]] = {}
    for plan in sorted(plan_arrays):
        spec = _validated_plan_spec(plan, plans_metadata[plan])
        values = plan_arrays[plan]
        counts = _validate_plan_districts(
            block_geoid, values, plan=plan, apportionment=spec["apportionment"]
        )
        recorded = plans_metadata[plan].get("districts")
        if recorded is not None and recorded != sum(counts.values()):
            raise ValueError(
                f"CD plan {plan!r} metadata records {recorded} districts, "
                f"the array has {sum(counts.values())}."
            )
        plan_sources[plan] = dict(plans_metadata[plan])

    return UsCdPlanRegistry(
        block_geoid=block_geoid,
        population=population,
        plans=plan_arrays,
        plan_sources=plan_sources,
        metadata=metadata,
        sha256=digest,
    )


def packaged_us_cd_plan_registry_provenance() -> dict[str, Any]:
    """The packaged build receipt: artifact SHA-256, sources, per-plan summary."""

    resource = files("microcosm.build.us_runtime").joinpath(
        US_CD_PLAN_REGISTRY_PROVENANCE_RESOURCE
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    digest = payload.get("output_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("US CD plan registry provenance has no valid output_sha256.")
    return payload


def load_pinned_us_cd_plan_registry(path: str | Path) -> UsCdPlanRegistry:
    """Load the registry, refusing any file but the one the package pins."""

    return load_us_cd_plan_registry(
        path, expected_sha256=packaged_us_cd_plan_registry_provenance()["output_sha256"]
    )


def check_cd_plan_registry_against_block_ladder(
    registry: UsCdPlanRegistry,
    *,
    block_geoid: np.ndarray,
    population: np.ndarray,
    congressional_district_geoid: np.ndarray,
    plan: str = PRIMARY_CD_PLAN,
) -> dict[str, Any]:
    """Differential check of the registry against a block ladder's own arrays.

    The ladder and the registry are built independently from the same Census
    sources, so they must cover the same blocks with the same populations and
    agree exactly on ``plan`` (the ladder's primary CD plan). Raises on any
    difference; returns a small receipt on agreement.
    """

    _require_aligned_1d(
        {
            "block_geoid": np.asarray(block_geoid),
            "population": np.asarray(population),
            "congressional_district_geoid": np.asarray(congressional_district_geoid),
        },
        label="Block ladder",
    )
    ladder_blocks = _int64_array(np.asarray(block_geoid), label="ladder block_geoid")
    order = np.argsort(ladder_blocks, kind="stable")
    ladder_blocks = ladder_blocks[order]
    ladder_population = _int64_array(np.asarray(population), label="ladder population")[
        order
    ]
    ladder_cd = _int64_array(
        np.asarray(congressional_district_geoid),
        label="ladder congressional_district_geoid",
    )[order]
    if plan not in registry.plans:
        raise KeyError(
            f"CD plan {plan!r} is not in the registry ({registry.plan_ids})."
        )
    if len(ladder_blocks) != len(registry.block_geoid) or not np.array_equal(
        ladder_blocks, registry.block_geoid
    ):
        only_ladder = np.setdiff1d(ladder_blocks, registry.block_geoid)
        only_registry = np.setdiff1d(registry.block_geoid, ladder_blocks)
        raise ValueError(
            "Block ladder and CD plan registry cover different blocks: "
            f"{len(only_ladder)} only in the ladder (e.g. "
            f"{[f'{b:015d}' for b in only_ladder[:3].tolist()]}), "
            f"{len(only_registry)} only in the registry (e.g. "
            f"{[f'{b:015d}' for b in only_registry[:3].tolist()]})."
        )
    population_mismatch = ladder_population != registry.population
    if population_mismatch.any():
        raise ValueError(
            f"Block ladder and CD plan registry disagree on the population of "
            f"{int(population_mismatch.sum())} block(s), e.g. "
            f"{[f'{b:015d}' for b in ladder_blocks[population_mismatch][:3].tolist()]}."
        )
    cd_mismatch = ladder_cd != registry.plans[plan]
    if cd_mismatch.any():
        raise ValueError(
            f"Block ladder and CD plan registry disagree on the {plan} district of "
            f"{int(cd_mismatch.sum())} block(s), e.g. "
            f"{[f'{b:015d}' for b in ladder_blocks[cd_mismatch][:3].tolist()]}."
        )
    return {
        "plan": plan,
        "blocks": int(len(ladder_blocks)),
        "population": int(ladder_population.sum()),
        "agreement": "exact",
    }


def summarize_cd_plan_registry(registry: UsCdPlanRegistry) -> dict[str, Any]:
    """Per-plan totals for the build receipt, re-checking conservation.

    For every plan, district populations summed within each state must equal
    that state's population; a failure here means the arrays are corrupt.
    """

    state_population = registry.state_population()
    plans: dict[str, Any] = {}
    for plan in registry.plan_ids:
        district_population = registry.district_population(plan)
        by_state: dict[str, int] = {}
        districts_by_state: dict[str, int] = {}
        for district, value in district_population.items():
            by_state[district[:2]] = by_state.get(district[:2], 0) + value
            districts_by_state[district[:2]] = (
                districts_by_state.get(district[:2], 0) + 1
            )
        if by_state != state_population:
            raise ValueError(f"CD plan {plan!r} does not conserve state populations.")
        values = np.asarray(list(district_population.values()), dtype=np.int64)
        primary = registry.plans.get(PRIMARY_CD_PLAN)
        changed_states: list[str] = []
        if primary is not None and plan != PRIMARY_CD_PLAN:
            differs = registry.plans[plan] != primary
            changed_states = sorted(
                {
                    f"{state:02d}"
                    for state in (
                        registry.block_geoid[differs] // _BLOCK_STATE_DIVISOR
                    ).tolist()
                }
            )
        plans[plan] = {
            "districts": int(len(district_population)),
            "districts_by_state": dict(sorted(districts_by_state.items())),
            "min_district_population": int(values.min()),
            "max_district_population": int(values.max()),
            "states_differing_from_primary": changed_states,
        }
    return {
        "blocks": int(len(registry)),
        "population": int(registry.population.sum()),
        "states": len(state_population),
        "primary_plan": PRIMARY_CD_PLAN,
        "plans": plans,
    }


def _sorted_assignment(
    blocks: np.ndarray, districts: np.ndarray, *, label: str
) -> CdBlockAssignment:
    blocks = _int64_array(np.asarray(blocks), label=f"{label} block_geoid")
    districts = _int64_array(np.asarray(districts), label=f"{label} district_geoid")
    if blocks.ndim != 1 or blocks.shape != districts.shape:
        raise ValueError(
            f"{label}: block and district arrays must be aligned 1-D arrays."
        )
    _validate_block_geoids(blocks, label=label)
    order = np.argsort(blocks, kind="stable")
    blocks = blocks[order]
    districts = districts[order]
    duplicated = blocks[1:][np.diff(blocks) == 0]
    if len(duplicated):
        raise ValueError(
            f"{label} assigns {len(duplicated)} block(s) more than once, e.g. "
            f"{int(duplicated[0]):015d}."
        )
    return CdBlockAssignment(blocks, districts)


def _align(
    assignment: CdBlockAssignment, blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Districts of ``blocks`` (any order) under ``assignment``, and a coverage mask."""

    if len(assignment) == 0:
        return np.zeros(len(blocks), dtype=np.int64), np.zeros(len(blocks), dtype=bool)
    index = np.searchsorted(assignment.block_geoid, blocks)
    clipped = np.minimum(index, len(assignment) - 1)
    covered = assignment.block_geoid[clipped] == blocks
    return np.where(covered, assignment.district_geoid[clipped], 0), covered


def _validated_plan_spec(plan: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not _PLAN_ID.fullmatch(plan):
        raise ValueError(
            f"CD plan id {plan!r} must be lowercase letters, digits and underscores."
        )
    if not isinstance(spec, Mapping):
        raise ValueError(f"CD plan {plan!r} metadata must be a mapping.")
    missing = [key for key in REQUIRED_PLAN_SOURCE_KEYS if not spec.get(key)]
    if missing:
        raise ValueError(f"CD plan {plan!r} metadata is missing {missing}.")
    if spec["vintage"] != plan:
        raise ValueError(
            f"CD plan {plan!r} records vintage {spec['vintage']!r}; they must match."
        )
    if spec["apportionment"] not in US_HOUSE_APPORTIONMENT:
        raise ValueError(
            f"CD plan {plan!r} apportionment {spec['apportionment']!r} is not one of "
            f"{sorted(US_HOUSE_APPORTIONMENT)}."
        )
    source_files = spec["source_files"]
    if not isinstance(source_files, Mapping):
        raise ValueError(f"CD plan {plan!r} source_files must be a mapping.")
    for key, entry in source_files.items():
        if not (
            isinstance(entry, Mapping)
            and entry.get("url")
            and re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ValueError(
                f"CD plan {plan!r} source file {key!r} must record a url and a SHA-256."
            )
    return dict(spec)


def _validate_plan_districts(
    blocks: np.ndarray, districts: np.ndarray, *, plan: str, apportionment: str
) -> dict[str, int]:
    """Check state consistency and district rosters; return counts by state."""

    if (districts <= 0).any():
        raise ValueError(f"CD plan {plan!r} has non-positive district geoids.")
    block_state = blocks // _BLOCK_STATE_DIVISOR
    mismatched = districts // 100 != block_state
    if mismatched.any():
        examples = [
            f"{block:015d} -> {district:04d}"
            for block, district in zip(
                blocks[mismatched][:5].tolist(),
                districts[mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(
            f"CD plan {plan!r} puts {int(mismatched.sum())} block(s) in another "
            f"state's district: {examples}."
        )
    unique = np.unique(districts)
    rosters: dict[str, list[int]] = {}
    for district in unique.tolist():
        rosters.setdefault(f"{district // 100:02d}", []).append(district % 100)
    expected = expected_district_counts(apportionment)
    wrong = {}
    for state, roster in sorted(rosters.items()):
        count = expected.get(state)
        want = (
            None
            if count is None
            else ([0] if count == 1 else list(range(1, count + 1)))
        )
        if roster != want:
            wrong[state] = {"districts": roster, "apportioned": count}
    if wrong:
        raise ValueError(
            f"CD plan {plan!r} district rosters differ from the {apportionment} "
            f"apportionment (at-large and delegate: 00; otherwise 01..n): {wrong}."
        )
    counts = {state: len(roster) for state, roster in rosters.items()}
    return counts


def _require_aligned_1d(arrays: Mapping[str, np.ndarray], *, label: str) -> None:
    """Refuse any array that is not 1-D or not the length of the others."""

    shapes = {name: tuple(np.shape(values)) for name, values in arrays.items()}
    not_1d = {name: shape for name, shape in shapes.items() if len(shape) != 1}
    if not_1d:
        raise ValueError(f"{label} arrays must be one-dimensional; got {not_1d}.")
    if len(set(shapes.values())) > 1:
        raise ValueError(f"{label} arrays are not aligned: {shapes}.")


def _validate_block_geoids(blocks: np.ndarray, *, label: str) -> None:
    bad = (blocks < _MIN_BLOCK_GEOID) | (blocks >= _MAX_BLOCK_GEOID)
    if bad.any():
        raise ValueError(
            f"{label} values must be 15-digit block geoids; invalid: "
            f"{blocks[bad][:5].tolist()}."
        )


def _sum_by(keys: np.ndarray, values: np.ndarray, *, width: int) -> dict[str, int]:
    unique, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(
        inverse, weights=values.astype(np.float64), minlength=len(unique)
    )
    # Exact integer sums: bincount's float64 is exact below 2**53, far above
    # the US population.
    return {
        f"{int(key):0{width}d}": int(total)
        for key, total in zip(unique.tolist(), sums.tolist(), strict=True)
    }


def _column_index(columns: list[str], names: tuple[str, ...], *, label: str) -> int:
    for name in names:
        if name in columns:
            return columns.index(name)
    raise ValueError(f"{label} header {columns} has none of the columns {list(names)}.")


def _required_header(iterator: Iterable[str], *, source: str) -> str:
    for line in iterator:
        stripped = line.strip().lstrip("﻿")
        if stripped:
            return stripped
    raise ValueError(f"{source} is empty.")


def _block_geoid(value: str, *, source: str) -> int:
    if not (value.isdigit() and len(value) == 15):
        raise ValueError(f"{source}: block geoid must be 15 digits, got {value!r}.")
    return int(value)


def _int64_array(values: np.ndarray, *, label: str) -> np.ndarray:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{label} must be an integer array, got {values.dtype}.")
    return values.astype(np.int64, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CD_PLAN_ARRAY_PREFIX",
    "PRIMARY_CD_PLAN",
    "US_CD_PLAN_REGISTRY_KIND",
    "US_HOUSE_APPORTIONMENT",
    "US_HOUSE_APPORTIONMENT_SOURCE",
    "BlockRelationship",
    "CdBlockAssignment",
    "UsCdPlanRegistry",
    "assemble_us_cd_plan_registry",
    "assignment_agreement",
    "cd_block_assignment",
    "check_cd_plan_registry_against_block_ladder",
    "concat_cd_block_assignments",
    "crosswalk_plan_to_2020_blocks",
    "expected_district_counts",
    "fill_unassigned_blocks",
    "load_pinned_us_cd_plan_registry",
    "load_us_cd_plan_registry",
    "packaged_us_cd_plan_registry_provenance",
    "parse_block_relationship_2010_2020",
    "parse_cd_block_assignment",
    "replace_state_assignments",
    "summarize_cd_plan_registry",
]
