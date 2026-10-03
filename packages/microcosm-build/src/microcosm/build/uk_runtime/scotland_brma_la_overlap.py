"""How Scotland's Broad Rental Market Areas overlap its council areas.

ONS's Price Index of Private Rents publishes Scottish rent levels for the 18
BRMAs; the local target surface is bound at council (local-authority) grain.
``scotland_brma_la_overlap.json`` measures the intersections in households
from Scotland's Census 2022 and the official BRMA geography, and this module
reads it and answers the structural questions the private-rent parity concern
rests on (microcosm#1090): which councils a BRMA covers, whether any grouping
makes BRMAs a union of whole councils, and which council is the same place as
one BRMA.

The resource is a measurement, not a translation: nothing here turns a BRMA
rent into a council target.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources as importlib_resources
from typing import Any

UK_SCOTLAND_BRMA_LA_OVERLAP_RESOURCE = "scotland_brma_la_overlap.json"
UK_SCOTLAND_BRMA_LA_OVERLAP_SCHEMA_VERSION = 1
SCOTLAND_BRMA_COUNT = 18
#: ``private_rented_households`` is the measure PIPR's rents describe;
#: ``all_households`` needs no within-ward assumption and is the cross-check.
OVERLAP_WEIGHTS = ("private_rented_households", "all_households")
#: A council and a BRMA coincide when each holds at least this share of the
#: other's private-rented households.
COINCIDENT_MIN_SHARE = 0.99
#: Relative tolerance of the cells against the census total they are rounded
#: from (four decimals over 63 cells).
_TOTAL_RTOL = 1e-6


@dataclass(frozen=True)
class BrmaCouncilOverlap:
    """One BRMA x council intersection and its share of each side."""

    brma: str
    local_authority: str
    households: float
    share_of_council: float
    share_of_brma: float


def load_uk_scotland_brma_la_overlap() -> Mapping[str, Any]:
    """Load and validate the packaged overlap resource."""

    return _load()


@cache
def _load() -> Mapping[str, Any]:
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_SCOTLAND_BRMA_LA_OVERLAP_RESOURCE)
        .read_text()
    )
    validate_uk_scotland_brma_la_overlap(payload)
    return payload


def validate_uk_scotland_brma_la_overlap(payload: Mapping[str, Any]) -> None:
    """Refuse an overlap table that does not cover both rosters or add up."""

    label = UK_SCOTLAND_BRMA_LA_OVERLAP_RESOURCE
    if payload.get("schema_version") != UK_SCOTLAND_BRMA_LA_OVERLAP_SCHEMA_VERSION:
        raise ValueError(f"{label}: schema_version does not match the runtime.")
    brmas = payload.get("brmas")
    cells = payload.get("cells")
    totals = payload.get("totals")
    if not isinstance(brmas, Mapping) or len(brmas) != SCOTLAND_BRMA_COUNT:
        raise ValueError(f"{label}: expected {SCOTLAND_BRMA_COUNT} BRMAs.")
    codes = [str(entry.get("pipr_area_code", "")) for entry in brmas.values()]
    if len(set(codes)) != len(codes) or not all(
        code.startswith("S33") for code in codes
    ):
        raise ValueError(f"{label}: BRMAs need distinct S33 PIPR area codes.")
    if not isinstance(cells, list) or not cells or not isinstance(totals, Mapping):
        raise ValueError(f"{label}: cells and totals are required.")

    councils = _scottish_council_roster()
    seen: set[tuple[str, str]] = set()
    for cell in cells:
        key = (str(cell.get("brma", "")), str(cell.get("local_authority", "")))
        if key in seen:
            raise ValueError(f"{label}: duplicate cell {key}.")
        seen.add(key)
        if key[0] not in brmas:
            raise ValueError(f"{label}: cell names unknown BRMA {key[0]!r}.")
        if key[1] not in councils:
            raise ValueError(
                f"{label}: cell names {key[1]!r}, which is not a Scottish "
                "local authority in the local-area crosswalk."
            )
        for weight in OVERLAP_WEIGHTS:
            value = cell.get(weight)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{label}: cell {key} has invalid {weight}.")
        if cell["all_households"] <= 0:
            # A cell exists because an output area's households sit in it.
            raise ValueError(f"{label}: cell {key} holds no households.")
    if {brma for brma, _ in seen} != set(brmas):
        raise ValueError(f"{label}: a BRMA has no cell.")
    if {council for _, council in seen} != councils:
        raise ValueError(f"{label}: a Scottish local authority has no cell.")
    if totals.get("cells") != len(cells):
        raise ValueError(f"{label}: totals.cells does not match the cells.")
    for weight in OVERLAP_WEIGHTS:
        total = totals.get(weight)
        summed = math.fsum(float(cell[weight]) for cell in cells)
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or not math.isclose(summed, total, rel_tol=_TOTAL_RTOL, abs_tol=0.0)
        ):
            raise ValueError(
                f"{label}: {weight} cells sum to {summed!r}, not the declared "
                f"total {total!r}."
            )


def _scottish_council_roster() -> frozenset[str]:
    crosswalk = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text()
    )
    return frozenset(
        str(area_id)
        for area_id in crosswalk["levels"]["local_authority"]["area_ids"]
        if str(area_id).startswith("S")
    )


def scotland_brma_council_overlaps(
    weight: str = "private_rented_households",
) -> tuple[BrmaCouncilOverlap, ...]:
    """Every intersection holding ``weight``, with its share of each side."""

    if weight not in OVERLAP_WEIGHTS:
        raise ValueError(f"unknown overlap weight {weight!r}; use {OVERLAP_WEIGHTS}.")
    cells = [
        cell
        for cell in load_uk_scotland_brma_la_overlap()["cells"]
        if float(cell[weight]) > 0
    ]
    by_council: dict[str, float] = {}
    by_brma: dict[str, float] = {}
    for cell in cells:
        value = float(cell[weight])
        council, brma = str(cell["local_authority"]), str(cell["brma"])
        by_council[council] = by_council.get(council, 0.0) + value
        by_brma[brma] = by_brma.get(brma, 0.0) + value
    return tuple(
        BrmaCouncilOverlap(
            brma=str(cell["brma"]),
            local_authority=str(cell["local_authority"]),
            households=float(cell[weight]),
            share_of_council=float(cell[weight])
            / by_council[str(cell["local_authority"])],
            share_of_brma=float(cell[weight]) / by_brma[str(cell["brma"])],
        )
        for cell in cells
    )


def scotland_brma_council_blocks(
    weight: str = "private_rented_households",
    *,
    min_share: float = 0.0,
) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
    """Smallest groups in which a set of BRMAs equals a set of whole councils.

    Each block is ``(brmas, councils)``: the connected pieces of the overlap.
    A rent total is additive over a block and over nothing finer, so a block
    is the finest area at which BRMA rents and council rows describe the same
    households. ``min_share`` drops an intersection that is below that share
    of both its council and its BRMA before grouping, to show how much of the
    joining is slivers; the default drops nothing.
    """

    if not 0.0 <= min_share <= 1.0:
        raise ValueError("min_share must lie in [0, 1].")
    links: dict[str, set[str]] = {}
    overlaps = scotland_brma_council_overlaps(weight)
    for overlap in overlaps:
        brma, council = f"brma:{overlap.brma}", f"la:{overlap.local_authority}"
        links.setdefault(brma, set())
        links.setdefault(council, set())
        if max(overlap.share_of_council, overlap.share_of_brma) >= min_share:
            links[brma].add(council)
            links[council].add(brma)
    blocks: list[tuple[frozenset[str], frozenset[str]]] = []
    seen: set[str] = set()
    for start in sorted(links):
        if start in seen:
            continue
        block: set[str] = set()
        pending = [start]
        while pending:
            node = pending.pop()
            if node in block:
                continue
            block.add(node)
            pending.extend(links[node] - block)
        seen |= block
        blocks.append(
            (
                frozenset(n.removeprefix("brma:") for n in block if n[:5] == "brma:"),
                frozenset(n.removeprefix("la:") for n in block if n[:3] == "la:"),
            )
        )
    return tuple(blocks)


def scotland_coincident_councils(
    *,
    min_share: float = COINCIDENT_MIN_SHARE,
) -> Mapping[str, str]:
    """Councils that are one BRMA's private renters, and it theirs.

    ``council -> brma`` for each pair where the intersection holds at least
    ``min_share`` of the council's and of the BRMA's private-rented
    households. For such a pair the BRMA's published rent describes the
    council's renters up to the remaining share on each side.
    """

    if not 0.5 < min_share <= 1.0:
        raise ValueError("min_share must lie in (0.5, 1].")
    return {
        overlap.local_authority: overlap.brma
        for overlap in scotland_brma_council_overlaps()
        if overlap.share_of_council >= min_share and overlap.share_of_brma >= min_share
    }


__all__ = [
    "COINCIDENT_MIN_SHARE",
    "OVERLAP_WEIGHTS",
    "SCOTLAND_BRMA_COUNT",
    "UK_SCOTLAND_BRMA_LA_OVERLAP_RESOURCE",
    "UK_SCOTLAND_BRMA_LA_OVERLAP_SCHEMA_VERSION",
    "BrmaCouncilOverlap",
    "load_uk_scotland_brma_la_overlap",
    "scotland_brma_council_blocks",
    "scotland_brma_council_overlaps",
    "scotland_coincident_councils",
    "validate_uk_scotland_brma_la_overlap",
]
