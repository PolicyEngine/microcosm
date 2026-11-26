"""US cross-grain nesting declarations (microcosm#1760).

``us/cross_grain_declarations.json`` holds every US choice the shared
cross-grain operator (:mod:`microcosm.build.cross_grain`) and its helpers
(:mod:`microcosm.build.cross_grain_controls`) need: the order of the geography
levels, the levels that hold controls, and, for each mechanism that makes the
rows of a lower level from a control at a higher one, its allocator, the rule
for its control, its bounds, the source of its member lists and its reviewed
registers with their reasons. The key names follow
``uk/cross_grain_declarations.json`` wherever the meaning is the same.

This module reads the file and hands out typed views of it. It does no
arithmetic and reads no target; the adapter that runs the operator is
:mod:`microcosm.build.us_runtime.target_nesting`.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources as importlib_resources
from typing import Any

from microcosm.build.cross_grain_controls import ALLOCATORS

US_CROSS_GRAIN_DECLARATIONS_RESOURCE = "cross_grain_declarations.json"

#: The mechanisms of the issue's rows 5 to 8, by their declared id.
DISTRICT_SOI = "state_congressional_district_soi"
COUNTY_POPULATION = "state_county_population"
COUNTY_SOI = "state_county_soi"
COUNTY_ACS_HOUSEHOLDS = "county_state_acs_households"

_CONTROL_KINDS = (
    "bound_row",
    "bound_row_or_scaled_by_sibling_ratio",
    "sum_of_lower_rows",
)


@functools.lru_cache(maxsize=1)
def load_us_cross_grain_declarations() -> Mapping[str, Any]:
    """The declarations file, checked for the fields every reader relies on."""

    payload = json.loads(
        importlib_resources.files("microcosm.build.us")
        .joinpath(US_CROSS_GRAIN_DECLARATIONS_RESOURCE)
        .read_text(encoding="utf-8")
    )
    if payload.get("schema_version") != 1:
        raise ValueError(
            "US cross-grain declarations schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    grains = tuple(payload["grain_precedence"])
    unknown_controls = [g for g in payload["control_grains"] if g not in grains]
    if unknown_controls:
        raise ValueError(
            f"US cross-grain control_grains {unknown_controls} are not in "
            "grain_precedence."
        )
    for mechanism_id, mechanism in payload["mechanisms"].items():
        higher, lower = mechanism["higher_grain"], mechanism["lower_grain"]
        if (
            higher not in payload["control_grains"]
            or lower not in grains
            or grains.index(higher) >= grains.index(lower)
        ):
            raise ValueError(
                f"US cross-grain mechanism {mechanism_id!r} must name a control "
                f"grain above its lower grain, got {higher!r} over {lower!r}."
            )
        if mechanism["allocator"] not in ALLOCATORS:
            raise ValueError(
                f"US cross-grain mechanism {mechanism_id!r} names unknown "
                f"allocator {mechanism['allocator']!r}; expected one of "
                f"{sorted(ALLOCATORS)}."
            )
        if mechanism["control"]["kind"] not in _CONTROL_KINDS:
            raise ValueError(
                f"US cross-grain mechanism {mechanism_id!r} names unknown "
                f"control kind {mechanism['control']['kind']!r}; expected one "
                f"of {_CONTROL_KINDS}."
            )
    return payload


def us_cross_grain_mechanism(mechanism_id: str) -> Mapping[str, Any]:
    """One declared mechanism; an unknown id is refused."""

    mechanisms = load_us_cross_grain_declarations()["mechanisms"]
    if mechanism_id not in mechanisms:
        raise ValueError(
            f"Unknown US cross-grain mechanism {mechanism_id!r}; declared: "
            f"{sorted(mechanisms)}."
        )
    return mechanisms[mechanism_id]


@dataclass(frozen=True)
class DistrictSoiNesting:
    """The declared choices of the state-to-district SOI mechanism."""

    vintage_rule: str
    allocator: str
    level_bridges: Mapping[str, str]
    factor_band: float
    internal_rtol: float
    closure_rtol: float
    reviewed_crosswalk_sha256: str
    defective_lower_measures: Mapping[str, str]
    unparented_lower_measures: Mapping[str, str]
    excluded_legs: Mapping[str, str]


@functools.lru_cache(maxsize=1)
def us_district_soi_nesting() -> DistrictSoiNesting:
    declared = us_cross_grain_mechanism(DISTRICT_SOI)
    registers = declared["registers"]
    return DistrictSoiNesting(
        vintage_rule=str(declared["vintage_rule"]),
        allocator=str(declared["allocator"]),
        level_bridges=dict(declared["control"]["level_bridges"]["siblings"]),
        factor_band=declared["factor_band"]["tolerance"],
        internal_rtol=declared["self_consistency"]["rtol"],
        closure_rtol=declared["closure"]["rtol"],
        reviewed_crosswalk_sha256=str(declared["members"]["reviewed_crosswalk_sha256"]),
        defective_lower_measures=dict(registers["defective_lower_measures"]["entries"]),
        unparented_lower_measures=dict(
            registers["unparented_lower_measures"]["entries"]
        ),
        excluded_legs=dict(registers["excluded_legs"]["entries"]),
    )


@dataclass(frozen=True)
class CountySoiNesting:
    """The declared choices of the state-to-county SOI mechanism."""

    allocator: str
    required_measures: tuple[str, ...]
    #: Absent county -> host county, as five-digit integer codes.
    combined_cells: Mapping[int, int]
    combined_cell_basis: str


@functools.lru_cache(maxsize=1)
def us_county_soi_nesting() -> CountySoiNesting:
    declared = us_cross_grain_mechanism(COUNTY_SOI)
    members = declared["members"]
    return CountySoiNesting(
        allocator=str(declared["allocator"]),
        required_measures=tuple(str(m) for m in declared["required_measures"]),
        combined_cells={
            int(cell["absent"]): int(cell["host"]) for cell in members["combined_cells"]
        },
        combined_cell_basis=str(members["combined_cell_basis"]),
    )
