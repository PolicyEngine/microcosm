"""US cross-grain nesting declarations (microcosm#1760).

``us/cross_grain_declarations.json`` holds every US choice the shared
cross-grain operator (:mod:`microcosm.build.cross_grain`) and its helpers
(:mod:`microcosm.build.cross_grain_controls`) need: the order of the geography
levels, the levels that hold controls, and, for each mechanism that makes the
rows of a lower level from a control at a higher one, its allocator, the rule
for its control, its bounds, the source of its member lists and its reviewed
registers with their reasons. The key names follow
``uk/cross_grain_declarations.json`` wherever the meaning is the same.

The file's ``inventory`` lists each US family that two geography levels
measure, with the route that ties the levels or the reason none does.
:func:`us_cross_grain_coverage_violations` checks a set of compiled specs
against it. That check is a receipt and a test of the declaration: no build
calls it and it refuses nothing.

This module reads the file and hands out typed views of it. It does no
arithmetic on targets; the adapter that runs the operator is
:mod:`microcosm.build.us_runtime.target_nesting`.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
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
#: Inventory routes that need a written reason.
_REASONED_ROUTES = frozenset(
    {
        "compile_rule",
        "same_source",
        "no_higher_control",
        "not_reconciled",
        "not_in_the_fit",
    }
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
    _validate_inventory(payload)
    return payload


def _validate_inventory(payload: Mapping[str, Any]) -> None:
    """Refuse an inventory entry that names no route, level or reason."""

    grains = tuple(payload["grain_precedence"])
    inventory = payload["inventory"]
    routes = tuple(inventory["routes"])
    seen: set[str] = set()
    for entry in inventory["entries"]:
        entry_id = str(entry.get("id", ""))
        if not entry_id or entry_id in seen:
            raise ValueError(
                f"US cross-grain inventory entry id {entry_id!r} is blank or repeated."
            )
        seen.add(entry_id)
        levels = tuple(entry.get("levels") or ())
        higher = entry.get("higher_level")
        if (
            not entry.get("family")
            or not levels
            or any(level not in grains for level in levels)
            or higher not in grains
            or any(grains.index(higher) > grains.index(level) for level in levels)
        ):
            raise ValueError(
                f"US cross-grain inventory entry {entry_id!r} must name a "
                "family, its levels and a higher level that is not below them."
            )
        route = entry.get("route")
        if route not in routes:
            raise ValueError(
                f"US cross-grain inventory entry {entry_id!r} names unknown "
                f"route {route!r}; expected one of {routes}."
            )
        if not isinstance(entry.get("compiled"), bool) or not entry.get("issues"):
            raise ValueError(
                f"US cross-grain inventory entry {entry_id!r} must say whether "
                "its rows are compiled and name an issue."
            )
        if route == "mechanism":
            mechanism = payload["mechanisms"].get(entry.get("mechanism"))
            if (
                mechanism is None
                or mechanism["issue_row"] != entry.get("issue_row")
                or mechanism["family"] != entry["family"]
                or (mechanism["lower_grain"],) != levels
                or mechanism["higher_grain"] != higher
            ):
                raise ValueError(
                    f"US cross-grain inventory entry {entry_id!r} does not "
                    "match the mechanism it names."
                )
        elif route in _REASONED_ROUTES and not str(entry.get("reason", "")).strip():
            raise ValueError(
                f"US cross-grain inventory entry {entry_id!r} has route "
                f"{route!r} and no reason."
            )
        if route == "compile_rule" and not (entry.get("code") and entry.get("marker")):
            raise ValueError(
                f"US cross-grain inventory entry {entry_id!r} must name the "
                "compile rule's code and the marker it writes."
            )


def us_cross_grain_inventory() -> tuple[Mapping[str, Any], ...]:
    """The inventory entries, in declared order."""

    return tuple(load_us_cross_grain_declarations()["inventory"]["entries"])


def _marker_matches(marker: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    value = metadata.get(str(marker["metadata_key"]))
    if value is None:
        return False
    if "value" in marker:
        return str(value) == str(marker["value"])
    if "value_prefix" in marker:
        return str(value).startswith(str(marker["value_prefix"]))
    return True


def us_cross_grain_coverage_violations(
    specs: Iterable[Any],
    *,
    level_key: str = "ledger_geography_level",
) -> list[dict[str, str]]:
    """Every gap between a set of compiled specs and the inventory.

    ``specs`` are compiled target specs (anything with ``family`` and
    ``metadata``). The top level of ``grain_precedence`` nests in nothing. A
    violation is one of:

    * ``family_level_without_route``: a family has rows at a lower level and
      no inventory entry names that family and level;
    * ``stale_inventory_entry``: an entry says its rows are compiled, and the
      specs hold none at one of its levels;
    * ``compile_rule_marker_absent``: a compile rule's marker is on no row of
      one of its declared levels.

    Returns the violations; an empty list means the inventory covers the
    specs. The caller decides what a violation means: this function refuses
    nothing.
    """

    declared = load_us_cross_grain_declarations()
    top = declared["grain_precedence"][0]
    compiled: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for spec in specs:
        level = spec.metadata.get(level_key)
        if level is None or level == top:
            continue
        compiled.setdefault((str(spec.family), str(level)), []).append(spec.metadata)
    entries = us_cross_grain_inventory()
    routed = {
        (str(entry["family"]), str(level))
        for entry in entries
        for level in entry["levels"]
    }
    violations = [
        {
            "kind": "family_level_without_route",
            "family": family,
            "level": level,
            "where": "compiled specs",
        }
        for family, level in sorted(set(compiled) - routed)
    ]
    for entry in entries:
        if not entry["compiled"]:
            continue
        for level in entry["levels"]:
            rows = compiled.get((str(entry["family"]), str(level)))
            if not rows:
                violations.append(
                    {
                        "kind": "stale_inventory_entry",
                        "family": str(entry["family"]),
                        "level": str(level),
                        "where": str(entry["id"]),
                    }
                )
            elif entry["route"] == "compile_rule" and not any(
                _marker_matches(entry["marker"], metadata) for metadata in rows
            ):
                violations.append(
                    {
                        "kind": "compile_rule_marker_absent",
                        "family": str(entry["family"]),
                        "level": str(level),
                        "where": str(entry["id"]),
                    }
                )
    return violations


def us_cross_grain_coverage_receipt(specs: Iterable[Any]) -> dict[str, Any]:
    """The coverage of a set of compiled specs, as a receipt.

    It records the violations and never raises on one.
    """

    specs = tuple(specs)
    violations = us_cross_grain_coverage_violations(specs)
    return {
        "declarations": US_CROSS_GRAIN_DECLARATIONS_RESOURCE,
        "checked_specs": len(specs),
        "inventory_entries": len(us_cross_grain_inventory()),
        "violations": violations,
        "covered": not violations,
    }


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
