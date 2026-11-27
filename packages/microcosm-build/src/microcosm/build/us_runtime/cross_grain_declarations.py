"""US cross-grain nesting declarations (microcosm#1760).

``us/cross_grain_declarations.json`` holds the US choices of the shared
cross-grain operator (:mod:`microcosm.build.cross_grain`), under the key names
of ``uk/cross_grain_declarations.json`` where the meaning is the same:

- ``grain_precedence``, ``control_grains``, ``signature_fields``;
- ``bridge_selectors``: the bridges between two sources of one quantity, one
  entry for all the measures of a pair of sources;
- ``derived_controls``: a control that no source publishes as one row;
- ``same_measurement``: one source at two levels, where nothing else is
  declared;
- ``relations`` and ``local_routes``: the pairs that are not reconciled and
  the lower targets with no higher control, each with a reason;
- ``members`` and ``registers``: what each leg must hold, and which rows of
  a source enter the surface;
- ``until_the_feed_repin``: what goes when the feed is re-pinned.

This module reads the file and hands out typed views of it.
:func:`us_cross_grain_coverage_violations` checks a set of compiled specs
against it. That check is a receipt and a test of the declaration: no build
calls it and it refuses nothing. The adapter that runs the operator is
:mod:`microcosm.build.us_runtime.target_nesting`.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources as importlib_resources
from typing import Any

from microcosm.build.cross_grain import CrossGrainBridgeSelector
from microcosm.build.cross_grain_controls import ALLOCATORS

US_CROSS_GRAIN_DECLARATIONS_RESOURCE = "cross_grain_declarations.json"

#: The four passes of the local build, by their declared id.
DISTRICT_SOI = "historic_table_2_state_vs_district_file"
COUNTY_SOI = "state_soi_vs_county_file"
COUNTY_ACS_HOUSEHOLDS = "acs_households_state_is_the_county_sum"
COUNTY_POPULATION = "ladder_population_state_vs_county"

_RELATION_KINDS = ("subset", "independent")
_LOCAL_ROUTE_KINDS = ("no_higher_control",)
#: The derivations that the adapter can run for a derived control.
US_DERIVATIONS = ("sum_of_lower_rows",)
#: Every key of the file. A key outside this set would be read by nothing.
_KEYS = frozenset(
    {
        "schema_version",
        "description",
        "grain_precedence",
        "control_grains",
        "signature_fields",
        "bridges",
        "bridge_selectors",
        "derived_controls",
        "same_measurement",
        "partitions",
        "relations",
        "local_routes",
        "members",
        "registers",
        "until_the_feed_repin",
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
    _validate(payload)
    return payload


def _validate(payload: Mapping[str, Any]) -> None:
    unknown = sorted(set(payload) - _KEYS)
    if unknown:
        raise ValueError(
            f"US cross-grain declarations hold unknown key(s) {unknown}; "
            "nothing would read them."
        )
    # The US passes take their bridges from ``bridge_selectors`` and declare no
    # partition. A bridge by id or a partition written here would not run.
    for key in ("bridges", "partitions"):
        if payload[key]:
            raise ValueError(
                f"US cross-grain declarations hold {key!r} entries, and no US "
                "pass runs them yet; declare a bridge selector, or add the pass."
            )
    grains = tuple(payload["grain_precedence"])
    controls = tuple(payload["control_grains"])
    unknown_controls = [grain for grain in controls if grain not in grains]
    if unknown_controls:
        raise ValueError(
            f"US cross-grain control_grains {unknown_controls} are not in "
            "grain_precedence."
        )

    def nests(pass_id: str, higher: object, lower: object) -> None:
        if (
            higher not in controls
            or lower not in grains
            or grains.index(higher) >= grains.index(lower)
        ):
            raise ValueError(
                f"US cross-grain pass {pass_id!r} must name a control grain "
                f"above its lower grain, got {higher!r} over {lower!r}."
            )

    pass_ids: list[str] = []
    for selector in payload["bridge_selectors"]:
        nests(
            selector["bridge_id"],
            selector["higher"].get("grain"),
            selector["lower"].get("grain"),
        )
        if not selector["pair_on"] or not str(selector.get("reason", "")).strip():
            raise ValueError(
                f"US cross-grain bridge selector {selector['bridge_id']!r} must "
                "name what its sides are paired on, and give a reason."
            )
        pass_ids.append(selector["bridge_id"])
    for derived in payload["derived_controls"]:
        nests(derived["bridge_id"], derived["higher_grain"], derived["lower_grain"])
        if derived["derive"] not in US_DERIVATIONS:
            raise ValueError(
                f"US cross-grain derived control {derived['bridge_id']!r} names "
                f"unknown derivation {derived['derive']!r}; expected one of "
                f"{US_DERIVATIONS}."
            )
        pass_ids.append(derived["bridge_id"])
    for same in payload["same_measurement"]:
        nests(same["id"], same["higher_grain"], same["lower_grain"])
        pass_ids.append(same["id"])
    if len(set(pass_ids)) != len(pass_ids):
        raise ValueError(f"US cross-grain pass ids repeat: {sorted(pass_ids)}.")
    allocators = payload["until_the_feed_repin"]["allocators"]
    for pass_id, name in allocators.items():
        if pass_id not in pass_ids or name not in ALLOCATORS:
            raise ValueError(
                f"US cross-grain allocator {name!r} for {pass_id!r} must name a "
                f"declared pass and one of {sorted(ALLOCATORS)}."
            )
    for relation in payload["relations"]:
        targets = [str(target) for target in relation["targets"]]
        subset_sides = {str(relation.get("part")), str(relation.get("whole"))}
        if (
            len(targets) != 2
            or relation.get("relation") not in _RELATION_KINDS
            or (relation["relation"] == "subset" and subset_sides != set(targets))
            or not str(relation.get("reason", "")).strip()
        ):
            raise ValueError(
                f"US cross-grain relation over {targets} is malformed: it needs "
                f"two targets, one of {_RELATION_KINDS}, a part and a whole for "
                "a subset, and a reason."
            )
        levels = {
            target: grains.index(_relation_target(target, grains)[2])
            for target in targets
        }
        if relation["relation"] == "subset" and (
            levels[str(relation["part"])] <= levels[str(relation["whole"])]
        ):
            raise ValueError(
                f"US cross-grain relation over {targets} is malformed: the part "
                "of a subset is at a lower level than its whole."
            )
    for target, route in payload["local_routes"].items():
        _route_target(target, grains)
        if (
            route.get("route") not in _LOCAL_ROUTE_KINDS
            or not str(route.get("reason", "")).strip()
        ):
            raise ValueError(
                f"US cross-grain local route {target!r} must be one of "
                f"{_LOCAL_ROUTE_KINDS} with a reason."
            )
    transitional = payload["until_the_feed_repin"]
    for entry in (*transitional["compile_rules"], *transitional["known_gaps"]):
        levels = tuple(entry.get("levels") or ())
        if (
            not entry.get("id")
            or not entry.get("family")
            or not levels
            or any(level not in grains for level in levels)
            or entry.get("higher_level") not in grains
            or not str(entry.get("reason", "")).strip()
            or not entry.get("issues")
            or ("roles" in entry and not entry["roles"])
        ):
            raise ValueError(
                f"US cross-grain entry {entry.get('id')!r} must name a family, "
                "its levels, a reason and an issue."
            )
    for rule in transitional["compile_rules"]:
        if not rule.get("code") or not rule.get("marker"):
            raise ValueError(
                f"US cross-grain compile rule {rule['id']!r} must name its code "
                "and the marker it writes."
            )


def _relation_target(target: str, grains: tuple[str, ...]) -> tuple[str, str, str]:
    """``family.role@level`` as ``(family, role, level)``."""

    name, _, level = target.partition("@")
    family, _, role = name.partition(".")
    if not family or not role or level not in grains:
        raise ValueError(
            f"US cross-grain relation target {target!r} must read "
            "family.role@level, with a declared level."
        )
    return family, role, level


def _route_target(target: str, grains: tuple[str, ...]) -> tuple[str, str]:
    """``family@level`` as ``(family, level)``."""

    family, _, level = target.partition("@")
    if not family or level not in grains:
        raise ValueError(
            f"US cross-grain local route {target!r} must read family@level, "
            "with a declared level."
        )
    return family, level


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def _selector_entry(bridge_id: str) -> Mapping[str, Any]:
    for selector in load_us_cross_grain_declarations()["bridge_selectors"]:
        if selector["bridge_id"] == bridge_id:
            return selector
    raise ValueError(f"Unknown US cross-grain bridge selector {bridge_id!r}.")


def us_bridge_selector(
    bridge_id: str, *, bounded: bool = True
) -> CrossGrainBridgeSelector:
    """A declared selector as the operator takes it.

    ``bounded=False`` leaves the declared bound out, for a caller that turns
    the bound off.
    """

    declared = _selector_entry(bridge_id)
    bound = (
        {
            "max_factor_shift": declared.get("max_factor_shift"),
            "min_factor_shift": declared.get("min_factor_shift"),
            "shift_around": declared.get("shift_around", "one"),
            "out_of_bound": declared.get("out_of_bound", "refuse"),
        }
        if bounded and declared.get("max_factor_shift") is not None
        else {}
    )
    return CrossGrainBridgeSelector(
        bridge_id=str(declared["bridge_id"]),
        concept=str(declared["concept"]),
        higher=dict(declared["higher"]),
        lower=dict(declared["lower"]),
        pair_on=tuple(str(name) for name in declared["pair_on"]),
        paired_values=declared.get("paired_values"),
        **bound,
    )


def us_pass_grains(pass_id: str) -> tuple[str, str]:
    """``(higher grain, lower grain)`` of a declared pass."""

    declared = load_us_cross_grain_declarations()
    for selector in declared["bridge_selectors"]:
        if selector["bridge_id"] == pass_id:
            return str(selector["higher"]["grain"]), str(selector["lower"]["grain"])
    for entry in declared["derived_controls"]:
        if entry["bridge_id"] == pass_id:
            return str(entry["higher_grain"]), str(entry["lower_grain"])
    for entry in declared["same_measurement"]:
        if entry["id"] == pass_id:
            return str(entry["higher_grain"]), str(entry["lower_grain"])
    raise ValueError(f"Unknown US cross-grain pass {pass_id!r}.")


def us_derivation(pass_id: str) -> str:
    """The declared derivation of the control of a derived-control pass."""

    for entry in load_us_cross_grain_declarations()["derived_controls"]:
        if entry["bridge_id"] == pass_id:
            return str(entry["derive"])
    raise ValueError(f"Unknown US cross-grain derived control {pass_id!r}.")


def us_allocator_name(pass_id: str) -> str | None:
    """The allocator of a pass until the feed is re-pinned, if it has one."""

    return load_us_cross_grain_declarations()["until_the_feed_repin"]["allocators"].get(
        pass_id
    )


@dataclass(frozen=True)
class DistrictSoiNesting:
    """The declared choices of the state-to-district SOI pass."""

    vintage_rule: str
    max_factor_shift: float
    min_factor_shift: float
    #: District-file-only measure -> sibling measure, until the feed re-pin.
    level_bridges: Mapping[str, str]
    closure_rtol: float
    reviewed_crosswalk_sha256: str
    defective_lower_measures: Mapping[str, str]
    unparented_lower_measures: Mapping[str, str]
    excluded_legs: Mapping[str, str]

    @property
    def factor_band(self) -> float:
        """The bound as a ratio to the median: one plus the highest shift."""

        return 1.0 + self.max_factor_shift


@functools.lru_cache(maxsize=1)
def us_district_soi_nesting() -> DistrictSoiNesting:
    declared = load_us_cross_grain_declarations()
    selector = _selector_entry(DISTRICT_SOI)
    registers = declared["registers"]
    transitional = declared["until_the_feed_repin"]
    return DistrictSoiNesting(
        vintage_rule=str(selector["vintage_rule"]),
        max_factor_shift=selector["max_factor_shift"],
        min_factor_shift=selector["min_factor_shift"],
        level_bridges=dict(transitional["level_bridges"]["siblings"]),
        closure_rtol=transitional["closure_rtol"],
        reviewed_crosswalk_sha256=str(
            declared["members"]["congressional_district"]["reviewed_crosswalk_sha256"]
        ),
        defective_lower_measures=dict(registers["defective_lower_measures"]["entries"]),
        unparented_lower_measures=dict(
            registers["unparented_lower_measures"]["entries"]
        ),
        excluded_legs=dict(registers["excluded_legs"]["entries"]),
    )


@dataclass(frozen=True)
class CountySoiNesting:
    """The declared choices of the state-to-county SOI pass."""

    required_measures: tuple[str, ...]
    max_factor_shift: float
    min_factor_shift: float
    #: Absent county -> host county, as five-digit integer codes.
    combined_cells: Mapping[int, int]
    combined_cell_basis: str

    @property
    def factor_band(self) -> float:
        """The bound as a ratio to the median: one plus the highest shift."""

        return 1.0 + self.max_factor_shift


@functools.lru_cache(maxsize=1)
def us_county_soi_nesting() -> CountySoiNesting:
    declared = load_us_cross_grain_declarations()
    combined = [
        cell
        for cell in declared["members"]["county"]["combined"]
        if cell["bridge_id"] == COUNTY_SOI
    ]
    bases = {str(cell["basis"]) for cell in combined}
    if len(bases) > 1:
        raise ValueError("US combined county cells of one pass must share a basis.")
    selector = _selector_entry(COUNTY_SOI)
    return CountySoiNesting(
        required_measures=tuple(
            str(measure) for measure in selector["required_measures"]
        ),
        max_factor_shift=selector["max_factor_shift"],
        min_factor_shift=selector["min_factor_shift"],
        combined_cells={int(cell["absent"]): int(cell["host"]) for cell in combined},
        combined_cell_basis=next(iter(bases), ""),
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


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
    role_key: str = "target_role",
) -> list[dict[str, str]]:
    """Every gap between a set of compiled specs and the declarations.

    ``specs`` are compiled target specs (anything with ``family``, ``value``
    and ``metadata``). The top level of ``grain_precedence`` nests in
    nothing. A family at a lower level is covered by a bridge selector, a
    derived control, a same-measurement entry, a relation, a local route, a
    compile rule or a known gap. A violation is one of:

    * ``family_level_without_route``: a family has rows at a lower level
      that nothing covers;
    * ``role_at_two_levels_without_relation``: rows of one family and one
      role (``role_key``) are at the top level and at a lower level, and no
      relation names that role, nor a compile rule or a known gap that lists
      it under ``roles``. A family is covered as a whole, so without this a
      new role in a covered family would pass unseen;
    * ``stale_declaration``: a relation, a local route, a compile rule or a
      known gap names rows that the specs do not hold;
    * ``compile_rule_marker_absent``: a compile rule's marker is on no row of
      one of its levels;
    * ``subset_exceeded``: the parts of a ``subset`` relation sum to more
      than their whole, beyond the declared ``max_excess``;
    * ``subset_whole_not_one_positive_row``: the whole of a ``subset``
      relation is more than one row, or is not positive, so the sum of its
      parts cannot be compared with it.

    What the check does not read: a row with no ``level_key`` in its
    metadata. The rows that a pass makes (the ladder population, the county
    households) are such rows, and their pass declares them.

    Returns the violations. The caller decides what one means: this function
    refuses nothing.
    """

    declared = load_us_cross_grain_declarations()
    grains = tuple(declared["grain_precedence"])
    top = grains[0]
    rows: dict[tuple[str, str], list[Any]] = {}
    for spec in specs:
        level = spec.metadata.get(level_key)
        if level is not None:
            rows.setdefault((str(spec.family), str(level)), []).append(spec)
    compiled = {key for key in rows if key[1] != top}
    transitional = declared["until_the_feed_repin"]
    covered: set[tuple[str, str]] = set()
    for selector in declared["bridge_selectors"]:
        covered.add((str(selector["family"]), str(selector["lower"]["grain"])))
    for entry in (*declared["derived_controls"], *declared["same_measurement"]):
        covered.add((str(entry["family"]), str(entry["lower_grain"])))
    violations: list[dict[str, str]] = []

    def stale(family: str, level: str, where: str) -> None:
        violations.append(
            {
                "kind": "stale_declaration",
                "family": family,
                "level": level,
                "where": where,
            }
        )

    related: set[tuple[str, str]] = set()
    for relation in declared["relations"]:
        sides = {}
        for target in relation["targets"]:
            family, role, level = _relation_target(str(target), grains)
            related.add((family, role))
            held = [
                spec
                for spec in rows.get((family, level), ())
                if spec.metadata.get(role_key) == role
            ]
            sides[str(target)] = held
            if level != top:
                covered.add((family, level))
            if not held:
                stale(family, level, str(target))
        if relation["relation"] == "subset":
            part, whole = sides[str(relation["part"])], sides[str(relation["whole"])]
            if not part or not whole:
                continue  # A side with no row is a stale declaration, above.
            if len(whole) != 1 or not whole[0].value > 0:
                violations.append(
                    {
                        "kind": "subset_whole_not_one_positive_row",
                        "family": str(whole[0].family),
                        "level": str(whole[0].metadata.get(level_key)),
                        "where": f"{relation['whole']} holds {len(whole)} row(s)",
                    }
                )
                continue
            ratio = math.fsum(spec.value for spec in part) / whole[0].value
            if ratio > 1.0 + float(relation.get("max_excess", 0.0)):
                violations.append(
                    {
                        "kind": "subset_exceeded",
                        "family": str(part[0].family),
                        "level": str(part[0].metadata.get(level_key)),
                        "where": f"{relation['part']} is {ratio!r} of its whole",
                    }
                )
    for target in declared["local_routes"]:
        family, level = _route_target(str(target), grains)
        covered.add((family, level))
        if (family, level) not in rows:
            stale(family, level, str(target))
    for entry in (*transitional["compile_rules"], *transitional["known_gaps"]):
        for role in entry.get("roles", ()):
            related.add((str(entry["family"]), str(role)))
            if not any(
                spec.metadata.get(role_key) == role
                for level in entry["levels"]
                for spec in rows.get((str(entry["family"]), str(level)), ())
            ):
                stale(str(entry["family"]), "/".join(entry["levels"]), str(role))
        for level in entry["levels"]:
            key = (str(entry["family"]), str(level))
            covered.add(key)
            if key not in rows:
                stale(*key, str(entry["id"]))
            elif "marker" in entry and not any(
                _marker_matches(entry["marker"], spec.metadata) for spec in rows[key]
            ):
                violations.append(
                    {
                        "kind": "compile_rule_marker_absent",
                        "family": key[0],
                        "level": key[1],
                        "where": str(entry["id"]),
                    }
                )
    violations.extend(
        {
            "kind": "family_level_without_route",
            "family": family,
            "level": level,
            "where": "compiled specs",
        }
        for family, level in sorted(compiled - covered)
    )
    # A family is covered as a whole. A role that two levels measure must be
    # named: by a relation of its own, or by a compile rule or a known gap
    # that lists it.
    levels_of_role: dict[tuple[str, str], set[str]] = {}
    for (family, level), held in rows.items():
        for spec in held:
            role = spec.metadata.get(role_key)
            if role is not None:
                levels_of_role.setdefault((family, str(role)), set()).add(level)
    for (family, role), levels in sorted(levels_of_role.items()):
        if top in levels and len(levels) > 1 and (family, role) not in related:
            violations.extend(
                {
                    "kind": "role_at_two_levels_without_relation",
                    "family": family,
                    "level": level,
                    "where": f"{family}.{role}",
                }
                for level in sorted(levels - {top})
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
        "violations": violations,
        "covered": not violations,
    }
