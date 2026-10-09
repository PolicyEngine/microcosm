"""UK cross-grain declarations and the coverage check that enforces them.

``uk/cross_grain_declarations.json`` holds every country-owned declaration the
shared operator (:mod:`microcosm.build.cross_grain`) needs for the UK: grain
precedence, control grains, the bridges and partitions, and the reasons for
the targets that reach no control. microcosm#1123 made the declarations
exhaustive and enforced: :func:`uk_cross_grain_coverage_violations` refuses

* a local (constituency or local-authority) target with no route to a higher
  control: no national target with its exact measurement signature, no
  bridge, no partition and no declared ``local_routes`` reason;
* two targets at different grains whose measurements overlap (same concept,
  entity and mapping, one's filters a subset of the other's) and that are
  neither in one exact-signature group nor in one bridge nor declared under
  ``relations``;
* a declaration that names an unknown target or no longer applies.

Targets whose measurement block does not carry their population (the
binding's filters or groupby do) cannot be compared by signature and are
listed under ``signature_incomplete``; their relations are declared by hand.
``uk/target_doctrine_exceptions.json`` tolerates the gaps that still exist,
each naming the change that closes it; a tolerated entry that no longer
matches a violation is refused as stale.
"""

from __future__ import annotations

import functools
import itertools
import json
from collections.abc import Mapping
from importlib import resources as importlib_resources
from typing import Any

from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainPartition,
    _measurement_signature,
)

UK_CROSS_GRAIN_DECLARATIONS_RESOURCE = "cross_grain_declarations.json"
UK_TARGET_DOCTRINE_EXCEPTIONS_RESOURCE = "target_doctrine_exceptions.json"
UK_COVERAGE_VIOLATION_KINDS = frozenset(
    (
        "unknown_target",
        "unknown_local_route",
        "stale_local_route",
        "local_target_without_control",
        "stale_signature_incomplete",
        "undeclared_signature_incomplete",
        "malformed_relation",
        "undeclared_overlap",
    )
)
UK_LOCAL_GEOGRAPHY_LEVELS = frozenset({"constituency", "local_authority"})
_RELATION_KINDS = ("subset", "independent")
_LOCAL_ROUTE_KINDS = ("no_higher_control",)


def _load_resource(name: str) -> dict[str, Any]:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


@functools.lru_cache(maxsize=1)
def load_uk_cross_grain_declarations() -> Mapping[str, Any]:
    payload = _load_resource(UK_CROSS_GRAIN_DECLARATIONS_RESOURCE)
    if payload.get("schema_version") != 1:
        raise ValueError(
            "UK cross-grain declarations schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    return payload


@functools.lru_cache(maxsize=1)
def load_uk_target_doctrine_exceptions() -> Mapping[str, Any]:
    payload = _load_resource(UK_TARGET_DOCTRINE_EXCEPTIONS_RESOURCE)
    if payload.get("schema_version") != 1:
        raise ValueError(
            "UK target doctrine exceptions schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    return payload


def uk_cross_grain_bridges(
    declarations: Mapping[str, Any] | None = None,
) -> tuple[CrossGrainBridge, ...]:
    declared = declarations or load_uk_cross_grain_declarations()
    return tuple(
        CrossGrainBridge(
            bridge_id=str(entry["bridge_id"]),
            concept=str(entry["concept"]),
            higher_target_ids=tuple(str(value) for value in entry["higher_target_ids"]),
            lower_side=str(entry["lower_side"]),
        )
        for entry in declared["bridges"]
    )


def uk_cross_grain_partitions(
    declarations: Mapping[str, Any] | None = None,
) -> tuple[CrossGrainPartition, ...]:
    declared = declarations or load_uk_cross_grain_declarations()
    return tuple(
        CrossGrainPartition(
            partition_id=str(entry["partition_id"]),
            parent_target_id=str(entry["parent_target_id"]),
            member_target_ids=tuple(str(value) for value in entry["member_target_ids"]),
            kind=str(entry.get("kind", "exhaustive")),
        )
        for entry in declared.get("partitions", ())
    )


def uk_cross_grain_grain(
    metadata: Mapping[str, Any],
    geography_level: str,
    geography_id: str,
    *,
    declarations: Mapping[str, Any] | None = None,
) -> str:
    """The grain a national target cell sits at on the reconciliation surface.

    A declared ``cross_grain_grain`` stamp wins (the region-tier fan-outs put
    Wales, Scotland and Northern Ireland beside the nine English regions). A
    country-level cell on a nation code (England, Wales, Scotland, Northern
    Ireland) sits at the nation grain, between the UK/GB rows and the regions;
    every other cell keeps its geography level.
    """

    declared = declarations or load_uk_cross_grain_declarations()
    stamp = metadata.get("cross_grain_grain")
    if stamp:
        return str(stamp)
    if geography_level == "country" and geography_id in set(
        declared["nation_geography_ids"]
    ):
        return "nation"
    return geography_level


def _signature_fields(declarations: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(field) for field in declarations["signature_fields"])


def _incomplete_targets(declarations: Mapping[str, Any]) -> dict[str, str]:
    incomplete: dict[str, str] = {}
    for group in declarations.get("signature_incomplete", ()):
        for target_id in group["targets"]:
            incomplete[str(target_id)] = str(group["reason"])
    return incomplete


def measurement_signature_is_incomplete(target: Mapping[str, Any]) -> bool:
    """The binding narrows a population its measurement block leaves open."""

    measurement = target.get("measurement") or {}
    binding = (target.get("bindings") or {}).get("policyengine") or {}
    return not measurement.get("filters") and bool(
        binding.get("filters")
        or binding.get("household_conditions")
        or binding.get("groupby_variable")
    )


def _filter_set(target: Mapping[str, Any]) -> frozenset[str]:
    measurement = target.get("measurement") or {}
    return frozenset(
        json.dumps(predicate, sort_keys=True)
        for predicate in measurement.get("filters") or ()
    )


def uk_cross_grain_overlap_candidates(
    contract: Mapping[str, Mapping[str, Any]],
    declarations: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Target pairs at different grains whose measured populations overlap.

    Same concept, entity and mapping, and one target's filter set a subset of
    the other's. Pairs in one exact-signature group or one bridge are already
    reconciled and are left out; so are targets declared signature-incomplete.
    """

    bridged: dict[str, set[str]] = {}
    for bridge in uk_cross_grain_bridges(declarations):
        for side in (*bridge.higher_target_ids, bridge.lower_side):
            bridged.setdefault(side.removeprefix("contract:"), set()).add(
                bridge.bridge_id
            )
    for entry in declarations.get("band_bridges", ()):
        for side in (entry["higher_target_id"], *entry["lower_target_ids"]):
            bridged.setdefault(str(side), set()).add(str(entry["bridge_id"]))
    incomplete = _incomplete_targets(declarations)
    by_measure: dict[tuple[Any, ...], list[str]] = {}
    for target_id, target in contract.items():
        if target_id in incomplete:
            continue
        measurement = target.get("measurement") or {}
        key = (
            measurement.get("concept"),
            measurement.get("entity"),
            measurement.get("map_to"),
        )
        by_measure.setdefault(key, []).append(target_id)
    pairs: list[tuple[str, str]] = []
    for target_ids in by_measure.values():
        for left, right in itertools.combinations(sorted(target_ids), 2):
            left_levels = set(contract[left].get("geography_levels") or ())
            right_levels = set(contract[right].get("geography_levels") or ())
            if left_levels == right_levels:
                continue
            left_filters = _filter_set(contract[left])
            right_filters = _filter_set(contract[right])
            if left_filters == right_filters:
                continue
            if not (left_filters <= right_filters or right_filters <= left_filters):
                continue
            if bridged.get(left, set()) & bridged.get(right, set()):
                continue
            pairs.append((left, right))
    return pairs


def uk_cross_grain_coverage_violations(
    contract: Mapping[str, Mapping[str, Any]],
    declarations: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Every coverage defect of the declarations against the contract."""

    declared = declarations or load_uk_cross_grain_declarations()
    fields = _signature_fields(declared)
    violations: list[dict[str, str]] = []

    def known(target_id: str, where: str) -> bool:
        if target_id in contract:
            return True
        violations.append(
            {"kind": "unknown_target", "target_id": target_id, "where": where}
        )
        return False

    bridges = uk_cross_grain_bridges(declared)
    partitions = uk_cross_grain_partitions(declared)
    bridge_lower = set()
    for bridge in bridges:
        for side in (*bridge.higher_target_ids, bridge.lower_side):
            known(side.removeprefix("contract:"), f"bridge {bridge.bridge_id}")
        bridge_lower.add(bridge.lower_side.removeprefix("contract:"))
    for entry in declared.get("band_bridges", ()):
        for target_id in (entry["higher_target_id"], *entry["lower_target_ids"]):
            known(str(target_id), f"band bridge {entry['bridge_id']}")
    partition_members = set()
    for partition in partitions:
        for target_id in (partition.parent_target_id, *partition.member_target_ids):
            known(target_id, f"partition {partition.partition_id}")
        partition_members.update(partition.member_target_ids)

    incomplete_ids = set(_incomplete_targets(declared))
    national_signatures = {
        _measurement_signature(target, fields)
        for target_id, target in contract.items()
        if set(target.get("geography_levels") or ()) - UK_LOCAL_GEOGRAPHY_LEVELS
        and target_id not in incomplete_ids
    }
    local_routes = declared.get("local_routes") or {}
    for target_id, route in local_routes.items():
        known(target_id, "local_routes")
        if route.get("route") not in _LOCAL_ROUTE_KINDS:
            violations.append(
                {
                    "kind": "unknown_local_route",
                    "target_id": target_id,
                    "where": str(route.get("route")),
                }
            )
    for target_id, target in sorted(contract.items()):
        levels = set(target.get("geography_levels") or ())
        if not levels & UK_LOCAL_GEOGRAPHY_LEVELS:
            continue
        covered = (
            _measurement_signature(target, fields) in national_signatures
            or target_id in bridge_lower
            or target_id in partition_members
        )
        if covered and target_id in local_routes:
            violations.append(
                {
                    "kind": "stale_local_route",
                    "target_id": target_id,
                    "where": "local_routes",
                }
            )
        elif not covered and target_id not in local_routes:
            violations.append(
                {
                    "kind": "local_target_without_control",
                    "target_id": target_id,
                    "where": "contract",
                }
            )

    incomplete = _incomplete_targets(declared)
    for target_id in incomplete:
        if known(target_id, "signature_incomplete") and not (
            measurement_signature_is_incomplete(contract[target_id])
        ):
            violations.append(
                {
                    "kind": "stale_signature_incomplete",
                    "target_id": target_id,
                    "where": "signature_incomplete",
                }
            )
    for target_id, target in sorted(contract.items()):
        if measurement_signature_is_incomplete(target) and target_id not in incomplete:
            violations.append(
                {
                    "kind": "undeclared_signature_incomplete",
                    "target_id": target_id,
                    "where": "contract",
                }
            )

    declared_pairs: set[tuple[str, str]] = set()
    for relation in declared.get("relations", ()):
        targets = [str(value) for value in relation["targets"]]
        if len(targets) != 2 or relation.get("relation") not in _RELATION_KINDS:
            violations.append(
                {
                    "kind": "malformed_relation",
                    "target_id": ",".join(targets),
                    "where": str(relation.get("relation")),
                }
            )
            continue
        for target_id in targets:
            known(target_id, "relations")
        declared_pairs.add(tuple(sorted(targets)))
    for left, right in uk_cross_grain_overlap_candidates(contract, declared):
        if (left, right) not in declared_pairs:
            violations.append(
                {
                    "kind": "undeclared_overlap",
                    "target_id": f"{left},{right}",
                    "where": "contract",
                }
            )
    return violations


def assert_uk_cross_grain_coverage(
    contract: Mapping[str, Mapping[str, Any]],
    *,
    declarations: Mapping[str, Any] | None = None,
    exceptions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse an undeclared or stale coverage gap; return the tolerated ones."""

    violations = uk_cross_grain_coverage_violations(contract, declarations)
    # The ledger also tolerates uprating holds (``undeclared_hold``), which
    # ``uk_runtime.uprating_holds`` checks; coverage reads only its own kinds.
    tolerated_entries = [
        entry
        for entry in (exceptions or load_uk_target_doctrine_exceptions())["entries"]
        if entry["kind"] in UK_COVERAGE_VIOLATION_KINDS
    ]
    tolerated = {(entry["kind"], entry["target_id"]) for entry in tolerated_entries}
    found = {(violation["kind"], violation["target_id"]) for violation in violations}
    untolerated = sorted(found - tolerated)
    stale = sorted(tolerated - found)
    if untolerated or stale:
        raise ValueError(
            "UK cross-grain coverage refused: undeclared gap(s) "
            f"{untolerated}; stale doctrine exception(s) {stale}. Declare the "
            "route in uk/cross_grain_declarations.json or retire the exception."
        )
    return {
        "violations_tolerated": [
            entry
            for entry in tolerated_entries
            if (entry["kind"], entry["target_id"]) in found
        ],
    }
