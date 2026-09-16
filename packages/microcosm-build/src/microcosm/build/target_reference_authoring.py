"""Author Ledger target-reference resources from country contracts.

Country packages own target declarations and country-specific pinning policy.
This module owns the shared mechanics: candidate reference construction,
fan-out over native dimension values, compilation through the real Ledger
resolver, and membership reporting.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from microcosm.build.ledger_targets import (  # pyright: ignore[reportPrivateUsage]
    LedgerHierarchyMetadataError,
    LedgerTargetReference,
    _assertion_allowed,
    _fact_matches_selector,
    _not_after_target_period,
    _period_key,
    _period_key_from_value,
    compile_ledger_target_references,
    reference_fact_selectors,
)

if TYPE_CHECKING:
    from microcosm.calibrate import TargetRegistry

FanoutName = Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]
#: Country-supplied application of a declared ``uprating_index``: takes the
#: compiled reference and its registry, returns the registry with the value
#: transported to the reference period and the factor declared on the spec
#: metadata. The authoring records the applied value in the membership so the
#: committed surface, the parity receipts and the runtime carry one value.
UpratingApplier = Callable[
    ["LedgerTargetReference", "TargetRegistry"], "TargetRegistry"
]
GeographyPin = Mapping[str, str]
#: ``(target, geography_level, geography_id, entity) -> extra reference metadata``
#: for one geography fan-out row. Values are strings (reference metadata is a
#: string mapping); a country encodes structured values, such as a predicate,
#: as JSON.
GeographyFanoutMetadata = Callable[
    [Mapping[str, Any], str, str, str], Mapping[str, str]
]
#: Ordered ``(geography_level, geography_id)`` pairs one target fans out over.
GeographyFanout = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AuthoredTargetReferences:
    """Generated references and their compile-membership report."""

    references: tuple[dict[str, Any], ...]
    membership_report: dict[str, Any]

    @property
    def status_counts(self) -> Mapping[str, int]:
        """Membership status counts keyed by report status."""

        return self.membership_report["status_counts"]


@dataclass(frozen=True)
class TargetReferenceAuthoringConfig:
    """Country-supplied authoring policy."""

    target_period: int | str
    geography_pins: Mapping[str, GeographyPin] = field(default_factory=dict)
    fanout_name: FanoutName | None = None
    sum_target_ids: frozenset[str] = frozenset()
    value_operation_by_target_id: Mapping[str, str] = field(default_factory=dict)
    selector_pins_by_target_id: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    reference_metadata_by_target_id: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    signed_exclusions_by_target_id: Mapping[str, str] = field(default_factory=dict)
    #: Row-level sign-outs inside a dimension fan-out: ``target_id -> {
    #: (dimension, json-encoded value): rationale }``. The matching fan-out
    #: row is recorded as ``signed_excluded`` and never compiled, while the
    #: target's other rows stay active. A declared row that no fan-out row
    #: matches is stale and refuses the run.
    signed_row_exclusions_by_target_id: Mapping[str, Mapping[tuple[str, str], str]] = (
        field(default_factory=dict)
    )
    #: Targets that fan out over a declared geography roster instead of pinning
    #: one geography: one reference per ``(geography_level, geography_id)``,
    #: named ``target_id@geography_id`` with its own measure column, the same
    #: way area-grain references are authored. A target listed here must not
    #: also carry a geography pin. Every roster cell must compile; an absent
    #: fact refuses the run rather than deferring silently.
    geography_fanout_by_target_id: Mapping[str, GeographyFanout] = field(
        default_factory=dict
    )
    #: Optional per-row metadata for geography fan-out rows (for example the
    #: predicate that scopes the measure to the row's geography).
    geography_fanout_metadata: GeographyFanoutMetadata | None = None
    #: Fan-out cells composed from lower-grain member facts instead of a fact
    #: of their own: ``target_id -> cell geography_id -> (member_level,
    #: member_geography_ids)``. The cell keeps its own geography in the row's
    #: metadata (and ``composed_from_level``), while its selector names the
    #: member level and the member ids, and each aggregating operand takes
    #: ``expected_member_count`` = the member count. A publisher that prints no
    #: row for the cell (English regions in the MHCLG council taxbase,
    #: microcosm#929) is bound as a consumer rollup of the rows it does print.
    geography_composition_by_target_id: Mapping[
        str, Mapping[str, tuple[str, tuple[str, ...]]]
    ] = field(default_factory=dict)
    #: Publisher recodings among composition members: ``member_level ->
    #: roster code -> alias codes`` (the same declaration the area rosters
    #: carry as ``code_aliases``), so a member list can name both spellings
    #: while the expected member count stays one per authority.
    geography_composition_aliases: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    binding_vocabulary: frozenset[str] = frozenset()
    source_fact_feed: str = ""
    uprating_appliers: Mapping[str, UpratingApplier] = field(default_factory=dict)


@dataclass(frozen=True)
class AreaSignedDeferral:
    """Signed area-level compile deferral for one contract target.

    Missing cells are the default use. ``defer_if_compiles`` is an explicit
    opt-in for a separate adjudication that prevents an available fact from
    binding; ordinary deferrals remain stale once their cell compiles.
    """

    target_id: str
    geography_level: str
    reason_id: str
    rationale: str
    area_ids: tuple[str, ...]
    defer_if_compiles: bool = False

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("AreaSignedDeferral.target_id must be non-empty.")
        if not self.geography_level:
            raise ValueError("AreaSignedDeferral.geography_level must be non-empty.")
        if not self.reason_id:
            raise ValueError("AreaSignedDeferral.reason_id must be non-empty.")
        if not self.rationale:
            raise ValueError("AreaSignedDeferral.rationale must be non-empty.")
        if not self.area_ids:
            raise ValueError("AreaSignedDeferral.area_ids must be non-empty.")
        object.__setattr__(
            self,
            "area_ids",
            tuple(dict.fromkeys(str(area_id) for area_id in self.area_ids)),
        )


@dataclass(frozen=True)
class AreaTargetReferenceAuthoringConfig:
    """Country-supplied authoring policy for area-grain targets."""

    target_period: int | str
    areas_by_geography_level: Mapping[str, Iterable[str]]
    area_signed_deferrals: tuple[AreaSignedDeferral, ...] = ()
    value_operation_by_target_id: Mapping[str, str] = field(default_factory=dict)
    selector_pins_by_target_id: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    reference_metadata_by_target_id: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    #: Per-target area rosters narrower than the shared roster:
    #: ``target_id -> geography_level -> area ids``. A target scoped this way
    #: authors cells only for the listed areas (a publication that covers one
    #: nation, microcosm#929); areas outside the scope are neither candidates
    #: nor deferrals, and the membership report records the scope.
    area_scope_by_target_id: Mapping[str, Mapping[str, frozenset[str]]] = field(
        default_factory=dict
    )
    #: Publisher recodings of roster areas: ``geography_level -> area_id ->
    #: alias codes``. A cell keeps the roster code as its identity and selects
    #: facts under the roster code or any alias; the row's metadata records the
    #: aliases so the surface can accept a fact stamped with one.
    area_id_aliases: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    binding_vocabulary: frozenset[str] = frozenset()
    source_fact_feed: str = ""

    def normalized_areas(self) -> dict[str, tuple[str, ...]]:
        """Return area rosters with duplicate ids removed in declared order."""

        rosters: dict[str, tuple[str, ...]] = {}
        for level, area_ids in self.areas_by_geography_level.items():
            values = tuple(dict.fromkeys(str(area_id) for area_id in area_ids))
            if not values:
                raise ValueError(
                    f"Area roster for geography level {level!r} must be non-empty."
                )
            rosters[str(level)] = values
        return rosters


def _validated_reference_targets(
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Require every declared target to resolve its value from Chronicle."""

    selected: list[Mapping[str, Any]] = []
    for target in contract.get("targets", ()):
        if not isinstance(target, Mapping):
            raise ValueError("Every target declaration must be a mapping.")
        target_id = str(target.get("target_id") or "").strip()
        if "materialization" in target:
            raise ValueError(
                f"Contract target {target_id!r} declares unsupported "
                "materialization; target values must resolve from Chronicle."
            )
        selector = target.get("ledger_selector")
        if not isinstance(selector, Mapping) or not selector:
            raise ValueError(
                f"Contract target {target_id!r} must declare a non-empty "
                "ledger_selector."
            )
        selected.append(target)
    return tuple(selected)


def author_target_references(
    contract: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
    config: TargetReferenceAuthoringConfig,
) -> AuthoredTargetReferences:
    """Build active target references and a membership report."""

    fact_rows = tuple(facts)
    reference_targets = _validated_reference_targets(contract)
    _validate_hierarchy_contract(contract)
    facts_by_source = _facts_by_source(fact_rows)
    _validate_contract_bindings(contract, config.binding_vocabulary)
    active_rows: list[dict[str, Any]] = []
    target_entries: dict[str, Any] = {}
    geography_pin_report: dict[str, Any] = {}
    genuine_sum_residue: list[str] = []
    uprating_holds: list[dict[str, str]] = []

    for target in reference_targets:
        target_id = str(target["target_id"])
        pin = config.geography_pins.get(target_id, {})
        geography_fanout = config.geography_fanout_by_target_id.get(target_id)
        if geography_fanout:
            if pin:
                raise ValueError(
                    f"target {target_id!r} declares both a geography pin "
                    f"{dict(pin)!r} and a geography fan-out; a fanned-out "
                    "target takes its geography from the roster only."
                )
            geography_pin_report[target_id] = {
                "geography_fanout": [
                    {"geography_level": str(level), "geography_id": str(area_id)}
                    for level, area_id in geography_fanout
                ]
            }
            composition = config.geography_composition_by_target_id.get(target_id)
            if composition:
                geography_pin_report[target_id]["geography_composition"] = {
                    str(area_id): {
                        "member_level": str(member_level),
                        "member_count": _composition_member_count(
                            config,
                            str(member_level),
                            tuple(str(member) for member in member_ids),
                        ),
                    }
                    for area_id, (member_level, member_ids) in sorted(
                        composition.items()
                    )
                }
        else:
            geography_pin_report[target_id] = dict(pin)
        source_facts = _source_prefilter(fact_rows, facts_by_source, target)
        selector = _target_selector(target, pin, config)
        signed_exclusion = config.signed_exclusions_by_target_id.get(target_id)
        if signed_exclusion is not None:
            matched = [
                fact for fact in source_facts if _fact_matches_selector(fact, selector)
            ]
            eligible = [
                fact
                for fact in matched
                if _not_after_target_period(
                    _period_key(fact),
                    _period_key_from_value(config.target_period),
                )
            ]
            target_entries[target_id] = {
                "status": "signed_excluded",
                "candidates": [
                    {
                        "name": target_id,
                        "status": "signed_excluded",
                        "matched_fact_count_overall": len(matched),
                        "matched_fact_count_at_or_before_period": len(eligible),
                        "signed_rationale": signed_exclusion,
                    }
                ],
            }
            continue
        candidates = tuple(
            _candidate_rows(
                target,
                source_facts,
                pin,
                config,
                hierarchy_catalog=contract["hierarchy"],
            )
        )
        candidate_entries: list[dict[str, Any]] = []
        row_exclusions = dict(
            config.signed_row_exclusions_by_target_id.get(target_id, {})
        )
        unmatched_row_exclusions = set(row_exclusions)
        for row in candidates:
            signed_row = _signed_row_exclusion(row, row_exclusions)
            if signed_row is not None:
                row_key, rationale = signed_row
                unmatched_row_exclusions.discard(row_key)
                matched = [
                    fact
                    for fact in source_facts
                    if _fact_matches_selector(fact, row["ledger_selector"])
                ]
                candidate_entries.append(
                    {
                        "name": row["name"],
                        "status": "signed_excluded",
                        "matched_fact_count_overall": len(matched),
                        "matched_fact_count_at_or_before_period": len(
                            [
                                fact
                                for fact in matched
                                if _not_after_target_period(
                                    _period_key(fact),
                                    _period_key_from_value(config.target_period),
                                )
                            ]
                        ),
                        "signed_row": {"dimension": row_key[0], "value": row_key[1]},
                        "signed_rationale": rationale,
                    }
                )
                continue
            reference = LedgerTargetReference(**row)
            matched = [
                fact
                for fact in source_facts
                if _fact_matches_selector(fact, row["ledger_selector"])
            ]
            eligible = [fact for fact in matched if _eligible_fact(reference, fact)]
            entry: dict[str, Any] = {
                "name": row["name"],
                "status": "",
                "matched_fact_count_overall": len(matched),
                "matched_fact_count_at_or_before_period": len(eligible),
            }
            row_geography_level = str(row["metadata"].get("geography_level") or "")
            if row_geography_level:
                entry["geography_level"] = row_geography_level
                entry["geography_id"] = str(row["metadata"].get("geography_id") or "")
            if reference.period_match_policy == "source_window":
                entry["matched_fact_count_in_source_window"] = sum(
                    _assertion_allowed(reference, fact) for fact in matched
                )
            compile_facts = matched
            operand_selectors = reference_fact_selectors(reference)[1:]
            if operand_selectors:
                # A ratio-scaled cell resolves its quotient from facts the
                # row selector never matches (national rows for a region
                # cell), so the compile sees the source's facts that match
                # any operand selector as well.
                compile_facts = [
                    fact
                    for fact in source_facts
                    if fact in matched
                    or any(
                        _fact_matches_selector(fact, selector)
                        for selector in operand_selectors
                    )
                ]
            try:
                registry = compile_ledger_target_references(
                    compile_facts,
                    [reference],
                    country=str(contract["country"]),
                )
            except LedgerHierarchyMetadataError:
                raise
            except ValueError as error:
                entry["status"] = _classify_deferral(error, matched, eligible)
                entry["error"] = _compact_compile_error(entry["status"])
                if geography_fanout:
                    # A roster cell that does not compile is a hole in a
                    # declared partition, not a deferral: refuse, the way the
                    # area-grain authoring refuses an unsigned absence.
                    raise ValueError(
                        f"Unsigned geography fan-out absence for {target_id!r} "
                        f"at {entry.get('geography_level')!r}/"
                        f"{entry.get('geography_id')!r}: {entry['status']}."
                    ) from error
            else:
                spec = registry.specs[0]
                resolved_period = spec.metadata.get("ledger_fact_period", "")
                _apply_uprating_hold(row, resolved_period, config.target_period)
                uprating: dict[str, str] = {}
                if row.get("uprating_index") is not None:
                    registry = _apply_declared_uprating(row, registry, config)
                    applied = registry.specs[0]
                    uprating = {
                        "index": str(row["uprating_index"]),
                        "factor": str(applied.metadata["uprating_factor"]),
                        "value_before_uprating": str(spec.value),
                    }
                    spec = applied
                if "uprating_from_period" in row:
                    uprating_holds.append(
                        {
                            "name": row["name"],
                            "from": str(row["uprating_from_period"]),
                            "to": str(row["uprating_to_period"]),
                            **uprating,
                        }
                    )
                if row.get("value_operation") == "sum":
                    genuine_sum_residue.append(row["name"])
                active_rows.append(row)
                entry.update(
                    {
                        "status": "active",
                        "resolved_period": spec.period,
                        "resolved_value": spec.value,
                        "resolved_fact_period": resolved_period,
                        "resolved_fact_key": _json_safe_ledger_id(
                            spec.metadata.get("ledger_aggregate_fact_key", "")
                        ),
                    }
                )
                if uprating:
                    entry["uprating"] = uprating
            candidate_entries.append(entry)
        if unmatched_row_exclusions:
            stale = sorted(
                f"{dimension}={value}" for dimension, value in unmatched_row_exclusions
            )
            raise ValueError(
                f"Stale row-level signed exclusion for {target_id!r}: no fan-out "
                f"row carries {stale!r}."
            )
        target_entries[target_id] = {
            "status": _target_status(candidate_entries),
            "candidates": candidate_entries,
        }

    status_counts = Counter(
        entry["status"]
        for target in target_entries.values()
        for entry in target["candidates"]
    )
    report = {
        "source_fact_feed": config.source_fact_feed,
        "target_period": config.target_period,
        "candidate_count": sum(
            len(target["candidates"]) for target in target_entries.values()
        ),
        "contract_target_count": len(reference_targets),
        "active_reference_count": len(active_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "geography_pins": geography_pin_report,
        "genuine_sum_residue": sorted(genuine_sum_residue),
        "uprating_holds": uprating_holds,
        "targets": target_entries,
    }
    return AuthoredTargetReferences(tuple(active_rows), report)


def author_area_target_references(
    contract: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
    config: AreaTargetReferenceAuthoringConfig,
) -> AuthoredTargetReferences:
    """Build area-grain target references over a declared geography roster."""

    fact_rows = tuple(facts)
    reference_targets = _validated_reference_targets(contract)
    _validate_hierarchy_contract(contract)
    areas_by_level = config.normalized_areas()
    signed = _area_deferral_index(config, contract, areas_by_level)
    facts_by_area = _facts_by_geography_id(fact_rows)
    _validate_contract_bindings(contract, config.binding_vocabulary)

    active_rows: list[dict[str, Any]] = []
    target_entries: dict[str, Any] = {}
    uprating_holds: list[dict[str, str]] = []

    for target in reference_targets:
        target_id = str(target["target_id"])
        level_entries: dict[str, Any] = {}
        for geography_level in target.get("geography_levels", ()):
            geography_level = str(geography_level)
            area_ids = areas_by_level.get(geography_level)
            if area_ids is None:
                # A declared level the roster lacks must refuse, not drop out:
                # skipping here yields no candidates, no references, and a
                # membership status of not_applicable -- a whole level could
                # leave the surface with the report agreeing nothing is wrong
                # (PR #795 round-3 review finding 2).
                raise ValueError(
                    f"target {target_id!r} declares geography level "
                    f"{geography_level!r}, which the area roster does not "
                    "carry."
                )
            scope = config.area_scope_by_target_id.get(target_id, {}).get(
                geography_level
            )
            if scope is not None:
                unknown = sorted(set(scope) - set(area_ids))
                if unknown:
                    raise ValueError(
                        f"target {target_id!r} area scope at {geography_level!r} "
                        f"names areas the roster does not carry: {unknown!r}."
                    )
                area_ids = tuple(area_id for area_id in area_ids if area_id in scope)
                if not area_ids:
                    raise ValueError(
                        f"target {target_id!r} area scope at {geography_level!r} "
                        "leaves no roster area."
                    )
            candidates: list[dict[str, Any]] = []
            for area_id in area_ids:
                key = (target_id, geography_level, area_id)
                selector = _area_target_selector(
                    target,
                    geography_level=geography_level,
                    area_id=area_id,
                    config=config,
                )
                row = _area_reference_row(
                    target,
                    selector,
                    geography_level=geography_level,
                    area_id=area_id,
                    config=config,
                    hierarchy_catalog=contract["hierarchy"],
                )
                # A recoded authority's facts are indexed under the alias
                # code; the cell's pool carries both spellings.
                aliases = config.area_id_aliases.get(geography_level, {}).get(
                    area_id, ()
                )
                source_facts = (
                    *facts_by_area.get(area_id, ()),
                    *(
                        fact
                        for alias in aliases
                        for fact in facts_by_area.get(alias, ())
                    ),
                )
                matched = [
                    fact
                    for fact in source_facts
                    if _fact_matches_selector(fact, selector)
                ]
                reference = LedgerTargetReference(**row)
                eligible = [fact for fact in matched if _eligible_fact(reference, fact)]
                signed_deferral = signed.get(key)
                entry: dict[str, Any] = {
                    "name": row["name"],
                    "target_id": target_id,
                    "geography_level": geography_level,
                    "geography_id": area_id,
                    "status": "",
                    "matched_fact_count_overall": len(matched),
                    "matched_fact_count_at_or_before_period": len(eligible),
                }
                try:
                    registry = compile_ledger_target_references(
                        matched,
                        [reference],
                        country=str(contract["country"]),
                    )
                except LedgerHierarchyMetadataError:
                    raise
                except ValueError as error:
                    status = _classify_area_deferral(error, matched, eligible)
                    entry["status"] = status
                    entry["error"] = _compact_compile_error(status)
                    if signed_deferral is None:
                        raise ValueError(
                            "Unsigned local target absence for "
                            f"{target_id!r} at {geography_level!r}/{area_id!r}: "
                            f"{status}."
                        ) from error
                    entry["signed_reason_id"] = signed_deferral.reason_id
                    entry["signed_rationale"] = signed_deferral.rationale
                else:
                    if (
                        signed_deferral is not None
                        and not signed_deferral.defer_if_compiles
                    ):
                        raise ValueError(
                            "Stale area signed deferral for "
                            f"{target_id!r} at {geography_level!r}/{area_id!r}: "
                            "the candidate now compiles."
                        )
                    if signed_deferral is not None:
                        entry.update(
                            {
                                "status": "signed_deferred",
                                "signed_reason_id": signed_deferral.reason_id,
                                "signed_rationale": signed_deferral.rationale,
                            }
                        )
                        candidates.append(entry)
                        continue
                    spec = registry.specs[0]
                    resolved_period = str(spec.metadata.get("ledger_fact_period", ""))
                    _apply_uprating_hold(row, resolved_period, config.target_period)
                    if "uprating_from_period" in row:
                        uprating_holds.append(
                            {
                                "name": row["name"],
                                "target_id": target_id,
                                "geography_level": geography_level,
                                "geography_id": area_id,
                                "from": str(row["uprating_from_period"]),
                                "to": str(row["uprating_to_period"]),
                            }
                        )
                    active_rows.append(row)
                    entry.update(
                        {
                            "status": "active",
                            "resolved_period": spec.period,
                            "resolved_value": spec.value,
                            "resolved_fact_period": resolved_period,
                            "resolved_fact_key": _json_safe_ledger_id(
                                spec.metadata.get("ledger_aggregate_fact_key", "")
                            ),
                        }
                    )
                candidates.append(entry)
            level_entries[geography_level] = {
                "status": _target_status(candidates),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        target_entries[target_id] = {
            "status": _target_status(
                [
                    entry
                    for level in level_entries.values()
                    for entry in level["candidates"]
                ]
            ),
            "geography_levels": level_entries,
        }

    status_counts = Counter(
        entry["status"]
        for target in target_entries.values()
        for level in target["geography_levels"].values()
        for entry in level["candidates"]
    )
    report = {
        "source_fact_feed": config.source_fact_feed,
        "target_period": config.target_period,
        "candidate_count": sum(
            level["candidate_count"]
            for target in target_entries.values()
            for level in target["geography_levels"].values()
        ),
        "contract_target_count": len(reference_targets),
        "active_reference_count": len(active_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "areas_by_geography_level": {
            level: list(area_ids) for level, area_ids in areas_by_level.items()
        },
        "area_scope_by_target_id": {
            target_id: {
                level: sorted(area_ids) for level, area_ids in sorted(scopes.items())
            }
            for target_id, scopes in sorted(config.area_scope_by_target_id.items())
        },
        "signed_deferrals": [
            {
                "target_id": deferral.target_id,
                "geography_level": deferral.geography_level,
                "reason_id": deferral.reason_id,
                "rationale": deferral.rationale,
                "area_ids": list(deferral.area_ids),
                **({"defer_if_compiles": True} if deferral.defer_if_compiles else {}),
            }
            for deferral in config.area_signed_deferrals
        ],
        "uprating_holds": uprating_holds,
        "holds_by_target": dict(
            sorted(Counter(row["target_id"] for row in uprating_holds).items())
        ),
        "targets": target_entries,
    }
    return AuthoredTargetReferences(tuple(active_rows), report)


def target_references_resource(
    *,
    country: str,
    description: str,
    authored: AuthoredTargetReferences,
    hierarchy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a serialisable ``target_references.json`` resource."""

    normalized_references = []
    target_categories: dict[str, str] = {}
    target_labels: dict[str, str] = {}
    for reference in authored.references:
        seed = reference.get("hierarchy")
        if not isinstance(seed, Mapping):
            raise ValueError(
                f"Authored reference {reference.get('name')!r} has no hierarchy seed."
            )
        category = seed.get("category")
        if not isinstance(category, Mapping) or not category.get("id"):
            raise ValueError(
                f"Authored reference {reference.get('name')!r} has no category id."
            )
        contract_target_id = str(
            (reference.get("metadata") or {}).get("contract_target_id") or ""
        )
        if not contract_target_id:
            raise ValueError(
                f"Authored reference {reference.get('name')!r} has no contract target id."
            )
        category_id = str(category["id"])
        previous = target_categories.setdefault(contract_target_id, category_id)
        if previous != category_id:
            raise ValueError(
                f"Contract target {contract_target_id!r} resolves to conflicting "
                f"hierarchy categories {previous!r} and {category_id!r}."
            )
        target_label = str(seed.get("target_label") or "").strip()
        if target_label:
            previous_label = target_labels.setdefault(contract_target_id, target_label)
            if previous_label != target_label:
                raise ValueError(
                    f"Contract target {contract_target_id!r} resolves to conflicting "
                    f"target labels {previous_label!r} and {target_label!r}."
                )
        normalized_references.append(
            {key: value for key, value in reference.items() if key != "hierarchy"}
        )
    return {
        "schema_version": 2,
        "country": country,
        "description": description,
        "hierarchy": {
            **dict(hierarchy),
            "target_categories": dict(sorted(target_categories.items())),
            "target_labels": dict(sorted(target_labels.items())),
        },
        "allowed_value_operations": [
            "identity",
            "sum",
            "difference",
            "calendar_year_average",
            "latest_plateau",
            "count_x_mean",
            "scaled_by_ratio",
            *sorted(
                {
                    row["value_operation"]
                    for row in authored.references
                    if row.get("value_operation")
                    in {
                        "monthly_window_average",
                        "monthly_window_sum_average",
                        "linear_combination",
                    }
                }
            ),
        ],
        "target_references": normalized_references,
    }


def _validate_contract_bindings(
    contract: Mapping[str, Any],
    binding_vocabulary: frozenset[str],
) -> None:
    if not binding_vocabulary:
        return
    for target in contract.get("targets", ()):
        binding = target["bindings"]["policyengine"]
        unknown = sorted(set(binding) - binding_vocabulary)
        if unknown:
            raise ValueError(
                f"Contract target {target['target_id']!r} carries unknown "
                f"policyengine binding keys {unknown!r}."
            )


def _validate_hierarchy_contract(contract: Mapping[str, Any]) -> None:
    """Require each flat target declaration to reference one catalog category."""

    if contract.get("schema_version") != 2:
        raise ValueError(
            "Calibration target contracts must use schema_version 2 to declare "
            "their normalized hierarchy."
        )
    hierarchy = contract.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise ValueError("Calibration target contract hierarchy must be a mapping.")
    providers = hierarchy.get("providers")
    categories = hierarchy.get("categories")
    if not isinstance(providers, Mapping) or not providers:
        raise ValueError("Calibration target hierarchy.providers must be non-empty.")
    if not isinstance(categories, Mapping) or not categories:
        raise ValueError("Calibration target hierarchy.categories must be non-empty.")
    for provider_id, provider in providers.items():
        if not str(provider_id) or not isinstance(provider, Mapping):
            raise ValueError("Each hierarchy provider must be a named mapping.")
        if not str(provider.get("label") or "").strip():
            raise ValueError(f"Hierarchy provider {provider_id!r} needs a label.")
    for category_id, category in categories.items():
        if not str(category_id) or not isinstance(category, Mapping):
            raise ValueError("Each hierarchy category must be a named mapping.")
        provider_id = str(category.get("provider_id") or "")
        if provider_id not in providers:
            raise ValueError(
                f"Hierarchy category {category_id!r} references unknown provider "
                f"{provider_id!r}."
            )
        if not str(category.get("label") or "").strip():
            raise ValueError(f"Hierarchy category {category_id!r} needs a label.")
    for target in contract.get("targets", ()):
        category_id = str(target.get("category_id") or "")
        if category_id not in categories:
            raise ValueError(
                f"Contract target {target.get('target_id')!r} references unknown "
                f"hierarchy category {category_id!r}."
            )
        label = target.get("label")
        if label is not None and not str(label).strip():
            raise ValueError(
                f"Contract target {target.get('target_id')!r} label must be a "
                "non-empty string when declared."
            )


def _hierarchy_seed(
    target: Mapping[str, Any], hierarchy_catalog: Mapping[str, Any]
) -> dict[str, Any]:
    category_id = str(target["category_id"])
    category = hierarchy_catalog["categories"][category_id]
    provider_id = str(category["provider_id"])
    provider = hierarchy_catalog["providers"][provider_id]
    seed = {
        "provider": {"id": provider_id, "label": str(provider["label"])},
        "category": {
            "id": category_id,
            "label": str(category["label"]),
            "provider_id": provider_id,
        },
    }
    target_label = str(target.get("label") or "").strip()
    if target_label:
        seed["target_label"] = target_label
    return seed


def _candidate_rows(
    target: Mapping[str, Any],
    facts: tuple[Mapping[str, Any], ...],
    geography_pin: GeographyPin,
    config: TargetReferenceAuthoringConfig,
    hierarchy_catalog: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    geography_fanout = config.geography_fanout_by_target_id.get(
        str(target["target_id"])
    )
    if geography_fanout:
        yield from _geography_fanout_rows(
            target, geography_fanout, config, hierarchy_catalog=hierarchy_catalog
        )
        return
    selector = _target_selector(target, geography_pin, config)
    if "groupby_dimension" in selector:
        rows = _fanout_rows(
            target,
            facts,
            selector,
            config,
            hierarchy_catalog=hierarchy_catalog,
        )
        if rows:
            yield from rows
            return
    yield _reference_row(
        target,
        selector,
        config,
        hierarchy_catalog=hierarchy_catalog,
    )


def _composition_member_count(
    config: TargetReferenceAuthoringConfig,
    member_level: str,
    members: tuple[str, ...],
) -> int:
    """Members of a composition, counting a recoded authority once.

    A member list may carry an authority under its roster code and its alias
    code(s) (``geography_composition_aliases``); the publisher prints one row
    per authority per vintage, so the count of rows the composition expects is
    the count of authorities, not of spellings.
    """

    aliases = config.geography_composition_aliases.get(member_level, {})
    alias_codes = {code for codes in aliases.values() for code in codes}
    return len([member for member in members if member not in alias_codes])


def _geography_fanout_rows(
    target: Mapping[str, Any],
    areas: GeographyFanout,
    config: TargetReferenceAuthoringConfig,
    *,
    hierarchy_catalog: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    """One reference per roster cell, named and measured ``target_id@area``.

    The selector is the contract selector plus the cell's geography, exactly
    as a pinned reference is built, so every other declaration (value
    operation, member count, period policy, uprating holds) follows the
    contract unchanged. The row's metadata carries the cell's geography under
    the same keys area-grain references use, plus whatever the country's
    ``geography_fanout_metadata`` adds for the cell.
    """

    target_id = str(target["target_id"])
    seen: set[str] = set()
    for geography_level, geography_id in areas:
        level = str(geography_level)
        area_id = str(geography_id)
        if not level or not area_id:
            raise ValueError(
                f"target {target_id!r} geography fan-out has a blank cell "
                f"({geography_level!r}, {geography_id!r})."
            )
        if area_id in seen:
            raise ValueError(
                f"target {target_id!r} geography fan-out lists {area_id!r} twice."
            )
        seen.add(area_id)
        composition = config.geography_composition_by_target_id.get(target_id, {}).get(
            area_id
        )
        if composition is None:
            selector = _target_selector(
                target,
                {"geography_level": level, "geography_id": area_id},
                config,
            )
        else:
            member_level, member_ids = composition
            members = tuple(dict.fromkeys(str(member) for member in member_ids))
            if not member_level or not members:
                raise ValueError(
                    f"target {target_id!r} composition for {area_id!r} needs a "
                    "member level and at least one member id."
                )
            selector = _target_selector(
                target,
                {"geography_level": str(member_level), "geography_id": list(members)},
                config,
            )
        row = _reference_row(
            target,
            selector,
            config,
            name=f"{target_id}@{area_id}",
            hierarchy_catalog=hierarchy_catalog,
        )
        metadata: dict[str, str] = {
            **row["metadata"],
            "geography_level": level,
            "geography_id": area_id,
        }
        if composition is not None:
            member_level, member_ids = composition
            members = tuple(dict.fromkeys(str(member) for member in member_ids))
            if row.get("value_operation") not in {"sum", "linear_combination"}:
                raise ValueError(
                    f"target {target_id!r} composes {area_id!r} from "
                    f"{len(members)} member rows but declares value_operation "
                    f"{row.get('value_operation', 'identity')!r}; a composed "
                    "cell needs an aggregating operation."
                )
            member_count = _composition_member_count(config, member_level, members)
            if row.get("value_operation") == "sum":
                row["expected_member_count"] = member_count
            else:
                row["value_operands"] = [
                    {**dict(operand), "expected_member_count": member_count}
                    if operand.get("expected_member_count") is None
                    else dict(operand)
                    for operand in row.get("value_operands", ())
                ]
            metadata["composed_from_level"] = str(member_level)
            metadata["composed_member_count"] = str(member_count)
        if config.geography_fanout_metadata is not None:
            extra = config.geography_fanout_metadata(
                target, level, area_id, str(row["entity"])
            )
            for key, value in extra.items():
                if not isinstance(value, str):
                    raise ValueError(
                        f"target {target_id!r} geography fan-out metadata "
                        f"{key!r} must be a string, got {type(value).__name__}."
                    )
                metadata[str(key)] = value
        row["metadata"] = metadata
        yield row


def _target_selector(
    target: Mapping[str, Any],
    geography_pin: GeographyPin,
    config: TargetReferenceAuthoringConfig,
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    selector = {**dict(target["ledger_selector"]), **dict(geography_pin)}
    pins = config.selector_pins_by_target_id.get(target_id, {})
    for key, value in pins.items():
        if key == "dimension_values" and isinstance(value, Mapping):
            existing = selector.get("dimension_values")
            selector[key] = {
                **(dict(existing) if isinstance(existing, Mapping) else {}),
                **dict(value),
            }
            continue
        selector[key] = value
    return selector


def _area_target_selector(
    target: Mapping[str, Any],
    *,
    geography_level: str,
    area_id: str,
    config: AreaTargetReferenceAuthoringConfig,
) -> dict[str, Any]:
    aliases = tuple(config.area_id_aliases.get(geography_level, {}).get(area_id, ()))
    selector = {
        **dict(target["ledger_selector"]),
        "geography_level": geography_level,
        "geography_id": [area_id, *aliases] if aliases else area_id,
    }
    pins = config.selector_pins_by_target_id.get(str(target["target_id"]), {})
    for key, value in pins.items():
        if key == "dimension_values" and isinstance(value, Mapping):
            existing = selector.get("dimension_values")
            selector[key] = {
                **(dict(existing) if isinstance(existing, Mapping) else {}),
                **dict(value),
            }
            continue
        selector[key] = value
    return selector


def _area_reference_row(
    target: Mapping[str, Any],
    selector: Mapping[str, Any],
    *,
    geography_level: str,
    area_id: str,
    config: AreaTargetReferenceAuthoringConfig,
    hierarchy_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    binding = target["bindings"]["policyengine"]
    row: dict[str, Any] = {
        "name": f"{target_id}@{area_id}",
        "ledger_selector": dict(selector),
        "entity": binding.get("from_entity") or binding.get("map_to") or "household",
        "measure": binding["metric_name"],
        "family": target["family"],
        "period": config.target_period,
        "metadata": {
            "contract_target_id": target_id,
            "measure_kind": "prepared_column",
            "geography_level": geography_level,
            "geography_id": area_id,
            **(
                {
                    "geography_id_aliases": ",".join(
                        config.area_id_aliases[geography_level][area_id]
                    )
                }
                if config.area_id_aliases.get(geography_level, {}).get(area_id)
                else {}
            ),
            **dict(config.reference_metadata_by_target_id.get(target_id, {})),
        },
        "hierarchy": _hierarchy_seed(target, hierarchy_catalog),
    }
    assertion_policy = target.get("assertion_policy")
    if assertion_policy is not None:
        row["assertion_policy"] = assertion_policy
    value_operation = config.value_operation_by_target_id.get(target_id)
    if value_operation is not None and value_operation != "identity":
        row["value_operation"] = value_operation
    operands = target.get("value_operands")
    if operands is not None:
        row["value_operands"] = operands
    if target.get("expected_member_count") is not None:
        row["expected_member_count"] = target["expected_member_count"]
    return row


def _fanout_rows(
    target: Mapping[str, Any],
    facts: tuple[Mapping[str, Any], ...],
    selector: Mapping[str, Any],
    config: TargetReferenceAuthoringConfig,
    hierarchy_catalog: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    dimension = str(selector["groupby_dimension"])
    matches = [fact for fact in facts if _fact_matches_selector(fact, selector)]
    existing_pins = selector.get("dimension_values") or {}
    pinned_dimension_names = (
        frozenset(str(key) for key in existing_pins)
        if isinstance(existing_pins, Mapping)
        else frozenset()
    )
    pins: dict[tuple[str, str], tuple[str, Any, Mapping[str, Any]]] = {}
    for fact in matches:
        dimension_name, dimension_value = _native_groupby_pin(
            fact,
            dimension,
            pinned_dimension_names=pinned_dimension_names,
        )
        if not dimension_name:
            continue
        key = (dimension_name, json.dumps(dimension_value, sort_keys=True))
        pins.setdefault(key, (dimension_name, dimension_value, fact))
    rows = []
    for _, (dimension_name, dimension_value, fact) in sorted(pins.items()):
        merged_dimension_values = {
            **(dict(existing_pins) if isinstance(existing_pins, Mapping) else {}),
            dimension_name: dimension_value,
        }
        pinned_selector = {
            **dict(selector),
            "dimension_values": merged_dimension_values,
        }
        name = (
            config.fanout_name(target, fact)
            if config.fanout_name is not None
            else _fallback_fanout_name(target, fact)
        )
        if not name:
            continue
        rows.append(
            _reference_row(
                target,
                pinned_selector,
                config,
                name=name,
                hierarchy_catalog=hierarchy_catalog,
            )
        )
    return tuple(rows)


def _facts_by_source(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        source = _source_name(fact)
        if source:
            buckets.setdefault(source, []).append(fact)
    return {key: tuple(value) for key, value in buckets.items()}


def _source_prefilter(
    facts: tuple[Mapping[str, Any], ...],
    facts_by_source: Mapping[str, tuple[Mapping[str, Any], ...]],
    target: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    source_name = target["ledger_selector"].get("source_name")
    if isinstance(source_name, str) and source_name:
        return facts_by_source.get(source_name, ())
    return facts


def _source_name(fact: Mapping[str, Any]) -> str:
    return str(
        fact.get("source", {}).get("source_name")
        or fact.get("observed_measure", {}).get("source_name")
        or ""
    )


def _facts_by_geography_id(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        area_id = str(fact.get("geography", {}).get("id") or "")
        if area_id:
            buckets.setdefault(area_id, []).append(fact)
    return {key: tuple(value) for key, value in buckets.items()}


def _area_deferral_index(
    config: AreaTargetReferenceAuthoringConfig,
    contract: Mapping[str, Any],
    areas_by_level: Mapping[str, tuple[str, ...]],
) -> dict[tuple[str, str, str], AreaSignedDeferral]:
    target_levels = {
        (str(target["target_id"]), str(level))
        for target in contract.get("targets", ())
        for level in target.get("geography_levels", ())
    }
    index: dict[tuple[str, str, str], AreaSignedDeferral] = {}
    for deferral in config.area_signed_deferrals:
        if (deferral.target_id, deferral.geography_level) not in target_levels:
            raise ValueError(
                "Area signed deferral references undeclared target/geography "
                f"{deferral.target_id!r}/{deferral.geography_level!r}."
            )
        roster = set(areas_by_level.get(deferral.geography_level, ()))
        outside = sorted(set(deferral.area_ids) - roster)
        if outside:
            raise ValueError(
                "Area signed deferral references area id(s) outside the roster "
                f"for {deferral.geography_level!r}: {outside!r}."
            )
        for area_id in deferral.area_ids:
            key = (deferral.target_id, deferral.geography_level, area_id)
            if key in index:
                raise ValueError(f"Duplicate area signed deferral for {key!r}.")
            index[key] = deferral
    return index


def _native_groupby_pin(
    fact: Mapping[str, Any],
    dimension: str,
    *,
    pinned_dimension_names: frozenset[str] = frozenset(),
) -> tuple[str, Any]:
    dimensions = fact.get("dimensions") or {}
    if not isinstance(dimensions, Mapping):
        return "", None
    candidates = (
        dimension,
        dimension.split(".")[-1],
        dimension.replace(".", "_"),
    )
    for candidate in candidates:
        if candidate in dimensions:
            if candidate in pinned_dimension_names:
                return "", None
            return candidate, dimensions[candidate]
    unpinned_dimensions = [
        (str(key), value)
        for key, value in dimensions.items()
        if str(key) not in pinned_dimension_names
    ]
    if len(unpinned_dimensions) == 1:
        return unpinned_dimensions[0]
    if len(dimensions) == 1:
        key, value = next(iter(dimensions.items()))
        if str(key) in pinned_dimension_names:
            return "", None
        return str(key), value
    return "", None


def _reference_row(
    target: Mapping[str, Any],
    selector: Mapping[str, Any],
    config: TargetReferenceAuthoringConfig,
    *,
    name: str | None = None,
    hierarchy_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    binding = target["bindings"]["policyengine"]
    row: dict[str, Any] = {
        "name": name or target_id,
        "ledger_selector": dict(selector),
        "entity": binding.get("from_entity") or binding.get("map_to") or "household",
        "measure": name or binding["metric_name"],
        "family": target["family"],
        "period": config.target_period,
        "metadata": {
            "contract_target_id": target_id,
            "measure_kind": "prepared_column",
            **dict(config.reference_metadata_by_target_id.get(target_id, {})),
        },
        "hierarchy": _hierarchy_seed(target, hierarchy_catalog),
    }
    assertion_policy = target.get("assertion_policy")
    if assertion_policy is not None:
        row["assertion_policy"] = assertion_policy
    period_match_policy = target.get("period_match_policy")
    if period_match_policy is not None:
        row["period_match_policy"] = period_match_policy
    uprating_index = target.get("uprating_index")
    if uprating_index is not None:
        row["uprating_index"] = str(uprating_index)
    value_operation = config.value_operation_by_target_id.get(target_id)
    if value_operation is None and target_id in config.sum_target_ids:
        value_operation = "sum"
    if value_operation is not None and value_operation != "identity":
        row["value_operation"] = value_operation
    operands = target.get("value_operands")
    if operands is not None:
        row["value_operands"] = operands
    if target.get("expected_member_count") is not None:
        row["expected_member_count"] = target["expected_member_count"]
    dimension_values = selector.get("dimension_values")
    if value_operation == "sum" and isinstance(dimension_values, Mapping):
        member_count = 1
        has_list = False
        for value in dimension_values.values():
            if isinstance(value, list):
                has_list = True
                member_count *= len(value)
        if has_list:
            row["expected_member_count"] = member_count
    return row


def _fallback_fanout_name(target: Mapping[str, Any], fact: Mapping[str, Any]) -> str:
    value_id = str(fact.get("layout", {}).get("groupby_value_id") or "detail")
    return f"{target['target_id']}.{value_id}"


def _json_safe_ledger_id(value: str) -> str:
    return value.replace(".", "_").replace(":", "_")


def _compact_compile_error(status: str) -> str:
    return f"ledger_reference_compile_status_{status}"


def _eligible_fact(reference: LedgerTargetReference, fact: Mapping[str, Any]) -> bool:
    return _not_after_target_period(
        _period_key(fact),
        _period_key_from_value(reference.period),
    ) and _assertion_allowed(reference, fact)


def _classify_deferral(
    error: ValueError,
    matched: list[Mapping[str, Any]],
    eligible: list[Mapping[str, Any]],
) -> str:
    message = str(error)
    if "source_projection" in message:
        return "source_projection_policy"
    if not eligible or "at or before target period" in message:
        return "no_fact_at_or_before_period"
    if "matched multiple Ledger facts" in message:
        return "multi_fact"
    if "invalid value" in message or "missing value" in message:
        return "value_not_finite_or_missing"
    geography_ids = {
        str(fact.get("geography", {}).get("id", ""))
        for fact in eligible
        if fact.get("geography", {}).get("level") == "country"
    }
    if len(geography_ids) > 1:
        return "geography_ambiguous_at_country_level"
    if not matched:
        return "no_fact_at_or_before_period"
    return "signed_deferral_compile_error"


def _classify_area_deferral(
    error: ValueError,
    matched: list[Mapping[str, Any]],
    eligible: list[Mapping[str, Any]],
) -> str:
    if not matched:
        return "no_fact_for_area"
    return _classify_deferral(error, matched, eligible)


def _target_status(candidate_entries: list[dict[str, Any]]) -> str:
    if not candidate_entries:
        return "not_applicable"
    # A row signed out of a fan-out is an adjudicated absence, not a deferral:
    # it neither makes the target partial nor hides a real deferral.
    considered = [
        entry for entry in candidate_entries if entry["status"] != "signed_excluded"
    ]
    if not considered:
        return "signed_excluded"
    if any(entry["status"] == "active" for entry in considered):
        if all(entry["status"] == "active" for entry in considered):
            return "active"
        return "partially_active"
    statuses = sorted({entry["status"] for entry in considered})
    return statuses[0] if len(statuses) == 1 else "multiple_deferrals"


def _signed_row_exclusion(
    row: Mapping[str, Any],
    row_exclusions: Mapping[tuple[str, str], str],
) -> tuple[tuple[str, str], str] | None:
    """The row-level sign-out this fan-out row matches, if any."""

    if not row_exclusions:
        return None
    dimension_values = row.get("ledger_selector", {}).get("dimension_values")
    if not isinstance(dimension_values, Mapping):
        return None
    for dimension, value in dimension_values.items():
        key = (str(dimension), json.dumps(value, sort_keys=True))
        rationale = row_exclusions.get(key)
        if rationale is not None:
            return key, rationale
    return None


def _apply_declared_uprating(
    row: Mapping[str, Any],
    registry: TargetRegistry,
    config: TargetReferenceAuthoringConfig,
) -> TargetRegistry:
    """Apply the country's applier for the reference's declared ``uprating_index``.

    A declared index without an applier fails closed: the surface would
    otherwise record a value the runtime cannot reproduce.
    """

    index = str(row["uprating_index"])
    applier = config.uprating_appliers.get(index)
    if applier is None:
        raise ValueError(
            f"Reference {row['name']!r} declares uprating_index {index!r}, but the "
            "country supplies no applier for it."
        )
    applied = applier(LedgerTargetReference(**row), registry)
    if len(applied.specs) != len(registry.specs) or any(
        "uprating_factor" not in spec.metadata for spec in applied.specs
    ):
        raise ValueError(
            f"Uprating applier for {index!r} must return the same specs with "
            "uprating_factor declared on each."
        )
    return applied


def _apply_uprating_hold(
    row: dict[str, Any],
    resolved_period: str,
    target_period: int | str,
) -> None:
    if not resolved_period or row.get("period_match_policy") == "source_window":
        return
    source_key = _period_key_from_value(resolved_period)
    target_key = _period_key_from_value(target_period)
    if source_key[0] and target_key[0] and source_key[1] < target_key[1]:
        row["uprating_from_period"] = resolved_period
        row["uprating_to_period"] = target_period


def is_finite_value(value: object) -> bool:
    """Return whether a fact value is finite when coerced to float."""

    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False
