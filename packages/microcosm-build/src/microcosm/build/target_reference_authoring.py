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
from typing import Any

from microcosm.build.ledger_targets import (  # pyright: ignore[reportPrivateUsage]
    LedgerTargetReference,
    _assertion_allowed,
    _fact_matches_selector,
    _not_after_target_period,
    _period_key,
    _period_key_from_value,
    compile_ledger_target_references,
)

FanoutName = Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]
GeographyPin = Mapping[str, str]


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
    signed_exclusions_by_target_id: Mapping[str, str] = field(default_factory=dict)
    binding_vocabulary: frozenset[str] = frozenset()
    source_fact_feed: str = ""


def author_target_references(
    contract: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
    config: TargetReferenceAuthoringConfig,
) -> AuthoredTargetReferences:
    """Build active target references and a membership report."""

    fact_rows = tuple(facts)
    facts_by_source = _facts_by_source(fact_rows)
    _validate_contract_bindings(contract, config.binding_vocabulary)
    active_rows: list[dict[str, Any]] = []
    target_entries: dict[str, Any] = {}
    geography_pin_report: dict[str, Any] = {}
    genuine_sum_residue: list[str] = []
    uprating_holds: list[dict[str, str]] = []

    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        pin = config.geography_pins.get(target_id, {})
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
        candidates = tuple(_candidate_rows(target, source_facts, pin, config))
        candidate_entries: list[dict[str, Any]] = []
        for row in candidates:
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
            try:
                registry = compile_ledger_target_references(
                    matched,
                    [reference],
                    country=str(contract["country"]),
                )
            except ValueError as error:
                entry["status"] = _classify_deferral(error, matched, eligible)
                entry["error"] = _compact_compile_error(entry["status"])
            else:
                spec = registry.specs[0]
                resolved_period = spec.metadata.get("ledger_fact_period", "")
                _apply_uprating_hold(row, resolved_period, config.target_period)
                if "uprating_from_period" in row:
                    uprating_holds.append(
                        {
                            "name": row["name"],
                            "from": str(row["uprating_from_period"]),
                            "to": str(row["uprating_to_period"]),
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
            candidate_entries.append(entry)
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
        "contract_target_count": len(tuple(contract.get("targets", ()))),
        "active_reference_count": len(active_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "geography_pins": geography_pin_report,
        "genuine_sum_residue": sorted(genuine_sum_residue),
        "uprating_holds": uprating_holds,
        "targets": target_entries,
    }
    return AuthoredTargetReferences(tuple(active_rows), report)


def target_references_resource(
    *,
    country: str,
    description: str,
    authored: AuthoredTargetReferences,
) -> dict[str, Any]:
    """Return a serialisable ``target_references.json`` resource."""

    return {
        "country": country,
        "description": description,
        "allowed_value_operations": [
            "identity",
            "sum",
            "calendar_year_average",
            "latest_plateau",
        ],
        "target_references": list(authored.references),
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


def _candidate_rows(
    target: Mapping[str, Any],
    facts: tuple[Mapping[str, Any], ...],
    geography_pin: GeographyPin,
    config: TargetReferenceAuthoringConfig,
) -> Iterable[dict[str, Any]]:
    selector = _target_selector(target, geography_pin, config)
    if "groupby_dimension" in selector:
        rows = _fanout_rows(target, facts, selector, config)
        if rows:
            yield from rows
            return
    yield _reference_row(target, selector, config)


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


def _fanout_rows(
    target: Mapping[str, Any],
    facts: tuple[Mapping[str, Any], ...],
    selector: Mapping[str, Any],
    config: TargetReferenceAuthoringConfig,
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
        rows.append(_reference_row(target, pinned_selector, config, name=name))
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
        },
    }
    assertion_policy = target.get("assertion_policy")
    if assertion_policy is not None:
        row["assertion_policy"] = assertion_policy
    value_operation = config.value_operation_by_target_id.get(target_id)
    if value_operation is None and target_id in config.sum_target_ids:
        value_operation = "sum"
    if value_operation is not None and value_operation != "identity":
        row["value_operation"] = value_operation
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


def _target_status(candidate_entries: list[dict[str, Any]]) -> str:
    if any(entry["status"] == "active" for entry in candidate_entries):
        if all(entry["status"] == "active" for entry in candidate_entries):
            return "active"
        return "partially_active"
    statuses = sorted({entry["status"] for entry in candidate_entries})
    return statuses[0] if len(statuses) == 1 else "multiple_deferrals"


def _apply_uprating_hold(
    row: dict[str, Any],
    resolved_period: str,
    target_period: int | str,
) -> None:
    if not resolved_period:
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
