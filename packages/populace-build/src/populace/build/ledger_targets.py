"""Ledger fact catalog adapters for Populace target profiles.

Ledger owns source-backed facts. Populace owns the active subset and the
model-variable mapping needed to compile a fact into a calibration target.
This module is intentionally duck-typed against the Ledger/Arch aggregate fact
schema so Populace can consume exported JSONL catalogs without importing the
Ledger implementation package at runtime.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from populace.calibrate import TargetRegistry, TargetSpec

SUPPORTED_AGGREGATIONS = frozenset({"count", "sum", "mean"})


@dataclass(frozen=True)
class LedgerTargetMapping:
    """How Populace maps Ledger facts to model-ready target rows.

    A Ledger fact is source truth; it is not automatically a Populace target.
    Populace must know which model column estimates the fact's measure, and
    optionally which model boolean column implements the fact's scoped
    population. Facts without a mapping are reported as unsupported.
    """

    measure_by_concept: Mapping[str, str] = field(default_factory=dict)
    measure_by_source_record_id: Mapping[str, str] = field(default_factory=dict)
    filter_by_source_record_id: Mapping[str, str] = field(default_factory=dict)
    filter_by_domain: Mapping[str, str] = field(default_factory=dict)
    signed_by_concept: frozenset[str] = frozenset()
    signed_by_source_record_id: frozenset[str] = frozenset()
    entity_by_ledger_entity: Mapping[str, str] = field(default_factory=dict)
    family_by_source_name: Mapping[str, str] = field(default_factory=dict)
    family_by_concept: Mapping[str, str] = field(default_factory=dict)
    default_family: str = "ledger"


@dataclass(frozen=True)
class UnsupportedLedgerTarget:
    """A Ledger fact Populace deliberately did not activate."""

    reason: str
    identifier: str
    concept: str = ""
    source_record_id: str = ""


@dataclass(frozen=True)
class LedgerTargetSelection:
    """Selected Populace targets plus explicit unsupported Ledger facts."""

    specs: tuple[TargetSpec, ...]
    unsupported: tuple[UnsupportedLedgerTarget, ...]

    def to_registry(self, *, country: str) -> TargetRegistry:
        """Build a Populace target registry from the selected specs."""
        return TargetRegistry(self.specs, country=country)


def select_ledger_targets(
    facts: Iterable[object],
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> LedgerTargetSelection:
    """Select the Ledger facts Populace can target.

    Args:
        facts: Ledger/Arch ``AggregateFact`` objects or JSON-like mappings.
        mapping: Explicit Populace model-variable mapping.
        period: Optional period override for every resulting target.

    Returns:
        A selection with model-ready :class:`TargetSpec` rows and unsupported
        facts with reasons. Unsupported facts are a feature: the target catalog
        can be broader than Populace's current build surface.
    """

    specs: list[TargetSpec] = []
    unsupported: list[UnsupportedLedgerTarget] = []
    for fact in facts:
        try:
            specs.append(target_spec_from_ledger_fact(fact, mapping, period=period))
        except _UnsupportedFactError as exc:
            unsupported.append(exc.unsupported)
    return LedgerTargetSelection(tuple(specs), tuple(unsupported))


def select_ledger_targets_from_jsonl(
    path: str | Path,
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> LedgerTargetSelection:
    """Select Ledger facts from a JSON Lines consumer-contract file."""
    with Path(path).open() as file:
        return select_ledger_targets(
            _jsonl_rows(file),
            mapping,
            period=period,
        )


def target_spec_from_ledger_fact(
    fact: object,
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> TargetSpec:
    """Convert one supported Ledger fact to a Populace target spec.

    The target's ``name`` is the Ledger stable fact identifier when available,
    otherwise its source-record ID. Human-facing tooling should display the
    structured metadata fields, not this internal key.
    """

    source_record_id = _source_record_id(fact)
    fact_key = _fact_key(fact)
    identifier = fact_key or source_record_id
    if not identifier:
        raise _unsupported("missing_identifier", fact)

    concept = _primary_measure_concept(fact)
    if not concept:
        raise _unsupported("missing_measure_concept", fact)

    aggregation = _str_at(fact, "aggregation", "method")
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise _unsupported(f"unsupported_aggregation:{aggregation}", fact)

    measure = mapping.measure_by_source_record_id.get(source_record_id)
    if not measure:
        for candidate in _measure_concepts(fact):
            measure = mapping.measure_by_concept.get(candidate)
            if measure:
                break
    if not measure:
        raise _unsupported("missing_model_measure_mapping", fact)

    ledger_entity = _str_at(fact, "entity", "name")
    entity = mapping.entity_by_ledger_entity.get(ledger_entity, ledger_entity)
    if not entity:
        raise _unsupported("missing_entity", fact)

    value = _at(fact, "value")
    if value is None:
        raise _unsupported("missing_value", fact)
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _unsupported("invalid_value", fact) from exc
    if not math.isfinite(numeric_value):
        raise _unsupported("invalid_value", fact)

    target_period = period if period is not None else _at(fact, "period", "value")
    if target_period is None:
        raise _unsupported("missing_period", fact)

    source_name = _source_name(fact)
    family = (
        mapping.family_by_concept.get(concept)
        or mapping.family_by_source_name.get(source_name)
        or source_name
        or mapping.default_family
    )

    filter_column = mapping.filter_by_source_record_id.get(source_record_id)
    if filter_column is None and _requires_detail_filter_mapping(fact):
        raise _unsupported("missing_model_filter_mapping", fact)
    domain = _domain(fact)
    if filter_column is None and _requires_domain_filter_mapping(fact):
        filter_column = mapping.filter_by_domain.get(domain)
    if filter_column is None and _requires_filter_mapping(fact):
        raise _unsupported("missing_model_filter_mapping", fact)
    if numeric_value < 0 and not _allows_signed_target(fact, mapping):
        raise _unsupported("missing_signed_target_mapping", fact)

    try:
        return TargetSpec(
            name=identifier,
            entity=entity,
            measure=measure,
            aggregation=aggregation,
            value=numeric_value,
            filter=filter_column,
            period=target_period,
            source=_source_citation(fact),
            family=family,
            signed=numeric_value < 0,
            metadata=_ledger_metadata(fact, fact_key=fact_key),
        )
    except (TypeError, ValueError) as exc:
        raise _unsupported(f"invalid_target_spec:{type(exc).__name__}", fact) from exc


class _UnsupportedFactError(Exception):
    def __init__(self, unsupported: UnsupportedLedgerTarget) -> None:
        super().__init__(unsupported.reason)
        self.unsupported = unsupported


def _unsupported(reason: str, fact: object) -> _UnsupportedFactError:
    return _UnsupportedFactError(
        UnsupportedLedgerTarget(
            reason=reason,
            identifier=_fact_key(fact) or _source_record_id(fact),
            concept=_primary_measure_concept(fact),
            source_record_id=_source_record_id(fact),
        )
    )


def _at(obj: object, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _jsonl_rows(lines: Iterable[str]) -> Iterable[object]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid Ledger JSONL row {line_number}: {exc.msg}"
            ) from exc


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    if value is None:
        return ""
    return str(value)


def _fact_key(fact: object) -> str:
    return (
        _str_at(fact, "aggregate_fact_key")
        or _str_at(fact, "fact_key")
        or _str_at(fact, "legacy_fact_key")
    )


def _source_record_id(fact: object) -> str:
    return _str_at(fact, "source_record_id") or _str_at(
        fact, "lineage", "source_record_id"
    )


def _measure_concepts(fact: object) -> tuple[str, ...]:
    concepts = (
        _str_at(fact, "measure", "concept"),
        _str_at(fact, "concept_alignment", "canonical_concept"),
        _str_at(fact, "observed_measure", "source_concept"),
        _str_at(fact, "measure", "source_concept"),
    )
    return tuple(dict.fromkeys(concept for concept in concepts if concept))


def _primary_measure_concept(fact: object) -> str:
    concepts = _measure_concepts(fact)
    return concepts[0] if concepts else ""


def _source_measure_concept(fact: object) -> str:
    return (
        _str_at(fact, "observed_measure", "source_concept")
        or _str_at(fact, "measure", "source_concept")
        or _primary_measure_concept(fact)
    )


def _source_name(fact: object) -> str:
    return _str_at(fact, "source", "source_name") or _str_at(
        fact, "observed_measure", "source_name"
    )


def _dimensions(fact: object) -> Mapping[str, object]:
    filters = _at(fact, "filters")
    if isinstance(filters, Mapping):
        return filters
    dimensions = _at(fact, "dimensions")
    if isinstance(dimensions, Mapping):
        return dimensions
    return {}


def _domain(fact: object) -> str:
    return _str_at(fact, "domain") or _str_at(fact, "universe_constraints", "domain")


def _constraint_rows(fact: object) -> tuple[object, ...]:
    constraints = _at(fact, "constraints")
    if isinstance(constraints, list | tuple):
        return tuple(constraints)
    universe_constraints = _at(fact, "universe_constraints", "constraints")
    if isinstance(universe_constraints, list | tuple):
        return tuple(universe_constraints)
    return ()


def _requires_filter_mapping(fact: object) -> bool:
    return _requires_detail_filter_mapping(fact) or _requires_domain_filter_mapping(
        fact
    )


def _requires_detail_filter_mapping(fact: object) -> bool:
    if _constraint_rows(fact):
        return True
    return any(
        not _is_unscoped_dimension_value(value) for value in _dimensions(fact).values()
    )


def _requires_domain_filter_mapping(fact: object) -> bool:
    return not _is_unscoped_domain(_domain(fact))


def _is_unscoped_domain(domain: str) -> bool:
    normalized = domain.strip().lower().replace("_", " ")
    return normalized in {"", "all", "all households", "all persons", "population"}


def _is_unscoped_dimension_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    normalized = str(value).strip().lower().replace("_", " ")
    return normalized in {"", "all", "all returns", "total", "totals", "overall"}


def _allows_signed_target(fact: object, mapping: LedgerTargetMapping) -> bool:
    source_record_id = _source_record_id(fact)
    if source_record_id in mapping.signed_by_source_record_id:
        return True
    return any(
        concept in mapping.signed_by_concept for concept in _measure_concepts(fact)
    )


def _source_citation(fact: object) -> str:
    parts = [
        _source_name(fact),
        _str_at(fact, "source", "source_table")
        or _str_at(fact, "observed_measure", "source_table"),
        _str_at(fact, "source", "source_file"),
        _str_at(fact, "source", "vintage"),
        _str_at(fact, "source", "url"),
    ]
    citation = " | ".join(part for part in parts if part)
    return citation or "PolicyEngine Ledger source-backed fact"


def _ledger_metadata(fact: object, *, fact_key: str) -> dict[str, str]:
    metadata = {
        "ledger_source": "policyengine_ledger",
        "ledger_fact_key": fact_key,
        "ledger_source_record_id": _source_record_id(fact),
        "ledger_aggregate_fact_key": _str_at(fact, "aggregate_fact_key"),
        "ledger_semantic_fact_key": _str_at(fact, "semantic_fact_key"),
        "ledger_legacy_fact_key": _str_at(fact, "legacy_fact_key"),
        "ledger_observed_measure_key": _str_at(fact, "observed_measure_key"),
        "ledger_dimension_set_key": _str_at(fact, "dimension_set_key"),
        "ledger_universe_constraint_set_key": _str_at(
            fact, "universe_constraint_set_key"
        ),
        "ledger_measure_concept": _primary_measure_concept(fact),
        "ledger_measure_unit": _str_at(fact, "measure", "unit")
        or _str_at(fact, "observed_measure", "unit"),
        "ledger_source_concept": _source_measure_concept(fact),
        "ledger_concept_relation": _str_at(fact, "measure", "concept_relation")
        or _str_at(fact, "concept_alignment", "relation"),
        "ledger_concept_authority": _str_at(fact, "measure", "concept_authority")
        or _str_at(fact, "concept_alignment", "authority"),
        "ledger_legal_vintage": _str_at(fact, "measure", "legal_vintage")
        or _str_at(fact, "concept_alignment", "legal_vintage"),
        "ledger_period_type": _str_at(fact, "period", "type"),
        "ledger_geography_level": _str_at(fact, "geography", "level"),
        "ledger_geography_id": _str_at(fact, "geography", "id"),
        "ledger_geography_name": _str_at(fact, "geography", "name"),
        "ledger_geography_vintage": _str_at(fact, "geography", "vintage"),
        "ledger_entity_name": _str_at(fact, "entity", "name"),
        "ledger_entity_role": _str_at(fact, "entity", "role"),
        "ledger_domain": _domain(fact),
        "ledger_layout_record_set_id": _str_at(fact, "layout", "record_set_id"),
        "ledger_layout_groupby_dimension": _str_at(fact, "layout", "groupby_dimension"),
        "ledger_layout_groupby_value_id": _str_at(fact, "layout", "groupby_value_id"),
        "ledger_layout_measure_id": _str_at(fact, "layout", "measure_id"),
    }
    constraint_rows = _constraint_rows(fact)
    if constraint_rows:
        metadata["ledger_universe_constraint_count"] = str(len(constraint_rows))
    for key, value in sorted(_dimensions(fact).items()):
        if value is not None:
            metadata[f"ledger_filter_{key}"] = str(value)
    return {key: value for key, value in metadata.items() if value}
