"""Shared materialization of declarative Ledger target bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from microcosm.calibrate import TargetRegistry

Provider = Callable[[Any, Mapping[str, Any], int | str], np.ndarray]


@dataclass(frozen=True)
class MaterializationSkip:
    """One target measure the interpreter could not prepare."""

    name: str
    measure: str
    reason: str


@dataclass(frozen=True)
class TargetMaterializationResult:
    """Prepared target columns and skipped-measure diagnostics."""

    adapter: Any
    registry: TargetRegistry
    skipped: tuple[MaterializationSkip, ...] = field(default_factory=tuple)

    def report(self) -> dict[str, object]:
        """Return a serialisable diagnostic report."""

        return {
            "prepared_count": len(self.registry) - len(self.skipped),
            "skipped_count": len(self.skipped),
            "skipped": [skip.__dict__ for skip in self.skipped],
        }


def materialize_target_bindings(
    adapter: Any,
    registry: TargetRegistry,
    contract_targets: Mapping[str, Mapping[str, Any]],
    *,
    period: int | str,
    providers: Mapping[str, Provider] | None = None,
) -> TargetMaterializationResult:
    """Prepare measure columns declared by compiled Ledger target specs."""

    provider_registry = {**default_provider_registry(), **(providers or {})}
    skipped: list[MaterializationSkip] = []
    for spec in registry.specs:
        contract_target_id = spec.metadata.get("contract_target_id")
        target = contract_targets.get(str(contract_target_id))
        if target is None:
            skipped.append(
                MaterializationSkip(
                    name=spec.name,
                    measure=spec.measure,
                    reason="missing_contract_target",
                )
            )
            continue
        binding = target["bindings"]["policyengine"]
        kind = binding.get("kind")
        try:
            if kind:
                provider = provider_registry.get(str(kind))
                if provider is None:
                    raise ValueError(f"unsupported binding kind {kind!r}")
                values = provider(adapter, binding, period)
            else:
                values = _prepared_column_values(adapter, spec.entity, binding)
            values = _apply_household_conditions(
                adapter,
                spec.entity,
                values,
                binding.get("household_conditions", ()),
            )
            adapter.set_column(spec.entity, spec.measure, values)
        except (KeyError, TypeError, ValueError) as error:
            skipped.append(
                MaterializationSkip(
                    name=spec.name,
                    measure=spec.measure,
                    reason=str(error),
                )
            )
    return TargetMaterializationResult(adapter, registry, tuple(skipped))


def default_provider_registry() -> dict[str, Provider]:
    """Return the generic provider engines keyed by binding kind."""

    return {
        "parameter_gated_threshold": parameter_gated_threshold,
        "baseline_flag_crosstab": baseline_flag_crosstab,
        "input_substitution_counterfactual": input_substitution_counterfactual,
    }


def parameter_gated_threshold(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    """Materialize an indicator/value gated by a period parameter."""

    gate = float(adapter.parameter(str(binding["gate_parameter"]), period))
    gated = _column(
        adapter, str(binding.get("from_entity") or "person"), binding["gated_variable"]
    )
    comparison = str(binding.get("gate_comparison") or ">")
    mask = _compare(gated, comparison, gate)
    value_variable = binding.get("value_variable")
    if value_variable and value_variable != "person_count":
        return np.where(mask, _column(adapter, "person", str(value_variable)), 0.0)
    return mask.astype(float)


def baseline_flag_crosstab(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    """Materialize a baseline affected-flag crosstab column."""

    del period
    entity = str(binding.get("from_entity") or "household")
    flag = _column(adapter, entity, binding["affected_flag_variable"]).astype(bool)
    value_variable = str(binding.get("value_variable") or "")
    count_of = str(binding.get("count_of") or "")
    if value_variable and value_variable not in {"household_count", "person_count"}:
        values = _column(adapter, entity, value_variable)
    elif count_of and count_of not in {"household", "person", "household_count"}:
        values = _column(adapter, entity, count_of)
    else:
        values = np.ones_like(flag, dtype=float)
    return np.where(flag, values, 0.0)


def input_substitution_counterfactual(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    """Materialize an output delta under a caller-owned counterfactual."""

    if hasattr(adapter, "counterfactual_delta"):
        return np.asarray(adapter.counterfactual_delta(binding, period), dtype=float)
    raise ValueError("adapter does not provide counterfactual_delta")


def _prepared_column_values(
    adapter: Any,
    entity: str,
    binding: Mapping[str, Any],
) -> np.ndarray:
    entity = str(binding.get("from_entity") or entity)
    if "value_expression" in binding:
        values = _expression(adapter, entity, str(binding["value_expression"]))
    else:
        values = _column(adapter, entity, binding["value_variable"])
    mask = np.ones_like(values, dtype=bool)
    for predicate in binding.get("filters", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    for predicate in binding.get("household_conditions", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    return np.where(mask, values, 0.0)


def _apply_household_conditions(
    adapter: Any,
    entity: str,
    values: np.ndarray,
    predicates: object,
) -> np.ndarray:
    conditions = tuple(predicates or ())
    if not conditions:
        return values
    mask = np.ones_like(values, dtype=bool)
    for predicate in conditions:
        if hasattr(adapter, "household_condition_mask"):
            condition_mask = adapter.household_condition_mask(entity, predicate)
        else:
            condition_mask = _predicate_mask(adapter, entity, predicate)
        mask &= np.asarray(condition_mask, dtype=bool)
    return np.where(mask, values, 0.0)


def _predicate_mask(
    adapter: Any,
    default_entity: str,
    predicate: Mapping[str, Any],
) -> np.ndarray:
    entity = str(predicate.get("entity") or default_entity)
    variable = str(predicate.get("variable") or predicate.get("concept"))
    values = _column(adapter, entity, variable)
    return _compare(values, str(predicate["operator"]), predicate["value"])


def _compare(values: np.ndarray, operator: str, expected: object) -> np.ndarray:
    if operator == "==":
        return values == expected
    if operator == "!=":
        return values != expected
    if operator == "in":
        return np.isin(values, list(expected))
    numeric = values.astype(float)
    threshold = float(expected)
    if operator == ">":
        return numeric > threshold
    if operator == ">=":
        return numeric >= threshold
    if operator == "<":
        return numeric < threshold
    if operator == "<=":
        return numeric <= threshold
    raise ValueError(f"unsupported predicate operator {operator!r}")


def _expression(adapter: Any, entity: str, expression: str) -> np.ndarray:
    parts = [part.strip() for part in expression.split("+")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"unsupported value_expression {expression!r}")
    total: np.ndarray | None = None
    for part in parts:
        values = _column(adapter, entity, part)
        total = values.copy() if total is None else total + values
    if total is None:
        raise ValueError(f"unsupported value_expression {expression!r}")
    return total


def _column(adapter: Any, entity: str, variable: object) -> np.ndarray:
    if str(variable) in {"person_count", "household_count", "benunit_count"}:
        return np.ones(_entity_length(adapter, entity), dtype=float)
    if hasattr(adapter, "column"):
        return np.asarray(adapter.column(entity, str(variable)), dtype=float)
    table = adapter.tables[entity]
    return np.asarray(table[str(variable)], dtype=float)


def _entity_length(adapter: Any, entity: str) -> int:
    if hasattr(adapter, "entity_length"):
        return int(adapter.entity_length(entity))
    table = adapter.tables[entity]
    if isinstance(table, Mapping):
        first = next(iter(table.values()))
        return len(first)
    if hasattr(table, "__len__"):
        return len(table)
    raise TypeError(f"cannot determine row count for entity {entity!r}")
