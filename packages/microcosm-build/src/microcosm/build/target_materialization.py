"""Shared materialization of declarative Ledger target bindings."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from microcosm.calibrate import TargetRegistry

Provider = Callable[[Any, Mapping[str, Any], int | str], np.ndarray]

# Banded measures declare their slice in two places: the contract binding
# names the variable to band on (``groupby_variable``), and each compiled
# spec carries its own band's lower edge in Ledger filter metadata. Two
# encodings are in use — a numeric lower bound (HMRC SPI income bands) and a
# published range label (DWP award bands, in monthly units, hence
# ``band_period_factor``) — and both reduce to one lower edge, because no
# reference anywhere declares an upper bound. A band's upper edge is
# therefore its sibling's lower edge within the same contract target, and the
# top band runs to infinity.
_BAND_LOWER_BOUND_SUFFIX = "_lower_bound"
_LEDGER_FILTER_PREFIX = "ledger_filter_"
_RANGE_LABEL = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(?:to|–|—|-)\s*(?:£\s*)?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OPEN_LABEL = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(?:or more|or over|and over|and above|\+)",
    re.IGNORECASE,
)


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


def _band_lower_edge(
    spec: Any, binding: Mapping[str, Any]
) -> float | None:
    """This spec's band lower edge in model units, or None if unbanded.

    Reads the compiled spec's Ledger filter metadata: a ``*_lower_bound``
    entry when the publisher gives numeric edges, otherwise a range label
    such as ``"£500.01 to £600.00"`` or ``"5 or more"``. ``band_period_factor``
    converts published units to the model's (monthly award bands to annual).
    Non-band filters — ``ledger_filter_family_type = "Single, no children"``
    — carry no numeric range and are ignored here; the binding's own
    ``filters`` list applies those.
    """

    metadata = getattr(spec, "metadata", None) or {}
    factor = float(binding.get("band_period_factor", 1) or 1)
    for key, value in sorted(metadata.items()):
        if not key.startswith(_LEDGER_FILTER_PREFIX):
            continue
        if key.endswith(_BAND_LOWER_BOUND_SUFFIX):
            return float(str(value).replace(",", "")) * factor
    for key, value in sorted(metadata.items()):
        if not key.startswith(_LEDGER_FILTER_PREFIX):
            continue
        match = _RANGE_LABEL.search(str(value)) or _OPEN_LABEL.search(str(value))
        if match is not None:
            return float(match.group(1).replace(",", "")) * factor
    return None


def _band_edges_by_group(
    registry: TargetRegistry,
    contract_targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[float]]:
    """Sorted band lower edges per contract target that declares bands.

    Grouped by contract target rather than by dimension: two measures can
    share a dimension while publishing different band sets, and taking the
    union of their edges would slice each measure on the other's boundaries.
    """

    edges: dict[str, set[float]] = {}
    for spec in registry.specs:
        contract_target_id = str(
            (getattr(spec, "metadata", None) or {}).get("contract_target_id", "")
        )
        target = contract_targets.get(contract_target_id)
        if target is None:
            continue
        binding = target.get("bindings", {}).get("policyengine", {})
        if not binding.get("groupby_variable"):
            continue
        lower = _band_lower_edge(spec, binding)
        if lower is not None:
            edges.setdefault(contract_target_id, set()).add(lower)
    return {key: sorted(values) for key, values in edges.items()}


def _band_bounds(
    spec: Any,
    binding: Mapping[str, Any],
    band_edges: Sequence[float],
) -> tuple[float, float]:
    """The half-open ``[lower, upper)`` this spec's band covers.

    Raises rather than returning an unsliced column: a banded measure whose
    band cannot be read would otherwise report the whole population as if it
    were one band, which is the failure this function exists to end.
    """

    lower = _band_lower_edge(spec, binding)
    if lower is None:
        raise ValueError(
            f"banded measure {getattr(spec, 'measure', '?')!r} declares "
            f"groupby_variable {binding['groupby_variable']!r} but its spec "
            "carries no readable band edge"
        )
    upper = math.inf
    for edge in band_edges:
        if edge > lower:
            upper = edge
            break
    return lower, upper


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
    band_edges = _band_edges_by_group(registry, contract_targets)
    skipped: list[MaterializationSkip] = []
    for spec in registry.specs:
        if hasattr(adapter, "has_column") and adapter.has_column(
            spec.entity, spec.measure
        ):
            continue
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
                band = None
                if binding.get("groupby_variable"):
                    band = _band_bounds(
                        spec,
                        binding,
                        band_edges.get(str(contract_target_id), ()),
                    )
                values = _prepared_column_values(
                    adapter, spec.entity, binding, band=band
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
        return np.where(
            mask,
            _column(
                adapter,
                str(binding.get("from_entity") or "person"),
                str(value_variable),
            ),
            0.0,
        )
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
    mask = flag
    for predicate in binding.get("filters", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    for predicate in binding.get("household_conditions", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    return np.where(mask, values, 0.0)


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
    *,
    band: tuple[float, float] | None = None,
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
    if band is not None:
        lower, upper = band
        banded = _column(adapter, entity, binding["groupby_variable"]).astype(float)
        mask &= (banded >= lower) & (banded < upper)
    return np.where(mask, values, 0.0)


def _predicate_mask(
    adapter: Any,
    default_entity: str,
    predicate: Mapping[str, Any],
) -> np.ndarray:
    if "reduce" in predicate and hasattr(adapter, "household_condition"):
        return np.asarray(adapter.household_condition(predicate), dtype=bool)
    entity = str(predicate.get("entity") or default_entity)
    variable = str(predicate.get("variable") or predicate.get("concept"))
    values = _column(adapter, entity, variable)
    if "operator" in predicate:
        operator, expected = str(predicate["operator"]), predicate["value"]
    else:
        operator, expected = "==", predicate["equals"]
    return _compare(values, operator, expected)


def _compare(values: np.ndarray, operator: str, expected: object) -> np.ndarray:
    if operator == "==":
        return values == expected
    if operator == "!=":
        return values != expected
    if operator == "in":
        return np.isin(values, expected)
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
    if hasattr(adapter, "column"):
        return np.asarray(adapter.column(entity, str(variable)))
    table = adapter.tables[entity]
    return np.asarray(table[str(variable)])
