"""Reporting strata from declared income, unit weights and assigned receipt.

Policy choices belong to the classifier declaration. These helpers do not
infer programs, tax treatment or annualisation from column names. A variant
uses :func:`carry_baseline_deciles` to preserve the baseline's unit membership,
then reports its own weighted cell shares.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum
from numbers import Integral, Real

import numpy as np
import pandas as pd

from microcosm.frame.schema import EntitySchema
from microcosm.frame.weights import WeightKind

__all__ = [
    "annual_unit_income",
    "carry_baseline_deciles",
    "classify_receipt_groups",
    "decile_weighted_shares",
    "validate_income_classifier",
    "validate_receipt_classifier",
    "weighted_deciles",
]


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text.")
    return value


def _mapping(value, fields, name):
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} requires exactly {sorted(fields)}.")
    return dict(value)


def _sequence(value, name):
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence.")
    return list(value)


def _bin_count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("n_bins must be a positive integer.")
    return int(value)


def _common(spec):
    for name in ("entity", "person_entity", "unit_id_column", "membership_column"):
        _text(spec[name], name)
    if spec["entity"] == spec["person_entity"]:
        raise ValueError("The reporting entity must differ from the person entity.")


def validate_income_classifier(spec: Mapping) -> dict:
    """Validate the complete executable annual-gross-income declaration."""

    result = _mapping(
        spec,
        {
            "entity",
            "person_entity",
            "unit_id_column",
            "membership_column",
            "weight_entity",
            "expected_weight_kind",
            "tie_key",
            "boundary_scenario",
            "n_bins",
            "rank_method",
            "components",
            "excluded_columns",
        },
        "Income classifier",
    )
    _common(result)
    for name in ("weight_entity", "tie_key", "boundary_scenario"):
        _text(result[name], name)
    if result["expected_weight_kind"] not in {kind.value for kind in WeightKind}:
        raise ValueError("expected_weight_kind must declare a known weight kind.")
    result["n_bins"] = _bin_count(result["n_bins"])
    if result["rank_method"] != "weighted_midpoint":
        raise ValueError("rank_method must be weighted_midpoint.")
    excluded = []
    for row in _sequence(result["excluded_columns"], "excluded_columns"):
        row = _mapping(row, {"entity", "column"}, "Excluded income column")
        excluded.append(
            (_text(row["entity"], "excluded entity"), _text(row["column"], "column"))
        )
    if len(set(excluded)) != len(excluded):
        raise ValueError("Excluded income columns must be distinct.")
    components = _sequence(result["components"], "components")
    if not components:
        raise ValueError("Income components must be nonempty.")
    seen = set()
    validated = []
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("Each income component must be an object.")
        fields = {"entity", "column", "kind", "tax_basis", "annualisation_factor"}
        if component.get("kind") == "benefit":
            fields.add("receipt_column")
        component = _mapping(component, fields, "Income component")
        entity = _text(component["entity"], "component entity")
        column = _text(component["column"], "component column")
        if entity not in {result["entity"], result["person_entity"]}:
            raise ValueError(
                "Income components must belong to persons or reporting units."
            )
        if component["kind"] not in {"market", "benefit"}:
            raise ValueError("Income component kind must be market or benefit.")
        if component["tax_basis"] != "before_tax":
            raise ValueError(
                "Every income component must be before_tax; net income is refused."
            )
        factor = component["annualisation_factor"]
        if (
            isinstance(factor, bool)
            or not isinstance(factor, Real)
            or not np.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("annualisation_factor must be finite and positive.")
        key = (entity, column)
        if key in excluded:
            raise ValueError(f"Excluded income component {entity}.{column} is refused.")
        if key in seen:
            raise ValueError("Income components must be distinct.")
        seen.add(key)
        if component["kind"] == "benefit":
            _text(component["receipt_column"], "receipt_column")
        validated.append(component)
    result["components"] = validated
    return result


def validate_receipt_classifier(spec: Mapping) -> dict:
    """Validate the shared member predicates and their reporting priority."""

    result = _mapping(
        spec,
        {
            "entity",
            "person_entity",
            "unit_id_column",
            "membership_column",
            "predicates",
            "priority",
            "default",
        },
        "Receipt classifier",
    )
    _common(result)
    _text(result["default"], "default")
    predicates = result["predicates"]
    if not isinstance(predicates, Mapping) or not predicates:
        raise ValueError("Receipt predicates must be a nonempty object.")
    validated = {}
    for name, predicate in predicates.items():
        _text(name, "predicate name")
        predicate = _mapping(
            predicate, {"entity", "columns", "reduction"}, "Receipt predicate"
        )
        if predicate["entity"] != result["person_entity"]:
            raise ValueError("Receipt predicates must read the declared person entity.")
        if predicate["reduction"] != "any":
            raise ValueError("Receipt predicates must use any-member reduction.")
        columns = _sequence(predicate["columns"], "predicate columns")
        if not columns or any(not isinstance(c, str) or not c.strip() for c in columns):
            raise ValueError("Receipt predicate columns must be nonempty text.")
        if len(set(columns)) != len(columns):
            raise ValueError("Receipt predicate columns must be distinct.")
        validated[name] = predicate
    priority = _sequence(result["priority"], "priority")
    if (
        not all(isinstance(name, str) for name in priority)
        or len(set(priority)) != len(priority)
        or set(priority) != set(validated)
        or result["default"] in validated
    ):
        raise ValueError(
            "Priority must name every predicate once and exclude the default."
        )
    result["predicates"] = validated
    return result


def _integer_ids(values, name, *, unique=True):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or any(
        isinstance(value, bool | np.bool_) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError(f"{name} must contain nonmissing integer ids.")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique ids.")
    return values


def _tie_ids(values, size):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or len(values) != size:
        raise ValueError("Tie ids must have one id per unit.")
    integers = all(
        isinstance(v, Integral) and not isinstance(v, bool | np.bool_) for v in values
    )
    strings = all(isinstance(v, str) and v for v in values)
    if not (integers or strings) or len(set(values)) != size:
        raise ValueError("Tie ids must be unique integers or unique nonempty strings.")
    return values


def _numeric(values, name):
    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite numeric vector.")
    result = source.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric vector.")
    return result


def _weights(values, size):
    weights = _numeric(values, "Weights")
    if len(weights) != size or (weights < 0).any() or not (weights > 0).any():
        raise ValueError(
            "Weights must align to units, be nonnegative and have positive mass."
        )
    return weights


def _column(tables, entity, column):
    if entity not in tables or column not in tables[entity]:
        raise ValueError(f"Required classifier column {entity}.{column} is missing.")
    return tables[entity][column]


def _relations(tables, spec):
    units = _integer_ids(
        _column(tables, spec["entity"], spec["unit_id_column"]), "Unit ids"
    )
    schema = EntitySchema(
        person_entity=spec["person_entity"], group_entities=(spec["entity"],)
    )
    person_ids = _integer_ids(
        _column(tables, spec["person_entity"], schema.person_id_column), "Person ids"
    )
    membership = _integer_ids(
        _column(tables, spec["person_entity"], spec["membership_column"]),
        "Unit memberships",
        unique=False,
    )
    rows = pd.Index(units).get_indexer(membership)
    if (rows < 0).any():
        raise ValueError("Unit memberships contain orphan ids.")
    if len(units) == 0 or (np.bincount(rows, minlength=len(units)) == 0).any():
        raise ValueError("Every reporting unit must have at least one member.")
    order = np.lexsort((person_ids, rows))
    starts = np.flatnonzero(np.r_[True, np.diff(rows[order]) != 0])
    return units, rows, order, starts


def _assigned_flags(values, name):
    flags = values.to_numpy(dtype=object)
    if any(not isinstance(value, bool | np.bool_) for value in flags):
        raise ValueError(
            f"Assigned receipt {name} must contain boolean flags without missing values."
        )
    return flags.astype(bool)


def annual_unit_income(tables: Mapping[str, pd.DataFrame], spec: Mapping) -> np.ndarray:
    """Annual before-tax assigned-receipt income in reporting-unit row order.

    Person components sum in stable person-id order within each unit. Benefit
    amounts are nonnegative engine payments multiplied by assigned receipt;
    market income can include losses. Neither entitlement nor the reporting
    outcome is implicitly included.
    """

    spec = validate_income_classifier(spec)
    units, _, order, starts = _relations(tables, spec)
    contributions = []
    try:
        with np.errstate(over="raise", invalid="raise"):
            for component in spec["components"]:
                entity, column = component["entity"], component["column"]
                values = _numeric(_column(tables, entity, column), f"{entity}.{column}")
                if component["kind"] == "benefit":
                    if (values < 0).any():
                        raise ValueError("Benefit payments must be nonnegative.")
                    receipt_column = component["receipt_column"]
                    flags = _assigned_flags(
                        _column(tables, entity, receipt_column),
                        f"{entity}.{receipt_column}",
                    )
                    values = values * flags
                values = values * component["annualisation_factor"]
                if entity == spec["person_entity"]:
                    values = np.add.reduceat(values[order], starts)
                contributions.append(values)
            result = np.sum(contributions, axis=0)
    except FloatingPointError as error:
        raise ValueError("Annual unit income must remain finite.") from error
    if len(result) != len(units) or not np.isfinite(result).all():
        raise ValueError("Annual unit income must remain finite and align to unit ids.")
    return result


def decile_weighted_shares(labels, weights, n_bins) -> list[float]:
    """Weighted shares for fixed unit cells under the current scenario weights."""

    n_bins = _bin_count(n_bins)
    labels = _integer_ids(labels, "Decile labels", unique=False)
    if any(label < 1 or label > n_bins for label in labels):
        raise ValueError("Decile labels must lie within the declared bins.")
    weights = _weights(weights, len(labels))
    scaled = weights / weights.max()
    total = fsum(scaled)
    return [fsum(scaled[labels == label]) / total for label in range(1, n_bins + 1)]


def _exact_midpoint_labels(sorted_weights, n_bins):
    """Compare float64 dyadic midpoints to bin boundaries without rounding."""

    ratios = [float(weight).as_integer_ratio() for weight in sorted_weights]
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    integer_weights = [
        numerator << (exponent - (denominator.bit_length() - 1))
        for numerator, denominator in ratios
    ]
    # Every denominator is a power of two. The common denominator cancels
    # from the midpoint ratio, leaving exact Python integer prefix sums and
    # division. In particular, equality with k/n_bins enters bin k + 1;
    # there is no tolerance that could promote a genuinely lower midpoint.
    twice_total = 2 * sum(integer_weights)
    preceding = 0
    labels = np.empty(len(integer_weights), dtype=np.int64)
    for row, weight in enumerate(integer_weights):
        labels[row] = min(
            n_bins * (2 * preceding + weight) // twice_total + 1, n_bins
        )
        preceding += weight
    return labels


def weighted_deciles(incomes, weights, tie_ids, n_bins, *, rank_method):
    """Assign individual units by their sorted cumulative-weight midpoints.

    Sort by income then the declared unique tie id. A unit's rank is the
    weight preceding it plus half its own weight, divided by total weight.
    Bins are left-closed: a midpoint exactly on a boundary enters the higher
    bin. Zero-weight units use the cumulative mass at their position, clipped
    to the final bin at rank one. An all-zero population is refused.
    """

    n_bins = _bin_count(n_bins)
    if rank_method != "weighted_midpoint":
        raise ValueError("rank_method must be weighted_midpoint.")
    incomes = _numeric(incomes, "Incomes")
    weights = _weights(weights, len(incomes))
    tie_ids = _tie_ids(tie_ids, len(incomes))
    order = np.lexsort((tie_ids, incomes))
    sorted_labels = _exact_midpoint_labels(weights[order], n_bins)
    labels = np.empty(len(incomes), dtype=np.int64)
    labels[order] = sorted_labels
    return labels, {
        "rank_method": rank_method,
        "n_units": len(labels),
        "zero_weight_units": int(np.count_nonzero(weights == 0)),
        "weighted_shares": decile_weighted_shares(labels, weights, n_bins),
    }


def carry_baseline_deciles(
    unit_ids, baseline_ids, baseline_deciles, n_bins
) -> np.ndarray:
    """Align fixed baseline cells by id, refusing population changes or invalid cells."""

    n_bins = _bin_count(n_bins)
    unit_ids = _integer_ids(unit_ids, "Variant unit ids")
    baseline_ids = _integer_ids(baseline_ids, "Baseline unit ids")
    deciles = _integer_ids(baseline_deciles, "Baseline deciles", unique=False)
    if len(baseline_ids) != len(deciles) or any(
        value < 1 or value > n_bins for value in deciles
    ):
        raise ValueError(
            "Baseline deciles must align to ids and lie in the declared bins."
        )
    if not len(unit_ids) or set(unit_ids) != set(baseline_ids):
        raise ValueError("Variant and baseline must contain exactly the same unit ids.")
    rows = pd.Index(baseline_ids).get_indexer(unit_ids)
    return deciles[rows].astype(np.int64)


def classify_receipt_groups(tables: Mapping[str, pd.DataFrame], spec: Mapping):
    """Classify units by declared any-member assigned receipt and priority.

    Mixed units satisfy more than one reporting predicate. Every input flag
    is validated even if a higher-priority predicate already selects a unit.
    """

    spec = validate_receipt_classifier(spec)
    units, _, order, starts = _relations(tables, spec)
    predicates = {}
    for name, predicate in spec["predicates"].items():
        flags = [
            _assigned_flags(
                _column(tables, predicate["entity"], column),
                f"{predicate['entity']}.{column}",
            )
            for column in predicate["columns"]
        ]
        person_match = np.logical_or.reduce(flags)
        predicates[name] = np.logical_or.reduceat(person_match[order], starts)
    groups = np.full(len(units), spec["default"], dtype=object)
    for name in reversed(spec["priority"]):
        groups[predicates[name]] = name
    result = pd.Series(
        groups, index=pd.Index(units, name=spec["unit_id_column"]), name="receipt_group"
    )
    return result, {
        "n_units": len(units),
        "mixed_units": int(
            np.count_nonzero(np.sum(list(predicates.values()), axis=0) > 1)
        ),
        "group_counts": {
            name: int(np.count_nonzero(groups == name))
            for name in [*spec["priority"], spec["default"]]
        },
        "predicate_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in predicates.items()
        },
    }
