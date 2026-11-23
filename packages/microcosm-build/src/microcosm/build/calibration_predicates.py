"""Validated calibration predicates shared by preparation and receipt assignment.

Receipt assignment precedes calibration preparation, so evaluating a predicate
must validate its declared raw inputs rather than trust existing indicator columns.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


def validate_calibration_inputs(
    tables: Mapping[str, pd.DataFrame] | None, bindings: Mapping[str, Any]
) -> tuple[Mapping[str, Any], list[Any] | tuple[Any, ...]]:
    """Validate every declared input using preparation's original contract.

    Return the declarations so callers can validate their outputs without repeating
    schema or input validation. ``None`` validates the schema without data, for
    activation preflight. No table or column is mutated.
    """
    if bindings.get("schema_version") != 1:
        raise ValueError("Calibration bindings require schema_version 1.")
    required = bindings.get("required_inputs")
    domains = bindings.get("input_domains")
    columns = bindings.get("columns")
    if not isinstance(required, Mapping) or not isinstance(domains, Mapping):
        raise ValueError("Calibration bindings need required_inputs and input_domains.")
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("Calibration bindings need nonempty columns.")
    for entity, names in required.items():
        if tables is not None and entity not in tables:
            raise ValueError(f"Calibration preparation needs the {entity!r} table.")
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError(f"Calibration required_inputs for {entity!r} are invalid.")
        entity_domains = domains.get(entity, {})
        for name in names:
            if tables is not None and name not in tables[entity]:
                raise ValueError(f"Calibration preparation needs {entity}.{name}.")
            domain = (
                entity_domains.get(name)
                if isinstance(entity_domains, Mapping)
                else None
            )
            if not isinstance(domain, Mapping):
                raise ValueError(f"Calibration input {entity}.{name} needs a domain.")
            if tables is None:
                _input_domain_kind(domain, f"{entity}.{name}")
            else:
                _validate_input(tables[entity][name], domain, f"{entity}.{name}")

    return required, columns


def predicate_mask(
    table: pd.DataFrame,
    predicates: object,
    *,
    entity: str,
    name: str,
    validated_inputs: Collection[str],
) -> np.ndarray:
    """Evaluate a conjunction after its caller has validated all named inputs."""
    if not isinstance(predicates, (list, tuple)) or not predicates:
        raise ValueError(f"Calibration output {entity}.{name} needs predicates.")
    mask = np.ones(len(table), dtype=bool)
    for predicate in predicates:
        if not isinstance(predicate, Mapping):
            raise ValueError("Calibration predicates must be objects.")
        column = predicate.get("column")
        if column not in validated_inputs or "value" not in predicate:
            raise ValueError(f"Calibration predicate needs a validated {entity} input.")
        values = table[column]
        operator, value = predicate.get("operator"), predicate["value"]
        if operator == "eq":
            part = values.eq(value)
        elif operator == "ge":
            part = values.ge(value)
        elif operator == "gt":
            part = values.gt(value)
        elif operator == "lt":
            part = values.lt(value)
        elif operator == "in" and isinstance(value, (list, tuple)) and value:
            part = values.isin(value)
        else:
            raise ValueError(f"Unsupported calibration predicate {operator!r}.")
        mask &= part.to_numpy(dtype=bool)
    return mask


def calibration_predicate_mask(
    tables: Mapping[str, pd.DataFrame],
    bindings: Mapping[str, Any],
    *,
    entity: str,
    name: str,
) -> np.ndarray:
    """Resolve one declared predicate filter and validate before evaluating it.

    Receipt strata use raw required inputs only. Roster-derived inputs remain owned
    by calibration preparation; neither stale prepared indicators nor unvalidated
    table columns can stand in for a declared predicate input.
    """
    required, columns = validate_calibration_inputs(tables, bindings)
    matches = [
        entry
        for entry in columns
        if isinstance(entry, Mapping)
        and entry.get("entity") == entity
        and entry.get("name") == name
    ]
    if entity not in required or len(matches) != 1:
        raise ValueError(
            f"Calibration filter {entity}.{name} needs exactly one declared predicate column."
        )
    entry = matches[0]
    if "constant" in entry or "interval" in entry:
        raise ValueError(f"Calibration filter {entity}.{name} must use predicates.")
    return predicate_mask(
        tables[entity],
        entry.get("predicates"),
        entity=entity,
        name=name,
        validated_inputs=required[entity],
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def _identity(value: object) -> tuple[type, object]:
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, Integral))
        or isinstance(value, str)
        and not value
    ):
        raise ValueError(
            "Calibration membership identities must be strings or integers."
        )
    return (str, value) if isinstance(value, str) else (int, int(value))


def _validate_input(values: pd.Series, domain: Mapping[str, Any], name: str) -> None:
    if values.isna().any():
        raise ValueError(f"Calibration input {name} contains missing values.")
    kind = _input_domain_kind(domain, name)
    if kind == "identity":
        for value in values.tolist():
            _identity(value)
        valid = True
    elif kind == "signed_integer":
        valid = all(
            isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            for value in values.tolist()
        )
    elif kind == "nonnegative_integer":
        valid = all(
            _finite_number(value) and value >= 0 and value == int(value)
            for value in values.tolist()
        )
    elif kind == "nonnegative_number":
        valid = all(_finite_number(value) and value >= 0 for value in values.tolist())
    else:
        allowed = domain["values"]
        valid = all(
            any(type(value) is type(member) and value == member for member in allowed)
            for value in values.tolist()
        )
    if not valid:
        raise ValueError(f"Calibration input {name} is outside its declared domain.")


def _input_domain_kind(domain: Mapping[str, Any], name: str) -> str:
    """Share domain-schema validation between activation and table validation."""
    kind = domain.get("kind")
    if kind in (
        "identity",
        "signed_integer",
        "nonnegative_integer",
        "nonnegative_number",
    ):
        return kind
    if isinstance(domain.get("values"), (list, tuple)) and domain["values"]:
        return "values"
    raise ValueError(f"Calibration input {name} has an unsupported domain.")
