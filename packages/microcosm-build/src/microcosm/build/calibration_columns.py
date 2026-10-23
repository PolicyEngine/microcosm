"""Prepare simple calibration columns from an explicit, engine-free contract.

Geography and canonical concepts must already be present. This helper creates
only constants and conjunctions of declared predicates; receipt assignment and
statistical-unit construction belong to their own producers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


def prepare_calibration_columns(
    tables: Mapping[str, pd.DataFrame], bindings: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    """Return independent tables with the contract's prepared columns.

    All required inputs are validated before any column is produced. Existing
    prepared columns are recomputed, so a changed geography or age cannot retain
    stale indicators. Missing/unknown inputs never become all-false categories.
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
        if entity not in tables:
            raise ValueError(f"Calibration preparation needs the {entity!r} table.")
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError(f"Calibration required_inputs for {entity!r} are invalid.")
        entity_domains = domains.get(entity, {})
        for name in names:
            if name not in tables[entity]:
                raise ValueError(f"Calibration preparation needs {entity}.{name}.")
            domain = entity_domains.get(name)
            if not isinstance(domain, Mapping):
                raise ValueError(f"Calibration input {entity}.{name} needs a domain.")
            _validate_input(tables[entity][name], domain, f"{entity}.{name}")

    out = {entity: table.copy(deep=True) for entity, table in tables.items()}
    seen: set[tuple[str, str]] = set()
    for entry in columns:
        if not isinstance(entry, Mapping):
            raise ValueError("Calibration column entries must be objects.")
        entity, name = entry.get("entity"), entry.get("name")
        if entity not in required or not isinstance(name, str) or not name:
            raise ValueError("Calibration column needs a declared entity and name.")
        key = (entity, name)
        if key in seen or name in required[entity] or name.endswith("_id"):
            raise ValueError(
                f"Calibration output {entity}.{name} is repeated or structural."
            )
        seen.add(key)
        if "constant" in entry:
            value = entry["constant"]
            if "predicates" in entry or not _finite_number(value):
                raise ValueError(f"Calibration constant {entity}.{name} is invalid.")
            out[entity][name] = value
            continue
        predicates = entry.get("predicates")
        if not isinstance(predicates, (list, tuple)) or not predicates:
            raise ValueError(f"Calibration output {entity}.{name} needs predicates.")
        mask = np.ones(len(out[entity]), dtype=bool)
        for predicate in predicates:
            if not isinstance(predicate, Mapping):
                raise ValueError("Calibration predicates must be objects.")
            column = predicate.get("column")
            if column not in required[entity] or "value" not in predicate:
                raise ValueError(
                    f"Calibration predicate needs a validated {entity} input."
                )
            values = out[entity][column]
            operator, value = predicate.get("operator"), predicate["value"]
            if operator == "eq":
                part = values.eq(value)
            elif operator == "ge":
                part = values.ge(value)
            elif operator == "lt":
                part = values.lt(value)
            elif operator == "in" and isinstance(value, (list, tuple)) and value:
                part = values.isin(value)
            else:
                raise ValueError(f"Unsupported calibration predicate {operator!r}.")
            mask &= part.to_numpy(dtype=bool)
        out[entity][name] = mask
    return out


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def _validate_input(values: pd.Series, domain: Mapping[str, Any], name: str) -> None:
    if values.isna().any():
        raise ValueError(f"Calibration input {name} contains missing values.")
    if domain.get("kind") == "nonnegative_integer":
        valid = all(
            _finite_number(value) and value >= 0 and value == int(value)
            for value in values.tolist()
        )
    elif isinstance(domain.get("values"), (list, tuple)) and domain["values"]:
        allowed = domain["values"]
        valid = all(
            any(type(value) is type(member) and value == member for member in allowed)
            for value in values.tolist()
        )
    else:
        raise ValueError(f"Calibration input {name} has an unsupported domain.")
    if not valid:
        raise ValueError(f"Calibration input {name} is outside its declared domain.")
