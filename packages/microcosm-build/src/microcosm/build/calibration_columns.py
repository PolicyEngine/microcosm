"""Prepare simple calibration columns from an explicit, engine-free contract.

Geography and canonical concepts must already be present. This helper creates
constants, validated roster counts, conjunctions of declared predicates and
interval indicators classified by :mod:`microcosm.build.reference_intervals`;
receipt assignment and statistical-unit construction belong to their own producers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.reference_intervals import (
    interval_indicators,
    validate_interval_document,
)


def prepare_calibration_columns(
    tables: Mapping[str, pd.DataFrame],
    bindings: Mapping[str, Any],
    *,
    intervals: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return independent tables with the contract's prepared columns.

    All required inputs are validated before any column is produced. Existing
    prepared columns are recomputed, so a changed geography or age cannot retain
    stale indicators. Missing/unknown inputs never become all-false categories.
    An interval output names its entity, an effective table of ``intervals``
    and one of its bands; every record of each named table is validated, and
    any refusal raises, before an output exists.
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

    interval_columns = _interval_columns(tables, columns, required, intervals)
    out = {entity: table.copy(deep=True) for entity, table in tables.items()}
    derived = _prepare_roster_counts(out, bindings.get("roster_counts", ()), required)
    prepared: dict[str, dict[str, object]] = {}
    seen: set[tuple[str, str]] = set()
    for entry in columns:
        if not isinstance(entry, Mapping):
            raise ValueError("Calibration column entries must be objects.")
        entity, name = entry.get("entity"), entry.get("name")
        if entity not in required or not isinstance(name, str) or not name:
            raise ValueError("Calibration column needs a declared entity and name.")
        key = (entity, name)
        if (
            key in seen
            or name in required[entity]
            or name in derived.get(entity, ())
            or name.endswith("_id")
        ):
            raise ValueError(
                f"Calibration output {entity}.{name} is repeated or structural."
            )
        seen.add(key)
        if "interval" in entry:
            prepared.setdefault(entity, {})[name] = interval_columns[key]
            continue
        if "constant" in entry:
            value = entry["constant"]
            if "predicates" in entry or not _finite_number(value):
                raise ValueError(f"Calibration constant {entity}.{name} is invalid.")
            prepared.setdefault(entity, {})[name] = value
            continue
        predicates = entry.get("predicates")
        if not isinstance(predicates, (list, tuple)) or not predicates:
            raise ValueError(f"Calibration output {entity}.{name} needs predicates.")
        mask = np.ones(len(out[entity]), dtype=bool)
        for predicate in predicates:
            if not isinstance(predicate, Mapping):
                raise ValueError("Calibration predicates must be objects.")
            column = predicate.get("column")
            if (
                column not in {*required[entity], *derived.get(entity, ())}
                or "value" not in predicate
            ):
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
        prepared.setdefault(entity, {})[name] = mask
    for entity, values in prepared.items():
        original_order = list(out[entity].columns)
        computed = pd.DataFrame(values, index=out[entity].index)
        out[entity] = pd.concat(
            [out[entity].drop(columns=list(values), errors="ignore"), computed], axis=1
        )
        order = original_order + [name for name in values if name not in original_order]
        out[entity] = out[entity].loc[:, order]
    return out


def _interval_columns(
    tables: Mapping[str, pd.DataFrame],
    columns: object,
    required: Mapping[str, Any],
    intervals: Mapping[str, Any] | None,
) -> dict[tuple[str, str], np.ndarray]:
    """Classify every named table once; refuse before any output is produced."""
    entries = [
        entry for entry in columns if isinstance(entry, Mapping) and "interval" in entry
    ]
    if not entries:
        return {}
    if intervals is None:
        raise ValueError("Interval outputs need the effective interval document.")
    document = validate_interval_document(intervals)
    requested: dict[str, list[tuple[str, str, str]]] = {}
    for entry in entries:
        entity, name, binding = (
            entry.get("entity"),
            entry.get("name"),
            entry["interval"],
        )
        if set(entry) != {"entity", "name", "interval"}:
            raise ValueError(
                f"Interval output {entity}.{name} declares only its entity, name "
                "and interval."
            )
        if not isinstance(binding, Mapping) or set(binding) != {"table", "band"}:
            raise ValueError("An interval output names exactly its table and band.")
        table = document.get(binding["table"])
        if table is None or binding["band"] not in table.band_ids:
            raise ValueError(
                f"Interval output {entity}.{name} names an undeclared table or band."
            )
        if entity not in required or entity != table.entity:
            raise ValueError(
                f"Interval output {entity}.{name} must name its table's entity."
            )
        if name != table.indicators[table.band_ids.index(binding["band"])]:
            raise ValueError(
                f"Interval output {entity}.{name} must be its band's declared indicator."
            )
        if (
            table.declaration["source_period"] != table.declaration["target_period"]
            and not table.scaled
        ):
            raise ValueError(
                f"Interval table {table.id!r} has source-period cuts; scale it to its "
                "target period before preparing indicators."
            )
        requested.setdefault(table.id, []).append((entity, name, binding["band"]))
    result: dict[tuple[str, str], np.ndarray] = {}
    computed = {}
    for table_id in requested:
        table = document[table_id]
        data = tables.get(table.entity)
        if data is None or any(
            column not in data
            for column in (table.income_column, table.universe_column)
        ):
            raise ValueError(
                f"Interval table {table_id!r} needs {table.entity}."
                f"{table.income_column} and {table.entity}.{table.universe_column}."
            )
        computed[table_id] = interval_indicators(
            table,
            data[table.income_column].tolist(),
            data[table.universe_column].tolist(),
        )
    for table_id, outputs in requested.items():
        for entity, name, _ in outputs:
            result[entity, name] = computed[table_id][name]
    return result


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


def _prepare_roster_counts(
    tables: dict[str, pd.DataFrame], entries: object, required: Mapping[str, Any]
) -> dict[str, set[str]]:
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Calibration roster_counts must be a list.")
    derived: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Calibration roster count must be an object.")
        entity, members, name = (
            entry.get("entity"),
            entry.get("member_entity"),
            entry.get("name"),
        )
        group_id, member_id, membership = (
            entry.get("entity_id"),
            entry.get("member_id"),
            entry.get("membership"),
        )
        if (
            entity not in required
            or members not in required
            or not isinstance(name, str)
            or not name
            or name.endswith("_id")
            or name in required[entity]
            or name in derived.get(entity, ())
            or group_id not in required[entity]
            or member_id not in required[members]
            or membership not in required[members]
            or not isinstance(entry.get("representation"), str)
            or not entry["representation"]
        ):
            raise ValueError(
                "Calibration roster count needs declared identities and representation."
            )
        groups = [_identity(value) for value in tables[entity][group_id].tolist()]
        identities = [_identity(value) for value in tables[members][member_id].tolist()]
        if len(set(groups)) != len(groups) or len(set(identities)) != len(identities):
            raise ValueError(
                "Calibration roster contains duplicate membership identities."
            )
        counts = dict.fromkeys(groups, 0)
        for value in tables[members][membership].tolist():
            key = _identity(value)
            if key not in counts:
                raise ValueError("Calibration roster contains orphan membership.")
            counts[key] += 1
        if any(count == 0 for count in counts.values()):
            raise ValueError("Calibration roster contains empty households or groups.")
        tables[entity][name] = [counts[key] for key in groups]
        derived.setdefault(entity, set()).add(name)
    return derived


def _validate_input(values: pd.Series, domain: Mapping[str, Any], name: str) -> None:
    if values.isna().any():
        raise ValueError(f"Calibration input {name} contains missing values.")
    if domain.get("kind") == "identity":
        for value in values.tolist():
            _identity(value)
        valid = True
    elif domain.get("kind") == "signed_integer":
        valid = all(
            isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            for value in values.tolist()
        )
    elif domain.get("kind") == "nonnegative_integer":
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
