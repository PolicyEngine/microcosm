"""Prepare simple calibration columns from an explicit, engine-free contract.

Geography and canonical concepts must already be present. This helper creates
constants, validated roster counts, conjunctions of declared predicates and
interval indicators classified by :mod:`microcosm.build.reference_intervals`;
receipt assignment and statistical-unit construction belong to their own producers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.calibration_predicates import (
    _finite_number,
    _identity,
    predicate_mask,
    validate_calibration_inputs,
)
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
    required, columns = validate_calibration_inputs(tables, bindings)
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
        prepared.setdefault(entity, {})[name] = predicate_mask(
            out[entity],
            entry.get("predicates"),
            entity=entity,
            name=name,
            validated_inputs={*required[entity], *derived.get(entity, ())},
        )
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
