"""Carry assigned household location strings to persons.

``geography.derive@1`` owns household strings only. ``geography.person_location@1``
projects declared household string columns to person string columns through
the structural household membership. Its ``columns`` parameter is a closed
canonical JSON object from household column to person column. Every person
receives exactly their own household's value, in the person table's row
order. A person whose household id names no household row, a missing
household value or a non-string value refuses, naming the rows. The kernel
changes no weights, rows, memberships or household columns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

__all__ = [
    "GEOGRAPHY_PERSON_LOCATION",
    "GeographyPersonLocationKernel",
    "register_person_location_kernels",
]

_DEPENDENCIES = ("numpy", "pandas")
#: The household partition and the person membership that projects it.
_SCHEMA = EntitySchema(group_entities=("household",))
_HOUSEHOLD = "household"


def _columns(params: Mapping[str, object]) -> dict[str, str]:
    if set(params) != {"columns"} or not isinstance(params["columns"], str):
        raise ValueError(
            "geography.person_location@1 requires exactly its canonical columns map."
        )
    try:
        columns = json.loads(params["columns"])
    except ValueError as error:
        raise ValueError("Person location columns must be valid JSON.") from error
    if (
        not isinstance(columns, dict)
        or not columns
        or canonical_json(columns).decode("utf-8") != params["columns"]
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in columns.items()
        )
        or len(set(columns.values())) != len(columns)
    ):
        raise ValueError(
            "Person location columns must be a canonical nonempty map of distinct "
            "household and person column names."
        )
    structural = {
        _SCHEMA.person_id_column,
        _SCHEMA.membership_column(_HOUSEHOLD),
        _SCHEMA.id_column(_HOUSEHOLD),
    }
    if structural & (set(columns) | set(columns.values())):
        raise ValueError("Person location columns cannot name structural columns.")
    return columns


def _refuse(mask, ids, message):
    if mask.any():
        raise ValueError(
            f"geography.person_location@1 {message}; rows {ids[mask].tolist()!r}."
        )


class GeographyPersonLocationKernel(KernelBase):
    """Project household location strings to persons through membership."""

    ref = "geography.person_location@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return source_hash(type(self), canonical_json, dependencies=_DEPENDENCIES)

    def run(self, context: KernelContext) -> KernelResult:
        columns = _columns(context.params)
        declared = {(o.entity, o.column, o.dtype) for o in context.node.outputs}
        if declared != {(_SCHEMA.person_entity, c, "string") for c in columns.values()}:
            raise ValueError(
                "Person location owned outputs must be exactly the mapped person "
                "string columns."
            )
        if (
            _HOUSEHOLD not in context.tables
            or _SCHEMA.person_entity not in context.tables
        ):
            raise ValueError("Person location requires household and person tables.")
        households = context.tables[_HOUSEHOLD]
        persons = context.tables[_SCHEMA.person_entity]
        missing = sorted(set(columns) - set(households.columns))
        if missing:
            raise ValueError(f"Person location requires household columns {missing!r}.")
        household_ids = pd.Index(households[_SCHEMA.id_column(_HOUSEHOLD)])
        if not household_ids.is_unique:
            raise ValueError("Person location requires unique household ids.")
        membership = persons[_SCHEMA.membership_column(_HOUSEHOLD)].to_numpy()
        person_ids = persons[_SCHEMA.person_id_column].to_numpy()
        positions = household_ids.get_indexer(membership)
        _refuse(positions < 0, person_ids, "person household ids match no household")
        index = pd.Index(person_ids, name=_SCHEMA.person_id_column)
        result = {}
        receipt = {}
        for source, target in columns.items():
            values = households[source].to_numpy(dtype=object)
            invalid = np.asarray(
                [not isinstance(value, str) for value in values], dtype=bool
            )
            _refuse(
                invalid,
                household_ids.to_numpy(),
                f"household {source} must hold a string on every household",
            )
            projected = values[positions]
            result[(_SCHEMA.person_entity, target)] = pd.Series(
                pd.array(projected, dtype="string"), index=index, name=target
            )
            receipt[target] = {"source": f"{_HOUSEHOLD}.{source}", "n": len(projected)}
        return KernelResult(columns=result, receipt={"columns": receipt})


GEOGRAPHY_PERSON_LOCATION = GeographyPersonLocationKernel()


def register_person_location_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Install the person-location kernel; repeating it is idempotent."""
    registry.register(GEOGRAPHY_PERSON_LOCATION)
    return registry
