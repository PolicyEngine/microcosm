"""``measure.combine@1``: declared signed integer-cent components summed into a measure.

A declaration document, never this module, names the entity, every component
column and its sign, the add-backs, the floor, the scale and the three output
columns. For each record the kernel computes, in Python integers,

``x = sum(sign_k * c_k) + sum(max(0, -(sign_j * c_j)) for each add-back j)``

so an add-back component counts only its gains (its losses are not deducted).
It writes ``x`` in cents, a unit column equal to ``max(floor, x) / scale``
(correctly rounded binary64), and a universe status: ``included`` where
``x > 0``, ``excluded`` where ``x <= 0`` and ``unresolved`` where any
component is null. An unresolved record's measure is null and its unit is
NaN; it is never read as zero. A measure outside int64 is refused by name.

The node may declare ``measurement_id``; the kernel carries it into its
receipt for the composition proof that reads it, and computes nothing from it.

Declaration (``measure`` parameter, canonical JSON, with ``measure_sha256``)::

    {"entity": "person",
     "components": [{"column": "...", "sign": 1}, {"column": "...", "sign": -1}],
     "add_backs": ["<a component column>"],
     "floor": <integer cents>, "scale": <positive integer>,
     "outputs": {"measure": "...", "unit": "...", "universe_status": "..."}}
"""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum

import numpy as np
import pandas as pd

import microcosm.build.reference_intervals as reference_intervals_module
from microcosm.build.reference_intervals import UNIVERSE_STATES
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token

from . import graph_inputs
from .graph_inputs import (
    canonical_document_param,
    require_params,
    sha256_param,
    sha256_text,
    string_param,
)

__all__ = [
    "MEASURE_COMBINE",
    "MEASURE_COMBINE_REF",
    "MeasureCombineKernel",
    "combine_measure",
    "register_measure_kernels",
    "validate_measure_declaration",
]

MEASURE_COMBINE_REF = "measure.combine@1"
INCLUDED, EXCLUDED, UNRESOLVED = UNIVERSE_STATES
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_OUTPUT_DTYPES = {
    "measure": frozenset({"Int64", "int64"}),
    "unit": frozenset({"float64"}),
    "universe_status": frozenset({"string"}),
}


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def validate_measure_declaration(document: Mapping[str, object]) -> dict:
    """Check a declaration's shape; every value comes from the document."""

    ref = MEASURE_COMBINE_REF
    if not isinstance(document, Mapping) or set(document) != {
        "entity",
        "components",
        "add_backs",
        "floor",
        "scale",
        "outputs",
    }:
        raise ValueError(
            f"{ref} declares exactly entity, components, add_backs, floor, scale "
            "and outputs."
        )
    if not _name(document["entity"]):
        raise ValueError(f"{ref} entity must be a nonempty name.")
    components = document["components"]
    if (
        not isinstance(components, list)
        or not components
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"column", "sign"}
            or not _name(row["column"])
            or not _integer(row["sign"])
            or row["sign"] not in {-1, 1}
            for row in components
        )
    ):
        raise ValueError(
            f"{ref} components are a nonempty list of {{column, sign}} with sign "
            "1 or -1."
        )
    columns = [row["column"] for row in components]
    if len(set(columns)) != len(columns):
        raise ValueError(f"{ref} component columns must be distinct.")
    add_backs = document["add_backs"]
    if (
        not isinstance(add_backs, list)
        or any(not _name(name) for name in add_backs)
        or len(set(add_backs)) != len(add_backs)
        or not set(add_backs) <= set(columns)
    ):
        raise ValueError(
            f"{ref} add_backs must name distinct declared component columns."
        )
    if not _integer(document["floor"]):
        raise ValueError(f"{ref} floor must be an integer number of minor units.")
    if not _integer(document["scale"]) or document["scale"] <= 0:
        raise ValueError(f"{ref} scale must be a positive integer.")
    if not _INT64_MIN <= document["floor"] <= _INT64_MAX:
        raise ValueError(f"{ref} floor must fit int64.")
    outputs = document["outputs"]
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != set(_OUTPUT_DTYPES)
        or any(not _name(name) for name in outputs.values())
        or len(set(outputs.values())) != len(outputs)
        or set(outputs.values()) & set(columns)
    ):
        raise ValueError(
            f"{ref} outputs name distinct measure, unit and universe_status "
            "columns, none of them a component."
        )
    return {
        "entity": document["entity"],
        "components": [dict(row) for row in components],
        "add_backs": list(add_backs),
        "floor": document["floor"],
        "scale": document["scale"],
        "outputs": dict(outputs),
    }


def _component_values(series: pd.Series, column: str) -> list[int | None]:
    """Exact Python integers, ``None`` where the component is null."""
    dtype = series.dtype
    if isinstance(dtype, pd.Int64Dtype) or (
        isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.integer)
    ):
        return [None if pd.isna(value) else int(value) for value in series.tolist()]
    raise ValueError(
        f"{MEASURE_COMBINE_REF} component {column!r} must be an integer minor-unit "
        f"column, not {dtype}."
    )


def combine_measure(
    declaration: Mapping[str, object],
    columns: Mapping[str, pd.Series],
    weights: np.ndarray,
) -> tuple[list[int | None], list[str], dict]:
    """Combine one entity's components; returns measures, statuses and a receipt.

    ``columns`` holds each declared component; ``weights`` are the entity's
    weights in the same row order. All arithmetic is in Python integers.
    """

    components = declaration["components"]
    add_backs = set(declaration["add_backs"])
    values = {
        row["column"]: _component_values(columns[row["column"]], row["column"])
        for row in components
    }
    size = len(weights)
    if any(len(column) != size for column in values.values()):
        raise ValueError(
            f"{MEASURE_COMBINE_REF} components and weights differ in length."
        )
    measures: list[int | None] = []
    statuses: list[str] = []
    back_totals = {name: 0 for name in declaration["add_backs"]}
    for index in range(size):
        signed = [
            (row["column"], row["sign"] * values[row["column"]][index])
            if values[row["column"]][index] is not None
            else (row["column"], None)
            for row in components
        ]
        if any(value is None for _, value in signed):
            measures.append(None)
            statuses.append(UNRESOLVED)
            continue
        backs = {
            column: max(0, -value) for column, value in signed if column in add_backs
        }
        total = sum(value for _, value in signed) + sum(backs.values())
        if not _INT64_MIN <= total <= _INT64_MAX:
            raise ValueError(
                f"{MEASURE_COMBINE_REF} refuses int64_overflow: record {index} sums "
                f"to {total}, outside int64."
            )
        for column, value in backs.items():
            back_totals[column] += value
        measures.append(total)
        statuses.append(INCLUDED if total > 0 else EXCLUDED)

    resolved = [index for index, status in enumerate(statuses) if status != UNRESOLVED]
    weights = np.asarray(weights, dtype=np.float64)

    def weighted(indices) -> float:
        return fsum(float(weights[index]) for index in indices)

    receipt_components = []
    for row in components:
        column = row["column"]
        signed = [row["sign"] * values[column][index] for index in resolved]
        receipt_components.append(
            {
                "column": column,
                "sign": row["sign"],
                "add_back": column in add_backs,
                "n_null": sum(value is None for value in values[column]),
                "unweighted_signed_total": sum(signed),
                "weighted_signed_total": fsum(
                    float(weights[index]) * value
                    for index, value in zip(resolved, signed, strict=True)
                ),
            }
        )
    negative = [index for index in resolved if measures[index] < 0]
    zero = [index for index in resolved if measures[index] == 0]
    receipt = {
        "entity": declaration["entity"],
        "n": size,
        "components": receipt_components,
        "add_backs": [
            {"column": name, "total": back_totals[name]}
            for name in declaration["add_backs"]
        ],
        "sum_measure": sum(measures[index] for index in resolved),
        "universe_statuses": {
            state: statuses.count(state) for state in UNIVERSE_STATES
        },
        "below_zero": {"n": len(negative), "weighted": weighted(negative)},
        "at_zero": {"n": len(zero), "weighted": weighted(zero)},
        "floor": declaration["floor"],
        "scale": declaration["scale"],
    }
    # The receipt's identities, checked exactly before anything is written.
    if receipt["sum_measure"] != sum(
        row["unweighted_signed_total"] for row in receipt_components
    ) + sum(back_totals.values()):
        raise AssertionError("components and add-backs must sum to the measure.")
    return measures, statuses, receipt


class MeasureCombineKernel(KernelBase):
    """``measure.combine@1``: one measure from declared signed components."""

    ref = MEASURE_COMBINE_REF
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            reference_intervals_module,
            # canonical_document_param checks the measure document's spelling
            # through the encoder (microcosm#931).
            canonical_json,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(
            context,
            ref,
            required=frozenset({"measure", "measure_sha256"}),
            optional=frozenset({"measurement_id"}),
        )
        document = canonical_document_param(context, ref, "measure")
        if sha256_param(context, ref, "measure_sha256") != sha256_text(
            context.params["measure"].encode("utf-8")
        ):
            raise ValueError(f"{ref} measure_sha256 differs from its declaration.")
        declaration = validate_measure_declaration(document)
        measurement_id = (
            string_param(context, ref, "measurement_id")
            if "measurement_id" in context.params
            else None
        )
        entity = declaration["entity"]
        if any(item.rows != ROWS_ALL for item in context.node.inputs):
            raise ValueError(f"{ref} requires whole-row slices.")
        declared_inputs = {
            (item.entity, column)
            for item in context.node.inputs
            for column in item.columns
        }
        expected_inputs = {(entity, row["column"]) for row in declaration["components"]}
        if declared_inputs != expected_inputs:
            raise ValueError(
                f"{ref} inputs must slice exactly the declared components on {entity!r}."
            )
        outputs = declaration["outputs"]
        declared_outputs = {
            (item.entity, item.column): item for item in context.node.outputs
        }
        if set(declared_outputs) != {(entity, name) for name in outputs.values()}:
            raise ValueError(
                f"{ref} outputs must equal the declaration's three columns on "
                f"{entity!r}."
            )
        dtypes = {}
        for role, name in outputs.items():
            item = declared_outputs[entity, name]
            if item.rewrite or item.dtype not in _OUTPUT_DTYPES[role]:
                raise ValueError(
                    f"{ref} output {name!r} ({role}) is a new column of dtype "
                    f"{sorted(_OUTPUT_DTYPES[role])}."
                )
            dtypes[role] = item.dtype
        table = context.tables[entity]
        weights = context.weights[entity].values
        measures, statuses, receipt = combine_measure(
            declaration,
            {row["column"]: table[row["column"]] for row in declaration["components"]},
            weights,
        )
        if dtypes["measure"] == "int64" and any(value is None for value in measures):
            raise ValueError(
                f"{ref} has unresolved records; declare the measure Int64 so they "
                "stay null rather than zero."
            )
        floor, scale = declaration["floor"], declaration["scale"]
        # Python's int / int is correctly rounded for any magnitude.
        units = [
            float("nan") if value is None else max(floor, value) / scale
            for value in measures
        ]
        index = pd.Index(table[f"{entity}_id"].to_numpy(copy=True), name=f"{entity}_id")
        columns = {
            (entity, outputs["measure"]): pd.Series(
                pd.array(measures, dtype=dtype_for_token(dtypes["measure"])),
                index=index,
                name=outputs["measure"],
            ),
            (entity, outputs["unit"]): pd.Series(
                np.asarray(units, dtype=np.float64), index=index, name=outputs["unit"]
            ),
            (entity, outputs["universe_status"]): pd.Series(
                pd.array(statuses, dtype=dtype_for_token("string")),
                index=index,
                name=outputs["universe_status"],
            ),
        }
        receipt = {
            **receipt,
            "outputs": dict(outputs),
            "measurement_id": measurement_id,
        }
        return KernelResult(columns=columns, receipt=receipt)


MEASURE_COMBINE = MeasureCombineKernel()


def register_measure_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register ``measure.combine@1`` explicitly and idempotently."""
    registry.register(MEASURE_COMBINE)
    return registry
