"""Income evidence for ``measurement_prefit``: real producers, no placeholders.

R2C-18 wires a ``required_evidence`` alias only where a graph node really
produces that artifact. The gate checks presence by alias, so each kernel here
refuses, or carries the run's own content, on its own:

``transport.income_identity@1``
    Recomputes the stand-in measure's exact identities from the run's columns,
    in Python integers and independently of ``measure.combine@1``. The first
    failing record raises by name, so the artifact exists only when every
    identity holds.
``transport.omission_register@1``
    Binds the declared omission register to the run: its items verbatim (each
    size exactly as declared), beside the totals of what the measure does
    include at pre-fit weights. It adds no model-side figure for an omitted
    item, because the register's sizes mean IRD's.
``transport.native_band_report@1``
    Sets the model, at the run's pre-fit and post-fit weights, beside every
    native row of the publisher's table: the row's count and amount from the
    pinned feed, its cut aged by the coarse table's own factor, and the
    model's weighted count and amount in that band. The native rows must tile
    each coarse band exactly, by cuts and by facts, or the kernel raises. It
    carries no pass or fail.

Each kernel emits one canonical JSON artifact whose header has the shape of
:func:`~microcosm.build.transport.evidence_contract.evidence_header`. The
producer hash is the kernel's own implementation hash: ``evidence_contract``
hashes only the kernels it lists, and this module stays outside the gate
battery's binding closure, so adding it re-keys no battery.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np
import pandas as pd

import microcosm.build.ledger_artifact as ledger_artifact_module
import microcosm.build.ledger_targets as ledger_targets_module
import microcosm.build.reference_intervals as reference_intervals_module
import microcosm.build.transport.evidence_contract as evidence_contract_module
import microcosm.build.transport.measure_kernels as measure_kernels_module
import microcosm.calibrate.artifacts as calibrate_artifacts_module
from microcosm.build.ledger_artifact import LexicalFloat, load_ledger_consumer_artifact
from microcosm.build.ledger_targets import fact_matches_selector
from microcosm.build.reference_intervals import (
    UNIVERSE_STATES,
    boundary_diagnostics,
    classify_records,
    interval_table,
    scale_interval_table,
    source_interval_table,
    validate_interval_document,
)
from microcosm.calibrate.artifacts import decode_problem, decode_solution
from microcosm.frame import EntitySchema
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from microcosm.graph import (
    ROWS_ALL,
    ArtifactType,
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

from . import graph_inputs
from .evidence_contract import frame_column_sha256, param_sha256
from .graph_inputs import (
    canonical_document_param,
    require_params,
    sha256_param,
    sha256_text,
    single_source,
    string_param,
)
from .measure_kernels import validate_measure_declaration

__all__ = [
    "IDENTITY_RECEIPTS_TYPE",
    "INCOME_IDENTITIES",
    "INCOME_IDENTITY",
    "INCOME_IDENTITY_REF",
    "NATIVE_BAND_REPORT",
    "NATIVE_BAND_REPORT_REF",
    "NATIVE_BAND_REPORT_TYPE",
    "NIL_MODEL_REASON",
    "OMISSION_REGISTER",
    "OMISSION_REGISTER_REF",
    "OMISSION_REGISTER_TYPE",
    "WEIGHT_BASES",
    "IncomeIdentityError",
    "IncomeIdentityKernel",
    "NativeBandReportKernel",
    "OmissionRegisterKernel",
    "TilingError",
    "aged_native_table",
    "coarse_members",
    "income_identity_receipts",
    "native_band_report",
    "omission_register_totals",
    "register_income_evidence_kernels",
]

INCOME_IDENTITY_REF = "transport.income_identity@1"
IDENTITY_RECEIPTS_TYPE = ArtifactType("microcosm.transport.income-identity-receipts", 1)
OMISSION_REGISTER_REF = "transport.omission_register@1"
OMISSION_REGISTER_TYPE = ArtifactType("microcosm.transport.omission-register", 1)
NATIVE_BAND_REPORT_REF = "transport.native_band_report@1"
NATIVE_BAND_REPORT_TYPE = ArtifactType("microcosm.transport.native-band-report", 1)
#: The weight vectors the native-band report reads, in report order.
WEIGHT_BASES = ("pre_fit", "post_fit")
#: Why the report gives a null, not a 0, for the model's nil row.
NIL_MODEL_REASON = (
    "Null, not 0: no record in the measure's universe sits at zero, because the "
    "universe excludes every record at or below zero (the identity receipts' "
    "universe_partition). The model has no count comparable to the publisher's "
    "nil row."
)
_VALUE_KINDS = ("count", "money")
_SCALING_KEYS = (
    "factor_id",
    "operational_factor",
    "factor_declaration_sha256",
    "factor_receipt_sha256",
)
#: Fields the native and coarse tables must share for one to tile the other.
_SHARED_TABLE_FIELDS = (
    "entity",
    "currency",
    "input_unit",
    "income_column",
    "universe_column",
    "source_period",
    "target_period",
    "positive_band_closure",
    "negative_income_policy",
)
#: The identities ``transport.income_identity@1`` recomputes, in order. The
#: declaration's ``identity_receipts`` list must name exactly these.
INCOME_IDENTITIES = (
    "component_sum",
    "add_back",
    "unit_rule",
    "universe_partition",
    "int64_range",
)
INCLUDED, EXCLUDED, UNRESOLVED = UNIVERSE_STATES
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_DEPENDENCIES = ("numpy", "pandas")


class IncomeIdentityError(ValueError):
    """One identity fails at one record; the message names both."""


def _header(
    kind: str,
    *,
    kernel: str,
    implementation_hash: str,
    node: str,
    artifacts: Mapping[str, bytes],
    frame: Mapping[str, str],
    params: Mapping[str, object],
) -> dict[str, object]:
    """``evidence_header``'s shape under this kernel's own implementation hash."""

    return {
        "schema_version": 1,
        "kind": kind,
        "producer": {
            "kernel": kernel,
            "implementation_hash": implementation_hash,
            "node": node,
        },
        "inputs": {
            "artifacts": {
                alias: sha256_text(payload)
                for alias, payload in sorted(artifacts.items())
            },
            "frame": dict(sorted(frame.items())),
            "params": {
                name: param_sha256(value) for name, value in sorted(params.items())
            },
        },
    }


def _frame_digests(context: KernelContext) -> dict[str, str]:
    """Each sliced column's digest except the entity id, keyed ``entity.column``."""

    digests = {}
    for entity, table in sorted(context.tables.items()):
        id_column = f"{entity}_id"
        if id_column not in table.columns:
            continue
        for column in table.columns:
            if column != id_column:
                digests[f"{entity}.{column}"] = frame_column_sha256(
                    entity, str(column), table[id_column], table[column]
                )
    return digests


def _integers(series: pd.Series, ref: str, column: str) -> list[int | None]:
    """Exact Python integers, ``None`` where null; refuse a non-integer dtype."""

    dtype = series.dtype
    if isinstance(dtype, pd.Int64Dtype) or (
        isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.integer)
    ):
        return [None if pd.isna(value) else int(value) for value in series.tolist()]
    raise ValueError(f"{ref} column {column!r} must be integer cents, not {dtype}.")


def _statuses(series: pd.Series) -> list[str | None]:
    return [None if pd.isna(value) else str(value) for value in series.tolist()]


def _declared_identities(document: Mapping[str, object], ref: str) -> list[dict]:
    rows = document.get("identities")
    if (
        set(document) != {"description", "identities"}
        or not isinstance(document.get("description"), str)
        or not isinstance(rows, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"id", "statement"}
            or not isinstance(row["statement"], str)
            or not row["statement"]
            for row in rows
        )
    ):
        raise ValueError(
            f"{ref} identities declare a description and a list of {{id, statement}}."
        )
    if [row["id"] for row in rows] != list(INCOME_IDENTITIES):
        raise ValueError(
            f"{ref} identities must name exactly {list(INCOME_IDENTITIES)}, in order."
        )
    return [dict(row) for row in rows]


def income_identity_receipts(
    declaration: Mapping[str, object],
    columns: Mapping[str, pd.Series],
    ids: Sequence[object],
    *,
    ref: str = INCOME_IDENTITY_REF,
) -> dict[str, object]:
    """Check every identity on every record; return the run's exact totals.

    ``columns`` holds each component and the three declared outputs of
    ``declaration`` (a ``measure.combine@1`` declaration). Raises
    :class:`IncomeIdentityError` at the first record where an identity fails.
    """

    components = declaration["components"]
    add_backs = list(declaration["add_backs"])
    outputs = declaration["outputs"]
    floor, scale = declaration["floor"], declaration["scale"]
    values = {
        row["column"]: _integers(columns[row["column"]], ref, row["column"])
        for row in components
    }
    measures = _integers(columns[outputs["measure"]], ref, outputs["measure"])
    unit_series = columns[outputs["unit"]]
    if unit_series.dtype != np.float64:
        raise ValueError(f"{ref} unit column {outputs['unit']!r} must be float64.")
    units = unit_series.to_numpy(dtype=np.float64).tolist()
    statuses = _statuses(columns[outputs["universe_status"]])
    size = len(ids)
    if any(len(column) != size for column in (*values.values(), measures, units)):
        raise ValueError(f"{ref} columns differ in length.")

    def fail(identity: str, index: int, detail: str) -> IncomeIdentityError:
        return IncomeIdentityError(
            f"{ref} identity {identity} fails at record {index} "
            f"({declaration['entity']} id {ids[index]!r}): {detail}"
        )

    signed_totals = {row["column"]: 0 for row in components}
    back_totals = {column: 0 for column in add_backs}
    counts = {state: 0 for state in UNIVERSE_STATES}
    sum_measure = 0
    for index in range(size):
        raw = [values[row["column"]][index] for row in components]
        status, measure, unit = statuses[index], measures[index], units[index]
        if any(value is None for value in raw):
            if status != UNRESOLVED:
                raise fail(
                    "universe_partition",
                    index,
                    f"status {status!r} with a null component",
                )
            if measure is not None:
                raise fail("component_sum", index, "a measure with a null component")
            if not math.isnan(unit):
                raise fail("unit_rule", index, f"unit {unit!r} with a null component")
            counts[UNRESOLVED] += 1
            continue
        signed = {
            row["column"]: row["sign"] * value
            for row, value in zip(components, raw, strict=True)
        }
        backs = {column: max(0, -signed[column]) for column in add_backs}
        total = sum(signed.values()) + sum(backs.values())
        if not _INT64_MIN <= total <= _INT64_MAX:
            raise fail("int64_range", index, f"the components sum to {total}")
        if measure != total:
            raise fail(
                "component_sum", index, f"measure {measure!r}, components {total}"
            )
        expected = INCLUDED if total > 0 else EXCLUDED
        if status != expected:
            raise fail(
                "universe_partition",
                index,
                f"status {status!r} where the measure {total} gives {expected!r}",
            )
        # Python's int / int is correctly rounded, so the comparison is exact;
        # float.hex also tells -0.0 from 0.0.
        rule = max(floor, total) / scale
        if not isinstance(unit, float) or unit.hex() != rule.hex():
            raise fail("unit_rule", index, f"unit {unit!r}, rule {rule!r}")
        for column, value in signed.items():
            signed_totals[column] += value
        for column, value in backs.items():
            back_totals[column] += value
        counts[status] += 1
        sum_measure += total
    if sum(counts.values()) != size:
        raise IncomeIdentityError(f"{ref} identity universe_partition fails in total.")
    if sum_measure != sum(signed_totals.values()) + sum(back_totals.values()):
        raise IncomeIdentityError(f"{ref} identity add_back fails in total.")
    return {
        "entity": declaration["entity"],
        "n": size,
        "universe_statuses": counts,
        "unresolved": counts[UNRESOLVED],
        "components": [
            {
                "column": row["column"],
                "sign": row["sign"],
                "add_back": row["column"] in back_totals,
                "n_null": sum(value is None for value in values[row["column"]]),
                "resolved_signed_total": signed_totals[row["column"]],
            }
            for row in components
        ],
        "add_backs": [
            {"column": column, "total": back_totals[column]} for column in add_backs
        ],
        "sum_measure": sum_measure,
        "floor": floor,
        "scale": scale,
        "outputs": dict(outputs),
    }


class IncomeIdentityKernel(KernelBase):
    """``transport.income_identity@1``: the stand-in measure's exact identities."""

    ref = INCOME_IDENTITY_REF
    artifact = "identity_receipts"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            measure_kernels_module,
            evidence_contract_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(
            context,
            ref,
            required=frozenset(
                {"measure", "measure_sha256", "measurement_id", "identities"}
            ),
        )
        document = canonical_document_param(context, ref, "measure")
        if sha256_param(context, ref, "measure_sha256") != sha256_text(
            context.params["measure"].encode("utf-8")
        ):
            raise ValueError(f"{ref} measure_sha256 differs from its declaration.")
        declaration = validate_measure_declaration(document)
        measurement_id = string_param(context, ref, "measurement_id")
        identities = _declared_identities(
            canonical_document_param(context, ref, "identities"), ref
        )
        entity = declaration["entity"]
        outputs = declaration["outputs"]
        if context.node.outputs:
            raise ValueError(f"{ref} writes no columns.")
        if [item.name for item in context.node.artifact_outputs] != [self.artifact]:
            raise ValueError(f"{ref} declares one artifact, {self.artifact!r}.")
        if context.node.artifact_outputs[0].type != IDENTITY_RECEIPTS_TYPE:
            raise ValueError(f"{ref} declares {self.artifact!r} as its own type.")
        if any(item.rows != ROWS_ALL for item in context.node.inputs):
            raise ValueError(f"{ref} requires whole-row slices.")
        declared = {
            (item.entity, column)
            for item in context.node.inputs
            for column in item.columns
        }
        expected = {(entity, row["column"]) for row in declaration["components"]} | {
            (entity, name) for name in outputs.values()
        }
        if declared != expected:
            raise ValueError(
                f"{ref} inputs must slice exactly the declared components and the "
                f"three outputs on {entity!r}."
            )
        table = context.tables[entity]
        names = [row["column"] for row in declaration["components"]]
        receipts = income_identity_receipts(
            declaration,
            {name: table[name] for name in (*names, *outputs.values())},
            table[f"{entity}_id"].tolist(),
        )
        params = {
            name: context.params[name]
            for name in ("measure", "measure_sha256", "measurement_id", "identities")
        }
        document = {
            **_header(
                "transport_income_identity",
                kernel=ref,
                implementation_hash=self.implementation_hash(),
                node=context.node.id,
                artifacts={},
                frame=_frame_digests(context),
                params=params,
            ),
            "measurement_id": measurement_id,
            "measure_sha256": context.params["measure_sha256"],
            "identities": identities,
            **receipts,
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "n": receipts["n"],
                "unresolved": receipts["unresolved"],
                "artifact_sha256": sha256_text(payload),
            },
        )


INCOME_IDENTITY = IncomeIdentityKernel()


def _register_items(document: Mapping[str, object], ref: str) -> dict[str, object]:
    items = document.get("items")
    if (
        set(document) != {"description", "items"}
        or not isinstance(document.get("description"), str)
        or not isinstance(items, list)
        or not items
        or any(
            not isinstance(row, Mapping)
            or not {"item", "size", "reason"} <= set(row)
            or not isinstance(row["item"], str)
            or not row["item"]
            for row in items
        )
    ):
        raise ValueError(
            f"{ref} register declares a description and a nonempty list of items, "
            "each with item, size and reason."
        )
    names = [row["item"] for row in items]
    if len(set(names)) != len(names):
        raise ValueError(f"{ref} register items must be distinct.")
    return {"description": document["description"], "items": [dict(r) for r in items]}


def _weight_vector_sha256(ids: Sequence[object], weights: np.ndarray) -> str:
    """SHA-256 of the weights as little-endian float64, in ascending id order."""

    order = np.argsort(np.asarray(ids), kind="stable")
    values = np.asarray(weights, dtype="<f8")[order]
    return sha256_text(values.tobytes())


def omission_register_totals(
    declaration: Mapping[str, object],
    columns: Mapping[str, pd.Series],
    weights: np.ndarray,
    *,
    ref: str = OMISSION_REGISTER_REF,
) -> dict[str, object]:
    """What the measure includes, at the given weights: exact and weighted totals.

    Signed totals run over resolved records (every component present), as
    ``measure.combine@1``'s receipt does; ``below_zero``, ``at_zero`` and
    ``unresolved`` read the measure column itself.
    """

    components = declaration["components"]
    add_backs = list(declaration["add_backs"])
    values = {
        row["column"]: _integers(columns[row["column"]], ref, row["column"])
        for row in components
    }
    measure_column = declaration["outputs"]["measure"]
    measures = _integers(columns[measure_column], ref, measure_column)
    weights = np.asarray(weights, dtype=np.float64)
    size = len(weights)
    if any(len(column) != size for column in (*values.values(), measures)):
        raise ValueError(f"{ref} columns and weights differ in length.")
    resolved = [
        index
        for index in range(size)
        if all(values[row["column"]][index] is not None for row in components)
    ]

    def mass(indices) -> dict[str, object]:
        return {
            "n": len(indices),
            "weighted": math.fsum(float(weights[index]) for index in indices),
        }

    totals = []
    for row in components:
        column = row["column"]
        signed = [row["sign"] * values[column][index] for index in resolved]
        totals.append(
            {
                "column": column,
                "sign": row["sign"],
                "add_back": column in add_backs,
                "unweighted_signed_total": sum(signed),
                "weighted_signed_total": math.fsum(
                    float(weights[index]) * value
                    for index, value in zip(resolved, signed, strict=True)
                ),
            }
        )
    backs = []
    for column in add_backs:
        sign = next(row["sign"] for row in components if row["column"] == column)
        gains = [max(0, -(sign * values[column][index])) for index in resolved]
        backs.append(
            {
                "column": column,
                "total": sum(gains),
                "weighted_total": math.fsum(
                    float(weights[index]) * value
                    for index, value in zip(resolved, gains, strict=True)
                ),
            }
        )
    return {
        "entity": declaration["entity"],
        "n": size,
        "components": totals,
        "add_backs": backs,
        "below_zero": mass(
            [i for i in range(size) if measures[i] is not None and measures[i] < 0]
        ),
        "at_zero": mass([i for i in range(size) if measures[i] == 0]),
        "unresolved": mass([i for i in range(size) if measures[i] is None]),
    }


class OmissionRegisterKernel(KernelBase):
    """``transport.omission_register@1``: the declared register, bound to the run."""

    ref = OMISSION_REGISTER_REF
    artifact = "omission_register"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            measure_kernels_module,
            evidence_contract_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        names = (
            "register",
            "income_measurements_sha256",
            "measure",
            "measure_sha256",
            "measurement_id",
        )
        require_params(context, ref, required=frozenset(names))
        register = _register_items(
            canonical_document_param(context, ref, "register"), ref
        )
        resource_sha256 = sha256_param(context, ref, "income_measurements_sha256")
        document = canonical_document_param(context, ref, "measure")
        if sha256_param(context, ref, "measure_sha256") != sha256_text(
            context.params["measure"].encode("utf-8")
        ):
            raise ValueError(f"{ref} measure_sha256 differs from its declaration.")
        declaration = validate_measure_declaration(document)
        measurement_id = string_param(context, ref, "measurement_id")
        entity = declaration["entity"]
        measure_column = declaration["outputs"]["measure"]
        if context.node.outputs:
            raise ValueError(f"{ref} writes no columns.")
        if [item.name for item in context.node.artifact_outputs] != [self.artifact]:
            raise ValueError(f"{ref} declares one artifact, {self.artifact!r}.")
        if context.node.artifact_outputs[0].type != OMISSION_REGISTER_TYPE:
            raise ValueError(f"{ref} declares {self.artifact!r} as its own type.")
        if any(item.rows != ROWS_ALL for item in context.node.inputs):
            raise ValueError(f"{ref} requires whole-row slices.")
        declared = {
            (item.entity, column)
            for item in context.node.inputs
            for column in item.columns
        }
        component_names = [row["column"] for row in declaration["components"]]
        expected = {(entity, name) for name in (*component_names, measure_column)}
        if declared != expected:
            raise ValueError(
                f"{ref} inputs must slice exactly the declared components and the "
                f"measure on {entity!r}."
            )
        table = context.tables[entity]
        if entity not in context.weights:
            raise ValueError(f"{ref} needs the population's resolved {entity} weights.")
        weights = context.weights[entity].values
        totals = omission_register_totals(
            declaration,
            {name: table[name] for name in (*component_names, measure_column)},
            weights,
        )
        params = {name: context.params[name] for name in names}
        body = {
            **_header(
                "transport_omission_register",
                kernel=ref,
                implementation_hash=self.implementation_hash(),
                node=context.node.id,
                artifacts={},
                frame=_frame_digests(context),
                params=params,
            ),
            "measurement_id": measurement_id,
            "measure_sha256": context.params["measure_sha256"],
            "income_measurements_sha256": resource_sha256,
            "register_sha256": sha256_text(context.params["register"].encode("utf-8")),
            "register": register,
            "weights": {
                "entity": entity,
                "basis": "pre_fit",
                "sha256": _weight_vector_sha256(
                    table[f"{entity}_id"].to_numpy(), weights
                ),
            },
            "included": totals,
        }
        payload = canonical_json(body)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "items": len(register["items"]),
                "artifact_sha256": sha256_text(payload),
            },
        )


OMISSION_REGISTER = OmissionRegisterKernel()


# ---------------------------------------------------------------------------
# transport.native_band_report@1
# ---------------------------------------------------------------------------


class TilingError(ValueError):
    """The native rows don't tile a coarse band, by cuts or by facts."""


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _exact(value: object, label: str) -> Fraction:
    """An exact number from a fact value: an integer, a lexical decimal or text."""

    if isinstance(value, LexicalFloat):
        token = value.lexical_token
    elif isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{label} must be an exact number, not {value!r}.")
    else:
        token = str(value)
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be an exact number, not {value!r}.") from error


def aged_native_table(
    intervals: Mapping[str, object], native_table: str, coarse_table: str
) -> tuple[dict, dict]:
    """The native table at the coarse table's own scaling, and that scaling.

    The effective interval document scales only the tables a reference
    profile names, and passes every other table through at its source cuts.
    The coarse table's ``scaling`` block records the exact factor that aged
    it. Rescaling the coarse table's source with that block must reproduce
    the coarse table, and the same block then ages the native table. A
    native table that a profile already scaled must carry that same block.
    """

    tables = validate_interval_document(intervals)
    raw = {row["id"]: row for row in intervals["tables"]}
    if native_table == coarse_table:
        raise ValueError("The native and coarse tables must differ.")
    for table_id in (native_table, coarse_table):
        if table_id not in tables:
            raise ValueError(f"The effective intervals hold no table {table_id!r}.")
    native, coarse = raw[native_table], raw[coarse_table]
    differ = [key for key in _SHARED_TABLE_FIELDS if native.get(key) != coarse.get(key)]
    if differ:
        raise ValueError(
            f"Native table {native_table!r} and coarse table {coarse_table!r} "
            f"differ in {differ}."
        )
    scaling = coarse.get("scaling") if tables[coarse_table].scaled else None
    if not isinstance(scaling, Mapping):
        raise ValueError(
            f"Coarse table {coarse_table!r} carries no scaling to age the native "
            "rows with."
        )
    block = {key: copy.deepcopy(scaling[key]) for key in _SCALING_KEYS}
    keys = {
        "factor": block["operational_factor"],
        "factor_id": block["factor_id"],
        "factor_declaration_sha256": block["factor_declaration_sha256"],
        "factor_receipt_sha256": block["factor_receipt_sha256"],
    }
    rescaled = scale_interval_table(source_interval_table(coarse), **keys)
    if canonical_json(rescaled) != canonical_json(coarse):
        raise ValueError(
            f"Rescaling coarse table {coarse_table!r}'s source with its own "
            "scaling doesn't reproduce it."
        )
    if tables[native_table].scaled:
        theirs = native.get("scaling", {})
        if {key: theirs.get(key) for key in _SCALING_KEYS} != block:
            raise ValueError(
                f"Native table {native_table!r} is scaled by another factor than "
                f"coarse table {coarse_table!r}."
            )
        return copy.deepcopy(dict(native)), block
    return scale_interval_table(native, **keys), block


def _positive_intervals(declaration: Mapping) -> list[tuple[str, int, int | None]]:
    """Each positive band's source interval ``(lower, upper]``; ``None`` is open."""

    lower = declaration["first_lower_cut_cents"]
    rows = []
    for band in declaration["bands"]:
        if band["kind"] == "positive":
            rows.append((band["id"], lower, band["upper_cent"]))
            lower = band["upper_cent"]
    return rows


def coarse_members(
    native_source: Mapping, coarse_source: Mapping
) -> dict[str, list[str]]:
    """The native bands each coarse band sums, read from the two source tables.

    A native band belongs to the coarse band whose interval contains its own.
    Native bands wholly below the coarse table's first cut belong to none.
    Raises :class:`TilingError` when a native band crosses a coarse cut, or
    when a coarse band's members don't cover its interval exactly.
    """

    coarse = _positive_intervals(coarse_source)
    first = coarse[0][1]
    members: dict[str, list[str]] = {name: [] for name, _, _ in coarse}
    spans: dict[str, list[tuple[int, int | None]]] = {name: [] for name, _, _ in coarse}

    def inside(lower, upper, coarse_lower, coarse_upper) -> bool:
        return lower >= coarse_lower and (
            coarse_upper is None or (upper is not None and upper <= coarse_upper)
        )

    for band, lower, upper in _positive_intervals(native_source):
        if upper is not None and upper <= first:
            continue
        owners = [
            name
            for name, coarse_lower, coarse_upper in coarse
            if inside(lower, upper, coarse_lower, coarse_upper)
        ]
        if len(owners) != 1:
            raise TilingError(
                f"Native band {band!r} ({lower}, {upper}] crosses a coarse cut."
            )
        members[owners[0]].append(band)
        spans[owners[0]].append((lower, upper))
    for name, coarse_lower, coarse_upper in coarse:
        span = spans[name]
        if not span or span[0][0] != coarse_lower or span[-1][1] != coarse_upper:
            raise TilingError(
                f"The native bands don't cover coarse band {name!r} "
                f"({coarse_lower}, {coarse_upper}] exactly."
            )
    return members


def _native_references(
    document: Mapping[str, object], table: Mapping
) -> dict[str, dict[str, Mapping]]:
    """Each native band's one count and one money diagnostic reference."""

    rows = document.get("diagnostic_references")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The diagnostic references declare a nonempty list.")
    bands = {band["indicator"]: band["id"] for band in table["bands"]}
    found: dict[str, dict[str, Mapping]] = {band: {} for band in bands.values()}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each diagnostic reference is an object.")
        name = row.get("name")
        band = bands.get(row.get("filter"))
        metadata = row.get("metadata")
        kind = (
            metadata.get("reference_value_kind")
            if isinstance(metadata, Mapping)
            else None
        )
        selector = row.get("ledger_selector")
        if band is None:
            raise ValueError(
                f"Diagnostic reference {name!r} filters on no band of the native table."
            )
        if kind not in _VALUE_KINDS or kind in found[band]:
            raise ValueError(
                f"Diagnostic reference {name!r} must be its band's one count or one "
                "money row."
            )
        if (
            not isinstance(selector, Mapping)
            or not isinstance(selector.get("aggregate_fact_key"), str)
            or not selector["aggregate_fact_key"]
        ):
            raise ValueError(
                f"Diagnostic reference {name!r} must pin one fact by aggregate_fact_key."
            )
        found[band][kind] = row
    missing = sorted(
        band for band, kinds in found.items() if set(kinds) != set(_VALUE_KINDS)
    )
    if missing:
        raise ValueError(f"Native bands {missing} lack a count or a money reference.")
    return found


def _coarse_references(
    document: Mapping[str, object], table: Mapping
) -> dict[tuple[str, str], Mapping]:
    """Each coarse band's count and money reference rows, by the band's names."""

    rows = document.get("target_references")
    if not isinstance(rows, list):
        raise ValueError("The target references declare a list.")
    by_name = {row.get("name"): row for row in rows if isinstance(row, Mapping)}
    found = {}
    for band in table["bands"]:
        for kind, key in (("count", "count_reference"), ("money", "amount_reference")):
            row = by_name.get(band.get(key))
            if row is None or not isinstance(row.get("ledger_selector"), Mapping):
                raise ValueError(
                    f"Coarse band {band['id']!r} names no {kind} reference with a "
                    "ledger selector."
                )
            found[band["id"], kind] = row
    return found


def _matching_facts(
    facts: Sequence[Mapping], selectors: Mapping[str, Mapping]
) -> dict[str, list[Mapping]]:
    """Every fact each named selector matches.

    Facts are first narrowed to the selector's own ``record_set_id``. That
    narrowing loses nothing, because a selector matches only facts that also
    match each of its fields alone.
    """

    pools: dict[object, Sequence[Mapping]] = {}
    found = {}
    for name, selector in selectors.items():
        record_set = selector.get("record_set_id")
        if record_set not in pools:
            pools[record_set] = (
                [
                    fact
                    for fact in facts
                    if fact_matches_selector(fact, {"record_set_id": record_set})
                ]
                if record_set
                else facts
            )
        found[name] = [
            fact for fact in pools[record_set] if fact_matches_selector(fact, selector)
        ]
    return found


def _fact_key(fact: Mapping, label: str) -> str:
    key = fact.get("aggregate_fact_key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"{label} matched a fact without an aggregate_fact_key.")
    return key


def _fact_value(fact: Mapping, kind: str, currency: str, label: str) -> Fraction:
    """A count as a nonnegative integer; an amount in the table's currency."""

    value = _exact(fact.get("value"), label)
    if kind == "count":
        if value.denominator != 1 or value < 0:
            raise ValueError(
                f"{label} must be a nonnegative integer count, not {value}."
            )
        return value
    measure = fact.get("observed_measure")
    unit = measure.get("unit") if isinstance(measure, Mapping) else None
    if unit != currency.lower():
        raise ValueError(f"{label} is in {unit!r}, not the table's {currency!r}.")
    return value


def native_band_report(
    intervals: Mapping[str, object],
    *,
    native_table: str,
    coarse_table: str,
    references: Mapping[str, object],
    diagnostic_references: Mapping[str, object],
    facts: Sequence[Mapping],
    person_ids: Sequence[object],
    amounts: Sequence[object],
    statuses: Sequence[object],
    source_households: Sequence[object],
    weights: Mapping[str, tuple[np.ndarray, str]],
    measurement_id: str,
) -> dict[str, object]:
    """The model against every native row, beside the publisher's own values.

    ``intervals`` is the effective interval document calibration prepared
    with. ``weights`` maps each of :data:`WEIGHT_BASES` to a record-aligned
    weight vector and its weight kind. Raises :class:`TilingError` unless
    the native rows tile every coarse band exactly, by cuts and by facts.
    """

    if set(weights) != set(WEIGHT_BASES):
        raise ValueError(f"The report reads exactly the {list(WEIGHT_BASES)} weights.")
    aged, scaling = aged_native_table(intervals, native_table, coarse_table)
    table = interval_table(aged)
    coarse = {row["id"]: row for row in intervals["tables"]}[coarse_table]
    members = coarse_members(source_interval_table(aged), source_interval_table(coarse))
    owner = {band: name for name, bands in members.items() for band in bands}
    factor = Fraction(
        scaling["operational_factor"]["numerator"],
        scaling["operational_factor"]["denominator"],
    )
    currency = aged["currency"]
    natives = _native_references(diagnostic_references, aged)
    coarse_rows = _coarse_references(references, coarse)
    selectors = {
        ("native", band, kind): row["ledger_selector"]
        for band, kinds in natives.items()
        for kind, row in kinds.items()
    }
    selectors.update(
        {
            ("coarse", band, kind): row["ledger_selector"]
            for (band, kind), row in coarse_rows.items()
        }
    )
    matched = _matching_facts(facts, selectors)
    pinned: dict[tuple[str, str], Mapping] = {}
    for band, kinds in natives.items():
        for kind, row in kinds.items():
            found = matched["native", band, kind]
            if len(found) != 1:
                raise ValueError(
                    f"Diagnostic reference {row['name']!r} matches {len(found)} facts "
                    "in the pinned feed; it must pin exactly one."
                )
            pinned[band, kind] = found[0]
    for kind in _VALUE_KINDS:
        keys = [_fact_key(pinned[band, kind], f"{kind} rows") for band in natives]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Two native {kind} rows pin the same fact.")
    values = {
        (band, kind): _fact_value(fact, kind, currency, f"Native {band} {kind} fact")
        for (band, kind), fact in pinned.items()
    }

    coarse_report = []
    for band in coarse["bands"]:
        name = band["id"]
        totals = {}
        for kind in _VALUE_KINDS:
            label = f"Coarse band {name!r}'s {kind} reference"
            found = matched["coarse", name, kind]
            keys = [_fact_key(fact, label) for fact in found]
            member_keys = [
                _fact_key(pinned[member, kind], label) for member in members[name]
            ]
            if len(set(keys)) != len(keys) or set(keys) != set(member_keys):
                raise TilingError(
                    f"{label} matches {len(keys)} facts, not exactly its "
                    f"{len(member_keys)} member native rows' facts."
                )
            total = sum(
                (_fact_value(fact, kind, currency, label) for fact in found),
                Fraction(0),
            )
            if total != sum(
                (values[member, kind] for member in members[name]), Fraction(0)
            ):
                raise TilingError(f"{label} doesn't sum to its member native rows.")
            totals[kind] = total
        coarse_report.append(
            {
                "band": name,
                "source_label": band.get("source_label"),
                "members": list(members[name]),
                "count_reference": coarse_rows[name, "count"]["name"],
                "amount_reference": coarse_rows[name, "money"]["name"],
                "publisher": {
                    "count": int(totals["count"]),
                    "amount": _rational(totals["money"]),
                    "aged_amount": _rational(totals["money"] * factor),
                },
            }
        )

    diagnostics = {
        basis: boundary_diagnostics(
            table,
            person_ids,
            amounts,
            statuses,
            vector,
            weight_record_ids=person_ids,
            weight_kind=kind,
        )
        for basis, (vector, kind) in weights.items()
    }
    positions = classify_records(table, amounts, statuses)
    if len(source_households) != len(positions):
        raise ValueError("Source households must be record-aligned.")
    supports: dict[int, set] = {}
    for position, household in zip(positions, source_households, strict=True):
        supports.setdefault(position, set()).add(household)
    rows = []
    for index, band in enumerate(aged["bands"]):
        measures = {
            basis: diagnostics[basis]["bands"][index]["declared"]
            for basis in WEIGHT_BASES
        }
        empty_nil = index == table.nil_index and all(
            measure["records"] == 0 for measure in measures.values()
        )
        positive = band["kind"] == "positive"
        rows.append(
            {
                "band": band["id"],
                "kind": band["kind"],
                "source_label": band.get("source_label"),
                "source_upper_cent": band.get("source_upper_cent")
                if positive
                else None,
                "aged_upper_cent": band.get("upper_cent") if positive else None,
                "coarse_band": owner.get(band["id"]),
                "publisher": {
                    "count": int(values[band["id"], "count"]),
                    "amount": _rational(values[band["id"], "money"]),
                    "aged_amount": _rational(values[band["id"], "money"] * factor),
                    "count_reference": natives[band["id"]]["count"]["name"],
                    "amount_reference": natives[band["id"]]["money"]["name"],
                    "count_fact": _fact_key(pinned[band["id"], "count"], "count rows"),
                    "amount_fact": _fact_key(pinned[band["id"], "money"], "money rows"),
                },
                "model": None if empty_nil else measures,
                "model_reason": NIL_MODEL_REASON if empty_nil else None,
                "source_households": None
                if empty_nil
                else len(supports.get(index, ())),
            }
        )
    mass = {
        basis: {
            key: report[key]
            for key in (
                "weight_kind",
                "records",
                "included_records",
                "below_first_cut",
                "floored_negative_income",
                "cuts",
                "digests",
            )
        }
        for basis, report in diagnostics.items()
    }
    return {
        "native_table": native_table,
        "coarse_table": coarse_table,
        "scaling": scaling,
        "units": {
            "publisher_amount": currency,
            "model_amount": aged["input_unit"],
            "model_column": aged["income_column"],
        },
        "disclosure": (
            f"The model columns read {aged['income_column']}, the stand-in measure "
            f"{measurement_id}, not the publisher's taxable income, at the run's "
            "pre-fit and post-fit weights. Each cut is the native table's source "
            "cut aged by the coarse table's own factor. Publisher counts and "
            "amounts are the pinned feed's source-period facts, with each amount "
            "also shown aged by that factor and each count left unchanged. The "
            "native rows stay diagnostic (R2C-4): measurement_prefit checks only "
            "that the battery wires this report, no gate reads its rows, and it "
            "carries no pass or fail."
        ),
        "rows": rows,
        "coarse": coarse_report,
        "mass": mass,
    }


class NativeBandReportKernel(KernelBase):
    """``transport.native_band_report@1``: the model against every native row."""

    ref = NATIVE_BAND_REPORT_REF
    artifact = "native_band_report"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=_DEPENDENCIES,
    )
    _names = (
        "native_table",
        "coarse_table",
        "references",
        "diagnostic_references",
        "diagnostic_references_sha256",
        "chronicle_feed_pin",
        "weight_entity",
        "measurement_id",
    )
    _aliases = ("preparation", "problem", "solution")

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            evidence_contract_module,
            reference_intervals_module,
            ledger_artifact_module,
            ledger_targets_module,
            calibrate_artifacts_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(context, ref, required=frozenset(self._names))
        native_id = string_param(context, ref, "native_table")
        coarse_id = string_param(context, ref, "coarse_table")
        weight_entity = string_param(context, ref, "weight_entity")
        measurement_id = string_param(context, ref, "measurement_id")
        references = canonical_document_param(context, ref, "references")
        diagnostics = canonical_document_param(context, ref, "diagnostic_references")
        diagnostics_sha256 = sha256_param(context, ref, "diagnostic_references_sha256")
        feed_pin = canonical_document_param(context, ref, "chronicle_feed_pin")
        if context.node.outputs:
            raise ValueError(f"{ref} writes no columns.")
        if [item.name for item in context.node.artifact_outputs] != [self.artifact]:
            raise ValueError(f"{ref} declares one artifact, {self.artifact!r}.")
        if context.node.artifact_outputs[0].type != NATIVE_BAND_REPORT_TYPE:
            raise ValueError(f"{ref} declares {self.artifact!r} as its own type.")
        edges = {edge.name for edge in context.node.artifact_inputs}
        if edges != set(self._aliases) or set(context.artifacts) != set(self._aliases):
            raise ValueError(f"{ref} reads exactly {list(self._aliases)} artifacts.")
        source = single_source(context, ref)
        payloads = {alias: context.artifacts[alias].payload for alias in self._aliases}
        preparation = json.loads(payloads["preparation"])
        intervals = (
            preparation.get("effective_intervals")
            if isinstance(preparation, Mapping)
            else None
        )
        if not isinstance(intervals, Mapping) or sha256_text(
            canonical_json(intervals)
        ) != preparation.get("effective_intervals_sha256"):
            raise ValueError(
                f"{ref} preparation carries no authenticated effective intervals."
            )
        tables = validate_interval_document(intervals)
        if native_id not in tables:
            raise ValueError(f"The effective intervals hold no table {native_id!r}.")
        native = tables[native_id]
        schema = EntitySchema(group_entities=(weight_entity,))
        person_entity = schema.person_entity
        if native.entity != person_entity:
            raise ValueError(f"{ref} reports a {person_entity} table.")
        if any(item.rows != ROWS_ALL for item in context.node.inputs):
            raise ValueError(f"{ref} requires whole-row slices.")
        declared = {
            (item.entity, column)
            for item in context.node.inputs
            for column in item.columns
        }
        expected = {
            (person_entity, native.income_column),
            (person_entity, native.universe_column),
            (weight_entity, SOURCE_HOUSEHOLD_ID_COLUMN),
        }
        if declared != expected:
            raise ValueError(f"{ref} inputs must slice exactly {sorted(expected)}.")
        problem = decode_problem(payloads["problem"])
        solution = decode_solution(
            payloads["solution"],
            problem_sha256=problem.sha256,
            entity_ids=problem.entity_ids,
        )
        if problem.problem.weight_entity != weight_entity:
            raise ValueError(
                f"{ref} broadcasts {weight_entity!r} weights, but the problem "
                f"calibrates {problem.problem.weight_entity!r}."
            )
        person = context.tables[person_entity]
        units = context.tables[weight_entity]
        ids = units[schema.entity_id_column(weight_entity)].tolist()
        if tuple(ids) != tuple(problem.entity_ids):
            raise ValueError(
                f"{ref} frame {weight_entity} ids differ from the problem."
            )
        calibrated = np.asarray(solution.weights, dtype=np.float64)
        frame_weights = context.weights.get(weight_entity)
        if frame_weights is None or not np.array_equal(
            np.asarray(frame_weights.values, dtype=np.float64), calibrated
        ):
            raise ValueError(
                f"{ref} frame {weight_entity} weights differ from the calibration "
                "solution."
            )
        design = np.asarray(problem.problem.initial_weights.values, dtype=np.float64)
        positions = {value: index for index, value in enumerate(ids)}
        try:
            rows = [
                positions[value]
                for value in person[schema.membership_column(weight_entity)].tolist()
            ]
        except KeyError as error:
            raise ValueError(
                f"{ref} has a {person_entity} outside the problem's {weight_entity} axis."
            ) from error
        sources = units[SOURCE_HOUSEHOLD_ID_COLUMN].tolist()
        artifact = load_ledger_consumer_artifact(
            context.sources[source], feed_pin=feed_pin, lexical_floats=True
        )
        report = native_band_report(
            intervals,
            native_table=native_id,
            coarse_table=coarse_id,
            references=references,
            diagnostic_references=diagnostics,
            facts=artifact.facts,
            person_ids=person[schema.person_id_column].tolist(),
            amounts=person[native.income_column].tolist(),
            statuses=person[native.universe_column].tolist(),
            source_households=[sources[row] for row in rows],
            weights={
                "pre_fit": (
                    design[rows],
                    problem.problem.initial_weights.kind.value,
                ),
                "post_fit": (calibrated[rows], frame_weights.kind.value),
            },
            measurement_id=measurement_id,
        )
        params = {name: context.params[name] for name in self._names}
        document = {
            **_header(
                "transport_native_band_report",
                kernel=ref,
                implementation_hash=self.implementation_hash(),
                node=context.node.id,
                artifacts=payloads,
                frame=_frame_digests(context),
                params=params,
            ),
            "measurement_id": measurement_id,
            "diagnostic_references_sha256": diagnostics_sha256,
            "feed": {
                "facts_sha256": artifact.facts_sha256,
                "manifest_sha256": artifact.manifest_sha256,
            },
            **report,
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "rows": len(report["rows"]),
                "coarse_bands": len(report["coarse"]),
                "artifact_sha256": sha256_text(payload),
            },
        )


NATIVE_BAND_REPORT = NativeBandReportKernel()


def register_income_evidence_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register the income evidence kernels explicitly and idempotently."""

    registry.register(INCOME_IDENTITY)
    registry.register(OMISSION_REGISTER)
    registry.register(NATIVE_BAND_REPORT)
    return registry
