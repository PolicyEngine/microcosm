"""Spec-declared unit reporting classifiers and baseline cell transport.

The baseline classifier emits a typed unit-id mapping. Other populations read
that mapping through an explicit artifact edge, retaining the baseline cells
while reporting their own weighted shares. No income is recomputed on that
path. Receipt classifications reduce assigned member flags using the declared
priority and record mixed-unit counts in each scenario's node receipt.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from microcosm.frame import reporting
from microcosm.graph import (
    ROWS_ALL,
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
from microcosm.graph.population import dtype_for_token

from . import graph_inputs
from .artifact_types import INCOME_CELLS_TYPE
from .graph_inputs import (
    canonical_document_param,
    context_frame,
    entities_param,
    require_outputs,
    require_params,
    sha256_param,
    string_param,
)

__all__ = [
    "REPORTING_INCOME_DECILE",
    "REPORTING_CARRY_INCOME_DECILE",
    "REPORTING_RECEIPT_GROUP",
    "IncomeDecileKernel",
    "CarryIncomeDecileKernel",
    "ReceiptGroupKernel",
    "decode_income_cells",
    "register_reporting_kernels",
]


def decode_income_cells(payload: bytes) -> dict:
    """Validate a complete baseline mapping before admitting it as an input."""
    try:
        document = json.loads(payload)
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        raise ValueError("Invalid baseline income cells JSON.") from error
    required = {
        "schema_version",
        "kind",
        "baseline_scenario",
        "entity",
        "unit_id_column",
        "n_bins",
        "classifier_sha256",
        "unit_ids",
        "deciles",
    }
    if (
        not isinstance(document, dict)
        or set(document) != required
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != "reporting_income_cells"
    ):
        raise ValueError("Invalid baseline income cells schema.")
    for name in ("baseline_scenario", "entity", "unit_id_column"):
        if not isinstance(document[name], str) or not document[name]:
            raise ValueError(f"Baseline income cells need {name}.")
    digest = document["classifier_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or not set(digest) <= set("0123456789abcdef")
    ):
        raise ValueError("Invalid baseline classifier digest.")
    if type(document["n_bins"]) is not int or document["n_bins"] < 1:
        raise ValueError("Invalid baseline n_bins.")
    ids, labels = document["unit_ids"], document["deciles"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(value) is not int for value in ids)
        or ids != sorted(set(ids))
        or not isinstance(labels, list)
        or len(labels) != len(ids)
        or any(
            type(value) is not int or not 1 <= value <= document["n_bins"]
            for value in labels
        )
    ):
        raise ValueError(
            "Baseline cells need unique sorted unit ids and valid deciles."
        )
    return document


def _classifier(context, ref, *, income):
    sha256_param(context, ref, "classifier_sha256")
    spec = canonical_document_param(context, ref, "classifier")
    validator = (
        reporting.validate_income_classifier
        if income
        else reporting.validate_receipt_classifier
    )
    return validator(spec)


def _contract(context, ref, entity, column, dtype, *, carry=False):
    if context.node.sources or context.sources:
        raise ValueError(f"{ref} declares no sources.")
    if any(
        item.rows != ROWS_ALL for item in (*context.node.inputs, *context.node.outputs)
    ):
        raise ValueError(f"{ref} requires all-row slices.")
    outputs = context.node.outputs
    if len(outputs) != 1 or (
        outputs[0].entity,
        outputs[0].column,
        outputs[0].dtype,
    ) != (entity, column, dtype):
        raise ValueError(f"{ref} must own exactly {entity}.{column} as {dtype}.")
    if carry:
        if (
            len(context.node.artifact_inputs) != 1
            or context.node.artifact_inputs[0].name != "baseline_cells"
            or context.node.artifact_inputs[0].type != INCOME_CELLS_TYPE
            or set(context.artifacts) != {"baseline_cells"}
            or context.artifacts["baseline_cells"].type != INCOME_CELLS_TYPE
        ):
            raise ValueError(f"{ref} needs exactly the typed baseline_cells artifact.")
    elif context.node.artifact_inputs or context.artifacts:
        raise ValueError(f"{ref} declares no artifact inputs.")
    require_outputs(context, ref)


def _weighted_frame(context, ref, spec):
    entities = entities_param(context, ref, "entities")
    if entities[0] != spec["person_entity"] or spec["entity"] not in entities:
        raise ValueError(f"{ref} entity schema disagrees with its classifier.")
    frame = context_frame(
        context, ref, entities=entities, weight_entity=spec["weight_entity"]
    )
    weights = frame.resolve_weights(spec["entity"])
    if weights.kind.value != spec["expected_weight_kind"]:
        raise ValueError(
            f"{ref} requires {spec['expected_weight_kind']!r} unit weights."
        )
    return frame, weights.values


def _column(table, spec, name, values):
    return {
        (spec["entity"], name): pd.Series(
            values, index=pd.Index(table[spec["unit_id_column"]].to_numpy(copy=True))
        )
    }


class _ReportingKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            sys.modules[__name__],
            reporting,
            graph_inputs,
            dependencies=self.capabilities.dependencies,
        )


_INCOME_PARAMS = frozenset({"classifier", "classifier_sha256", "scenario", "entities"})


class IncomeDecileKernel(_ReportingKernel):
    """Compute cells once on the declared baseline, exporting their unit ids."""

    ref = "reporting.income_decile@1"

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, self.ref, required=_INCOME_PARAMS)
        spec = _classifier(context, self.ref, income=True)
        scenario = string_param(context, self.ref, "scenario")
        if scenario != spec["boundary_scenario"]:
            raise ValueError(
                f"{self.ref} only computes the declared baseline scenario."
            )
        _contract(context, self.ref, spec["entity"], "income_decile", "int64")
        frame, weights = _weighted_frame(context, self.ref, spec)
        table = frame.table(spec["entity"])
        if spec["tie_key"] not in table:
            raise ValueError(f"Missing stable tie key {spec['tie_key']!r}.")
        income = reporting.annual_unit_income(
            {entity: frame.table(entity) for entity in frame.schema.entities}, spec
        )
        labels, diagnostics = reporting.weighted_deciles(
            income,
            weights,
            table[spec["tie_key"]].to_numpy(),
            spec["n_bins"],
            rank_method=spec["rank_method"],
        )
        cells = table.loc[:, [spec["unit_id_column"]]].copy()
        cells["income_decile"] = labels
        cells = cells.sort_values(spec["unit_id_column"])
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": "reporting_income_cells",
                "baseline_scenario": scenario,
                "entity": spec["entity"],
                "unit_id_column": spec["unit_id_column"],
                "n_bins": spec["n_bins"],
                "classifier_sha256": context.params["classifier_sha256"],
                "unit_ids": cells[spec["unit_id_column"]].tolist(),
                "deciles": cells["income_decile"].tolist(),
            }
        )
        decode_income_cells(payload)
        return KernelResult(
            columns=_column(table, spec, "income_decile", labels),
            artifacts={"baseline_cells": payload},
            receipt={"scenario": scenario, **diagnostics},
        )


class CarryIncomeDecileKernel(_ReportingKernel):
    """Join a baseline artifact by unit id; only shares use the current weights."""

    ref = "reporting.carry_income_decile@1"

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, self.ref, required=_INCOME_PARAMS)
        spec = _classifier(context, self.ref, income=True)
        scenario = string_param(context, self.ref, "scenario")
        _contract(
            context, self.ref, spec["entity"], "income_decile", "int64", carry=True
        )
        baseline = decode_income_cells(context.artifacts["baseline_cells"].payload)
        expected = {
            "baseline_scenario": spec["boundary_scenario"],
            "entity": spec["entity"],
            "unit_id_column": spec["unit_id_column"],
            "n_bins": spec["n_bins"],
            "classifier_sha256": context.params["classifier_sha256"],
        }
        if any(baseline[key] != value for key, value in expected.items()):
            raise ValueError("Baseline cells disagree with the declared classifier.")
        frame, weights = _weighted_frame(context, self.ref, spec)
        table = frame.table(spec["entity"])
        labels = reporting.carry_baseline_deciles(
            table[spec["unit_id_column"]].to_numpy(),
            baseline["unit_ids"],
            baseline["deciles"],
            spec["n_bins"],
        )
        return KernelResult(
            columns=_column(table, spec, "income_decile", labels),
            receipt={
                "scenario": scenario,
                "baseline_scenario": baseline["baseline_scenario"],
                "n_units": len(table),
                "weighted_shares": reporting.decile_weighted_shares(
                    labels, weights, spec["n_bins"]
                ),
            },
        )


class ReceiptGroupKernel(_ReportingKernel):
    """Assigned receipt groups, using the declared shared unit predicates."""

    ref = "reporting.receipt_group@1"

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=frozenset({"classifier", "classifier_sha256", "scenario"}),
        )
        spec = _classifier(context, self.ref, income=False)
        _contract(context, self.ref, spec["entity"], "receipt_group", "string")
        groups, diagnostics = reporting.classify_receipt_groups(context.tables, spec)
        groups = groups.astype(dtype_for_token("string"))
        return KernelResult(
            columns={(spec["entity"], "receipt_group"): groups},
            receipt={
                "scenario": string_param(context, self.ref, "scenario"),
                **diagnostics,
            },
        )


REPORTING_INCOME_DECILE = IncomeDecileKernel()
REPORTING_CARRY_INCOME_DECILE = CarryIncomeDecileKernel()
REPORTING_RECEIPT_GROUP = ReceiptGroupKernel()


def register_reporting_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Explicit, idempotent registration; implementation hashes bind no spec."""
    for kernel in (
        REPORTING_INCOME_DECILE,
        REPORTING_CARRY_INCOME_DECILE,
        REPORTING_RECEIPT_GROUP,
    ):
        registry.register(kernel)
    return registry
