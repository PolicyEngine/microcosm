"""A visible graph operation for reconciling component draws to a signed total.

The caller owns the meanings, source qualification and modeling choices. This
operator preserves its inputs and exposes the adjustments as separate columns;
it does not assign tax treatment or turn a modeled component into an observation.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from microcosm.fit import signed_reconciliation as projection
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

SUMMARY_TYPE = ArtifactType("microcosm.fit.signed_total_reconciliation_summary", 1)
PROTOCOL = "microcosm.fit.signed-total-reconciliation.v1"


def _require(condition, reason):
    if not condition:
        raise ValueError("GRAPH_SIGNED_RECONCILIATION_" + reason)


def _name(value):
    return type(value) is str and bool(value) and value.strip() == value


def _diagnostics(components, prefix):
    return (
        tuple(prefix + "_adjustment_" + c for c in components),
        tuple(prefix + "_active_bound_" + c for c in components),
        prefix + "_residual",
        prefix + "_objective",
    )


def signed_reconciliation_node(
    node_id: str,
    *,
    population: str,
    entity: str,
    anchor: str,
    draws: tuple[str, ...],
    components: tuple[str, ...],
    nonnegative: tuple[bool, ...],
    scales: tuple[float, ...],
    atol: float,
    rtol: float,
    diagnostic_prefix: str,
) -> Node:
    """Declare the complete numeric rule, with no implicit component or scale.

    The input draw names and output component names must differ. Original draw
    and anchor columns remain available alongside component adjustments, bound
    activity, residual and projection objective. All rows must have a finite
    anchor and finite draws. Source unknowns require an explicit upstream
    selection or modeled completion, never a zero supplied by this operator.
    """
    _require(all(_name(v) for v in (entity, anchor, diagnostic_prefix)), "NAMES")
    for values in (draws, components):
        _require(
            type(values) is tuple
            and bool(values)
            and all(_name(c) for c in values)
            and len(set(values)) == len(values),
            "COLUMN_ROSTER",
        )
    _require(len(draws) == len(components), "COMPONENT_COUNT")
    _require(
        type(nonnegative) is tuple
        and len(nonnegative) == len(components)
        and all(type(v) is bool for v in nonnegative),
        "BOUND_MASK",
    )
    _require(
        type(scales) is tuple
        and len(scales) == len(components)
        and all(type(v) is float for v in scales)
        and type(atol) is type(rtol) is float,
        "NUMERIC_PARAMETERS",
    )
    # The pure implementation is the single authority for numeric parameter
    # domains. An empty batch validates them without inventing a country model.
    projection.reconcile_signed_total(
        np.empty((0, len(draws))),
        np.empty(0),
        components=components,
        nonnegative=np.array(nonnegative, dtype=bool),
        scales=np.array(scales),
        atol=atol,
        rtol=rtol,
    )
    adjustments, active, residual, objective = _diagnostics(
        components, diagnostic_prefix
    )
    output_names = (*components, *adjustments, *active, residual, objective)
    _require(
        len(set(output_names)) == len(output_names)
        and len({anchor, *draws}) == 1 + len(draws)
        and not set(output_names).intersection((anchor, *draws, f"{entity}_id")),
        "COLUMN_COLLISION",
    )
    return Node(
        node_id,
        SignedReconciliationKernel.ref,
        population=population,
        inputs=(Slice(entity, (anchor, *draws)),),
        outputs=tuple(
            Owned(entity, c, "bool" if c in active else "float64") for c in output_names
        ),
        params={
            "entity": entity,
            "anchor": anchor,
            "draws": draws,
            "components": components,
            "nonnegative": nonnegative,
            "scales": scales,
            "atol": atol,
            "rtol": rtol,
            "diagnostic_prefix": diagnostic_prefix,
        },
        artifact_outputs=(ArtifactOutput("summary", SUMMARY_TYPE),),
        description="Reconcile joint component draws to a supplied signed total; retain original draws, adjustments and bound diagnostics.",
    )


class SignedReconciliationKernel(KernelBase):
    ref = "fit.signed_reconciliation@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            projection,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        expected = signed_reconciliation_node(
            context.node.id,
            population=context.node.population,
            **dict(context.params),
        )
        _require(context.node.normative() == expected.normative(), "DECLARATION")
        params = expected.params
        entity = params["entity"]
        _require(
            set(context.tables) == {entity}
            and not context.sources
            and not context.artifacts,
            "CONTEXT",
        )
        table = context.tables[entity]
        id_column = f"{entity}_id"
        _require(
            id_column in table
            and table[id_column].dtype == np.dtype("int64")
            and table[id_column].is_unique,
            "ROW_IDENTITY",
        )
        for name in (params["anchor"], *params["draws"]):
            _require(
                name in table
                and table[name].dtype == np.dtype("float64")
                and np.isfinite(table[name].to_numpy()).all(),
                "FINITE_FLOAT64_INPUT:" + name,
            )
        result = projection.reconcile_signed_total(
            table.loc[:, list(params["draws"])].to_numpy(copy=True),
            table[params["anchor"]].to_numpy(copy=True),
            components=params["components"],
            nonnegative=np.array(params["nonnegative"], dtype=bool),
            scales=np.array(params["scales"]),
            atol=params["atol"],
            rtol=params["rtol"],
        )
        index = pd.Index(table[id_column].to_numpy(copy=True), name=id_column)
        adjustments, active, residual, objective = _diagnostics(
            result.components, params["diagnostic_prefix"]
        )
        columns = {}
        for j, component in enumerate(result.components):
            for name, array in (
                (component, result.values[:, j]),
                (adjustments[j], result.adjustments[:, j]),
                (active[j], result.active_bounds[:, j]),
            ):
                columns[(entity, name)] = pd.Series(
                    array.copy(), index=index, name=name
                )
        columns[(entity, residual)] = pd.Series(
            result.residuals.copy(), index=index, name=residual
        )
        columns[(entity, objective)] = pd.Series(
            result.objective.copy(), index=index, name=objective
        )
        summary = {
            "protocol": PROTOCOL,
            "rule": dict(params),
            "rows": len(table),
            "max_absolute_residual": float(np.max(abs(result.residuals), initial=0)),
            "adjusted_rows": int(np.any(result.adjustments != 0, axis=1).sum()),
            "active_bounds_by_component": {
                name: int(result.active_bounds[:, j].sum())
                for j, name in enumerate(result.components)
            },
        }
        return KernelResult(
            columns=columns,
            artifacts={"summary": canonical_json(summary)},
            receipt={"protocol": PROTOCOL, "rows": len(table)},
        )
