"""Actual graph replay and numerical boundaries for anchored component draws."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit.graph_signed_reconciliation import (
    SignedReconciliationKernel,
    signed_reconciliation_node,
)
from microcosm.fit.signed_reconciliation import reconcile_signed_total
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)


def rule(**changes):
    return signed_reconciliation_node(
        "reconcile",
        **{
            "population": "source",
            "entity": "tax_unit",
            "anchor": "reported_property",
            "draws": (
                "draw_interest",
                "draw_retirement_interest",
                "draw_dividend",
                "draw_property",
            ),
            "components": ("interest", "retirement_interest", "dividend", "property"),
            "nonnegative": (True, True, True, False),
            "scales": (1.0, 1.0, 1.0, 1.0),
            "atol": 1e-10,
            "rtol": 1e-12,
            "diagnostic_prefix": "property_reconciliation",
            **changes,
        },
    )


def table():
    return pd.DataFrame(
        {
            "tax_unit_id": np.array([2**53 + 17, 19, 31], dtype=np.int64),
            "reported_property": [-50.0, 0.0, 130.0],
            "draw_interest": [20.0, 10.0, 30.0],
            "draw_retirement_interest": [10.0, 20.0, 40.0],
            "draw_dividend": [10.0, 30.0, 50.0],
            "draw_property": [-40.0, -60.0, 10.0],
        }
    )


def context(values=None, node=None):
    values = table() if values is None else values
    node = rule() if node is None else node
    return KernelContext(
        node,
        {"tax_unit": values},
        {}
        if len(values) == 0
        else {
            "tax_unit": Weights(
                np.array([1.0, 2.0, 0.0])[: len(values)], WeightKind.DESIGN
            )
        },
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(7),
    )


def test_matches_pure_projection_preserves_input_and_exact_ids():
    ctx = context()
    before = ctx.tables["tax_unit"].copy(deep=True)
    rng = json.dumps(ctx.rng.bit_generator.state, sort_keys=True)
    output = SignedReconciliationKernel().run(ctx)
    p = ctx.params
    expected = reconcile_signed_total(
        before[list(p["draws"])].to_numpy(),
        before[p["anchor"]].to_numpy(),
        components=p["components"],
        nonnegative=np.array(p["nonnegative"]),
        scales=np.array(p["scales"]),
        atol=p["atol"],
        rtol=p["rtol"],
    )
    for j, c in enumerate(p["components"]):
        result = output.columns[("tax_unit", c)]
        np.testing.assert_array_equal(result, expected.values[:, j])
        np.testing.assert_array_equal(result.index, before.tax_unit_id)
        assert result.index.dtype == np.dtype("int64")
        np.testing.assert_array_equal(
            output.columns[("tax_unit", "property_reconciliation_adjustment_" + c)],
            expected.adjustments[:, j],
        )
    # A zero net total need not mean no component income.
    assert output.columns[("tax_unit", "interest")].iloc[1] > 0
    assert output.columns[("tax_unit", "property")].iloc[1] < 0
    pd.testing.assert_frame_equal(before, ctx.tables["tax_unit"])
    assert json.dumps(ctx.rng.bit_generator.state, sort_keys=True) == rng
    assert not output.weights and output.frame is None


def test_row_permutation_preserves_id_association_and_diagnostics():
    kernel = SignedReconciliationKernel()
    original = kernel.run(context())
    permuted = kernel.run(context(table().iloc[[2, 0, 1]]))
    for key in original.columns:
        pd.testing.assert_series_equal(
            original.columns[key].sort_index(), permuted.columns[key].sort_index()
        )
    assert original.artifacts == permuted.artifacts


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", ["reported_property", "draw_interest"])
def test_unknown_or_nonfinite_anchor_or_draw_refuses(column, bad):
    values = table()
    values.loc[0, column] = bad
    with pytest.raises(ValueError, match="FINITE_FLOAT64_INPUT"):
        SignedReconciliationKernel().run(context(values))


@pytest.mark.parametrize(
    "change",
    [
        {"scales": (1.0, 0.0, 1.0, 1.0)},
        {"scales": (1.0, float("inf"), 1.0, 1.0)},
        {"nonnegative": (True, True, True, True)},
        {"nonnegative": (1, True, True, False)},
        {"components": ("interest", "interest", "dividend", "property")},
        {"anchor": "draw_interest"},
        {
            "components": (
                "draw_interest",
                "retirement_interest",
                "dividend",
                "property",
            )
        },
        {"atol": -1.0},
        {"draws": ("draw_interest",)},
    ],
)
def test_invalid_numeric_rule_or_column_collision_refuses_at_declaration(change):
    with pytest.raises(ValueError):
        rule(**change)


def test_direct_context_cannot_change_declared_scales_or_output_roster():
    ctx = context()
    altered = replace(ctx, params={**ctx.params, "scales": (10.0, 1.0, 1.0, 1.0)})
    with pytest.raises(ValueError, match="DECLARATION"):
        SignedReconciliationKernel().run(altered)
    altered = replace(ctx, node=replace(ctx.node, outputs=ctx.node.outputs[:-1]))
    with pytest.raises(ValueError, match="DECLARATION"):
        SignedReconciliationKernel().run(altered)


@pytest.mark.parametrize(
    "defect", ("id_float", "id_duplicate", "nullable", "bool_draw")
)
def test_no_silent_identity_or_dtype_coercion(defect):
    values = table()
    if defect == "id_float":
        values["tax_unit_id"] = values.tax_unit_id.astype(float)
    elif defect == "id_duplicate":
        values.loc[1, "tax_unit_id"] = values.loc[0, "tax_unit_id"]
    elif defect == "nullable":
        values["draw_interest"] = pd.array(values.draw_interest, dtype="Float64")
    else:
        values["draw_interest"] = True
    with pytest.raises(ValueError):
        SignedReconciliationKernel().run(context(values))


def test_empty_projection_has_explicit_zero_diagnostics():
    result = SignedReconciliationKernel().run(context(table().iloc[:0]))
    assert all(len(s) == 0 for s in result.columns.values())
    summary = json.loads(result.artifacts["summary"])
    assert summary["rows"] == summary["adjusted_rows"] == 0
    assert summary["max_absolute_residual"] == 0


class SourceKernel(KernelBase):
    ref = "test.signed_reconciliation_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, ctx):
        return KernelResult(frame=load_source("frame-store", ctx.sources["source"]))


def test_actual_cold_and_required_graph_replay_preserves_complete_frame(tmp_path):
    values = table().sort_values("tax_unit_id").reset_index(drop=True)
    ids = values.tax_unit_id.to_numpy(copy=True)
    frame = Frame(
        {
            "person": pd.DataFrame({"person_id": ids, "person_tax_unit_id": ids}),
            "tax_unit": values,
        },
        EntitySchema(group_entities=("tax_unit",)),
        {"tax_unit": Weights(np.array([1.0, 2.0, 0.0]), WeightKind.DESIGN)},
        metadata={"survey": "invented", "preserve": {"version": 2}},
    )
    source = Node(
        "source",
        SourceKernel.ref,
        sources=("source",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned("tax_unit", c, "float64") for c in values if c != "tax_unit_id"
        ),
    )
    node = rule()
    graph = compile_graph(
        Graph(
            country="us",
            sources=(SourceRef("source", "frame-store"),),
            nodes=(source, node),
        )
    )
    kernels = KernelRegistry()
    kernels.register(SourceKernel())
    kernels.register(SignedReconciliationKernel())
    store = ContentStore(tmp_path)
    key = hashlib.sha256(values.to_json().encode()).hexdigest()
    sources = {"source": store.put_frame(key, frame)}
    observed = {}
    first = run_graph(
        graph,
        sources=sources,
        store=store,
        kernels=kernels,
        _population_observer=lambda name, pop: observed.update({name: pop}),
    )
    complete = observed["reconcile"].frame
    pd.testing.assert_frame_equal(complete.person, frame.person)
    pd.testing.assert_frame_equal(complete.table("tax_unit")[list(values)], values)
    np.testing.assert_array_equal(
        complete.weights_for("tax_unit").values, frame.weights_for("tax_unit").values
    )
    assert complete.metadata == frame.metadata
    assert not any(n.hit for n in first.nodes.values())
    second = run_graph(
        graph,
        sources=sources,
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert first.key == second.key
    assert all(n.hit for n in second.nodes.values())
    assert (
        first.node("reconcile").opaque_artifacts
        == second.node("reconcile").opaque_artifacts
    )
    changed = compile_graph(
        Graph(
            country="us",
            sources=(SourceRef("source", "frame-store"),),
            nodes=(source, rule(scales=(2.0, 1.0, 1.0, 1.0))),
        )
    )
    third = run_graph(changed, sources=sources, store=store, kernels=kernels)
    assert third.node("source").hit and not third.node("reconcile").hit
    assert third.node("reconcile").key != first.node("reconcile").key
