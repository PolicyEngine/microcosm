"""Invented receiving-version contracts; authored without runtime acceptance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_property_tax_leaves as tax
from microcosm.build.us_runtime import graph_survey_completion as completion
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.schema import LinkSpec
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Graph,
    Node,
    Numeric,
    NumericScope,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.decl import GraphError
from microcosm.graph.executor import _apply_result, _project_context
from microcosm.graph.keys import opaque_artifact_key
from microcosm.graph.population import Population

ORDER_TYPE = ArtifactType("test.survey_completion_order", 1)
ORDER = ArtifactInput("parent", "source", "ordering", ORDER_TYPE)


def frame(*, defect=None):
    people = pd.DataFrame(
        {
            "person_id": np.array([2**53 + 7, 2**53 + 9], dtype=np.int64),
            "person_household_id": np.array([31, 32], dtype=np.int64),
            "source_native_person_id": np.array([2**53 + 51, 2**53 + 53]),
            "age": [9, 45],
            "known": pd.array([None, True], dtype="boolean"),
            "geography": pd.array(["001", "002"], dtype="string"),
        }
    )
    for column in (*tax.PROPERTY_COMPONENTS, *tax.TAX_LEAF_COLUMNS):
        people[column] = [np.nan, 4.0]
    groups = pd.DataFrame({"household_id": [31, 32], "size": [1, 1]})
    if defect == "group_index":
        groups.index = pd.Index([8, 5], name="retained_index")
    elif defect == "id_only":
        groups = groups[["household_id"]]
    tables = {"person": people, "household": groups}
    links = ()
    if defect == "links":
        links = (LinkSpec("connections", "person", "household"),)
        tables["connections"] = people[["person_id"]].assign(household_id=[31, 32])
    value = Frame(
        tables,
        EntitySchema(group_entities=("household",), links=links),
        {"household": Weights([1.0, 0.0], WeightKind.DESIGN)},
        metadata={"keep": {"nested": (1, "value")}},
    )
    if defect == "orphan":
        value.person.loc[1, "person_household_id"] = 31
    return value


def declaration(value):
    return completion.completion_receiving_node(
        value, population="source", ordering=(ORDER,)
    )


def context(value, node=None):
    node = declaration(value) if node is None else node
    parent = Population.from_frame(value, "source")
    artifact = ArtifactValue(
        b"invented-parent-ordering",
        ORDER_TYPE,
        opaque_artifact_key("a" * 64, "ordering"),
        "a" * 64,
        NumericScope(Numeric.BITWISE),
    )
    return parent, _project_context(
        node,
        parent,
        key="b" * 64,
        sources={},
        tolerances={},
        numerics={},
        artifacts={"parent": artifact},
    )


def test_receiving_uses_real_filter_preserves_values_zero_weight_and_lineage():
    before = frame()
    parent, ctx = context(before)
    result = completion.CompletionReceivingKernel().run(ctx)
    assert result.frame is None and result.keep.all()
    assert result.keep.index.tolist() == before.person.person_id.tolist()
    assert not result.columns and not result.artifacts
    after = _apply_result(ctx.node, result, parent)
    for entity in before.entities:
        pd.testing.assert_frame_equal(after.frame.table(entity), before.table(entity))
    assert after.frame.schema == before.schema
    assert after.frame.metadata == before.metadata
    assert after.frame.links == before.links
    pd.testing.assert_series_equal(after.frame.strata, before.strata)
    np.testing.assert_array_equal(
        after.frame.weights_for("household").values,
        before.weights_for("household").values,
    )
    assert after.frame.weights_for("household").kind is WeightKind.DESIGN
    assert after.mass_ledger[:-1] == parent.mass_ledger
    assert after.mass_ledger[-1].operation == "filter"
    assert after.mass_ledger[-1].before_total == after.mass_ledger[-1].after_total
    assert after.version == completion.NODE
    assert set(after.owners) == set(parent.owners)
    assert set(after.owners.values()) == {completion.NODE}
    assert result.receipt["source_authority"] is False


def test_receiving_reads_nonstructural_identity_and_keeps_explicit_parent_edge():
    value = frame()
    node = declaration(value)
    assert node.id == completion.NODE and node.kernel == completion.KERNEL
    assert node.base == "source" and node.structural is StructuralDelta.FILTER
    assert node.artifact_inputs == (ORDER,)
    person = next(s for s in node.inputs if s.entity == "person")
    assert "source_native_person_id" in person.columns
    assert "person_id" not in person.columns
    assert "person_household_id" not in person.columns
    assert node.outputs == ()


@pytest.mark.parametrize("defect", ["group_index", "id_only", "orphan", "links"])
def test_receiving_refuses_unsupported_shape_before_declaration(defect):
    with pytest.raises(ValueError):
        declaration(frame(defect=defect))


@pytest.mark.parametrize("defect", ["missing", "wrong_type", "wrong_key", "extra"])
def test_receiving_rechecks_exact_typed_ordering_artifacts(defect):
    _, ctx = context(frame())
    values = dict(ctx.artifacts)
    if defect == "missing":
        values.clear()
    elif defect == "extra":
        values["other"] = values["parent"]
    elif defect == "wrong_type":
        values["parent"] = replace(values["parent"], type=ArtifactType("test.other", 1))
    else:
        values["parent"] = replace(values["parent"], key="0" * 64)
    with pytest.raises(ValueError, match="SURVEY_COMPLETION_"):
        completion.CompletionReceivingKernel().run(replace(ctx, artifacts=values))


def test_receiving_rechecks_membership_and_declaration_after_projection():
    _, ctx = context(frame())
    changed = dict(ctx.tables)
    changed["person"] = changed["person"].copy(deep=True)
    changed["person"].loc[1, "person_household_id"] = 31
    with pytest.raises(ValueError, match="ORPHAN_GROUP"):
        completion.CompletionReceivingKernel().run(replace(ctx, tables=changed))
    node = replace(ctx.node, params={**ctx.node.params, "mode": "drop_unknown"})
    with pytest.raises(ValueError, match="DECLARATION"):
        completion.CompletionReceivingKernel().run(
            replace(ctx, node=node, params=node.params)
        )


def test_receiving_refuses_duplicate_ordering_and_masked_rows():
    value = frame()
    with pytest.raises(ValueError, match="ORDERING_EDGES"):
        completion.completion_receiving_node(
            value, population="source", ordering=(ORDER, ORDER)
        )
    _, ctx = context(value)
    masked = replace(ctx.node.inputs[0], rows="known")
    node = replace(ctx.node, inputs=(masked, *ctx.node.inputs[1:]))
    with pytest.raises(ValueError, match="FULL_SLICES"):
        completion.CompletionReceivingKernel().run(
            replace(ctx, node=node, params=node.params)
        )


def test_separate_completion_and_tax_versions_avoid_rewrite_reader_cycle():
    value = frame()
    source = Node(
        "source",
        "test.completion_source@1",
        sources=("invented",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, name, str(value.table(entity)[name].dtype))
            for entity in value.entities
            for name in value.table(entity)
            if name not in {"person_id", "household_id", "person_household_id"}
        ),
        artifact_outputs=(ArtifactOutput("ordering", ORDER_TYPE),),
    )
    receiving = declaration(value)
    child = Node(
        "test.child",
        "test.child@1",
        population=completion.NODE,
        inputs=tuple(receiving.inputs),
        outputs=tuple(
            Owned("person", c, "float64", rewrite=True)
            for c in (tax.PROPERTY_COMPONENTS[0], tax.PROPERTY_COMPONENTS[2])
        ),
    )
    later = replace(receiving, id="test.tax_receiving", base=completion.NODE)
    tax_node = Node(
        "test.tax",
        "test.tax@1",
        population=later.id,
        inputs=(
            Slice("person", (tax.PROPERTY_COMPONENTS[0], tax.PROPERTY_COMPONENTS[2])),
        ),
        outputs=tuple(
            Owned("person", c, "float64", rewrite=True) for c in tax.TAX_LEAF_COLUMNS
        ),
    )
    graph = Graph(
        "us",
        (SourceRef("invented", "frame-store"),),
        (source, receiving, child, later, tax_node),
    )
    compiled = compile_graph(graph)
    assert (
        compiled.order.index(child.id)
        < compiled.order.index(later.id)
        < compiled.order.index(tax_node.id)
    )
    # In a shared version child reads tax leaves while tax reads child O/D.
    with pytest.raises(GraphError, match="Cycle:"):
        compile_graph(
            replace(
                graph,
                nodes=(
                    source,
                    receiving,
                    child,
                    replace(tax_node, population=completion.NODE),
                ),
            )
        )
    # A rewrite needs a base version carrying the incumbent.
    with pytest.raises(GraphError, match="CREATE version"):
        compile_graph(
            replace(graph, nodes=(source, replace(child, population="source")))
        )
    # Two writers in the valid FILTER version must fail ownership validation,
    # independently of the CREATE prohibition and the reader-cycle check.
    second_child = replace(child, id="test.second_child", kernel="test.second_child@1")
    with pytest.raises(GraphError, match="owned by both"):
        compile_graph(replace(graph, nodes=(source, receiving, child, second_child)))
