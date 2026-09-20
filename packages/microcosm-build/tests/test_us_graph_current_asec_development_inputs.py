"""Invented source/receiving components, never full financial-run issuance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_income_routing import routing_arguments
from test_us_puf55_observed_recipients import _invented_values
from test_us_puf55_survey_recipients import _frames

from microcosm.build.us_runtime import graph_atomic_survey_financial as host
from microcosm.build.us_runtime import graph_current_asec_development_inputs as graph
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys
from microcosm.graph.population import Population


def _pure_fixture():
    fixed, frame = _invented_values(rules=graph.fixed.DEVELOPMENT_RULES)
    original, _, _ = _frames()
    frame.person.drop(columns=list(graph.OUTPUTS), inplace=True)
    qualified = graph.routing.CurrentAsecIncomeRoutingValues(
        fixed.source_basis, pd.DataFrame(), {"invented_component_only": True}
    )
    origins = pd.DataFrame(
        {"native_person_id": original.person.person_id.to_numpy(), "source": "asec"},
        index=pd.Index(original.person.person_id.to_numpy(), name="person_id"),
    )
    return qualified, origins, frame


def test_keyed_signed_values_unknowns_and_unchanged_frame():
    qualified, originals, receiving = _pure_fixture()
    before = graph.source._frame_identity(receiving)
    receiving.person.iloc[:] = receiving.person.iloc[::-1].to_numpy()
    values = graph.columns(qualified, originals, receiving.person)
    assert values["person", "rental_income"].loc[1] == -40
    assert values["person", "farm_operations_income"].loc[1001] == -90
    assert np.isnan(values["person", "taxable_private_pension_income"].loc[2])
    assert values["person", "farm_operations_income"].loc[2] == -90
    for name in graph.OUTPUTS:
        a, b = (
            values["person", name].loc[range(1, 11)],
            values["person", name].loc[range(1001, 1011)],
        )
        np.testing.assert_array_equal(a, b)
    receiving.person.iloc[:] = receiving.person.iloc[::-1].to_numpy()
    assert graph.source._frame_identity(receiving) == before
    assert not graph.codec.decode_json(graph.qualification_payload(qualified))[
        "rule_metadata"
    ]["observed_taxable_amount_claim"]


@pytest.mark.parametrize(
    "change", ("incumbent", "origin", "clone", "basis", "nonfinite")
)
def test_refuses_unqualified_join_or_rewrite(change):
    qualified, originals, receiving = _pure_fixture()
    if change == "incumbent":
        receiving.person[graph.OUTPUTS[0]] = 123.0
    elif change == "origin":
        receiving.person.loc[0, graph.provenance.spine_source_id_column("person")] = 999
    elif change == "clone":
        receiving.person.loc[
            0, graph.provenance.support_clone_index_column("person")
        ] = 1
    elif change == "basis":
        qualified.person.loc[1, "native_person_id"] = 999
    else:
        qualified.person.loc[1, "farm_known_amount"] = np.inf
    with pytest.raises(ValueError):
        graph.columns(qualified, originals, receiving.person)


def _clone_component(frame):
    """Explicit fixture receiving support; this function issues no graph owner."""
    tables = {}
    for entity in frame.entities:
        original = frame.table(entity)
        arms = []
        for arm in (0, 1):
            value = original.copy(deep=True)
            value[graph.provenance.support_source_id_column(entity)] = value[
                entity + "_id"
            ]
            value[graph.provenance.support_clone_index_column(entity)] = np.full(
                len(value), arm, dtype=np.int64
            )
            value[entity + "_id"] += arm * 100000
            if entity == "person":
                for group in frame.schema.group_entities:
                    value["person_" + group + "_id"] += arm * 100000
            arms.append(value)
        tables[entity] = pd.concat(arms, ignore_index=True)
    return Frame(
        tables,
        frame.schema,
        {
            "household": Weights(
                np.tile(frame.weights_for("household").values / 2, 2),
                WeightKind.IMPORTANCE,
            )
        },
        pd.concat((frame.strata, frame.strata), ignore_index=True),
        metadata=frame.metadata,
    )


def test_genuine_routing_two_real_nodes_cold_required_and_reconstruction(
    tmp_path, monkeypatch
):
    (tmp_path / "sources").mkdir()
    arguments = routing_arguments(tmp_path / "sources", monkeypatch)
    prepared = graph.source.prepare_authenticated_survey_population(**arguments)
    entry = prepared._checked()
    receiving = _clone_component(entry[2].frame)
    original_stamp = graph.source._frame_identity(receiving)
    source_node = Node(
        "invented.receiving",
        "invented.development_receiving@1",
        structural=StructuralDelta.CREATE,
        sources=("invented",),
        outputs=tuple(
            Owned(
                e, c, graph.population_ops.token_for_dtype(receiving.table(e)[c].dtype)
            )
            for e in receiving.entities
            for c in receiving.table(e)
            if c != receiving.schema.entity_id_column(e)
            and not (
                e == "person"
                and c
                in {
                    receiving.schema.membership_column(g)
                    for g in receiving.schema.group_entities
                }
            )
        ),
    )

    class Receiving(KernelBase):
        ref = source_node.kernel
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "a" * 64

        def run(self, context):
            return KernelResult(frame=receiving)

    boundary = host._DevelopmentBoundary(
        prepared, entry, Population.from_frame(receiving, source_node.id), {}
    )
    nodes = graph.nodes(
        boundary.qualified,
        receiving,
        population=source_node.id,
        host_edges=(),
        host_pins={},
    )
    boundary.nodes = nodes
    compiled = compile_graph(
        Graph("us", (SourceRef("invented", "raw-bytes-v1"),), (source_node, *nodes))
    )
    registry = KernelRegistry()
    registry.register(Receiving())
    for node in nodes:
        registry.register(graph.DevelopmentKernel(boundary, node))
    path = tmp_path / "component-source.txt"
    path.write_text("Invented receiving support; genuine routing owns source values.")
    store = ContentStore(tmp_path / "store")
    sources = {"invented": path}
    _, source_keys = _source_paths_and_keys(compiled, sources, store)
    keys, _ = _all_node_keys(compiled, registry, source_keys)
    boundary.keys = tuple(sorted(keys.items()))
    observed = {}
    cold = run_graph(
        compiled,
        sources=sources,
        kernels=registry,
        store=store,
        _population_observer=lambda n, p: observed.update({n: p}),
    )
    warm = run_graph(
        compiled, sources=sources, kernels=registry, store=store, resume="require"
    )
    assert warm.key == cold.key and all(n.hit for n in warm.nodes.values())
    expected = Population.from_frame(receiving, source_node.id)
    for node in nodes:
        result = graph.result(
            boundary.qualified, boundary.originals, node, expected.frame.person
        )
        for name, payload in result.artifacts.items():
            assert payload == store.load_bytes(
                warm.node(node.id).opaque_artifacts[name]
            )
        expected = graph.population_ops.patch(expected, node, result)
        host.atomic.same_replayed_population(expected, observed[node.id])
    host.atomic.same_replayed_population(
        boundary.complement(expected), Population.from_frame(receiving, source_node.id)
    )
    assert graph.source._frame_identity(receiving) == original_stamp
    asec = expected.frame.person[graph.provenance.support_channel_column("person")].eq(
        "asec"
    )
    assert expected.frame.person.loc[~asec, list(graph.OUTPUTS)].isna().all().all()
    native = expected.frame.person[graph.provenance.spine_source_id_column("person")]
    assert (asec & native.eq(105)).sum() == 2
    assert (asec & native.eq(107)).sum() == 2
    assert (
        expected.frame.person.loc[asec & native.eq(105), "rental_income"].eq(-400).all()
    )
    assert (
        expected.frame.person.loc[asec & native.eq(107), "farm_operations_income"]
        .eq(-9000)
        .all()
    )
    boundary.requalify()
    expected.frame.person.loc[0, graph.OUTPUTS[0]] = 123.0
    with pytest.raises(ValueError, match="DEVELOPMENT_VALUE_CHANGED"):
        boundary.complement(expected)
    basis = boundary.qualified.person
    basis.loc[basis.native_person_id.eq(105), "net_property_known_amount"] = -399.0
    with pytest.raises(ValueError, match="DEVELOPMENT_SOURCE_OR_RECEIVING_CHANGED"):
        boundary.pure()


@pytest.mark.parametrize(
    "binding",
    (
        "targets",
        "node",
        "type",
        "capability",
        "method",
        "fraction",
        "mapper",
        "origins",
    ),
)
def test_live_contract_refuses_rebinding_before_source_io(monkeypatch, binding):
    if binding == "targets":
        monkeypatch.setattr(
            graph.fixed, "DEVELOPMENT_TARGETS", graph.fixed.DEVELOPMENT_TARGETS[::-1]
        )
    elif binding == "node":
        monkeypatch.setattr(graph, "ATTACH_NODE", "other.attach")
    elif binding == "type":
        monkeypatch.setattr(
            graph, "PROJECTION_TYPE", replace(graph.PROJECTION_TYPE, schema_version=2)
        )
    elif binding == "capability":
        monkeypatch.setattr(
            graph.DevelopmentKernel,
            "capabilities",
            replace(graph.DevelopmentKernel.capabilities, dependencies=()),
        )
    elif binding == "method":
        monkeypatch.setattr(graph.DevelopmentKernel, "run", lambda *_: None)
    elif binding == "mapper":
        monkeypatch.setattr(graph.attachment, "attach_columns", lambda *_: {})
    elif binding == "origins":
        monkeypatch.setattr(graph.original_source, "_origins", lambda *_: None)
    else:
        monkeypatch.setattr(graph.fixed.leaves, "TAXABLE_PENSION_FRACTION", 0.111)
    assert graph._live() != graph._LIVE
    with pytest.raises(ValueError, match="DEVELOPMENT_IMPLEMENTATION_CHANGED"):
        host._DevelopmentBoundary(None, None, None, {})


@pytest.mark.parametrize("development", (False, True))
@pytest.mark.parametrize("completion", (False, True))
def test_puf_receives_actual_terminal_after_completion(development, completion):
    # Selector contract only. No fabricated issuance or execution receipt.
    actual = host._financial_terminal(
        property_income=object(),
        rebase_property_taxes=completion,
        completion_boundary=object() if completion else None,
        development_boundary=object() if development else None,
    )
    assert actual == (
        host._tax_module().GATE_NODE
        if completion
        else graph.ATTACH_NODE
        if development
        else host._property_module().ATTACH_NODE
    )


def test_additions_compose_with_actual_child_roles_tax_declarations_and_compact_roster():
    """Real fragment factories/compiler; invented values, no full51 execution."""
    from types import SimpleNamespace

    from test_us_child_property_income_graph import (
        _options,
        _qualified,
        _receiving_frame,
    )
    from test_us_current_asec_income_routing import _pure_rows
    from test_us_current_survey_household_roles import _qualified as role_values
    from test_us_current_survey_household_roles import _table as role_table

    from microcosm.build.us_runtime import graph_child_property_income as child
    from microcosm.build.us_runtime import graph_current_survey_household_roles as roles
    from microcosm.build.us_runtime import graph_property_tax_leaves as tax
    from microcosm.build.us_runtime import graph_survey_completion as receiving_graph
    from microcosm.build.us_runtime import graph_survey_completion_host as completion
    from microcosm.graph import ArtifactInput, ArtifactOutput, ArtifactType

    children, origins = _qualified()
    frame = _receiving_frame(children, origins)
    for name in (*host.values.OUTPUTS, *tax.PROPERTY_COMPONENTS):
        if name not in frame.person:
            frame.person[name] = np.zeros(len(frame.person), dtype=np.float64)
    ordering_type = ArtifactType("invented.development_parent", 1)
    create = Node(
        host.financial.ATTACH_NODE,
        "invented.property_parent@1",
        structural=StructuralDelta.CREATE,
        sources=(child.SOURCE_NAME,),
        outputs=tuple(
            Owned(e, c, graph.population_ops.token_for_dtype(frame.table(e)[c].dtype))
            for e in frame.entities
            for c in frame.table(e)
            if c != frame.schema.entity_id_column(e)
            and not (
                e == "person"
                and c
                in {
                    frame.schema.membership_column(g)
                    for g in frame.schema.group_entities
                }
            )
        ),
        artifact_outputs=(ArtifactOutput("parent", ordering_type),),
    )
    parent_edge = ArtifactInput("parent", create.id, "parent", ordering_type)
    subset = origins.loc[origins.source.eq("asec")]
    raw, ages = _pure_rows(
        len(subset), children.recipients.loc[subset.index, "source_age"].tolist()
    )
    basis = graph.routing.project_income_routing(raw, ages)
    basis.index = pd.Index(subset.index, name="person_id")
    basis["native_person_id"] = subset.selected_receiving_person_id.to_numpy()
    qualified = graph.routing.CurrentAsecIncomeRoutingValues(
        basis, pd.DataFrame(), {"invented_declaration_only": True}
    )
    additions = graph.nodes(
        qualified,
        frame,
        population=create.id,
        host_edges=(parent_edge,),
        host_pins={},
        after_columns=host.values.OUTPUTS,
    )
    base = compile_graph(
        Graph(
            "us", (SourceRef(child.SOURCE_NAME, "raw-bytes-v1"),), (create, *additions)
        )
    )
    assert graph.ATTACH_NODE in host._compact_retained_roster(base, None, False)
    original_axis = origins[["source"]].copy()
    original_axis["native_person_id"] = origins.selected_receiving_person_id
    augmented = graph.source._copy_source(frame)
    for (_, name), vector in graph.columns(
        qualified, original_axis, frame.person
    ).items():
        augmented.person[name] = vector.to_numpy(copy=True)
    after = ArtifactInput(
        "development_binding", graph.ATTACH_NODE, "binding", graph.BINDING_TYPE
    )
    receiving_node = receiving_graph.completion_receiving_node(
        augmented, population=create.id, ordering=(after,)
    )
    qualified_roles = role_values(
        role_table(
            tuple(
                (
                    r.Index,
                    r.selected_receiving_person_id,
                    r.source,
                    1 if r.source == "asec" else 20,
                    1,
                    0,
                )
                for r in origins.itertuples()
            )
        )
    )
    role_nodes = roles.current_survey_household_roles_nodes(
        qualified_roles,
        augmented,
        receiving_version=receiving_node.id,
        host_pins={},
        after=after,
        host_edges=(),
    )
    with_roles = graph.source._copy_source(augmented)
    for (_, name), vector in roles.roles.household_role_columns_for_population(
        qualified_roles, with_roles
    ).items():
        with_roles.person[name] = vector.array.copy()
    parent = Population.from_frame(with_roles, receiving_node.id)
    pins = {
        after.name: {
            "producer_key": "a" * 64,
            "artifact_key": "b" * 64,
            "payload_sha256": "c" * 64,
        }
    }
    child_nodes = child.child_property_nodes(
        children,
        origins,
        parent,
        options=_options(),
        host_edges=(after,),
        host_pins=pins,
    )
    finished = graph.source._copy_source(with_roles)
    finished.person[child.STATUS] = pd.array(
        ["invented"] * len(finished.person), dtype="string"
    )
    finished.person[child.IMPUTED] = False
    tax_nodes = tax.property_tax_leaf_nodes(
        finished,
        population=receiving_node.id,
        projection=replace(parent_edge, name="projection"),
        reconciliation=replace(parent_edge, name="reconciliation"),
        atol=0.0,
        rtol=0.0,
        completion=ArtifactInput(
            "child_verification", child.VERIFY, "verification", child.VERIFICATION_TYPE
        ),
    )
    boundary = SimpleNamespace(
        household_roles=True,
        receiving_node=receiving_node,
        role_nodes=role_nodes,
        child=SimpleNamespace(nodes=child_nodes),
        base=SimpleNamespace(compiled=base),
    )
    extensions = completion._completion_nodes(boundary, tax_nodes)
    compiled = compile_graph(
        replace(base.graph, nodes=(*base.graph.nodes, *extensions))
    )
    order = compiled.order
    assert (
        order.index(graph.ATTACH_NODE)
        < order.index(receiving_node.id)
        < order.index(roles.BIND_NODE)
        < order.index(child.ATTACH)
        < order.index(tax.GATE_NODE)
    )
    assert set(graph.OUTPUTS).isdisjoint(
        o.column for n in extensions for o in n.outputs
    )
    assert host._compact_retained_roster(compiled, None, False, boundary) == tuple(
        n for n in order if n not in base.order
    )
    assert set(graph.OUTPUTS) <= set(
        c for selection in tax_nodes[0].inputs for c in selection.columns
    )
