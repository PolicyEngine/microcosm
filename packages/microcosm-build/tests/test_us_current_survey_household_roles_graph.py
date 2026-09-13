"""The household-role fragment on a tiny real graph: invented values, actual executor.

The Frame, the source bytes and the qualified values here are invented and
claim no source authority. Everything they are run through — compile_graph,
run_graph, the ContentStore, population patching and the replay comparison —
is the real machinery, so cold and required execution, artifact identity and
complete receiving-state preservation are checked against it rather than
against a stub.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_us_current_survey_household_roles import (
    _frame,
    _qualified,
    _receiving,
    _table,
)

from microcosm.build.us_runtime import current_survey_household_roles as roles
from microcosm.build.us_runtime import graph_current_survey_household_roles as graph
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    GraphError,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejectedError,
    NumericScope,
    Owned,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)
from microcosm.graph import population as populations

RECEIVING = "invented.receiving"
ALLOCATED = "invented.allocated"
ORDERING_TYPE = ArtifactType("invented.household_roles_ordering", 1)


def _ordering_edge(producer=RECEIVING):
    return ArtifactInput("ordering", producer, "ordering", ORDERING_TYPE)


def _invented_source(receiving, people):
    class InventedSource(KernelBase):
        ref = "invented.household_roles_receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "a" * 64  # Test-only fixed producer for the invented Frame.

        def run(self, context):
            return KernelResult(
                frame=receiving, artifacts={"ordering": b"invented ordering input"}
            )

    node = Node(
        RECEIVING,
        InventedSource.ref,
        sources=(graph.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned("person", column, graph._dtype(people[column]))
            for column in people
            if column != "person_id"
        ),
        artifact_outputs=(ArtifactOutput("ordering", ORDERING_TYPE),),
    )
    return node, InventedSource()


def _graph_case(tmp_path, incumbent=None, *, owner_mutation=None):
    qualified = _qualified()
    people = _receiving(qualified.rows, incumbent=incumbent).person
    receiving = _frame(people)
    create, source_kernel = _invented_source(receiving, people)
    after = _ordering_edge()
    state = SimpleNamespace(calls=[])
    expected_seal = graph.household_role_projection_seal(qualified)

    def current():
        assert graph.household_role_projection_seal(qualified) == expected_seal
        state.calls.append(True)
        if state.mutate is not None:
            state.mutate(len(state.calls), state)

    state.mutate = owner_mutation
    state.current = current

    nodes, kernels = graph.current_survey_household_roles_kernels(
        qualified,
        receiving,
        receiving_version=create.id,
        host_pins={},
        after=after,
        host_edges=(),
        require_current=current,
    )
    state.kernels = kernels
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(graph.SOURCE_NAME, "raw-bytes-v1"),),
            (create, *nodes),
        )
    )
    registry = KernelRegistry()
    registry.register(source_kernel)
    for kernel in kernels:
        registry.register(kernel)
    path = tmp_path / "invented-source.txt"
    path.write_bytes(b"invented household-role fixture only")
    store = ContentStore(tmp_path / "store")
    arguments = {
        "sources": {graph.SOURCE_NAME: path},
        "store": store,
        "kernels": registry,
    }
    cold = run_graph(compiled, **arguments)
    observed = {}
    by_node = {}

    def capture(node_id, population):
        observed[population.version] = population
        by_node[node_id] = population

    warm = run_graph(
        compiled, **arguments, resume="require", _population_observer=capture
    )
    return SimpleNamespace(
        qualified=qualified,
        people=people,
        receiving=receiving,
        by_node=by_node,
        nodes=nodes,
        compiled=compiled,
        cold=cold,
        warm=warm,
        store=store,
        calls=state.calls,
        observed=observed,
    )


def test_actual_graph_cold_and_required_bind_both_clones_and_keep_everything_else(
    tmp_path,
):
    import pandas as pd

    case = _graph_case(tmp_path)
    assert len(case.compiled.order) == 3
    assert case.cold.key == case.warm.key
    assert all(receipt.hit for receipt in case.warm.nodes.values())
    # Before and after each of the two cold fragment nodes; replay invokes none.
    assert len(case.calls) == 4
    output = case.warm.population(RECEIVING)
    pd.testing.assert_frame_equal(
        output.person[case.receiving.person.columns],
        case.receiving.person,
        check_exact=True,
    )
    column = output.person[roles.CANONICAL_COLUMN]
    assert str(column.dtype) == "boolean"
    original = output.person[roles.support_source_id_column("person")].to_numpy()
    values = column.to_numpy(na_value=pd.NA)
    assert set(values[original == 10]) == {True}
    assert set(values[original == 20]) == {False}
    assert all(value is pd.NA for value in values[original == 30])
    projection = case.warm.population(graph.SOURCE_NODE)
    assert set(roles.COLUMNS) <= set(projection.person.columns)
    assert roles.CANONICAL_COLUMN not in projection.person


def test_private_codes_stay_in_the_typed_artifact_and_the_public_receipt_is_closed(
    tmp_path,
):
    case = _graph_case(tmp_path)
    binding = case.warm.node(graph.BIND_NODE)
    document = json.loads(case.store.load_bytes(binding.opaque_artifacts["binding"]))
    assert set(document) == roles.PUBLIC_RECEIPT_KEYS
    assert document["acs_allocation_provenance"] == "unresolved"
    assert document["equivalent_allocation_evidence_claim"] is False
    assert document["source_admission_issued"] is False
    assert document["release_eligible"] is False
    projection = case.warm.node(graph.SOURCE_NODE)
    payload = case.store.load_bytes(projection.opaque_artifacts["projection"])
    assert payload == case.qualified.projection
    # The rowwise codes live only in the private typed artifact.
    assert roles.CODE_COLUMN in payload.decode()
    assert roles.CODE_COLUMN not in json.dumps(document)


def test_actual_graph_independent_complete_reconstruction(tmp_path):
    case = _graph_case(tmp_path)
    expected = {RECEIVING: populations.Population.from_frame(case.receiving, RECEIVING)}
    for node_id in case.compiled.order:
        if node_id == RECEIVING:
            continue
        node = case.compiled.graph.node(node_id)
        artifacts = {}
        for edge in node.artifact_inputs:
            producer = case.warm.node(edge.producer)
            key = producer.opaque_artifacts[edge.artifact]
            artifacts[edge.name] = ArtifactValue(
                case.store.load_bytes(key),
                edge.type,
                key,
                producer.key,
                NumericScope(),
            )
        version = case.compiled.versions[node_id]
        expected[version] = graph.expected_survey_household_roles_population(
            node_id,
            expected.get(version),
            qualified=case.qualified,
            node=node,
            artifacts=artifacts,
        )
    for version, population in expected.items():
        graph.replay.same_replayed_population(population, case.observed[version])


def test_materialized_verifier_accepts_the_run_and_refuses_a_wrong_declaration(
    tmp_path,
):
    case = _graph_case(tmp_path)
    node = case.compiled.graph.node(graph.BIND_NODE)
    # The population the bind node actually received, not a rebuilt lookalike.
    incoming = case.by_node[RECEIVING]
    receipt = graph.verify_materialized_survey_household_roles(
        case.observed[RECEIVING], incoming, qualified=case.qualified, node=node
    )
    assert set(receipt) == roles.PUBLIC_RECEIPT_KEYS
    assert receipt["declared_rewrite"] is False
    assert receipt["filled_cells"] == 4 and receipt["unresolved_cells"] == 2
    # A declaration that disagrees with the population it is applied to would
    # let a create silently overwrite an incumbent leaf; it must refuse here.
    wrong = replace(node, params={**dict(node.params), "declared_rewrite": True})
    with pytest.raises(ValueError, match="RECEIVING_DECLARATION"):
        graph.verify_materialized_survey_household_roles(
            case.observed[RECEIVING], incoming, qualified=case.qualified, node=wrong
        )


def test_projection_artifact_bytes_cannot_be_replaced():
    qualified = _qualified()
    receiving = _frame(_receiving(qualified.rows).person)
    bind = graph.current_survey_household_roles_nodes(
        qualified,
        receiving,
        receiving_version="country.receiving",
        host_pins={},
        after=_ordering_edge("country.amounts"),
        host_edges=(),
    )[1]
    artifacts = {
        graph.PROJECTION_ALIAS: ArtifactValue(
            b"{}", graph.PROJECTION_TYPE, "0" * 64, "1" * 64, NumericScope()
        ),
        "ordering": ArtifactValue(
            b"x", ORDERING_TYPE, "2" * 64, "3" * 64, NumericScope()
        ),
    }
    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD"):
        graph._check_artifacts(bind, artifacts, qualified, {})
    with pytest.raises(ValueError, match="ARTIFACT_ROSTER"):
        graph._check_artifacts(bind, {}, qualified, {})


def test_host_edge_pins_are_checked_against_the_incoming_artifact():
    qualified = _qualified()
    receiving = _frame(_receiving(qualified.rows).person)
    edge = ArtifactInput(
        "preparation", "survey_population.create", "preparation", ArtifactType("p", 2)
    )
    pins = {
        "preparation": {
            "producer_key": "a" * 64,
            "artifact_key": "b" * 64,
            "payload_sha256": "c" * 64,
        }
    }
    projection = graph.current_survey_household_roles_nodes(
        qualified,
        receiving,
        receiving_version="country.receiving",
        host_pins=pins,
        after=_ordering_edge("country.amounts"),
        host_edges=(edge,),
    )[0]
    assert edge in projection.artifact_inputs
    artifacts = {
        "ordering": ArtifactValue(
            b"x", ORDERING_TYPE, "2" * 64, "3" * 64, NumericScope()
        ),
        "preparation": ArtifactValue(
            b"payload", edge.type, "b" * 64, "a" * 64, NumericScope()
        ),
    }
    with pytest.raises(ValueError, match="HOST_EDGE_PIN"):
        graph._check_artifacts(projection, artifacts, qualified, pins)


def test_qualified_mutation_changes_the_seal_without_touching_a_fresh_value():
    qualified = _qualified()
    fresh = _qualified()
    seal = graph.household_role_projection_seal(qualified)
    assert graph.household_role_projection_seal(fresh) == seal
    qualified.rows.loc[10, roles.CODE_COLUMN] = 0
    with pytest.raises(ValueError, match="PROJECTION_BINDING"):
        graph.household_role_projection_seal(qualified)
    assert graph.household_role_projection_seal(fresh) == seal


def test_declared_rewrite_preserves_known_incumbents_through_the_real_patch():
    """The rewrite path goes through the executor's own patch, not a private writer."""
    import pandas as pd

    qualified = _qualified()
    people = _receiving(qualified.rows, incumbent={20: False}).person
    receiving = _frame(people)
    node = graph.current_survey_household_roles_nodes(
        qualified,
        receiving,
        receiving_version=RECEIVING,
        host_pins={},
        after=_ordering_edge("country.amounts"),
        host_edges=(),
    )[1]
    assert node.outputs[0].rewrite and node.outputs[0].dtype == "boolean"
    incoming = populations.Population.from_frame(receiving, RECEIVING)
    result = graph._result(qualified, node, receiving.person)
    patched = populations.patch(incoming, node, result)
    column = patched.frame.person[roles.CANONICAL_COLUMN]
    original = patched.frame.person[roles.support_source_id_column("person")].to_numpy()
    values = column.to_numpy(na_value=pd.NA)
    assert set(values[original == 20]) == {False}  # Incumbent preserved.
    assert set(values[original == 10]) == {True}  # Source-known null filled.
    assert all(value is pd.NA for value in values[original == 30])
    assert patched.owners[("person", roles.CANONICAL_COLUMN)] == graph.BIND_NODE
    # The owned leaf is the one column this node is allowed to change; every
    # other column of every entity must survive the rewrite untouched.
    for entity in receiving.entities:
        before = receiving.table(entity)
        unowned = [c for c in before.columns if c != roles.CANONICAL_COLUMN]
        pd.testing.assert_frame_equal(
            patched.frame.table(entity)[unowned], before[unowned], check_exact=True
        )
    assert list(patched.frame.person.columns) == list(receiving.person.columns)


def test_a_rewrite_needs_a_receiving_version_with_a_base():
    """A host that adopts the rewrite must open a version below its CREATE node."""
    qualified = _qualified()
    people = _receiving(qualified.rows, incumbent={20: False}).person
    receiving = _frame(people)
    create, _kernel = _invented_source(receiving, people)
    nodes = graph.current_survey_household_roles_nodes(
        qualified,
        receiving,
        receiving_version=create.id,
        host_pins={},
        after=_ordering_edge(),
        host_edges=(),
    )
    assert nodes[1].outputs[0].rewrite
    with pytest.raises(GraphError, match="rewrite"):
        compile_graph(
            Graph(
                "us",
                (SourceRef(graph.SOURCE_NAME, "raw-bytes-v1"),),
                (create, *nodes),
            )
        )


def test_country_host_edges_name_the_real_typed_ancestry():
    edges = graph.country_host_edges()
    assert {edge.name for edge in edges} == {
        "preparation",
        "allocation",
        "frame_context",
    }
    assert all(type(edge) is ArtifactInput for edge in edges)
    assert {edge.producer for edge in edges} == {
        "survey_population.create",
        "survey_population.allocate",
    }


def test_projection_branch_keeps_the_original_support_and_adds_only_source_columns():
    qualified = _qualified()
    projected = graph._projection_frame(qualified)
    assert type(projected) is Frame
    person = projected.person
    assert set(roles.COLUMNS) <= set(person.columns)
    assert roles.CANONICAL_COLUMN not in person
    assert person.person_id.tolist() == qualified.rows.index.tolist()
    for entity in qualified.source_frame.entities:
        assert set(projected.table(entity).columns) <= set(
            qualified.source_frame.table(entity).columns
        ) | set(roles.COLUMNS)


def _rewrite_graph_case(tmp_path):
    """A receiving version opened below the CREATE, carrying an incumbent leaf.

    A rewrite needs a version with a base to carry the incumbent from, so the
    invented host opens one with an ordinary REWEIGHT that conserves mass and
    changes no cell. That is the shape a country host has after its own
    allocation, clone and geography stages.
    """
    qualified = _qualified()
    people = _receiving(qualified.rows, incumbent={20: False}).person
    receiving = _frame(people)
    create, source_kernel = _invented_source(receiving, people)
    allocated = Weights(
        receiving.weights_for("household").values.copy(), WeightKind.IMPORTANCE
    )

    class InventedAllocation(KernelBase):
        ref = "invented.household_roles_allocate@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
        )

        def implementation_hash(self):
            return "b" * 64  # Test-only fixed producer for the invented version.

        def run(self, context):
            return KernelResult(weights=allocated)

    allocate = Node(
        ALLOCATED,
        InventedAllocation.ref,
        structural=StructuralDelta.REWEIGHT,
        base=create.id,
        weights=WeightTransition("household", "importance", "conserve"),
        mass="conserve",
    )
    after = _ordering_edge()
    state = SimpleNamespace(calls=[])
    expected_seal = graph.household_role_projection_seal(qualified)

    def current():
        assert graph.household_role_projection_seal(qualified) == expected_seal
        state.calls.append(True)

    nodes, kernels = graph.current_survey_household_roles_kernels(
        qualified,
        receiving,
        receiving_version=allocate.id,
        host_pins={},
        after=after,
        host_edges=(),
        require_current=current,
    )
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(graph.SOURCE_NAME, "raw-bytes-v1"),),
            (create, allocate, *nodes),
        )
    )
    registry = KernelRegistry()
    registry.register(source_kernel)
    registry.register(InventedAllocation())
    for kernel in kernels:
        registry.register(kernel)
    path = tmp_path / "invented-source.txt"
    path.write_bytes(b"invented household-role fixture only")
    store = ContentStore(tmp_path / "store")
    arguments = {
        "sources": {graph.SOURCE_NAME: path},
        "store": store,
        "kernels": registry,
    }
    cold = run_graph(compiled, **arguments)
    observed = {}
    by_node = {}

    def capture(node_id, population):
        observed[population.version] = population
        by_node[node_id] = population

    warm = run_graph(
        compiled, **arguments, resume="require", _population_observer=capture
    )
    return SimpleNamespace(
        qualified=qualified,
        receiving=receiving,
        by_node=by_node,
        nodes=nodes,
        compiled=compiled,
        cold=cold,
        warm=warm,
        store=store,
        calls=state.calls,
        observed=observed,
    )


def test_actual_graph_rewrite_preserves_incumbents_cold_and_required(tmp_path):
    import pandas as pd

    case = _rewrite_graph_case(tmp_path)
    assert len(case.compiled.order) == 4
    assert case.nodes[1].population == ALLOCATED
    assert case.nodes[1].outputs[0].rewrite
    assert case.nodes[1].outputs[0].dtype == "boolean"
    assert case.cold.key == case.warm.key
    assert all(receipt.hit for receipt in case.warm.nodes.values())
    assert len(case.calls) == 4
    output = case.warm.population(ALLOCATED)
    column = output.person[roles.CANONICAL_COLUMN]
    assert str(column.dtype) == "boolean"
    original = output.person[roles.support_source_id_column("person")].to_numpy()
    assert column[original == 20].tolist() == [False, False]  # Incumbent preserved.
    assert column[original == 10].tolist() == [True, True]  # Known null filled.
    assert column[original == 30].isna().tolist() == [True, True]
    unowned = [c for c in case.receiving.person.columns if c != roles.CANONICAL_COLUMN]
    pd.testing.assert_frame_equal(
        output.person[unowned], case.receiving.person[unowned], check_exact=True
    )
    assert list(output.person.columns) == list(case.receiving.person.columns)
    binding = case.warm.node(graph.BIND_NODE)
    document = json.loads(case.store.load_bytes(binding.opaque_artifacts["binding"]))
    assert document["declared_rewrite"] is True
    assert document["receiving_version"] == ALLOCATED
    assert document["preserved_cells"] == 2
    assert document["filled_cells"] == 2
    assert document["unresolved_cells"] == 2
    assert set(document) == roles.PUBLIC_RECEIPT_KEYS


def test_rewrite_run_reconstructs_and_refuses_a_conflicting_incumbent(tmp_path):
    case = _rewrite_graph_case(tmp_path)
    node = case.compiled.graph.node(graph.BIND_NODE)
    incoming = case.by_node[ALLOCATED]
    receipt = graph.verify_materialized_survey_household_roles(
        case.observed[ALLOCATED], incoming, qualified=case.qualified, node=node
    )
    assert receipt["preserved_cells"] == 2 and receipt["declared_rewrite"] is True
    conflicting = _qualified(
        _table(
            (
                (10, 2, "asec", 1, 1, 0),
                (20, 3, "acs", 20, 1, 0),
                (30, 5, "asec", 6, 0, 1),
            )
        )
    )
    with pytest.raises(ValueError, match="CANONICAL_CONFLICT"):
        graph.verify_materialized_survey_household_roles(
            case.observed[ALLOCATED], incoming, qualified=conflicting, node=node
        )


@pytest.mark.parametrize("callback_number", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "mutation",
    [
        "callable",
        "code",
        "defaults",
        "constant",
        "graph_callable",
        "graph_constant",
        "callback_identity",
        "callback_defaults",
        "callback_closure",
    ],
)
def test_first_and_final_owner_callbacks_cannot_change_live_implementation(
    tmp_path, monkeypatch, callback_number, mutation
):
    """First/final callbacks on each node cannot change executable behavior."""

    def replacement(*args, **kwargs):
        raise AssertionError("changed binder must not execute")

    restored_cells = []

    def mutate(number, state):
        if number != callback_number:
            return
        if mutation == "callable":
            monkeypatch.setattr(roles, "_bind", replacement)
        elif mutation == "code":
            monkeypatch.setattr(roles._bind, "__code__", replacement.__code__)
        elif mutation == "defaults":
            monkeypatch.setattr(roles._bind, "__defaults__", (None,))
        elif mutation == "constant":
            monkeypatch.setattr(roles, "ACS_OBSERVATION_YEAR", 2099)
        elif mutation == "graph_callable":
            monkeypatch.setattr(graph, "_result", replacement)
        elif mutation == "graph_constant":
            monkeypatch.setattr(graph, "MAX_ARTIFACT_BYTES", 1)
        elif mutation == "callback_identity":
            state.kernels[(number - 1) // 2].require_current = lambda: None
        elif mutation == "callback_defaults":
            monkeypatch.setattr(state.current, "__defaults__", (None,))
        else:
            freevars = dict(
                zip(
                    state.current.__code__.co_freevars,
                    state.current.__closure__,
                    strict=True,
                )
            )
            cell = freevars["expected_seal"]
            restored_cells.append((cell, cell.cell_contents))
            cell.cell_contents = ("changed closure",)

    try:
        with pytest.raises(
            NodeRejectedError, match="IMPLEMENTATION_CHANGED|CALLBACK_CHANGED"
        ):
            _graph_case(tmp_path, owner_mutation=mutate)
    finally:
        for cell, value in restored_cells:
            cell.cell_contents = value


def test_opaque_callable_owner_cannot_claim_an_executable_callback_seal():
    class Owner:
        def __call__(self):
            raise AssertionError("unsealed owner must not execute")

    qualified = _qualified()
    receiving = _frame(_receiving(qualified.rows).person)
    with pytest.raises(ValueError, match="HOST_CALLBACK"):
        graph.current_survey_household_roles_kernels(
            qualified,
            receiving,
            receiving_version=RECEIVING,
            host_pins={},
            after=_ordering_edge(),
            host_edges=(),
            require_current=Owner(),
        )
