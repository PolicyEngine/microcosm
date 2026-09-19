"""Real hours source owner and tiny invented receiving graph, without PUF fits."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_current_survey_health_coverage import _frame
from test_us_current_survey_hours_source import actual, receiving_people  # noqa: F401

from microcosm.build.us_runtime import graph_current_survey_hours as graph
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
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
    artifact_edges,
    compile_graph,
    run_graph,
)
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys


@pytest.fixture(scope="module")
def case(actual, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("hours-fragment")
    qualified = actual.values
    people = receiving_people(qualified)
    people["unrelated_value"] = 17.25
    receiving = _frame(people)
    before = graph.source.source._frame_identity(receiving)
    attachment_type = ArtifactType("invented.hours_parent", 1)

    class Receiving(KernelBase):
        ref = "invented.hours_receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "a" * 64

        def run(self, context):
            return KernelResult(frame=receiving, artifacts={"attachment": b"parent"})

    create = Node(
        "invented.receiving",
        Receiving.ref,
        structural=StructuralDelta.CREATE,
        sources=(graph.SOURCE_NAME,),
        outputs=tuple(
            Owned("person", c, graph.population_ops.token_for_dtype(people[c].dtype))
            for c in people
            if c != "person_id"
        ),
        artifact_outputs=(ArtifactOutput("attachment", attachment_type),),
    )
    after = ArtifactInput(
        "housing_attachment", create.id, "attachment", attachment_type
    )
    nodes = graph.hours_nodes(
        qualified, receiving, receiving_version=create.id, after=after
    )
    compiled = compile_graph(
        Graph("us", (SourceRef(graph.SOURCE_NAME, "raw-bytes-v1"),), (create, *nodes))
    )

    def current():
        assert graph.source.source._frame_identity(receiving) == before

    kernels = KernelRegistry()
    kernels.register(Receiving())
    for kernel in graph.hours_kernels(
        qualified,
        receiving,
        receiving_version=create.id,
        after=after,
        require_current=current,
    ):
        kernels.register(kernel)
    path = root / "invented-source.txt"
    path.write_bytes(b"invented receiving; hours use actual retained source owner")
    store = ContentStore(root / "store")
    args = dict(sources={graph.SOURCE_NAME: path}, store=store, kernels=kernels)
    qualified.validate()
    cold = run_graph(compiled, **args)
    warm = run_graph(compiled, **args, resume="require")
    qualified.validate()
    current()
    return SimpleNamespace(
        qualified=qualified,
        receiving=receiving,
        nodes=nodes,
        compiled=compiled,
        cold=cold,
        warm=warm,
        store=store,
        args=args,
        current=current,
        after=after,
        version=create.id,
    )


def test_real_fragment_cold_required_preserves_clones_and_nonowned_values(case):
    assert case.cold.key == case.warm.key
    assert all(n.hit for n in case.warm.nodes.values())
    output = case.warm.population(case.version)
    expected = graph.source.borrow_cloned_hours_columns(
        case.qualified, case.receiving.person
    )
    for (_entity, name), values in expected.items():
        pd.testing.assert_series_equal(
            output.person.set_index("person_id")[name], values
        )
    for entity in case.receiving.entities:
        pd.testing.assert_frame_equal(
            output.table(entity)[case.receiving.table(entity).columns],
            case.receiving.table(entity),
            check_exact=True,
        )
    assert (
        output.weights_for("household").values.tolist()
        == case.receiving.weights_for("household").values.tolist()
    )


def test_graph_hours_and_nullable_provenance_survive_development_storage(
    case, tmp_path
):
    from microcosm.build.us_runtime import native_survey_handoff as handoff

    frame = case.warm.population(case.version)
    inventory = handoff.native_survey_input_inventory(frame)
    hours = next(row for row in inventory if row["variable"] == graph.hours.TARGET)
    assert hours["status"] == "present" and hours["missing_values"] == 0
    assert not hours["source_signal_verified"] and not hours["applicability_verified"]
    report = {
        "protocol": handoff.PROTOCOL,
        "input_inventory": inventory,
        "invented_storage_test": True,
        "release_eligible": False,
    }
    output = tmp_path / "hours-checkpoint"
    # A genuine hours fragment is insufficient to issue an enrichment owner.
    # Exercise only storage; public handoff still requires the retained full host.
    handoff._write_checkpoint(frame, report, output)
    restored = handoff.load_native_survey_development_checkpoint(output)
    assert not restored.owner_live_verified
    assert restored.report == report
    handoff.same_replayed_frame(frame, restored.frame)
    for name in (graph.hours.TARGET, "hours_provenance", "hours_policy"):
        pd.testing.assert_series_equal(frame.person[name], restored.frame.person[name])


def artifacts_for(case, node):
    inputs = case.warm.node(node.id).typed_artifacts["inputs"]
    return {
        edge.name: artifact_edges.value_from_descriptor(
            case.store.load_bytes(inputs[edge.name]["key"]), inputs[edge.name]
        )
        for edge in node.artifact_inputs
    }


def test_replay_reconstructs_every_artifact_and_column_from_independent_owner(case):
    current = graph.population_ops.Population.from_frame(case.receiving, case.version)
    for node in case.nodes:
        result = graph.hours_result(
            case.qualified, node, artifacts_for(case, node), current.frame.person
        )
        record = case.warm.node(node.id)
        for name, expected in result.artifacts.items():
            descriptor = record.typed_artifacts["outputs"][name]
            assert case.store.load_bytes(descriptor["key"]) == expected
        current = graph.population_ops.patch(current, node, result)
    pd.testing.assert_frame_equal(
        current.frame.person,
        case.warm.population(case.version).person,
        check_exact=True,
    )
    assert current.mass_ledger == case.warm.mass_ledger(case.version)
    params = str([node.params for node in case.nodes])
    assert "2024HU" not in params and "PERIDNUM" not in params
    assert graph.hours.AGE15_POLICY in params
    assert graph.hours.UNDER15_POLICY in params
    assert len(case.nodes[-1].inputs[0].columns) == 4
    assert set(case.nodes[-1].inputs[0].columns).isdisjoint(
        {"unrelated_value", graph.hours.TARGET}
    )


@pytest.mark.parametrize("alias", ["hours_projection", "hours_asec", "hours_acs"])
def test_attachment_refuses_altered_upstream_payload(case, alias):
    node = case.nodes[-1]
    artifacts = artifacts_for(case, node)
    artifacts[alias] = replace(artifacts[alias], payload=b"{}")
    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD"):
        graph.hours_result(case.qualified, node, artifacts, case.receiving.person)


def kernels_for(case, qualified=None, callback=None):
    return graph.hours_kernels(
        case.qualified if qualified is None else qualified,
        case.receiving,
        receiving_version=case.version,
        after=case.after,
        require_current=case.current if callback is None else callback,
    )


def test_copied_or_revoked_owner_cannot_issue_kernels(case):
    with pytest.raises(ValueError, match="PROJECTION_OBJECT_CHANGED"):
        kernels_for(case, replace(case.qualified))
    callback = case.qualified._revalidate
    try:
        object.__setattr__(case.qualified, "_revalidate", None)
        with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
            kernels_for(case)
    finally:
        object.__setattr__(case.qualified, "_revalidate", callback)
    case.qualified.validate()


def test_policy_identity_changes_source_and_every_downstream_key(case):
    _, source_keys = _source_paths_and_keys(
        case.compiled, case.args["sources"], case.store
    )
    before, _ = _all_node_keys(case.compiled, case.args["kernels"], source_keys)
    changed = replace(
        case.nodes[0],
        params={**case.nodes[0].params, "under15_policy": "different-explicit-policy"},
    )
    compiled = compile_graph(
        replace(
            case.compiled.graph,
            nodes=(case.compiled.graph.nodes[0], changed, *case.nodes[1:]),
        )
    )
    after, _ = _all_node_keys(compiled, case.args["kernels"], source_keys)
    assert before[case.version] == after[case.version]
    assert all(before[node.id] != after[node.id] for node in case.nodes)


def test_existing_hours_owner_refuses_before_attachment(case):
    people = case.receiving.person.copy(deep=True)
    people[graph.hours.TARGET] = 40.0
    with pytest.raises(ValueError, match="ATTACH_OWNERSHIP_COLLISION"):
        graph.hours_nodes(
            case.qualified,
            _frame(people),
            receiving_version=case.version,
            after=case.after,
        )


def test_final_host_callback_cannot_change_hours_before_kernel_returns(case):
    table = case.qualified.person_hours
    pid = table.index[0]
    previous = table.at[pid, graph.hours.TARGET]
    calls = 0

    def callback():
        nonlocal calls
        calls += 1
        if calls == 2:
            table.at[pid, graph.hours.TARGET] = previous + 1

    kernel = kernels_for(case, callback=callback)[0]
    try:
        with pytest.raises(ValueError, match="FINAL_HOURS_CHANGED"):
            kernel.run(
                SimpleNamespace(
                    node=case.nodes[0],
                    artifacts=artifacts_for(case, case.nodes[0]),
                    tables={},
                )
            )
    finally:
        table.at[pid, graph.hours.TARGET] = previous
    assert calls == 2
    case.qualified.validate()


def test_final_source_io_cannot_mutate_receiving_owner_before_kernel_returns(case):
    kernel = kernels_for(case)[0]
    table = case.receiving.person
    previous = table.loc[table.index[0], "unrelated_value"]
    previous_profile = sys.getprofile()
    calls = 0

    def profile(frame, event, argument):
        nonlocal calls
        if event == "return" and frame.f_code is type(case.qualified).validate.__code__:
            calls += 1
            if calls == 2:
                table.loc[table.index[0], "unrelated_value"] = previous + 1

    try:
        sys.setprofile(profile)
        with pytest.raises(AssertionError):
            kernel.run(
                SimpleNamespace(
                    node=case.nodes[0],
                    artifacts=artifacts_for(case, case.nodes[0]),
                    tables={},
                )
            )
    finally:
        sys.setprofile(previous_profile)
        table.loc[table.index[0], "unrelated_value"] = previous
    assert calls == 2
    case.qualified.validate()
    case.current()
