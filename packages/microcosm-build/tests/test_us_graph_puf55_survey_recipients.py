"""Actual typed recipient preparation and replay over invented originals."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_graph_current_survey_geography import _detached_context
from test_us_puf55_survey_recipients import recipient_financial_run  # noqa: F401

from microcosm.build.us_runtime import graph_puf55_survey_recipients as graph
from microcosm.graph import StructuralDelta, compile_graph, run_graph
from microcosm.graph.artifact_edges import typed_contracts
from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys


@pytest.fixture(scope="module")
def recipient_graph(recipient_financial_run):  # noqa: F811
    case = recipient_financial_run
    run = case.run
    nodes = graph.puf55_survey_recipient_nodes(case.qualified)
    compiled = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *nodes))
    )
    kernels = (
        graph.Puf55SurveyRecipientProjectionKernel(run),
        graph.Puf55SurveyRecipientMatrixKernel(run),
    )
    for kernel in kernels:
        run.kernels.register(kernel)
    contexts = {}

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is graph._Kernel.run.__code__:
            context = frame.f_locals["context"]
            contexts[context.node.id] = context

    old = sys.getprofile()
    sys.setprofile(trace)
    try:
        cold = run_graph(
            compiled, sources=run.sources, store=run.store, kernels=run.kernels
        )
    finally:
        sys.setprofile(old)
    assert set(contexts) == {graph.PROJECTION_NODE, graph.MATRIX_NODE}
    warm = run_graph(
        compiled,
        sources=run.sources,
        store=run.store,
        kernels=run.kernels,
        resume="require",
    )
    yield SimpleNamespace(
        case=case,
        run=run,
        nodes=nodes,
        compiled=compiled,
        kernels=kernels,
        contexts=contexts,
        cold=cold,
        warm=warm,
    )
    graph.financial.check_atomic_survey_financial_run(run)


def _verified_artifacts(case, manifest):
    """Check actual producer keys and typed closure before reading artifacts."""
    run = case.run
    _, sources = _source_paths_and_keys(case.compiled, run.sources, run.store)
    keys, implementations = _all_node_keys(case.compiled, run.kernels, sources)
    payloads = {}
    for node in case.nodes:
        record = manifest.node(node.id)
        assert record.key == keys[node.id]
        assert record.kernel_impl_hash == implementations[node.id]
        assert record.typed_artifacts == typed_contracts(
            case.compiled, node, keys, run.kernels
        )
        assert set(record.opaque_artifacts) == {a.name for a in node.artifact_outputs}
        for artifact in node.artifact_outputs:
            key = record.opaque_artifacts[artifact.name]
            assert key == graph.opaque_artifact_key(keys[node.id], artifact.name)
            payloads[node.id, artifact.name] = run.store.load_bytes(key)
    matrices = tuple(
        (name, payloads[graph.MATRIX_NODE, graph._NAMES[name]])
        for name, _ in case.case.qualified.matrices
    )
    return payloads[graph.PROJECTION_NODE, "projection"], matrices


def test_actual_twenty_one_node_cold_required_replay_keeps_complete_population(
    recipient_graph,
):
    case = recipient_graph
    # The financial run graph (19 nodes since geography moved after the
    # initial clone) plus exactly the projection and matrix nodes below.
    assert len(case.compiled.order) == 19 + 2
    assert case.compiled.graph.sources == case.run.compiled.graph.sources
    assert set(case.run.sources) == {
        graph.source_graph.SOURCE_NAME,
        graph.financial.reconstruction.blocks.SOURCE,
    }
    assert all(node.sources == tuple(sorted(case.run.sources)) for node in case.nodes)
    assert set(case.compiled.order) == {
        *case.run.compiled.order,
        graph.PROJECTION_NODE,
        graph.MATRIX_NODE,
    }
    order = case.compiled.order
    assert (
        order.index(graph.financial.financial.ATTACH_NODE)
        < order.index(graph.PROJECTION_NODE)
        < order.index(graph.MATRIX_NODE)
    )
    assert all(
        node.structural is StructuralDelta.NONE
        and not node.outputs
        and node.weights is None
        for node in case.nodes
    )
    assert case.cold.key == case.warm.key
    assert all(case.cold.node(node_id).hit for node_id in case.run.compiled.order)
    assert all(node.hit for node in case.warm.nodes.values())
    assert tuple(o.name for o in case.nodes[1].artifact_outputs) == (
        "matrix_nine",
        "matrix_eight",
    )
    for manifest in (case.cold, case.warm):
        projection, matrices = _verified_artifacts(case, manifest)
        got = graph.verify_materialized_puf55_survey_recipients(
            case.run, projection=projection, matrices=matrices
        )
        assert got.receipt == case.case.qualified.receipt
        assert got.matrices == case.case.qualified.matrices
        after = manifest.population(case.run.financial_population.version)
        graph.financial.survey._same_frame(case.run.financial_population.frame, after)
        for entity in after.entities:
            pd.testing.assert_frame_equal(
                after.table(entity),
                case.run.financial_population.frame.table(entity),
                check_exact=True,
            )
        assert (
            manifest.mass_ledger(case.run.financial_population.version)
            == case.run.financial_population.mass_ledger
        )
    graph.financial._pure_run(case.run, graph.financial._run_entry(case.run))


@pytest.mark.parametrize(
    "defect",
    (
        "parameter",
        "source_roster",
        "survey_source_missing",
        "support_source_path",
        "artifact_payload",
        "artifact_key",
        "context_money",
        "route_roster",
    ),
)
def test_kernel_refuses_unbound_declaration_artifact_or_receiving_values(
    recipient_graph, defect
):
    case = recipient_graph
    context = _detached_context(case.contexts[graph.MATRIX_NODE])
    if defect == "parameter":
        context = replace(
            context, params={**context.params, "financial_run_sha256": "0" * 64}
        )
    elif defect == "source_roster":
        context = replace(context, sources={"invented": "/not-read"})
    elif defect == "survey_source_missing":
        sources = dict(context.sources)
        del sources[graph.source_graph.SOURCE_NAME]
        context = replace(context, sources=sources)
    elif defect == "support_source_path":
        sources = dict(context.sources)
        name = graph.financial.reconstruction.blocks.SOURCE
        sources[name] = sources[name].with_name("not-the-issued-support.npz")
        context = replace(context, sources=sources)
    elif defect == "artifact_payload":
        artifacts = dict(context.artifacts)
        artifacts["financial_matrix"] = replace(
            artifacts["financial_matrix"], payload=b"wrong"
        )
        context = replace(context, artifacts=artifacts)
    elif defect == "artifact_key":
        artifacts = dict(context.artifacts)
        artifacts["financial_projection"] = replace(
            artifacts["financial_projection"], producer_key="0" * 64
        )
        context = replace(context, artifacts=artifacts)
    elif defect == "context_money":
        table = context.tables["person"]
        table.loc[table.index[0], "employment_income_before_lsr"] += 1
    else:
        node = replace(context.node, artifact_outputs=())
        context = replace(context, node=node)
    # A real previously qualified context/value pair isolates contract negatives
    # from expensive source rebuilds. Actual cold/replay above executes kernels.
    with pytest.raises((ValueError, AssertionError)):
        case.kernels[1]._context(context, case.case.qualified)


@pytest.mark.parametrize(
    "target",
    ("artifact", "receipt", "source_owner", "baseline_and_result", "mutable_artifact"),
)
def test_last_owner_check_seals_graph_outputs_and_population(recipient_graph, target):
    case, fired = recipient_graph, []
    context = _detached_context(case.contexts[graph.MATRIX_NODE])
    original = None
    table = case.run.financial_population.frame.person
    index = table.index[0]
    if target == "source_owner":
        original = table.loc[index, "employment_income_before_lsr"]

    def trace(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is graph.financial.check_atomic_survey_financial_run.__code__
            and caller is not None
            and caller.f_code is graph._Kernel.run.__code__
            and "result" in caller.f_locals
            and not fired
        ):
            fired.append(True)
            result = caller.f_locals["result"]
            if target == "artifact":
                result.artifacts[next(iter(result.artifacts))] = b"wrong"
            elif target == "receipt":
                result.receipt["release_eligible"] = True
            elif target == "baseline_and_result":
                key = next(iter(result.artifacts))
                result.artifacts[key] = b"wrong"
                caller.f_locals["outputs"][key] = b"wrong"
            elif target == "mutable_artifact":
                key = next(iter(result.artifacts))
                result.artifacts[key] = bytearray(result.artifacts[key])
            else:
                table.loc[index, "employment_income_before_lsr"] += 1

    old = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(
            ValueError, match="GRAPH_FINAL_RESULT|FINANCIAL_RUN_.*CHANGED"
        ):
            case.kernels[1].run(context)
    finally:
        sys.setprofile(old)
        if original is not None:
            table.loc[index, "employment_income_before_lsr"] = original
    assert fired == [True]


@pytest.mark.parametrize("defect", ("projection", "missing_route", "matrix_bytes"))
def test_materialized_replay_refuses_projection_or_route_matrix_substitution(
    recipient_graph, defect
):
    case = recipient_graph
    projection, matrices = _verified_artifacts(case, case.warm)
    if defect == "projection":
        projection = b"{}"
    elif defect == "missing_route":
        matrices = ()
    else:
        matrices = ((matrices[0][0], b"wrong"), *matrices[1:])
    with pytest.raises(ValueError, match="MATERIALIZED_ARTIFACTS"):
        graph.verify_materialized_puf55_survey_recipients(
            case.run, projection=projection, matrices=matrices
        )
