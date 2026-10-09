"""The full graph preserves dense and exact-count numerical paths and replay."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import dataset_size
from microcosm.build.uk_runtime.graph_calibration import (
    UKGraphCalibrationConfig,
    register_uk_calibration_kernels,
    restore_uk_graph_result,
    uk_calibration_nodes,
)
from microcosm.build.uk_runtime.graph_terminal import FULL_GATE_REPORT_TYPE
from microcosm.calibrate import Target, TargetSet, build_constraint_matrix, calibrate
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    decode_calibration_result,
    decode_problem,
    encode_problem,
)
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
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
from test_support.microcosm_build.uk_full_calibration_graph import (
    preflight_payload,
)
from test_support.microcosm_build.uk_local_rowwise import _clone_frame


def source_frame():
    frame = _clone_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"]["marker"] = [5, 7, 9]
    tables["person"]["age"] = [30, 40, 50]
    tables["benunit"]["eligible"] = [True, True, False]
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def targets():
    return TargetSet(
        [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
    )


def binding():
    return {
        "mass_reason": "Fixture selected constraints",
        "max_weight_ratio": 10.0,
        "target_loss_weights": [1.0],
        "target_loss_cap": 10.0,
        "selector": "all",
    }


def problem_payload():
    frame = source_frame()
    return encode_problem(
        build_constraint_matrix(frame, targets(), weight_entity="household"),
        entity_ids=frame.table("household")["household_id"].tolist(),
        bindings=binding(),
    )


class Source(KernelBase):
    ref = "uk.test.solver-source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context):
        return KernelResult(
            frame=source_frame(),
            artifacts={
                "problem": problem_payload(),
                "preflight": preflight_payload(context.params["preflight_passed"]),
            },
        )


def compiled(k, checkpoint_identity=None, preflight_passed=True):
    calibration = uk_calibration_nodes(
        base="pool",
        columns={
            ("household", "marker"): "int64",
            ("person", "age"): "int64",
            ("benunit", "eligible"): "bool",
        },
        problem_producer="pool",
        config=UKGraphCalibrationConfig(
            epochs=2, learning_rate=0.02, seed=7, dataset_households=k
        ),
        checkpoint_identity=checkpoint_identity,
    )
    graph = Graph(
        "uk",
        (
            SourceRef("fixture", "raw-bytes-v1"),
            *(
                ()
                if checkpoint_identity is None
                else (
                    SourceRef("uk_size_checkpoint_manifest", "raw-bytes-v1"),
                    SourceRef("uk_size_checkpoint_arrays", "raw-bytes-v1"),
                )
            ),
        ),
        (
            Node(
                "pool",
                Source.ref,
                sources=("fixture",),
                params={"preflight_passed": preflight_passed},
                structural=StructuralDelta.CREATE,
                outputs=(
                    Owned("household", "marker", "int64"),
                    Owned("person", "age", "int64"),
                    Owned("benunit", "eligible", "bool"),
                ),
                artifact_outputs=(
                    ArtifactOutput("problem", PROBLEM_TYPE),
                    ArtifactOutput("preflight", FULL_GATE_REPORT_TYPE),
                ),
            ),
            *(
                replace(
                    node,
                    artifact_inputs=(
                        *node.artifact_inputs,
                        ArtifactInput(
                            "preflight", "pool", "preflight", FULL_GATE_REPORT_TYPE
                        ),
                    ),
                )
                if node.id == calibration.dense_producer
                else node
                for node in calibration.nodes
            ),
        ),
    )
    return compile_graph(graph), calibration


def registry():
    kernels = KernelRegistry()
    kernels.register(Source())
    return register_uk_calibration_kernels(kernels)


@pytest.mark.parametrize("k", [None, 2, 3])
def test_graph_preserves_numerical_path_and_complete_resume(k, tmp_path, monkeypatch):
    frame = source_frame()
    dense = calibrate(
        frame,
        targets(),
        weight_entity="household",
        epochs=2,
        learning_rate=0.02,
        seed=7,
        mass="free",
        **binding_solver(),
    )
    expected = (
        dense
        if k is None
        else dataset_size.refit_uk_dataset_size(
            frame, dense, households=k, epochs=2, learning_rate=0.02, seed=7
        ).result
    )
    graph, endpoints = compiled(k)
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    store = ContentStore(tmp_path / "store")
    first = run_graph(
        graph, sources={"fixture": fixture}, store=store, kernels=registry()
    )
    actual = first.population(endpoints.population)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), expected.frame.table(entity)
        )
    np.testing.assert_array_equal(
        actual.weights_for("household").values, expected.weights
    )
    assert actual.mass_log == expected.frame.mass_log
    assert actual.weights_for("household").kind.value == "calibrated"

    def payload(producer, name):
        return store.load_bytes(first.node(producer).opaque_artifacts[name])

    restored = restore_uk_graph_result(
        frame,
        problem_payload=payload(endpoints.problem_producer, "problem"),
        result_payload=payload(endpoints.result_producer, "result"),
        solution_payload=payload(endpoints.solution_producer, "solution"),
        original_problem_payload=problem_payload(),
    )
    assert restored.frame.mass_log == expected.frame.mass_log
    np.testing.assert_array_equal(restored.initial_weights, expected.initial_weights)
    np.testing.assert_array_equal(restored.loss_trajectory, expected.loss_trajectory)
    assert restored.options == expected.options
    if k == 2:
        assert graph.versions["uk.full.size_search"] == "pool"
        assert graph.versions["uk.full.size_refit"] == "pool"
        assert graph.graph.node("uk.full.selected").structural is StructuralDelta.FILTER
    if k == 3:
        receipt = json.loads(
            store.load_bytes(first.node("uk.full.size_refit").opaque_artifacts["size"])
        )
        assert receipt["method"] == "full_pool"
    # Identity is source-authored; prohibit execution while retaining the
    # original kernel source identity in this simulated fresh registry.
    original_registry = registry()
    for kernel in original_registry.as_mapping().values():
        monkeypatch.setattr(
            kernel, "run", lambda *a, **kw: pytest.fail("replay executed")
        )
    replay = run_graph(
        graph,
        sources={"fixture": fixture},
        store=store,
        kernels=original_registry,
        resume="require",
    )
    np.testing.assert_array_equal(
        replay.population(endpoints.population).weights_for("household").values,
        expected.weights,
    )


def binding_solver():
    value = binding()
    value.pop("selector")
    return value


@pytest.mark.parametrize("k", [None, 2, 3])
def test_orrery_summary_replays_dense_and_size_artifacts_without_solving(
    k, tmp_path, monkeypatch
):
    from microcosm.build.uk_runtime.orrery_contract import UK_SUMMARY_PROVIDERS
    from microcosm.graph import (
        collect_execution_evidence,
        graph_schema,
        orrery_document,
        record_run_binding,
    )

    graph, endpoints = compiled(k)
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        graph, sources={"fixture": fixture}, store=store, kernels=registry()
    )
    expected = restore_uk_graph_result(
        source_frame(),
        problem_payload=store.load_bytes(
            manifest.nodes[endpoints.problem_producer].opaque_artifacts["problem"]
        ),
        result_payload=store.load_bytes(
            manifest.nodes[endpoints.result_producer].opaque_artifacts["result"]
        ),
        solution_payload=store.load_bytes(
            manifest.nodes[endpoints.solution_producer].opaque_artifacts["solution"]
        ),
        original_problem_payload=problem_payload(),
    )
    import microcosm.calibrate.solve as solver

    monkeypatch.setattr(
        solver, "calibrate", lambda *a, **kw: pytest.fail("export reran solver")
    )
    run = record_run_binding(graph, manifest, attempt_id="fixture", phase="numerical")
    evidence = collect_execution_evidence(
        graph_schema(graph),
        runs=[run],
        store=store,
        artifact_summaries=UK_SUMMARY_PROVIDERS,
    )
    summary = next(
        item["data"]
        for item in evidence["summaries"]
        if item["node"] == endpoints.result_producer and item["artifact"] == "result"
    )
    rows = summary["tables"]["targets"]
    assert rows[0]["target"] == expected.diagnostics[0].target
    assert rows[0]["achieved"] == expected.diagnostics[0].final_estimate
    assert (
        rows[0]["residual"]
        == expected.diagnostics[0].final_estimate - expected.diagnostics[0].target
    )
    assert rows[0]["uncertainty"]["status"] == "not_recorded"
    assert summary["overview"]["final_weights"]["total"] == float(
        expected.weights.sum()
    )
    assert summary["overview"]["final_weights"]["records"] == (3 if k is None else k)
    if k is not None:
        size = next(
            item["data"]["overview"]
            for item in evidence["summaries"]
            if item["artifact"] == "size"
        )
        assert size["method"] == (
            "full_pool" if k == 3 else "contribution_informed_l0_exact_count_refit"
        )
    preflight = next(
        item["data"]
        for item in evidence["summaries"]
        if item["node"] == "pool" and item["artifact"] == "preflight"
    )
    assert preflight["overview"]["kind"] == "uk_full_gate_report"
    assert preflight["overview"]["phase"] == "preflight"
    assert preflight["overview"]["enforcement"]["artifact_permitted"] is True
    assert preflight["overview"]["status_counts"]["passed"] == len(
        preflight["tables"]["gates"]
    )
    assert {row["status"] for row in preflight["tables"]["gates"]} == {"passed"}
    assert "details" not in preflight["tables"]["gates"][0]
    serialized = json.dumps(evidence)
    assert '"household_ids":' not in serialized
    assert '"pool_row_indices":' not in serialized
    assert '"inclusion_probabilities":' not in serialized
    document = orrery_document(graph, execution=evidence)
    result_node = next(
        node
        for node in document["nodes"]
        if json.loads(node["id"]) == ["operation", endpoints.result_producer]
    )
    assert result_node["data"]["artifact_summaries"]
    pool = next(
        node
        for node in document["nodes"]
        if json.loads(node["id"]) == ["operation", "pool"]
    )
    assert pool["data"]["artifact_summaries"]["preflight"]["tables"]["gates"]["rows"]
    assert [badge["label"] for badge in pool["statuses"]] == [
        "Execution: completed",
        "Computed: numerical",
    ]
    single_file = [
        artifact
        for artifact in document["artifacts"]
        if artifact["label"].endswith("(bytes, 1 file)")
    ]
    assert single_file and all(len(a["sha256"]) == 64 for a in single_file)


def test_reused_draw_skips_rng_and_rejects_changed_binding(monkeypatch):
    frame = source_frame()
    dense = calibrate(frame, targets(), epochs=2)
    options = dict(households=2, epochs=2, learning_rate=0.02, seed=7)
    selection = dataset_size.select_uk_dataset_size(frame, dense, **options)
    draw = dataset_size.draw_uk_dataset_size(
        frame, dense, selection=selection, households=2, seed=7
    )
    expected = dataset_size.refit_uk_dataset_size(
        frame, dense, selection=selection, draw=draw, **options
    )
    monkeypatch.setattr(
        dataset_size, "select_exact_k", lambda *a, **kw: pytest.fail("draw repeated")
    )
    again = dataset_size.refit_uk_dataset_size(
        frame, dense, selection=selection, draw=draw, **options
    )
    np.testing.assert_array_equal(expected.result.weights, again.result.weights)
    from dataclasses import replace

    with pytest.raises(ValueError, match="differs from its selection"):
        dataset_size.refit_uk_dataset_size(
            frame,
            dense,
            selection=selection,
            draw=replace(draw, probabilities_sha256="0" * 64),
            **options,
        )


def test_completed_result_can_rebuild_without_optimizer():
    # A separately serialized result is also independently inspectable; the
    # graph cache is not the only way to obtain diagnostics on resume.
    from microcosm.calibrate.artifacts import encode_calibration_result

    frame = source_frame()
    problem = decode_problem(problem_payload())
    result = calibrate(frame, problem.to_target_set(), epochs=2)
    replay = decode_calibration_result(
        encode_calibration_result(
            result, entity_ids=problem.entity_ids, problem_sha256=problem.sha256
        ),
        frame=frame,
        problem=problem,
    )
    assert replay.problem.names == result.problem.names
    np.testing.assert_array_equal(replay.initial_weights, result.initial_weights)


def test_external_search_checkpoint_is_imported_without_repeating_solves(
    tmp_path, monkeypatch
):
    from microcosm.build.uk_runtime import graph_calibration, size_checkpoint
    from microcosm.graph.errors import NodeRejectedError

    frame = source_frame()
    dense = calibrate(
        frame,
        targets(),
        weight_entity="household",
        epochs=2,
        learning_rate=0.02,
        seed=7,
        mass="free",
        **binding_solver(),
    )
    selection = dataset_size.select_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7
    )
    expected = dataset_size.refit_uk_dataset_size(
        frame,
        dense,
        selection=selection,
        households=2,
        epochs=2,
        learning_rate=0.02,
        seed=7,
    )
    identity = {"pool": "fixture", "K": 1, "selector": "all", "k": 2}
    directory = tmp_path / "legacy"
    size_checkpoint.write_uk_size_checkpoint(
        directory, frame=frame, dense=dense, selection=selection, identity=identity
    )
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    sources = {
        "fixture": fixture,
        "uk_size_checkpoint_manifest": directory
        / size_checkpoint.SIZE_CHECKPOINT_MANIFEST_FILENAME,
        "uk_size_checkpoint_arrays": directory
        / size_checkpoint.SIZE_CHECKPOINT_ARRAYS_FILENAME,
    }
    monkeypatch.setattr(
        graph_calibration,
        "calibrate",
        lambda *a, **kw: pytest.fail("dense solve repeated"),
    )
    monkeypatch.setattr(
        dataset_size,
        "select_uk_dataset_size",
        lambda *a, **kw: pytest.fail("search repeated"),
    )
    graph, endpoints = compiled(2, checkpoint_identity=identity)
    result = run_graph(
        graph,
        sources=sources,
        store=ContentStore(tmp_path / "store"),
        kernels=registry(),
    )
    np.testing.assert_array_equal(
        result.population(endpoints.population).weights_for("household").values,
        expected.result.weights,
    )
    assert "uk.full.size_checkpoint_import" in graph.predecessors["uk.full.dense"]
    assert "uk.full.size_checkpoint_import" in graph.predecessors["uk.full.size_search"]
    for changed in ({**identity, "K": 2}, {"K": 1}):
        drift, _ = compiled(2, checkpoint_identity=changed)
        with pytest.raises(NodeRejectedError, match="identity"):
            run_graph(
                drift,
                sources=sources,
                store=ContentStore(tmp_path / "drift"),
                kernels=registry(),
            )


def test_changing_k_reuses_dense_but_source_bytes_invalidate_it(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"first source")
    store = ContentStore(tmp_path / "store")
    first_graph, _ = compiled(2)
    run_graph(
        first_graph, sources={"fixture": fixture}, store=store, kernels=registry()
    )
    second_graph, _ = compiled(3)
    kernels = registry()
    for ref in (Source.ref, "uk.full.dense@1"):
        monkeypatch.setattr(
            kernels.get(ref),
            "run",
            lambda *a, **kw: pytest.fail("k reran upstream work"),
        )
    resized = run_graph(
        second_graph, sources={"fixture": fixture}, store=store, kernels=kernels
    )
    assert resized.node("pool").hit
    assert resized.node("uk.full.dense").hit
    assert not resized.node("uk.full.size_search").hit
    fixture.write_bytes(b"changed source")
    rebuilt = run_graph(
        second_graph, sources={"fixture": fixture}, store=store, kernels=registry()
    )
    assert not rebuilt.node("pool").hit
    assert not rebuilt.node("uk.full.dense").hit


@pytest.mark.parametrize("imported", [False, True])
def test_blocking_preflight_refuses_before_dense_or_checkpoint_work(imported):
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.graph_calibration import UKDenseSolveKernel

    artifacts = {"preflight": SimpleNamespace(payload=preflight_payload(False))}
    if imported:
        # Deliberately unreadable result: the preflight must refuse before
        # decoding an imported solution, just as it must before optimization.
        artifacts["imported_dense"] = SimpleNamespace(payload=b"must not read")
    with pytest.raises(ValueError, match="refused by the source preflight"):
        UKDenseSolveKernel().run(SimpleNamespace(artifacts=artifacts, params={}))


def test_size_kernels_forward_phased_epochs_to_the_registered_observer(
    tmp_path, monkeypatch
):
    """One observer sees the dense, search and refit epochs, tagged by phase.

    Restores the candidate tool's ``size_candidate_stages_the_search_and_refit_phases``
    contract on the graph (receipts R5, row 19): the size search and refit
    kernels forward their epochs through the callback
    ``register_uk_calibration_kernels`` registers, tagged ``size_search`` and
    ``size_refit`` by ``dataset_size``, after the dense solve's untagged
    epochs, and the driver's stderr line names the phase. The observer is
    instance state: the implementation hashes are an unobserved registry's,
    and the observed run replays under one without executing a kernel.
    """
    from microcosm.build.uk_runtime.solve_progress import uk_solve_progress_callback

    events: list[dict] = []
    observer = events.append
    observed = KernelRegistry()
    observed.register(Source())
    register_uk_calibration_kernels(observed, progress_callback=observer)
    unobserved = registry()
    for ref in ("uk.full.dense@1", "uk.full.size_search@1", "uk.full.size_refit@1"):
        assert observed.get(ref).progress_callback is observer
        assert unobserved.get(ref).progress_callback is None
        assert (
            observed.get(ref).implementation_hash()
            == unobserved.get(ref).implementation_hash()
        )
    graph, endpoints = compiled(2)
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    store = ContentStore(tmp_path / "store")
    first = run_graph(
        graph, sources={"fixture": fixture}, store=store, kernels=observed
    )
    assert not first.node("uk.full.size_search").hit
    epochs = [event for event in events if event["kind"] == "calibration_epoch"]
    assert list(dict.fromkeys(event.get("phase") for event in epochs)) == [
        None,
        "size_search",
        "size_refit",
    ]
    for event in epochs:
        assert 1 <= int(event["epoch"]) <= int(event["epochs"]) == 2
        assert "loss" in event
    # Only the size search reports anything but epochs (probes, the stop).
    assert {
        event["kind"] for event in events if event.get("phase") != "size_search"
    } == {"calibration_epoch"}
    lines: list[str] = []
    render = uk_solve_progress_callback(lines.append, every=1)
    for event in events:
        render(event)
    assert any(" dense solve: epoch " in line for line in lines)
    assert any(" search: epoch " in line or " probe " in line for line in lines)
    assert any(" refit: epoch " in line for line in lines)
    # The observer never entered a node key: the observed run replays under
    # an unobserved registry without executing a kernel.
    for kernel in unobserved.as_mapping().values():
        monkeypatch.setattr(
            kernel, "run", lambda *a, **kw: pytest.fail("replay executed")
        )
    replay = run_graph(
        graph,
        sources={"fixture": fixture},
        store=store,
        kernels=unobserved,
        resume="require",
    )
    np.testing.assert_array_equal(
        replay.population(endpoints.population).weights_for("household").values,
        first.population(endpoints.population).weights_for("household").values,
    )


def test_select_warm_starts_the_search_and_refuses_bad_hints(monkeypatch):
    """microcosm#1115: ``initial_lambda`` reaches the shared search as its
    warm-start penalty, is recorded on the selection and refused when invalid."""
    from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig

    frame = source_frame()
    dense = calibrate(frame, targets(), epochs=2)
    seen = {}
    real = dataset_size.calibrate

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(dataset_size, "calibrate", spy)
    options = dict(households=2, epochs=2, learning_rate=0.02, seed=7)
    cold = dataset_size.select_uk_dataset_size(frame, dense, **options)
    assert seen["l0_lambda"] == 0.0
    assert cold.initial_lambda is None
    assert cold.selection.options["budget_search"]["initial_lambda"] is None
    warm = dataset_size.select_uk_dataset_size(
        frame, dense, initial_lambda=1e-3, **options
    )
    assert seen["l0_lambda"] == 1e-3
    assert warm.initial_lambda == 1e-3
    assert warm.selection.options["budget_search"]["initial_lambda"] == 1e-3
    assert warm.selection.options["budget_search"]["probes"][0]["l0_lambda"] == 1e-3
    for bad in (0.0, -1e-3, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="initial_lambda"):
            dataset_size.select_uk_dataset_size(
                frame, dense, initial_lambda=bad, **options
            )
        with pytest.raises(ValueError, match="initial_lambda"):
            UKGraphCalibrationConfig(selection_initial_lambda=bad)
    assert (
        UKGraphCalibrationConfig(selection_initial_lambda=2e-6).selection_initial_lambda
        == 2e-6
    )
    # The refit reuses a warm-searched selection like any other.
    draw = dataset_size.draw_uk_dataset_size(
        frame, dense, selection=warm, households=2, seed=7
    )
    refit = dataset_size.refit_uk_dataset_size(
        frame, dense, selection=warm, draw=draw, initial_lambda=1e-3, **options
    )
    assert refit.receipt["selection_l0_lambda"] == warm.selection.l0_lambda
