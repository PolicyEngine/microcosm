"""The national release role on the graph: kernels, composition, evidence.

These tests stand on the calibration seam suite's synthetic frame, sidecar
and register (``test_support.microcosm_build.uk_calibration_run``): the same
inputs the seam solved, now bound as a spine checkpoint and solved through
``uk.full.national_targets`` → ``uk.full.national_problem`` →
``uk.full.dense`` → ``uk.full.calibrated`` → ``uk.full.gates.calibrated``.
The register materialises from the frame's own columns (no engine), the
seam's ``measure_resolver=None`` route.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.build.uk_runtime import calibration_run, full_targets, graph_national
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
)
from microcosm.build.uk_runtime.graph_build import (
    UKBoundSpineKernel,
    bound_spine_graph,
)
from microcosm.build.uk_runtime.graph_calibration import register_uk_calibration_kernels
from microcosm.build.uk_runtime.graph_national import (
    NATIONAL_GATES_NODE,
    NATIONAL_PROBLEM_NODE,
    NATIONAL_TARGETS_NODE,
    UKNationalBuildConfig,
    register_uk_national_kernels,
    uk_national_graph,
)
from microcosm.build.uk_runtime.national_calibration import (
    CalibrationFrameAdapter,
    national_calibration_mass_reason,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UKNationalSolveDoctrine,
    uk_doctrine_with_overrides,
    uk_national_target_loss_weights,
)
from microcosm.build.uk_runtime.national_frame import write_uk_national_frame
from microcosm.calibrate import calibrate
from microcosm.calibrate.artifacts import decode_problem
from microcosm.frame import WeightKind
from microcosm.graph import ContentStore, KernelRegistry, compile_graph, run_graph
from test_support.microcosm_build import uk_calibration_run as seam
from test_support.microcosm_build.uk_calibration_run import (  # noqa: F401
    _cgt_projection,
    _signing_key,
)


@pytest.fixture(autouse=True)
def _admin_totals(monkeypatch):
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (seam._admin_anchor_values(), []),
    )


def _patch_target_inputs(monkeypatch, registry):
    """Stand in for the pinned Chronicle artifact: the fixture register."""
    artifact = SimpleNamespace(
        facts=None,
        facts_sha256="1" * 64,
        manifest_sha256="2" * 64,
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-national-fixture",
        },
    )
    monkeypatch.setattr(
        full_targets, "load_ledger_consumer_artifact", lambda path, **kwargs: artifact
    )
    monkeypatch.setattr(
        full_targets,
        "require_committed_uk_chronicle_feed_pin",
        lambda facts_sha256, **kwargs: SimpleNamespace(
            to_dict=lambda: {"facts_sha256": "1" * 64, "manifest_sha256": "2" * 64}
        ),
    )
    monkeypatch.setattr(
        full_targets,
        "compile_uk_target_registry",
        lambda facts, target_period: SimpleNamespace(registry=registry, unsupported=()),
    )
    monkeypatch.setattr(
        full_targets, "load_uk_calibration_measure_exclusions", lambda p: ()
    )
    monkeypatch.setattr(
        full_targets,
        "apply_uk_calibration_measure_exclusions",
        lambda reg, exclusions, now=None: (reg, {}),
    )
    monkeypatch.setattr(
        full_targets, "_ledger_provenance", lambda artifact: {"artifact_id": "fixture"}
    )
    return artifact


def _bound_inputs(tmp_path, frame):
    input_h5 = tmp_path / "spine.h5"
    write_uk_national_frame(frame, input_h5)
    sidecar_path, report_path, _sidecar = seam._bound_checkpoint(tmp_path, frame)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "consumer_facts.jsonl").write_text("{}\n")
    return {
        "uk_spine": input_h5,
        "uk_spine_evidence": sidecar_path,
        "uk_spine_gates": report_path,
        "uk_ledger_facts": ledger,
    }


def _config(**overrides):
    doctrine, receipt = uk_doctrine_with_overrides(epochs=5, **overrides)
    return UKNationalBuildConfig(
        calibration_year=2025,
        time_period="2023",
        source_year=2023,
        doctrine=doctrine,
        doctrine_overrides=receipt,
    )


def _registry_for_run():
    kernels = KernelRegistry()
    kernels.register(UKBoundSpineKernel())
    register_uk_calibration_kernels(kernels)
    register_uk_national_kernels(kernels, resolver_factory=None)
    return kernels


def _run(tmp_path, monkeypatch, config=None, *, endpoint=None):
    pytest.importorskip("tables")
    frame = seam._frame()
    registry = seam._registry()
    _patch_target_inputs(monkeypatch, registry)
    sources = _bound_inputs(tmp_path, frame)
    national = uk_national_graph(
        config or _config(),
        spine=bound_spine_graph(frame),
        spine_population="uk.full.spine_checkpoint",
        review_date="2026-09-29",
    )
    graph = national.graph
    if endpoint is not None:
        compiled = compile_graph(graph)
        needed, pending = {endpoint}, [endpoint]
        while pending:
            for parent in compiled.predecessors[pending.pop()]:
                if parent not in needed:
                    needed.add(parent)
                    pending.append(parent)
        from dataclasses import replace

        graph = replace(
            graph, nodes=tuple(node for node in graph.nodes if node.id in needed)
        )
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compile_graph(graph),
        sources=sources,
        store=store,
        kernels=_registry_for_run(),
    )
    return national, manifest, store, frame, registry


def _payload(manifest, store, node, artifact):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[artifact])


def test_national_graph_composes_the_bound_checkpoint_line():
    frame = seam._frame()
    national = uk_national_graph(
        _config(),
        spine=bound_spine_graph(frame),
        spine_population="uk.full.spine_checkpoint",
        review_date="2026-09-29",
    )
    compiled = compile_graph(national.graph)
    assert list(compiled.order) == [
        "uk.full.spine_checkpoint",
        NATIONAL_TARGETS_NODE,
        NATIONAL_PROBLEM_NODE,
        "uk.full.dense",
        "uk.full.calibrated",
        NATIONAL_GATES_NODE,
    ]
    dense = national.graph.node("uk.full.dense")
    # The national doctrine drives the shared solve node; its admission is the
    # bound spine's provenance, not a preflight battery.
    assert dense.params == {
        "epochs": 5,
        "learning_rate": 0.02,
        "seed": 0,
        "admission": "bound_spine",
    }
    assert [item.name for item in dense.artifact_inputs] == [
        "problem",
        "spine_provenance",
    ]
    problem = national.graph.node(NATIONAL_PROBLEM_NODE)
    assert problem.params["target_weight_rule"] == "family_equal"
    assert problem.params["target_loss_cap"] == 10.0
    assert problem.params["max_weight_ratio"] == 10.0
    assert {source.name for source in national.graph.sources} == {
        "uk_spine",
        "uk_spine_evidence",
        "uk_spine_gates",
        "uk_ledger_facts",
    }
    inventory = national.operation_inventory()
    assert inventory["release_role"] == "national"
    assert inventory["configuration"]["doctrine"]["epochs"] == 5
    assert inventory["configuration"]["doctrine_overrides"] == {
        "epochs": {"default": 1500, "effective": 5}
    }
    assert national.population == "uk.full.calibrated"


def test_national_graph_refuses_a_population_without_bound_provenance():
    frame = seam._frame()
    spine = bound_spine_graph(frame)
    from dataclasses import replace

    unbound = replace(
        spine,
        nodes=tuple(replace(node, artifact_outputs=()) for node in spine.nodes),
    )
    with pytest.raises(ValueError, match="bound spine checkpoint population"):
        uk_national_graph(
            _config(),
            spine=unbound,
            spine_population="uk.full.spine_checkpoint",
            review_date="2026-09-29",
        )


def test_national_problem_binds_the_seam_doctrine(tmp_path, monkeypatch):
    national, manifest, store, frame, registry = _run(
        tmp_path, monkeypatch, endpoint=NATIONAL_PROBLEM_NODE
    )
    targets = json.loads(_payload(manifest, store, NATIONAL_TARGETS_NODE, "registry"))
    assert targets["kind"] == "uk_national_target_registry"
    assert targets["calibration_year"] == 2025
    assert targets["register_completeness"]["approved_reference_count"] == 1
    assert targets["chronicle_feed_pin"] == {
        "facts_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
    }
    problem = decode_problem(
        _payload(manifest, store, NATIONAL_PROBLEM_NODE, "problem")
    )
    assert problem.entity_ids == tuple(frame.table("household")["household_id"])
    row_name = registry.specs[0].to_target().row_name
    assert problem.problem.names == (row_name,)
    bindings = problem.bindings
    families = [spec.family for spec in registry.specs]
    assert bindings["release_role"] == "national"
    assert bindings["mass_reason"] == national_calibration_mass_reason(families)
    assert (
        bindings["target_loss_weights"]
        == uk_national_target_loss_weights(families, rule="family_equal").tolist()
    )
    assert bindings["target_weight_rule"] == "family_equal"
    assert bindings["target_loss_cap"] == 10.0
    assert bindings["max_weight_ratio"] == 10.0
    assert bindings["mass_rule"] == "free"
    # The dense node solves under free mass, the default scales and no L0
    # penalty; the doctrine the problem binds is one it honours.
    from microcosm.build.uk_runtime.graph_calibration import (
        refuse_unhonoured_solve_doctrine,
    )

    refuse_unhonoured_solve_doctrine(bindings)
    assert bindings["bound_families"] == ["national/dwp_universal_credit"]
    assert bindings["register_sha256"] == registry.version
    assert bindings["measure_resolution"]["mode"] == "frame_only"
    assert bindings["measure_resolution"]["target_materialization"] == {
        "prepared_count": 1,
        "skipped_count": 0,
        "skipped": [],
    }
    assert problem.target_metadata[0]["materialization"] == "uk_national_measure"
    assert problem.target_metadata[0]["geography_level"] == "country"
    # The matrix is the seam's: the register materialised on the adapter and
    # compiled row for row.
    adapter = CalibrationFrameAdapter(frame)
    from microcosm.build.uk_runtime.ledger_targets import materialize_uk_ledger_targets
    from microcosm.calibrate import build_constraint_matrix

    materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=registry
    )
    expected = build_constraint_matrix(
        adapter.prepared_frame(), registry.to_target_set(), "household"
    )
    assert np.array_equal(problem.problem.matrix.toarray(), expected.matrix.toarray())
    assert np.array_equal(problem.problem.target_vector, expected.target_vector)


def test_national_solve_is_the_seam_solve(tmp_path, monkeypatch):
    national, manifest, store, frame, registry = _run(
        tmp_path, monkeypatch, endpoint="uk.full.calibrated"
    )
    calibrated = manifest.population("uk.full.calibrated")
    weights = calibrated.weights_for("household")
    assert weights.kind is WeightKind.CALIBRATED
    # The seam solved the same register on the same frame with the same
    # doctrine: an in-process calibrate call under the national doctrine
    # reproduces the graph's weights exactly.
    doctrine = UKNationalSolveDoctrine(epochs=5)
    adapter = CalibrationFrameAdapter(frame)
    from microcosm.build.uk_runtime.ledger_targets import materialize_uk_ledger_targets

    materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=registry
    )
    families = [spec.family for spec in registry.specs]
    result = calibrate(
        adapter.prepared_frame(),
        registry.to_target_set(),
        weight_entity="household",
        epochs=doctrine.epochs,
        learning_rate=doctrine.learning_rate,
        mass=doctrine.mass_rule,
        mass_reason=national_calibration_mass_reason(families),
        max_weight_ratio=doctrine.max_weight_ratio,
        seed=doctrine.seed,
        l0_lambda=doctrine.l0_lambda,
        target_loss_cap=doctrine.target_loss_cap,
        target_loss_weights=uk_national_target_loss_weights(
            families, rule=doctrine.target_weight_rule
        ),
    )
    assert np.array_equal(weights.values, result.weights)
    # The calibration mass record is the seam's, appended once.
    assert len(calibrated.mass_log) == len(frame.mass_log) + 1
    record = calibrated.mass_log[-1]
    assert record.reason == national_calibration_mass_reason(families)
    assert record.entity == "household"
    assert calibrated.mass_log[:-1] == frame.mass_log


def test_national_gates_evaluate_the_seam_scope_and_record_the_evidence(
    tmp_path, monkeypatch
):
    national, manifest, store, frame, registry = _run(tmp_path, monkeypatch)
    report = json.loads(_payload(manifest, store, NATIONAL_GATES_NODE, "gate_report"))
    assert report["kind"] == "uk_national_gate_report"
    assert report["posture"] == "calibration_seam"
    assert report["policy_suffix"] == "calibration_seam_scope"
    assert report["scope"] == list(UK_CALIBRATION_GATE_SCOPE)
    assert set(report["scope_exclusions"]) == set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
    outcomes = {row["id"]: row for row in report["report"]["outcomes"]}
    assert set(outcomes) == set(UK_CALIBRATION_GATE_SCOPE)
    assert report["blocking"] == []
    assert report["artifact_permitted"] is True
    assert manifest.nodes[NATIONAL_GATES_NODE].receipt["outcome"] == "pass"
    evidence = json.loads(
        _payload(manifest, store, NATIONAL_GATES_NODE, "calibration_evidence")
    )
    calibration = evidence["calibration"]
    assert calibration["activated_reference_count"] == 1
    assert calibration["resolved_reference_count"] == 1
    assert calibration["matrix_target_count"] == 1
    assert calibration["max_weight_ratio_bound"] == 10.0
    assert calibration["weights"]["household_weight_kind"] == "calibrated"
    assert calibration["weights"]["household_weight_kind_chain"] == [
        {"stage": "staging", "kind": "design"},
        {"stage": "national_calibration", "kind": "calibrated"},
    ]
    assert calibration["weights"]["mass_log_records_before_calibration"] == 0
    assert calibration["weights"]["mass_log_records"] == 1
    assert calibration["weights"]["calibration_mass_change"]["entity"] == "household"
    assert calibration["solve"]["n_targets"] == 1
    assert calibration["solve"]["n_households"] == 4
    assert calibration["parameters"]["doctrine"]["target_weight_rule"] == "family_equal"
    assert calibration["parameters"]["doctrine"]["epochs"] == 5
    assert calibration["target_materialization"]["prepared_count"] == 1
    assert "measure_resolution" not in calibration
    row_name = registry.specs[0].to_target().row_name
    assert evidence["diagnostics"][0]["name"] == row_name
    assert evidence["target_geography_levels"] == {row_name: "country"}
    assert evidence["register"] == {
        "country": "uk",
        "version": registry.version,
        "compiled_count": 1,
        "excluded_count": 0,
        "calibrated_count": 1,
    }


def test_national_problem_refuses_an_empty_register(tmp_path, monkeypatch):
    from microcosm.calibrate import TargetRegistry

    pytest.importorskip("tables")
    frame = seam._frame()
    _patch_target_inputs(monkeypatch, TargetRegistry([], country="uk"))
    sources = _bound_inputs(tmp_path, frame)
    national = uk_national_graph(
        _config(),
        spine=bound_spine_graph(frame),
        spine_population="uk.full.spine_checkpoint",
        review_date="2026-09-29",
    )
    with pytest.raises(Exception, match="selects no target"):
        run_graph(
            compile_graph(national.graph),
            sources=sources,
            store=ContentStore(tmp_path / "store"),
            kernels=_registry_for_run(),
        )


def test_dense_kernel_refuses_bound_spine_admission_without_provenance():
    from microcosm.build.uk_runtime.graph_calibration import UKDenseSolveKernel

    context = SimpleNamespace(params={"admission": "bound_spine"}, artifacts={})
    with pytest.raises(ValueError, match="bound spine's provenance"):
        UKDenseSolveKernel._admit(context)
    with pytest.raises(ValueError, match="Unknown dense calibration admission"):
        UKDenseSolveKernel._admit(
            SimpleNamespace(params={"admission": "x"}, artifacts={})
        )
    # The dense line's admission is unchanged: absent parameter, preflight required.
    with pytest.raises(ValueError, match="source preflight artifact"):
        UKDenseSolveKernel._admit(SimpleNamespace(params={}, artifacts={}))


def test_graph_national_module_has_no_seam_stage_dependency():
    import inspect

    source = inspect.getsource(graph_national)
    assert "UKNationalCalibrationStage" not in source
    assert "run_uk_calibration" not in source


def test_dense_solve_refuses_a_bound_doctrine_it_cannot_honour():
    """``mass_rule``, ``scale_rule`` and ``l0_lambda`` are recorded by the
    national problem as the doctrine the solve ran under; the dense node
    refuses any value other than the one it hard-codes (#1057 round 1)."""
    from microcosm.build.uk_runtime.graph_calibration import (
        UK_DENSE_SOLVE_FIXED_DOCTRINE,
        refuse_unhonoured_solve_doctrine,
    )

    refuse_unhonoured_solve_doctrine({})
    refuse_unhonoured_solve_doctrine(dict(UK_DENSE_SOLVE_FIXED_DOCTRINE))
    refuse_unhonoured_solve_doctrine({"l0_lambda": 0})
    for key, value in (
        ("mass_rule", "pinned"),
        ("scale_rule", "unit"),
        ("l0_lambda", 0.5),
    ):
        with pytest.raises(ValueError, match=key):
            refuse_unhonoured_solve_doctrine(
                {**UK_DENSE_SOLVE_FIXED_DOCTRINE, key: value}
            )


def test_national_gate_node_reads_its_scope_from_the_posture():
    from microcosm.build.uk_runtime.rowwise_posture import UK_ROWWISE_NATIONAL_POSTURE

    national = uk_national_graph(
        _config(),
        spine=bound_spine_graph(seam._frame()),
        spine_population="uk.full.spine_checkpoint",
        review_date="2026-09-29",
    )
    node = national.graph.node(NATIONAL_GATES_NODE)
    assert json.loads(node.params["gate_scope"]) == list(
        UK_ROWWISE_NATIONAL_POSTURE.gate_scope
    )
    assert len(json.loads(node.params["gate_scope"])) == 7


def test_national_run_config_refuses_an_empty_register():
    from microcosm.build.uk_runtime.rowwise_posture import UK_ROWWISE_NATIONAL_POSTURE

    with pytest.raises(ValueError, match="compiled national register"):
        graph_national.national_run_config(
            posture=UK_ROWWISE_NATIONAL_POSTURE,
            config=_config(),
            registry_document={"national_registry": {}},
            driver_parameters={},
        )
