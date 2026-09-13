"""Host-owned aggregate snapshots on the actual invented fiscal graph."""

from __future__ import annotations

import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest
from test_us_graph_fiscal_dense_calibration import graph_fixture, setup

from microcosm.build.us_runtime import graph_fiscal_dense_calibration as stage
from microcosm.calibrate import TargetSnapshotCadence, TargetSnapshotObserver
from microcosm.calibrate.target_snapshots import (
    target_identity_digest,
    validate_target_snapshot,
)
from microcosm.graph import Graph, KernelRegistry, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json, normative

solve = importlib.import_module("microcosm.calibrate.solve")


def _observer(sink, *, every=3, run_id="invented-fiscal"):
    return TargetSnapshotObserver(
        sink=sink,
        run_id=run_id,
        candidate_id="invented-three-households",
        cadence=TargetSnapshotCadence(every=every),
        context={"country": "us"},
    )


def _context():
    _, measured, original = setup(incoming=[1.0, 0.0, 3.0])
    node = stage.fiscal_dense_calibration_node(
        measurement_node=measured.node,
        bounds_node="invented.bounds",
        epochs=6,
        learning_rate=0.03,
    )
    return replace(original, node=node, params=node.params)


def _same_result(a, b):
    assert a.weights.kind == b.weights.kind
    assert a.weights.values.tobytes() == b.weights.values.tobytes()
    assert a.artifacts == b.artifacts
    assert a.receipt == b.receipt
    assert a.frame is b.frame is None
    assert not a.columns and not b.columns


def test_default_none_and_enabled_observers_leave_outputs_and_identity_exact():
    context = _context()
    original_node = canonical_json(normative(context.node))
    seen = []
    kernels = (
        stage.FiscalDenseCalibrationKernel(),
        stage.FiscalDenseCalibrationKernel(target_snapshots=None),
        stage.FiscalDenseCalibrationKernel(target_snapshots=_observer(seen.append)),
        stage.FiscalDenseCalibrationKernel(
            target_snapshots=_observer(lambda _: None, every=1, run_id="other-run")
        ),
    )
    assert len({kernel.implementation_hash() for kernel in kernels}) == 1
    results = [kernel.run(context) for kernel in kernels]
    for result in results[1:]:
        _same_result(results[0], result)
    assert seen
    assert canonical_json(normative(context.node)) == original_node
    assert (
        b"invented-fiscal" not in original_node
        and b"target_snapshots" not in original_node
    )
    assert kernels[0]._target_snapshots is kernels[1]._target_snapshots is None
    with pytest.raises(TypeError, match="TargetSnapshotObserver or None"):
        stage.FiscalDenseCalibrationKernel(target_snapshots=lambda _: None)


def test_snapshot_cadence_uses_actual_measured_targets_and_loss_tensor(monkeypatch):
    context = _context()
    measured = stage.measurement.decode_fiscal_measurement(
        context.artifacts["measurement"].payload
    )
    actual_apply = solve._apply_constraint
    actual_estimates = []

    def record(matrix, weights):
        result = actual_apply(matrix, weights)
        actual_estimates.append(result.detach().cpu().numpy().copy())
        return result

    monkeypatch.setattr(solve, "_apply_constraint", record)
    seen = []
    output = stage.FiscalDenseCalibrationKernel(
        target_snapshots=_observer(seen.append)
    ).run(context)
    assert len(actual_estimates) == 6
    assert [item["epoch"] for item in seen] == [0, 2, 5, 6]
    names = tuple(spec.to_target().row_name for spec in measured.registry)
    identity = target_identity_digest(names, measured.target_values)
    for item in seen:
        validate_target_snapshot(item)
        assert item["targets_sha256"] == identity
        assert [row["name"] for row in item["targets"]] == list(names)
        assert [row["target"] for row in item["targets"]] == list(
            measured.target_values
        )
        assert item["best_retained"] == {
            "available": False,
            "epoch": None,
            "loss": None,
        }
        assert item["selection"] == {
            "rule": "closing_state",
            "constraint_mode": "grouped_upper_bounds",
            "grouped_preserve_zeros": True,
        }
    for item in seen[:-1]:
        assert item["iterate"] == "current" and item["precision"] == "float32"
        assert [row["estimate"] for row in item["targets"]] == list(
            actual_estimates[item["epoch"]].astype(np.float64)
        )
    assert seen[-1]["iterate"] == "selected"
    assert seen[-1]["precision"] == "float64"
    np.testing.assert_array_equal(
        [row["estimate"] for row in seen[-1]["targets"]],
        measured.matrix @ output.weights.values,
    )


def test_delivered_payload_mutation_is_detached_from_later_snapshots_and_results():
    context = _context()
    baseline = stage.FiscalDenseCalibrationKernel().run(context)
    delivered = []

    def mutate(payload):
        delivered.append(copy.deepcopy(payload))
        payload["context"]["country"] = "changed"
        payload["selection"]["rule"] = "changed"
        payload["best_retained"]["available"] = True
        payload["targets"][0]["estimate"] = 99999
        payload["targets"][0]["name"] = "changed"

    observer = _observer(mutate)
    result = stage.FiscalDenseCalibrationKernel(target_snapshots=observer).run(context)
    _same_result(baseline, result)
    assert len(delivered) == 4 and observer.context == {"country": "us"}
    for item in delivered:
        validate_target_snapshot(item)
        assert item["context"]["country"] == "us"
        assert item["selection"]["rule"] == "closing_state"
        assert item["targets"][0]["name"] != "changed"


@pytest.mark.parametrize("selected", [False, True], ids=["current", "selected"])
def test_sink_exception_propagates(selected):
    class SinkError(RuntimeError):
        pass

    def fail(payload):
        if (payload["iterate"] == "selected") == selected:
            raise SinkError("invented snapshot sink failure")

    with pytest.raises(SinkError, match="invented snapshot sink failure"):
        stage.FiscalDenseCalibrationKernel(target_snapshots=_observer(fail)).run(
            _context()
        )


@pytest.mark.parametrize("surface", ["measurement", "ids", "weights"])
def test_selected_sink_cannot_bypass_final_measurement_id_or_weight_checks(
    monkeypatch, surface
):
    context = _context()
    materialized = []
    actual_apply = solve._apply_weights

    def capture(*args, **kwargs):
        frame = actual_apply(*args, **kwargs)
        materialized.append(frame)
        return frame

    monkeypatch.setattr(solve, "_apply_weights", capture)

    def mutate(payload):
        if payload["iterate"] != "selected":
            return
        if surface == "measurement":
            context.tables["person"]["amount"] += 1
        elif surface == "ids":
            table = materialized[-1].table("household")
            table.loc[table.index[[0, 2]], "household_id"] = table.household_id.iloc[
                [2, 0]
            ].to_numpy()
        else:
            weights = materialized[-1].weights_for("household")
            changed = weights.values.copy()
            changed[0] += 0.01
            # Normal values are immutable. Inject a corrupted live container
            # to exercise the final acceptance fence after sink completion.
            object.__setattr__(weights, "values", changed)

    reason = (
        "PROJECTION_BINDING"
        if surface == "measurement"
        else "(?i)(ordered IDs|stored|accepted|weights)"
    )
    with pytest.raises(ValueError, match=reason):
        stage.FiscalDenseCalibrationKernel(target_snapshots=_observer(mutate)).run(
            context
        )


def _replace_kernel(arguments, observer):
    registry = KernelRegistry()
    for kernel in arguments["kernels"].as_mapping().values():
        registry.register(
            stage.FiscalDenseCalibrationKernel(target_snapshots=observer)
            if kernel.ref == stage.FiscalDenseCalibrationKernel.ref
            else kernel
        )
    return {**arguments, "kernels": registry}


def test_actual_graph_observer_config_keeps_keys_and_required_replay_is_silent(
    tmp_path, monkeypatch
):
    _, sources, nodes, arguments = graph_fixture(tmp_path / "off")
    compiled = compile_graph(Graph("us", sources, nodes))
    off = run_graph(compiled, **arguments)
    _, on_sources, on_nodes, on_arguments = graph_fixture(tmp_path / "on")
    seen = []
    on_arguments = _replace_kernel(on_arguments, _observer(seen.append, every=40))
    on_compiled = compile_graph(Graph("us", on_sources, on_nodes))
    assert canonical_json(normative(compiled.graph)) == canonical_json(
        normative(on_compiled.graph)
    )
    on = run_graph(on_compiled, **on_arguments)
    assert seen and on.key == off.key
    assert {n: r.key for n, r in on.nodes.items()} == {
        n: r.key for n, r in off.nodes.items()
    }
    node_id = nodes[-1].id
    assert (
        on.population(node_id).weights_for("household").values.tobytes()
        == off.population(node_id).weights_for("household").values.tobytes()
    )
    for name, key in on.node(node_id).opaque_artifacts.items():
        assert key == off.node(node_id).opaque_artifacts[name]
        assert on_arguments["store"].load_bytes(key) == arguments["store"].load_bytes(
            key
        )

    def forbidden(*args, **kwargs):
        pytest.fail("required cache replay executed optimizer or snapshot sink")

    monkeypatch.setattr(stage, "calibrate", forbidden)
    replay_arguments = _replace_kernel(
        on_arguments, _observer(forbidden, every=1, run_id="replay")
    )
    warm = run_graph(on_compiled, **replay_arguments, resume="require")
    assert warm.key == on.key and all(node.store_hit for node in warm.nodes.values())
    assert warm.node(node_id).opaque_artifacts == on.node(node_id).opaque_artifacts
