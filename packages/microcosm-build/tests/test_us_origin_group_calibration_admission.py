"""Invented solver-bypass admission and actual typed-artifact rekey probes."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from microcosm.build.us_runtime import origin_group_budget as budgets
from microcosm.build.us_runtime import origin_group_calibration_graph as stage
from microcosm.calibrate import GroupedUpperBounds
from microcosm.frame import WeightKind, Weights
from microcosm.graph import KernelResult
from microcosm.graph.canonical import canonical_json


def _peer():
    name = "origin_group_admission_graph_fixture"
    path = Path(__file__).with_name("test_us_origin_group_calibration_graph.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _invalid_kernel(calls, bound_side):
    def run(self, context):
        calls.append(context.node.id)
        budget = budgets.FrozenOriginBudget.from_payload(
            context.artifacts["budget"].payload
        )
        doc = budget.document
        values = context.weights["household"].values.copy()
        by_id = {identity: i for i, identity in enumerate(doc["household_ids"])}
        for group in doc["origins"]:
            bound = float.fromhex(group[bound_side + "_bound_hex"])
            assert bound == float.fromhex(group["absolute_bound_hex"])
            # Both roles are individually below the redundant row-DESIGN guard,
            # but their true shared sum is 1.5 times the selected inequality.
            for member in group["members"]:
                values[by_id[member["household_id"]]] = bound * 0.75
                assert bound * 0.75 <= float.fromhex(group["design_bound_hex"])
            assert (
                math.fsum(values[by_id[m["household_id"]]] for m in group["members"])
                > bound
            )
        return KernelResult(
            weights=Weights(values, WeightKind.CALIBRATED),
            artifacts={"diagnostics": canonical_json({"invented_bypass": True})},
            receipt={"invented_solver_bypass": True},
        )

    return run


@pytest.mark.parametrize(
    ("bound_side", "design_multiplier", "incoming_multiplier"),
    [("design", 2.0, 1e9), ("incoming", 1e9, 2.0)],
)
def test_cold_and_required_warm_refuse_solver_bypass(
    tmp_path, monkeypatch, bound_side, design_multiplier, incoming_multiplier
):
    helper = _peer()
    root = tmp_path / "invented"
    helper._fixture(root)
    bypass_calls = []
    monkeypatch.setattr(
        stage.GroupedDemographicCalibrationKernel,
        "run",
        _invalid_kernel(bypass_calls, bound_side),
    )
    options = dict(
        design_multiplier=design_multiplier, incoming_multiplier=incoming_multiplier
    )
    with pytest.raises(ValueError, match="exceed frozen absolute bounds"):
        helper._run(root, **options)
    assert bypass_calls == [stage.CALIBRATION_NODE]

    # Create a deliberately invalid legacy cache through the real Graph/store
    # path, without the integrating runner's observer. This is test setup only:
    # stop before its final admission code, then prove REQUIRED admission rejects.
    real_run_graph = stage.run_graph

    class SeededInvalidCacheError(Exception):
        pass

    seeded_keys = {}

    def seed(compiled, **kwargs):
        if stage.CALIBRATION_NODE not in {n.id for n in compiled.graph.nodes}:
            return real_run_graph(compiled, **kwargs)
        kwargs["_population_observer"] = None
        manifest = real_run_graph(compiled, **kwargs)
        seeded_keys.update(
            {name: receipt.key for name, receipt in manifest.nodes.items()}
        )
        raise SeededInvalidCacheError

    monkeypatch.setattr(stage, "run_graph", seed)
    with pytest.raises(SeededInvalidCacheError):
        helper._run(root, **options)
    assert bypass_calls == [stage.CALIBRATION_NODE] * 2
    assert stage.CALIBRATION_NODE in seeded_keys

    warm_calls = []

    def guarded_warm(compiled, **kwargs):
        registry = kwargs["kernels"]
        for ref in {n.kernel for n in compiled.graph.nodes}:
            kernel = registry.get(ref)

            def forbidden(context, ref=ref):
                warm_calls.append(ref)
                raise AssertionError("REQUIRED admission executed kernel " + ref)

            kernel.run = forbidden
        return real_run_graph(compiled, **kwargs)

    monkeypatch.setattr(stage, "run_graph", guarded_warm)
    with pytest.raises(ValueError, match="exceed frozen absolute bounds"):
        helper._run(root, warm=True, **options)
    assert warm_calls == []
    assert bypass_calls == [stage.CALIBRATION_NODE] * 2
    print(
        json.dumps(
            {
                "probe": "solver_bypass",
                "bound": bound_side,
                "cold_refused": True,
                "warm_refused": True,
                "actual_warm_kernel_call_count": len(warm_calls),
                "seeded_calibration_key": seeded_keys[stage.CALIBRATION_NODE],
            },
            sort_keys=True,
        )
    )


def test_only_group_map_typed_artifact_rekeys_actual_graph_and_refuses_authority(
    tmp_path, monkeypatch
):
    helper = _peer()
    root = tmp_path / "invented"
    helper._fixture(root)
    real_run_graph = stage.run_graph
    captured = {}

    def capture(compiled, **kwargs):
        if stage.CALIBRATION_NODE in {n.id for n in compiled.graph.nodes}:
            captured.update(compiled=compiled, kwargs=kwargs)
        return real_run_graph(compiled, **kwargs)

    monkeypatch.setattr(stage, "run_graph", capture)
    cold = helper._run(root)
    frozen_bytes = (root / "frozen-budget.json").read_bytes()
    original = budgets.FrozenOriginBudget.from_payload(frozen_bytes)
    document = original.document
    mapping = document["group_indices"]
    other = next(i for i in range(1, len(mapping)) if mapping[i] != mapping[0])
    mapping[0], mapping[other] = mapping[other], mapping[0]
    groups = GroupedUpperBounds(
        document["household_ids"],
        mapping,
        [float.fromhex(value) for value in document["absolute_bound_hex"]],
    )
    document["constraint_digest"] = groups.constraint_digest
    changed = budgets.FrozenOriginBudget.from_payload(canonical_json(document))
    assert {key for key in document if document[key] != original.document[key]} == {
        "group_indices",
        "constraint_digest",
    }
    changed_path = root / "map-only-budget.json"
    changed_path.write_bytes(changed.payload)

    calls = []

    def passthrough(context):
        calls.append(context.node.id)
        return KernelResult(
            weights=Weights(
                context.weights["household"].values.copy(), WeightKind.CALIBRATED
            ),
            artifacts={"diagnostics": canonical_json({"invented_key_probe": True})},
            receipt={"invented_key_probe": True},
        )

    kwargs = dict(captured["kwargs"])
    kwargs["kernels"].get(
        stage.GroupedDemographicCalibrationKernel.ref
    ).run = passthrough
    kwargs["sources"] = {**kwargs["sources"], stage.BUDGET_SOURCE: changed_path}
    kwargs["_population_observer"] = None
    changed_manifest = real_run_graph(captured["compiled"], **kwargs)
    changed_keys = {
        name: receipt.key for name, receipt in changed_manifest.nodes.items()
    }
    assert calls == [stage.CALIBRATION_NODE]
    assert {
        name for name in cold["keys"] if cold["keys"][name] != changed_keys[name]
    } == {stage.BUDGET_NODE, stage.CALIBRATION_NODE}
    assert (root / "frozen-budget.json").read_bytes() == frozen_bytes

    original_runner = stage.run_origin_group_calibration

    def offered_changed_budget(*args, **kwargs):
        kwargs["budget_path"] = changed_path
        return original_runner(*args, **kwargs)

    monkeypatch.setattr(stage, "run_origin_group_calibration", offered_changed_budget)
    with pytest.raises(budgets.OriginBudgetError, match="BUDGET_MATERIALIZED_IDENTITY"):
        helper._run(root, warm=True)
    print(
        json.dumps(
            {
                "probe": "map_only_rekey",
                "changed_nodes": [stage.BUDGET_NODE, stage.CALIBRATION_NODE],
                "original_constraint_digest": original.grouped_bounds.digest,
                "changed_constraint_digest": groups.digest,
                "fabricated_authority_refused": True,
            },
            sort_keys=True,
        )
    )
