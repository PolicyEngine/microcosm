"""Actual integrating-runner initial admission refuses before any fitting."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from microcosm.build.us_runtime import origin_group_budget as budgets
from microcosm.build.us_runtime import origin_group_calibration_graph as stage


def _peer(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "case", ["zero_initial", "positive_member_zero_design", "overbudget_initial"]
)
def test_complete_wrapper_preflight_refuses_without_grouped_kernel_or_optimizer(
    tmp_path, monkeypatch, case
):
    fixture = _peer("test_us_origin_group_budget")
    registry_fixture = _peer("test_us_origin_group_calibration_graph")
    manifest, store, kwargs, execution = fixture.invented_budget_graph(
        tmp_path,
        monkeypatch,
        positive=case != "zero_initial",
        positive_incoming_zero_design=case == "positive_member_zero_design",
        return_execution_args=True,
    )
    if case == "overbudget_initial":
        kwargs["design_multiplier"] = 0.125
    expected = budgets.freeze_origin_group_budget(manifest, store, **kwargs)
    document = expected.document
    initial = kwargs["expanded_population"].frame.weights_for("household").values
    assert document["group_count"] == 5
    assert len(document["household_ids"]) == 10
    assert sum(len(origin["members"]) for origin in document["origins"]) == 10
    assert set(document["group_indices"]) == set(range(5))
    if case == "zero_initial":
        assert document["zero_weight_households_with_lineage"] == 2
        assert np.count_nonzero(initial == 0) == 2
        # The frozen-incoming term makes this origin's capacity exactly zero.
        totals = expected.grouped_bounds.check(initial)
        zero = expected.grouped_bounds.absolute_bounds == 0
        assert np.count_nonzero(zero) == 1 and np.all(totals[zero] == 0)
    elif case == "positive_member_zero_design":
        assert np.all(initial > 0)
        zero_groups = [
            origin
            for origin in document["origins"]
            if float.fromhex(origin["prepared_design_hex"]) == 0
        ]
        assert len(zero_groups) == 1
        assert float.fromhex(zero_groups[0]["frozen_incoming_hex"]) > 0
        assert float.fromhex(zero_groups[0]["absolute_bound_hex"]) == 0
        with pytest.raises(ValueError, match="exceed"):
            expected.grouped_bounds.check(initial)
    else:
        assert np.all(initial > 0)
        assert np.all(expected.grouped_bounds.absolute_bounds > 0)
        with pytest.raises(ValueError, match="exceed"):
            expected.grouped_bounds.check(initial)

    calls = {"prefix": [], "grouped": [], "budget": [], "optimizer": []}

    def forbidden(category):
        def run(*args, **kw):
            calls[category].append(1)
            raise AssertionError("preflight executed " + category)

        return run

    for node in execution["prefix_graph"].nodes:
        execution["kernels"].get(node.kernel).run = forbidden("prefix")
    budget_kernel = stage.FrozenOriginBudgetKernel()
    budget_kernel.run = forbidden("budget")
    grouped_kernel = stage.GroupedDemographicCalibrationKernel()
    grouped_kernel.run = forbidden("grouped")
    execution["kernels"].register(budget_kernel)
    execution["kernels"].register(grouped_kernel)
    monkeypatch.setattr(torch.optim, "Adam", forbidden("optimizer"))
    budget_path = tmp_path / "frozen-preflight-budget.json"
    with pytest.raises(ValueError, match="strictly positive|exceed"):
        stage.run_origin_group_calibration(
            **execution,
            store=store,
            native_binding_node=kwargs["native_binding_node"],
            expanded_binding_node=kwargs["expanded_binding_node"],
            target_registry=registry_fixture._registry(),
            budget_path=budget_path,
            epochs=1,
            learning_rate=0.1,
            design_multiplier=kwargs["design_multiplier"],
            incoming_multiplier=kwargs["incoming_multiplier"],
            resume="auto",
        )
    assert budget_path.read_bytes() == expected.payload
    assert calls == {"prefix": [], "grouped": [], "budget": [], "optimizer": []}
    print(
        json.dumps(
            {
                "probe": "actual_wrapper_preflight",
                "case": case,
                "covered_household_rows": len(document["household_ids"]),
                "covered_origins": document["group_count"],
                "actual_kernel_and_optimizer_calls": calls,
            },
            sort_keys=True,
        )
    )
