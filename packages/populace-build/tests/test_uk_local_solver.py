from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import populace.build.uk_runtime.local_solver as local_solver
from populace.build.uk_runtime import (
    build_assigned_local_matrix,
    build_stacked_local_matrix,
    solve_assigned_local_weights,
    solve_stacked_local_weights,
)


def test_solve_stacked_local_weights_reduces_loss_and_reports_diagnostics() -> None:
    metrics = pd.DataFrame({"population": [1.0, 1.0]}, index=[101, 102])
    targets = pd.DataFrame({"code": ["E001", "S001"], "population": [1.5, 0.5]})
    problem = build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001", "S001"],
        household_ids=[101, 102],
    )

    result = solve_stacked_local_weights(
        problem,
        [1.0, 1.0],
        epochs=80,
        learning_rate=0.2,
        max_weight_ratio=10.0,
        seed=1,
    )

    assert result.weights.shape == (4,)
    assert result.initial_weights.tolist() == [0.5, 0.5, 0.5, 0.5]
    assert result.final_loss < result.initial_loss
    assert result.n_nonzero == 4
    assert result.diagnostics["area_code"].tolist() == ["E001", "S001"]
    np.testing.assert_allclose(result.diagnostics["target"], [1.5, 0.5])


def test_solve_assigned_local_weights_uses_household_weight_columns() -> None:
    metrics = pd.DataFrame({"population": [1.0, 1.0]}, index=[101, 102])
    targets = pd.DataFrame({"code": ["E001", "S001"], "population": [1.5, 0.5]})
    households = pd.DataFrame(
        {
            "household_id": [101, 102],
            "constituency_code_oa": ["E001", "S001"],
        }
    )
    problem = build_assigned_local_matrix(
        metrics,
        targets,
        household_frame=households,
        area_codes=["E001", "S001"],
        household_ids=[101, 102],
    )

    result = solve_assigned_local_weights(
        problem,
        [1.0, 1.0],
        epochs=80,
        learning_rate=0.2,
        max_weight_ratio=10.0,
        seed=1,
    )

    assert result.weights.shape == (2,)
    assert result.initial_weights.tolist() == [1.0, 1.0]
    assert result.final_loss < result.initial_loss
    assert result.diagnostics["area_code"].tolist() == ["E001", "S001"]


def test_solve_assigned_local_weights_can_upweight_zero_base_support() -> None:
    metrics = pd.DataFrame({"income": [1_000_000.0]}, index=[101])
    targets = pd.DataFrame({"code": ["E001"], "income": [1_000_000.0]})
    households = pd.DataFrame(
        {
            "household_id": [101],
            "constituency_code_oa": ["E001"],
        }
    )
    problem = build_assigned_local_matrix(
        metrics,
        targets,
        household_frame=households,
        area_codes=["E001"],
        household_ids=[101],
    )

    result = solve_assigned_local_weights(
        problem,
        [0.0],
        epochs=80,
        learning_rate=0.3,
        seed=1,
    )

    assert result.weights[0] > 0.01
    assert result.final_loss < 0.05


def test_solve_stacked_local_weights_uses_explicit_positive_floor() -> None:
    metrics = pd.DataFrame({"population": [1.0, 1.0]}, index=[101, 102])
    targets = pd.DataFrame({"code": ["E001"], "population": [1.0]})
    problem = build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001"],
        household_ids=[101, 102],
    )

    result = solve_stacked_local_weights(
        problem,
        [0.0, 1.0],
        epochs=5,
        min_initial_weight=1e-3,
    )

    assert result.initial_weights.tolist() == [1e-3, 1.0]


def test_solve_stacked_local_weights_rejects_nonpositive_initial_weights() -> None:
    metrics = pd.DataFrame({"population": [1.0, 1.0]}, index=[101, 102])
    targets = pd.DataFrame({"code": ["E001"], "population": [1.0]})
    problem = build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001"],
        household_ids=[101, 102],
    )

    with pytest.raises(ValueError, match="strictly positive"):
        solve_stacked_local_weights(
            problem,
            [0.0, 1.0],
            epochs=5,
            min_initial_weight=0.0,
        )


def test_solve_stacked_local_weights_uses_budget_search(monkeypatch) -> None:
    metrics = pd.DataFrame({"population": [1.0, 1.0, 1.0]}, index=[101, 102, 103])
    targets = pd.DataFrame({"code": ["E001"], "population": [2.0]})
    problem = build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001"],
        household_ids=[101, 102, 103],
    )
    calls = []

    def fake_budget_search(
        matrix,
        targets_tensor,
        target_loss_weights,
        target_loss_scales,
        target_loss_cap,
        initial_weights,
        *,
        target_records,
        **kwargs,
    ):
        assert isinstance(matrix, torch.Tensor)
        calls.append(target_records)
        weights = np.zeros_like(initial_weights)
        weights[:target_records] = 1.0
        return weights, np.asarray([1.0, 0.5]), 0.01, target_records

    monkeypatch.setattr(
        local_solver,
        "_calibrate_search_l0_lambda_for_budget",
        fake_budget_search,
    )

    low = solve_stacked_local_weights(
        problem,
        [1.0, 1.0, 1.0],
        target_records=1,
    )
    high = solve_stacked_local_weights(
        problem,
        [1.0, 1.0, 1.0],
        target_records=3,
    )

    assert calls == [1, 3]
    assert low.n_nonzero == 1
    assert high.n_nonzero == 3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target_records": 1.5}, "target_records"),
        ({"target_records": 1, "budget_iters": 0}, "budget_iters"),
    ],
)
def test_solve_stacked_local_weights_validates_budget_controls(kwargs, match) -> None:
    metrics = pd.DataFrame({"population": [1.0, 1.0]}, index=[101, 102])
    targets = pd.DataFrame({"code": ["E001"], "population": [1.0]})
    problem = build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001"],
        household_ids=[101, 102],
    )

    with pytest.raises(ValueError, match=match):
        solve_stacked_local_weights(problem, [1.0, 1.0], **kwargs)
