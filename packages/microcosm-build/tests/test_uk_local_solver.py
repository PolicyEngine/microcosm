from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import microcosm.build.uk_runtime.local_solver as local_solver
from microcosm.build.uk_runtime import (
    build_uk_rowwise_local_matrix,
    solve_prepared_local_weights,
)


def _problem():
    metrics = pd.DataFrame({"population": [1.0, 1.0, 1.0]}, index=[101, 102, 103])
    assigned = pd.Series(["E001", "E001", "S001"], index=[101, 102, 103])
    targets = pd.DataFrame({"code": ["E001", "S001"], "population": [2.0, 0.5]})
    return build_uk_rowwise_local_matrix(metrics, assigned, targets)


def test_solve_prepared_local_weights_reduces_loss_and_reports_diagnostics() -> None:
    problem = _problem()

    result = solve_prepared_local_weights(
        matrix=problem.matrix,
        targets=problem.targets,
        target_frame=problem.target_frame,
        initial_weights=np.ones(3),
        epochs=80,
        learning_rate=0.2,
        max_weight_ratio=10.0,
        seed=1,
    )

    assert result.weights.shape == (3,)
    assert result.initial_weights.tolist() == [1.0, 1.0, 1.0]
    assert result.final_loss < result.initial_loss
    assert result.n_nonzero == 3
    assert result.diagnostics["area_code"].tolist() == ["E001", "S001"]
    np.testing.assert_allclose(result.diagnostics["target"], [2.0, 0.5])


def test_solve_prepared_local_weights_rejects_nonpositive_initial_weights() -> None:
    problem = _problem()

    with pytest.raises(ValueError, match="strictly positive"):
        solve_prepared_local_weights(
            matrix=problem.matrix,
            targets=problem.targets,
            target_frame=problem.target_frame,
            initial_weights=np.array([0.0, 1.0, 1.0]),
            epochs=5,
        )


def test_solve_prepared_local_weights_uses_budget_search(monkeypatch) -> None:
    problem = _problem()
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

    low = solve_prepared_local_weights(
        matrix=problem.matrix,
        targets=problem.targets,
        target_frame=problem.target_frame,
        initial_weights=np.ones(3),
        target_records=1,
    )
    high = solve_prepared_local_weights(
        matrix=problem.matrix,
        targets=problem.targets,
        target_frame=problem.target_frame,
        initial_weights=np.ones(3),
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
def test_solve_prepared_local_weights_validates_budget_controls(kwargs, match) -> None:
    problem = _problem()

    with pytest.raises(ValueError, match=match):
        solve_prepared_local_weights(
            matrix=problem.matrix,
            targets=problem.targets,
            target_frame=problem.target_frame,
            initial_weights=np.ones(3),
            **kwargs,
        )
