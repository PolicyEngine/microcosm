"""The public solve of a prebuilt constraint system (microcosm#1585 item 8.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

import microcosm.calibrate.solve as solve_module
from microcosm.calibrate import Target, TargetSet, calibrate
from microcosm.calibrate.solve import (
    _optimize,
    _torch_constraint_matrix,
    default_target_loss_scales,
    solve_constraint_system,
)


def _problem():
    rng = np.random.default_rng(0)
    dense = rng.random((6, 50))
    dense[dense < 0.5] = 0.0
    matrix = sparse.csr_array(dense * 10.0)
    w0 = rng.uniform(0.5, 2.0, 50)
    return matrix, (matrix @ w0) * rng.uniform(0.8, 1.2, 6), w0


def _bits(values) -> list[int]:
    return np.asarray(values, dtype=np.float64).view(np.uint64).tolist()


@pytest.mark.parametrize("bakeoff", (False, True), ids=("defaults", "bakeoff"))
def test_is_the_optimize_call_it_replaces(bakeoff) -> None:
    """Bit for bit, the direct ``_optimize`` call the bake-off made, and the
    default free-mass solve."""
    matrix, targets, w0 = _problem()
    loss_weights = np.linspace(0.5, 2.0, 6)
    scales = default_target_loss_scales(targets)
    options = (
        {
            "target_loss_weights": loss_weights,
            "target_loss_scales": scales,
            "target_loss_cap": 1.0,
            "mass": "conserve",
            "max_weight_ratio": 5.0,
        }
        if bakeoff
        else {}
    )
    weights, trajectory = solve_constraint_system(
        matrix, targets, w0, epochs=40, **options
    )
    torch.manual_seed(0)
    expected = _optimize(
        _torch_constraint_matrix(matrix),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(loss_weights, dtype=torch.float32) if bakeoff else None,
        torch.tensor(scales, dtype=torch.float32),
        1.0 if bakeoff else 10.0,
        w0,
        epochs=40,
        learning_rate=0.02,
        conserve_mass=bakeoff,
        max_weight_ratio=5.0 if bakeoff else None,
        l0_lambda=0.0,
        l2_lambda=0.0,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
    )
    assert _bits(weights) == _bits(expected[0])
    assert _bits(trajectory) == _bits(expected[1])


def test_hands_the_callers_operator_to_the_module_optimize(monkeypatch) -> None:
    matrix, targets, w0 = _problem()
    operator = torch.tensor(matrix.toarray(), dtype=torch.float32)
    calls = []

    def recording(matrix, *args, **kwargs):
        calls.append((matrix, kwargs))
        return _optimize(matrix, *args, **kwargs)

    monkeypatch.setattr(solve_module, "_optimize", recording)
    solve_constraint_system(operator, targets, w0, epochs=5)
    ((passed, kwargs),) = calls
    assert passed is operator
    assert (kwargs["l0_lambda"], kwargs["target_records"]) == (0.0, None)


def test_is_calibrates_solve_on_its_compiled_problem(feasible_frame) -> None:
    frame, truths = feasible_frame(n=80)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=truths["population"] * 1.2,
                measure="household_count",
            ),
            Target(
                name="income",
                entity="household",
                value=truths["income"] * 1.1,
                measure="income",
            ),
        )
    )
    options = {"epochs": 60, "mass": "conserve", "max_weight_ratio": 4.0, "seed": 3}
    result = calibrate(frame, targets, **options)
    weights, trajectory = solve_constraint_system(
        result.problem.matrix,
        result.problem.target_vector,
        result.initial_weights,
        **options,
    )
    assert _bits(weights) == _bits(result.weights)
    assert _bits(trajectory) == _bits(result.loss_trajectory)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"mass": "neither"}, "mass must be"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"l2_basis": "other"}, "l2_basis must be one of"),
        ({"targets": np.full(6, np.nan)}, "targets must be one-dimensional and finite"),
        ({"initial_weights": np.ones(49)}, "matrix shape"),
        ({"initial_weights": np.zeros(50)}, "Weights cannot be all zero"),
    ),
)
def test_refuses_what_calibrate_and_a_frame_refuse(overrides, message) -> None:
    matrix, targets, w0 = _problem()
    arguments = {"targets": targets, "initial_weights": w0, "epochs": 5, **overrides}
    with pytest.raises(ValueError, match=message):
        solve_constraint_system(
            matrix,
            arguments.pop("targets"),
            arguments.pop("initial_weights"),
            **arguments,
        )
