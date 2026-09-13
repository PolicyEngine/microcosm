"""Ordinary overshoot contracts for the combined US-launch solver.

The count targets, starts and epoch budgets come from current main's
test_solve.py best-iterate cases. All populations are invented in this module.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from microcosm.calibrate import (
    GroupedUpperBounds,
    Target,
    TargetSet,
    TargetSnapshotCadence,
    TargetSnapshotObserver,
    calibrate,
    diagnostics_payload,
)
from microcosm.calibrate import solve as solve_module
from microcosm.calibrate.target_snapshots import EVERY_EPOCH, ITERATE_SELECTED
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _count_frame(weights: tuple[float, ...]) -> Frame:
    ids = tuple(range(len(weights)))
    return Frame(
        {
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
            "household": pd.DataFrame(
                {"household_id": ids, "household_count": np.ones(len(ids))}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights), WeightKind.DESIGN)},
    )


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
@pytest.mark.parametrize("fixed_zeros", [False, True], ids=["positive", "fixed-zeros"])
@pytest.mark.parametrize(
    "target,warm,epochs,selected_epoch",
    [
        pytest.param(2.01, 2.0, 2, 0, id="warm-start-overshoot"),
        pytest.param(1.5, None, 3, 2, id="intermediate-overshoot"),
    ],
)
def test_grouped_closing_state_and_ordinary_best_iterate_remain_distinct(
    monkeypatch, sparse, fixed_zeros, target, warm, epochs, selected_epoch
) -> None:
    """A worse closing step remains grouped; ordinary Adam selects the best.

    The bound of three is deliberately loose, so projection cannot account for
    the difference. Fixed-support cases add both a zero-only group and a zero
    row in the positive group while keeping the same active count problem.
    """
    monkeypatch.setattr(solve_module, "_SPARSE_MIN_CELLS", 1 if sparse else 1000)
    monkeypatch.setattr(solve_module, "_SPARSE_DENSITY_CUTOFF", 1.0)
    layouts = []
    apply_constraint = solve_module._apply_constraint

    def observe_layout(matrix, weights):
        layouts.append(matrix.layout)
        return apply_constraint(matrix, weights)

    monkeypatch.setattr(solve_module, "_apply_constraint", observe_layout)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=target,
                measure="household_count",
            ),
        )
    )
    common = {"epochs": epochs, "learning_rate": 0.2, "mass": "free", "seed": 0}
    ordinary = calibrate(
        _count_frame((1.0,)),
        targets,
        warm_start_weights=None if warm is None else np.array([warm]),
        **common,
    )
    initial = np.array([0.0, 1.0, 0.0] if fixed_zeros else [1.0])
    grouped_frame = _count_frame(tuple(initial))
    groups = GroupedUpperBounds(
        tuple(range(len(initial))),
        np.array([0, 1, 1] if fixed_zeros else [0]),
        np.array([0.0, 3.0] if fixed_zeros else [3.0]),
    )
    observations = []
    grouped = calibrate(
        grouped_frame,
        targets,
        warm_start_weights=None if warm is None else initial * warm,
        grouped_upper_bounds=groups,
        grouped_preserve_zeros=fixed_zeros,
        _post_projection_observer=observations.append,
        **common,
    )
    expected_layout = torch.sparse_csr if sparse else torch.strided
    assert layouts and all(layout == expected_layout for layout in layouts)
    assert ordinary.options["iterate_selection"] == "best_feasible_loss"
    receipt = ordinary.options["iterate_selection_receipt"]
    assert receipt["rule"] == "best_feasible_loss"
    assert receipt["selected_epoch"] == selected_epoch
    assert receipt["epochs_executed"] == epochs
    assert receipt["epoch_convention"] == "completed_optimizer_updates; zero is start"
    assert receipt["selected_loss_float32"] == ordinary.loss_trajectory[selected_epoch]
    assert receipt["closing_iterate_loss_float32"] > ordinary.final_loss + 0.01
    assert ordinary.final_loss == pytest.approx(
        abs(float(ordinary.weights.sum()) - target) / target, abs=1e-12
    )
    assert len(ordinary.loss_trajectory) == epochs
    assert ordinary.final_loss <= ordinary.loss_trajectory.min() + 1e-7

    assert grouped.options["iterate_selection"] == "closing_state"
    assert grouped.options["iterate_selection_receipt"] == {}
    assert grouped.gate_open_probabilities is None
    for result in (ordinary, grouped):
        assert result.options["gate_initialization_supplied"] is False
        assert result.options["budget_basis"] == "nonzero_count"
        assert result.options["feasible_draw_pi_hi"] is None
        assert result.options["budget_search"] is None
        payload = diagnostics_payload(result)
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
        assert payload["options"] == result.options
    assert "grouped_upper_bounds" not in ordinary.options
    assert "grouped_preserve_zeros" not in ordinary.options
    assert ordinary.realized_max_weight_ratio == float(
        (ordinary.weights / ordinary.initial_weights).max()
    )
    assert len(observations) == epochs + 1
    assert [item["epoch"] for item in observations] == list(range(epochs + 1))
    assert all(item["corrected_group_count"] == 0 for item in observations)
    expected_start = initial if warm is None else initial * warm
    assert observations[0]["weights"].tobytes() == expected_start.tobytes()
    observed_losses = []
    for item in observations:
        weights = item["weights"]
        assert weights.dtype == np.dtype("float64")
        assert item["household_ids"] == groups.household_ids
        np.testing.assert_array_equal(item["group_indices"], groups.group_indices)
        np.testing.assert_array_equal(item["absolute_bounds"], groups.absolute_bounds)
        np.testing.assert_array_equal(
            item["group_totals"], groups.check(weights, positive=not fixed_zeros)
        )
        observed_losses.append(abs(float(weights.sum()) - target) / target)
    assert int(np.argmin(observed_losses)) == selected_epoch
    assert observed_losses[-1] > min(observed_losses) + 0.01
    np.testing.assert_allclose(
        grouped.loss_trajectory, observed_losses[:-1], rtol=2e-6, atol=2e-7
    )
    assert grouped.final_loss == pytest.approx(observed_losses[-1], abs=1e-12)
    assert grouped.final_loss > ordinary.final_loss + 0.01
    assert grouped.weights.tobytes() == observations[-1]["weights"].tobytes()
    stored = grouped.frame.resolve_weights("household").values
    assert stored.tobytes() == grouped.weights.tobytes()
    assert grouped.initial_weights.tobytes() == initial.tobytes()
    assert grouped.options["grouped_upper_bounds"] == groups.diagnostics(
        grouped.weights, 0
    )
    for entity in ("person", "household"):
        pd.testing.assert_frame_equal(
            grouped.frame.table(entity), grouped_frame.table(entity)
        )
    if fixed_zeros:
        zero_mask = initial == 0
        for item in observations:
            assert item["weights"][zero_mask].tobytes() == initial[zero_mask].tobytes()
        assert stored[zero_mask].tobytes() == initial[zero_mask].tobytes()
        assert grouped.options["grouped_preserve_zeros"] == {
            "enabled": True,
            "fixed_zero_count": 2,
            "ordered_zero_mask_sha256": hashlib.sha256(bytes([1, 0, 1])).hexdigest(),
        }
        assert np.isfinite(grouped.realized_max_weight_ratio)
        assert grouped.realized_max_weight_ratio == grouped.weights[1]
    else:
        assert "grouped_preserve_zeros" not in grouped.options

    # Ordinary checkpoint reconstruction retains the merged options and
    # diagnostics. A grouped receipt alone cannot re-admit aligned group caps.
    def rebuild(result, frame):
        return solve_module.rebuild_calibration_result(
            frame,
            targets,
            weights=result.weights,
            loss_trajectory=result.loss_trajectory,
            l0_lambda=result.l0_lambda,
            n_nonzero=result.n_nonzero,
            target_loss_weights=result.target_loss_weights,
            target_loss_scales=result.target_loss_scales,
            target_loss_cap=result.target_loss_cap,
            options=result.options,
            closing_loss=result.final_loss,
        )

    restored = rebuild(ordinary, _count_frame((1.0,)))
    assert diagnostics_payload(restored) == diagnostics_payload(ordinary)
    assert restored.weights.tobytes() == ordinary.weights.tobytes()

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("grouped checkpoint compiled without constraints")

    monkeypatch.setattr(solve_module, "build_constraint_matrix", forbidden_compile)
    with pytest.raises(ValueError, match="original aligned constraints"):
        rebuild(grouped, grouped_frame)


@pytest.mark.parametrize("fixed_zeros", [False, True], ids=["positive", "fixed-zeros"])
@pytest.mark.parametrize(
    "target,warm,epochs,selected_epoch",
    [
        pytest.param(2.01, 2.0, 2, 0, id="warm-start-overshoot"),
        pytest.param(1.5, None, 3, 2, id="intermediate-overshoot"),
    ],
)
def test_snapshots_describe_each_solver_s_own_selection_on_the_same_oscillation(
    fixed_zeros, target, warm, epochs, selected_epoch
) -> None:
    """The same oscillating example, now observed by both solvers.

    Ordinary Adam's closing snapshot must name the earlier epoch it actually
    returned and admit a retained best; the grouped run's must name the closing
    epoch and say no best iterate was retained at all. A single reused emitter
    that inferred retain-best from an empty receipt would look right here only
    by accident, which is why the grouped label is asserted too.
    """
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=target,
                measure="household_count",
            ),
        )
    )
    common = {"epochs": epochs, "learning_rate": 0.2, "mass": "free", "seed": 0}

    def observer(seen):
        return TargetSnapshotObserver(
            sink=seen.append,
            run_id="oscillation",
            cadence=TargetSnapshotCadence(every=EVERY_EPOCH),
        )

    ordinary_seen = []
    ordinary = calibrate(
        _count_frame((1.0,)),
        targets,
        warm_start_weights=None if warm is None else np.array([warm]),
        target_snapshots=observer(ordinary_seen),
        **common,
    )
    initial = np.array([0.0, 1.0, 0.0] if fixed_zeros else [1.0])
    groups = GroupedUpperBounds(
        tuple(range(len(initial))),
        np.array([0, 1, 1] if fixed_zeros else [0]),
        np.array([0.0, 3.0] if fixed_zeros else [3.0]),
    )
    grouped_seen = []
    grouped = calibrate(
        _count_frame(tuple(initial)),
        targets,
        warm_start_weights=None if warm is None else initial * warm,
        grouped_upper_bounds=groups,
        grouped_preserve_zeros=fixed_zeros,
        target_snapshots=observer(grouped_seen),
        **common,
    )

    # The example is still an oscillation the two rules resolve differently.
    assert grouped.final_loss > ordinary.final_loss + 0.01

    ordinary_selected = [
        item for item in ordinary_seen if item["iterate"] == ITERATE_SELECTED
    ]
    grouped_selected = [
        item for item in grouped_seen if item["iterate"] == ITERATE_SELECTED
    ]
    assert len(ordinary_selected) == len(grouped_selected) == 1
    ordinary_selected = ordinary_selected[0]
    grouped_selected = grouped_selected[0]

    assert ordinary_selected["epoch"] == selected_epoch
    assert ordinary_selected["epoch"] < epochs
    assert ordinary_selected["best_retained"]["available"] is True
    assert ordinary_selected["selection"]["rule"] == "best_feasible_loss"
    assert ordinary_selected["selection"]["selected_epoch"] == selected_epoch

    assert grouped_selected["epoch"] == epochs
    assert grouped_selected["best_retained"] == {
        "available": False,
        "epoch": None,
        "loss": None,
    }
    assert grouped_selected["selection"] == {
        "rule": "closing_state",
        "constraint_mode": "grouped_upper_bounds",
        "grouped_preserve_zeros": fixed_zeros,
    }

    # Both closing snapshots are the float64 estimates their result returned.
    for selected, result in (
        (ordinary_selected, ordinary),
        (grouped_selected, grouped),
    ):
        assert selected["precision"] == "float64"
        assert selected["epochs"] == epochs
        assert [row["estimate"] for row in selected["targets"]] == [
            diagnostic.final_estimate for diagnostic in result.diagnostics
        ]
        assert selected["loss"] == float(result.closing_loss)

    # Every in-loop grouped snapshot reports the closing-state rule; the
    # ordinary run's carry its own best-iterate bookkeeping instead.
    for item in grouped_seen:
        assert item["best_retained"]["available"] is False
    assert any(
        item["best_retained"]["available"] is True
        for item in ordinary_seen
        if item["iterate"] != ITERATE_SELECTED
    )
