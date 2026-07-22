"""UK local solve doctrine: one uniform operator, declared bounds (#495 inc 4).

Doctrine (#492/#493): no per-target calibration knobs. The doctrine solve
exposes no per-target weight or scale parameters at all; its loss cap and
weight-ratio stretch are declared constants that a change must touch (and
therefore review); and the past-cap census makes the rows a solve wrote off
first-class diagnostics instead of silent triage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_TARGET_LOSS_CAP,
    UKLocalSolveDoctrine,
    build_stacked_local_matrix,
    past_cap_census,
    solve_stacked_local_weights,
    solve_uk_local_weights_under_doctrine,
)


def _toy_problem():
    metrics = pd.DataFrame({"population": [1.0, 1.0]}, index=[101, 102])
    targets = pd.DataFrame({"code": ["E001", "S001"], "population": [1.5, 0.5]})
    return build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001", "S001"],
        household_ids=[101, 102],
    )


def test_past_cap_census_classifies_every_transition() -> None:
    targets = np.array([100.0, 100.0, 100.0, 100.0])
    initial = np.array([100.0, 350.0, 300.0, 90.0])
    final = np.array([102.0, 120.0, 320.0, 350.0])
    frame = pd.DataFrame(
        {
            "target_index": [0, 1, 2, 3],
            "area_code": ["A", "B", "C", "D"],
            "metric": ["m", "m", "m", "m"],
        }
    )
    census = past_cap_census(
        initial,
        final,
        targets,
        target_loss_cap=1.0,
        target_frame=frame,
    )
    assert census["target_loss_cap"] == 1.0
    assert census["n_targets"] == 4
    assert census["past_at_init"] == 2
    assert census["past_at_final"] == 2
    assert census["escaped"] == 1
    assert census["frozen"] == 1
    assert census["pushed_out"] == 1
    (pushed,) = census["pushed_out_rows"]
    assert pushed["area_code"] == "D"
    assert pushed["initial_abs_relative_error"] == pytest.approx(0.1)
    assert pushed["final_abs_relative_error"] == pytest.approx(2.5)


def test_past_cap_census_defaults_scales_and_validates_shapes() -> None:
    targets = np.array([10.0, 0.0])
    # Default scale is max(|target|, 1): row 0 scale 10, row 1 scale 1.
    census = past_cap_census(
        np.array([10.0, 5.0]),
        np.array([25.0, 0.0]),
        targets,
        target_loss_cap=1.0,
    )
    assert census["past_at_init"] == 1  # |5-0|/1 = 5 > 1
    assert census["past_at_final"] == 1  # |25-10|/10 = 1.5 > 1
    assert census["pushed_out"] == 1
    assert census["escaped"] == 1
    assert census["pushed_out_rows"][0]["target_index"] == 0

    with pytest.raises(ValueError, match="align"):
        past_cap_census(
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            target_loss_cap=1.0,
        )


def test_solve_result_carries_past_cap_census() -> None:
    problem = _toy_problem()
    result = solve_stacked_local_weights(
        problem,
        [1.0, 1.0],
        epochs=40,
        learning_rate=0.2,
        max_weight_ratio=10.0,
        seed=1,
    )
    census = result.past_cap_census
    assert census is not None
    assert census["n_targets"] == len(problem.targets)
    assert census["target_loss_cap"] == 10.0
    assert 0 <= census["past_at_final"] <= census["n_targets"]


def test_doctrine_constants_are_the_declared_contract() -> None:
    assert UK_LOCAL_TARGET_LOSS_CAP == 10.0
    assert UK_LOCAL_MAX_WEIGHT_RATIO == 100.0
    assert UK_LOCAL_SOLVE_DOCTRINE.target_loss_cap == UK_LOCAL_TARGET_LOSS_CAP
    assert UK_LOCAL_SOLVE_DOCTRINE.max_weight_ratio == UK_LOCAL_MAX_WEIGHT_RATIO
    assert UK_LOCAL_SOLVE_DOCTRINE.scale_rule == "default_target_loss_scales"
    assert UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule == "uniform"


def test_doctrine_rejects_tampered_bounds() -> None:
    with pytest.raises(ValueError, match="target_loss_cap"):
        UKLocalSolveDoctrine(target_loss_cap=0.0)
    with pytest.raises(ValueError, match="target_loss_cap"):
        UKLocalSolveDoctrine(target_loss_cap=float("nan"))
    with pytest.raises(ValueError, match="max_weight_ratio"):
        UKLocalSolveDoctrine(max_weight_ratio=0.5)
    with pytest.raises(ValueError, match="scale_rule"):
        UKLocalSolveDoctrine(scale_rule="bespoke")
    with pytest.raises(ValueError, match="target_weight_rule"):
        UKLocalSolveDoctrine(target_weight_rule="per_target")


def test_doctrine_solve_exposes_no_per_target_knobs() -> None:
    import inspect

    parameters = inspect.signature(solve_uk_local_weights_under_doctrine).parameters
    assert "target_loss_weights" not in parameters
    assert "target_loss_scales" not in parameters
    assert "target_loss_cap" not in parameters
    assert "max_weight_ratio" not in parameters

    problem = _toy_problem()
    with pytest.raises(TypeError):
        solve_uk_local_weights_under_doctrine(
            problem,
            [1.0, 1.0],
            target_loss_weights=[1.0, 2.0],
        )


def test_doctrine_solve_applies_declared_bounds() -> None:
    problem = _toy_problem()
    result = solve_uk_local_weights_under_doctrine(
        problem,
        [1.0, 1.0],
        epochs=40,
        learning_rate=0.2,
        seed=1,
    )
    assert result.final_loss <= result.initial_loss
    assert result.past_cap_census is not None
    assert result.past_cap_census["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    # The declared stretch bound holds on the solved weights.
    stretched = result.weights / result.initial_weights
    assert float(np.nanmax(stretched)) <= UK_LOCAL_MAX_WEIGHT_RATIO * (1 + 1e-6)

    with pytest.raises(TypeError, match="UKLocalSolveDoctrine"):
        solve_uk_local_weights_under_doctrine(
            problem,
            [1.0, 1.0],
            doctrine="loose",
        )
