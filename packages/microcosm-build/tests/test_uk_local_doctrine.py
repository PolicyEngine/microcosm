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

from microcosm.build.uk_runtime import (
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


def test_doctrine_solve_exposes_no_knobs_and_no_injection_point() -> None:
    import inspect

    parameters = inspect.signature(solve_uk_local_weights_under_doctrine).parameters
    assert "target_loss_weights" not in parameters
    assert "target_loss_scales" not in parameters
    assert "target_loss_cap" not in parameters
    assert "max_weight_ratio" not in parameters
    # No doctrine parameter either: a caller cannot mint a locally revised
    # contract and route it through the release path.
    assert "doctrine" not in parameters

    problem = _toy_problem()
    with pytest.raises(TypeError):
        solve_uk_local_weights_under_doctrine(
            problem,
            [1.0, 1.0],
            target_loss_weights=[1.0, 2.0],
        )
    with pytest.raises(TypeError):
        solve_uk_local_weights_under_doctrine(
            problem,
            [1.0, 1.0],
            doctrine=UKLocalSolveDoctrine(target_loss_cap=77.0),
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
    assert np.isfinite(result.final_loss)
    assert result.past_cap_census is not None
    assert result.past_cap_census["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    # The declared stretch bound holds on the solved weights (the optimizer
    # applies a hard closing cap against the same initial weights).
    assert np.isfinite(result.weights).all()
    stretched = result.weights / result.initial_weights
    assert float(np.max(stretched)) <= UK_LOCAL_MAX_WEIGHT_RATIO * (1 + 1e-6)


def test_doctrine_solve_refuses_duplicate_target_surface() -> None:
    import dataclasses

    problem = _toy_problem()
    doctored_frame = pd.concat(
        [problem.target_frame, problem.target_frame.iloc[[0]]],
        ignore_index=True,
    )
    doctored = dataclasses.replace(problem, target_frame=doctored_frame)
    with pytest.raises(ValueError, match="per-target weights"):
        solve_uk_local_weights_under_doctrine(doctored, [1.0, 1.0], epochs=1)


def test_census_boundary_and_scale_refusals() -> None:
    targets = np.array([100.0, 100.0])
    # A row exactly AT the cap is not past it: strictly-greater semantics
    # match the torch loss, where a tie still carries gradient.
    census = past_cap_census(
        np.array([200.0, 100.0]),
        np.array([200.0, 100.0]),
        targets,
        target_loss_cap=1.0,
    )
    assert census["past_at_init"] == 0
    assert census["past_at_final"] == 0

    with pytest.raises(ValueError, match="finite and positive"):
        past_cap_census(
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            targets,
            target_loss_cap=1.0,
            target_loss_scales=np.array([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="estimates must be finite"):
        past_cap_census(
            np.array([np.nan, 1.0]),
            np.array([1.0, 1.0]),
            targets,
            target_loss_cap=1.0,
        )


def test_census_lists_every_pushed_out_row_unless_bounded() -> None:
    n = 130
    targets = np.full(n, 100.0)
    initial = np.full(n, 100.0)
    final = np.full(n, 100.0 + 100.0 * 5.0)
    unbounded = past_cap_census(initial, final, targets, target_loss_cap=1.0)
    assert unbounded["pushed_out"] == n
    assert len(unbounded["pushed_out_rows"]) == n
    assert unbounded["pushed_out_rows_truncated"] is False

    bounded = past_cap_census(
        initial, final, targets, target_loss_cap=1.0, max_listed_rows=100
    )
    assert len(bounded["pushed_out_rows"]) == 100
    assert bounded["pushed_out_rows_truncated"] is True


def test_runner_under_doctrine_refuses_knobs_and_persists_census(tmp_path) -> None:
    from microcosm.build.uk_runtime import (
        build_local_candidate,
        summarize_local_candidate,
        write_local_candidate_outputs,
    )

    areas = pd.DataFrame({"code": ["E001", "S001"], "country": ["England", "Scotland"]})
    targets = pd.DataFrame({"code": ["E001", "S001"], "population": [2.0, 2.0]})
    metrics = {
        "England": pd.DataFrame({"population": [1.0, 0.0]}, index=[101, 102]),
        "Scotland": pd.DataFrame({"population": [0.0, 1.0]}, index=[101, 102]),
    }
    households = pd.DataFrame(
        {
            "household_id": [101, 102],
            "household_weight": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="under_doctrine refuses"):
        build_local_candidate(
            area_type="constituency",
            area_frame=areas,
            targets=targets,
            metrics=metrics,
            household_frame=households,
            solver_options={"target_loss_cap": 2.0},
            under_doctrine=True,
        )

    result = build_local_candidate(
        area_type="constituency",
        area_frame=areas,
        targets=targets,
        metrics=metrics,
        household_frame=households,
        solver_options={"epochs": 40, "learning_rate": 0.2, "seed": 1},
        under_doctrine=True,
    )
    census = result.solve_result.past_cap_census
    assert census is not None
    assert census["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP

    summary = write_local_candidate_outputs(result, tmp_path)
    assert summary == summarize_local_candidate(result)
    assert summary["past_cap"]["n_targets"] == census["n_targets"]
    import json

    persisted = json.loads((tmp_path / "past_cap_census.json").read_text())
    assert persisted["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
