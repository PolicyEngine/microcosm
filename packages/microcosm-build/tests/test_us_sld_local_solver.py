"""Tests for the per-district SLD solver and its doctrine (populace#625)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.sld_local_doctrine import (
    US_SLD_LOCAL_MAX_WEIGHT_RATIO,
    US_SLD_LOCAL_SOLVE_DOCTRINE,
    US_SLD_LOCAL_TARGET_LOSS_CAP,
    UsSldLocalSolveDoctrine,
    solve_us_sld_chamber_under_doctrine,
    solve_us_sld_district_weights_under_doctrine,
)
from microcosm.build.us_runtime.sld_local_solver import (
    SldDistrictProblem,
    past_cap_census,
    solve_sld_chamber,
    solve_sld_district_weights,
)


def _problem(
    *,
    area_code: str = "610U900US49001",
    district_code: str = "001",
    target_scale: float = 1.0,
    base_weights: np.ndarray | None = None,
) -> SldDistrictProblem:
    """Five households, three targets solvable at weights ~2x the anchor."""
    matrix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],  # households
            [2.0, 1.0, 0.0, 3.0, 1.0],  # persons in an age band
            [0.0, 1.0, 1.0, 0.0, 1.0],  # households in an income bracket
        ]
    )
    weights = (
        np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        if base_weights is None
        else base_weights
    )
    achievable = matrix @ (weights * 2.0) * target_scale
    return SldDistrictProblem(
        area_type="sldu",
        area_code=area_code,
        state_fips="49",
        district_code=district_code,
        matrix=matrix,
        targets=achievable,
        target_frame=pd.DataFrame(
            {
                "area_type": ["sldu"] * 3,
                "area_code": [area_code] * 3,
                "metric": ["households", "age_0_to_4", "income_under_10000"],
            }
        ),
        household_ids=np.array([101, 102, 103, 104, 105]),
        base_weights=weights,
    )


def test_solve_hits_achievable_targets_from_the_anchor():
    result = solve_sld_district_weights(_problem(), epochs=400, seed=0)
    assert result.final_loss < result.initial_loss
    assert result.fraction_within_10pct == 1.0
    assert result.realized_max_weight_ratio <= 100.0
    assert result.past_cap_census["pushed_out"] == 0
    assert list(result.diagnostics["metric"]) == [
        "households",
        "age_0_to_4",
        "income_under_10000",
    ]


def test_solve_is_deterministic_from_seed():
    first = solve_sld_district_weights(_problem(), epochs=64, seed=3)
    second = solve_sld_district_weights(_problem(), epochs=64, seed=3)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_ratio_bound_holds_against_an_unreachable_target():
    problem = _problem(target_scale=1000.0)
    result = solve_sld_district_weights(
        problem,
        epochs=200,
        max_weight_ratio=5.0,
        seed=0,
    )
    assert result.realized_max_weight_ratio <= 5.0 + 1e-9
    assert result.weights.max() <= problem.base_weights.max() * 5.0 + 1e-6


def test_zero_base_weights_are_floored_and_counted():
    weights = np.array([10.0, 0.0, 10.0, 10.0, 0.0])
    result = solve_sld_district_weights(
        _problem(base_weights=weights),
        epochs=16,
        seed=0,
    )
    assert result.n_floored_base_weights == 2
    assert (result.initial_weights > 0).all()


def test_problem_validation_refuses_misaligned_shapes():
    problem = _problem()
    with pytest.raises(ValueError, match="columns"):
        SldDistrictProblem(
            area_type="sldu",
            area_code="x",
            state_fips="49",
            district_code="001",
            matrix=problem.matrix,
            targets=problem.targets,
            target_frame=problem.target_frame,
            household_ids=problem.household_ids[:-1],
            base_weights=problem.base_weights[:-1],
        )
    with pytest.raises(ValueError, match="area_type"):
        SldDistrictProblem(
            area_type="county",
            area_code="x",
            state_fips="49",
            district_code="001",
            matrix=problem.matrix,
            targets=problem.targets,
            target_frame=problem.target_frame,
            household_ids=problem.household_ids,
            base_weights=problem.base_weights,
        )


def test_past_cap_census_classifies_every_transition():
    census = past_cap_census(
        initial_estimates=np.array([0.0, 30.0, 30.0, 10.0]),
        final_estimates=np.array([10.0, 30.0, 9.5, 30.0]),
        targets=np.array([10.0, 10.0, 10.0, 10.0]),
        target_loss_cap=1.0,
        target_frame=pd.DataFrame(
            {
                "area_type": ["sldu"] * 4,
                "area_code": ["a"] * 4,
                "metric": ["m1", "m2", "m3", "m4"],
            }
        ),
    )
    assert census["past_at_init"] == 2
    assert census["past_at_final"] == 2
    assert census["escaped"] == 1
    assert census["frozen"] == 1
    assert census["pushed_out"] == 1
    assert census["pushed_out_rows"][0]["metric"] == "m4"
    assert census["past_at_init"] == census["escaped"] + census["frozen"]
    assert census["past_at_final"] == census["frozen"] + census["pushed_out"]


def test_chamber_solve_orders_and_rolls_up():
    problems = [
        _problem(area_code="610U900US49002", district_code="002"),
        _problem(area_code="610U900US49001", district_code="001"),
    ]
    result = solve_sld_chamber(problems, epochs=64, seed=0)
    assert [r.problem.area_code for r in result.district_results] == [
        "610U900US49001",
        "610U900US49002",
    ]
    assert list(result.long_weights.columns) == [
        "area_type",
        "area_code",
        "household_id",
        "weight",
        "weight_source",
    ]
    assert len(result.long_weights) == 10
    assert result.census_rollup["n_districts"] == 2
    assert result.census_rollup["pushed_out"] == sum(
        r.past_cap_census["pushed_out"] for r in result.district_results
    )
    shuffled = solve_sld_chamber(list(reversed(problems)), epochs=64, seed=0)
    pd.testing.assert_frame_equal(result.long_weights, shuffled.long_weights)


def test_chamber_solve_refuses_duplicate_codes_and_mixed_chambers():
    with pytest.raises(ValueError, match="unique"):
        solve_sld_chamber([_problem(), _problem()], epochs=8)
    lower = SldDistrictProblem(
        area_type="sldl",
        area_code="620L900US49001",
        state_fips="49",
        district_code="001",
        matrix=_problem().matrix,
        targets=_problem().targets,
        target_frame=_problem().target_frame,
        household_ids=_problem().household_ids,
        base_weights=_problem().base_weights,
    )
    with pytest.raises(ValueError, match="area_type"):
        solve_sld_chamber([_problem(), lower], epochs=8)


def test_doctrine_constants_are_pinned():
    assert US_SLD_LOCAL_TARGET_LOSS_CAP == 10.0
    assert US_SLD_LOCAL_MAX_WEIGHT_RATIO == 100.0
    assert US_SLD_LOCAL_SOLVE_DOCTRINE.scale_rule == "default_target_loss_scales"
    assert US_SLD_LOCAL_SOLVE_DOCTRINE.target_weight_rule == "uniform"
    assert US_SLD_LOCAL_SOLVE_DOCTRINE.anchor_rule == ("artifact_calibrated_weights")
    assert US_SLD_LOCAL_SOLVE_DOCTRINE.as_record() == {
        "target_loss_cap": 10.0,
        "max_weight_ratio": 100.0,
        "scale_rule": "default_target_loss_scales",
        "target_weight_rule": "uniform",
        "anchor_rule": "artifact_calibrated_weights",
        "min_initial_weight": 1e-4,
    }


def test_doctrine_refuses_tampered_bounds_and_unknown_rules():
    with pytest.raises(ValueError, match="target_loss_cap"):
        UsSldLocalSolveDoctrine(target_loss_cap=0.0)
    with pytest.raises(ValueError, match="max_weight_ratio"):
        UsSldLocalSolveDoctrine(max_weight_ratio=1.0)
    with pytest.raises(ValueError, match="scale_rule"):
        UsSldLocalSolveDoctrine(scale_rule="handcrafted")
    with pytest.raises(ValueError, match="target_weight_rule"):
        UsSldLocalSolveDoctrine(target_weight_rule="per_target")
    with pytest.raises(ValueError, match="anchor_rule"):
        UsSldLocalSolveDoctrine(anchor_rule="design_weights")


def test_doctrine_solve_signatures_are_structurally_knob_free():
    for function in (
        solve_us_sld_district_weights_under_doctrine,
        solve_us_sld_chamber_under_doctrine,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {
            "max_weight_ratio",
            "target_loss_cap",
            "target_loss_weights",
            "target_loss_scales",
            "min_initial_weight",
            "doctrine",
        }, function.__name__


def test_doctrine_solve_refuses_duplicate_metric_rows():
    problem = _problem()
    duplicated = SldDistrictProblem(
        area_type=problem.area_type,
        area_code=problem.area_code,
        state_fips=problem.state_fips,
        district_code=problem.district_code,
        matrix=problem.matrix,
        targets=problem.targets,
        target_frame=pd.DataFrame(
            {
                "area_type": ["sldu"] * 3,
                "area_code": [problem.area_code] * 3,
                "metric": ["households", "households", "income_under_10000"],
            }
        ),
        household_ids=problem.household_ids,
        base_weights=problem.base_weights,
    )
    with pytest.raises(ValueError, match="non-uniform target surface"):
        solve_us_sld_district_weights_under_doctrine(duplicated, epochs=8)
    with pytest.raises(ValueError, match="non-uniform target surface"):
        solve_us_sld_chamber_under_doctrine([duplicated], epochs=8)


def test_doctrine_chamber_solve_runs_clean():
    result = solve_us_sld_chamber_under_doctrine([_problem()], epochs=64)
    assert result.area_type == "sldu"
    assert len(result.district_results) == 1
