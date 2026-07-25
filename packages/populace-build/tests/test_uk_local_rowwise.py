"""Rowwise local solve surface (#495 increment 6b).

The US Build-N shape for the UK: one weight vector over the cloned rowwise
households, where each household supports only its assigned constituency's
target rows. The matrix builder fails closed on an assigned area the target
surface does not cover (the 650/650 requirement), and the solve runs under
the #503 doctrine — no per-target knobs, declared bounds, past-cap census on
every result, and initial weights that are the household base weights
directly (never split across areas: a rowwise household exists in exactly
one area).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_TARGET_LOSS_CAP,
    build_uk_rowwise_local_matrix,
    rowwise_area_support_summary,
    solve_uk_rowwise_weights_under_doctrine,
)


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": [2.0, 1.0, 3.0],
            "uc_households": [1.0, 0.0, 1.0],
        },
        index=[101, 102, 103],
    )


def _assigned() -> pd.Series:
    return pd.Series(["E001", "E001", "S001"], index=[101, 102, 103])


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["E001", "S001"],
            "population": [4.0, 2.0],
            "uc_households": [1.0, 1.0],
        }
    )


def test_matrix_builder_places_support_only_in_assigned_area() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    assert problem.matrix.shape == (4, 3)
    assert problem.area_codes == ("E001", "S001")
    assert problem.metric_names == ("population", "uc_households")
    assert problem.household_ids == (101, 102, 103)

    dense = problem.matrix.toarray()
    frame = problem.target_frame
    # E001 rows carry only households 101/102; S001 rows only household 103.
    e_pop = int(
        frame[(frame["area_code"] == "E001") & (frame["metric"] == "population")][
            "target_index"
        ].iloc[0]
    )
    assert dense[e_pop].tolist() == [2.0, 1.0, 0.0]
    s_pop = int(
        frame[(frame["area_code"] == "S001") & (frame["metric"] == "population")][
            "target_index"
        ].iloc[0]
    )
    assert dense[s_pop].tolist() == [0.0, 0.0, 3.0]
    np.testing.assert_allclose(problem.targets, [4.0, 1.0, 2.0, 1.0])
    assert frame["area_type"].unique().tolist() == ["constituency"]


def test_matrix_builder_fails_closed_on_uncovered_assigned_area() -> None:
    assigned = pd.Series(["E001", "E001", "X999"], index=[101, 102, 103])
    with pytest.raises(ValueError, match="X999"):
        build_uk_rowwise_local_matrix(_metrics(), assigned, _targets())


def test_matrix_builder_validates_alignment_and_finiteness() -> None:
    misaligned = pd.Series(["E001", "S001"], index=[101, 999])
    with pytest.raises(ValueError, match="align"):
        build_uk_rowwise_local_matrix(_metrics(), misaligned, _targets())

    bad = _metrics()
    bad.loc[101, "population"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_uk_rowwise_local_matrix(bad, _assigned(), _targets())


def test_rowwise_doctrine_solve_uses_base_weights_directly() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    base = [1.0, 1.0, 1.0]
    result = solve_uk_rowwise_weights_under_doctrine(
        problem,
        base,
        epochs=60,
        learning_rate=0.2,
        seed=1,
    )
    # Rowwise initial weights ARE the base weights (floored), never split
    # across areas.
    np.testing.assert_allclose(result.initial_weights, base)
    assert result.weights.shape == (3,)
    assert np.isfinite(result.final_loss)
    assert result.past_cap_census is not None
    assert result.past_cap_census["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    stretched = result.weights / result.initial_weights
    assert float(np.max(stretched)) <= UK_LOCAL_MAX_WEIGHT_RATIO * (1 + 1e-6)


def test_rowwise_doctrine_solve_exposes_no_knobs() -> None:
    import inspect

    parameters = inspect.signature(solve_uk_rowwise_weights_under_doctrine).parameters
    for forbidden in (
        "target_loss_weights",
        "target_loss_scales",
        "target_loss_cap",
        "max_weight_ratio",
        "doctrine",
    ):
        assert forbidden not in parameters

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(TypeError):
        solve_uk_rowwise_weights_under_doctrine(
            problem,
            [1.0, 1.0, 1.0],
            target_loss_weights=[1.0] * 4,
        )


def test_rowwise_doctrine_solve_refuses_duplicate_surface() -> None:
    import dataclasses

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    doctored = dataclasses.replace(
        problem,
        target_frame=pd.concat(
            [problem.target_frame, problem.target_frame.iloc[[0]]],
            ignore_index=True,
        ),
    )
    with pytest.raises(ValueError, match="per-target weights"):
        solve_uk_rowwise_weights_under_doctrine(doctored, [1.0, 1.0, 1.0], epochs=1)


def test_rowwise_area_support_summary_reports_all_target_areas() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    weights = np.array([2.0, 0.0, 5.0])
    support = rowwise_area_support_summary(
        problem,
        weights,
        source_household_ids=[1, 1, 2],
    )
    rows = {row.area_code: row for row in support.itertuples(index=False)}
    assert rows["E001"].nonzero_households == 1
    assert rows["E001"].nonzero_source_households == 1
    assert rows["E001"].weight_sum == pytest.approx(2.0)
    assert rows["S001"].nonzero_households == 1
    assert rows["S001"].weight_sum == pytest.approx(5.0)
    assert rows["E001"].effective_sample_size == pytest.approx(1.0)


def test_solver_refactor_preserves_stacked_behaviour() -> None:
    # Regression guard for the shared-core refactor: the stacked path still
    # splits base weights across areas and solves identically shaped output.
    from populace.build.uk_runtime import (
        build_stacked_local_matrix,
        solve_stacked_local_weights,
    )

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
        epochs=40,
        learning_rate=0.2,
        max_weight_ratio=10.0,
        seed=1,
    )
    assert result.weights.shape == (4,)
    assert result.initial_weights.tolist() == [0.5, 0.5, 0.5, 0.5]
    assert result.past_cap_census is not None


def test_matrix_builder_fails_closed_on_unreachable_nonzero_targets() -> None:
    # A target area with no assigned households cannot be hit; a nonzero
    # target there must refuse at build time, while a zero target is fine.
    targets = pd.DataFrame(
        {
            "code": ["E001", "S001", "W001"],
            "population": [4.0, 2.0, 5.0],
            "uc_households": [1.0, 1.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="W001/population"):
        build_uk_rowwise_local_matrix(_metrics(), _assigned(), targets)

    zero_ok = targets.copy()
    zero_ok.loc[zero_ok["code"] == "W001", "population"] = 0.0
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), zero_ok)
    assert problem.n_areas == 3


def test_matrix_builder_refuses_duplicate_and_metadata_metric_labels() -> None:
    duplicated = _metrics()
    duplicated.columns = ["population", "population"]
    with pytest.raises(ValueError, match="duplicate column label"):
        build_uk_rowwise_local_matrix(duplicated, _assigned(), _targets())

    metadata = _metrics().rename(columns={"uc_households": "area_index"})
    targets = _targets().rename(columns={"uc_households": "area_index"})
    with pytest.raises(ValueError, match="metadata"):
        build_uk_rowwise_local_matrix(metadata, _assigned(), targets)


def test_rowwise_solve_refuses_dead_rows_and_bad_core_inputs() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="zero"):
        solve_uk_rowwise_weights_under_doctrine(problem, [1.0, 0.0, 1.0], epochs=1)

    from populace.build.uk_runtime import solve_prepared_local_weights

    with pytest.raises(ValueError, match="finite"):
        solve_prepared_local_weights(
            matrix=problem.matrix,
            targets=problem.targets,
            target_frame=problem.target_frame,
            initial_weights=np.array([1.0, np.nan, 1.0]),
            epochs=1,
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        solve_prepared_local_weights(
            matrix=problem.matrix,
            targets=problem.targets,
            target_frame=problem.target_frame,
            initial_weights=np.ones((3, 1)),
            epochs=1,
        )


def test_calibration_mass_record_names_families_and_totals() -> None:
    from populace.build.uk_runtime import rowwise_calibration_mass_record

    record = rowwise_calibration_mass_record(
        [1.0, 2.0],
        [1.5, 2.5],
        bound_families=["census_households/constituency"],
    )
    assert record.entity == "household"
    assert record.old_total == pytest.approx(3.0)
    assert record.new_total == pytest.approx(4.0)
    assert "census_households/constituency" in record.reason

    with pytest.raises(ValueError, match="bound_families"):
        rowwise_calibration_mass_record([1.0], [1.0], bound_families=[])


def test_support_summary_normalizes_inputs() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="one-dimensional"):
        rowwise_area_support_summary(problem, np.ones((3, 1)))
    # Composite (tuple) source ids must be handled, not coerced into 2-D.
    support = rowwise_area_support_summary(
        problem,
        [1.0, 1.0, 1.0],
        source_household_ids=[(2023, 1), (2023, 1), (2023, 2)],
    )
    rows = {row.area_code: row for row in support.itertuples(index=False)}
    assert rows["E001"].nonzero_source_households == 1
