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

from microcosm.build.uk_runtime import (
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_TARGET_LOSS_CAP,
    build_uk_rowwise_local_matrix,
    rowwise_area_support_summary,
    rowwise_calibration_mass_reason,
    solve_uk_rowwise_weights_under_doctrine,
    uk_household_weight_kind,
    uk_national_frame,
)
from microcosm.frame import WeightKind


def _clone_frame(weights=(1.0, 1.0, 1.0)):
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 103],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "household_weight": list(weights),
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
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
    frame = _clone_frame(base)
    result = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        bound_families=["census_households/constituency"],
        epochs=60,
        learning_rate=0.2,
        seed=1,
    )
    # Rowwise initial weights ARE the frame's typed base weights, never
    # split across areas.
    np.testing.assert_allclose(result.initial_weights, base)
    assert result.weights.shape == (3,)
    assert np.isfinite(result.final_loss)
    assert result.past_cap_census is not None
    assert result.past_cap_census["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    stretched = result.weights / result.initial_weights
    assert float(np.max(stretched)) <= UK_LOCAL_MAX_WEIGHT_RATIO * (1 + 1e-6)

    # The declarative target expression and the hand-assembled sparse matrix
    # derive from the same numbers: the compiled initial estimates equal the
    # COO assembly's matvec row for row.
    np.testing.assert_allclose(
        result.diagnostics["initial_estimate"].to_numpy(dtype=np.float64),
        problem.matrix @ np.asarray(base, dtype=np.float64),
    )

    # The kernel product: CALIBRATED typed weights, the calibration mass
    # record naming the bound family, and the refreshed persisted column.
    assert uk_household_weight_kind(result.frame) is WeightKind.CALIBRATED
    record = result.frame.mass_log[-1]
    assert "census_households/constituency" in record.reason
    assert record.old_total == pytest.approx(float(np.sum(base)))
    assert record.new_total == pytest.approx(float(np.sum(result.weights)))
    np.testing.assert_allclose(
        result.frame.weights_for("household").values,
        result.weights,
    )


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
            _clone_frame(),
            problem,
            bound_families=["census_households/constituency"],
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
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame(),
            doctored,
            bound_families=["census_households/constituency"],
            epochs=1,
        )


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


def test_rowwise_solve_refuses_dead_rows_and_misaligned_frames() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="zero"):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame([1.0, 0.0, 1.0]),
            problem,
            bound_families=["census_households/constituency"],
            epochs=1,
        )

    # The same households in a different order are refused with the ordering
    # named, not a useless equal-counts message (and never realigned).
    reordered = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 103],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "household_weight": [1.0, 1.0, 1.0],
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )
    reordered_problem = build_uk_rowwise_local_matrix(
        _metrics().reindex([102, 101, 103]),
        _assigned().reindex([102, 101, 103]),
        _targets(),
    )
    with pytest.raises(ValueError, match="different order.*row 0.*101.*102"):
        solve_uk_rowwise_weights_under_doctrine(
            reordered,
            reordered_problem,
            bound_families=["census_households/constituency"],
            epochs=1,
        )

    # A frame whose household rows do not match the problem's households
    # cannot express the declared surface and is refused, not realigned.
    misaligned = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_household_id": [999],
                "person_benunit_id": [11],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11]}),
        household=pd.DataFrame({"household_id": [999], "household_weight": [1.0]}),
        time_period="2023",
    )
    with pytest.raises(ValueError, match="match the problem"):
        solve_uk_rowwise_weights_under_doctrine(
            misaligned,
            problem,
            bound_families=["census_households/constituency"],
            epochs=1,
        )


def test_calibration_mass_reason_names_families() -> None:
    reason = rowwise_calibration_mass_reason(["census_households/constituency"])
    assert "census_households/constituency" in reason
    assert "calibration" in reason

    with pytest.raises(ValueError, match="bound_families"):
        rowwise_calibration_mass_reason([])


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


def test_doctrine_solve_forwards_declared_bounds_to_the_front_door(
    monkeypatch,
) -> None:
    """The doctrine's reviewed constants ride into calibrate() explicitly.

    The pre-migration solver defaults (epochs 512, learning rate 0.15) and
    the doctrine bounds (ratio 100.0, cap 10.0) differ from calibrate()'s
    own defaults, so silent default-drift would change solve behaviour;
    this pin fails if any of them stops being forwarded.
    """

    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    forwarded: dict[str, object] = {}
    real_calibrate = local_rowwise.calibrate

    def spy(frame, targets, **kwargs):
        forwarded.update(kwargs)
        return real_calibrate(frame, targets, **kwargs)

    monkeypatch.setattr(local_rowwise, "calibrate", spy)
    solve_uk_rowwise_weights_under_doctrine(
        _clone_frame(),
        problem,
        bound_families=["census_households/constituency"],
        seed=3,
    )

    assert forwarded["epochs"] == 512
    assert forwarded["learning_rate"] == 0.15
    assert forwarded["max_weight_ratio"] == UK_LOCAL_MAX_WEIGHT_RATIO
    assert forwarded["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    assert forwarded["mass"] == "free"
    assert forwarded["seed"] == 3
    assert "census_households/constituency" in forwarded["mass_reason"]


def test_doctrine_solve_refuses_reordered_diagnostics(monkeypatch) -> None:
    """A front-door result whose diagnostics are reordered is refused by name.

    The evidence tables consume diagnostics positionally, and target values
    legitimately repeat on a local surface, so value equality alone could
    pass a reordering by coincidence; the solve asserts per-row name
    alignment against the declared surface instead.
    """

    import dataclasses

    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    real_calibrate = local_rowwise.calibrate

    def reordering(frame, targets, **kwargs):
        result = real_calibrate(frame, targets, **kwargs)
        return dataclasses.replace(
            result, diagnostics=tuple(reversed(result.diagnostics))
        )

    monkeypatch.setattr(local_rowwise, "calibrate", reordering)
    with pytest.raises(ValueError, match="not aligned.*row 0"):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame(),
            problem,
            bound_families=["census_households/constituency"],
            epochs=1,
        )
