"""Constraint-matrix compilation: shape, multi-period rows, skipped targets."""

from __future__ import annotations

import numpy as np
import pytest

from populace.calibrate import Target, TargetSet, build_constraint_matrix


def test_matrix_has_one_row_per_target_one_column_per_weight(feasible_frame) -> None:
    frame, truths = feasible_frame(n=150)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                aggregation="count",
                value=truths["population"],
            ),
            Target(
                name="income",
                entity="household",
                aggregation="sum",
                value=truths["income"],
                measure="income",
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.matrix.shape == (2, 150)
    assert len(problem.names) == 2
    # The count row is the all-ones indicator; the income row is per-record income.
    dense = problem.matrix.toarray()
    np.testing.assert_allclose(dense[0], np.ones(150))
    np.testing.assert_allclose(dense[1], frame.table("household")["income"].to_numpy())
    # b is the target vector, aligned to rows.
    np.testing.assert_allclose(
        problem.target_vector,
        [truths["population"], truths["income"]],
    )


def test_estimates_recover_the_weighted_aggregates(feasible_frame) -> None:
    frame, truths = feasible_frame(n=150)
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                aggregation="sum",
                value=truths["income"],
                measure="income",
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    w = frame.resolve_weights("household").values
    est = problem.estimates(w)
    np.testing.assert_allclose(est[0], truths["income"], rtol=1e-9)


def test_multi_period_targets_become_distinct_rows(multiperiod_frame) -> None:
    frame, t26, t30 = multiperiod_frame(n=120)
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                aggregation="sum",
                value=t26,
                measure="income_2026",
                period=2026,
            ),
            Target(
                name="income",
                entity="household",
                aggregation="sum",
                value=t30,
                measure="income_2030",
                period=2030,
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.matrix.shape == (2, 120)  # one weight vector, two period rows
    assert any("2026" in n for n in problem.names)
    assert any("2030" in n for n in problem.names)


def test_target_value_near_minus_one_compiles_under_scaled_mape(
    feasible_frame,
) -> None:
    """A target at -1 is no longer special: scaled MAPE has no ``b + 1`` denominator."""
    frame, _ = feasible_frame(n=50)
    targets = TargetSet(
        (
            Target(
                name="counts_to_neg_one",
                entity="household",
                aggregation="count",
                value=-1.0,
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.names == ("counts_to_neg_one@0",)
    assert problem.target_vector.tolist() == [-1.0]


def test_mean_target_one_below_current_mean_compiles(feasible_frame) -> None:
    """Mean rows whose compiled RHS lands at -1 are valid under scaled MAPE."""
    frame, _ = feasible_frame(n=50)
    current_mean = float(frame.table("household")["income"].to_numpy().mean())
    targets = TargetSet(
        (
            Target(
                name="mean_one_below",
                entity="household",
                aggregation="mean",
                value=current_mean - 1.0,
                measure="income",
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.names == ("mean_one_below@0",)
    assert problem.target_vector.tolist() == pytest.approx([-1.0])


def test_normal_target_value_compiles_fine(feasible_frame) -> None:
    """A target whose compiled RHS is far from -1 compiles without complaint."""
    frame, truths = feasible_frame(n=50)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                aggregation="count",
                value=truths["population"],
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.matrix.shape[0] == 1


def test_person_sum_target_collapses_onto_multi_person_households(
    multiperson_frame,
) -> None:
    """A person-level ``sum`` compiles onto household weights via group collapse.

    On a real multi-person frame the per-person values sum within each household
    (members share the household weight), so the household row's entry is the
    household's person-total, and the weighted row reproduces the true aggregate.
    """
    frame, age, person_household, weights = multiperson_frame()
    targets = TargetSet(
        (
            Target(
                name="total_age",
                entity="person",
                aggregation="sum",
                value=1.0,
                measure="age",
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets, "household")
    assert problem.matrix.shape[0] == 1
    assert problem.skipped == ()
    # The household row equals each household's summed person ages.
    expected_row = np.zeros(frame.n("household"))
    np.add.at(expected_row, person_household, age)
    np.testing.assert_allclose(problem.matrix.toarray()[0], expected_row)
    # The weighted estimate recovers the true weighted person-age total.
    true_total = float((age * weights[person_household]).sum())
    np.testing.assert_allclose(problem.estimates(weights)[0], true_total, rtol=1e-9)


def test_person_mean_target_compiles_on_multi_person_households(
    multiperson_frame,
) -> None:
    """A person-level ``mean`` compiles on a multi-person frame (Finding 5).

    The compiler builds the row at the person-aligned linearization point but
    previously called ``Target.offset`` with the *group* weight vector, so its
    internal ``filter_mask * weights`` broadcast (n_persons,) against
    (n_households,) and raised — the target was silently skipped with a raw-numpy
    reason. ``offset`` must see the same person-aligned weights the row does.
    """
    frame, age, person_household, weights = multiperson_frame()
    person_weights = weights[person_household]
    true_mean = float((age * person_weights).sum() / person_weights.sum())
    targets = TargetSet(
        (
            Target(
                name="mean_age",
                entity="person",
                aggregation="mean",
                value=true_mean * 1.1,
                measure="age",
            ),
        )
    )
    value = true_mean * 1.1
    problem = build_constraint_matrix(frame, targets, "household")
    assert problem.skipped == (), problem.skipped
    assert problem.matrix.shape[0] == 1
    # By the mean linearization, target_vector = value + (row @ w0 - current_mean)
    # and estimates(w0) = row @ w0, so estimates(w0) - target_vector + value
    # recovers the true current mean exactly — and the row solving to value would
    # land the mean on the target. This identity holds only if offset saw the
    # same person-aligned weights as the row (the Finding 5 fix).
    est_at_w0 = float(problem.estimates(weights)[0])
    recovered_mean = est_at_w0 - float(problem.target_vector[0]) + value
    np.testing.assert_allclose(recovered_mean, true_mean, rtol=1e-6)


def test_uncompilable_target_is_skipped_and_reported(feasible_frame) -> None:
    frame, truths = feasible_frame(n=100)
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                aggregation="sum",
                value=truths["income"],
                measure="income",
            ),
            Target(
                name="missing",
                entity="household",
                aggregation="sum",
                value=1.0,
                measure="not_a_column",
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.matrix.shape[0] == 1  # only the compilable target
    assert len(problem.skipped) == 1
    assert problem.skipped[0].target.name == "missing"
    assert "not_a_column" in problem.skipped[0].reason
