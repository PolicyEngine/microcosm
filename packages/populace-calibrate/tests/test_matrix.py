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
    np.testing.assert_allclose(
        dense[1], frame.table("household")["income"].to_numpy()
    )
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


def test_target_value_near_minus_one_is_rejected_with_a_named_error(
    feasible_frame,
) -> None:
    """A compiled RHS of exactly -1 zeroes the loss denominator ``(b + 1)``.

    The eCPS loss divides each residual by ``(target + 1)``. A target value of
    exactly -1 makes that denominator zero, so the loss is NaN, the gradients are
    NaN, every weight goes NaN, and the kernel raises an opaque "Weights must be
    finite" error naming neither the target nor the cause. The compiler must
    reject the offending target up front, naming it.
    """
    frame, _ = feasible_frame(n=50)
    targets = TargetSet(
        (
            Target(name="counts_to_neg_one", entity="household",
                   aggregation="count", value=-1.0),
        )
    )
    with pytest.raises(ValueError, match="counts_to_neg_one"):
        build_constraint_matrix(frame, targets)


def test_mean_target_one_below_current_mean_is_rejected(feasible_frame) -> None:
    """A ``mean`` whose compiled RHS lands at -1 is rejected too.

    A ``mean`` target compiles to a right-hand side of ``value - current_mean``;
    a value exactly 1 below the current mean lands the RHS at -1 — the same
    zero-denominator landmine — and must be named, not silently NaN'd.
    """
    frame, _ = feasible_frame(n=50)
    current_mean = float(frame.table("household")["income"].to_numpy().mean())
    targets = TargetSet(
        (
            Target(name="mean_one_below", entity="household",
                   aggregation="mean", value=current_mean - 1.0,
                   measure="income"),
        )
    )
    with pytest.raises(ValueError, match="mean_one_below"):
        build_constraint_matrix(frame, targets)


def test_normal_target_value_compiles_fine(feasible_frame) -> None:
    """A target whose compiled RHS is far from -1 compiles without complaint."""
    frame, truths = feasible_frame(n=50)
    targets = TargetSet(
        (
            Target(name="population", entity="household",
                   aggregation="count", value=truths["population"]),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.matrix.shape[0] == 1


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
