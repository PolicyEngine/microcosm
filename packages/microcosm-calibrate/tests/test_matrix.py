"""Constraint-matrix compilation: shape, multi-period rows, skipped targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def test_matrix_has_one_row_per_target_one_column_per_weight(feasible_frame) -> None:
    frame, truths = feasible_frame(n=150)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=truths["population"],
                measure="household_count",
            ),
            Target(
                name="income",
                entity="household",
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


def test_matrix_compiler_does_not_dense_stack_target_rows(
    feasible_frame, monkeypatch
) -> None:
    frame, truths = feasible_frame(n=150)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=truths["population"],
                measure="household_count",
            ),
            Target(
                name="income",
                entity="household",
                value=truths["income"],
                measure="income",
            ),
        )
    )

    def fail_vstack(*args, **kwargs):
        raise AssertionError("matrix compiler should not dense-stack rows")

    monkeypatch.setattr(np, "vstack", fail_vstack)
    problem = build_constraint_matrix(frame, targets)

    assert problem.matrix.shape == (2, 150)
    np.testing.assert_allclose(
        problem.estimates(frame.resolve_weights("household").values),
        [
            truths["population"],
            truths["income"],
        ],
    )


def test_estimates_recover_the_weighted_aggregates(feasible_frame) -> None:
    frame, truths = feasible_frame(n=150)
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
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
                value=t26,
                measure="income_2026",
                period=2026,
            ),
            Target(
                name="income",
                entity="household",
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
                value=-1.0,
                measure="household_count",
            ),
        )
    )
    problem = build_constraint_matrix(frame, targets)
    assert problem.names == ("counts_to_neg_one@0",)
    assert problem.target_vector.tolist() == [-1.0]


def test_normal_target_value_compiles_fine(feasible_frame) -> None:
    """A target whose compiled RHS is far from -1 compiles without complaint."""
    frame, truths = feasible_frame(n=50)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=truths["population"],
                measure="household_count",
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


def test_group_sum_target_collapses_onto_weighted_parent_group() -> None:
    """A benunit-grain row compiles onto household weights on a real nested frame."""

    weights = np.array([10.0, 20.0, 30.0])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(6, dtype="int64"),
                    "person_benunit_id": [0, 0, 1, 2, 3, 3],
                    "person_household_id": [0, 0, 0, 1, 2, 2],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(4, dtype="int64"),
                    "uc_receipt": [1.0, 0.0, 1.0, 1.0],
                }
            ),
            "household": pd.DataFrame(
                {"household_id": np.arange(3, dtype="int64")}
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="uc_benunits",
                entity="benunit",
                value=60.0,
                measure="uc_receipt",
                period=2025,
            ),
        )
    )

    problem = build_constraint_matrix(frame, targets, "household")

    assert problem.skipped == ()
    np.testing.assert_allclose(problem.matrix.toarray()[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(problem.estimates(weights), [60.0])


def test_uncompilable_target_is_skipped_and_reported(feasible_frame) -> None:
    frame, truths = feasible_frame(n=100)
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=truths["income"],
                measure="income",
            ),
            Target(
                name="missing",
                entity="household",
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
