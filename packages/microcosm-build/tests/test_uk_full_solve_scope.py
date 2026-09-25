"""A filtered full problem has zero local rows and uses the same solver."""

import numpy as np
import pytest
from test_uk_local_rowwise import _clone_frame

from microcosm.build.uk_runtime.local_rowwise import (
    UKRowwiseNationalRows,
    empty_uk_local_problem,
    finish_uk_full_solve,
    prepare_uk_full_solve,
    rotated_uk_local_holdout,
    solve_uk_dense_reference,
    solve_uk_rowwise_weights_under_doctrine,
)
from microcosm.calibrate import Target, TargetRegistry, TargetSet, TargetSpec


def national_rows():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="households",
                entity="household",
                value=6.0,
                measure="household_id",
                period=2026,
                family="fixture",
                source="fixture",
            )
        ],
        country="uk",
    )
    return UKRowwiseNationalRows(
        TargetSet(
            [
                Target(
                    name="households",
                    entity="household",
                    value=6.0,
                    measure=lambda frame: np.ones(frame.n("household")),
                    period=2026,
                )
            ]
        ),
        registry,
        ("fixture",),
    )


def test_zero_local_scope_uses_same_solver_and_has_no_fake_holdout():
    frame = _clone_frame()
    local = empty_uk_local_problem(frame.table("household")["household_id"])
    rows = national_rows()
    prepared = prepare_uk_full_solve(
        frame, local, bound_families=("national/fixture",), national_rows=rows
    )
    dense = solve_uk_dense_reference(prepared, epochs=8, seed=17)
    finished = finish_uk_full_solve(prepared, dense)
    existing = solve_uk_rowwise_weights_under_doctrine(
        frame,
        local,
        bound_families=("national/fixture",),
        national_rows=rows,
        epochs=8,
        seed=17,
    )
    np.testing.assert_array_equal(finished.weights, existing.weights)
    assert finished.diagnostics.empty
    assert len(finished.national_diagnostics) == 1
    assert dense.problem.n_targets == 1
    holdout = rotated_uk_local_holdout(frame, local, national_rows=rows)
    assert holdout["outcome"] == "not_applicable"
    assert holdout["n_folds"] == 0
    assert holdout["folds"] == []


def test_empty_total_surface_refuses_instead_of_manufacturing_constraints():
    frame = _clone_frame()
    local = empty_uk_local_problem(frame.table("household")["household_id"])
    with pytest.raises(ValueError, match="at least one selected target"):
        prepare_uk_full_solve(frame, local, bound_families=())
