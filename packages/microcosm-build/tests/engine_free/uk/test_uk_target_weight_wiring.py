"""Row-metadata weighting rules reach the solve, the holdout and the vocabulary mirrors."""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    build_uk_rowwise_local_matrix,
    rotated_uk_local_holdout,
    uk_national_frame,
)
from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_ROW_METADATA_RULES,
    UK_LOCAL_TARGET_WEIGHT_RULES,
    uk_local_target_loss_weights,
)
from microcosm.build.uk_runtime.local_rowwise import prepare_uk_full_solve
from microcosm.build.uk_runtime.rowwise_posture import UK_ROWWISE_DENSE_POSTURE
from microcosm.build.uk_runtime.target_weights import (
    uk_target_loss_weights_for_rows,
)
from microcosm.diagnostics.schema import UKMeasuredRotatedHoldout
from microcosm.frame import WeightKind

_IDS = [101, 102, 103, 104, 105]
_AREAS = ["E001", "W001", "S001", "N001", "E002"]


def _frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "person_household_id": _IDS,
                "person_benunit_id": [11, 12, 13, 14, 15],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13, 14, 15]}),
        household=pd.DataFrame({"household_id": _IDS, "household_weight": [1.0] * 5}),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )


def _problem():
    metrics = pd.DataFrame(
        {"households": np.ones(5), "tenure/social_rent": np.ones(5)},
        index=_IDS,
    )
    targets = pd.DataFrame(
        {"code": _AREAS, "households": [2.0] * 5, "tenure/social_rent": [1.0] * 5}
    )
    return build_uk_rowwise_local_matrix(
        metrics, pd.Series(_AREAS, index=_IDS), targets
    )


_FAMILIES = ["census_households/constituency", "tenure/constituency"]


def test_prepared_solve_binds_the_rule_computed_from_its_own_rows() -> None:
    problem = _problem()
    legacy = prepare_uk_full_solve(
        _frame(), problem, bound_families=_FAMILIES, target_weight_rule="grain_equal"
    )
    assert legacy.target_rows is None
    np.testing.assert_array_equal(
        legacy.target_loss_weights,
        uk_local_target_loss_weights(
            problem.target_frame["area_type"].astype(str).tolist(), rule="grain_equal"
        ),
    )
    for rule in UK_LOCAL_ROW_METADATA_RULES:
        prepared = prepare_uk_full_solve(
            _frame(), problem, bound_families=_FAMILIES, target_weight_rule=rule
        )
        rows = prepared.target_rows
        assert rows is not None
        assert rows.names == tuple(
            target.row_name for target in prepared.target_set.targets
        )
        assert set(rows.family) == {"census_households", "tenure"}
        np.testing.assert_array_equal(
            prepared.target_loss_weights,
            uk_target_loss_weights_for_rows(rows, rule=rule),
        )
    # One grain, two families: grain_family_equal halves the loss by family,
    # where grain_equal is uniform over the ten rows.
    family = prepare_uk_full_solve(
        _frame(),
        problem,
        bound_families=_FAMILIES,
        target_weight_rule="grain_family_equal",
    )
    metrics = problem.target_frame["metric"].to_numpy()
    assert family.target_loss_weights[metrics == "households"].sum() == pytest.approx(
        0.5
    )


def test_holdout_weights_held_rows_through_the_carrier_and_is_schema_valid() -> None:
    payload = rotated_uk_local_holdout(
        _frame(),
        _problem(),
        target_weight_rule="grain_family_equal",
        epochs=2,
        solve_seed=17,
    )
    assert payload["target_weight_rule"] == "grain_family_equal"
    assert payload["n_folds"] == 5
    assert np.isfinite(payload["mean_holdout_loss"])
    UKMeasuredRotatedHoldout.model_validate(payload)


def test_diagnostics_schema_admits_exactly_the_local_vocabulary() -> None:
    annotation = UKMeasuredRotatedHoldout.model_fields["target_weight_rule"].annotation
    assert typing.get_args(annotation) == UK_LOCAL_TARGET_WEIGHT_RULES


def test_dense_posture_allows_the_row_rules_and_keeps_its_default() -> None:
    assert UK_ROWWISE_DENSE_POSTURE.target_weight_rule == "grain_equal"
    assert UK_ROWWISE_DENSE_POSTURE.allowed_target_weight_rules == (
        "grain_equal",
        "uniform",
        *UK_LOCAL_ROW_METADATA_RULES,
    )
