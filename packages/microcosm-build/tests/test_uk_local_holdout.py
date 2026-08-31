"""The UK local rotated holdout runs five real training solves."""

import numpy as np
import pandas as pd

from microcosm.build.holdout import rotated_folds
from microcosm.build.uk_runtime import (
    build_uk_rowwise_local_matrix,
    rotated_uk_local_holdout,
    uk_national_frame,
)
from microcosm.frame import WeightKind


def test_rotated_local_holdout_runs_fixed_five_fold_actual_solves() -> None:
    household_ids = [101, 102, 103, 104, 105]
    area_codes = ["E001", "W001", "S001", "N001", "E002"]
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "person_household_id": household_ids,
                "person_benunit_id": [11, 12, 13, 14, 15],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13, 14, 15]}),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": [1.0] * 5,
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )
    problem = build_uk_rowwise_local_matrix(
        pd.DataFrame({"households": np.ones(5)}, index=household_ids),
        pd.Series(area_codes, index=household_ids),
        pd.DataFrame({"code": area_codes, "households": [2.0] * 5}),
    )

    payload = rotated_uk_local_holdout(
        frame,
        problem,
        epochs=2,
        solve_seed=17,
    )

    expected_folds = rotated_folds(5, n_folds=5, seed=20260529)
    assert payload["report_only"] is True
    assert payload["method"] == "rotated_folds"
    assert payload["n_folds"] == 5
    assert payload["seed"] == 20260529
    assert payload["solve_seed"] == 17
    assert [row["holdout_target_indices"] for row in payload["folds"]] == [
        fold.tolist() for fold in expected_folds
    ]
    assert all(row["n_train_targets"] == 4 for row in payload["folds"])
    assert all(row["n_holdout_targets"] == 1 for row in payload["folds"])
    assert payload["fold_losses"] == [0.5] * 5
    assert payload["mean_holdout_loss"] == 0.5
    assert payload["worst_holdout_loss"] == 0.5
