from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.calibrate import Target, TargetSet, relative_error_loss, score_targets
from populace.frame import EntitySchema, Frame, WeightKind, Weights


def _frame() -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "person_household_id": [10, 20],
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": [10, 20],
                    "income": [100.0, 200.0],
                    "household_count": [1.0, 1.0],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights([2.0, 3.0], WeightKind.CALIBRATED)},
    )


def test_score_targets_evaluates_existing_weights_without_calibrating() -> None:
    frame = _frame()
    targets = TargetSet(
        (
            Target(name="income", entity="household", value=1_000.0, measure="income"),
            Target(
                name="households",
                entity="household",
                value=5.0,
                measure="household_count",
            ),
        )
    )

    result = score_targets(frame, targets, target_loss_cap=1.0)

    np.testing.assert_allclose(result.weights, [2.0, 3.0])
    np.testing.assert_allclose(result.initial_weights, [2.0, 3.0])
    assert result.options["method"] == "score_only"
    assert result.gate_open_probabilities is None
    assert result.n_nonzero == 2
    assert [diagnostic.name for diagnostic in result.diagnostics] == [
        "income@0",
        "households@0",
    ]
    assert result.diagnostics[0].initial_estimate == 800.0
    assert result.diagnostics[0].final_estimate == 800.0
    assert result.diagnostics[1].final_estimate == 5.0
    expected = relative_error_loss(
        np.asarray([800.0, 5.0]),
        np.asarray([1_000.0, 5.0]),
        target_loss_cap=1.0,
    )
    assert result.initial_loss == pytest.approx(expected)
    assert result.final_loss == pytest.approx(expected)


def test_score_targets_accepts_explicit_weight_vector() -> None:
    frame = _frame()
    targets = TargetSet(
        (Target(name="income", entity="household", value=500.0, measure="income"),)
    )

    result = score_targets(frame, targets, weights=np.asarray([1.0, 2.0]))

    np.testing.assert_allclose(result.initial_weights, [2.0, 3.0])
    np.testing.assert_allclose(result.weights, [1.0, 2.0])
    assert result.diagnostics[0].final_estimate == 500.0
    assert result.final_loss == pytest.approx(0.0)


def test_score_targets_rejects_misaligned_weight_vector() -> None:
    frame = _frame()
    targets = TargetSet(
        (Target(name="income", entity="household", value=500.0, measure="income"),)
    )

    with pytest.raises(ValueError, match="weights must align"):
        score_targets(frame, targets, weights=np.asarray([1.0]))
