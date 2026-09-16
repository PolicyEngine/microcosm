"""Synthetic contracts for the experimental canonical-QRF schedule decoder."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import nsece_childcare_qrf as module
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.nsece_childcare_pooling import (
    with_childcare_household_size,
)

MONTH, DAYS, HOURS = US_CHILDCARE_ATTENDANCE_COLUMNS


def _rows():
    rows = pd.DataFrame(
        {
            "donor_id": ["a", "b", "c", "d"],
            "source_household_id": ["a", "b", "c", "d"],
            "age": [3, 3, 3, 4],
            "region": 1,
            "parent_work_status": 2,
            "income_band": 1,
            "household_income": 40000,
            "household_members": 3,
            "resident_parent_count": 2,
            "attendance_status": "complete",
            "child_weight": [1.0, 2.0, 3.0, 4.0],
            MONTH: [0.0, 4.0, 9.0, 22.0],
            DAYS: [0.0, 1.0, 2.0, 5.0],
            HOURS: [0.0, 1.0, 2.0, 10.0],
        }
    )
    return with_childcare_household_size(rows)


class _FakeFit:
    def predict_from_uniforms(self, rows, *, quantiles, sign_uniforms):
        key = module.QRF_CHILDCARE_SCHEDULE_CODE
        assert set(quantiles) == set(sign_uniforms) == {key}
        return pd.DataFrame({key: np.resize([0.0, 39.0, 100.0], len(rows))})


def test_decoder_retains_exact_age_joint_pairs_and_lower_ties(monkeypatch):
    captured = []

    def fit(rows, predictors, targets, **kwargs):
        captured.append((rows.copy(), predictors, targets, kwargs))
        return _FakeFit()

    monkeypatch.setattr(module, "fit", fit)
    model = module.QRFChildcareSchedules(_rows())
    target = _rows().iloc[[0]].assign(source_household_id="new")
    model.prepare(target)
    values, cumulative, weights = model.distribution(target.iloc[0])
    assert set(map(tuple, values)) == {
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 4.0),
    }
    assert cumulative[-1] == 1
    assert weights.sum() == pytest.approx(1)
    assert captured[0][3] == {
        "weights": "child_weight",
        "seed": 915,
        "n_estimators": 100,
    }
    assert captured[0][1] == list(module.QRF_CHILDCARE_PREDICTORS)


def test_qrf_cannot_score_its_own_training_households(monkeypatch):
    monkeypatch.setattr(module, "fit", lambda *a, **k: _FakeFit())
    rows = _rows()
    model = module.QRFChildcareSchedules(rows)
    with pytest.raises(ValueError, match="overlap"):
        model.prepare(rows)
    with pytest.raises(ValueError, match="overlap"):
        model.distribution(rows.iloc[0])
    with pytest.raises(ValueError, match="exact-age"):
        model.prepare(rows.iloc[[0]].assign(source_household_id="new", age=9))


def test_qrf_excludes_unknown_calendars_and_rejects_reconstructed_truth(monkeypatch):
    captured = []

    def fit(rows, *args, **kwargs):
        captured.append(rows.copy())
        return _FakeFit()

    monkeypatch.setattr(module, "fit", fit)
    rows = _rows()
    rows.loc[1, "attendance_status"] = "partial_calendar"
    rows.loc[1, [MONTH, DAYS, HOURS]] = np.nan
    module.QRFChildcareSchedules(rows)
    assert captured[0].donor_id.tolist() == ["a", "c", "d"]
    rows.loc[1, "attendance_status"] = "summary_bridge"
    with pytest.raises(ValueError, match="original measured"):
        module.QRFChildcareSchedules(rows)


def test_real_canonical_model_gives_valid_row_order_invariant_schedules():
    model = module.QRFChildcareSchedules(_rows(), n_estimators=4)
    targets = _rows().iloc[[0, 3]].assign(source_household_id=["new1", "new2"])
    model.prepare(targets)
    first = model.distribution(targets.iloc[0])
    model.prepare(targets.iloc[::-1])
    second = model.distribution(targets.iloc[0])
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert set(map(tuple, first[0])).issubset(
        {(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 2.0, 4.0)}
    )
