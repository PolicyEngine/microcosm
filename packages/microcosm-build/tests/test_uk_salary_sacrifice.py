from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime import salary_sacrifice
from microcosm.build.uk_runtime.cgt_structure import (
    HOUSEHOLD_IS_CGT_BAND_DONOR,
    HOUSEHOLD_IS_CGT_CLONE,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.salary_sacrifice import (
    SALSAC_OUTPUT,
    SALSAC_RATE_CAP,
    SALSAC_STAGE_TARGET,
    _assert_salary_sacrifice_stage_parameters,
    impute_salary_sacrifice,
    load_salary_sacrifice_anchor,
)


class _Fitted:
    value = 0.0

    def predict(self, predictors: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({SALSAC_OUTPUT: self.value}, index=predictors.index)


class _FakeQRF:
    training_frames: list[pd.DataFrame] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        assert n_estimators == 100
        assert seed == 42

    def fit(self, frame, predictors, targets, *, weights):
        assert predictors == ["age", "employment_income"]
        assert targets == [SALSAC_OUTPUT]
        assert weights == "_fit_weight"
        self.training_frames.append(frame.copy())
        return _Fitted()


def _frame(
    *,
    asked,
    salary_sacrifice_values,
    employee_pension,
    channels=None,
    clones=None,
    donors=None,
    weights=None,
):
    n = len(asked)
    ids = np.arange(1, n + 1, dtype="int64")
    channels = channels or ["frs"] * n
    clones = clones or [False] * n
    donors = donors or [False] * n
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": np.linspace(25, 55, n),
            "employment_income": np.full(n, 30_000.0),
            "salary_sacrifice_asked": asked,
            SALSAC_OUTPUT: salary_sacrifice_values,
            "employee_pension_contributions": employee_pension,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_support_channel": channels,
            HOUSEHOLD_IS_CGT_CLONE: clones,
            HOUSEHOLD_IS_CGT_BAND_DONOR: donors,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household,
        household_weights=np.ones(n) if weights is None else weights,
        time_period="2024",
    )


@lru_cache
def _stage():
    return load_country_spec("uk").sources.stage_map()["salary_sacrifice"]


def _drift(operation_index: int, parameter: str):
    stage = _stage()
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        operation.kind,
        {**operation.parameters, parameter: "__drift__"},
    )
    return replace(stage, operations=tuple(operations))


def test_qrf_preserves_asked_rows_and_excludes_nonbase_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeQRF.training_frames = []
    _Fitted.value = -25.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(
        asked=[1, 1, 1, 1, 0],
        salary_sacrifice_values=[10.0, 20.0, 30.0, 40.0, 0.0],
        employee_pension=[0.0] * 5,
        channels=["frs", "spi", "frs", "frs", "spi"],
        clones=[False, False, True, False, False],
        donors=[False, False, False, True, False],
    )

    result = impute_salary_sacrifice(frame)

    assert len(_FakeQRF.training_frames) == 1
    assert _FakeQRF.training_frames[0].index.tolist() == [0]
    assert result.frame.table("person")[SALSAC_OUTPUT].tolist() == [
        10.0,
        20.0,
        30.0,
        40.0,
        0.0,
    ]
    assert result.training_rows == 1
    assert result.prediction_rows == 1


def test_conversion_moves_full_pension_zeros_source_and_records_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Fitted.value = 0.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    n_donors = 100
    frame = _frame(
        asked=[1, *([0] * n_donors)],
        salary_sacrifice_values=[0.0] * (n_donors + 1),
        employee_pension=[0.0, *np.linspace(100.0, 1_000.0, n_donors)],
        weights=[1.0, *([100_000.0] * n_donors)],
    )

    result = impute_salary_sacrifice(frame)
    person = result.frame.table("person")
    converted = person[SALSAC_OUTPUT] > 0.0

    assert result.rate == SALSAC_RATE_CAP
    assert result.cap_bound is True
    assert 0 < result.converted_rows < n_donors
    assert (person.loc[converted, "employee_pension_contributions"] == 0.0).all()
    assert result.moved_amount == pytest.approx(
        person.loc[converted, SALSAC_OUTPUT].sum()
    )
    evidence = result.evidence()["headcount_receipt"]
    assert evidence["target"] == SALSAC_STAGE_TARGET
    assert evidence["converted_rows"] == result.converted_rows


def test_anchor_is_self_consistent() -> None:
    anchor = load_salary_sacrifice_anchor()
    assert anchor["hmrc_anchor"]["total_users"] * anchor["derived"][
        "staging_ratio"
    ] == pytest.approx(anchor["derived"]["stage_target"])


@pytest.mark.parametrize(
    "operation_index,parameter",
    [
        *[
            (0, name)
            for name in (
                "training_population",
                "target_population",
                "predictors",
                "targets",
                "weights",
                "weight_mapping",
                "seed",
                "n_estimators",
                "clamp_minimum",
                "preserve_asked_rows",
                "cache",
            )
        ],
        *[
            (1, name)
            for name in (
                "resource",
                "target",
                "donor_pool",
                "rate_cap",
                "move",
                "seed",
                "salt",
                "receipt",
            )
        ],
    ],
)
def test_manifest_drift_assert_covers_every_reviewed_parameter(
    operation_index: int, parameter: str
) -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_salary_sacrifice_stage_parameters(
            _drift(operation_index, parameter),
            anchor=load_salary_sacrifice_anchor(),
        )


def test_resource_drift_assert_rejects_anchor_change() -> None:
    anchor = dict(load_salary_sacrifice_anchor())
    anchor["derived"] = {**anchor["derived"], "stage_target": 1}
    with pytest.raises(ValueError, match="stage_target.*drifted"):
        _assert_salary_sacrifice_stage_parameters(_stage(), anchor=anchor)


def test_drift_assert_rejects_extra_keys_and_operations() -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_salary_sacrifice_stage_parameters(
            _drift(1, "undeclared_extra_key"),
            anchor=load_salary_sacrifice_anchor(),
        )
    stage = _stage()
    extra = replace(stage, operations=(*stage.operations, stage.operations[-1]))
    with pytest.raises(ValueError, match="operation order drifted"):
        _assert_salary_sacrifice_stage_parameters(
            extra, anchor=load_salary_sacrifice_anchor()
        )
