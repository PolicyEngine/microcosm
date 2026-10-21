from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime import salary_sacrifice
from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.salary_sacrifice import (
    SALSAC_MOVE_PAY_SHARE_CAP,
    SALSAC_OUTPUT,
    SALSAC_PRE_CONVERSION_PAY_COLUMN,
    SALSAC_RATE_CAP,
    SALSAC_STAGE_TARGET,
    UKSalarySacrificeStageTransform,
    _assert_salary_sacrifice_stage_parameters,
    impute_salary_sacrifice,
    load_salary_sacrifice_anchor,
)


class _Fitted:
    value = 0.0

    def predict_from_uniforms(
        self, predictors: pd.DataFrame, *, quantiles, sign_uniforms
    ) -> pd.DataFrame:
        # The stage keys each person's uniforms on their own id.
        for uniforms in (quantiles[SALSAC_OUTPUT], sign_uniforms[SALSAC_OUTPUT]):
            assert uniforms.shape == (len(predictors),)
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
    employment_income=None,
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
            "employment_income": (
                np.full(n, 30_000.0)
                if employment_income is None
                else np.asarray(employment_income, dtype=float)
            ),
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
            HOUSEHOLD_IS_CGT_SUPPORT_COPY: donors,
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


def test_qrf_predicts_only_for_people_not_asked_who_are_paid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given people not asked the question, with and without employment
    # income, one of them carrying a stale amount.
    _FakeQRF.training_frames = []
    _Fitted.value = 900.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(
        asked=[1, 1, 0, 0, 0],
        salary_sacrifice_values=[10.0, 20.0, 0.0, 0.0, 55.0],
        employee_pension=[0.0] * 5,
        employment_income=[30_000.0, 0.0, 25_000.0, 0.0, 0.0],
        weights=np.asarray([1.0, 1.0, 1.0, 2.0, 3.0]),
    )

    result = impute_salary_sacrifice(frame)

    # Then the paid person not asked is predicted for and the unpaid ones are
    # zero. The asked person without pay reported 20: a sacrifice comes out of
    # pay, so it is limited to none (uk-data#518).
    assert result.frame.table("person")[SALSAC_OUTPUT].tolist() == [
        10.0,
        0.0,
        900.0,
        0.0,
        0.0,
    ]
    assert result.prediction_rows == 1
    assert result.limited_to_pay_rows == 1
    assert result.limited_to_pay_mass == 1.0
    assert result.limited_to_pay_amount == 20.0
    assert result.not_asked_without_pay_rows == 2
    assert result.not_asked_without_pay_mass == 5.0
    qrf = result.evidence()["qrf"]
    assert qrf["not_asked_without_pay_rows"] == 2
    assert qrf["not_asked_without_pay_mass"] == 5.0
    assert "employment_income > 0" in qrf["target_population"]
    assert qrf["limited_to_pay_rows"] == 1
    assert qrf["limited_to_pay_amount"] == 20.0
    assert "employment_income" in qrf["amount_limit"]
    # The headcount the conversion tops up starts from true users only: the
    # asked person without pay is no longer one of them.
    assert result.pre_headcount == 2.0


def test_qrf_trains_only_on_asked_respondents_with_pay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A respondent without pay shows nothing about what paid people sacrifice."""

    _FakeQRF.training_frames = []
    _Fitted.value = 0.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(
        asked=[1, 1, 1, 0],
        salary_sacrifice_values=[10.0, 0.0, 30.0, 0.0],
        employee_pension=[0.0] * 4,
        employment_income=[30_000.0, 0.0, 20_000.0, 25_000.0],
    )

    result = impute_salary_sacrifice(frame)

    assert _FakeQRF.training_frames[0].index.tolist() == [0, 2]
    assert result.training_rows == 2


def test_the_forest_needs_an_asked_respondent_with_pay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(
        asked=[1, 0],
        salary_sacrifice_values=[0.0, 0.0],
        employee_pension=[0.0, 0.0],
        employment_income=[0.0, 25_000.0],
    )

    with pytest.raises(ValueError, match="no eligible asked FRS rows with pay"):
        impute_salary_sacrifice(frame)


def test_every_amount_is_limited_to_the_pay_that_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reported and predicted amounts alike, on FRS and SPI-channel rows.

    An SPI-channel row keeps its FRS twin's answer beside pay redrawn from the
    SPI, so its reported amount can sit above its pay.
    """

    _FakeQRF.training_frames = []
    _Fitted.value = 900.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(
        asked=[1, 1, 0, 0, 1],
        salary_sacrifice_values=[4_000.0, 4_000.0, 0.0, 0.0, 250.0],
        employee_pension=[0.0] * 5,
        employment_income=[30_000.0, 1_500.0, 500.0, 25_000.0, 250.0],
        channels=["frs", "spi", "frs", "spi", "frs"],
        weights=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    result = impute_salary_sacrifice(frame)
    person = result.frame.table("person")

    # The SPI-channel respondent's 4,000 against pay of 1,500 and the
    # prediction of 900 against pay of 500 are limited; an amount equal to pay
    # stays.
    assert person[SALSAC_OUTPUT].tolist() == [4_000.0, 1_500.0, 500.0, 900.0, 250.0]
    assert (person[SALSAC_OUTPUT] <= person["employment_income"]).all()
    assert result.limited_to_pay_rows == 2
    assert result.limited_to_pay_mass == 5.0
    assert result.limited_to_pay_amount == 2_500.0 + 400.0


def test_conversion_moves_a_contribution_within_half_the_pay_whole(
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
    # Sacrificed pay leaves employment_income (microcosm#1069 c9); records the
    # stage does not convert keep their reported pay.
    np.testing.assert_allclose(
        person.loc[converted, "employment_income"],
        30_000.0 - person.loc[converted, SALSAC_OUTPUT],
    )
    assert (person.loc[~converted, "employment_income"] == 30_000.0).all()
    # The converted record's pay before the conversion stays on the carrier,
    # zero for everyone else, so the rewrite reverses exactly (microcosm#1063).
    assert (person.loc[converted, SALSAC_PRE_CONVERSION_PAY_COLUMN] == 30_000.0).all()
    assert (person.loc[~converted, SALSAC_PRE_CONVERSION_PAY_COLUMN] == 0.0).all()
    assert SALSAC_PRE_CONVERSION_PAY_COLUMN in (
        UKSalarySacrificeStageTransform.output_columns()
    )
    evidence = result.evidence()["headcount_receipt"]
    assert evidence["target"] == SALSAC_STAGE_TARGET
    assert evidence["converted_rows"] == result.converted_rows
    assert evidence["pre_conversion_pay_column"] == SALSAC_PRE_CONVERSION_PAY_COLUMN
    # No contribution here is above half the pay, so none is split.
    assert evidence["move_pay_share_cap"] == SALSAC_MOVE_PAY_SHARE_CAP
    assert evidence["pay_share_capped_rows"] == 0
    assert evidence["retained_employee_amount"] == 0.0


def test_conversion_moves_at_most_half_the_pay_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sacrifice never exceeds the pay that remains (uk-data#518).

    The whole contribution used to move, and pay was floored at zero, which
    left a converted record with a sacrifice and no pay.
    """

    _Fitted.value = 0.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    n_donors = 100
    pay = np.asarray([30_000.0, *([1_000.0] * n_donors)])
    # Contributions from 100 to 1,000: above half the pay from 500 up, and at
    # most the whole pay, where the spine caps a net-pay contribution
    # (uk_runtime.pension_coherence).
    contributions = np.asarray([0.0, *np.linspace(100.0, 1_000.0, n_donors)])
    frame = _frame(
        asked=[1, *([0] * n_donors)],
        salary_sacrifice_values=[0.0] * (n_donors + 1),
        employee_pension=contributions,
        employment_income=pay,
        weights=[1.0, *([100_000.0] * n_donors)],
    )

    result = impute_salary_sacrifice(frame)
    person = result.frame.table("person")
    converted = (person[SALSAC_PRE_CONVERSION_PAY_COLUMN] > 0.0).to_numpy()

    assert 0 < result.converted_rows < n_donors
    moved = person.loc[converted, SALSAC_OUTPUT].to_numpy()
    np.testing.assert_array_equal(
        moved, np.minimum(contributions[converted], 0.5 * pay[converted])
    )
    # Pay falls by the amount moved, so pay plus sacrifice is the carrier and
    # the sacrifice is within the pay that remains.
    np.testing.assert_array_equal(
        person.loc[converted, "employment_income"].to_numpy() + moved,
        person.loc[converted, SALSAC_PRE_CONVERSION_PAY_COLUMN].to_numpy(),
    )
    assert (person[SALSAC_OUTPUT] <= person["employment_income"]).all()
    assert (person.loc[converted, "employment_income"] > 0.0).all()
    # What does not move stays an employee contribution, so the two columns
    # still add up to the contribution.
    np.testing.assert_allclose(
        person["employee_pension_contributions"].to_numpy()
        + np.where(converted, person[SALSAC_OUTPUT].to_numpy(), 0.0),
        contributions,
    )
    capped = converted & (contributions > 0.5 * pay)
    assert capped.any() and (converted & ~capped).any()
    assert result.pay_share_capped_rows == int(capped.sum())
    assert result.moved_amount == pytest.approx(moved.sum())
    assert result.retained_employee_amount == pytest.approx(
        (contributions[capped] - 0.5 * pay[capped]).sum()
    )


_AMOUNTS = st.sampled_from(
    [0.0, 0.01, 250.0, 499.99, 500.0, 1_000.0, 2_500.0, 40_000.0]
)


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.booleans(),  # asked
            _AMOUNTS,  # reported salary sacrifice
            _AMOUNTS,  # employee pension contribution
            _AMOUNTS,  # pay
            st.sampled_from(["frs", "spi"]),
        ),
        min_size=1,
        max_size=12,
    ),
    predicted=_AMOUNTS,
)
def test_no_amount_exceeds_pay_and_nothing_is_lost(rows, predicted) -> None:
    """Whatever is reported, predicted or converted, the sacrifice is within pay.

    The invariants of uk-data#518's rule: a sacrifice is at most the pay that
    remains and nil without pay; a converted record's pay and sacrifice add up
    to its pay before; and a contribution is moved or kept, never dropped.
    """

    # One asked, paid FRS respondent always leads, so the forest has a row.
    rows = [(True, 0.0, 0.0, 30_000.0, "frs"), *rows]
    asked = [1 if row[0] else 0 for row in rows]
    reported = np.asarray([row[1] if row[0] else 0.0 for row in rows])
    pay = np.asarray([row[3] for row in rows])
    # The spine caps net-pay contributions at pay before this stage
    # (uk_runtime.pension_coherence), so the stage only ever sees them there.
    contributions = np.minimum(np.asarray([row[2] for row in rows]), pay)
    frame = _frame(
        asked=asked,
        salary_sacrifice_values=reported,
        employee_pension=contributions,
        employment_income=pay,
        channels=[row[4] for row in rows],
        weights=np.full(len(rows), 1_000_000.0),
    )
    original_qrf, original_value = salary_sacrifice.QRF, _Fitted.value
    salary_sacrifice.QRF, _Fitted.value = _FakeQRF, predicted
    try:
        result = impute_salary_sacrifice(frame)
    finally:
        salary_sacrifice.QRF, _Fitted.value = original_qrf, original_value
    person = result.frame.table("person")
    sacrifice = person[SALSAC_OUTPUT].to_numpy()
    pay_after = person["employment_income"].to_numpy()
    carrier = person[SALSAC_PRE_CONVERSION_PAY_COLUMN].to_numpy()
    converted = carrier > 0.0

    assert (sacrifice >= 0.0).all() and (sacrifice <= pay_after).all()
    assert (sacrifice[pay_after == 0.0] == 0.0).all()
    np.testing.assert_array_equal(pay_after[~converted], pay[~converted])
    np.testing.assert_array_equal(
        pay_after[converted] + sacrifice[converted], carrier[converted]
    )
    np.testing.assert_array_equal(carrier[converted], pay[converted])
    kept = person["employee_pension_contributions"].to_numpy()
    np.testing.assert_array_equal(kept[~converted], contributions[~converted])
    np.testing.assert_allclose(
        kept[converted] + sacrifice[converted], contributions[converted]
    )
    assert (sacrifice[converted] <= SALSAC_MOVE_PAY_SHARE_CAP * pay[converted]).all()


def test_a_contribution_above_the_pay_that_remains_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spine caps net-pay contributions at pay; the stage refuses one above it."""

    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(
        asked=[1, 1],
        salary_sacrifice_values=[0.0, 0.0],
        employee_pension=[0.0, 3_000.0],
        employment_income=[30_000.0, 1_000.0],
    )

    with pytest.raises(ValueError, match="must not exceed the pay that remains"):
        impute_salary_sacrifice(frame)


def test_draws_move_only_for_the_people_whose_inputs_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each prediction reads uniforms keyed on the person's own id: reversing
    the people, changing one and adding another moves no other person's draw
    (Vahid's review of microcosm#1332)."""

    from test_support.microcosm_build.uk_keyed_draws import (
        assert_draws_are_keyed_on_ids,
    )

    drawn: dict[str, pd.Series] = {}

    class _Keyed(_Fitted):
        def predict_from_uniforms(self, predictors, *, quantiles, sign_uniforms):
            drawn["quantile"] = pd.Series(
                quantiles[SALSAC_OUTPUT], index=predictors.index
            )
            return pd.DataFrame({SALSAC_OUTPUT: 0.0}, index=predictors.index)

    class _KeyedQRF(_FakeQRF):
        def fit(self, frame, predictors, targets, *, weights):
            super().fit(frame, predictors, targets, weights=weights)
            return _Keyed()

    monkeypatch.setattr(salary_sacrifice, "QRF", _KeyedQRF)

    def draw(recipient: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        # The person rows keep the recipients' order; the group tables are
        # sorted by id, as a frame requires.
        groups = np.sort(ids)
        frame = uk_national_frame(
            person=recipient.assign(
                person_id=ids, person_benunit_id=ids, person_household_id=ids
            ),
            benunit=pd.DataFrame({"benunit_id": groups}),
            household=pd.DataFrame(
                {
                    "household_id": groups,
                    "household_support_channel": "frs",
                    HOUSEHOLD_IS_CGT_CLONE: False,
                    HOUSEHOLD_IS_CGT_SUPPORT_COPY: False,
                }
            ),
            household_weights=np.ones(len(ids)),
            time_period="2024",
        )
        drawn.clear()
        impute_salary_sacrifice(frame)
        quantile = pd.Series(np.nan, index=frame.table("person").index)
        quantile.loc[drawn["quantile"].index] = drawn["quantile"].to_numpy()
        return pd.DataFrame({"quantile": quantile.to_numpy()})

    rows = 6
    recipient = pd.DataFrame(
        {
            "age": np.linspace(25, 55, rows),
            "employment_income": np.full(rows, 30_000.0),
            "salary_sacrifice_asked": [1, 1, 0, 0, 0, 0],
            SALSAC_OUTPUT: [10.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            "employee_pension_contributions": np.zeros(rows),
        }
    )
    assert_draws_are_keyed_on_ids(
        draw,
        recipient,
        np.arange(101, 101 + rows, dtype="int64"),
        perturbed_column="age",
        perturbed_value=61.0,
    )


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
                "clamp_maximum",
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
                "move_pay_share_cap",
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
