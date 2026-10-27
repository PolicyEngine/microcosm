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
    BONUS_SACRIFICE_OUTPUT,
    BONUS_SACRIFICE_SOURCE,
    SALSAC_MOVE_PAY_SHARE_CAP,
    SALSAC_OUTPUT,
    SALSAC_PRE_CONVERSION_PAY_COLUMN,
    SALSAC_RATE_CAP,
    SALSAC_STAGE_TARGET,
    UKSalarySacrificeStageTransform,
    _assert_bonus_sacrifice_anchor,
    _assert_salary_sacrifice_stage_parameters,
    bonus_sacrifice_excess,
    convert_bonuses_to_bonus_sacrifice,
    impute_salary_sacrifice,
    load_bonus_sacrifice_anchor,
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
    bonus=None,
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
            BONUS_SACRIFICE_SOURCE: np.zeros(n) if bonus is None else bonus,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_source_id": ids,
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
                    "household_source_id": groups,
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
            BONUS_SACRIFICE_SOURCE: np.zeros(rows),
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
                "artifact_role",
                "resource",
                "feed_pin",
                "record_sets",
                "source_vintage",
                "mapped_build_period",
                "period_mapping",
                "require_before_source_read",
                "runtime_sha256_required",
                "fail_on_mismatch",
            )
        ],
        *[
            (1, name)
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
            (2, name)
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
        *[
            (3, name)
            for name in (
                "resource",
                "facts",
                "source",
                "output",
                "candidates",
                "move",
                "move_pay_share_cap",
                "spi_bonus_draw",
                "spi_cell_quantiles",
                "spi_min_donor_sources",
                "spi_salt",
                "threshold",
                "solve",
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
            _drift(2, "undeclared_extra_key"),
            anchor=load_salary_sacrifice_anchor(),
        )
    stage = _stage()
    extra = replace(stage, operations=(*stage.operations, stage.operations[-1]))
    with pytest.raises(ValueError, match="operation order drifted"):
        _assert_salary_sacrifice_stage_parameters(
            extra, anchor=load_salary_sacrifice_anchor()
        )


def _toy_bonus_anchor(**overrides) -> dict:
    anchor = {
        "engine": {"package": "policyengine-uk", "version": "toy"},
        "source": {},
        "obr_base_above_threshold": 0.0,
        "threshold": 2000,
        "data_year": 2024,
        "base_year": 2027,
        "earnings_growth": {"factor": 1.0},
        "weight_growth": {"factor": 1.0},
        "upper_earnings_limit": {"weekly": 50_000 / 52, "annual": 50_000.0},
    }
    anchor.update(overrides)
    return anchor


def test_bonus_excess_is_the_bonus_routes_share_above_the_threshold() -> None:
    """The threshold applies to the two routes summed (microcosm#1340)."""

    salary = np.asarray([0.0, 1_500.0, 3_000.0, 0.0])
    bonus = np.asarray([1_000.0, 1_000.0, 1_000.0, 3_000.0])
    np.testing.assert_allclose(
        bonus_sacrifice_excess(salary, bonus, earnings_growth=1.0),
        [0.0, 500.0, 1_000.0, 1_000.0],
    )
    # The amounts grow to the base year before the fixed threshold applies.
    np.testing.assert_allclose(
        bonus_sacrifice_excess(salary, bonus, earnings_growth=2.0),
        [0.0, 2_000.0, 2_000.0, 4_000.0],
    )


def _conversion(target: float, **kwargs):
    defaults = dict(
        bonus=np.asarray([10_000.0, 10_000.0, 10_000.0, 10_000.0, 0.0, 10_000.0]),
        employment_income=np.asarray(
            [100_000.0, 100_000.0, 100_000.0, 100_000.0, 100_000.0, 45_000.0]
        ),
        salary_sacrifice=np.zeros(6),
        weights=np.ones(6),
        uniforms=np.asarray([0.4, 0.1, 0.3, 0.2, 0.05, 0.01]),
        anchor=_toy_bonus_anchor(obr_base_above_threshold=target),
    )
    defaults.update(kwargs)
    return convert_bonuses_to_bonus_sacrifice(**defaults)


def test_bonus_candidates_have_a_bonus_and_pay_at_the_limit() -> None:
    """A sacrificed bonus sits beside pay, so pay itself must reach the limit."""

    result = _conversion(0.0)
    # No bonus (row 4), and pay under the limit (row 5).
    np.testing.assert_array_equal(
        result.candidates, [True, True, True, True, False, False]
    )
    assert not result.participants.any()


def test_bonus_participants_follow_their_uniforms_to_the_nearest_base() -> None:
    """Each candidate adds 8,000 above the threshold; 17,000 is nearest two."""

    result = _conversion(17_000.0)
    # The two candidates with the smallest uniforms: rows 1 (0.1) and 3 (0.2).
    np.testing.assert_array_equal(
        result.participants, [False, True, False, True, False, False]
    )
    np.testing.assert_allclose(result.sacrificed, [0, 10_000, 0, 10_000, 0, 0])
    assert result.projected_base == pytest.approx(16_000.0)
    assert result.all_candidates_fall_short is False
    # The engine's growth moves the base, not who is first.
    grown = _conversion(
        17_000.0,
        anchor=_toy_bonus_anchor(
            obr_base_above_threshold=17_000.0, weight_growth={"factor": 2.0}
        ),
    )
    np.testing.assert_array_equal(
        grown.participants, [False, True, False, False, False, False]
    )
    assert grown.projected_base == pytest.approx(16_000.0)


def test_every_candidate_converts_where_all_fall_short() -> None:
    result = _conversion(1e12)
    np.testing.assert_array_equal(result.participants, result.candidates)
    assert result.all_candidates_fall_short is True


def test_a_bonus_is_sacrificed_whole_up_to_half_the_pay() -> None:
    """María's ruling of 2026-10-24 (Vahid's review of #1383, item 5).

    The sacrifice is added beside pay, so no other route's share of the pay
    limits it: only half the pay does, as an imputation bound.
    """

    result = _conversion(
        1e12,
        bonus=np.asarray([80_000.0, 10_000.0, 10_000.0, 0.0, 0.0, 0.0]),
        employment_income=np.asarray([140_000.0, 100_000.0, 100_000.0, 0.0, 0.0, 0.0]),
        salary_sacrifice=np.asarray([0.0, 0.0, 95_000.0, 0.0, 0.0, 0.0]),
    )
    # Half of 140,000; the whole bonus twice, the salary route beside it.
    np.testing.assert_allclose(result.sacrificed, [70_000, 10_000, 10_000, 0, 0, 0])
    np.testing.assert_array_equal(
        result.pay_share_capped, [True, False, False, False, False, False]
    )


def _spi_draw(**kwargs):
    from microcosm.build.uk_runtime.salary_sacrifice import draw_spi_bonuses

    n_donors = 25
    defaults = dict(
        # 25 FRS donors at 100,000 with a tenth of pay as bonus, then two
        # SPI adults (one paid, one not) and one FRS adult with a bonus of
        # its own.
        bonus=np.asarray([10_000.0] * n_donors + [99_000.0, 5_000.0, 7_000.0]),
        employment_income=np.asarray(
            [100_000.0] * n_donors + [300_000.0, 0.0, 80_000.0]
        ),
        weights=np.ones(n_donors + 3),
        person_ids=np.arange(1, n_donors + 4),
        source_ids=np.arange(1, n_donors + 4),
        donors=np.asarray([True] * n_donors + [False, False, False]),
        spi=np.asarray([False] * n_donors + [True, True, False]),
    )
    defaults.update(kwargs)
    return draw_spi_bonuses(**defaults)


def test_an_spi_adult_draws_a_bonus_share_of_its_own_pay() -> None:
    """Vahid's review of #1383, item 6, and María's ruling of 2026-10-24.

    The SPI adult's twin's bonus is set aside: it takes a donor's tenth of
    pay, on its own pay. One without pay has none, and an FRS adult that is
    no donor keeps its own.
    """

    draw = _spi_draw()
    np.testing.assert_allclose(draw.bonus[-3:], [30_000.0, 0.0, 7_000.0])
    np.testing.assert_array_equal(draw.recipients[-3:], [True, False, False])
    np.testing.assert_allclose(draw.bonus[:-3], 10_000.0)


def test_spi_bonus_cells_hold_the_donor_floor_and_the_draw_is_keyed() -> None:
    """A cell with fewer than 20 donor households joins the one below.

    Five high-paid donors with half their pay as bonus are too few for a cell
    of their own, so they draw with the others. The draw is keyed on person
    ids, so shuffling the rows leaves each SPI adult's bonus where it was.
    """

    bonus = np.asarray([10_000.0] * 25 + [250_000.0] * 5 + [0.0])
    pay = np.asarray([100_000.0] * 25 + [500_000.0] * 5 + [600_000.0])
    ids = np.arange(1, 32)
    donors = np.asarray([True] * 30 + [False])
    spi = np.asarray([False] * 30 + [True])
    draw = _spi_draw(
        bonus=bonus,
        employment_income=pay,
        weights=np.ones(31),
        person_ids=ids,
        source_ids=ids,
        donors=donors,
        spi=spi,
    )
    assert draw.cells == 1
    assert draw.bonus[-1] in {60_000.0, 300_000.0}
    order = np.random.default_rng(7).permutation(31)
    shuffled = _spi_draw(
        bonus=bonus[order],
        employment_income=pay[order],
        weights=np.ones(31),
        person_ids=ids[order],
        source_ids=ids[order],
        donors=donors[order],
        spi=spi[order],
    )
    assert shuffled.bonus[np.flatnonzero(order == 30)[0]] == draw.bonus[-1]


def test_spi_bonuses_need_paid_frs_donors() -> None:
    with pytest.raises(ValueError, match="base-FRS adults with pay"):
        _spi_draw(donors=np.zeros(28, dtype=bool))


def test_the_stage_adds_bonus_sacrifice_beside_pay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Fitted.value = 0.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    n = 12
    frame = _frame(
        asked=[1] * n,
        salary_sacrifice_values=[0.0] * n,
        employee_pension=[0.0] * n,
        employment_income=[120_000.0] * n,
        bonus=[20_000.0] * n,
    )
    result = impute_salary_sacrifice(
        frame, bonus_anchor=_toy_bonus_anchor(obr_base_above_threshold=90_000.0)
    )
    person = result.frame.table("person")
    converted = person[BONUS_SACRIFICE_OUTPUT] > 0.0
    # Each converted bonus adds 18,000 above the threshold: five is nearest.
    assert int(converted.sum()) == 5
    assert (person.loc[converted, BONUS_SACRIFICE_OUTPUT] == 20_000.0).all()
    # A sacrificed bonus is never received, so pay stays as it is.
    assert (person["employment_income"] == 120_000.0).all()
    assert (person[SALSAC_PRE_CONVERSION_PAY_COLUMN] == 0.0).all()
    receipt = result.evidence()["bonus_sacrifice_receipt"]
    assert receipt["pay"] == "unchanged: a sacrificed bonus is never received"
    assert receipt["spi_drawn_support"] == 0
    assert receipt["target_base"] == 90_000.0
    # Five sources: the base and the band are withheld under ten.
    assert receipt["participant_support"] == "<10"
    assert receipt["projected_base"] == "<10"
    assert receipt["candidate_support"] == "10-29"
    assert BONUS_SACRIFICE_OUTPUT in UKSalarySacrificeStageTransform.output_columns()


def test_the_stage_refuses_a_frame_without_the_bonus_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Fitted.value = 0.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(asked=[1], salary_sacrifice_values=[0.0], employee_pension=[0.0])
    person = frame.table("person").drop(columns=BONUS_SACRIFICE_SOURCE)
    bare = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values,
        time_period="2024",
    )
    with pytest.raises(ValueError, match=BONUS_SACRIFICE_SOURCE):
        impute_salary_sacrifice(bare, bonus_anchor=_toy_bonus_anchor())


def test_the_stage_refuses_an_anchor_for_another_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Fitted.value = 0.0
    monkeypatch.setattr(salary_sacrifice, "QRF", _FakeQRF)
    frame = _frame(asked=[1], salary_sacrifice_values=[0.0], employee_pension=[0.0])
    with pytest.raises(ValueError, match="projects from 2023"):
        impute_salary_sacrifice(frame, bonus_anchor=_toy_bonus_anchor(data_year=2023))


def test_the_packaged_bonus_anchor_is_the_vendored_obr_row() -> None:
    anchor = load_bonus_sacrifice_anchor()
    _assert_bonus_sacrifice_anchor(anchor)
    assert anchor["base_year"] == 2027
    assert anchor["obr_base_above_threshold"] == 13.0e9
    with pytest.raises(ValueError, match="differs from the vendored row"):
        _assert_bonus_sacrifice_anchor({**anchor, "obr_base_above_threshold": 1.0})


def test_the_hmrc_salary_sacrifice_rows_bind_the_salary_route_alone() -> None:
    """ASHE, which HMRC's salary-sacrifice figures come from, covers no bonus
    sacrifice, so those rows keep reading the salary route (microcosm#1340)."""

    import json
    from importlib.resources import files

    targets = json.loads(
        files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text("utf-8")
    )["targets"]
    rows = [t for t in targets if t["target_id"].startswith("hmrc.salary_sacrifice")]
    assert rows
    for row in rows:
        text = json.dumps(row)
        assert BONUS_SACRIFICE_OUTPUT not in text, row["target_id"]
