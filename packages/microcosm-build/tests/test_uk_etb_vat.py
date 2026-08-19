from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.etb_vat import (
    UK_ETB_VAT_FIT_NAME,
    clean_etb_vat_table,
    donor_realized_ranges,
    impute_etb_vat,
    support_clip_to_donor,
)


def _raw_etb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2023, 2023, 2023],
            "adults": [9, 2, 1, " "],
            "childs": [9, 1, 0, 0],
            "noretd": [9, 0, 1, 0],
            "disinc": [9, 100.0, 200.0, 300.0],
            "totvat": [9, 20.0, 10.0, 30.0],
            "expdis": [9, 120.0, 110.0, 130.0],
            "hhold_adj_weight": [9, 3.0, 4.0, 5.0],
        }
    )


def test_etb_vat_cleaning_filters_2023_and_computes_target() -> None:
    donor = clean_etb_vat_table(_raw_etb())

    assert donor["is_adult"].tolist() == [2.0, 1.0]
    assert donor["is_child"].tolist() == [1.0, 0.0]
    assert donor["is_SP_age"].tolist() == [0.0, 1.0]
    assert donor["household_net_income"].tolist() == [5200.0, 10400.0]
    assert donor["weight"].tolist() == [3.0, 4.0]
    expected = [
        (20.0 * 0.975 / 0.20) / (120.0 - 20.0),
        (10.0 * 0.975 / 0.20) / (110.0 - 10.0),
    ]
    np.testing.assert_allclose(donor["full_rate_vat_expenditure_rate"], expected)


def test_etb_vat_cleaning_fails_loud_on_missing_rate() -> None:
    with pytest.raises(ValueError, match="standard_rate"):
        clean_etb_vat_table(_raw_etb(), standard_rate=np.nan)


def test_etb_vat_support_clip_and_ranges() -> None:
    donor = clean_etb_vat_table(_raw_etb())
    draws = pd.DataFrame({"full_rate_vat_expenditure_rate": [-99.0, 99.0]})

    clipped = support_clip_to_donor(draws, donor)

    assert clipped["full_rate_vat_expenditure_rate"].tolist() == [
        donor["full_rate_vat_expenditure_rate"].min(),
        donor["full_rate_vat_expenditure_rate"].max(),
    ]
    assert donor_realized_ranges(donor) == {
        "full_rate_vat_expenditure_rate": (
            float(donor["full_rate_vat_expenditure_rate"].min()),
            float(donor["full_rate_vat_expenditure_rate"].max()),
        )
    }


def test_etb_vat_weighted_fit_record(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFitted:
        def predict(self, recipient):
            return pd.DataFrame(
                {"full_rate_vat_expenditure_rate": [0.1]}, index=recipient.index
            )

    class _FakeModel:
        def __init__(self, *, n_estimators, seed):
            assert n_estimators == 100
            assert seed == 0

        def fit(self, donor, predictors, targets, *, weights):
            assert weights == "weight"
            assert targets == ["full_rate_vat_expenditure_rate"]
            return _FakeFitted()

    import microcosm.fit as fit_module

    monkeypatch.setattr(fit_module, "RegimeGatedQRF", _FakeModel)

    donor = clean_etb_vat_table(_raw_etb())
    draws, record = impute_etb_vat(
        donor,
        pd.DataFrame(
            {
                "is_adult": [1.0],
                "is_child": [0.0],
                "is_SP_age": [0.0],
                "household_net_income": [1.0],
            }
        ),
        seed=0,
    )

    assert draws["full_rate_vat_expenditure_rate"].tolist() == [0.1]
    assert record.fit_name == UK_ETB_VAT_FIT_NAME
    assert record.weight_kind == "explicit"


def test_recipient_predictors_aggregate_person_entities_to_household() -> None:
    # Regression for the licensed-build crash: is_adult / is_child / is_SP_age
    # materialize at person grain and must aggregate to household — direct
    # engine arrays fail on the person/household length mismatch.
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.etb_vat import recipient_predictors
    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    entities = {
        "is_adult": "person",
        "is_child": "person",
        "is_SP_age": "person",
        "household_net_income": "household",
    }
    values = {
        "is_adult": np.array([1.0, 1.0, 0.0, 1.0]),
        "is_child": np.array([0.0, 0.0, 1.0, 0.0]),
        "is_SP_age": np.array([0.0, 1.0, 0.0, 0.0]),
        "household_net_income": np.array([1e4, 2e4]),
    }

    class _FakeEngine:
        country = "uk"

        def variable_metadata(self, name):
            return SimpleNamespace(entity=entities[name])

        def materialize(self, frame, variables, period):
            return {variable: values[variable] for variable in variables}

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [10, 10, 10, 20],
            "person_benunit_id": [100, 100, 100, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [100, 200], "benunit_household_id": [10, 20]})
    household = pd.DataFrame(
        {"household_id": [10, 20], "household_weight": [1.0, 1.0]}
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )

    result = recipient_predictors(frame, _FakeEngine())

    assert result["is_adult"].tolist() == [2.0, 1.0]
    assert result["is_child"].tolist() == [1.0, 0.0]
    assert result["is_SP_age"].tolist() == [1.0, 0.0]
    assert result["household_net_income"].tolist() == [1e4, 2e4]
