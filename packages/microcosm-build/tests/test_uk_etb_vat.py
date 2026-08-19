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
