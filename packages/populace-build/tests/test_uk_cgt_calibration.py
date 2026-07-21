"""Tests for the UK capital gains constraint-frame materialization.

The declared HMRC facts only bind if the prepared person columns mean what the
facts say they mean. The threshold is policy-dependent, so these tests pin the
per-period annual exempt amount, the refusal to guess for an unmapped period,
and the two error paths that would otherwise calibrate a positive fact to a
zero row.
"""

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime.cgt_calibration import (
    UK_CGT_GAINS_AMOUNT_COLUMN,
    UK_CGT_TAXPAYER_COUNT_COLUMN,
    materialize_uk_cgt_calibration_frame,
    uk_cgt_annual_exempt_amount,
)
from populace.build.uk_runtime.national_build import UKNationalDataset
from populace.frame import WeightKind


def _dataset(gains, *, time_period="2023", drop_gains=False, weights=None):
    ids = np.arange(len(gains), dtype=int)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "capital_gains": np.asarray(gains, dtype=float),
        }
    )
    if drop_gains:
        person = person.drop(columns=["capital_gains"])
    return UKNationalDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": (
                    np.ones(len(ids)) if weights is None else np.asarray(weights, float)
                ),
            }
        ),
        time_period=time_period,
        household_weight_kind=WeightKind.IMPORTANCE,
    )


@pytest.mark.parametrize(
    ("period", "expected"),
    [("2022", 12_300.0), ("2023", 6_000.0), ("2024", 3_000.0)],
)
def test_annual_exempt_amount_tracks_the_period(period, expected):
    """The AEA moved £12,300 -> £6,000 -> £3,000; the threshold must follow."""
    assert uk_cgt_annual_exempt_amount(period) == expected


def test_unmapped_period_raises_rather_than_defaulting():
    """A wrong threshold silently changes what the taxpayer-count fact means."""
    with pytest.raises(ValueError, match="No reviewed CGT annual exempt amount"):
        uk_cgt_annual_exempt_amount("1999")


def test_default_threshold_comes_from_the_dataset_period():
    """2022-23 and 2023-24 must not classify the same person identically."""
    gains = [0.0, 8_000.0, 100_000.0]
    assert (
        materialize_uk_cgt_calibration_frame(
            _dataset(gains, time_period="2023")
        ).taxpayer_rows
        == 2
    )
    assert (
        materialize_uk_cgt_calibration_frame(
            _dataset(gains, time_period="2022")
        ).taxpayer_rows
        == 1
    )


def test_measure_columns_are_indicator_and_gated_amount():
    """Counts are 0/1; amounts are zeroed off-support, not merely copied."""
    result = materialize_uk_cgt_calibration_frame(
        _dataset([0.0, 5_000.0, 20_000.0, 100_000.0])
    )
    person = result.frame.table("person")
    assert person[UK_CGT_TAXPAYER_COUNT_COLUMN].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert person[UK_CGT_GAINS_AMOUNT_COLUMN].tolist() == [
        0.0,
        0.0,
        20_000.0,
        100_000.0,
    ]
    assert result.annual_exempt_amount == 6_000.0
    assert result.taxpayer_rows == 2
    assert result.minimum_positive_support_rows == 2


def test_explicit_threshold_overrides_the_period_default():
    result = materialize_uk_cgt_calibration_frame(
        _dataset([0.0, 5_000.0, 20_000.0]), annual_exempt_amount=10_000.0
    )
    assert result.annual_exempt_amount == 10_000.0
    assert result.taxpayer_rows == 1


def test_frame_carries_person_rows_and_household_weights():
    """Constraint rows are person-grain; the calibrated mass stays household."""
    result = materialize_uk_cgt_calibration_frame(
        _dataset([0.0, 20_000.0], weights=[3.0, 5.0])
    )
    assert set(result.frame.entities) == {"person", "household"}
    weights = result.frame.resolve_weights("household")
    assert weights.kind is WeightKind.IMPORTANCE
    assert np.asarray(weights.values).tolist() == [3.0, 5.0]


def test_registry_contains_exactly_the_two_declared_facts():
    result = materialize_uk_cgt_calibration_frame(_dataset([0.0, 20_000.0]))
    assert result.registry.country == "uk"
    assert {spec.name for spec in result.registry} == {
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
    }


def test_missing_capital_gains_column_fails_loudly():
    with pytest.raises(ValueError, match="capital_gains"):
        materialize_uk_cgt_calibration_frame(_dataset([0.0, 1.0], drop_gains=True))


def test_zero_support_refuses_to_calibrate_a_positive_fact():
    """Every person below the AEA means the £65.9bn fact has no support."""
    with pytest.raises(ValueError, match="no strictly positive-mass support"):
        materialize_uk_cgt_calibration_frame(_dataset([0.0, 100.0, 5_999.0]))
