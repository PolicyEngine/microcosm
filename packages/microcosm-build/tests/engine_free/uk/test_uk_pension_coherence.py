"""Pension contributions that cohere with the pay they come from (microcosm#1095)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.pension_coherence import (
    ASHE_EMPLOYER_RATE_CEILING,
    cap_net_pay_contributions,
    coherence_receipt,
    twin_rate_employer_contributions,
)
from microcosm.build.uk_runtime.receipt_counts import support_band
from microcosm.build.uk_runtime.spi_income import _spi_pension_coherence


def test_net_pay_contributions_are_capped_at_pay() -> None:
    capped, changed = cap_net_pay_contributions(
        np.array([500.0, 3_000.0, 200.0]), np.array([1_000.0, 2_000.0, -50.0])
    )

    np.testing.assert_array_equal(capped, [500.0, 2_000.0, 0.0])
    np.testing.assert_array_equal(changed, [False, True, True])


def test_the_net_pay_cap_refuses_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="one value per person"):
        cap_net_pay_contributions(np.zeros(2), np.zeros(3))


def test_the_forest_decides_membership_and_the_twin_rate_scales_the_amount() -> None:
    """A row the forest gave no employer contribution keeps none, whatever its
    twin's record; a member whose twin has a rate takes that rate times its
    own pay; any other member keeps its drawn amount, bounded by ASHE's top
    rate (María's ruling of 2026-10-21, after Vahid's review of
    microcosm#1332)."""

    employer, from_twin, bounded = twin_rate_employer_contributions(
        drawn=np.array([0.0, 4_000.0, 9_000.0, 50_000.0, 1_000.0, 3_000.0]),
        pay=np.array([100_000.0, 60_000.0, 80_000.0, 100_000.0, 0.0, 50_000.0]),
        twin_pay=np.array([20_000.0, 30_000.0, 0.0, 0.0, 25_000.0, 40_000.0]),
        twin_employer=np.array([1_000.0, 0.0, 0.0, 0.0, 2_500.0, 2_000.0]),
    )

    np.testing.assert_allclose(
        employer,
        [0.0, 4_000.0, 9_000.0, ASHE_EMPLOYER_RATE_CEILING * 100_000.0, 0.0, 2_500.0],
    )
    np.testing.assert_array_equal(from_twin, [False, False, False, False, True, True])
    np.testing.assert_array_equal(bounded, [False, False, False, True, False, False])


def test_a_receipt_withholds_amounts_resting_on_fewer_than_three_households() -> None:
    receipt = coherence_receipt(
        changed=np.array([True, True, False, True]),
        before=np.array([10.0, 20.0, 30.0, 40.0]),
        after=np.array([5.0, 10.0, 30.0, 0.0]),
        weights=np.ones(4),
        source_ids=np.array([1, 1, 2, 3]),
    )

    assert receipt == {
        "support": support_band(2),
        "weighted_before": "<3",
        "weighted_after": "<3",
    }


def test_a_receipt_gives_weighted_amounts_on_three_or_more_households() -> None:
    rows = 12
    receipt = coherence_receipt(
        changed=np.ones(rows, dtype=bool),
        before=np.full(rows, 10.0),
        after=np.full(rows, 4.0),
        weights=np.full(rows, 2.0),
        source_ids=np.arange(rows),
    )

    assert receipt["support"] == support_band(rows)
    assert receipt["weighted_before"] == pytest.approx(240.0)
    assert receipt["weighted_after"] == pytest.approx(96.0)


def test_the_spi_stage_applies_the_rule_to_its_recipients() -> None:
    """The SPI stage caps each recipient's net-pay contribution at its drawn
    pay, scales a member's employer contribution by its twin's rate, leaves a
    row the forest gave none at none, and counts each rule apart."""

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "person_household_id": [10, 20, 30, 40, 50],
            "employment_income": [100_000.0, 60_000.0, 50_000.0, 40_000.0, 30_000.0],
            "employee_pension_contributions": [5_000.0, 70_000.0, 0.0, 1_000.0, 0.0],
            "employer_pension_contributions": [
                2_000.0,
                4_000.0,
                20_000.0,
                3_000.0,
                0.0,
            ],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [10, 20, 30, 40, 50],
            "household_weight": [1.0] * 5,
            "household_source_id": [10, 20, 30, 40, 50],
        }
    )
    recipients = pd.Series([True, True, True, False, True])

    out, receipt = _spi_pension_coherence(
        person.copy(),
        household,
        spi_people=recipients,
        twin_pay=np.array([20_000.0, 30_000.0, 0.0, 25_000.0]),
        twin_employer=np.array([1_000.0, 0.0, 0.0, 2_500.0]),
    )

    np.testing.assert_allclose(
        out["employer_pension_contributions"],
        [5_000.0, 4_000.0, 15_000.0, 3_000.0, 0.0],
    )
    np.testing.assert_allclose(
        out["employee_pension_contributions"], [5_000.0, 60_000.0, 0.0, 1_000.0, 0.0]
    )
    assert set(receipt) == {
        "employee_capped_at_pay",
        "employer_at_twin_rate",
        "employer_bounded_by_ashe_ceiling",
    }
    for rule in receipt.values():
        assert rule["support"] == support_band(1)
        assert rule["weighted_before"] == rule["weighted_after"] == "<3"
