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


def test_the_twin_scheme_record_sets_membership_both_ways() -> None:
    """A member twin's rate carries onto the row's own pay, even where the
    forest drew nothing; a twin with pay outside any employer scheme gives
    none, whatever the forest drew; only a twin without pay leaves the
    forest's draw, bounded by ASHE's top rate (María's ruling of 2026-10-20)."""

    employer, from_twin, bounded = twin_rate_employer_contributions(
        drawn=np.array([0.0, 4_000.0, 9_000.0, 50_000.0, 1_000.0]),
        pay=np.array([100_000.0, 60_000.0, 80_000.0, 100_000.0, 0.0]),
        twin_pay=np.array([20_000.0, 30_000.0, 0.0, 0.0, 25_000.0]),
        twin_employer=np.array([1_000.0, 0.0, 0.0, 0.0, 2_500.0]),
    )

    np.testing.assert_allclose(
        employer,
        [5_000.0, 0.0, 9_000.0, ASHE_EMPLOYER_RATE_CEILING * 100_000.0, 0.0],
    )
    np.testing.assert_array_equal(from_twin, [True, True, False, False, True])
    np.testing.assert_array_equal(bounded, [False, False, False, True, False])


def test_a_receipt_withholds_amounts_resting_on_fewer_than_ten_households() -> None:
    receipt = coherence_receipt(
        changed=np.array([True, True, False, True]),
        before=np.array([10.0, 20.0, 30.0, 40.0]),
        after=np.array([5.0, 10.0, 30.0, 0.0]),
        weights=np.ones(4),
        source_ids=np.array([1, 1, 2, 3]),
    )

    assert receipt == {
        "support": support_band(2),
        "weighted_before": "<10",
        "weighted_after": "<10",
    }


def test_a_receipt_gives_weighted_amounts_on_ten_or_more_households() -> None:
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
    pay and takes its employer contribution from its twin's scheme record,
    and its receipt counts each rule apart."""

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [10, 20, 30, 40],
            "employment_income": [100_000.0, 60_000.0, 50_000.0, 40_000.0],
            "employee_pension_contributions": [5_000.0, 70_000.0, 0.0, 1_000.0],
            "employer_pension_contributions": [0.0, 4_000.0, 20_000.0, 3_000.0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [10, 20, 30, 40],
            "household_weight": [1.0, 1.0, 1.0, 1.0],
            "household_source_id": [10, 20, 30, 40],
        }
    )
    recipients = pd.Series([True, True, True, False])

    out, receipt = _spi_pension_coherence(
        person.copy(),
        household,
        spi_people=recipients,
        twin_pay=np.array([20_000.0, 30_000.0, 0.0]),
        twin_employer=np.array([1_000.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        out["employer_pension_contributions"], [5_000.0, 0.0, 15_000.0, 3_000.0]
    )
    np.testing.assert_allclose(
        out["employee_pension_contributions"], [5_000.0, 60_000.0, 0.0, 1_000.0]
    )
    assert set(receipt) == {
        "employee_capped_at_pay",
        "employer_at_twin_rate",
        "employer_none_outside_twin_scheme",
        "employer_bounded_by_ashe_ceiling",
    }
    for rule in receipt.values():
        assert rule["support"] == support_band(1)
        assert rule["weighted_before"] == rule["weighted_after"] == "<10"
