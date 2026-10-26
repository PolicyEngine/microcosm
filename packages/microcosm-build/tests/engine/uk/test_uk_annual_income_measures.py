"""The HMRC measures of each person's year run on policyengine-uk itself.

HMRC's income statistics count the income received over a tax year, and the
frame carries beside its current incomes the descriptors of each adult's year
(``frs_annual_income_spells``). The annual measures are the current measures
on a second simulation whose inputs are that year: the engine projects a
dataset's inputs to its later years as it loads it, so the year's inputs are
set at the measurement year and the other spells grow by the engine's own
factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.local_targets import compute_household_metrics
from microcosm.build.uk_runtime.measure_simulation import (
    UK_ANNUAL_INCOME_MEASURES,
    UKMeasureResolver,
    state_pension_annual_share,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind


def _frame(**descriptors):
    ids = np.arange(5, dtype=np.int64)
    person = {
        "person_id": ids,
        "person_benunit_id": ids,
        "person_household_id": ids,
        # Person 0 is paid all year; person 1 left a job in the year and
        # claims contributory ESA since; person 2 started a job half way
        # through; persons 3 and 4 are pensioners, 3 retiring in the year.
        "age": [45.0, 40.0, 30.0, 65.75, 75.0],
        "employment_income": [30_000.0, 0.0, 24_000.0, 0.0, 0.0],
        "self_employment_income": [0.0] * 5,
        "private_pension_income": [0.0, 0.0, 0.0, 9_000.0, 6_000.0],
        "state_pension_reported": [0.0, 0.0, 0.0, 11_500.0, 11_500.0],
        "esa_contrib_reported": [0.0, 4_000.0, 0.0, 0.0, 0.0],
        # Each lives alone: the FRS roles of five single-adult benefit units.
        "is_benunit_head": [True] * 5,
        "is_parent": [False] * 5,
        **descriptors,
    }
    return uk_national_frame(
        person=pd.DataFrame(person),
        benunit=pd.DataFrame({"benunit_id": ids, "dependent_children": 0}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [1.0] * 5,
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
                "main_residence_value": 250_000.0,
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


_DESCRIPTORS = {
    "employment_income_annual_share": [1.0, 1.0, 0.5, 1.0, 1.0],
    "employment_income_other_spells": [0.0, 15_000.0, 0.0, 20_000.0, 0.0],
    "self_employment_income_annual_share": [1.0] * 5,
    "self_employment_income_other_spells": [0.0] * 5,
    "private_pension_income_annual_share": [1.0, 1.0, 1.0, 0.5, 1.0],
    "contributory_benefits_annual_share": [1.0, 0.5, 1.0, 1.0, 1.0],
}


def test_the_annual_measures_tax_each_persons_year(tmp_path) -> None:
    resolver = UKMeasureResolver(
        simulation_source=None,
        frame=_frame(**_DESCRIPTORS),
        scratch_dir=tmp_path,
        year=2025,
    )
    pay, _ = resolver.compute("person", "employment_income")
    pay = np.asarray(pay, dtype=float)
    factor = pay[0] / 30_000.0
    assert factor > 1.0

    annual_pay, route = resolver.compute("person", "employment_income_annual")
    assert route.startswith("annual_income_simulation:")
    # The leaver's spell and the pensioner's last job grow by the engine's
    # own factor for pay; the entrant keeps half the year at today's rate.
    np.testing.assert_allclose(
        annual_pay,
        [pay[0], 15_000.0 * factor, pay[2] * 0.5, 20_000.0 * factor, 0.0],
        rtol=1e-6,
    )
    pension, _ = resolver.compute("person", "private_pension_income")
    annual_pension, _ = resolver.compute("person", "private_pension_income_annual")
    np.testing.assert_allclose(
        annual_pension, np.asarray(pension) * [1.0, 1.0, 1.0, 0.5, 1.0], rtol=1e-6
    )
    state_pension, _ = resolver.compute("person", "state_pension")
    months, _ = resolver.compute("person", "months_since_state_pension_age")
    annual_state_pension, _ = resolver.compute("person", "state_pension_annual")
    np.testing.assert_allclose(
        annual_state_pension,
        np.asarray(state_pension) * state_pension_annual_share(months),
        rtol=1e-6,
    )
    # Taxed by the engine on the year: the leaver's spell makes them a
    # taxpayer, and halving the entrant's pay lowers their tax.
    tax, _ = resolver.compute("person", "income_tax")
    annual_tax, _ = resolver.compute("person", "income_tax_annual")
    tax, annual_tax = np.asarray(tax, dtype=float), np.asarray(annual_tax, dtype=float)
    assert tax[1] == 0.0 and annual_tax[1] > 0.0
    assert annual_tax[2] < tax[2]
    assert annual_tax[0] == tax[0] and annual_tax[4] == tax[4]
    # Nobody here passes the Winter Fuel Payment charge's income test, so
    # the tax before the charge is the tax.
    pre_charge, _ = resolver.compute(
        "person", "income_tax_before_winter_fuel_payment_charge_annual"
    )
    np.testing.assert_allclose(pre_charge, annual_tax)
    total, _ = resolver.compute("person", "total_income_after_loss_relief_annual")
    assert np.asarray(total, dtype=float)[1] > 15_000.0 * factor
    profit, route = resolver.compute(
        "person", "self_employment_income_after_loss_relief_annual"
    )
    np.testing.assert_allclose(profit, 0.0)
    assert route.startswith("annual_income_simulation:engine_components:")
    band, _ = resolver.compute("person", "tax_band_annual")
    assert len(band) == 5
    receipt = resolver.receipt()["annual_income_measures"]
    assert receipt["route"] == "annual_income_simulation"
    factors = receipt["other_spells_engine_factors"]
    assert factors["employment_income_before_lsr"] == pytest.approx(factor, rel=1e-6)
    assert set(receipt["computed_measures"]) == set(UK_ANNUAL_INCOME_MEASURES)


def test_the_income_tax_rows_read_the_tax_before_the_winter_fuel_charge(
    tmp_path,
) -> None:
    """policyengine-uk adds the Winter Fuel Payment charge to income_tax from
    2.157.0. HMRC's statistics predate it, and the OBR window's receipts hold
    almost none of 2025's recovery, so those rows read the tax before it
    (María's ruling of 2026-10-25, microcosm#1095)."""

    frame = _frame(**_DESCRIPTORS)
    # The 75-year-old's pension takes their income past the charge's
    # GBP 35,000 test.
    frame.table("person").loc[4, "private_pension_income"] = 40_000.0
    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )

    annual_tax, _ = resolver.compute("person", "income_tax_annual")
    pre_charge, route = resolver.compute(
        "person", "income_tax_before_winter_fuel_payment_charge_annual"
    )

    annual_tax = np.asarray(annual_tax, dtype=float)
    pre_charge = np.asarray(pre_charge, dtype=float)
    assert route.startswith("annual_income_simulation:")
    assert pre_charge[4] < annual_tax[4]
    np.testing.assert_allclose(pre_charge[:4], annual_tax[:4])


def test_the_local_hmrc_rows_count_each_persons_year(tmp_path) -> None:
    resolver = UKMeasureResolver(
        simulation_source=None,
        frame=_frame(**_DESCRIPTORS),
        scratch_dir=tmp_path,
        year=2025,
    )

    metrics = compute_household_metrics(resolver, "constituency", period=2025)

    # The leaver is paid nothing now and pays no tax on it, but pays tax on
    # the year's spell, so their area counts an employee taxpayer and the
    # year's pay (each person is a household of their own).
    tax, _ = resolver.compute("person", "income_tax")
    annual_tax, _ = resolver.compute("person", "income_tax_annual")
    annual_pay, _ = resolver.compute("person", "employment_income_annual")
    assert np.asarray(tax, dtype=float)[1] == 0.0
    assert np.asarray(annual_tax, dtype=float)[1] > 0.0
    assert metrics["hmrc/employment_income/count"].iloc[1] == 1.0
    assert metrics["hmrc/employment_income/amount"].iloc[1] == pytest.approx(
        float(np.asarray(annual_pay)[1])
    )


def test_a_frame_without_descriptors_refuses_unless_it_is_the_incumbent(
    tmp_path,
) -> None:
    resolver = UKMeasureResolver(
        simulation_source=None, frame=_frame(), scratch_dir=tmp_path, year=2025
    )
    assert not resolver.knows("person", "income_tax_annual")
    resolver.use_current_income_for_annual_measures(reason="incumbent")
    annual_tax, route = resolver.compute("person", "income_tax_annual")
    tax, _ = resolver.compute("person", "income_tax")
    np.testing.assert_array_equal(annual_tax, tax)
    assert route.startswith("current_values_without_annual_income_descriptors:")
