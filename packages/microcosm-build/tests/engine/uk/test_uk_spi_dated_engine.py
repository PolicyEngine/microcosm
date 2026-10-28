"""The dated SPI measures run on policyengine-uk at the fact's own tax year.

``spi_2023_*`` carries the dataset's 2024-25 inputs back to 2023-24 by the
engine's own projection rules, runs the annual income simulation there and
scales the additive measures by each person's population factor. The engine's
own forward projection returns the carried-back inputs as stored (the
resolver refuses otherwise), and these tests hold the measures to that year's
policy and the engine's own growth rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_dated_measures import (
    population_factor_for_people,
    population_factors,
)
from microcosm.frame import WeightKind

_PRIVATE_PENSION_GROWTH = (
    "gov.economic_assumptions.yoy_growth.obr.private_pension_index"
)
_EARNINGS_GROWTH = "gov.economic_assumptions.yoy_growth.obr.average_earnings"


def _resolver(tmp_path, **household):
    """Five people living alone, as in the annual income measures' test.

    Person 0 is paid all year; person 1 left a job in the year and claims
    contributory ESA since; person 2 started a job half way through; persons
    3 and 4 are pensioners, 3 retiring in the year.
    """

    ids = np.arange(5, dtype=np.int64)
    person = {
        "person_id": ids,
        "person_benunit_id": ids,
        "person_household_id": ids,
        "age": [45.0, 40.0, 30.0, 65.75, 75.0],
        "employment_income": [30_000.0, 0.0, 24_000.0, 0.0, 0.0],
        "self_employment_income": [0.0] * 5,
        "private_pension_income": [0.0, 0.0, 0.0, 9_000.0, 6_000.0],
        "state_pension_reported": [0.0, 0.0, 0.0, 11_500.0, 11_500.0],
        "esa_contrib_reported": [0.0, 4_000.0, 0.0, 0.0, 0.0],
        "is_benunit_head": [True] * 5,
        "is_parent": [False] * 5,
        "employment_income_annual_share": [1.0, 1.0, 0.5, 1.0, 1.0],
        "employment_income_other_spells": [0.0, 15_000.0, 0.0, 20_000.0, 0.0],
        "self_employment_income_annual_share": [1.0] * 5,
        "self_employment_income_other_spells": [0.0] * 5,
        "private_pension_income_annual_share": [1.0, 1.0, 1.0, 0.5, 1.0],
        "contributory_benefits_annual_share": [1.0, 0.5, 1.0, 1.0, 1.0],
    }
    frame = uk_national_frame(
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
                **household,
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )
    return UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )


def _measure(resolver, name):
    values, route = resolver.compute("person", name)
    return np.asarray(values, dtype=float), route


@pytest.fixture(scope="module")
def resolver(tmp_path_factory):
    return _resolver(tmp_path_factory.mktemp("spi_dated"))


def test_the_count_is_each_persons_population_factor(resolver) -> None:
    count, route = _measure(resolver, "spi_2023_person_count")
    assert route == "spi_dated_population_factor:2023"
    age = np.asarray(resolver.simulation.calculate("age", 2024), dtype=float)
    raw = resolver.simulation.calculate("gender", 2024)
    sex = raw.decode_to_str() if hasattr(raw, "decode_to_str") else np.asarray(raw)
    np.testing.assert_allclose(
        count, population_factor_for_people(age, sex, population_factors(2023, 2025))
    )
    # Fewer people of every pensioner's age band in 2023-24 than in mid-2025.
    assert (count[3:] < 1.0).all()


def test_private_pensions_are_carried_back_by_the_engines_index(resolver) -> None:
    count, _ = _measure(resolver, "spi_2023_person_count")
    pension, route = _measure(resolver, "spi_2023_private_pension_income")
    assert route.endswith(":population_factor")
    growth = float(
        resolver.simulation.tax_benefit_system.parameters.get_child(
            _PRIVATE_PENSION_GROWTH
        )("2024")
    )
    # The 2024-25 pensions, carried back a year, over each person's year:
    # the retiree drew half the year's pension (the fixture's annual share).
    np.testing.assert_allclose(
        pension,
        np.array([0.0, 0.0, 0.0, 9_000.0 * 0.5, 6_000.0]) / (1 + growth) * count,
        rtol=1e-6,
    )


def test_the_state_pension_is_paid_at_2023_24_rates(resolver) -> None:
    count, _ = _measure(resolver, "spi_2023_person_count")
    dated, _ = _measure(resolver, "spi_2023_state_pension")
    current = np.asarray(resolver.simulation.calculate("state_pension", 2024))
    ratio = dated[4] / count[4] / current[4]
    # The 2024-25 triple lock raised the basic and new State Pension rates by
    # 8.5% and the additional pension by September 2023's CPI, 6.7%.
    assert 1 / 1.085 - 1e-4 <= ratio <= 1 / 1.067 + 1e-4


def test_the_band_and_the_filter_read_2023_24_income_and_tax(resolver) -> None:
    total, route = _measure(resolver, "spi_2023_total_income")
    assert route.startswith("spi_dated_annual_simulation:2023:")
    tax, _ = _measure(resolver, "spi_2023_income_tax")
    count, _ = _measure(resolver, "spi_2023_person_count")
    state_pension, _ = _measure(resolver, "spi_2023_state_pension")
    pension, _ = _measure(resolver, "spi_2023_private_pension_income")
    # The pensioner's only incomes are the two pensions, measured unscaled.
    assert total[4] == pytest.approx((state_pension[4] + pension[4]) / count[4])
    # The worker's pay at 2023-24 is 2024-25's carried back by average
    # earnings growth, and taxed at 2023-24's 20% basic rate above the
    # personal allowance, frozen at GBP 12,570 in both years.
    growth = float(
        resolver.simulation.tax_benefit_system.parameters.get_child(_EARNINGS_GROWTH)(
            "2024"
        )
    )
    pay = 30_000.0 / (1 + growth)
    assert total[0] == pytest.approx(pay, rel=1e-6)
    tax_2024 = np.asarray(resolver.simulation.calculate("income_tax", 2024))
    assert tax_2024[0] == pytest.approx(0.2 * (30_000.0 - 12_570.0), rel=1e-6)
    assert tax[0] == pytest.approx(0.2 * (pay - 12_570.0), rel=1e-6)


def test_the_receipt_records_the_round_trip(resolver) -> None:
    for name in (
        "spi_2023_person_count",
        "spi_2023_state_pension",
        "spi_2023_private_pension_income",
        "spi_2023_total_income",
        "spi_2023_income_tax",
    ):
        _measure(resolver, name)
    receipt = resolver.receipt()["spi_dated_measures"]
    assert receipt["calibration_period"] == 2025
    year = receipt["years"]["2023"]
    assert year["dataset_year"] == 2024
    assert year["round_trip_checked_inputs"] > 0
    assert year["indexed_inputs"] > 0
    assert "unused_pension_annual_allowance" in year["dropped_year_specific_inputs"]
    assert year["residential_property_ratio"] > 1.0
    # A repayment is compared wherever the engine kept the person's plan.
    assert year["round_trip_compared_where_unchanged"] == {
        "student_loan_repayments": "student_loan_plan"
    }
    assert "student_loan_repayments" not in year["round_trip_exempt_inputs"]


def test_the_held_inputs_do_not_move_the_measures(tmp_path) -> None:
    """The inputs the round trip exempts are held at 2024-25: none is income.

    The engine projects council tax, rent and water charges by region after
    its index step, so the carried-back file does not return them; the dated
    measures read none of them, so holding them changes no measure.
    """

    base = _resolver(tmp_path / "base")
    moved = _resolver(
        tmp_path / "moved",
        council_tax=2_400.0,
        rent=12_000.0,
        water_and_sewerage_charges=500.0,
        tenure_type="RENT_PRIVATELY",
    )
    for name in (
        "spi_2023_total_income",
        "spi_2023_income_tax",
        "spi_2023_state_pension",
        "spi_2023_private_pension_income",
    ):
        np.testing.assert_allclose(_measure(moved, name)[0], _measure(base, name)[0])
    exempt = moved.receipt()["spi_dated_measures"]["years"]["2023"][
        "round_trip_exempt_inputs"
    ]
    # Each was exercised: the engine's forward rule moved it.
    for name in ("council_tax", "rent", "water_and_sewerage_charges"):
        assert exempt[name]["largest_miss"] > 0.0, name
