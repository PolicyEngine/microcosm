"""policyengine-uk reads the stored trading allowance claim status (microcosm#1439).

The stage's own rules are tested engine-free in
tests/engine_free/uk/test_uk_trading_allowance_receipts.py. Here, on the locked
engine (2.174.1), the three statuses relieve what the stage's notes say, and the
status stays an engine input for every later lock bump (Vahid's items 1 and 3 on
microcosm#1439).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

STATUSES = ("CLAIMANT", "UNKNOWN", "NON_CLAIMANT")


def _relief(people: dict, period: int) -> list[float]:
    from policyengine_uk import Simulation

    names = list(people)
    situation = {
        "people": people,
        "benunits": {f"b_{name}": {"members": [name]} for name in names},
        "households": {f"h_{name}": {"members": [name]} for name in names},
    }
    simulation = Simulation(situation=situation)
    return np.asarray(
        simulation.calculate("trading_allowance_deduction", period), dtype=float
    ).tolist()


def test_the_claim_status_is_still_an_engine_input() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    status = system.variables["trading_allowance_claim_status"]
    assert not status.formulas
    assert {member.name for member in status.possible_values} == set(STATUSES)
    assert status.default_value.name == "UNKNOWN"
    receipts = system.variables["self_employment_gross_receipts"]
    assert not receipts.formulas
    assert receipts.default_value == 0


@pytest.mark.parametrize("period", [2025, 2028])
def test_a_small_profit_is_relieved_in_full_only_as_unknown(period: int) -> None:
    """A GBP 600 profit given for the year simulated: a CLAIMANT's is already net
    of that year's allowance and a NON_CLAIMANT's of larger expenses, so only the
    full-relief group's UNKNOWN relieves it, in full (ITTOIA s. 783AE)."""

    people = {
        status.lower(): {
            "age": {period: 40},
            "self_employment_income": {period: 600},
            "trading_allowance_claim_status": {period: status},
        }
        for status in STATUSES
    }
    assert _relief(people, period) == [0.0, 600.0, 0.0]


def _dataset(profit: float, status: str, years: tuple[int, ...]):
    from policyengine_uk.data import UKMultiYearDataset, UKSingleYearDataset

    def single(year: int):
        return UKSingleYearDataset(
            person=pd.DataFrame(
                {
                    "person_id": [1],
                    "person_benunit_id": [1],
                    "person_household_id": [1],
                    "age": [40],
                    "self_employment_income": [profit],
                    "trading_allowance_claim_status": [status],
                }
            ),
            benunit=pd.DataFrame({"benunit_id": [1]}),
            household=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
            fiscal_year=year,
        )

    return UKMultiYearDataset(datasets=[single(year) for year in years])


@pytest.mark.parametrize("period", [2028, 2032])
@pytest.mark.parametrize("profit", [1_500.0, 5_000.0])
def test_a_claimant_measured_in_2025_gains_the_raise_without_drift(
    period: int, profit: float
) -> None:
    """A claimant's 2025 profit is net of GBP 1,000, so at GBP 3,000 (from 6 April
    2027) they gain exactly min(profit, GBP 2,000), in 2027-28 and in 2031-32."""

    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=_dataset(profit, "CLAIMANT", (2025, period)))
    allowance = simulation.tax_benefit_system.parameters(
        period
    ).gov.hmrc.income_tax.allowances.trading_allowance
    assert allowance == 3_000
    relief = float(
        np.asarray(simulation.calculate("trading_allowance_deduction", period))[0]
    )
    assert relief == pytest.approx(min(profit, 2_000.0))


@pytest.mark.parametrize("profit", [3_000.0, 5_000.0])
def test_a_non_claimant_gains_nothing_from_the_raise(profit: float) -> None:
    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=_dataset(profit, "NON_CLAIMANT", (2025, 2028)))
    relief = float(
        np.asarray(simulation.calculate("trading_allowance_deduction", 2028))[0]
    )
    assert relief == 0.0
