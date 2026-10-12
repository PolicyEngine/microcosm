"""Synthetic FRS and SPI tables for the SPI housing shell tests.

FRS households' housing follows their income; SPI copies carry high incomes
and a parent's housing, so a stage that conditions on the recipient's own
income moves them into the housing that income implies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.spi_housing_shell import RENTED_TENURES

_REGIONS = ("LONDON", "NORTH_EAST", "NORTHERN_IRELAND")
_TYPES = ("couple_no_children_households", "lone_households_under_65")


def housing_shell_tables(n_frs: int = 600, n_spi: int = 200, seed: int = 0):
    """FRS households whose housing follows income; SPI copies with a parent's housing."""

    rng = np.random.default_rng(seed)
    n = n_frs + n_spi
    household_id = np.arange(1, n + 1)
    channel = np.array(["frs"] * n_frs + ["spi"] * n_spi)
    region = rng.choice(_REGIONS, size=n)
    household_type = rng.choice(_TYPES, size=n)
    single = household_type == "lone_households_under_65"
    # FRS incomes are modest; SPI copies carry high incomes.
    income = np.where(
        channel == "frs", rng.lognormal(10.2, 0.8, n), rng.lognormal(12.5, 0.6, n)
    )
    high = income > np.quantile(income[:n_frs], 0.6)
    tenure = np.where(
        high,
        rng.choice(["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE"], size=n),
        rng.choice(list(RENTED_TENURES), size=n),
    )
    accommodation = np.where(high, "HOUSE_DETACHED", "FLAT")
    rented = np.isin(tenure, RENTED_TENURES)
    mortgaged = tenure == "OWNED_WITH_MORTGAGE"
    ni = region == "NORTHERN_IRELAND"
    household = pd.DataFrame(
        {
            "household_id": household_id,
            "region": region,
            "ons_household_type": household_type,
            "council_tax_single_adult_raw": np.where(single, 1, 2),
            "household_support_channel": channel,
            "tenure_type": tenure,
            "accommodation_type": accommodation,
            "num_bedrooms": np.where(high, 4, 2).astype(np.int64),
            "council_tax_band": np.where(high, "F", "B"),
            "council_tax": np.where(ni, 0.0, np.where(high, 2800.0, 1300.0)),
            "council_tax_reported": np.where(ni, 0.0, np.where(high, 2800.0, 1300.0)),
            "rent": np.where(rented, 0.1 * income + 3000.0, 0.0),
            "mortgage_interest_repayment": np.where(mortgaged, 4000.0, 0.0),
            "mortgage_capital_repayment": np.where(mortgaged, 6000.0, 0.0),
            "structural_insurance_payments": np.where(rented, 0.0, 300.0),
            "housing_service_charges": np.zeros(n),
            "water_and_sewerage_charges": np.full(n, 400.0),
            "domestic_rates": np.where(ni, 1000.0, 0.0),
            "subrent": np.zeros(n),
            "council_tax_rebate": np.zeros(n),
        }
    )
    # One or two people per household; the first is the reference person.
    rows = []
    for index, hid in enumerate(household_id):
        rows.append((hid, 1, 45, income[index]))
        if not single[index]:
            rows.append((hid, 0, 43, 0.0))
    person = pd.DataFrame(
        rows,
        columns=[
            "person_household_id",
            "is_household_head",
            "age",
            "employment_income",
        ],
    )
    person["person_id"] = np.arange(1, len(person) + 1)
    person["person_benunit_id"] = person["person_household_id"]
    person["is_household_head"] = person["is_household_head"].astype(bool)
    for column in (
        "self_employment_income",
        "private_pension_income",
        "savings_interest_income",
        "dividend_income",
        "property_income",
        "other_investment_income",
        "state_pension_reported",
        "council_tax_benefit_reported",
    ):
        person[column] = 0.0
    hrp_rented = person["person_household_id"].map(
        pd.Series(rented, index=household_id)
    )
    person["housing_benefit_reported"] = np.where(
        person["is_household_head"] & hrp_rented, 2500.0, 0.0
    )
    benunit = pd.DataFrame({"benunit_id": household_id})
    weights = np.where(channel == "frs", 10.0, 5.0)
    return person, benunit, household, weights
