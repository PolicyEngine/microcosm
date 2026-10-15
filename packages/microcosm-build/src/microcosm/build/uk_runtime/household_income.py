"""Household gross income on the survey concept, from the engine's HBAI income.

The surveys whose income bands the UK stages match recipients to (LCFS p344p,
NEED's modelled gross household income, the NTS household income band) band
households on gross cash income: earnings, self-employment, investment and
pension income and cash benefits, before income tax, National Insurance,
council tax and pension contributions. policyengine-uk's
``household_gross_income`` is not that concept: it adds capital gains and the
imputed value of NHS, education and transport spending that its
``household_benefits`` counts (microcosm#1164). The engine's HBAI net income
with the deductions it takes to reach it added back is.

The ETB's disposable income (``disinc``) is that gross income less the direct
taxes: income tax, employees' National Insurance and council tax or Northern
Ireland domestic rates (``etb_disposable_income``). The engine's
``household_net_income`` is not that concept either: it also takes off the
indirect taxes and keeps the in-kind spending.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import Frame

#: The deductions policyengine-uk takes from HBAI income to reach
#: ``hbai_household_net_income`` (its ``HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS``),
#: less the ones the gross income does not add back; an engine-lane test fails
#: if the two lists together part from the engine's.
UK_HBAI_GROSS_INCOME_DEDUCTIONS = (
    "council_tax",
    "domestic_rates",
    "income_tax",
    "national_insurance",
    "student_loan_repayments",
    "employee_pension_contributions",
    "personal_pension_contributions",
    "maintenance_expenses",
    "external_child_payments",
    "LVT",
)
#: The engine's HBAI subtractions the gross income leaves subtracted.
#: ``trading_loss`` (policyengine-uk 2.131.0) is the year's self-employment
#: loss: the surveys' gross incomes count self-employment income net of losses,
#: so adding it back would band a loss-making household on its income before
#: the loss.
UK_HBAI_NET_INCOME_SUBTRACTS_NOT_ADDED_BACK = ("trading_loss",)
#: The engine variables a stage materializes to build the gross income.
UK_HBAI_GROSS_INCOME_VARIABLES = (
    "hbai_household_net_income",
    *UK_HBAI_GROSS_INCOME_DEDUCTIONS,
)
#: The direct taxes the ETB takes from gross income for disposable income.
UK_ETB_DIRECT_TAXES = (
    "income_tax",
    "national_insurance",
    "council_tax",
    "domestic_rates",
)


def household_level(
    frame: Frame, engine: object, variable: str, values: Any
) -> np.ndarray:
    """One materialized engine variable at household grain, persons summed."""

    household = frame.table("household")
    person = frame.table("person")
    declared = str(engine.variable_metadata(variable).entity)
    array = np.asarray(values)
    if declared == "household":
        return array.astype(float)
    if declared == "person":
        summed = (
            pd.Series(array.astype(float))
            .groupby(person["person_household_id"].to_numpy())
            .sum()
        )
        return summed.reindex(household["household_id"]).fillna(0.0).to_numpy()
    raise ValueError(f"unsupported entity {declared!r} for {variable}.")


def hbai_gross_income(frame: Frame, engine: object, materialized: Any) -> np.ndarray:
    """Household gross income: HBAI net income plus the engine's HBAI deductions.

    ``materialized`` holds every ``UK_HBAI_GROSS_INCOME_VARIABLES`` entry.
    """

    return sum(
        household_level(frame, engine, variable, materialized[variable])
        for variable in UK_HBAI_GROSS_INCOME_VARIABLES
    )


def etb_disposable_income(
    frame: Frame, engine: object, materialized: Any
) -> np.ndarray:
    """Household disposable income on the ETB's concept (``disinc``).

    Gross cash income (``hbai_gross_income``) less income tax, employees'
    National Insurance and council tax or domestic rates. ``materialized``
    holds every ``UK_HBAI_GROSS_INCOME_VARIABLES`` entry, the direct taxes
    among them.
    """

    gross = hbai_gross_income(frame, engine, materialized)
    return gross - sum(
        household_level(frame, engine, variable, materialized[variable])
        for variable in UK_ETB_DIRECT_TAXES
    )
