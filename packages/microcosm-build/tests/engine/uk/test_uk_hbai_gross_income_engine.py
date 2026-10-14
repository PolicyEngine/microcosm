"""The survey-concept gross income follows the engine's HBAI definition.

The LCFS NEED band and the NTS income band read a household's HBAI net income
with the deductions policyengine-uk takes to reach it added back
(microcosm#1164). If the engine changes that list, the gross income must move
with it, so this test fails until ``UK_HBAI_GROSS_INCOME_DEDUCTIONS`` and the
two stages' declared ``materialize_rules_engine_predictors`` match again.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.household_income import (
    UK_HBAI_GROSS_INCOME_DEDUCTIONS,
)


def test_gross_income_deductions_are_the_engines_hbai_subtracts() -> None:
    from policyengine_uk.variables.household.income.hbai_household_net_income import (
        HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS,
    )

    assert sorted(UK_HBAI_GROSS_INCOME_DEDUCTIONS) == sorted(
        HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS
    )
