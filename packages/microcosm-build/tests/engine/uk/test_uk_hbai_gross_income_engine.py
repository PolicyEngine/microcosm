"""The survey-concept gross income follows the engine's HBAI definition.

The LCFS NEED band and the NTS income band read a household's HBAI net income
with the deductions policyengine-uk takes to reach it added back
(microcosm#1164). If the engine changes that list, the gross income must move
with it, so this test fails until ``UK_HBAI_GROSS_INCOME_DEDUCTIONS`` (with the
stages' declared ``materialize_rules_engine_predictors``) and
``UK_HBAI_NET_INCOME_SUBTRACTS_NOT_ADDED_BACK`` together match the engine's
list again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.household_income import (
    UK_HBAI_GROSS_INCOME_DEDUCTIONS,
    UK_HBAI_GROSS_INCOME_VARIABLES,
    UK_HBAI_NET_INCOME_SUBTRACTS_NOT_ADDED_BACK,
    UK_HBAI_NET_INCOME_VARIABLE,
    hbai_gross_income,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def test_gross_income_deductions_are_the_engines_hbai_subtracts() -> None:
    from policyengine_uk.variables.household.income.hbai_household_net_income import (
        HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS,
    )

    assert not set(UK_HBAI_GROSS_INCOME_DEDUCTIONS) & set(
        UK_HBAI_NET_INCOME_SUBTRACTS_NOT_ADDED_BACK
    )
    assert sorted(
        UK_HBAI_GROSS_INCOME_DEDUCTIONS + UK_HBAI_NET_INCOME_SUBTRACTS_NOT_ADDED_BACK
    ) == sorted(HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS)


def test_gross_income_reads_hbai_income_before_the_reset_to_zero() -> None:
    """policyengine-uk 2.132.4 resets negative HBAI income to zero (pe-uk#2402).

    The build reads the figure before the reset: the surveys' incomes are not
    reset, and net income plus the deductions is gross income only before it.
    A household with no income that pays maintenance has negative HBAI income
    and no gross income.
    """

    person = pd.DataFrame(
        {
            "person_id": [11, 21],
            "person_benunit_id": [1, 2],
            "person_household_id": [1, 2],
            "age": [40.0, 40.0],
            "employment_income": [0.0, 30_000.0],
            "maintenance_expenses": [5_000.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1, 2], "would_claim_uc": [False, False]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["WALES", "WALES"],
            "council_tax": [0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(2),
        weight_kind=WeightKind.DESIGN,
        time_period="2025",
    )
    engine = PolicyEngineUKEngine()
    materialized = engine.materialize(
        frame, [*UK_HBAI_GROSS_INCOME_VARIABLES, "hbai_household_net_income"], 2025
    )
    before = np.asarray(materialized[UK_HBAI_NET_INCOME_VARIABLE], dtype=float)
    after = np.asarray(materialized["hbai_household_net_income"], dtype=float)

    # The engine resets the first household's income and leaves the second's.
    assert before[0] == pytest.approx(-5_000.0)
    assert after[0] == 0.0
    assert before[1] == pytest.approx(after[1])
    assert after[1] > 0.0

    gross = hbai_gross_income(frame, engine, materialized)
    assert gross[0] == pytest.approx(0.0, abs=1e-6)
    assert gross[1] == pytest.approx(30_000.0)
