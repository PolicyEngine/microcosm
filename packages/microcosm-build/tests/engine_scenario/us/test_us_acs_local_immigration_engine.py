"""The ACS immigration receipt's five-year-bar statuses match the pinned engine.

The receipt's adjustment-lag sensitivity (microcosm#1052 review) counts ACS
persons assigned a status the engine's five-year bar applies to. These tests
derive that set from policyengine-us's own Medicaid parameters and check that
the engine flips eligibility at the bar on ``years_since_us_entry``.
"""

from __future__ import annotations

import numpy as np

from microcosm.build.us_runtime.acs_local_immigration import (
    FIVE_YEAR_BAR_STATUSES,
    FIVE_YEAR_BAR_YEARS,
)

_INSTANT = "2024-01-01"


def test_five_year_bar_statuses_are_the_engines_bar_subject_statuses() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    eligibility = CountryTaxBenefitSystem().parameters.gov.hhs.medicaid.eligibility
    eligible = set(eligibility.eligible_immigration_statuses(_INSTANT))
    exempt = set(eligibility.bar_exempt_immigration_statuses(_INSTANT))
    assert set(FIVE_YEAR_BAR_STATUSES) == eligible - {"CITIZEN"} - exempt
    assert eligibility.five_year_bar_years(_INSTANT) == FIVE_YEAR_BAR_YEARS


def test_the_engine_reads_the_entry_clock_at_the_bar() -> None:
    """An LPR at 4 years is barred from Medicaid, at 5 years is not."""

    from policyengine_us import Simulation

    people = {
        f"p{years}": {
            "age": {2024: 40},
            "immigration_status_str": {2024: "LEGAL_PERMANENT_RESIDENT"},
            "years_since_us_entry": {2024: float(years)},
        }
        for years in (4, 5)
    }
    members = list(people)
    situation = {
        "people": people,
        "households": {"h": {"members": members, "state_name": {2024: "TX"}}},
        "spm_units": {"s": {"members": members}},
        "families": {"f": {"members": members}},
        "tax_units": {name: {"members": [name]} for name in members},
        "marital_units": {name: {"members": [name]} for name in members},
    }
    simulation = Simulation(situation=situation)
    eligible = np.asarray(
        simulation.calculate("is_medicaid_immigration_status_eligible", 2024)
    )
    assert eligible.tolist() == [False, True]
