from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_legacy_proxies import derive_frs_legacy_proxies
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def test_legacy_proxy_truth_table_and_jsa_hours_boundary() -> None:
    person = pd.DataFrame(
        {
            "age": [18, 18, 30, 30],
            "employment_status": [
                "UNEMPLOYED",
                "UNEMPLOYED",
                "SHORT_TERM_DISABLED",
                "LONG_TERM_DISABLED",
            ],
            "hours_worked": [
                15.99 * WEEKS_IN_YEAR,
                16 * WEEKS_IN_YEAR,
                0,
                0,
            ],
            "current_education": [
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
            ],
        }
    )

    result = derive_frs_legacy_proxies(
        person,
        employment_status_reported=[True, True, True, True],
        over_state_pension_age=np.array([False, False, False, False]),
        max_annual_hours=16 * WEEKS_IN_YEAR,
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [True, False, False, False]
    assert result["esa_health_condition_proxy"].tolist() == [False, False, True, True]
    assert result["esa_support_group_proxy"].tolist() == [False, False, False, True]


def test_legacy_proxies_follow_the_engines_state_pension_age_status() -> None:
    """Working age is the engine's is_SP_age, not a comparison of ages.

    From policyengine-uk#1899 the engine's per-person state_pension_age is a
    fractional age (66.0027 for a 66-year-old who reached it on their
    birthday), so a whole-year age compared with it would put most
    66-year-olds under State Pension age. The stage reads is_SP_age instead:
    two 66-year-olds with the same record, one over State Pension age and one
    not (as in 2026-27), get different proxies.
    """

    person = pd.DataFrame(
        {
            "age": [66, 66, 66, 66],
            "employment_status": [
                "UNEMPLOYED",
                "UNEMPLOYED",
                "LONG_TERM_DISABLED",
                "LONG_TERM_DISABLED",
            ],
            "hours_worked": [0, 0, 0, 0],
            "current_education": ["NOT_IN_EDUCATION"] * 4,
        }
    )

    result = derive_frs_legacy_proxies(
        person,
        employment_status_reported=[True] * 4,
        over_state_pension_age=np.array([False, True, False, True]),
        max_annual_hours=16 * WEEKS_IN_YEAR,
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [True, False, False, False]
    assert result["esa_health_condition_proxy"].tolist() == [False, False, True, False]
    assert result["esa_support_group_proxy"].tolist() == [False, False, True, False]

    # A fractional State Pension age or any other non-boolean is refused.
    with pytest.raises(ValueError, match="one boolean per person row"):
        derive_frs_legacy_proxies(
            person,
            employment_status_reported=[True] * 4,
            over_state_pension_age=np.full(4, 66.0027),
            max_annual_hours=16 * WEEKS_IN_YEAR,
        )
