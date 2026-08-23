from __future__ import annotations

import pandas as pd

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
        state_pension_age=[66, 66, 66, 66],
        max_annual_hours=16 * WEEKS_IN_YEAR,
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [True, False, False, False]
    assert result["esa_health_condition_proxy"].tolist() == [False, False, True, True]
    assert result["esa_support_group_proxy"].tolist() == [False, False, False, True]
