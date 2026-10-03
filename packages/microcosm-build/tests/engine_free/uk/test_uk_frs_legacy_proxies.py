from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.build.uk_runtime.frs_employment import (
    derive_employment_status_from_frs,
    derive_frs_employment,
)
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


@pytest.mark.parametrize("empstati", range(1, 12))
def test_esa_proxies_read_only_the_sick_or_disabled_codes(empstati) -> None:
    # A working-age adult with no hours: only EMPSTATI 9 (permanently
    # sick/disabled) and 10 (temporarily sick/injured) are ESA health states,
    # and only 9 is in the support group. Code 11 (other inactive) is neither.
    status = derive_frs_employment(
        pd.DataFrame({"person_id": [1]}),
        pd.DataFrame(
            {"person_id": [1], "empstati": [empstati], "mjobsect": [0], "sic": [0]}
        ),
    )["employment_status"]

    result = derive_frs_legacy_proxies(
        _working_age_person(status.to_numpy(), hours=[0]),
        employment_status_reported=[True],
        state_pension_age=[66],
        max_annual_hours=16 * WEEKS_IN_YEAR,
    )

    assert result["esa_health_condition_proxy"].tolist() == [empstati in (9, 10)]
    assert result["esa_support_group_proxy"].tolist() == [empstati == 9]
    assert result["legacy_jobseeker_proxy"].tolist() == [empstati == 5]


@given(
    st.lists(
        st.tuples(
            st.integers(0, 100),  # age
            st.sampled_from(range(1, 12)),  # EMPSTATI
            st.integers(60, 68),  # State Pension age
            st.integers(0, 3_000),  # annual hours worked
            st.booleans(),  # EMPSTATI reported
        ),
        min_size=1,
        max_size=60,
    )
)
def test_esa_proxies_follow_the_corrected_statuses(rows) -> None:
    age, codes, spa, hours, reported = map(np.array, zip(*rows, strict=True))
    status = derive_employment_status_from_frs(codes, np.ones(len(rows), bool))
    person = _working_age_person(status, hours=hours).assign(age=age)

    result = derive_frs_legacy_proxies(
        person,
        employment_status_reported=reported,
        state_pension_age=spa,
        max_annual_hours=16 * WEEKS_IN_YEAR,
    )
    health = result["esa_health_condition_proxy"].to_numpy()
    support = result["esa_support_group_proxy"].to_numpy()

    working_age = (age >= 16) & (age < spa)
    assert (
        health.tolist() == (reported & working_age & np.isin(codes, (9, 10))).tolist()
    )
    assert (
        support.tolist()
        == (reported & working_age & (codes == 9) & (hours <= 0)).tolist()
    )
    assert not (support & ~health).any()
    assert not health[codes == 11].any()


def _working_age_person(status, *, hours) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.full(len(status), 30),
            "employment_status": status,
            "hours_worked": hours,
            "current_education": "NOT_IN_EDUCATION",
        }
    )
