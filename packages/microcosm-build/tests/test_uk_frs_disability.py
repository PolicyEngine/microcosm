from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_disability import (
    UKDWPBaselineDisabilityRates,
    UKDWPDisabilityFlagRates,
    derive_frs_disability,
    uk_dwp_baseline_disability_rates,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def _baseline() -> UKDWPBaselineDisabilityRates:
    return UKDWPBaselineDisabilityRates(
        aa_lower=10,
        aa_higher=20,
        dla_sc_lower=10,
        dla_sc_middle=20,
        dla_sc_higher=30,
        dla_m_lower=10,
        dla_m_higher=20,
        pip_m_standard=10,
        pip_m_enhanced=20,
        pip_dl_standard=10,
        pip_dl_enhanced=20,
        instant="2023-01-01",
        source="fixture",
    )


def _flags() -> UKDWPDisabilityFlagRates:
    return UKDWPDisabilityFlagRates(
        aa_higher=20,
        dla_sc_higher=30,
        pip_dl_enhanced=20,
        instant="2023-01-01",
        source="fixture",
    )


def test_disability_category_threshold_and_overwrite() -> None:
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                (20 - 1) * WEEKS_IN_YEAR,
                (20 - 1.01) * WEEKS_IN_YEAR,
            ],
            "dla_sc_reported": [0, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, baseline_rates=_baseline(), flag_rates=_flags()
    )

    assert result["aa_category"].tolist() == ["HIGHER", "LOWER"]


def test_disability_flag_operator_asymmetry_and_afcs() -> None:
    boundary = (30 - 1) * WEEKS_IN_YEAR
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [0, 0],
            "dla_sc_reported": [boundary, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
            "sda_reported": [0, 0],
            "incapacity_benefit_reported": [0, 0],
            "iidb_reported": [0, 0],
            "afcs_reported": [0, 1],
            "esa_contrib_reported": [0, 0],
            "esa_income_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, baseline_rates=_baseline(), flag_rates=_flags()
    )

    assert result["is_enhanced_disabled_for_benefits"].tolist() == [False, False]
    assert result["is_severely_disabled_for_benefits"].tolist() == [True, True]


def test_dwp_reader_split_is_value_bearing() -> None:
    # The readers construct the real engine's parameter tree; the wheel gate
    # and the us-extra CI lane run without policyengine-uk, so skip there.
    # The baseline-vs-plain split carries different VALUES, not just
    # provenance: the baseline clone escapes policyengine-uk's fiscal-year
    # conversion (April-2022-era weekly rates at build period 2023), while
    # the plain tree the incumbent's flags read is fiscal-2023-24. A licensed
    # head-to-head against the incumbent's own output caught the one-row
    # flag divergence when both readers used the raw-file January values.
    pytest.importorskip("policyengine_uk")
    from microcosm.build.uk_runtime.frs_disability import (
        uk_dwp_disability_flag_rates,
    )

    rates = uk_dwp_baseline_disability_rates(2023)
    flag_rates = uk_dwp_disability_flag_rates(2023)

    assert rates.instant == "2023-01-01"
    assert np.isfinite(rates.aa_lower)
    assert rates.aa_higher == pytest.approx(92.40)
    assert flag_rates.aa_higher == pytest.approx(101.75)
    assert flag_rates.dla_sc_higher > rates.dla_sc_higher
