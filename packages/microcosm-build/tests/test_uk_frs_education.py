from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_education import (
    EDUCQUAL_MAP,
    derive_current_education,
    derive_frs_education,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def test_current_education_cascade_keeps_dead_post_secondary_shadowed() -> None:
    result = derive_current_education(
        fted=np.array([2, 1, 1, 1, 1, 1]),
        typeed2=np.array([9, 1, 2, 5, 7, 9]),
        age=np.array([20, 4, 10, 16, 19, 20]),
    )

    assert result.tolist() == [
        "NOT_IN_EDUCATION",
        "PRE_PRIMARY",
        "PRIMARY",
        "LOWER_SECONDARY",
        "UPPER_SECONDARY",
        "TERTIARY",
    ]


def test_educqual_map_pins_the_corrected_frs_value_labels() -> None:
    # Signed difference vs the incumbent, whose map was inverted (it read
    # low codes as school-level and 17-21 as degrees). Labels verified
    # against the FRS 2023-24 data dictionary (UKDS SN 9367, DOI
    # 10.5255/UKDA-SN-9367-2, adult table) and corroborated by the raw
    # aggregates: code 1 (Doctorate) is ~1.8% of adults at the highest mean
    # earnings; 18/19 are near-empty baccalaureates; the GCSE band (36-82)
    # carries the mass. Code 87 is undocumented in the dictionary and is
    # deliberately unmapped, falling to the fillna default.
    assert EDUCQUAL_MAP[1] == "TERTIARY"
    assert EDUCQUAL_MAP[3] == "TERTIARY"
    assert EDUCQUAL_MAP[11] == "POST_SECONDARY"
    assert EDUCQUAL_MAP[17] == "UPPER_SECONDARY"
    assert EDUCQUAL_MAP[26] == "POST_SECONDARY"
    assert EDUCQUAL_MAP[36] == "LOWER_SECONDARY"
    assert EDUCQUAL_MAP[82] == "LOWER_SECONDARY"
    assert EDUCQUAL_MAP[85] == "NOT_COMPLETED_PRIMARY"
    assert EDUCQUAL_MAP[86] == "LOWER_SECONDARY"
    assert 87 not in EDUCQUAL_MAP
    assert set(EDUCQUAL_MAP) == set(range(1, 87))


def test_training_qyp_ema_and_benefits_in_own_right() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "age": [19, 17, 20],
            "universal_credit_reported": [0, 1, 0],
            "jsa_contrib_reported": [0, 0, 0],
            "jsa_income_reported": [0, 0, 0],
            "esa_contrib_reported": [0, 0, 0],
            "esa_income_reported": [0, 0, 0],
        }
    )
    raw = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "fted": [1, 2, 2],
            "typeed2": [7, 0, 0],
            "educqual": [17, 86, 999],
            "train": [9, 10, np.nan],
            # 2023-24-shaped vintage: no adema pair, eduma/edumaamt instead.
            "eduma": [1, 1, 0],
            "edumaamt": [2.0, -1.0, 5.0],
            "chema": [0, 1, 1],
            "chemaamt": [0.0, 3.0, -1.0],
        }
    )

    result = derive_frs_education(person, raw)

    assert result["is_in_approved_training"].tolist() == [True, False, False]
    assert result[
        "age_started_or_accepted_current_education_or_training"
    ].tolist() == [18, 1000, 1000]
    assert result[
        "is_before_universal_credit_qualifying_young_person_terminal_date"
    ].tolist() == [True, False, False]
    # fill_with_mean semantics: a participant's sentinel −1 receives the
    # participant mean (2.0 adult / 3.0 child); a non-participant keeps a
    # positive reported amount (no flag-gating of reported values).
    assert result["adult_ema"].tolist() == [
        2.0 * WEEKS_IN_YEAR,
        2.0 * WEEKS_IN_YEAR,
        5.0 * WEEKS_IN_YEAR,
    ]
    assert result["child_ema"].tolist() == [
        0.0,
        3.0 * WEEKS_IN_YEAR,
        3.0 * WEEKS_IN_YEAR,
    ]
    assert result["receives_benefits_in_own_right"].tolist() == [False, True, False]


def test_adult_ema_prefers_the_adema_pair_when_present() -> None:
    person = pd.DataFrame({"person_id": [1, 2], "age": [17, 18]})
    raw = pd.DataFrame(
        {
            "person_id": [1, 2],
            "fted": [2, 2],
            "typeed2": [0, 0],
            "educqual": [3, 3],
            # adema pair present: it must win over a contradictory eduma pair
            # (the incumbent aliases eduma into adema only when adema is
            # absent).
            "adema": [1, 1],
            "ademaamt": [4.0, -1.0],
            "eduma": [1, 1],
            "edumaamt": [9.0, 9.0],
            "chema": [0, 0],
            "chemaamt": [0.0, 0.0],
        }
    )

    result = derive_frs_education(person, raw)

    assert result["adult_ema"].tolist() == [
        4.0 * WEEKS_IN_YEAR,
        4.0 * WEEKS_IN_YEAR,
    ]
