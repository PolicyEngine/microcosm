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


def test_educqual_map_pins_pinned_incumbent_codes() -> None:
    assert EDUCQUAL_MAP[1] == "NOT_COMPLETED_PRIMARY"
    assert EDUCQUAL_MAP[17] == "TERTIARY"
    assert EDUCQUAL_MAP[22] == "POST_SECONDARY"
    assert EDUCQUAL_MAP[85] == "POST_SECONDARY"
    assert EDUCQUAL_MAP[86] == "UPPER_SECONDARY"


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
            "emaamt": [2.0, -1.0, 0.0],
            "chemaamt": [0.0, 3.0, 0.0],
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
    assert result["adult_ema"].tolist() == [2.0 * WEEKS_IN_YEAR, 0.0, 0.0]
    assert result["child_ema"].tolist() == [0.0, 3.0 * WEEKS_IN_YEAR, 0.0]
    assert result["receives_benefits_in_own_right"].tolist() == [False, True, False]
