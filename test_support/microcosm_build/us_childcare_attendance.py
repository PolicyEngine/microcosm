# ruff: noqa: F401
"""Behavioral contracts for the opt-in childcare attendance donor primitive.

All donors below are synthetic test records, not survey estimates.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    impute_us_childcare_attendance,
)
from microcosm.frame import WeightKind, Weights

MONTH, DAYS, HOURS = US_CHILDCARE_ATTENDANCE_COLUMNS


def _donors():
    return pd.DataFrame(
        {
            "donor_id": ["none", "part", "full"],
            "age": [3, 3, 3],
            MONTH: [0, 13, 22],
            DAYS: [0.0, 3.0, 5.0],
            HOURS: [0.0, 4.0, 8.0],
        }
    )


def _people(count=1):
    return pd.DataFrame(
        {"person_source_id": [f"c:{i}" for i in range(count)], "age": [3] * count}
    )


def _impute(person, donor=None, weights=None, **kwargs):
    donor = _donors() if donor is None else donor
    return impute_us_childcare_attendance(
        person,
        donor,
        donor_weights=Weights(
            np.ones(len(donor)) if weights is None else np.asarray(weights),
            WeightKind.DESIGN,
        ),
        match_columns=kwargs.pop("match_columns", ("age",)),
        seed=kwargs.pop("seed", 42),
        **kwargs,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
