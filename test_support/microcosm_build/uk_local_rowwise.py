"""The three-household importance-weighted clone frame and its area
assignment, shared by the rowwise local solve tests, the full-graph
calibration tests and the terminal-node tests."""

from __future__ import annotations

import pandas as pd

from microcosm.build.uk_runtime import uk_national_frame
from microcosm.frame import WeightKind


def _clone_frame(weights=(1.0, 1.0, 1.0)):
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 103],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "household_weight": list(weights),
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )


def _assigned() -> pd.Series:
    return pd.Series(["E001", "E001", "S001"], index=[101, 102, 103])


__all__ = [name for name in globals() if not name.startswith("__")]
