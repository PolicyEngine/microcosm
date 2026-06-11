"""Shared fixture: a small person+household frame for stage-plan tests."""

import numpy as np
import pandas as pd
import pytest

from populace.frame import EntitySchema, Frame, WeightKind, Weights


@pytest.fixture
def small_frame() -> Frame:
    """Four persons in two households, with an income column."""
    person = pd.DataFrame(
        {
            "person_id": np.arange(4, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
            "income": np.asarray([100.0, 0.0, 250.0, 50.0]),
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([1, 2], dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=np.asarray([1000.0, 2000.0]), kind=WeightKind.DESIGN
            )
        },
    )
