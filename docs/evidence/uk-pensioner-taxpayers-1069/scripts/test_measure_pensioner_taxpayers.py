"""Properties of the recipient partition (microcosm#1069).

Run with the worktree venv:
    .venv/bin/python -m pytest -q docs/evidence/uk-pensioner-taxpayers-1069/scripts
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent))

from measure_pensioner_taxpayers import weighted_partition  # noqa: E402

rows = st.lists(
    st.tuples(
        st.booleans(),
        st.booleans(),
        st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False),
    ),
    min_size=0,
    max_size=200,
)


@settings(max_examples=200, deadline=None)
@given(rows)
def test_taxpaying_and_non_taxpaying_sum_to_the_recipients(data) -> None:
    recipient = np.array([row[0] for row in data], dtype=bool)
    taxpayer = np.array([row[1] for row in data], dtype=bool)
    weight = np.array([row[2] for row in data], dtype=float)
    split = weighted_partition(recipient, taxpayer, weight)
    total = split["recipients"]
    assert split["taxpaying"] + split["non_taxpaying"] == pytest.approx(
        total, rel=1e-12, abs=1e-6
    )
    assert 0 <= split["taxpaying"] <= total + 1e-6
    assert 0 <= split["non_taxpaying"] <= total + 1e-6
    # Only recipients count: non-recipients' weights move nothing.
    shifted = weight + np.where(recipient, 0.0, 5.0)
    assert weighted_partition(recipient, taxpayer, shifted) == split


def test_the_partition_refuses_negative_or_misaligned_weights() -> None:
    with pytest.raises(ValueError):
        weighted_partition([True], [True], [-1.0])
    with pytest.raises(ValueError):
        weighted_partition([True, False], [True], [1.0, 1.0])


def test_a_hand_computed_partition() -> None:
    split = weighted_partition(
        [True, True, True, False], [True, False, True, True], [2.0, 3.0, 5.0, 7.0]
    )
    assert split == {"recipients": 10.0, "taxpaying": 7.0, "non_taxpaying": 3.0}
