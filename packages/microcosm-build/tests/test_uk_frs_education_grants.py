from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_education_grants import (
    UKDSAPolicy,
    allocate_reported_education_grants,
    disabled_students_allowance_capacity,
)


def test_grant_allocator_caps_fraction_and_keeps_residual() -> None:
    result = allocate_reported_education_grants(
        [60.0, 300.0],
        {
            "childcare_grant": np.array([100.0, 100.0]),
            "parents_learning_allowance": np.array([100.0, 0.0]),
            "adult_dependants_grant": np.array([0.0, 0.0]),
            "dsa": np.array([0.0, 50.0]),
        },
    )

    assert result["childcare_grant"].tolist() == [30.0, 100.0]
    assert result["parents_learning_allowance"].tolist() == [30.0, 0.0]
    assert result["dsa"].tolist() == [0.0, 50.0]
    assert result["education_grants"].tolist() == [0.0, 150.0]


def test_pre_2025_dsa_capacity_is_aligned_zero_vector() -> None:
    person = pd.DataFrame(index=[10, 20, 30])

    result = disabled_students_allowance_capacity(
        person,
        capacities={},
        policy=UKDSAPolicy(maximum=100.0, instant="2023-01-01", source="fixture"),
        year=2023,
    )

    assert result.shape == (3,)
    assert result.tolist() == [0.0, 0.0, 0.0]
