"""Analytical geography-support expectations retained independently of the CLI."""

from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime import expected_uk_rowwise_area_support


def _household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "WALES"],
        }
    )


def _crosswalk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": "E0001",
                "lsoa_code": "E0101",
                "msoa_code": "E0201",
                "la_code": "E06000063",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 75,
            },
            {
                "oa_code": "E0002",
                "lsoa_code": "E0102",
                "msoa_code": "E0202",
                "la_code": "E06000064",
                "constituency_code": "E14000002",
                "region_code": "E12000007",
                "country": "England",
                "population": 25,
            },
            {
                "oa_code": "W0001",
                "lsoa_code": "W0101",
                "msoa_code": "W0201",
                "la_code": "W06000001",
                "constituency_code": "W07000041",
                "region_code": "W99999999",
                "country": "Wales",
                "population": 80,
            },
        ]
    )


def test_collision_free_expectation_matches_distribution_math() -> None:
    support = expected_uk_rowwise_area_support(
        _household_frame(),
        _crosswalk_frame(),
        n_clones=2,
    )
    rows = {
        (row.area_type, row.area_code): row.expected_rows
        for row in support.itertuples(index=False)
    }
    # Collision-free: one London household at K=2 splits 75/25 across the two
    # England constituencies; the Wales household lands on the single Welsh
    # one. (The real sampler's collision avoidance forces the two London
    # clones apart — the dry-run's realized support covers that.)
    assert rows[("constituency", "E14000001")] == pytest.approx(2 * 0.75)
    assert rows[("constituency", "E14000002")] == pytest.approx(2 * 0.25)
    assert rows[("constituency", "W07000041")] == pytest.approx(2.0)
    assert rows[("la", "E06000063")] == pytest.approx(1.5)
    assert rows[("la", "W06000001")] == pytest.approx(2.0)
    total_constituency = sum(
        value for (area_type, _), value in rows.items() if area_type == "constituency"
    )
    assert total_constituency == pytest.approx(2 * len(_household_frame()))


def test_collision_free_expectation_requires_covered_countries() -> None:
    crosswalk = _crosswalk_frame()
    england_only = crosswalk[crosswalk["country"] == "England"]
    with pytest.raises(ValueError, match="Wales"):
        expected_uk_rowwise_area_support(
            _household_frame(),
            england_only,
            n_clones=1,
        )
    support = expected_uk_rowwise_area_support(
        _household_frame(),
        england_only,
        n_clones=1,
        require_all_countries=False,
    )
    assert set(support["area_code"]) == {
        "E14000001",
        "E14000002",
        "E06000063",
        "E06000064",
    }
