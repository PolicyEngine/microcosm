"""Age-distribution payload assembly, isolated from policyengine-us."""

from __future__ import annotations

import json

import numpy as np
import pytest

from populace.build.us_runtime.demographics import (
    AGE_BANDS,
    DEMOGRAPHICS_SCHEMA_VERSION,
    compute_age_distribution,
    demographics_payload,
    write_demographics,
)


def test_bands_partition_ages_with_weighted_population():
    ages = np.array([2, 10, 40, 70, 90])
    weights = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    rows = compute_age_distribution(ages, weights, benchmark=None)
    by_label = {r["label"]: r for r in rows}
    assert by_label["0–4"]["population"] == 100.0
    assert by_label["5–17"]["population"] == 200.0
    assert by_label["35–44"]["population"] == 300.0
    assert by_label["65–74"]["population"] == 400.0
    assert by_label["75+"]["population"] == 500.0  # open-ended upper band
    # shares sum to 1 over the full population.
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)


def test_benchmark_relative_error_is_signed():
    ages = np.array([10, 10, 10])  # all in 5–17
    weights = np.array([20_000_000.0, 20_000_000.0, 20_000_000.0])  # 60M in 5–17
    rows = compute_age_distribution(ages, weights, benchmark={"5–17": 50_000_000})
    row = next(r for r in rows if r["label"] == "5–17")
    assert row["population"] == 60_000_000
    assert row["benchmark"] == 50_000_000
    assert row["relative_error"] == pytest.approx(0.2)  # 60M vs 50M = +20% over


def test_payload_shape_and_totals():
    ages = np.array([3, 30, 80])
    weights = np.array([10.0, 20.0, 30.0])
    payload = demographics_payload(
        ages, weights, period=2024, release_id="rel-a", benchmark=None
    )
    assert payload["schema_version"] == DEMOGRAPHICS_SCHEMA_VERSION
    assert payload["period"] == 2024
    assert payload["total_population"] == 60.0
    assert payload["release_id"] == "rel-a"
    assert len(payload["age_bands"]) == len(AGE_BANDS)


def test_write_round_trips(tmp_path):
    payload = demographics_payload(np.array([5]), np.array([1.0]), period=2024)
    path = write_demographics(payload, tmp_path / "demographics.json")
    loaded = json.loads(path.read_text())
    assert loaded["age_bands"][1]["label"] == "5–17"


def test_geography_coverage_counts_household_records_by_state_and_district(tmp_path):
    pytest.importorskip("tables")  # pandas HDF backend
    import pandas as pd

    from populace.build.us_runtime.demographics import geography_coverage_payload

    household = pd.DataFrame(
        {
            "household_id": range(7),
            "household_weight": [1.0] * 7,
            # Two AL households in AL-01, one in AL-02; four AK at-large.
            "state_fips": [1, 1, 1, 2, 2, 2, 2],
            "congressional_district_geoid": [101, 101, 102, 200, 200, 200, 200],
        }
    )
    path = tmp_path / "mini.h5"
    with pd.HDFStore(str(path)) as store:
        store.put("household", household, format="table")

    payload = geography_coverage_payload(path)
    assert payload["unit"] == "unweighted household records"
    states = payload["states"]
    assert states["counts"] == {"AL": 3, "AK": 4}
    assert states["n_geographies"] == 2
    assert states["n_under_50"] == 2
    districts = payload["congressional_districts"]
    assert districts["counts"] == {"AL-01": 2, "AL-02": 1, "AK-00": 4}
    assert districts["household_records_min"] == 1
    assert districts["household_records_max"] == 4
    assert districts["n_under_50"] == 3


def test_geography_coverage_without_district_column(tmp_path):
    pytest.importorskip("tables")  # pandas HDF backend
    import pandas as pd

    from populace.build.us_runtime.demographics import geography_coverage_payload

    household = pd.DataFrame(
        {
            "household_id": [0, 1],
            "household_weight": [1.0, 1.0],
            "state_fips": [1, 2],
        }
    )
    path = tmp_path / "mini.h5"
    with pd.HDFStore(str(path)) as store:
        store.put("household", household, format="table")

    payload = geography_coverage_payload(path)
    assert payload["congressional_districts"] is None
    assert payload["states"]["counts"] == {"AL": 1, "AK": 1}


def test_geography_coverage_payload_is_json_stable(tmp_path):
    pytest.importorskip("tables")  # pandas HDF backend
    import json

    import pandas as pd

    from populace.build.us_runtime.demographics import geography_coverage_payload

    household = pd.DataFrame(
        {
            "household_id": [0],
            "household_weight": [1.0],
            "state_fips": [49],
            "congressional_district_geoid": [4903],
        }
    )
    path = tmp_path / "mini.h5"
    with pd.HDFStore(str(path)) as store:
        store.put("household", household, format="table")

    payload = geography_coverage_payload(path)
    assert payload["congressional_districts"]["counts"] == {"UT-03": 1}
    json.dumps(payload, allow_nan=False)  # round-trips
