"""US congressional district geography assignment tests."""

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import (
    SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
    assign_congressional_districts_to_households,
    congressional_district_assignment_summary,
    congressional_district_distribution_from_ledger_facts,
    with_household_congressional_districts,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def test_soi_cd_distribution_uses_state_total_only_for_at_large_states() -> None:
    distribution = congressional_district_distribution_from_ledger_facts(
        [
            _soi_cd_fact(
                value=2_104_760,
                geography_id="0400000US01",
                geography_level="state",
                source_row_id="al_total",
            ),
            _soi_cd_fact(
                value=315_360,
                geography_id="5001700US0101",
                geography_level="congressional_district",
                source_row_id="al_01",
            ),
            _soi_cd_fact(
                value=285_600,
                geography_id="5001700US0102",
                geography_level="congressional_district",
                source_row_id="al_02",
            ),
            _soi_cd_fact(
                value=390_590,
                geography_id="0400000US02",
                geography_level="state",
                source_row_id="ak_total",
            ),
        ]
    )

    assert distribution["state_fips"].tolist() == ["01", "01", "02"]
    assert distribution["congressional_district_geoid"].tolist() == [
        "0101",
        "0102",
        "0200",
    ]
    assert distribution["is_state_total_proxy"].tolist() == [False, False, True]


def test_cd_assignment_is_state_constrained_and_index_independent() -> None:
    household = pd.DataFrame(
        {"household_id": [10, 20, 30, 40], "state_fips": [1, 1, 2, 2]},
        index=[100, 200, 300, 400],
    )
    distribution = pd.DataFrame(
        {
            "state_fips": ["01", "01", "02"],
            "congressional_district_geoid": ["0101", "0102", "0200"],
            "sampling_weight": [3.0, 1.0, 9.0],
            "is_state_total_proxy": [False, False, True],
        }
    )

    assigned = assign_congressional_districts_to_households(
        household,
        distribution,
        seed=0,
    )

    assert assigned.index.tolist() == [100, 200, 300, 400]
    assert set(assigned.loc[[100, 200], "congressional_district_geoid"]).issubset(
        {101, 102}
    )
    assert assigned.loc[[300, 400], "congressional_district_geoid"].tolist() == [
        200,
        200,
    ]


def test_cd_assignment_refuses_missing_state_distribution() -> None:
    household = pd.DataFrame({"household_id": [1], "state_fips": [6]})
    distribution = pd.DataFrame(
        {
            "state_fips": ["01"],
            "congressional_district_geoid": ["0101"],
            "sampling_weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing state_fips"):
        assign_congressional_districts_to_households(household, distribution)


def test_with_household_congressional_districts_preserves_frame_mass() -> None:
    frame = _minimal_us_frame()
    distribution = pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "congressional_district_geoid": ["0101", "0200"],
            "sampling_weight": [1.0, 1.0],
        }
    )

    assigned = with_household_congressional_districts(
        frame,
        distribution,
        seed=0,
    )

    assert assigned.table("household")["congressional_district_geoid"].tolist() == [
        101,
        200,
    ]
    assert assigned.weights_for("household").values.tolist() == [100.0, 300.0]
    assert assigned.strata.tolist() == frame.strata.tolist()
    summary = congressional_district_assignment_summary(
        assigned.table("household"),
        distribution,
    )
    assert summary["applied"] is True
    assert summary["assigned_congressional_districts"] == 2


def _soi_cd_fact(
    *,
    value: float,
    geography_id: str,
    geography_level: str,
    source_row_id: str,
) -> dict[str, object]:
    return {
        "value": value,
        "source": {"source_sha256": "sha"},
        "observed_measure": {"source_measure_id": "return_count"},
        "geography": {"id": geography_id, "level": geography_level},
        "layout": {
            "record_set_id": SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
            "groupby_value_id": source_row_id,
            "source_row_id": source_row_id,
        },
        "dimensions": {"filing_status": "all", "income_range": "all"},
        "period": {"value": 2023},
    }


def _minimal_us_frame() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2], dtype="int64"),
                "person_household_id": np.asarray([1, 2], dtype="int64"),
                "person_tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "person_spm_unit_id": np.asarray([100, 200], dtype="int64"),
                "person_family_id": np.asarray([1000, 2000], dtype="int64"),
                "person_marital_unit_id": np.asarray([10000, 20000], dtype="int64"),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([1, 2], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": np.asarray([10, 20])}),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    strata = pd.Series(["asec_2024", "asec_2024"], name="stratum")
    return Frame(tables, US_SCHEMA, weights, strata)
