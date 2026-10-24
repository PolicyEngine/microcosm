"""Both household ladders carry separate engine and calibration counties."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.county_vintage import (
    COUNTY_FIPS_2023_COLUMN,
    county_fips_2023_from_blocks,
    county_fips_2023_from_tracts,
    load_ct_planning_region_by_tract,
)
from microcosm.build.us_runtime.geography_ladder import (
    UsBlockLadder,
    assign_us_geography_ladder,
)
from microcosm.build.us_runtime.puma_ladder import (
    UsPumaLadder,
    assign_us_puma_ladder,
)
from microcosm.data.stored_inputs import undefined_stored_inputs


def _puma_ladder() -> UsPumaLadder:
    hartford = int(
        next(t for t in load_ct_planning_region_by_tract() if t.startswith("09003"))
    )
    return UsPumaLadder(
        puma=np.array([600100, 900100]),
        puma_population=np.array([2.0, 6.0]),
        cd_overlap_puma=np.array([600100, 900100]),
        cd_overlap_cd=np.array([601, 901]),
        cd_overlap_population=np.array([2.0, 6.0]),
        county_overlap_puma=np.array([600100, 900100, 900100]),
        county_overlap_county=np.array([6037, 9001, 9003]),
        county_overlap_population=np.array([2.0, 4.0, 2.0]),
        tract_overlap_puma=np.array([600100, 900100, 900100, 900100]),
        tract_overlap_tract=np.array([6037000100, 9001070100, 9001110100, hartford]),
        tract_overlap_population=np.array([2.0, 1.0, 3.0, 2.0]),
        metadata={},
    )


@pytest.mark.parametrize("assign_tract", [False, True])
def test_puma_assignment_carries_complete_counties_and_conserves_mass(
    assign_tract,
) -> None:
    household = pd.DataFrame(
        {
            "household_id": np.arange(600),
            "state_fips": [6] * 100 + [9] * 500,
            "puma": [600100] * 100 + [900100] * 500,
            "population_weight": np.arange(1, 601),
        },
        index=np.arange(600) + 1000,
    )
    assigned = assign_us_puma_ladder(
        household, _puma_ladder(), seed=7, assign_tract=assign_tract
    )

    assert not assigned[COUNTY_FIPS_2023_COLUMN].isna().any()
    assert assigned.index.equals(household.index)
    pd.testing.assert_series_equal(
        assigned["population_weight"], household["population_weight"]
    )
    non_ct = assigned["state_fips"] != 9
    assert (
        assigned.loc[non_ct, COUNTY_FIPS_2023_COLUMN]
        == assigned.loc[non_ct, "county_fips"]
    ).all()
    assert set(assigned.loc[~non_ct, "county_fips"]) == {"09001", "09003"}
    ct = assigned.loc[~non_ct]
    assert (
        ct.groupby("county_fips")["population_weight"].sum().sum()
        == ct.groupby(COUNTY_FIPS_2023_COLUMN)["population_weight"].sum().sum()
        == household.loc[~non_ct, "population_weight"].sum()
    )
    if assign_tract:
        np.testing.assert_array_equal(
            assigned[COUNTY_FIPS_2023_COLUMN],
            county_fips_2023_from_tracts(assigned["tract_geoid"]),
        )
        assert (assigned["county_fips"] == assigned["tract_geoid"].str[:5]).all()
    else:
        assert "tract_geoid" not in assigned
        # The planning region must respect the drawn legacy county as well
        # as the PUMA: Hartford support cannot leak into Fairfield rows.
        fairfield = ct["county_fips"] == "09001"
        assert set(ct.loc[fairfield, COUNTY_FIPS_2023_COLUMN]) == {"09120", "09140"}


def test_block_assignment_derives_2023_county_without_changing_2020_ids() -> None:
    blocks = np.array([90010701001000, 90011101001000, 90010201011000, 60370001001000])
    ladder = UsBlockLadder(
        block_geoid=blocks,
        population=np.ones(4),
        congressional_district_geoid=np.array([901, 902, 903, 601]),
        sldu=np.array(["001"] * 4),
        sldl=np.array(["001"] * 4),
        place_fips=np.ones(4, dtype=int),
        cbsa_code=np.ones(4, dtype=int),
        metadata={},
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "state_fips": [9, 9, 9, 6],
            "congressional_district_geoid": [901, 902, 903, 601],
            "population_weight": [30, 20, 10, 40],
        }
    )
    assigned = assign_us_geography_ladder(household, ladder, seed=3)

    assert assigned["county_fips"].tolist() == ["09001"] * 3 + ["06037"]
    np.testing.assert_array_equal(
        assigned[COUNTY_FIPS_2023_COLUMN], county_fips_2023_from_blocks(blocks)
    )
    assert assigned[COUNTY_FIPS_2023_COLUMN].tolist() == [
        "09120",
        "09140",
        "09190",
        "06037",
    ]
    assert (assigned["tract_geoid"] == assigned["block_geoid"].str[:11]).all()
    assert assigned["population_weight"].equals(household["population_weight"])
    assert not assigned[COUNTY_FIPS_2023_COLUMN].isna().any()


def test_new_county_column_is_registered_stored_geography() -> None:
    assert (
        undefined_stored_inputs(
            [COUNTY_FIPS_2023_COLUMN], engine_variables={"county_fips"}
        )
        == ()
    )
