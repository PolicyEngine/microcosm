from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    ROWWISE_GEOGRAPHY_COLUMNS,
    assign_household_geography,
    clone_entity_frame,
    geography_coverage_summary,
    id_multiplier_for_values,
    prepare_geography_crosswalk,
    validate_geography_coverage,
)


def crosswalk_rows() -> pd.DataFrame:
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
                "population": 100,
            },
            {
                "oa_code": "E0002",
                "lsoa_code": "E0102",
                "msoa_code": "E0202",
                "la_code": "E06000064",
                "constituency_code": "E14000002",
                "region_code": "E12000001",
                "country": "England",
                "population": 200,
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
            {
                "oa_code": "S0001",
                "lsoa_code": "S0101",
                "msoa_code": "S0201",
                "la_code": "S12000033",
                "constituency_code": "S14000001",
                "region_code": "S99999999",
                "country": "Scotland",
                "population": 90,
            },
            {
                "oa_code": "N20000001",
                "lsoa_code": "N20000001",
                "msoa_code": "N21000001",
                "la_code": "N09000001",
                "constituency_code": "N05000001",
                "region_code": "N99999999",
                "country": "Northern Ireland",
                "population": 70,
            },
        ]
    )


def test_prepare_geography_crosswalk_requires_constituencies() -> None:
    crosswalk = crosswalk_rows()
    crosswalk.loc[crosswalk["country"] == "Northern Ireland", "constituency_code"] = ""

    with pytest.raises(ValueError, match="constituency_code.*blank"):
        prepare_geography_crosswalk(crosswalk)


def test_coverage_summary_tolerates_blank_constituency_when_non_strict() -> None:
    crosswalk = crosswalk_rows()
    crosswalk.loc[crosswalk["country"] == "Northern Ireland", "constituency_code"] = (
        np.nan
    )

    summary = geography_coverage_summary(
        crosswalk,
        {"constituency": ["N05000001"]},
    )

    row = summary.iloc[0]
    assert int(row["covered_areas"]) == 0
    assert row["missing_area_codes"] == ("N05000001",)


def test_validate_geography_coverage_flags_missing_ni_and_new_las() -> None:
    stale_crosswalk = crosswalk_rows()
    stale_crosswalk = stale_crosswalk[
        ~stale_crosswalk["country"].eq("Northern Ireland")
    ].copy()
    stale_crosswalk = stale_crosswalk[~stale_crosswalk["la_code"].eq("E06000064")]

    with pytest.raises(ValueError, match="Northern Ireland"):
        validate_geography_coverage(
            stale_crosswalk,
            required_countries=[
                "England",
                "Wales",
                "Scotland",
                "Northern Ireland",
            ],
            area_codes_by_type={
                "la": ["E06000063", "E06000064", "N09000001"],
                "constituency": ["E14000001", "N05000001"],
            },
        )


def test_geography_coverage_summary_reports_missing_target_codes() -> None:
    summary = geography_coverage_summary(
        crosswalk_rows(),
        {
            "la": ["E06000063", "E06000066", "N09000001"],
            "constituency": ["N05000001"],
        },
    )

    la_row = summary.set_index("area_type").loc["la"]
    assert int(la_row["target_areas"]) == 3
    assert int(la_row["covered_areas"]) == 2
    assert la_row["missing_area_codes"] == ("E06000066",)
    assert float(la_row["coverage_share"]) == pytest.approx(2 / 3)
    assert int(la_row["sampleable_areas"]) == 2
    assert la_row["unsampleable_area_codes"] == ()


def test_geography_coverage_summary_flags_zero_probability_areas() -> None:
    crosswalk = crosswalk_rows()
    crosswalk.loc[crosswalk["la_code"] == "E06000064", "population"] = 0

    summary = geography_coverage_summary(
        crosswalk,
        {"la": ["E06000063", "E06000064"]},
    )

    row = summary.iloc[0]
    assert int(row["covered_areas"]) == 2
    assert int(row["sampleable_areas"]) == 1
    assert row["unsampleable_area_codes"] == ("E06000064",)

    with pytest.raises(ValueError, match="unsampleable"):
        validate_geography_coverage(
            crosswalk,
            area_codes_by_type={"la": ["E06000063", "E06000064"]},
        )


def test_assign_household_geography_clones_weights_and_provenance() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101, 102, 103, 104],
            "household_weight": [10.0, 20.0, 30.0, 40.0],
            "region": ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"],
        }
    )

    assignment = assign_household_geography(
        households,
        crosswalk_rows(),
        n_clones=2,
        seed=1,
        source_year=2023,
    )

    result = assignment.household
    assert len(result) == 8
    assert result["household_id"].is_unique
    assert result["household_weight"].sum() == pytest.approx(100.0)
    assert set(ROWWISE_GEOGRAPHY_COLUMNS).issubset(result.columns)
    assert set(result["source_household_key"]) == {
        "2023:101",
        "2023:102",
        "2023:103",
        "2023:104",
    }
    assert result.loc[result["region"] == "LONDON", "oa_code"].str.startswith("E").all()
    assert result.loc[result["region"] == "WALES", "oa_code"].str.startswith("W").all()
    assert (
        result.loc[result["region"] == "SCOTLAND", "oa_code"].str.startswith("S").all()
    )
    assert (
        result.loc[result["region"] == "NORTHERN_IRELAND", "oa_code"]
        .str.startswith("N")
        .all()
    )


def test_assign_household_geography_preserves_frs_region_constraint() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101],
            "household_weight": [10.0],
            "region": ["LONDON"],
        }
    )

    assignment = assign_household_geography(
        households,
        crosswalk_rows(),
        n_clones=10,
        seed=2,
    )

    result = assignment.household
    assert result["region_code_oa"].unique().tolist() == ["E12000007"]
    assert result["oa_code"].unique().tolist() == ["E0001"]


def test_assign_household_geography_raises_when_country_missing() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101],
            "household_weight": [10.0],
            "region": ["NORTHERN_IRELAND"],
        }
    )
    no_ni = crosswalk_rows()[lambda df: df["country"] != "Northern Ireland"]

    with pytest.raises(ValueError, match="Northern Ireland"):
        assign_household_geography(households, no_ni)


def test_assign_household_geography_accepts_numeric_country_column() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101, 102],
            "household_weight": [10.0, 20.0],
            "frs_country": [1, 4],
        }
    )

    assignment = assign_household_geography(
        households,
        crosswalk_rows(),
        country_column="frs_country",
    )

    result = assignment.household
    assert result.loc[result["frs_country"] == 1, "oa_code"].str.startswith("E").all()
    assert result.loc[result["frs_country"] == 4, "oa_code"].str.startswith("N").all()


def test_assign_household_geography_accepts_integral_float_country_column() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101, 102],
            "household_weight": [10.0, 20.0],
            "frs_country": [1.0, 4.0],
        }
    )

    assignment = assign_household_geography(
        households,
        crosswalk_rows(),
        country_column="frs_country",
    )

    result = assignment.household
    assert result.loc[result["frs_country"] == 1.0, "oa_code"].str.startswith("E").all()
    assert result.loc[result["frs_country"] == 4.0, "oa_code"].str.startswith("N").all()


def test_assign_household_geography_can_allow_missing_country_for_pilots() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101, 102],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "NORTHERN_IRELAND"],
        }
    )
    no_ni = crosswalk_rows()[lambda df: df["country"] != "Northern Ireland"]

    assignment = assign_household_geography(
        households,
        no_ni,
        require_all_countries=False,
    )

    result = assignment.household
    assert result.loc[result["region"] == "LONDON", "oa_code"].str.startswith("E").all()
    assert (result.loc[result["region"] == "NORTHERN_IRELAND", "oa_code"] == "").all()


def test_assign_household_geography_rejects_too_small_household_multiplier() -> None:
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "LONDON"],
        }
    )

    with pytest.raises(ValueError, match="household_id.*id_multiplier"):
        assign_household_geography(
            households,
            crosswalk_rows(),
            n_clones=2,
            id_multiplier=1,
        )


def test_assign_household_geography_rejects_zero_population_distribution() -> None:
    households = pd.DataFrame(
        {
            "household_id": [101],
            "household_weight": [10.0],
            "region": ["WALES"],
        }
    )
    crosswalk = crosswalk_rows()
    crosswalk.loc[crosswalk["country"] == "Wales", "population"] = 0

    with pytest.raises(ValueError, match="zero total population"):
        assign_household_geography(households, crosswalk)

    assignment = assign_household_geography(
        households,
        crosswalk,
        allow_zero_population_distribution=True,
    )
    assert assignment.household["oa_code"].tolist() == ["W0001"]


def test_clone_entity_frame_remaps_linked_ids() -> None:
    people = pd.DataFrame(
        {
            "person_id": [101001, 101002],
            "person_household_id": [101, 101],
            "person_benunit_id": [10101, 10101],
        }
    )

    cloned = clone_entity_frame(
        people,
        id_columns=["person_id", "person_household_id", "person_benunit_id"],
        n_clones=2,
        id_multiplier=1_000_000,
        clone_index_column="clone_index",
    )

    assert cloned["person_id"].tolist() == [101001, 101002, 1101001, 1101002]
    assert cloned["person_household_id"].tolist() == [101, 101, 1000101, 1000101]
    assert cloned["person_benunit_id"].tolist() == [10101, 10101, 1010101, 1010101]
    assert cloned["clone_index"].tolist() == [0, 0, 1, 1]


def test_clone_entity_frame_rejects_too_small_multiplier() -> None:
    people = pd.DataFrame(
        {
            "person_id": [101001, 102001],
            "person_household_id": [101, 102],
        }
    )

    with pytest.raises(ValueError, match="id_multiplier"):
        clone_entity_frame(
            people,
            id_columns=["person_id", "person_household_id"],
            n_clones=2,
            id_multiplier=1000,
        )


def test_id_multiplier_can_cover_all_linked_entity_ids() -> None:
    households = pd.DataFrame({"household_id": [101, 102]})
    people = pd.DataFrame(
        {
            "person_id": [101001, 102001],
            "person_household_id": [101, 102],
        }
    )
    benunits = pd.DataFrame({"benunit_id": [10101, 10201]})

    multiplier = id_multiplier_for_values(
        households["household_id"],
        people["person_id"],
        people["person_household_id"],
        benunits["benunit_id"],
    )

    assert multiplier == 1_000_000
    household_assignment = assign_household_geography(
        pd.DataFrame(
            {
                "household_id": [101, 102],
                "household_weight": [1.0, 1.0],
                "region": ["LONDON", "LONDON"],
            }
        ),
        crosswalk_rows(),
        n_clones=2,
        id_multiplier=multiplier,
    )
    cloned_people = clone_entity_frame(
        people,
        id_columns=["person_id", "person_household_id"],
        n_clones=2,
        id_multiplier=household_assignment.id_multiplier,
    )

    assert household_assignment.household["household_id"].tolist() == [
        101,
        102,
        1000101,
        1000102,
    ]
    assert set(cloned_people["person_household_id"]).issubset(
        set(household_assignment.household["household_id"])
    )


def test_assign_household_geography_is_seed_stable() -> None:
    households = pd.DataFrame(
        {
            "household_id": np.arange(1, 21),
            "household_weight": np.ones(20),
            "region": ["LONDON"] * 20,
        }
    )

    first = assign_household_geography(households, crosswalk_rows(), seed=123)
    second = assign_household_geography(households, crosswalk_rows(), seed=123)

    assert first.household["oa_code"].tolist() == second.household["oa_code"].tolist()
