from __future__ import annotations

import pandas as pd
import pytest

from populace.build.uk import (
    build_complete_uk_geography_crosswalk,
    build_northern_ireland_crosswalk,
    geography_coverage_summary,
    infer_ni_dz_constituencies_from_postcodes,
    update_england_wales_lad_codes,
    validate_geography_coverage,
)


def base_crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": "E0001",
                "lsoa_code": "E0101",
                "msoa_code": "E0201",
                "la_code": "E07000026",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 100,
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


def ni_hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oa_code": ["N20000001", "N20000002"],
            "lsoa_code": ["N20000001", "N20000002"],
            "msoa_code": ["N21000001", "N21000001"],
            "la_code": ["N09000001", "N09000002"],
        }
    )


def ni_population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Census 2021 Data Zone Code": ["N20000001", "N20000002"],
            "Count": [738, 331],
        }
    )


def test_update_england_wales_lad_codes_repairs_new_unitary_codes() -> None:
    repaired = update_england_wales_lad_codes(
        base_crosswalk(),
        pd.DataFrame(
            {
                "OA21CD": ["E0001", "W0001"],
                "LAD23CD": ["E06000063", "W06000001"],
            }
        ),
    )

    assert repaired.loc[repaired["oa_code"] == "E0001", "la_code"].item() == (
        "E06000063"
    )
    assert repaired.loc[repaired["oa_code"] == "W0001", "la_code"].item() == (
        "W06000001"
    )


def test_update_england_wales_lad_codes_rejects_missing_lookup_rows() -> None:
    with pytest.raises(ValueError, match="missing OA code"):
        update_england_wales_lad_codes(
            base_crosswalk(),
            pd.DataFrame({"OA21CD": ["E0001"], "LAD23CD": ["E06000063"]}),
        )


def test_update_england_wales_lad_codes_rejects_blank_lad23_codes() -> None:
    with pytest.raises(ValueError, match="blank LAD23"):
        update_england_wales_lad_codes(
            base_crosswalk(),
            pd.DataFrame(
                {
                    "OA21CD": ["E0001", "W0001"],
                    "LAD23CD": ["E06000063", pd.NA],
                }
            ),
        )


def test_infer_ni_dz_constituencies_from_active_postcode_mode() -> None:
    postcode_oa = pd.DataFrame(
        {
            "pcds": ["BT1 1AA", "BT1 1AB", "BT1 1AC", "BT2 2AA", "BT2 2AB"],
            "doterm": [pd.NA, pd.NA, "202401", pd.NA, pd.NA],
            "oa21cd": [
                "N20000001",
                "N20000001",
                "N20000001",
                "N20000002",
                "N20000002",
            ],
        }
    )
    postcode_constituency = pd.DataFrame(
        {
            "pcd": ["BT1 1AA", "BT1 1AB", "BT1 1AC", "BT2 2AA", "BT2 2AB"],
            "pconcd": [
                "N05000014",
                "N05000014",
                "N05000001",
                "N05000002",
                "N05000003",
            ],
        }
    )

    inferred = infer_ni_dz_constituencies_from_postcodes(
        postcode_oa,
        postcode_constituency,
    )

    assert inferred["oa_code"].tolist() == ["N20000001", "N20000002"]
    assert inferred["constituency_code"].tolist() == ["N05000014", "N05000002"]
    assert inferred["postcode_count"].tolist() == [2, 1]


def test_infer_ni_dz_constituencies_reports_unmatched_postcodes() -> None:
    inferred = infer_ni_dz_constituencies_from_postcodes(
        pd.DataFrame(
            {
                "pcds": ["BT1 1AA", "BT1 1AB", "BT2 2AA"],
                "doterm": [pd.NA, pd.NA, pd.NA],
                "oa21cd": ["N20000001", "N20000001", "N20000002"],
            }
        ),
        pd.DataFrame(
            {
                "pcd": ["BT1 1AA", "BT2 2AA"],
                "pconcd": ["N05000014", "N05000002"],
            }
        ),
        max_unmatched_active_postcode_share=0.5,
    )

    assert inferred.attrs["active_ni_postcode_count"] == 3
    assert inferred.attrs["unmatched_active_ni_postcode_count"] == 1


def test_infer_ni_dz_constituencies_rejects_excess_unmatched_postcodes() -> None:
    with pytest.raises(ValueError, match="missing too many active NI postcodes"):
        infer_ni_dz_constituencies_from_postcodes(
            pd.DataFrame(
                {
                    "pcds": ["BT1 1AA", "BT1 1AB"],
                    "doterm": [pd.NA, pd.NA],
                    "oa21cd": ["N20000001", "N20000001"],
                }
            ),
            pd.DataFrame({"pcd": ["BT1 1AA"], "pconcd": ["N05000014"]}),
            max_unmatched_active_postcode_share=0.1,
        )


def test_build_northern_ireland_crosswalk_rows() -> None:
    ni = build_northern_ireland_crosswalk(
        ni_hierarchy(),
        ni_population(),
        pd.DataFrame(
            {
                "oa_code": ["N20000001", "N20000002"],
                "constituency_code": ["N05000014", "N05000002"],
            }
        ),
        expected_dz_count=None,
    )

    assert ni["oa_code"].tolist() == ["N20000001", "N20000002"]
    assert ni["country"].unique().tolist() == ["Northern Ireland"]
    assert ni["region_code"].unique().tolist() == ["N99999999"]
    assert ni["population"].tolist() == [738, 331]


def test_build_complete_crosswalk_validates_targets() -> None:
    dz_constituencies = pd.DataFrame(
        {
            "oa_code": ["N20000001", "N20000002"],
            "constituency_code": ["N05000014", "N05000002"],
        }
    )

    complete = build_complete_uk_geography_crosswalk(
        base_crosswalk(),
        ew_oa_lad23_lookup=pd.DataFrame(
            {
                "OA21CD": ["E0001", "W0001"],
                "LAD23CD": ["E06000063", "W06000001"],
            }
        ),
        ni_dz_hierarchy=ni_hierarchy(),
        ni_dz_population=ni_population(),
        ni_dz_constituencies=dz_constituencies,
        expected_ni_dz_count=None,
    )

    summary = geography_coverage_summary(
        complete,
        {
            "constituency": ["E14000001", "W07000041", "N05000002", "N05000014"],
            "la": ["E06000063", "W06000001", "N09000001", "N09000002"],
        },
    )
    validate_geography_coverage(
        complete,
        required_countries=["England", "Wales", "Northern Ireland"],
        area_codes_by_type={
            "constituency": ["E14000001", "W07000041", "N05000002", "N05000014"],
            "la": ["E06000063", "W06000001", "N09000001", "N09000002"],
        },
    )

    assert summary["coverage_share"].tolist() == [1.0, 1.0]
    assert summary["sampleable_share"].tolist() == [1.0, 1.0]


def test_build_northern_ireland_crosswalk_rejects_missing_pcon() -> None:
    with pytest.raises(ValueError, match="NI DZ source codes differ"):
        build_northern_ireland_crosswalk(
            ni_hierarchy(),
            ni_population(),
            pd.DataFrame(
                {
                    "oa_code": ["N20000001"],
                    "constituency_code": ["N05000014"],
                }
            ),
            expected_dz_count=None,
        )


def test_build_northern_ireland_crosswalk_rejects_blank_pcon() -> None:
    with pytest.raises(ValueError, match="blank PCON"):
        build_northern_ireland_crosswalk(
            ni_hierarchy(),
            ni_population(),
            pd.DataFrame(
                {
                    "oa_code": ["N20000001", "N20000002"],
                    "constituency_code": ["N05000014", pd.NA],
                }
            ),
            expected_dz_count=None,
        )


def test_build_northern_ireland_crosswalk_rejects_source_code_mismatch() -> None:
    population = pd.concat(
        [
            ni_population(),
            pd.DataFrame(
                {"Census 2021 Data Zone Code": ["N20000003"], "Count": [52]}
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="NI DZ source codes differ"):
        build_northern_ireland_crosswalk(
            ni_hierarchy(),
            population,
            pd.DataFrame(
                {
                    "oa_code": ["N20000001", "N20000002"],
                    "constituency_code": ["N05000014", "N05000002"],
                }
            ),
            expected_dz_count=None,
        )


def test_build_northern_ireland_crosswalk_enforces_expected_dz_count() -> None:
    with pytest.raises(ValueError, match="expected 2 DZ2021"):
        build_northern_ireland_crosswalk(
            ni_hierarchy().iloc[:1],
            ni_population(),
            pd.DataFrame(
                {
                    "oa_code": ["N20000001", "N20000002"],
                    "constituency_code": ["N05000014", "N05000002"],
                }
            ),
            expected_dz_count=2,
        )


def test_build_complete_crosswalk_drops_noncanonical_stale_ni_rows() -> None:
    stale_ni = pd.DataFrame(
        [
            {
                "oa_code": "N09999999",
                "lsoa_code": "N09999999",
                "msoa_code": "N09999999",
                "la_code": "N09000099",
                "constituency_code": "N05000099",
                "region_code": "N99999999",
                "country": "NORTHERN_IRELAND",
                "population": 100,
            }
        ]
    )
    complete = build_complete_uk_geography_crosswalk(
        pd.concat([base_crosswalk(), stale_ni], ignore_index=True),
        ew_oa_lad23_lookup=pd.DataFrame(
            {
                "OA21CD": ["E0001", "W0001"],
                "LAD23CD": ["E06000063", "W06000001"],
            }
        ),
        ni_dz_hierarchy=ni_hierarchy(),
        ni_dz_population=ni_population(),
        ni_dz_constituencies=pd.DataFrame(
            {
                "oa_code": ["N20000001", "N20000002"],
                "constituency_code": ["N05000014", "N05000002"],
            }
        ),
        expected_ni_dz_count=None,
    )

    assert "N09999999" not in set(complete["oa_code"])
    assert set(complete.loc[complete["country"] == "Northern Ireland", "oa_code"]) == {
        "N20000001",
        "N20000002",
    }
