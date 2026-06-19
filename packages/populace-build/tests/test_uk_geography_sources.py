from __future__ import annotations

import io
import urllib.error
import zipfile

import pandas as pd
import pytest

import populace.build.uk.geography_sources as geography_sources
from populace.build.uk import (
    build_complete_uk_geography_crosswalk,
    build_england_wales_crosswalk,
    build_northern_ireland_crosswalk,
    build_official_uk_geography_crosswalk,
    build_scotland_crosswalk,
    geography_coverage_summary,
    infer_ni_dz_constituencies_from_postcodes,
    load_scotland_oa_constituencies,
    load_scotland_oa_lau_lookup,
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


def ew_hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OA21CD": ["E0001", "W0001"],
            "LSOA21CD": ["E0101", "W0101"],
            "MSOA21CD": ["E0201", "W0201"],
            "LAD22CD": ["E07000026", "W06000001"],
        }
    )


def ew_population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geography code": ["E0001", "W0001"],
            "Residence type: Total; measures: Value": [100, 80],
        }
    )


def ew_constituencies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OA21CD": ["E0001", "W0001"],
            "PCON25CD": ["E14000001", "W07000041"],
        }
    )


def ew_lad23_lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OA21CD": ["E0001", "W0001"],
            "LAD23CD": ["E06000063", "W06000001"],
        }
    )


def england_lad_region_lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LAD22CD": ["E07000026"],
            "RGN22CD": ["E12000007"],
        }
    )


def scotland_oa_dz_iz() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OA22": ["S0001", "S0002"],
            "DZ22": ["S0101", "S0102"],
            "IZ22": ["S0201", "S0202"],
        }
    )


def scotland_oa_lau() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oa_code": ["S0001", "S0002"],
            "la_code": ["S12000033", "S12000005"],
        }
    )


def scotland_constituencies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OA22": ["S0001", "S0002"],
            "UKPC24": ["S14000001", "S14000002"],
        }
    )


def scotland_population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OutputArea2022": ["S0001", "S0002"],
            "UsualResidentPopulation": [90, 75],
        }
    )


def zipped_csv_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return buffer.getvalue()


def test_read_url_bytes_retries_transient_http_errors(monkeypatch) -> None:
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "Internal Server Error",
                {},
                None,
            )
        return Response()

    monkeypatch.setattr(geography_sources.urllib.request, "urlopen", fake_urlopen)

    assert geography_sources._read_url_bytes("https://example.test", retry_delay=0) == (
        b"ok"
    )
    assert len(calls) == 2


def test_build_england_wales_crosswalk_from_official_sources() -> None:
    crosswalk = build_england_wales_crosswalk(
        ew_hierarchy(),
        ew_population(),
        ew_constituencies(),
        ew_lad23_lookup(),
        england_lad_region_lookup(),
        expected_oa_count=None,
    )

    assert crosswalk["oa_code"].tolist() == ["E0001", "W0001"]
    assert crosswalk["la_code"].tolist() == ["E06000063", "W06000001"]
    assert crosswalk["region_code"].tolist() == ["E12000007", "W99999999"]
    assert crosswalk["country"].tolist() == ["England", "Wales"]
    assert crosswalk["population"].tolist() == [100.0, 80.0]


def test_build_england_wales_crosswalk_rejects_source_mismatch() -> None:
    with pytest.raises(ValueError, match="E/W OA source codes differ"):
        build_england_wales_crosswalk(
            ew_hierarchy(),
            ew_population().iloc[:1],
            ew_constituencies(),
            ew_lad23_lookup(),
            england_lad_region_lookup(),
            expected_oa_count=None,
        )


def test_load_scotland_oa_lau_lookup_maps_lau_to_council_area(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: zipped_csv_bytes(
            {
                "OA22_LAU25_L1.csv": (
                    "OutputArea2022Code,LAU2025Level1Code\n"
                    "S0001,S30000001\n"
                    "S0002,S30000002\n"
                ),
                "CA19 - LAU25L1 - ITL25L2 - ITL25L3 Lookup.csv": (
                    "CouncilArea2019Code,LAU2025Level1Code\n"
                    "S12000033,S30000001\n"
                    "S12000005,S30000002\n"
                ),
            }
        ),
    )

    lookup = load_scotland_oa_lau_lookup("memory://scotland-lau.zip")

    assert lookup.to_dict("records") == [
        {"oa_code": "S0001", "la_code": "S12000033"},
        {"oa_code": "S0002", "la_code": "S12000005"},
    ]


def test_load_scotland_constituencies_selects_oa_mapping_csv(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: zipped_csv_bytes(
            {
                "Code to Name Lookup UKPC24.csv": (
                    "UKParliamentaryConstituency2024Code,"
                    "UKParliamentaryConstituency2024Name\n"
                    "S14000001,Aberdeen North\n"
                ),
                "OA22_UKPC24.CSV": (
                    "OA22,UKPC24,UKPC24Name\n"
                    "S0001,S14000001,Aberdeen North\n"
                    "S0002,S14000002,Aberdeen South\n"
                ),
            }
        ),
    )

    lookup = load_scotland_oa_constituencies("memory://scotland-pcon.zip")

    assert lookup["oa_code"].tolist() == ["S0001", "S0002"]
    assert lookup["constituency_code"].tolist() == ["S14000001", "S14000002"]


def test_build_scotland_crosswalk_from_official_sources() -> None:
    crosswalk = build_scotland_crosswalk(
        scotland_oa_dz_iz(),
        scotland_oa_lau(),
        scotland_constituencies(),
        scotland_population(),
        expected_oa_count=None,
    )

    assert crosswalk["oa_code"].tolist() == ["S0001", "S0002"]
    assert crosswalk["lsoa_code"].tolist() == ["S0101", "S0102"]
    assert crosswalk["msoa_code"].tolist() == ["S0201", "S0202"]
    assert crosswalk["la_code"].tolist() == ["S12000033", "S12000005"]
    assert crosswalk["region_code"].unique().tolist() == ["S99999999"]
    assert crosswalk["country"].unique().tolist() == ["Scotland"]


def test_build_scotland_crosswalk_rejects_source_mismatch() -> None:
    with pytest.raises(ValueError, match="Scotland OA source codes differ"):
        build_scotland_crosswalk(
            scotland_oa_dz_iz(),
            scotland_oa_lau().iloc[:1],
            scotland_constituencies(),
            scotland_population(),
            expected_oa_count=None,
        )


def test_build_official_uk_geography_crosswalk_from_source_frames() -> None:
    complete = build_official_uk_geography_crosswalk(
        ew_oa_hierarchy=ew_hierarchy(),
        ew_oa_population=ew_population(),
        ew_oa_constituencies=ew_constituencies(),
        ew_oa_lad23_lookup=ew_lad23_lookup(),
        england_lad_region_lookup=england_lad_region_lookup(),
        scotland_oa_dz_iz=scotland_oa_dz_iz(),
        scotland_oa_lau=scotland_oa_lau(),
        scotland_oa_constituencies=scotland_constituencies(),
        scotland_oa_population=scotland_population(),
        ni_dz_hierarchy=ni_hierarchy(),
        ni_dz_population=ni_population(),
        ni_dz_constituencies=pd.DataFrame(
            {
                "oa_code": ["N20000001", "N20000002"],
                "constituency_code": ["N05000014", "N05000002"],
            }
        ),
        expected_england_wales_oa_count=None,
        expected_scotland_oa_count=None,
        expected_ni_dz_count=None,
    )

    validate_geography_coverage(
        complete,
        required_countries=["England", "Wales", "Scotland", "Northern Ireland"],
        area_codes_by_type={
            "constituency": [
                "E14000001",
                "W07000041",
                "S14000001",
                "S14000002",
                "N05000002",
                "N05000014",
            ],
            "la": [
                "E06000063",
                "W06000001",
                "S12000033",
                "S12000005",
                "N09000001",
                "N09000002",
            ],
        },
    )
    assert len(complete) == 6


def test_build_official_uk_geography_crosswalk_accepts_normalized_frames() -> None:
    complete = build_official_uk_geography_crosswalk(
        ew_oa_hierarchy=pd.DataFrame(
            {
                "oa_code": ["E0001", "W0001"],
                "lsoa_code": ["E0101", "W0101"],
                "msoa_code": ["E0201", "W0201"],
                "la_code": ["E07000026", "W06000001"],
            }
        ),
        ew_oa_population=pd.DataFrame(
            {"oa_code": ["E0001", "W0001"], "population": [100, 80]}
        ),
        ew_oa_constituencies=pd.DataFrame(
            {
                "oa_code": ["E0001", "W0001"],
                "constituency_code": ["E14000001", "W07000041"],
            }
        ),
        ew_oa_lad23_lookup=pd.DataFrame(
            {"oa_code": ["E0001", "W0001"], "lad23_code": ["E06000063", "W06000001"]}
        ),
        england_lad_region_lookup=pd.DataFrame(
            {"la_code": ["E07000026"], "region_code": ["E12000007"]}
        ),
        scotland_oa_dz_iz=pd.DataFrame(
            {
                "oa_code": ["S0001", "S0002"],
                "lsoa_code": ["S0101", "S0102"],
                "msoa_code": ["S0201", "S0202"],
            }
        ),
        scotland_oa_lau=scotland_oa_lau(),
        scotland_oa_constituencies=pd.DataFrame(
            {
                "oa_code": ["S0001", "S0002"],
                "constituency_code": ["S14000001", "S14000002"],
            }
        ),
        scotland_oa_population=pd.DataFrame(
            {"oa_code": ["S0001", "S0002"], "population": [90, 75]}
        ),
        ni_dz_hierarchy=ni_hierarchy(),
        ni_dz_population=ni_population(),
        ni_dz_constituencies=pd.DataFrame(
            {
                "oa_code": ["N20000001", "N20000002"],
                "constituency_code": ["N05000014", "N05000002"],
            }
        ),
        expected_england_wales_oa_count=None,
        expected_scotland_oa_count=None,
        expected_ni_dz_count=None,
    )

    assert complete["country"].tolist() == [
        "England",
        "Wales",
        "Scotland",
        "Scotland",
        "Northern Ireland",
        "Northern Ireland",
    ]


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
            "doterm": [pd.NA, "", "202401", " ", pd.NA],
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
            pd.DataFrame({"Census 2021 Data Zone Code": ["N20000003"], "Count": [52]}),
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
