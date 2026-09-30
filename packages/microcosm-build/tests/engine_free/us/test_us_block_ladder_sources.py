"""US block-ladder source parser tests."""

import json

import numpy as np
import pytest

from microcosm.build.us_runtime.block_ladder_sources import (
    US_STATES,
    assemble_us_block_ladder,
    cbsa_delineated_states,
    ct_planning_region_by_block,
    parse_baf_county_subdivision_file,
    parse_baf_district_file,
    parse_baf_place_file,
    parse_cbsa_delineations,
    parse_ct_planning_region_crosswalk,
    parse_national_cd_bef,
    parse_pl_geo_blocks,
    us_block_ladder_cbsa_coverage_failures,
)


def test_us_states_covers_the_fifty_states_plus_dc() -> None:
    assert len(US_STATES) == 51
    fips = [entry[0] for entry in US_STATES]
    assert len(set(fips)) == 51
    assert "11" in fips  # DC
    assert "72" not in fips  # PR is outside the state spine


def test_parse_national_cd_bef_normalizes_delegate_and_drops_unassigned() -> None:
    result = parse_national_cd_bef(
        [
            "GEOID,CDFP\n",
            "100010401001000,00\n",
            "110010001001000,98\n",
            "360010001001000,20\n",
            "360610001001001,ZZ\n",
        ]
    )

    assert result == {
        100010401001000: 1000,
        110010001001000: 1100,
        360010001001000: 3620,
    }


def test_parse_national_cd_bef_refuses_bad_district_codes() -> None:
    with pytest.raises(ValueError, match="two-digit code"):
        parse_national_cd_bef(["GEOID,CDFP\n", "100010401001000,7\n"])
    with pytest.raises(ValueError, match="header"):
        parse_national_cd_bef(["BLOCKID|DISTRICT\n"])
    with pytest.raises(ValueError, match="more than once"):
        parse_national_cd_bef(
            [
                "GEOID,CDFP\n",
                "100010401001000,00\n",
                "100010401001000,01\n",
            ]
        )


def test_parse_baf_district_file_normalizes_unassigned_markers() -> None:
    result = parse_baf_district_file(
        [
            "BLOCKID|DISTRICT\n",
            "100010401001000|001\n",
            "100010401001001|ZZZ\n",
            "100010401001002|00A\n",
        ],
        label="test SLDU",
    )

    assert result == {
        100010401001000: "001",
        100010401001001: "",
        100010401001002: "00A",
    }


def test_parse_baf_district_file_refuses_odd_codes() -> None:
    with pytest.raises(ValueError, match="3-character"):
        parse_baf_district_file(
            ["BLOCKID|DISTRICT\n", "100010401001000|0001\n"],
            label="test SLDU",
        )


def test_parse_baf_place_file_maps_blank_to_zero() -> None:
    result = parse_baf_place_file(
        [
            "BLOCKID|PLACEFP\n",
            "100010401001000|77580\n",
            "100010401001001|\n",
        ],
        label="test place",
    )

    assert result == {100010401001000: 77580, 100010401001001: 0}


def test_parse_pl_geo_blocks_keeps_populated_blocks_and_validates_totals() -> None:
    def geo_row(summary_level: str, geocode: str, population: int) -> str:
        # The real 2020 legacy geoheader has 97 pipe-delimited fields.
        fields = [""] * 97
        fields[2] = summary_level
        fields[9] = geocode
        fields[90] = str(population)
        return "|".join(fields) + "\n"

    blocks = parse_pl_geo_blocks(
        [
            geo_row("040", "10", 30),
            geo_row("750", "100010401001000", 20),
            geo_row("750", "100010401001001", 0),
            geo_row("750", "100010401001002", 10),
        ],
        state_fips="10",
    )

    assert blocks == {100010401001000: 20, 100010401001002: 10}

    with pytest.raises(ValueError, match="state row records"):
        parse_pl_geo_blocks(
            [
                geo_row("040", "10", 31),
                geo_row("750", "100010401001000", 20),
                geo_row("750", "100010401001002", 10),
            ],
            state_fips="10",
        )


def test_parse_cbsa_delineations_reads_past_title_rows_and_footnotes() -> None:
    result = parse_cbsa_delineations(
        [
            ("List 1.", None, None),
            (None, None, None),
            (
                "CBSA Code",
                "Metropolitan Division Code",
                "CSA Code",
                "CBSA Title",
                "Metropolitan/Micropolitan Statistical Area",
                "Metropolitan Division Title",
                "CSA Title",
                "County/County Equivalent",
                "State Name",
                "FIPS State Code",
                "FIPS County Code",
                "Central/Outlying County",
            ),
            (
                "35620",
                "35614",
                "408",
                "New York-Newark-Jersey City, NY-NJ",
                "Metropolitan Statistical Area",
                "New York-Jersey City-White Plains, NY-NJ",
                "New York-Newark, NY-NJ-CT-PA",
                "New York County",
                "New York",
                "36",
                "061",
                "Central",
            ),
            (
                "10100",
                None,
                None,
                "Aberdeen, SD",
                "Micropolitan Statistical Area",
                None,
                None,
                "Brown County",
                "South Dakota",
                "46",
                "013",
                "Central",
            ),
            ("Note: The 2023 delineations are based on OMB Bulletin 23-01.",),
        ]
    )

    assert result == {"36061": 35620, "46013": 10100}


def test_parse_cbsa_delineations_accepts_numeric_cells() -> None:
    header = (
        "CBSA Code",
        "FIPS State Code",
        "FIPS County Code",
    )

    result = parse_cbsa_delineations(
        [
            header,
            (35620.0, 36.0, 61.0),
        ]
    )

    assert result == {"36061": 35620}


def test_parse_cbsa_delineations_refuses_malformed_fips_on_a_data_row() -> None:
    header = (
        "CBSA Code",
        "FIPS State Code",
        "FIPS County Code",
    )

    with pytest.raises(ValueError, match="malformed"):
        parse_cbsa_delineations(
            [
                header,
                ("35620", "NY", "061"),
            ]
        )


def test_parse_cbsa_delineations_refuses_conflicting_assignments() -> None:
    header = (
        "CBSA Code",
        "FIPS State Code",
        "FIPS County Code",
    )
    with pytest.raises(ValueError, match="both CBSA"):
        parse_cbsa_delineations(
            [
                header,
                ("35620", "36", "061"),
                ("10100", "36", "061"),
            ]
        )


def test_assemble_us_block_ladder_round_trips_through_the_loader(tmp_path) -> None:
    from microcosm.build.us_runtime import load_us_block_ladder

    metadata = {
        "schema_version": 1,
        "kind": "us_block_ladder",
        "block_vintage": "2020_tabulation_blocks",
        "sampling_basis": "population",
        "layers": {
            layer: {"vintage": "test", "source": "test source"}
            for layer in ("congressional_district", "sldu", "sldl", "place", "cbsa")
        },
    }
    payload = assemble_us_block_ladder(
        block_population={100010401001000: 20, 360010001001000: 30},
        cd_by_block={100010401001000: 1000, 360010001001000: 3620},
        sldu_by_block={100010401001000: "001"},
        sldl_by_block={},
        place_by_block={360010001001000: 51000},
        cbsa_by_county={"36001": 10580},
        metadata=metadata,
    )
    path = tmp_path / "ladder.npz"
    np.savez_compressed(path, **payload)

    ladder = load_us_block_ladder(path)

    assert ladder.block_geoid.tolist() == [100010401001000, 360010001001000]
    assert ladder.population.tolist() == [20.0, 30.0]
    assert ladder.sldu.tolist() == ["001", ""]
    assert ladder.place_fips.tolist() == [0, 51000]
    assert ladder.cbsa_code.tolist() == [0, 10580]
    assert json.loads(json.dumps(dict(ladder.metadata)))["sampling_basis"] == (
        "population"
    )


def test_assemble_refuses_populated_block_without_a_district() -> None:
    with pytest.raises(ValueError, match="no congressional district"):
        assemble_us_block_ladder(
            block_population={100010401001000: 20},
            cd_by_block={},
            sldu_by_block={},
            sldl_by_block={},
            place_by_block={},
            cbsa_by_county={},
            metadata={},
        )


# --------------------------------------------------------------------------
# Connecticut: 2022 planning regions against 2020 block counties
# --------------------------------------------------------------------------

# Rows copied from Census's ct_cou_to_cousub_crosswalk.txt (BOM already
# stripped, as the builder reads it): a quoted two-line header, town rows,
# "County subdivisions not defined" rows listed under several planning
# regions, and a glossary after blank lines.
_CT_CROSSWALK = (
    '"STATEFP\n(INCITS38)"|"OLD_COUNTYFP\n(INCITS31)"|OLD_COUNTY_NAMELSAD|'
    '"NEW_COUNTYFP\n(INCITS31)"|NEW_COUNTY_NAMELSAD|COUSUBFP|OLD_COUSUB_GEOID|'
    'NEW_COUSUB_GEOID|COUSUB_NAMELSAD|"COUSUBNS\n(INCITS446)"|COUSUB_LSAD|'
    "COUSUB_FUNCSTAT|COUSUB_CLASSFP\n"
    "09|001|Fairfield County|120|Greater Bridgeport Planning Region|08070|"
    "0900108070|0912008070|Bridgeport town|00213396|43|C|T5\n"
    "09|001|Fairfield County|120|Greater Bridgeport Planning Region|00000|"
    "0900100000|0912000000|County subdivisions not defined|00000000|00|F|Z9\n"
    "09|001|Fairfield County|140|Naugatuck Valley Planning Region|68170|"
    "0900168170|0914068170|Shelton town|00213504|43|C|T5\n"
    "09|001|Fairfield County|190|Western Connecticut Planning Region|00000|"
    "0900100000|0919000000|County subdivisions not defined|00000000|00|F|Z9\n"
    "09|001|Fairfield County|190|Western Connecticut Planning Region|73070|"
    "0900173070|0919073070|Stamford town|00213511|43|C|T5\n"
    "09|009|New Haven County|170|South Central Connecticut Planning Region|52070|"
    "0900952070|0917052070|New Haven town|00213471|43|C|T5\n"
    "\n"
    "\n"
    "GLOSSARY\n"
    "STATEFP = State FIPS Code / ANSI INCITS 38 Code\n"
)

# Real 2020 blocks (BAF MCD layer): Bridgeport, Shelton, Stamford (all in
# 2020 Fairfield County 09001) and New Haven (09009).
_BRIDGEPORT_BLOCK = 90010701001000
_SHELTON_BLOCK = 90011101001000
_STAMFORD_BLOCK = 90010201011000
_NEW_HAVEN_BLOCK = 90091401011000
_CT_MCD = [
    "BLOCKID|COUNTYFP|COUSUBFP\n",
    "090010701001000|001|08070\n",
    "090011101001000|001|68170\n",
    "090010201011000|001|73070\n",
    "090091401011000|009|52070\n",
]
# OMB 2023 delineation rows for these planning regions.
_CT_CBSA_BY_COUNTY = {
    "09120": 14860,  # Greater Bridgeport → Bridgeport-Stamford-Danbury
    "09140": 47930,  # Naugatuck Valley → Waterbury-Shelton
    "09170": 35300,  # South Central → New Haven
    "09190": 14860,  # Western → Bridgeport-Stamford-Danbury
}
_DE_BLOCK = 100010401001000


def _ladder_metadata(**extra: object) -> dict:
    return {
        "schema_version": 1,
        "kind": "us_block_ladder",
        "block_vintage": "2020_tabulation_blocks",
        "sampling_basis": "population",
        "layers": {
            layer: {"vintage": "test", "source": "test source"}
            for layer in ("congressional_district", "sldu", "sldl", "place", "cbsa")
        },
        **extra,
    }


def _assemble(
    block_population: dict[int, int],
    *,
    cbsa_by_county: dict[str, int],
    cbsa_county_by_block: dict[int, str] | None = None,
    metadata: dict | None = None,
) -> dict[str, np.ndarray]:
    return assemble_us_block_ladder(
        block_population=block_population,
        cd_by_block={block: (block // 10**13) * 100 + 1 for block in block_population},
        sldu_by_block={},
        sldl_by_block={},
        place_by_block={},
        cbsa_by_county=cbsa_by_county,
        metadata=metadata if metadata is not None else _ladder_metadata(),
        cbsa_county_by_block=cbsa_county_by_block,
    )


def _ct_planning_regions() -> dict[int, str]:
    return ct_planning_region_by_block(
        [_BRIDGEPORT_BLOCK, _SHELTON_BLOCK, _STAMFORD_BLOCK, _NEW_HAVEN_BLOCK],
        cousub_by_block=parse_baf_county_subdivision_file(
            _CT_MCD, label="BlockAssign_ST09_CT_MCD.txt"
        ),
        planning_region_by_cousub=parse_ct_planning_region_crosswalk(
            _CT_CROSSWALK.splitlines(keepends=True)
        ),
    )


def test_parse_ct_planning_region_crosswalk_reads_the_census_layout() -> None:
    result = parse_ct_planning_region_crosswalk(_CT_CROSSWALK.splitlines(keepends=True))

    # Towns only: the not-defined rows name no single planning region.
    assert result == {
        "0900108070": "09120",
        "0900168170": "09140",
        "0900173070": "09190",
        "0900952070": "09170",
    }
    # Line endings, or a BOM the caller did not strip, do not change the parse.
    assert parse_ct_planning_region_crosswalk(_CT_CROSSWALK.splitlines()) == result
    assert (
        parse_ct_planning_region_crosswalk(
            ("\ufeff" + _CT_CROSSWALK).splitlines(keepends=True)
        )
        == result
    )


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("|08070|0900108070|", "|08070|0900108071|", "disagree with its code"),
        ("09|009|New Haven", "10|009|New Haven", "STATEFP '10'"),
        ("|52070|0900952070|", "|5207|0900952070|", "COUSUBFP '5207'"),
        ("Shelton town|00213504|43|C|T5", "Shelton town|00213504|43|C", "fields"),
    ],
)
def test_parse_ct_planning_region_crosswalk_refuses_malformed_rows(
    old: str, new: str, match: str
) -> None:
    assert _CT_CROSSWALK.count(old) == 1
    with pytest.raises(ValueError, match=match):
        parse_ct_planning_region_crosswalk(
            _CT_CROSSWALK.replace(old, new).splitlines(keepends=True)
        )


def test_parse_ct_planning_region_crosswalk_refuses_a_town_in_two_regions() -> None:
    conflicting = _CT_CROSSWALK.replace(
        "\n\n\nGLOSSARY",
        "\n09|001|Fairfield County|190|Western Connecticut Planning Region|08070|"
        "0900108070|0919008070|Bridgeport town|00213396|43|C|T5\n\n\nGLOSSARY",
    )
    with pytest.raises(ValueError, match="both 09120 and 09190"):
        parse_ct_planning_region_crosswalk(conflicting.splitlines(keepends=True))


def test_parse_ct_planning_region_crosswalk_refuses_a_changed_header() -> None:
    with pytest.raises(ValueError, match="header must be"):
        parse_ct_planning_region_crosswalk(
            _CT_CROSSWALK.replace("NEW_COUNTY_NAMELSAD", "NEW_NAME").splitlines(
                keepends=True
            )
        )


def test_parse_baf_county_subdivision_file_keys_towns_by_2020_county() -> None:
    result = parse_baf_county_subdivision_file(_CT_MCD, label="MCD")

    assert result == {
        _BRIDGEPORT_BLOCK: "0900108070",
        _SHELTON_BLOCK: "0900168170",
        _STAMFORD_BLOCK: "0900173070",
        _NEW_HAVEN_BLOCK: "0900952070",
    }


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("090010701001000|009|08070\n", "block geoid's county is 001"),
        ("090010701001000|001|8070\n", "COUSUBFP '8070'"),
        ("090010701001000|001\n", "three fields"),
    ],
)
def test_parse_baf_county_subdivision_file_refuses_bad_rows(
    row: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_baf_county_subdivision_file([_CT_MCD[0], row], label="MCD")


def test_ct_planning_region_by_block_follows_each_town() -> None:
    # Fairfield County (09001) splits across three planning regions, so the
    # 2020 county alone cannot name a block's county-equivalent.
    assert _ct_planning_regions() == {
        _BRIDGEPORT_BLOCK: "09120",
        _SHELTON_BLOCK: "09140",
        _STAMFORD_BLOCK: "09190",
        _NEW_HAVEN_BLOCK: "09170",
    }


def test_ct_planning_region_by_block_refuses_blocks_it_cannot_place() -> None:
    crosswalk = parse_ct_planning_region_crosswalk(
        _CT_CROSSWALK.splitlines(keepends=True)
    )
    with pytest.raises(ValueError, match="no county subdivision"):
        ct_planning_region_by_block(
            [_BRIDGEPORT_BLOCK, _DE_BLOCK],
            cousub_by_block={},
            planning_region_by_cousub=crosswalk,
        )
    # A populated block in an undefined (water) area names no planning region.
    with pytest.raises(ValueError, match="absent from the planning-region"):
        ct_planning_region_by_block(
            [_BRIDGEPORT_BLOCK],
            cousub_by_block={_BRIDGEPORT_BLOCK: "0900100000"},
            planning_region_by_cousub=crosswalk,
        )
    # Non-CT blocks are not its business.
    assert (
        ct_planning_region_by_block(
            [_DE_BLOCK], cousub_by_block={}, planning_region_by_cousub=crosswalk
        )
        == {}
    )


def test_assemble_looks_up_connecticut_cbsa_by_planning_region(tmp_path) -> None:
    from microcosm.build.us_runtime import load_us_block_ladder

    cbsa_by_county = {**_CT_CBSA_BY_COUNTY, "10001": 20100}
    blocks = {
        _BRIDGEPORT_BLOCK: 40,
        _SHELTON_BLOCK: 30,
        _STAMFORD_BLOCK: 20,
        _NEW_HAVEN_BLOCK: 10,
        _DE_BLOCK: 5,
    }
    metadata = _ladder_metadata(
        cbsa_delineated_states=cbsa_delineated_states(cbsa_by_county, ["09", "10"])
    )
    payload = _assemble(
        blocks,
        cbsa_by_county=cbsa_by_county,
        cbsa_county_by_block=_ct_planning_regions(),
        metadata=metadata,
    )
    path = tmp_path / "ladder.npz"
    np.savez_compressed(path, **payload)
    ladder = load_us_block_ladder(path)

    cbsa = dict(
        zip(ladder.block_geoid.tolist(), ladder.cbsa_code.tolist(), strict=True)
    )
    assert cbsa == {
        _BRIDGEPORT_BLOCK: 14860,
        _SHELTON_BLOCK: 47930,
        _STAMFORD_BLOCK: 14860,
        _NEW_HAVEN_BLOCK: 35300,
        _DE_BLOCK: 20100,
    }
    assert us_block_ladder_cbsa_coverage_failures(ladder) == []


def test_assemble_refuses_ct_blocks_keyed_by_2020_county() -> None:
    # The pre-fix join: CT blocks looked up by 09001/09009 reach none of the
    # planning-region rows, which used to leave every CT block at CBSA 0.
    with pytest.raises(ValueError, match=r"4 CBSA-delineated county-equivalent"):
        _assemble(
            {_BRIDGEPORT_BLOCK: 40, _NEW_HAVEN_BLOCK: 10},
            cbsa_by_county=_CT_CBSA_BY_COUNTY,
        )


def test_assemble_refuses_a_state_only_partly_remapped() -> None:
    with pytest.raises(ValueError, match="lack a county-equivalent"):
        _assemble(
            {_BRIDGEPORT_BLOCK: 40, _NEW_HAVEN_BLOCK: 10},
            cbsa_by_county={"09120": 14860},
            cbsa_county_by_block={_BRIDGEPORT_BLOCK: "09120"},
        )


@pytest.mark.parametrize(
    ("county", "match"),
    [("10001", "in another state"), ("0912", "5-digit"), (9120, "5-digit")],
)
def test_assemble_refuses_malformed_county_equivalents(
    county: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _assemble(
            {_BRIDGEPORT_BLOCK: 40},
            cbsa_by_county={},
            cbsa_county_by_block={_BRIDGEPORT_BLOCK: county},
        )


def test_assemble_ignores_remap_entries_for_unpopulated_blocks() -> None:
    # The MCD layer covers every block, populated or not; entries for blocks
    # outside the ladder (before, between and after its rows) change nothing.
    regions = {
        **_ct_planning_regions(),
        90010101011000: "09190",
        90010900001000: "09120",
        99999999999999: "09170",
    }
    payload = _assemble(
        {_BRIDGEPORT_BLOCK: 40, _NEW_HAVEN_BLOCK: 10},
        cbsa_by_county={"09120": 14860, "09170": 35300},
        cbsa_county_by_block=regions,
    )

    assert payload["block_geoid"].tolist() == [_BRIDGEPORT_BLOCK, _NEW_HAVEN_BLOCK]
    assert payload["cbsa_code"].tolist() == [14860, 35300]


def test_assemble_ignores_delineation_rows_outside_the_built_states() -> None:
    # A smoke build of Delaware alone must not trip over Connecticut rows.
    payload = _assemble({_DE_BLOCK: 5}, cbsa_by_county=_CT_CBSA_BY_COUNTY)

    assert payload["cbsa_code"].tolist() == [0]


def _write_loaded_ladder(tmp_path, payload: dict[str, np.ndarray]):
    from microcosm.build.us_runtime import load_us_block_ladder

    path = tmp_path / "ladder.npz"
    np.savez_compressed(path, **payload)
    return load_us_block_ladder(path)


def _legacy_ct_payload(**metadata_extra: object) -> dict[str, np.ndarray]:
    """A ladder shaped like the pre-fix national build: CT all at CBSA 0."""
    payload = _assemble(
        {_BRIDGEPORT_BLOCK: 40, _NEW_HAVEN_BLOCK: 10, _DE_BLOCK: 5},
        cbsa_by_county={"10001": 20100},
        metadata=_ladder_metadata(**metadata_extra),
    )
    assert payload["cbsa_code"].tolist() == [0, 0, 20100]
    return payload


def test_cbsa_coverage_check_flags_the_legacy_connecticut_ladder(tmp_path) -> None:
    # No cbsa_delineated_states record (artifacts built before it): every
    # state present is taken as delineated, so all-zero CT fails.
    ladder = _write_loaded_ladder(tmp_path, _legacy_ct_payload())

    failures = us_block_ladder_cbsa_coverage_failures(ladder)

    assert len(failures) == 1
    assert failures[0].startswith("state 09: all 2 populated blocks")


def test_cbsa_coverage_check_follows_the_recorded_states(tmp_path) -> None:
    recorded = _write_loaded_ladder(
        tmp_path, _legacy_ct_payload(cbsa_delineated_states=["09", "10"])
    )
    assert [
        failure[:8] for failure in us_block_ladder_cbsa_coverage_failures(recorded)
    ] == ["state 09"]
    # A state the delineation does not cover may legitimately be all zero.
    uncovered = _write_loaded_ladder(
        tmp_path, _legacy_ct_payload(cbsa_delineated_states=["10"])
    )
    assert us_block_ladder_cbsa_coverage_failures(uncovered) == []
    malformed = _write_loaded_ladder(
        tmp_path, _legacy_ct_payload(cbsa_delineated_states="09")
    )
    assert "must be a list" in us_block_ladder_cbsa_coverage_failures(malformed)[0]


def test_cbsa_delineated_states_keeps_built_states_with_delineated_territory() -> None:
    assert cbsa_delineated_states(
        {"09120": 14860, "10001": 20100, "72127": 41980}, ["09", "10", "11"]
    ) == ["09", "10"]


# --------------------------------------------------------------------------
# Invariants of the CBSA join, for all inputs
# --------------------------------------------------------------------------

_STATES = ("09", "10", "36")


def _block_strategy(st):
    return st.builds(
        lambda state, county, tract_block: int(
            f"{state}{county:03d}{tract_block:010d}"
        ),
        st.sampled_from(_STATES),
        st.integers(1, 4),
        st.integers(1, 10**10 - 1),
    )


def test_cbsa_join_invariants_hold_for_generated_ladders(tmp_path_factory) -> None:
    """For any blocks, delineation and CT remap:

    1. the join refuses exactly when a delineated county-equivalent in a
       built state is reached by no populated block;
    2. otherwise each block's cbsa_code is the delineation's code for its
       county-equivalent (0 when undelineated);
    3. every other array is independent of the CBSA inputs;
    4. a ladder the join accepts passes the loader-level coverage check.
    """
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from microcosm.build.us_runtime import load_us_block_ladder

    @settings(max_examples=200, deadline=None)
    @given(
        blocks=st.dictionaries(
            _block_strategy(st), st.integers(1, 1000), min_size=1, max_size=12
        ),
        regions=st.lists(st.sampled_from(["09110", "09120", "09190"]), max_size=12),
        delineation=st.dictionaries(
            st.sampled_from(
                [
                    "09001",
                    "09002",
                    "09110",
                    "09120",
                    "09190",
                    "10001",
                    "10003",
                    "36001",
                    "36004",
                ]
            ),
            st.integers(10000, 99999),
            max_size=6,
        ),
        remap_ct=st.booleans(),
    )
    def check(blocks, regions, delineation, remap_ct) -> None:
        ordered = sorted(blocks)
        ct_blocks = [block for block in ordered if f"{block:015d}"[:2] == "09"]
        remap = (
            {
                block: regions[index % len(regions)]
                for index, block in enumerate(ct_blocks)
            }
            if remap_ct and regions
            else {}
        )
        key = {block: remap.get(block, f"{block:015d}"[:5]) for block in ordered}
        built = {f"{block:015d}"[:2] for block in ordered}
        unreached = {
            county
            for county in delineation
            if county[:2] in built and county not in set(key.values())
        }
        if unreached:
            with pytest.raises(ValueError, match="CBSA-delineated county-equivalent"):
                _assemble(
                    blocks, cbsa_by_county=delineation, cbsa_county_by_block=remap
                )
            return
        metadata = _ladder_metadata(
            cbsa_delineated_states=cbsa_delineated_states(delineation, built)
        )
        payload = _assemble(
            blocks,
            cbsa_by_county=delineation,
            cbsa_county_by_block=remap,
            metadata=metadata,
        )
        assert payload["cbsa_code"].tolist() == [
            delineation.get(key[block], 0) for block in ordered
        ]
        baseline = _assemble(blocks, cbsa_by_county={}, metadata=metadata)
        for name, values in payload.items():
            if name != "cbsa_code":
                np.testing.assert_array_equal(values, baseline[name])
        path = tmp_path_factory.mktemp("ladder") / "ladder.npz"
        np.savez_compressed(path, **payload)
        assert us_block_ladder_cbsa_coverage_failures(load_us_block_ladder(path)) == []

    check()


def test_ct_crosswalk_round_trips_generated_towns() -> None:
    """Any town table written in the Census layout parses back to itself."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    header = _CT_CROSSWALK.split("COUSUB_CLASSFP\n", 1)[0] + "COUSUB_CLASSFP\n"

    @settings(max_examples=200, deadline=None)
    @given(
        towns=st.dictionaries(
            st.tuples(st.integers(1, 15), st.integers(1, 99999)),
            st.integers(110, 190),
            min_size=1,
            max_size=20,
        ),
        undefined=st.lists(
            st.tuples(st.integers(1, 15), st.integers(110, 190)), max_size=4
        ),
    )
    def check(towns, undefined) -> None:
        rows = [
            f"09|{old:03d}|X County|{new:03d}|Y Planning Region|{cousub:05d}|"
            f"09{old:03d}{cousub:05d}|09{new:03d}{cousub:05d}|Z town|0|43|A|T1\n"
            for (old, cousub), new in towns.items()
        ] + [
            f"09|{old:03d}|X County|{new:03d}|Y Planning Region|00000|"
            f"09{old:03d}00000|09{new:03d}00000|County subdivisions not defined|"
            "00000000|00|F|Z9\n"
            for old, new in undefined
        ]
        text = header + "".join(rows) + "\n\nGLOSSARY\n"
        assert parse_ct_planning_region_crosswalk(text.splitlines(keepends=True)) == {
            f"09{old:03d}{cousub:05d}": f"09{new:03d}"
            for (old, cousub), new in towns.items()
        }

    check()
