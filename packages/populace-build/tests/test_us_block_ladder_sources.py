"""US block-ladder source parser tests."""

import json

import numpy as np
import pytest

from populace.build.us_runtime.block_ladder_sources import (
    US_STATES,
    assemble_us_block_ladder,
    parse_baf_district_file,
    parse_baf_place_file,
    parse_cbsa_delineations,
    parse_national_cd_bef,
    parse_pl_geo_blocks,
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
        fields = [""] * 93
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
    from populace.build.us_runtime import load_us_block_ladder

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
