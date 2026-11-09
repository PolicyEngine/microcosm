"""Tests for the pure PUMA-ladder source parsers and assembler."""

import json

import numpy as np
import pytest

from microcosm.build.us_runtime.puma_ladder_sources import (
    assemble_us_puma_ladder,
    parse_tract_to_puma_relationship,
)

# Four populated blocks: PUMA 0100100 spans counties 01001/01003 and CDs
# 101/102; PUMA 0100200 sits in county 01003; PUMA 0200100 is the state-02
# at-large district. Block ints drop the leading state zero, exactly as the
# P.L. 94-171 and CD BEF parsers return them.
_BLOCK_POPULATION = {
    10010001001000: 900,  # tract 01001000100, county 01001
    10030001001000: 100,  # tract 01003000100, county 01003
    10030002001000: 500,  # tract 01003000200, county 01003
    20130001001000: 400,  # tract 02013000100, county 02013
}
_CD_BY_BLOCK = {
    10010001001000: 101,
    10030001001000: 102,
    10030002001000: 102,
    20130001001000: 200,
}
_TRACT_TO_PUMA = {
    1001000100: 100100,
    1003000100: 100100,
    1003000200: 100200,
    2013000100: 200100,
}
_METADATA = {
    "schema_version": 1,
    "kind": "us_puma_ladder",
    "puma_vintage": "2020_puma",
    "sampling_basis": "population",
    "layers": {
        "congressional_district": {"vintage": "119th_congress", "source": "cd119"},
        "county": {"vintage": "2020_census", "source": "tract-to-puma"},
        "tract": {"vintage": "2020_census", "source": "tract-to-puma"},
    },
}


def _relationship_lines(extra: list[str] | None = None) -> list[str]:
    lines = [
        "﻿STATEFP,COUNTYFP,TRACTCE,PUMA5CE",
        "01,001,000100,00100",
        "01,003,000100,00100",
        "01,003,000200,00200",
        "02,013,000100,00100",
        "72,001,000100,00100",  # Puerto Rico — filtered out.
    ]
    return lines + (extra or [])


def test_parse_tract_to_puma_filters_territories_and_builds_geoids() -> None:
    mapping = parse_tract_to_puma_relationship(
        _relationship_lines(), allowed_state_fips=frozenset({"01", "02"})
    )

    assert mapping == _TRACT_TO_PUMA
    # The Puerto Rico tract is excluded by the state filter.
    assert 72001000100 not in mapping


def test_parse_tract_to_puma_keeps_all_states_without_a_filter() -> None:
    mapping = parse_tract_to_puma_relationship(_relationship_lines())

    assert 72001000100 in mapping
    assert mapping[72001000100] == 7200100


def test_parse_tract_to_puma_rejects_a_wrong_header() -> None:
    with pytest.raises(ValueError, match="header must be"):
        parse_tract_to_puma_relationship(["STATE,COUNTY,TRACT,PUMA", "01,001,1,1"])


def test_parse_tract_to_puma_rejects_a_malformed_row() -> None:
    with pytest.raises(ValueError, match="four fields"):
        parse_tract_to_puma_relationship(
            ["STATEFP,COUNTYFP,TRACTCE,PUMA5CE", "01,001,000100"]
        )


def test_parse_tract_to_puma_rejects_a_bad_width() -> None:
    with pytest.raises(ValueError, match="5-digit"):
        parse_tract_to_puma_relationship(
            ["STATEFP,COUNTYFP,TRACTCE,PUMA5CE", "01,001,000100,100"]
        )


def test_parse_tract_to_puma_rejects_conflicting_pumas() -> None:
    with pytest.raises(ValueError, match="both PUMA"):
        parse_tract_to_puma_relationship(
            [
                "STATEFP,COUNTYFP,TRACTCE,PUMA5CE",
                "01,001,000100,00100",
                "01,001,000100,00200",
            ]
        )


def test_assemble_builds_conserving_overlap_tables() -> None:
    payload = assemble_us_puma_ladder(
        block_population=_BLOCK_POPULATION,
        cd_by_block=_CD_BY_BLOCK,
        tract_to_puma=_TRACT_TO_PUMA,
        metadata=_METADATA,
    )

    assert payload["puma"].tolist() == [100100, 100200, 200100]
    assert payload["puma_population"].tolist() == [1000, 500, 400]

    # CD overlap sorted by (puma, cd), conserving each PUMA's population.
    assert list(
        zip(
            payload["cd_overlap_puma"].tolist(),
            payload["cd_overlap_cd"].tolist(),
            payload["cd_overlap_population"].tolist(),
            strict=True,
        )
    ) == [
        (100100, 101, 900),
        (100100, 102, 100),
        (100200, 102, 500),
        (200100, 200, 400),
    ]
    # County overlap: PUMA 0100100 straddles counties 01001 and 01003.
    assert list(
        zip(
            payload["county_overlap_puma"].tolist(),
            payload["county_overlap_county"].tolist(),
            payload["county_overlap_population"].tolist(),
            strict=True,
        )
    ) == [
        (100100, 1001, 900),
        (100100, 1003, 100),
        (100200, 1003, 500),
        (200100, 2013, 400),
    ]
    assert payload["tract_overlap_tract"].tolist() == [
        1001000100,
        1003000100,
        1003000200,
        2013000100,
    ]
    metadata = json.loads(str(payload["metadata_json"]))
    assert metadata["kind"] == "us_puma_ladder"


def test_assemble_refuses_a_block_with_no_puma() -> None:
    with pytest.raises(ValueError, match="tract absent from the tract-to-PUMA"):
        assemble_us_puma_ladder(
            block_population={**_BLOCK_POPULATION, 30070001001000: 50},
            cd_by_block={**_CD_BY_BLOCK, 30070001001000: 301},
            tract_to_puma=_TRACT_TO_PUMA,  # no tract 3007000100
            metadata=_METADATA,
        )


def test_assemble_refuses_a_block_with_no_congressional_district() -> None:
    with pytest.raises(ValueError, match="no congressional district"):
        assemble_us_puma_ladder(
            block_population=_BLOCK_POPULATION,
            cd_by_block={
                key: value
                for key, value in _CD_BY_BLOCK.items()
                if key != 20130001001000
            },
            tract_to_puma=_TRACT_TO_PUMA,
            metadata=_METADATA,
        )


def test_assemble_ignores_zero_population_blocks() -> None:
    payload = assemble_us_puma_ladder(
        block_population={**_BLOCK_POPULATION, 10010001009000: 0},
        cd_by_block=_CD_BY_BLOCK,
        tract_to_puma=_TRACT_TO_PUMA,
        metadata=_METADATA,
    )

    # The zero-population block never needs a CD or tract lookup and does not
    # change any PUMA's conserved total.
    assert payload["puma_population"].tolist() == [1000, 500, 400]
    assert np.asarray(payload["cd_overlap_population"]).sum() == 1900
