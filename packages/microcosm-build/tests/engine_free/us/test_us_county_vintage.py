"""2020 geography resolves to complete, population-conserving 2023 counties."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.county_vintage import (
    COUNTY_FIPS_2023_COLUMN,
    county_fips_2023_from_blocks,
    county_fips_2023_from_tracts,
    load_ct_planning_region_by_town,
    load_ct_planning_region_by_tract,
)
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
from test_support.paths import paths_for

_PATHS = paths_for("microcosm-build")
_DATA = _PATHS.package / "src/microcosm/build/us_runtime/data"
_NATIONAL_SUPPORT = _PATHS.tests / "fixtures/us_county_vintage_tract_support.csv"
_CT_REGIONS = {f"09{county}" for county in range(110, 191, 10)}
_CT_OLD_COUNTIES = {f"09{county:03d}" for county in range(1, 16, 2)}


def _ct_tract_rows() -> list[dict[str, str]]:
    with (_DATA / "ct_2020_tract_to_2023_county.csv").open(newline="") as source:
        return list(csv.DictReader(source))


def test_original_crosswalk_assigns_all_169_towns_to_one_planning_region() -> None:
    with (_DATA / "ct_cou_to_cousub_crosswalk.txt").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        rows = csv.reader(source, delimiter="|")
        header = next(rows)
        columns = {name.split("\n", 1)[0]: index for index, name in enumerate(header)}
        regions_by_town: dict[str, set[str]] = {}
        for row in rows:
            if not row or not any(row):
                break
            if row[columns["COUSUBFP"]] == "00000":
                continue
            town = row[columns["OLD_COUSUB_GEOID"]]
            region = row[columns["STATEFP"]] + row[columns["NEW_COUNTYFP"]]
            regions_by_town.setdefault(town, set()).add(region)

    assert len(regions_by_town) == 169
    assert all(len(regions) == 1 for regions in regions_by_town.values())
    assert load_ct_planning_region_by_town() == {
        town: next(iter(regions)) for town, regions in regions_by_town.items()
    }
    assert set(load_ct_planning_region_by_town().values()) == _CT_REGIONS


def test_real_fairfield_blocks_follow_three_different_planning_regions() -> None:
    # Existing BAF examples: Bridgeport, Shelton and Stamford all have the
    # same 2020 county prefix, so county-only replacement cannot resolve them.
    blocks = ["090010701001000", "090011101001000", "090010201011000"]

    assert COUNTY_FIPS_2023_COLUMN == "county_fips_2023"
    assert {block[:5] for block in blocks} == {"09001"}
    assert county_fips_2023_from_blocks(blocks).tolist() == [
        "09120",
        "09140",
        "09190",
    ]
    assert county_fips_2023_from_tracts([block[:11] for block in blocks]).tolist() == [
        "09120",
        "09140",
        "09190",
    ]


@pytest.mark.parametrize("as_integer", [False, True])
def test_every_source_tract_gets_one_value_with_non_ct_identity(as_integer) -> None:
    # This is the full county support universe, not a rebuilt ladder artifact:
    # one Census tract per non-CT county and every town-defined CT tract.
    with _NATIONAL_SUPPORT.open(newline="") as source:
        tracts = [row["tract_geoid"] for row in csv.DictReader(source)]
    values = [int(tract) for tract in tracts] if as_integer else tracts
    counties = county_fips_2023_from_tracts(values)
    legacy = np.asarray([tract[:5] for tract in tracts])
    non_ct = np.asarray([not tract.startswith("09") for tract in tracts])

    assert len(tracts) == len(set(tracts)) == 4015
    assert len(set(legacy)) == 3143
    assert {tract[:2] for tract in tracts} == set(US_STATE_FIPS_TO_POSTAL)
    assert counties.shape == (len(tracts),)
    assert not pd.isna(counties).any()
    assert all(isinstance(county, str) and len(county) == 5 for county in counties)
    np.testing.assert_array_equal(counties[non_ct], legacy[non_ct])
    assert len(set(counties)) == 3144
    assert set(counties[~non_ct]) == _CT_REGIONS
    assert not set(counties).intersection(_CT_OLD_COUNTIES)


def test_national_support_fixture_matches_its_provenance_digest() -> None:
    provenance = json.loads(
        _NATIONAL_SUPPORT.with_suffix(".csv.provenance.json").read_text()
    )

    assert (
        hashlib.sha256(_NATIONAL_SUPPORT.read_bytes()).hexdigest()
        == provenance["output_sha256"]
    )
    source = provenance["source_files"]["tract_to_puma"]
    assert source["url"].startswith("https://www2.census.gov/")
    assert len(source["sha256"]) == 64


def test_ct_tract_mapping_covers_population_and_conserves_county_totals() -> None:
    rows = _ct_tract_rows()
    tracts = [row["tract_geoid"] for row in rows]
    population = np.asarray(
        [int(row["population_2020"]) for row in rows], dtype=np.int64
    )
    legacy = np.asarray([tract[:5] for tract in tracts])
    counties = county_fips_2023_from_tracts(tracts)
    legacy_totals = pd.Series(population).groupby(legacy).sum()
    region_totals = pd.Series(population).groupby(counties).sum()

    assert len(tracts) == len(set(tracts)) == 880
    assert set(legacy_totals.index) == _CT_OLD_COUNTIES
    assert set(region_totals.index) == _CT_REGIONS
    assert legacy_totals.sum() == region_totals.sum() == population.sum() == 3_605_944
    assert not pd.isna(counties).any()
    assert load_ct_planning_region_by_tract() == dict(
        zip(tracts, counties, strict=True)
    )


@pytest.mark.parametrize(
    ("mapper", "valid"),
    [
        (county_fips_2023_from_blocks, "090010701001000"),
        (county_fips_2023_from_tracts, "09001070100"),
    ],
)
def test_mapping_preserves_empty_vectors_duplicates_and_input_order(
    mapper, valid
) -> None:
    assert mapper([]).shape == (0,)
    assert mapper([valid, int(valid), valid]).tolist() == ["09120"] * 3
    assert mapper(iter([valid, valid])).tolist() == ["09120"] * 2


@pytest.mark.parametrize("invalid", [None, np.nan, pd.NA, "", "invalid", True, 1.5])
@pytest.mark.parametrize(
    "mapper", [county_fips_2023_from_blocks, county_fips_2023_from_tracts]
)
def test_mapping_refuses_missing_or_invalid_geography(mapper, invalid) -> None:
    with pytest.raises(ValueError):
        mapper([invalid])


@pytest.mark.parametrize(
    ("mapper", "unknown"),
    [
        (county_fips_2023_from_blocks, "090019999991000"),
        (county_fips_2023_from_tracts, "09001999999"),
    ],
)
def test_mapping_refuses_unknown_ct_tracts_instead_of_returning_null(mapper, unknown):
    with pytest.raises(ValueError, match="Connecticut|tract"):
        mapper([unknown])


def test_mapping_refuses_undefined_water_block_even_when_its_tract_is_mapped():
    # The original BAF assigns no town to this coastal-water block. Other
    # blocks in tract 09001990000 have towns, so a tract-prefix-only lookup
    # would silently give this uninhabitable block a planning region.
    assert "09001990000" in load_ct_planning_region_by_tract()
    with pytest.raises(ValueError, match="no defined town"):
        county_fips_2023_from_blocks(["090019900000001"])


@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(
                [
                    ("090010701001000", "09120"),
                    ("090011101001000", "09140"),
                    ("090010201011000", "09190"),
                    ("090091401011000", "09170"),
                    ("010010201001000", "01001"),
                    ("020130001001000", "02013"),
                    ("110010062021000", "11001"),
                    ("360610001001000", "36061"),
                ]
            ),
            st.integers(min_value=0, max_value=10**9),
            st.booleans(),
        ),
        max_size=100,
    )
)
@settings(max_examples=75, deadline=None)
def test_mapping_is_complete_order_preserving_and_conserves_generated_mass(rows):
    blocks = [int(pair[0]) if integral else pair[0] for pair, _, integral in rows]
    expected = [pair[1] for pair, _, _ in rows]
    population = np.asarray([mass for _, mass, _ in rows], dtype=np.int64)

    counties = county_fips_2023_from_blocks(blocks)
    assert counties.tolist() == expected
    assert not pd.isna(counties).any()
    np.testing.assert_array_equal(
        county_fips_2023_from_blocks(blocks[::-1]), counties[::-1]
    )
    assert (
        county_fips_2023_from_tracts([pair[0][:11] for pair, _, _ in rows]).tolist()
        == expected
    )
    assert pd.Series(population).groupby(counties).sum().sum() == population.sum()


def test_existing_full_ladder_has_3144_counties_without_rebuilding() -> None:
    artifact = os.environ.get("MICROCOSM_US_BLOCK_LADDER_ARTIFACT")
    if not artifact:
        pytest.skip(
            "Set MICROCOSM_US_BLOCK_LADDER_ARTIFACT to an existing national NPZ."
        )
    from microcosm.build.us_runtime.geography_ladder import load_us_block_ladder

    ladder = load_us_block_ladder(Path(artifact))
    states = {f"{int(state):02d}" for state in np.unique(ladder.block_geoid // 10**13)}
    assert states == set(US_STATE_FIPS_TO_POSTAL)
    counties = county_fips_2023_from_blocks(ladder.block_geoid)
    legacy = np.asarray([f"{int(block) // 10**10:05d}" for block in ladder.block_geoid])
    is_ct = (ladder.block_geoid // 10**13) == 9

    assert not pd.isna(counties).any()
    assert len(set(counties)) == 3144
    np.testing.assert_array_equal(counties[~is_ct], legacy[~is_ct])
    ct_population = pd.Series(ladder.population[is_ct])
    assert (
        ct_population.groupby(legacy[is_ct]).sum().sum()
        == ct_population.groupby(counties[is_ct]).sum().sum()
    )
