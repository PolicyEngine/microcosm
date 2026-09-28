"""CPS ASEC source geography for the block-location draw.

Covers the List 4 parser (Census's entirely-identified county list), the join
of ASEC household geography onto frame households, and the draw inputs it
builds: identified-county households draw within their county, ``GTCO = 0``
households within their state minus the doubly confirmed counties, and the
CBSA narrowing turns on only when the source's delineation matches the
ladder's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.block_location import (
    CpsIdentifiedCountyList,
    draw_us_block_locations,
    load_us_location_ladder,
)
from microcosm.build.us_runtime.cps_identified_county_sources import (
    parse_cps_identified_county_list,
)
from microcosm.build.us_runtime.cps_source_geography import (
    CPS_ASEC_SURVEY_YEAR_OFFSET,
    cps_source_geography,
    household_source_keys,
    load_cps_household_geography,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_block_location import (
    synthetic_blocks,
    write_location_ladder,
)

_RENDERED = """\
 List 4: FIPS County Code ........................................ F-18

SPECIFIC METROPOLITAN AREAS                                        F-17
                     List 4: FIPS County Codes
Please note that these county codes must be used in conjunction with state codes to create
unique county identifiers as county codes start with 001 in each state. Counties are only
included on this list if the entire county is identified.

FIPS
County               County
Code                 Name                 State

                                         Alabama

003                  Baldwin
081                  Lee

SPECIFIC METROPOLITAN AREAS                       F-25
                                   Connecticut

005               Litchfield*
                                    New York

061               New York
* Counties marked with an asterisk (*) are also single county Micropolitan
12300         Augusta-Waterville, ME                       Kennebec              005
"""


def test_parser_reads_list_4_and_stops_at_its_footnote() -> None:
    counties = parse_cps_identified_county_list(_RENDERED.splitlines())
    assert [(c.state_fips, c.county_code) for c in counties] == [
        ("01", "003"),
        ("01", "081"),
        ("09", "005"),
        ("36", "061"),
    ]
    assert [c.single_county_micropolitan for c in counties] == [
        False,
        False,
        True,
        False,
    ]
    assert counties[3].county_name == "New York"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("nothing here", "no 'List 4"),
        ("List 4: FIPS County Codes\n003    Baldwin\n", "precedes every state"),
        (
            "List 4: FIPS County Codes\nAlabama\n003    Baldwin\n003    Baldwin\n",
            "listed twice",
        ),
        ("List 4: FIPS County Codes\nAlabama\n* Counties marked\n", "no counties"),
    ],
)
def test_parser_refuses_malformed_lists(text, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_cps_identified_county_list(text.splitlines())


def _cps_table() -> pd.DataFrame:
    """Two income years of synthetic ASEC households over the synthetic ladder:
    county 01001 coded in both years, 36001 coded only in 2024, 01003 never."""

    rows = []
    for year in (2023, 2024):
        for seq, (state, gtco, cbsa) in enumerate(
            [
                (1, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (36, 1 if year == 2024 else 0, 0),
                (36, 0, 0),
                (2, 0, 0),
            ],
            start=1,
        ):
            rows.append((year, 100 * year + seq, state, gtco, cbsa, 1500.0))
    return pd.DataFrame(
        rows,
        columns=[
            "source_year",
            "source_household_id",
            "state_fips",
            "gtco",
            "gtcbsa",
            "person_weight",
        ],
    )


def _official(counties: set[int]) -> dict[int, CpsIdentifiedCountyList]:
    return {
        year + CPS_ASEC_SURVEY_YEAR_OFFSET: CpsIdentifiedCountyList(
            asec_year=year + CPS_ASEC_SURVEY_YEAR_OFFSET,
            counties=frozenset(counties),
            source=f"https://example.test/cpsmar{(year + 1) % 100}.pdf",
            source_sha256="f" * 64,
        )
        for year in (2023, 2024)
    }


def test_source_geography_joins_each_household_and_builds_the_rule(tmp_path) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    cps = _cps_table()
    frame_rows = cps.sample(frac=1, random_state=0).reset_index(drop=True)
    geography = cps_source_geography(
        ladder,
        cps,
        source_year=frame_rows["source_year"],
        source_household_id=frame_rows["source_household_id"],
        state_fips=frame_rows["state_fips"],
        official=_official({1001, 36001, 1005}),
    )
    coded = frame_rows["gtco"].to_numpy() > 0
    assert (
        geography.county_fips[coded]
        == frame_rows["state_fips"].to_numpy()[coded] * 1000 + 1
    ).all()
    assert (geography.county_fips[~coded] == 0).all()
    assert (geography.identification_group[coded] == -1).all()
    assert (
        geography.identification_group[~coded]
        == frame_rows["source_year"].to_numpy()[~coded]
    ).all()
    # Excluded only with both confirmations, per year: 36001 is coded in 2024
    # only; 1005 is listed but never coded.
    assert geography.identified_counties == {
        2023: frozenset({1001}),
        2024: frozenset({1001, 36001}),
    }
    groups = geography.record["groups"]
    assert groups["2023"]["listed_not_coded"] == ["01005", "36001"]
    assert groups["2024"]["asec_year"] == 2025
    assert geography.record["counties_whose_exclusion_varies_across_groups"] == [
        "36001"
    ]
    assert geography.cbsa_code is None
    assert geography.unidentified_county_share is None
    # Draws honour the rule.
    draw = draw_us_block_locations(
        ladder,
        household_key=np.arange(len(frame_rows)),
        state_fips=frame_rows["state_fips"],
        seed=3,
        county_fips=geography.county_fips,
        identification_group=geography.identification_group,
        identified_counties=geography.identified_counties,
    )
    county = ladder.block_geoid[draw.block_index] // 10**10
    assert (county[coded] == geography.county_fips[coded]).all()
    for row in np.flatnonzero(~coded):
        year = int(frame_rows.at[row, "source_year"])
        assert county[row] not in geography.identified_counties[year]


def test_source_geography_refuses_missing_rows_and_state_disagreement(
    tmp_path,
) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    cps = _cps_table()
    with pytest.raises(ValueError, match="no ASEC source row"):
        cps_source_geography(
            ladder,
            cps,
            source_year=[2023],
            source_household_id=[999],
            state_fips=[1],
            official=_official(set()),
        )
    with pytest.raises(ValueError, match="GESTFIPS"):
        cps_source_geography(
            ladder,
            cps,
            source_year=[2023],
            source_household_id=[202301],
            state_fips=[2],
            official=_official(set()),
        )


def test_cbsa_narrowing_applies_only_when_the_delineations_agree(tmp_path) -> None:
    arrays = synthetic_blocks(8)
    arrays["cbsa_code"] = np.where(
        arrays["block_geoid"] // 10**10 == 1001, 11111, 0
    ).astype(np.int32)
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz", arrays))
    cps = _cps_table()
    cps.loc[cps["gtco"] > 0, "gtcbsa"] = np.where(
        cps.loc[cps["gtco"] > 0, "state_fips"] == 1, 11111, 0
    )
    agreed = cps_source_geography(
        ladder,
        cps,
        source_year=cps["source_year"],
        source_household_id=cps["source_household_id"],
        state_fips=cps["state_fips"],
        official=_official({1001}),
    )
    assert agreed.record["cbsa_narrowing"]["applied"]
    assert agreed.cbsa_code is not None
    off = cps_source_geography(
        ladder,
        cps,
        source_year=cps["source_year"],
        source_household_id=cps["source_household_id"],
        state_fips=cps["state_fips"],
        official=_official({1001}),
        cbsa_narrowing=False,
    )
    assert not off.record["cbsa_narrowing"]["applied"]
    cps.loc[(cps["state_fips"] == 1) & (cps["gtco"] > 0), "gtcbsa"] = 22222
    disagreed = cps_source_geography(
        ladder,
        cps,
        source_year=cps["source_year"],
        source_household_id=cps["source_household_id"],
        state_fips=cps["state_fips"],
        official=_official({1001}),
    )
    assert not disagreed.record["cbsa_narrowing"]["applied"]
    assert (
        disagreed.record["cbsa_narrowing"]["vintage_agreement"]["disagreeing_pairs"]
        == 1
    )


def test_load_household_geography_reads_each_years_file(tmp_path) -> None:
    sources = {}
    for year in (2023, 2024):
        path = tmp_path / f"census_cps_{year}.h5"
        pd.DataFrame(
            {
                "H_SEQ": [1, 2],
                "GESTFIPS": [1, 36],
                "GTCO": [1, 0],
                "GTCBSA": [0, 35620],
                "HSUP_WGT": [150000.0, 90000.0],
                "H_NUMPER": [2, 3],
            }
        ).to_hdf(path, key="household")
        sources[year] = path
    table = load_cps_household_geography(sources)
    assert table["source_year"].tolist() == [2023, 2023, 2024, 2024]
    assert table["gtco"].tolist() == [1, 0, 1, 0]
    assert table["person_weight"].tolist() == [300000.0, 270000.0] * 2
    pd.DataFrame(
        {
            "H_SEQ": [1, 1],
            "GESTFIPS": [1, 1],
            "GTCO": [0, 0],
            "GTCBSA": [0, 0],
            "HSUP_WGT": [1.0, 1.0],
            "H_NUMPER": [1, 1],
        }
    ).to_hdf(tmp_path / "dup.h5", key="household")
    with pytest.raises(ValueError, match="repeats H_SEQ"):
        load_cps_household_geography({2022: tmp_path / "dup.h5"})


def _frame_with_source_keys(keys: list[tuple[int, int]]) -> Frame:
    household_ids = np.arange(1, len(keys) + 1, dtype=np.int64)
    person_household = np.repeat(household_ids, 2)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.arange(1, len(person_household) + 1),
                "person_household_id": person_household,
                "person_tax_unit_id": person_household,
                "person_spm_unit_id": person_household,
                "person_family_id": person_household,
                "person_marital_unit_id": person_household,
                "source_year": np.repeat([k[0] for k in keys], 2),
                "source_household_id": np.repeat([k[1] for k in keys], 2),
            }
        ),
        "household": pd.DataFrame(
            {"household_id": household_ids, "state_fips": np.ones(len(keys), int)}
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids}),
        "family": pd.DataFrame({"family_id": household_ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household_ids}),
    }
    weights = {"household": Weights(np.ones(len(keys)), WeightKind.DESIGN)}
    return Frame(tables, US_SCHEMA, weights)


def test_household_source_keys_come_from_members() -> None:
    frame = _frame_with_source_keys([(2023, 7), (2024, 7), (2023, 9)])
    year, household = household_source_keys(frame)
    assert year.tolist() == [2023, 2024, 2023]
    assert household.tolist() == [7, 7, 9]
    person = frame.table("person").copy()
    person.loc[1, "source_household_id"] = 8
    broken = Frame(
        {**{e: frame.table(e) for e in frame.entities}, "person": person},
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    with pytest.raises(ValueError, match="disagree"):
        household_source_keys(broken)


def test_guarantee_is_read_from_the_list_preamble() -> None:
    from microcosm.build.us_runtime.cps_identified_county_sources import (
        list_carries_entire_county_guarantee,
    )

    assert list_carries_entire_county_guarantee(_RENDERED.splitlines())
    without = _RENDERED.replace(
        " Counties are only\nincluded on this list if the entire county is identified.",
        "",
    )
    assert "entire county" not in without
    assert not list_carries_entire_county_guarantee(without.splitlines())
    # The sentence may wrap anywhere across the preamble's lines.
    wrapped = _RENDERED.replace(
        "included on this list if the entire county is identified.",
        "included on this list if the entire\n   county is identified.",
    )
    assert list_carries_entire_county_guarantee(wrapped.splitlines())


def test_packaged_lists_record_which_years_carry_the_guarantee() -> None:
    from microcosm.build.us_runtime.block_location import (
        load_cps_asec_identified_counties,
    )

    lists = load_cps_asec_identified_counties()
    assert {year: entry.entire_county_guarantee for year, entry in lists.items()} == {
        2023: True,
        2024: True,
        2025: True,
        2026: False,
    }


def test_a_year_without_the_guarantee_excludes_nothing() -> None:
    """ASEC 2026 drops the whole-county sentence and its file has GTCO=0
    households inside listed, fully coded counties (all of Delaware), so for
    such a year a GTCO=0 record draws from its whole state."""

    from microcosm.build.us_runtime.block_location import cps_excluded_county_sets

    listed = {
        2025: CpsIdentifiedCountyList(
            asec_year=2025,
            counties=frozenset({1001, 1003, 1005}),
            source="https://example.test/cpsmar25.pdf",
            source_sha256="a" * 64,
            entire_county_guarantee=True,
        ),
        2026: CpsIdentifiedCountyList(
            asec_year=2026,
            counties=frozenset({1001, 1003, 1005}),
            source="https://example.test/cpsmar26.pdf",
            source_sha256="b" * 64,
            entire_county_guarantee=False,
        ),
    }
    coded = {2024: {1001, 1003}, 2025: {1001, 1003}}
    excluded, record = cps_excluded_county_sets(
        coded, listed, asec_year_of_group={2024: 2025, 2025: 2026}
    )
    assert excluded == {2024: frozenset({1001, 1003}), 2025: frozenset()}
    assert record["groups"]["2025"]["basis"] == "no_whole_county_guarantee_state_draw"
    assert record["groups"]["2025"]["listed_and_coded_counties"] == 2
    assert record["groups"]["2024"]["basis"] == "official_list_and_coded"


def test_every_state_keeps_candidate_blocks_for_its_gtco_zero_records(
    tmp_path,
) -> None:
    """A state whose every county is excluded would leave its GTCO=0 records
    nowhere to go; the draw fails closed on that, never falls back."""

    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    counties = {int(c) for c in np.unique(ladder.block_geoid // 10**10)}
    state_one = {c for c in counties if c // 1000 == 1}
    with pytest.raises(ValueError, match="no candidate block"):
        draw_us_block_locations(
            ladder,
            household_key=[1],
            state_fips=[1],
            seed=0,
            identification_group=[2024],
            identified_counties={2024: frozenset(state_one)},
        )


def test_parser_stops_at_the_next_section_without_a_footnote() -> None:
    """The 2026 documentation has no footnote after List 4; the list ends at
    the first line that is not a state, county or page header."""

    rendered = _RENDERED.split("* Counties marked")[0] + (
        "                Source of the Data and Accuracy of the Estimates\n"
        "001          Not a county\n"
    )
    counties = parse_cps_identified_county_list(rendered.splitlines())
    assert [(c.state_fips, c.county_code) for c in counties][-1] == ("36", "061")
    assert len(counties) == 4


def test_the_official_list_must_match_its_provenance_digest(
    tmp_path, monkeypatch
) -> None:
    import shutil

    from microcosm.build.us_runtime import block_location
    from microcosm.build.us_runtime.block_location import (
        default_cps_asec_identified_counties_path,
        load_cps_asec_identified_counties,
    )

    packaged = default_cps_asec_identified_counties_path()
    copy = tmp_path / packaged.name
    shutil.copy(packaged, copy)
    shutil.copy(
        packaged.with_name(packaged.name + ".provenance.json"),
        copy.with_name(copy.name + ".provenance.json"),
    )
    assert load_cps_asec_identified_counties(copy)
    copy.write_text(copy.read_text().replace(",003,Baldwin,", ",005,Baldwin,", 1))
    with pytest.raises(ValueError, match="provenance records"):
        load_cps_asec_identified_counties(copy)
    # The packaged list must ship with its provenance.
    bare = tmp_path / "bare" / packaged.name
    bare.parent.mkdir()
    shutil.copy(packaged, bare)
    monkeypatch.setattr(
        block_location, "default_cps_asec_identified_counties_path", lambda: bare
    )
    with pytest.raises(FileNotFoundError, match="lacks"):
        load_cps_asec_identified_counties()


def test_record_names_the_packaged_list_digest(tmp_path) -> None:
    from microcosm.build.us_runtime.block_location import (
        default_cps_asec_identified_counties_path,
        file_sha256,
    )

    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    cps = _cps_table()
    geography = cps_source_geography(
        ladder,
        cps,
        source_year=cps["source_year"],
        source_household_id=cps["source_household_id"],
        state_fips=cps["state_fips"],
    )
    assert geography.record["official_list_csv_sha256"] == file_sha256(
        default_cps_asec_identified_counties_path()
    )


def test_remainder_weighting_never_applies_to_a_year_without_the_guarantee(
    tmp_path,
) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    cps = _cps_table()
    lists = _official({1001})
    lists[2025] = CpsIdentifiedCountyList(
        asec_year=2025,
        counties=frozenset({1001}),
        source="https://example.test/cpsmar25.pdf",
        source_sha256="c" * 64,
        entire_county_guarantee=False,
    )
    geography = cps_source_geography(
        ladder,
        cps,
        source_year=cps["source_year"],
        source_household_id=cps["source_household_id"],
        state_fips=cps["state_fips"],
        official=lists,
        partial_county_remainder=True,
    )
    assert geography.identified_counties[2024] == frozenset()
    assert 2024 not in geography.unidentified_county_share
    assert 2023 in geography.unidentified_county_share
    assert geography.record["partial_county_remainder"][
        "skipped_groups_without_guarantee"
    ] == [2024]
