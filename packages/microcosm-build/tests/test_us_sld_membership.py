"""Tests for 2024-vintage SLD membership assignment (populace#625)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.sld_membership import (
    SLD_ASSIGNMENT_METHODS,
    SLD_MEMBERSHIP_COLUMNS,
    UsSldMembershipLadder,
    assign_us_sld_membership,
    load_us_sld_membership_ladder,
    parse_national_sld24_bef,
    us_sld_membership_gate,
)


def _ladder() -> UsSldMembershipLadder:
    """Two Utah tracts and one Nebraska tract, with matching cells.

    Tract 49049000100 lies wholly in (005, 040); tract 49049000200 splits
    60/40 between (005, 040) and (006, 041). The Nebraska tract exercises
    the unicameral no-lower-chamber path (sldl == ""). Keys follow the
    national conventions (puma = state*100000+puma5, cd = state*100+district,
    county = state*1000+county).
    """
    return UsSldMembershipLadder(
        tract_geoid=np.array(
            [49049000100, 49049000200, 49049000200, 31055000100],
            dtype=np.int64,
        ),
        tract_sldu=np.array(["005", "005", "006", "010"]),
        tract_sldl=np.array(["040", "040", "041", ""]),
        tract_population=np.array([1000, 600, 400, 500], dtype=np.int64),
        cell_puma=np.array([4904901, 4904901, 4904901, 3103101], dtype=np.int64),
        cell_cd=np.array([4903, 4903, 4903, 3102], dtype=np.int64),
        cell_county=np.array([49049, 49049, 49049, 31055], dtype=np.int64),
        cell_sldu=np.array(["005", "005", "006", "010"]),
        cell_sldl=np.array(["040", "040", "041", ""]),
        cell_population=np.array([1000, 600, 400, 500], dtype=np.int64),
    )


def _save_ladder(path, ladder: UsSldMembershipLadder, **overrides) -> None:
    arrays = {
        "tract_geoid": ladder.tract_geoid,
        "tract_sldu": ladder.tract_sldu,
        "tract_sldl": ladder.tract_sldl,
        "tract_population": ladder.tract_population,
        "cell_puma": ladder.cell_puma,
        "cell_cd": ladder.cell_cd,
        "cell_county": ladder.cell_county,
        "cell_sldu": ladder.cell_sldu,
        "cell_sldl": ladder.cell_sldl,
        "cell_population": ladder.cell_population,
        "meta_boundary_vintage": np.array("2024_state_legislative_districts"),
        "meta_source_kind": np.array("census_2024_sld_bef"),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)


def test_parse_national_sld24_bef_parses_and_drops_unassigned():
    lines = [
        "GEOID,SLDUST",
        "490490001001000,005",
        "490490001001001,ZZZ",
        "310550001001000,010",
    ]
    parsed = parse_national_sld24_bef(lines, chamber="upper")
    assert parsed == {490490001001000: "005", 310550001001000: "010"}


def test_parse_national_sld24_bef_refuses_wrong_header_and_chamber():
    with pytest.raises(ValueError, match="GEOID,SLDLST"):
        parse_national_sld24_bef(["GEOID,SLDUST"], chamber="lower")
    with pytest.raises(ValueError, match="chamber"):
        parse_national_sld24_bef(["GEOID,SLDUST"], chamber="senate")
    with pytest.raises(ValueError, match="15 digits"):
        parse_national_sld24_bef(
            ["GEOID,SLDUST", "12345,005"],
            chamber="upper",
        )
    with pytest.raises(ValueError, match="3 characters"):
        parse_national_sld24_bef(
            ["GEOID,SLDUST", "490490001001000,05"],
            chamber="upper",
        )


def test_load_refuses_unconserved_population(tmp_path):
    ladder = _ladder()
    path = tmp_path / "ladder.npz"
    _save_ladder(path, ladder, cell_population=ladder.cell_population + 1)
    with pytest.raises(ValueError, match="not conserved"):
        load_us_sld_membership_ladder(path)


def test_load_refuses_missing_or_wrong_vintage_metadata(tmp_path):
    ladder = _ladder()
    bare = tmp_path / "bare.npz"
    np.savez(
        bare,
        tract_geoid=ladder.tract_geoid,
        tract_sldu=ladder.tract_sldu,
        tract_sldl=ladder.tract_sldl,
        tract_population=ladder.tract_population,
        cell_puma=ladder.cell_puma,
        cell_cd=ladder.cell_cd,
        cell_county=ladder.cell_county,
        cell_sldu=ladder.cell_sldu,
        cell_sldl=ladder.cell_sldl,
        cell_population=ladder.cell_population,
    )
    with pytest.raises(ValueError, match="meta_boundary_vintage"):
        load_us_sld_membership_ladder(bare)
    stale = tmp_path / "stale.npz"
    _save_ladder(
        stale,
        ladder,
        meta_boundary_vintage=np.array("2020_baf"),
    )
    with pytest.raises(ValueError, match="boundary vintage"):
        load_us_sld_membership_ladder(stale)


def test_load_round_trips_a_valid_artifact(tmp_path):
    ladder = _ladder()
    path = tmp_path / "ladder.npz"
    _save_ladder(path, ladder)
    loaded = load_us_sld_membership_ladder(path)
    assert loaded.tract_geoid.tolist() == ladder.tract_geoid.tolist()
    assert loaded.district_codes("upper")["49"] == frozenset({"005", "006"})
    assert loaded.district_codes("lower")["49"] == frozenset({"040", "041"})
    assert loaded.district_codes("upper")["31"] == frozenset({"010"})
    assert "31" not in loaded.district_codes("lower")


def test_assign_tract_exact_and_split_draw_are_deterministic():
    households = pd.DataFrame(
        {
            "state_fips": [49, 49],
            "tract_geoid": [49049000100, 49049000200],
        }
    )
    first = assign_us_sld_membership(households, _ladder(), seed=7)
    second = assign_us_sld_membership(households, _ladder(), seed=7)
    assert list(first.columns[-3:]) == list(SLD_MEMBERSHIP_COLUMNS)
    assert first["sld_assignment_method"].tolist() == [
        "tract_exact",
        "tract_split_draw",
    ]
    assert first["sld_upper_code"][0] == "005"
    assert first["sld_lower_code"][0] == "040"
    assert first["sld_upper_code"][1] in {"005", "006"}
    pd.testing.assert_frame_equal(first, second)


def test_assign_split_draw_follows_population_shares():
    households = pd.DataFrame(
        {
            "state_fips": [49] * 400,
            "tract_geoid": [49049000200] * 400,
        }
    )
    assigned = assign_us_sld_membership(households, _ladder(), seed=0)
    share_005 = float((assigned["sld_upper_code"] == "005").mean())
    assert 0.5 < share_005 < 0.7  # population share is 0.6


def test_assign_acs_rows_condition_on_puma_cd_county_with_fallbacks():
    households = pd.DataFrame(
        {
            "state_fips": [49, 49, 49, 49],
            "puma": [4904901, 4904901, 4904901, 4904999],
            "congressional_district_geoid": [4903, 4999, None, 4903],
            "county_fips": [49049, 49049, None, 49049],
        }
    )
    assigned = assign_us_sld_membership(households, _ladder(), seed=1)
    assert assigned["sld_assignment_method"].tolist() == [
        "puma_cd_county_draw",
        "puma_county_draw",
        "puma_draw",
        "unassigned",
    ]
    assert assigned["sld_upper_code"][0] in {"005", "006"}
    assert assigned["sld_upper_code"][3] == ""


def test_assign_ignores_cross_state_geography_components():
    """A wrong-state PUMA/CD/county must never yield a same-code district.

    District codes repeat across states (California also has an 005), so a
    cross-state key that happens to exist elsewhere would otherwise assign
    a plausible-looking wrong district — the reviewed exploit.
    """
    households = pd.DataFrame(
        {
            "state_fips": [49],
            # A California-convention PUMA/CD/county on a Utah row.
            "puma": [600101],
            "congressional_district_geoid": [652],
            "county_fips": [6037],
        }
    )
    assigned = assign_us_sld_membership(households, _ladder(), seed=0)
    assert assigned["sld_assignment_method"].tolist() == ["unassigned"]
    assert assigned["sld_upper_code"][0] == ""


def test_assign_falls_through_zero_population_cells():
    ladder = UsSldMembershipLadder(
        tract_geoid=np.array([49049000100], dtype=np.int64),
        tract_sldu=np.array(["005"]),
        tract_sldl=np.array(["040"]),
        tract_population=np.array([100], dtype=np.int64),
        cell_puma=np.array([4904901, 4904901], dtype=np.int64),
        cell_cd=np.array([4903, 4904], dtype=np.int64),
        cell_county=np.array([49049, 49049], dtype=np.int64),
        cell_sldu=np.array(["005", "006"]),
        cell_sldl=np.array(["040", "041"]),
        cell_population=np.array([0, 100], dtype=np.int64),
    )
    households = pd.DataFrame(
        {
            "state_fips": [49],
            "puma": [4904901],
            "congressional_district_geoid": [4903],
            "county_fips": [49049],
        }
    )
    assigned = assign_us_sld_membership(households, ladder, seed=0)
    # The exact (puma, cd, county) cell exists but has zero population, so
    # the draw falls through to (puma, county), which has support.
    assert assigned["sld_assignment_method"].tolist() == ["puma_county_draw"]
    assert assigned["sld_upper_code"][0] == "006"


def test_assign_nebraska_rows_carry_empty_lower_codes():
    households = pd.DataFrame(
        {
            "state_fips": [31],
            "tract_geoid": [31055000100],
        }
    )
    assigned = assign_us_sld_membership(households, _ladder(), seed=0)
    assert assigned["sld_upper_code"][0] == "010"
    assert assigned["sld_lower_code"][0] == ""
    assert assigned["sld_assignment_method"][0] == "tract_exact"


def test_gate_passes_a_clean_assignment_and_counts_methods():
    households = pd.DataFrame(
        {
            "state_fips": [49, 49, 31],
            "tract_geoid": [49049000100, 49049000200, 31055000100],
        }
    )
    assigned = assign_us_sld_membership(households, _ladder(), seed=0)
    gate = us_sld_membership_gate(assigned, _ladder())
    assert gate.passed, gate.failures
    assert set(gate.details["method_counts"]) <= set(SLD_ASSIGNMENT_METHODS)
    assert gate.details["n_rows"] == 3


def test_gate_flags_codes_outside_the_state_district_set():
    assigned = pd.DataFrame(
        {
            "state_fips": [49],
            "sld_upper_code": ["999"],
            "sld_lower_code": ["040"],
            "sld_assignment_method": ["tract_exact"],
        }
    )
    gate = us_sld_membership_gate(assigned, _ladder())
    assert not gate.passed
    assert any("upper-chamber code" in failure for failure in gate.failures)


def test_gate_flags_missing_lower_outside_unicameral_states():
    assigned = pd.DataFrame(
        {
            "state_fips": [49],
            "sld_upper_code": ["005"],
            "sld_lower_code": [""],
            "sld_assignment_method": ["puma_draw"],
        }
    )
    gate = us_sld_membership_gate(assigned, _ladder())
    assert not gate.passed
    assert any("lower-chamber code" in failure for failure in gate.failures)


def test_gate_flags_excess_unassigned_share():
    assigned = pd.DataFrame(
        {
            "state_fips": [49, 49],
            "sld_upper_code": ["005", ""],
            "sld_lower_code": ["040", ""],
            "sld_assignment_method": ["tract_exact", "unassigned"],
        }
    )
    gate = us_sld_membership_gate(assigned, _ladder())
    assert not gate.passed
    assert any("unassigned row share" in failure for failure in gate.failures)


def test_assemble_builds_conserved_overlap_tables():
    from microcosm.build.us_runtime.sld_membership import (
        assemble_us_sld_membership_ladder,
    )

    block = lambda state, tract, suffix: int(  # noqa: E731
        f"{state:02d}{tract:09d}{suffix:04d}"
    )
    ut_a = block(49, 49049000100 % 10**9, 1000)
    ut_b = block(49, 49049000100 % 10**9, 1001)
    ne_a = block(31, 31055000100 % 10**9, 1000)
    arrays = assemble_us_sld_membership_ladder(
        block_population={ut_a: 60, ut_b: 40, ne_a: 25},
        sldu_by_block={ut_a: "005", ut_b: "006", ne_a: "010"},
        sldl_by_block={ut_a: "040", ut_b: "041"},
        cd_by_block={ut_a: 4903, ut_b: 4903},
        tract_to_puma={ut_a // 10**4: 4901, ne_a // 10**4: 3101},
    )
    assert arrays["tract_population"].sum() == 125
    assert arrays["cell_population"].sum() == 125
    assert arrays["n_assigned_blocks"][0] == 3
    assert arrays["n_blocks_without_cd"][0] == 1  # the Nebraska block
    lower_codes = set(arrays["cell_sldl"].tolist())
    assert "" in lower_codes  # Nebraska has no lower chamber
    ne_rows = arrays["cell_puma"] == 3101
    assert arrays["cell_cd"][ne_rows].tolist() == [0]


def test_assemble_refuses_populated_block_without_puma_mapping():
    import pytest as _pytest

    from microcosm.build.us_runtime.sld_membership import (
        assemble_us_sld_membership_ladder,
    )

    populated = 49049000100 * 10**4 + 1000
    with _pytest.raises(ValueError, match="no PUMA mapping"):
        assemble_us_sld_membership_ladder(
            block_population={populated: 10},
            sldu_by_block={populated: "005"},
            sldl_by_block={},
            cd_by_block={},
            tract_to_puma={},
        )
    # A zero-population unmapped block is silently outside the universe.
    arrays = assemble_us_sld_membership_ladder(
        block_population={populated: 0, populated + 1: 5},
        sldu_by_block={populated: "005", populated + 1: "005"},
        sldl_by_block={populated + 1: "040"},
        cd_by_block={},
        tract_to_puma={(populated + 1) // 10**4: 4901},
    )
    assert arrays["tract_population"].sum() == 5
