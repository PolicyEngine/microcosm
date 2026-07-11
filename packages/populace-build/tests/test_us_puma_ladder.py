"""US PUMA-anchored geography-ladder tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import (
    US_PUMA_LADDER_COLUMNS,
    assign_us_puma_ladder,
    load_us_puma_ladder,
    us_puma_ladder_assignment_summary,
    us_puma_ladder_gate,
    with_household_us_puma_ladder,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _ladder_metadata(**overrides: object) -> dict:
    layers = {
        "congressional_district": {
            "vintage": "119th_congress",
            "source": "Census 119th Congressional District BEF (NationalCD119.txt)",
        },
        "county": {
            "vintage": "2020_census",
            "source": "Census 2020 Census Tract to 2020 PUMA relationship file",
        },
        "tract": {
            "vintage": "2020_census",
            "source": "Census 2020 Census Tract to 2020 PUMA relationship file",
        },
    }
    metadata = {
        "schema_version": 1,
        "kind": "us_puma_ladder",
        "puma_vintage": "2020_puma",
        "sampling_basis": "population",
        "layers": layers,
    }
    metadata.update(overrides)
    return metadata


def _write_ladder(path: Path, **overrides: object) -> Path:
    # PUMA 0100100 spans CDs 101/102 (900/100) and counties 01001/01003;
    # PUMA 0100200 is CD 102 in county 01003; PUMA 0200100 is the state-02
    # at-large district. Every overlap table conserves its PUMA population.
    arrays: dict[str, np.ndarray] = {
        "puma": np.asarray([100100, 100200, 200100], dtype=np.int64),
        "puma_population": np.asarray([1000.0, 500.0, 400.0]),
        "cd_overlap_puma": np.asarray([100100, 100100, 100200, 200100], dtype=np.int64),
        "cd_overlap_cd": np.asarray([101, 102, 102, 200], dtype=np.int64),
        "cd_overlap_population": np.asarray([900.0, 100.0, 500.0, 400.0]),
        "county_overlap_puma": np.asarray(
            [100100, 100100, 100200, 200100], dtype=np.int64
        ),
        "county_overlap_county": np.asarray([1001, 1003, 1003, 2013], dtype=np.int64),
        "county_overlap_population": np.asarray([900.0, 100.0, 500.0, 400.0]),
        "tract_overlap_puma": np.asarray(
            [100100, 100100, 100200, 200100], dtype=np.int64
        ),
        "tract_overlap_tract": np.asarray(
            [1001000100, 1003000100, 1003000200, 2013000100], dtype=np.int64
        ),
        "tract_overlap_population": np.asarray([900.0, 100.0, 500.0, 400.0]),
        "metadata_json": np.asarray(json.dumps(_ladder_metadata())),
    }
    arrays.update({k: np.asarray(v) for k, v in overrides.items()})
    np.savez_compressed(path, **arrays)
    return path


def _ladder(tmp_path: Path):
    return load_us_puma_ladder(_write_ladder(tmp_path / "puma_ladder.npz"))


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #


def test_load_round_trips_arrays_and_vintages(tmp_path) -> None:
    ladder = _ladder(tmp_path)

    assert len(ladder) == 3
    assert ladder.layer_vintages == {
        "puma": "2020_puma",
        "congressional_district": "119th_congress",
        "county": "2020_census",
        "tract": "2020_census",
    }
    assert ladder.puma_population.dtype == np.float64
    assert ladder.county_overlap_county.dtype == np.int32


def test_load_refuses_missing_array_key(tmp_path) -> None:
    path = tmp_path / "puma_ladder.npz"
    _write_ladder(path)
    with np.load(path) as payload:
        arrays = {k: payload[k] for k in payload.files if k != "cd_overlap_cd"}
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="missing required key"):
        load_us_puma_ladder(path)


def test_load_refuses_missing_layer_vintage(tmp_path) -> None:
    metadata = _ladder_metadata()
    del metadata["layers"]["tract"]
    path = _write_ladder(
        tmp_path / "puma_ladder.npz", metadata_json=np.asarray(json.dumps(metadata))
    )

    with pytest.raises(ValueError, match="vintage_policy: error"):
        load_us_puma_ladder(path)


def test_load_refuses_nonpositive_population(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "puma_ladder.npz",
        puma_population=np.asarray([1000.0, 0.0, 400.0]),
    )

    with pytest.raises(ValueError, match="positive"):
        load_us_puma_ladder(path)


def test_load_refuses_duplicate_or_unsorted_pumas(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "puma_ladder.npz",
        puma=np.asarray([100200, 100100, 200100], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="sorted ascending"):
        load_us_puma_ladder(path)


def test_load_refuses_overlap_not_conserving_population(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "puma_ladder.npz",
        cd_overlap_population=np.asarray([800.0, 100.0, 500.0, 400.0]),
    )

    with pytest.raises(ValueError, match="must conserve exactly"):
        load_us_puma_ladder(path)


def test_load_refuses_overlap_puma_absent_from_anchor(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "puma_ladder.npz",
        cd_overlap_puma=np.asarray([100100, 100100, 100200, 999999], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="absent from the anchor"):
        load_us_puma_ladder(path)


def test_load_refuses_cd_state_mismatch(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "puma_ladder.npz",
        cd_overlap_cd=np.asarray([101, 102, 102, 600], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="state prefix must match"):
        load_us_puma_ladder(path)


# --------------------------------------------------------------------------- #
# Assignment — ACS spine (known PUMA) and ASEC spine (drawn PUMA)
# --------------------------------------------------------------------------- #


def test_acs_rows_keep_their_puma_and_derive_the_ladder(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame(
        {
            "household_id": [10, 20],
            "state_fips": [1, 2],
            "puma": [100100, 200100],
        },
        index=[100, 200],
    )

    assigned = assign_us_puma_ladder(household, ladder, seed=0)

    assert assigned.index.tolist() == [100, 200]
    for column in US_PUMA_LADDER_COLUMNS:
        assert column in assigned.columns
    # The state-02 at-large PUMA has exactly one CD and one county: fully
    # determined regardless of seed.
    assert assigned.loc[200, "puma"] == "0200100"
    assert assigned.loc[200, "congressional_district_geoid"] == 200
    assert assigned.loc[200, "county_fips"] == "02013"
    # The state-01 ACS household keeps PUMA 0100100 and draws within it.
    assert assigned.loc[100, "puma"] == "0100100"
    assert assigned.loc[100, "congressional_district_geoid"] in {101, 102}
    assert assigned.loc[100, "county_fips"] in {"01001", "01003"}


def test_asec_rows_draw_a_puma_within_state(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    # No puma column at all — the pure-ASEC spine.
    household = pd.DataFrame({"household_id": [1, 2], "state_fips": [1, 2]})

    assigned = assign_us_puma_ladder(household, ladder, seed=0)

    assert set(assigned.loc[assigned["state_fips"] == 1, "puma"]).issubset(
        {"0100100", "0100200"}
    )
    assert assigned.loc[assigned["state_fips"] == 2, "puma"].tolist() == ["0200100"]


def test_mixed_acs_and_asec_frame(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [1, 1, 2],
            # Row 0 is ACS (known PUMA); rows 1-2 are ASEC (NaN → drawn).
            "puma": [100200, np.nan, np.nan],
        }
    )

    assigned = assign_us_puma_ladder(household, ladder, seed=1)

    assert assigned.loc[0, "puma"] == "0100200"
    assert assigned.loc[0, "congressional_district_geoid"] == 102
    assert assigned.loc[2, "puma"] == "0200100"
    assert assigned.loc[1, "puma"] in {"0100100", "0100200"}


def test_assignment_is_deterministic_and_population_weighted(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame(
        {
            "household_id": range(600),
            "state_fips": [1] * 600,
            "puma": [100100] * 600,
        }
    )

    first = assign_us_puma_ladder(household, ladder, seed=7)
    second = assign_us_puma_ladder(household, ladder, seed=7)

    assert first["congressional_district_geoid"].tolist() == (
        second["congressional_district_geoid"].tolist()
    )
    # CD 101 carries 900/1000 of PUMA 0100100's population.
    share_101 = (first["congressional_district_geoid"] == 101).mean()
    assert 0.85 < share_101 < 0.95
    # County 01001 carries the same 900/1000.
    assert 0.85 < (first["county_fips"] == "01001").mean() < 0.95


def test_state_to_puma_draw_matches_population_weights(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"household_id": range(6000), "state_fips": [1] * 6000})

    assigned = assign_us_puma_ladder(household, ladder, seed=3)

    # PUMA 0100100 holds 1000/1500 of state 01's population.
    share = (assigned["puma"] == "0100100").mean()
    assert abs(share - (1000 / 1500)) < 0.03


def test_assign_tract_derives_a_consistent_county(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame(
        {"household_id": range(200), "state_fips": [1] * 200, "puma": [100100] * 200}
    )

    assigned = assign_us_puma_ladder(household, ladder, seed=0, assign_tract=True)

    assert "tract_geoid" in assigned.columns
    tracts = assigned["tract_geoid"]
    assert set(tracts).issubset({"01001000100", "01003000100"})
    # County derives structurally from the drawn tract — never an independent
    # draw that could disagree.
    assert (assigned["county_fips"] == tracts.str[:5]).all()


def test_congressional_district_is_stable_across_the_tract_flag(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame(
        {"household_id": range(200), "state_fips": [1] * 200, "puma": [100100] * 200}
    )

    without_tract = assign_us_puma_ladder(household, ladder, seed=5)
    with_tract = assign_us_puma_ladder(household, ladder, seed=5, assign_tract=True)

    assert without_tract["congressional_district_geoid"].tolist() == (
        with_tract["congressional_district_geoid"].tolist()
    )


def test_assignment_requires_state_fips(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"household_id": [1], "puma": [100100]})

    with pytest.raises(ValueError, match="must contain 'state_fips'"):
        assign_us_puma_ladder(household, ladder)


def test_assignment_refuses_puma_absent_from_ladder(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"household_id": [1], "state_fips": [1], "puma": [109999]})

    with pytest.raises(ValueError, match="share a PUMA vintage"):
        assign_us_puma_ladder(household, ladder)


def test_assignment_refuses_puma_state_disagreement(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"household_id": [1], "state_fips": [2], "puma": [100100]})

    with pytest.raises(ValueError, match="disagrees with state_fips"):
        assign_us_puma_ladder(household, ladder)


def test_assignment_refuses_state_without_pumas(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"household_id": [1], "state_fips": [5]})

    with pytest.raises(ValueError, match="no PUMAs for state_fips 05"):
        assign_us_puma_ladder(household, ladder)


def test_assignment_refuses_mismatched_cd_vintage(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"household_id": [1], "state_fips": [1], "puma": [100100]})

    with pytest.raises(ValueError, match="does not match the vintage"):
        assign_us_puma_ladder(
            household,
            ladder,
            expected_congressional_district_vintage="120th_congress",
        )


# --------------------------------------------------------------------------- #
# Frame integration, summary, and gate
# --------------------------------------------------------------------------- #


def test_with_household_preserves_frame_mass(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    frame = _minimal_us_frame()

    assigned = with_household_us_puma_ladder(
        frame,
        ladder,
        seed=0,
        expected_congressional_district_vintage="119th_congress",
    )

    household = assigned.table("household")
    assert household["puma"].tolist() == ["0100200", "0200100"]
    assert assigned.weights_for("household").values.tolist() == [100.0, 300.0]
    assert assigned.strata.tolist() == frame.strata.tolist()

    summary = us_puma_ladder_assignment_summary(
        household,
        ladder,
        weight_values=assigned.weights_for("household").values,
    )
    assert summary["applied"] is True
    assert summary["ladder_pumas"] == 3
    assert summary["assigned_puma_values"] == 2
    assert summary["layer_vintages"]["congressional_district"] == "119th_congress"


def test_gate_passes_on_a_consistent_ladder() -> None:
    household, weights = _gated_household()

    result = us_puma_ladder_gate(household, weights)

    assert result.passed, result.failures
    assert 0.005 <= result.details["nyc_weighted_household_share"] <= 0.06


def test_gate_fails_when_nyc_collapses_to_zero() -> None:
    household, weights = _gated_household()
    household.loc[household["county_fips"].isin(["36061"]), ["county_fips", "puma"]] = [
        "36001",
        "3600100",
    ]

    result = us_puma_ladder_gate(household, weights)

    assert not result.passed
    assert any("in_nyc collapse" in failure for failure in result.failures)


def test_gate_fails_on_state_prefix_inconsistency() -> None:
    household, weights = _gated_household()
    household.loc[household.index[0], "county_fips"] = "06037"

    result = us_puma_ladder_gate(household, weights)

    assert not result.passed
    assert any("disagree with the expected" in f for f in result.failures)


def test_gate_fails_when_columns_are_missing() -> None:
    household, weights = _gated_household()
    household = household.drop(columns=["county_fips"])

    result = us_puma_ladder_gate(household, weights)

    assert not result.passed
    assert any("missing geography column" in f for f in result.failures)


def _gated_household() -> tuple[pd.DataFrame, np.ndarray]:
    """A weighted household table whose NYC mass sits inside the gate bounds.

    NYC is 2.6% of national weight and 41.9% of New York State's.
    """

    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "state_fips": [36, 36, 6, 48],
            "puma": ["3603801", "3600100", "0603701", "4800100"],
            "congressional_district_geoid": [3612, 3620, 653, 4802],
            "county_fips": ["36061", "36001", "06037", "48001"],
        }
    )
    weights = np.asarray([2.6, 3.6, 73.8, 20.0])
    return household, weights


def _minimal_us_frame() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2], dtype="int64"),
                "person_household_id": np.asarray([1, 2], dtype="int64"),
                "person_tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "person_spm_unit_id": np.asarray([100, 200], dtype="int64"),
                "person_family_id": np.asarray([1000, 2000], dtype="int64"),
                "person_marital_unit_id": np.asarray([10000, 20000], dtype="int64"),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([1, 2], dtype="int64"),
                "puma": np.asarray([100200, 200100], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": np.asarray([10, 20])}),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    strata = pd.Series(["acs_2023", "acs_2023"], name="stratum")
    return Frame(tables, US_SCHEMA, weights, strata)
