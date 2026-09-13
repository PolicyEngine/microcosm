"""US PUMA-anchored geography-ladder tests."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    US_PUMA_LADDER_COLUMNS,
    assign_us_puma_ladder,
    load_us_puma_ladder,
    us_puma_ladder_assignment_summary,
    us_puma_ladder_gate,
    us_puma_ladder_joint_support_gate,
    with_household_us_puma_ladder,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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
        "schema_version": 2,
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
        "joint_overlap_puma": np.asarray(
            [100100, 100100, 100200, 200100], dtype=np.int64
        ),
        "joint_overlap_tract": np.asarray(
            [1001000100, 1003000100, 1003000200, 2013000100], dtype=np.int64
        ),
        "joint_overlap_cd": np.asarray([101, 102, 102, 200], dtype=np.int64),
        "joint_overlap_population": np.asarray([900, 100, 500, 400], dtype=np.int64),
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


def test_county_cd_pairs_have_actual_joint_support(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"state_fips": [1] * 1000, "puma": [100100] * 1000})
    assigned = assign_us_puma_ladder(household, ladder, seed=578)
    pairs = set(
        zip(
            assigned["county_fips"],
            assigned["congressional_district_geoid"],
            strict=True,
        )
    )
    assert pairs == {("01001", 101), ("01003", 102)}
    assert us_puma_ladder_joint_support_gate(assigned, ladder).passed
    # Both marginals and the state prefix are valid; their pairing is not.
    assigned.loc[0, ["county_fips", "congressional_district_geoid"]] = ["01001", 102]
    result = us_puma_ladder_joint_support_gate(assigned, ladder)
    assert not result.passed
    assert result.details["unsupported_rows"] == 1
    assert any(
        "no joint PUMA/county/CD support" in reason for reason in result.failures
    )


def test_tract_flag_preserves_the_complete_coarse_assignment(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame(
        {"state_fips": [1, 1, 2] * 20, "puma": [100100, np.nan, np.nan] * 20}
    )
    before = household.copy(deep=True)
    coarse = assign_us_puma_ladder(household, ladder, seed=20260909)
    fine = assign_us_puma_ladder(household, ladder, seed=20260909, assign_tract=True)
    pd.testing.assert_frame_equal(coarse, fine.drop(columns=["tract_geoid"]))
    pd.testing.assert_frame_equal(household, before)
    assert coarse.loc[household["puma"].notna(), "puma"].eq("0100100").all()
    assert us_puma_ladder_joint_support_gate(fine, ladder, assign_tract=True).passed
    fine.loc[0, "tract_geoid"] = "01003000200"
    assert not us_puma_ladder_joint_support_gate(fine, ladder, assign_tract=True).passed


def test_old_marginal_only_artifact_requires_rebuild(tmp_path) -> None:
    path = _write_ladder(tmp_path / "v1.npz")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: payload[name]
            for name in payload.files
            if not name.startswith("joint_")
        }
    arrays["metadata_json"] = np.asarray(json.dumps(_ladder_metadata(schema_version=1)))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="missing required key"):
        load_us_puma_ladder(path)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (
            {"joint_overlap_population": np.asarray([899, 100, 500, 400])},
            "exactly reproduce the anchor",
        ),
        (
            {"joint_overlap_cd": np.asarray([102, 101, 102, 200])},
            "exactly reproduce the congressional_district",
        ),
        (
            {
                "joint_overlap_tract": np.asarray(
                    [1003000100, 1003000100, 1003000200, 2013000100]
                )
            },
            "exactly reproduce the county",
        ),
        (
            {"joint_overlap_population": np.asarray([900.0, 100.0, 500.0, 400.0])},
            "integer array",
        ),
        (
            {"joint_overlap_population": np.asarray([900, 0, 500, 400])},
            "positive integer",
        ),
        (
            {"joint_overlap_population": np.asarray([[900, 100, 500, 400]])},
            "one-dimensional",
        ),
        (
            {"joint_overlap_cd": np.asarray([201, 102, 102, 200])},
            "state prefix must match",
        ),
        (
            {
                "joint_overlap_population": np.asarray(
                    [900, 100, 500, 2**63], dtype=np.uint64
                )
            },
            "outside signed int64",
        ),
        (
            {"joint_overlap_puma": np.asarray([100100, 100200, 100100, 200100])},
            "unique and sorted",
        ),
    ],
)
def test_load_rejects_invalid_joint_support(tmp_path, overrides, reason) -> None:
    path = _write_ladder(tmp_path / "invalid-joint.npz", **overrides)
    with pytest.raises(ValueError, match=reason):
        load_us_puma_ladder(path)


def test_mutated_ladder_arrays_refuse_before_assignment(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    ladder.joint_overlap_population[0] -= 1
    household = pd.DataFrame({"state_fips": [1], "puma": [100100]})
    with pytest.raises(ValueError, match="exactly reproduce the anchor"):
        assign_us_puma_ladder(household, ladder)
    assert not us_puma_ladder_joint_support_gate(household, ladder).passed


def test_joint_support_gate_rejects_vintage_and_nonintegral_geoids(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"state_fips": [1], "puma": [100100]})
    assigned = assign_us_puma_ladder(household, ladder)
    assert not us_puma_ladder_joint_support_gate(
        assigned, ladder, expected_congressional_district_vintage="118th_congress"
    ).passed
    assigned["congressional_district_geoid"] = [101.5]
    assert not us_puma_ladder_joint_support_gate(assigned, ladder).passed


def test_direct_ladder_duplicate_anchor_is_rejected(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    # Aggregate counts still match, but duplicate anchors would bias the ASEC draw.
    ladder = replace(
        ladder,
        puma=np.asarray([100100, 100100, 100200, 200100], dtype=np.int64),
        puma_population=np.asarray([500.0, 500.0, 500.0, 400.0]),
    )
    with pytest.raises(ValueError, match="anchors must be unique"):
        assign_us_puma_ladder(pd.DataFrame({"state_fips": [1]}), ladder)


def test_live_joint_support_requires_tracts_to_nest_in_one_puma(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    household = pd.DataFrame({"state_fips": [1], "puma": [100100]})
    assigned = assign_us_puma_ladder(household, ladder)
    # Counts and every marginal remain exact, but this corrupts tract nesting.
    ladder.joint_overlap_tract[2] = ladder.joint_overlap_tract[1]
    ladder.tract_overlap_tract[2] = ladder.tract_overlap_tract[1]
    with pytest.raises(ValueError, match="one tract to multiple PUMAs"):
        assign_us_puma_ladder(household, ladder)
    result = us_puma_ladder_joint_support_gate(assigned, ladder)
    assert not result.passed
    assert any("one tract to multiple PUMAs" in reason for reason in result.failures)


@pytest.mark.parametrize("layer", ["county", "tract"])
def test_joint_assignment_rejects_wrong_census_vintage(tmp_path, layer) -> None:
    metadata = _ladder_metadata()
    metadata["layers"][layer]["vintage"] = "2010_census"
    path = _write_ladder(
        tmp_path / "puma_ladder.npz", metadata_json=np.asarray(json.dumps(metadata))
    )
    with pytest.raises(ValueError, match="2020 Census county/tract"):
        load_us_puma_ladder(path)
