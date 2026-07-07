"""UK output-area-anchored geography-ladder tests.

Coverage mirrors ``test_us_geography_ladder.py``: artifact round-trip and
vintage refusal, the assignment writing the full ladder from the OA anchor,
deterministic proportional sampling (two-stage: constituency by household
counts, OA by population), the region-coverage refusal (the Scotland/NI
NotImplemented path), constituency-vintage refusal, and the summary/gate
surface. It adds the pure assembler/join round-trip that has no US analogue
because the UK OA codes are not structural prefixes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    LADDER_OA_COLUMNS,
    UK_GEOGRAPHY_LADDER_COLUMNS,
    assemble_uk_oa_ladder,
    assign_uk_geography_ladder,
    join_uk_oa_ladder_layers,
    load_uk_oa_ladder,
    uk_geography_ladder_assignment_summary,
    uk_geography_ladder_gate,
)


def _ladder_metadata(**overrides: object) -> dict:
    source = "ONS Open Geography Portal / Nomis Census 2021"
    layers = {
        "constituency": {"vintage": "2024_pcon", "source": source},
        "lsoa": {"vintage": "2021_census", "source": source},
        "msoa": {"vintage": "2021_census", "source": source},
        "local_authority": {"vintage": "2023_april_lad", "source": source},
        "ward": {"vintage": "2022_wd", "source": source},
        "itl": {"vintage": "2021_itl", "source": source},
        "region": {"vintage": "2022_rgn", "source": source},
    }
    metadata = {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "oa_vintage": "2021_census",
        "constituency_sampling_basis": "census_2021_household_counts",
        "oa_sampling_basis": "census_2021_usual_resident_population",
        "layers": layers,
    }
    metadata.update(overrides)
    return metadata


def _ladder_arrays(**overrides: object) -> dict:
    # Two London constituencies whose household counts diverge sharply from
    # their populations (E14000001 is population-heavy but household-light,
    # E14000002 the reverse), and E14000001 in turn has one dominant-population
    # OA. That lets a test separate the stage-one (household) weight from the
    # stage-two (population) weight.
    arrays: dict[str, np.ndarray] = {
        "oa_code": np.asarray(
            ["E00000001", "E00000002", "E00000003", "W00000001"], dtype="U9"
        ),
        "population": np.asarray([990.0, 10.0, 300.0, 400.0]),
        "households": np.asarray([100.0, 100.0, 600.0, 150.0]),
        "constituency_code": np.asarray(
            ["E14000001", "E14000001", "E14000002", "W07000041"], dtype="U9"
        ),
        "region_code": np.asarray(
            ["E12000007", "E12000007", "E12000007", "W99999999"], dtype="U9"
        ),
        "lsoa_code": np.asarray(
            ["E01000001", "E01000001", "E01000002", "W01000001"], dtype="U9"
        ),
        "msoa_code": np.asarray(
            ["E02000001", "E02000001", "E02000002", "W02000001"], dtype="U9"
        ),
        "local_authority_code": np.asarray(
            ["E09000001", "E09000001", "E09000002", "W06000001"], dtype="U9"
        ),
        "ward_code": np.asarray(
            ["E05000001", "E05000001", "E05000002", "W05000001"], dtype="U9"
        ),
        "itl3_code": np.asarray(["TLI31", "TLI31", "TLI32", "TLL11"], dtype="U5"),
        "metadata_json": np.asarray(json.dumps(_ladder_metadata())),
    }
    arrays.update({k: np.asarray(v) for k, v in overrides.items()})
    return arrays


def _write_ladder(path: Path, **overrides: object) -> Path:
    np.savez_compressed(path, **_ladder_arrays(**overrides))
    return path


def _household() -> pd.DataFrame:
    # Region given four ways: enum member name, enum value, GSS code, and the
    # Wales member name — all must resolve.
    return pd.DataFrame(
        {
            "household_id": [10, 20, 30, 40],
            "region": ["LONDON", "London", "E12000007", "WALES"],
        },
        index=[100, 200, 300, 400],
    )


# ---------------------------------------------------------------------------
# Loader: round-trip and vintage refusal
# ---------------------------------------------------------------------------


def test_load_uk_oa_ladder_round_trips_arrays_and_vintages(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))

    assert len(ladder) == 4
    assert ladder.layer_vintages == {
        "oa": "2021_census",
        "constituency": "2024_pcon",
        "lsoa": "2021_census",
        "msoa": "2021_census",
        "local_authority": "2023_april_lad",
        "ward": "2022_wd",
        "itl": "2021_itl",
        "region": "2022_rgn",
    }
    assert ladder.population.dtype == np.float64
    assert ladder.households.dtype == np.float64


def test_load_refuses_missing_array_key(tmp_path) -> None:
    path = tmp_path / "ladder.npz"
    _write_ladder(path)
    with np.load(path) as payload:
        arrays = {k: payload[k] for k in payload.files if k != "ward_code"}
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="missing required key"):
        load_uk_oa_ladder(path)


def test_load_refuses_missing_layer_vintage(tmp_path) -> None:
    metadata = _ladder_metadata()
    del metadata["layers"]["itl"]
    path = _write_ladder(
        tmp_path / "ladder.npz",
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="vintage_policy: error"):
        load_uk_oa_ladder(path)


def test_load_refuses_missing_sampling_basis(tmp_path) -> None:
    metadata = _ladder_metadata()
    del metadata["constituency_sampling_basis"]
    path = _write_ladder(
        tmp_path / "ladder.npz",
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="constituency_sampling_basis"):
        load_uk_oa_ladder(path)


def test_load_refuses_invalid_gss_code(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "ladder.npz",
        oa_code=np.asarray(["E00000001", "NOTACODE", "E00000003", "W00000001"]),
    )

    with pytest.raises(ValueError, match="GSS codes"):
        load_uk_oa_ladder(path)


def test_load_refuses_nonpositive_population(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "ladder.npz",
        population=np.asarray([990.0, 0.0, 300.0, 400.0]),
    )

    with pytest.raises(ValueError, match="positive"):
        load_uk_oa_ladder(path)


def test_load_refuses_duplicate_output_areas(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "ladder.npz",
        oa_code=np.asarray(
            ["E00000001", "E00000001", "E00000003", "W00000001"], dtype="U9"
        ),
    )

    with pytest.raises(ValueError, match="unique"):
        load_uk_oa_ladder(path)


def test_load_refuses_region_with_zero_household_weight(tmp_path) -> None:
    # Every OA in London carries zero households: the stage-one constituency
    # draw would be undefined for the region.
    path = _write_ladder(
        tmp_path / "ladder.npz",
        households=np.asarray([0.0, 0.0, 0.0, 150.0]),
    )

    with pytest.raises(ValueError, match="zero household weight"):
        load_uk_oa_ladder(path)


def test_load_refuses_invalid_itl_code(tmp_path) -> None:
    path = _write_ladder(
        tmp_path / "ladder.npz",
        itl3_code=np.asarray(["TLI31", "TLI31", "XX99", "TLL11"]),
    )

    with pytest.raises(ValueError, match="ITL3"):
        load_uk_oa_ladder(path)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def test_assignment_writes_the_full_ladder_from_the_oa_anchor(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))

    assigned = assign_uk_geography_ladder(_household(), ladder, seed=0)

    assert assigned.index.tolist() == [100, 200, 300, 400]
    for column in UK_GEOGRAPHY_LADDER_COLUMNS:
        assert column in assigned.columns
    # Wales has exactly one OA, so its whole ladder row is checkable.
    wales = assigned.loc[400]
    assert wales["oa_code"] == "W00000001"
    assert wales["lsoa_code"] == "W01000001"
    assert wales["msoa_code"] == "W02000001"
    assert wales["local_authority_code"] == "W06000001"
    assert wales["ward_code"] == "W05000001"
    assert wales["constituency_code"] == "W07000041"
    assert wales["region_code"] == "W99999999"
    assert wales["itl3_code"] == "TLL11"
    assert wales["itl2_code"] == "TLL1"
    assert wales["itl1_code"] == "TLL"
    # London households draw a London OA only.
    assert set(assigned.loc[[100, 200, 300], "oa_code"]).issubset(
        {"E00000001", "E00000002", "E00000003"}
    )
    assert (assigned.loc[[100, 200, 300], "region_code"] == "E12000007").all()


def test_assignment_is_deterministic_and_two_stage_weighted(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))
    household = pd.DataFrame(
        {"household_id": range(6000), "region": ["LONDON"] * 6000}
    )

    first = assign_uk_geography_ladder(household, ladder, seed=7)
    second = assign_uk_geography_ladder(household, ladder, seed=7)

    assert first["oa_code"].tolist() == second["oa_code"].tolist()
    # Stage one is household-weighted: E14000001 (200 households) vs E14000002
    # (600 households) -> ~0.25, NOT the population-weighted 0.77.
    e1_share = (first["constituency_code"] == "E14000001").mean()
    assert 0.21 < e1_share < 0.29
    # Stage two is population-weighted within the constituency: E00000001 (pop
    # 990) dominates E00000002 (pop 10) inside E14000001, ~0.99, NOT the
    # household-weighted 0.5.
    within = first[first["constituency_code"] == "E14000001"]
    dominant = (within["oa_code"] == "E00000001").mean()
    assert dominant > 0.95


def test_assignment_requires_region(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))
    household = pd.DataFrame({"household_id": [1]})

    with pytest.raises(ValueError, match="assign regions first"):
        assign_uk_geography_ladder(household, ladder)


def test_assignment_refuses_region_missing_from_ladder(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))
    household = pd.DataFrame({"household_id": [1], "region": ["SCOTLAND"]})

    with pytest.raises(ValueError, match="no output areas for household region"):
        assign_uk_geography_ladder(household, ladder)


def test_assignment_refuses_mismatched_constituency_vintage(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))

    with pytest.raises(ValueError, match="does not match the vintage"):
        assign_uk_geography_ladder(
            _household(),
            ladder,
            expected_constituency_vintage="2010_pcon",
        )


def test_assignment_accepts_matching_constituency_vintage(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))

    assigned = assign_uk_geography_ladder(
        _household(),
        ladder,
        expected_constituency_vintage="2024_pcon",
    )
    assert "constituency_code" in assigned.columns


def test_assignment_summary_reports_per_rung_coverage(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))
    assigned = assign_uk_geography_ladder(_household(), ladder, seed=0)
    weights = np.asarray([10.0, 10.0, 10.0, 70.0])

    summary = uk_geography_ladder_assignment_summary(
        assigned, ladder, weight_values=weights
    )

    assert summary["applied"] is True
    assert summary["ladder_output_areas"] == 4
    assert summary["constituency_code_nonempty_weighted_share"] == 1.0
    assert summary["assigned_region_code_values"] == 2
    # London holds 30 of 100 weight units here.
    assert summary["london_weighted_household_share"] == pytest.approx(0.30)
    assert summary["layer_vintages"]["constituency"] == "2024_pcon"
    assert summary["constituency_sampling_basis"] == "census_2021_household_counts"


def test_assignment_summary_reports_not_applied_before_assignment(tmp_path) -> None:
    ladder = load_uk_oa_ladder(_write_ladder(tmp_path / "ladder.npz"))

    summary = uk_geography_ladder_assignment_summary(_household(), ladder)

    assert summary["applied"] is False
    assert "constituency_code_nonempty_weighted_share" not in summary


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _gated_household() -> tuple[pd.DataFrame, np.ndarray]:
    """A weighted household table whose shares sit inside the gate bounds.

    London carries 13% of the weight (matching its share of England & Wales
    households); ward, ITL, and constituency cover everyone.
    """

    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["LONDON", "SOUTH_EAST", "NORTH_WEST", "WALES"],
            "oa_code": ["E00000001", "E00000010", "E00000020", "W00000001"],
            "lsoa_code": ["E01000001", "E01000010", "E01000020", "W01000001"],
            "msoa_code": ["E02000001", "E02000010", "E02000020", "W02000001"],
            "local_authority_code": [
                "E09000001",
                "E07000010",
                "E08000020",
                "W06000001",
            ],
            "ward_code": ["E05000001", "E05000010", "E05000020", "W05000001"],
            "constituency_code": [
                "E14000001",
                "E14000010",
                "E14000020",
                "W07000041",
            ],
            "region_code": ["E12000007", "E12000008", "E12000002", "W99999999"],
            "itl3_code": ["TLI31", "TLJ31", "TLD31", "TLL11"],
            "itl2_code": ["TLI3", "TLJ3", "TLD3", "TLL1"],
            "itl1_code": ["TLI", "TLJ", "TLD", "TLL"],
        }
    )
    weights = np.asarray([13.0, 40.0, 37.0, 10.0])
    return household, weights


def test_gate_passes_on_a_consistent_ladder() -> None:
    household, weights = _gated_household()

    result = uk_geography_ladder_gate(household, weights)

    assert result.passed, result.failures
    assert 0.08 <= result.details["london_weighted_household_share"] <= 0.20
    assert result.details["constituency_nonempty_weighted_share"] == 1.0


def test_gate_fails_when_london_collapses_to_zero() -> None:
    household, weights = _gated_household()
    household.loc[
        household["region_code"] == "E12000007", "region_code"
    ] = "E12000002"

    result = uk_geography_ladder_gate(household, weights)

    assert not result.passed
    assert any("region-collapse" in failure for failure in result.failures)


def test_gate_fails_on_itl_prefix_inconsistency() -> None:
    household, weights = _gated_household()
    household.loc[household.index[0], "itl2_code"] = "TLZ9"

    result = uk_geography_ladder_gate(household, weights)

    assert not result.passed
    assert any("disagree with the ITL3 prefix" in f for f in result.failures)


def test_gate_fails_when_columns_are_missing() -> None:
    household, weights = _gated_household()
    household = household.drop(columns=["ward_code"])

    result = uk_geography_ladder_gate(household, weights)

    assert not result.passed
    assert any("missing geography column" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Assembler and join (pure)
# ---------------------------------------------------------------------------


def _base_crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": "E00000001",
                "lsoa_code": "E01000001",
                "msoa_code": "E02000001",
                "la_code": "E09000001",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 990.0,
            },
            {
                "oa_code": "E00000003",
                "lsoa_code": "E01000002",
                "msoa_code": "E02000002",
                "la_code": "E09000002",
                "constituency_code": "E14000002",
                "region_code": "E12000007",
                "country": "England",
                "population": 300.0,
            },
            {
                "oa_code": "W00000001",
                "lsoa_code": "W01000001",
                "msoa_code": "W02000001",
                "la_code": "W06000001",
                "constituency_code": "W07000041",
                "region_code": "W99999999",
                "country": "Wales",
                "population": 400.0,
            },
        ]
    )


def _oa_households() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oa_code": ["E00000001", "E00000003", "W00000001"],
            "households": [200, 300, 150],
        }
    )


def _oa_ward() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oa_code": ["E00000001", "E00000003", "W00000001"],
            "ward_code": ["E05000001", "E05000002", "W05000001"],
        }
    )


def _lad_itl() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "local_authority_code": ["E09000001", "E09000002", "W06000001"],
            "itl3_code": ["TLI31", "TLI32", "TLL11"],
        }
    )


def test_join_and_assemble_round_trip(tmp_path) -> None:
    joined = join_uk_oa_ladder_layers(
        _base_crosswalk(),
        oa_households=_oa_households(),
        oa_ward=_oa_ward(),
        lad_itl=_lad_itl(),
    )
    assert list(joined.columns) == list(LADDER_OA_COLUMNS)

    payload = assemble_uk_oa_ladder(joined, _ladder_metadata())
    path = tmp_path / "assembled.npz"
    np.savez_compressed(path, **payload)
    ladder = load_uk_oa_ladder(path)

    assert len(ladder) == 3
    assert set(ladder.oa_code.tolist()) == {"E00000001", "E00000003", "W00000001"}
    # The E09000001 OA inherits its LAD's ITL3.
    index = ladder.oa_code.tolist().index("E00000001")
    assert ladder.itl3_code[index] == "TLI31"
    assert ladder.local_authority_code[index] == "E09000001"


def test_assemble_drops_unpopulated_output_areas() -> None:
    joined = join_uk_oa_ladder_layers(
        _base_crosswalk(),
        oa_households=_oa_households(),
        oa_ward=_oa_ward(),
        lad_itl=_lad_itl(),
    )
    joined.loc[joined["oa_code"] == "E00000003", "population"] = 0.0

    payload = assemble_uk_oa_ladder(joined, _ladder_metadata())

    assert "E00000003" not in payload["oa_code"].tolist()
    assert len(payload["oa_code"]) == 2


def test_join_raises_on_unmatched_household_count() -> None:
    households = _oa_households()
    households = households[households["oa_code"] != "W00000001"]

    with pytest.raises(ValueError, match="OA household count"):
        join_uk_oa_ladder_layers(
            _base_crosswalk(),
            oa_households=households,
            oa_ward=_oa_ward(),
            lad_itl=_lad_itl(),
        )


def test_join_raises_on_unmatched_lad_itl() -> None:
    lad_itl = _lad_itl()
    lad_itl = lad_itl[lad_itl["local_authority_code"] != "W06000001"]

    with pytest.raises(ValueError, match="LAD ITL"):
        join_uk_oa_ladder_layers(
            _base_crosswalk(),
            oa_households=_oa_households(),
            oa_ward=_oa_ward(),
            lad_itl=lad_itl,
        )
