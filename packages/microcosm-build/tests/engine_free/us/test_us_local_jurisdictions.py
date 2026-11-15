"""US local-jurisdiction flags: registry pin, derivation properties and gates.

Invariants exercised here (stated for every input, then checked by
Hypothesis):

- a flag is true exactly when the household's county (county rule) or its
  state and place (place rule) equal the registry geography;
- over a block ladder, the population of the blocks whose household gets a
  flag equals the population of the jurisdiction's blocks;
- derivation is deterministic and commutes with any reordering of household
  or person rows, and each person carries its household's value;
- a line without a place rung never writes a place-rule flag, and missing or
  malformed geography raises instead of yielding ``False``;
- deriving a selection of a derived support equals deriving the selected rows
  from scratch, and leaves every flag the kept rows' geography decides;
- deriving a support whose flags are already complete returns it unchanged;
- a selection held to the rungs of the support it came from carries exactly
  that support's flags, and each flag its own geography adds is recorded as
  unavailable rather than written;
- after any sequence of geography lines and artifact steps, no place-rule or
  county-rule flag is true on a household whose county lies outside the
  flag's counties, and no line leaves a block outside its household's county;
- a table in which a row's block lies outside the row's county supplies no
  rung: reading its rungs raises, so no place-rule flag is derived from it.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.build.us_local_jurisdictions import (
    BLOCK_GEOGRAPHY_RUNGS,
    JURISDICTION_FLAG_DERIVED,
    JURISDICTION_FLAG_UNAVAILABLE,
    PUMA_GEOGRAPHY_RUNGS,
    US_LOCAL_JURISDICTION_REGISTRY_RESOURCE,
    US_LOCAL_JURISDICTION_REGISTRY_SHA256,
    apply_us_local_jurisdiction_flags,
    check_us_local_jurisdictions_against_block_ladder,
    load_us_local_jurisdiction_registry,
    packaged_us_local_jurisdiction_registry,
    rederive_us_local_jurisdiction_flags,
    us_local_jurisdiction_artifact_receipt,
    us_local_jurisdiction_flag_failures,
    us_local_jurisdiction_flag_values,
    us_local_jurisdiction_rungs_present,
    with_us_local_jurisdiction_flags,
)
from microcosm.build.us_runtime.base_pool import (
    ACS_2024_1YR_SPINE,
    ASEC_PUF_SPINE,
    spine_column,
    with_optional_acs_spine,
)
from microcosm.build.us_runtime.block_location import (
    US_CD_PLAN_COLUMN_PREFIX,
    attach_congressional_district_plans,
    load_us_location_ladder,
    location_geography_columns,
    us_block_location_gate,
    with_household_us_block_location,
)
from microcosm.build.us_runtime.geography_ladder import (
    US_GEOGRAPHY_LADDER_COLUMNS,
    assign_us_geography_ladder,
    us_geography_ladder_assignment_summary,
    us_geography_ladder_gate,
    with_household_us_geography_ladder,
)
from microcosm.build.us_runtime.puma_ladder import (
    US_PUMA_LADDER_BLOCK_ANCHORED_COLUMNS,
    US_PUMA_LADDER_BLOCK_PLAN_COLUMN_PREFIX,
    US_PUMA_LADDER_COLUMNS,
    US_PUMA_LADDER_TRACT_COLUMN,
    UsPumaLadder,
    assign_us_puma_ladder,
    us_puma_ladder_assignment_summary,
    us_puma_ladder_gate,
    with_household_us_puma_ladder,
)
from microcosm.frame import US_SCHEMA, Frame
from test_support.microcosm_build.us_block_location import (
    local_jurisdiction_blocks,
    write_location_ladder,
)
from test_support.microcosm_build.us_local_jurisdictions import (
    FIXTURE_BLOCK_BY_NAME,
    FIXTURE_BLOCKS,
    FixtureBlock,
    block_ladder,
    households_on_blocks,
    us_frame,
)
from test_support.paths import paths_for

REGISTRY_PATH = (
    paths_for("microcosm-build").package
    / "src/microcosm/build"
    / US_LOCAL_JURISDICTION_REGISTRY_RESOURCE
)
REGISTRY = packaged_us_local_jurisdiction_registry()
HOUSEHOLD_FLAGS = ("in_denver", "in_wilmington", "in_yonkers")
PERSON_FLAGS = ("pa_philadelphia_wage_tax_resident",)
PLACE_FLAGS = ("in_wilmington", "in_yonkers")

# The registry geographies, written out independently of the registry file:
# a code mistyped there fails these oracles.
EXPECTED_RULES = {
    "in_denver": ("county", "08031"),
    "pa_philadelphia_wage_tax_resident": ("county", "42101"),
    "in_yonkers": ("place", "36", "84000"),
    "in_wilmington": ("place", "10", "77580"),
}


def _oracle(flag: str, county: str, place: str) -> bool:
    rule = EXPECTED_RULES[flag]
    if rule[0] == "county":
        return county == rule[1]
    return county[:2] == rule[1] and place == rule[2]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_packaged_registry_bytes_match_the_pinned_sha256() -> None:
    payload = REGISTRY_PATH.read_bytes()

    assert hashlib.sha256(payload).hexdigest() == US_LOCAL_JURISDICTION_REGISTRY_SHA256
    assert REGISTRY.sha256 == US_LOCAL_JURISDICTION_REGISTRY_SHA256


def test_registry_maps_each_input_only_flag_to_its_2020_geography() -> None:
    by_flag = {item.flag: item for item in REGISTRY.jurisdictions}

    assert set(by_flag) == set(EXPECTED_RULES)
    assert {flag: by_flag[flag].entity for flag in by_flag} == {
        "in_denver": "household",
        "in_wilmington": "household",
        "in_yonkers": "household",
        "pa_philadelphia_wage_tax_resident": "person",
    }
    for flag, rule in EXPECTED_RULES.items():
        item = by_flag[flag]
        assert item.rule == rule[0]
        if rule[0] == "county":
            assert item.county_fips == rule[1]
        else:
            assert (item.state_fips, item.place_fips) == rule[1:]
    assert by_flag["in_denver"].coextensive_place_fips == "20000"
    assert by_flag["pa_philadelphia_wage_tax_resident"].coextensive_place_fips == (
        "60000"
    )


def test_registry_records_block_ladder_checks_that_match_its_populations() -> None:
    document = json.loads(REGISTRY_PATH.read_text())
    populations = {item.flag: item.population_2020 for item in REGISTRY.jurisdictions}
    national = document["sources"]["population"]["national_total_2020"]

    checks = document["block_ladder_checks"]
    assert {check["block_ladder_sha256"][:8] for check in checks} == {
        "6840b990",
        "7ba39b95",
    }
    for check in checks:
        assert check["population"] == national
        assert {
            flag: entry["population"] for flag, entry in check["jurisdictions"].items()
        } == populations
        for flag in ("in_denver", "pa_philadelphia_wage_tax_resident"):
            entry = check["jurisdictions"][flag]
            assert entry["coextensive_place_blocks"] == entry["blocks"]


def test_packaged_registry_refuses_bytes_that_differ_from_the_pin(tmp_path) -> None:
    edited = tmp_path / "registry.json"
    edited.write_bytes(REGISTRY_PATH.read_bytes() + b" ")

    with pytest.raises(ValueError, match="do not match the pinned SHA-256"):
        load_us_local_jurisdiction_registry(
            edited, expected_sha256=US_LOCAL_JURISDICTION_REGISTRY_SHA256
        )


def _write_registry(path: Path, jurisdictions: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "us_local_jurisdiction_registry",
                "schema_version": 1,
                "jurisdictions": jurisdictions,
            }
        )
    )
    return path


def _county_entry(**overrides: object) -> dict:
    entry: dict[str, object] = {
        "flag": "in_denver",
        "entity": "household",
        "jurisdiction": "Denver",
        "rule": "county",
        "state_fips": "08",
        "county_fips": "08031",
        "place_fips": None,
        "coextensive_place_fips": "20000",
        "population_2020": 700,
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"county_fips": "8031"}, "5-digit string"),
        ({"county_fips": 8031}, "5-digit string"),
        ({"state_fips": "09"}, "is not in state"),
        ({"rule": "tract"}, "rule must be one of"),
        ({"entity": "tax_unit"}, "entity must be one of"),
        ({"place_fips": "20000"}, "county rule names county_fips and no place_fips"),
        ({"population_2020": 0}, "must be positive"),
        ({"population_2020": True}, "must be an integer"),
        ({"flag": "In_Denver"}, "lowercase identifier"),
    ],
)
def test_registry_loader_refuses_malformed_entries(
    tmp_path, overrides: dict, message: str
) -> None:
    path = _write_registry(tmp_path / "registry.json", [_county_entry(**overrides)])

    with pytest.raises(ValueError, match=message):
        load_us_local_jurisdiction_registry(path)


def test_registry_loader_refuses_repeated_flags_and_geographies(tmp_path) -> None:
    repeated_flag = _write_registry(
        tmp_path / "flag.json", [_county_entry(), _county_entry(county_fips="08005")]
    )
    repeated_geography = _write_registry(
        tmp_path / "geography.json",
        [_county_entry(), _county_entry(flag="in_denver_too")],
    )

    with pytest.raises(ValueError, match="repeats a flag"):
        load_us_local_jurisdiction_registry(repeated_flag)
    with pytest.raises(ValueError, match="repeats a geography"):
        load_us_local_jurisdiction_registry(repeated_geography)


# --------------------------------------------------------------------------- #
# Derivation properties
# --------------------------------------------------------------------------- #

# Registry codes, their neighbours and near misses, plus arbitrary codes.
_COUNTIES = st.one_of(
    st.sampled_from(
        [
            "08031",
            "42101",
            "36119",
            "10003",
            "08005",
            "08032",
            "18031",
            "42103",
            "42045",
            "06037",
            "37129",
            "36005",
        ]
    ),
    st.from_regex(r"\A[0-9]{5}\Z"),
)
_PLACES = st.one_of(
    st.sampled_from(
        ["", "84000", "77580", "20000", "60000", "84001", "77581", "04000"]
    ),
    st.from_regex(r"\A[0-9]{5}\Z"),
)
_GEOGRAPHY_ROWS = st.lists(st.tuples(_COUNTIES, _PLACES), min_size=1, max_size=40)


def _household_table(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": np.arange(1, len(rows) + 1),
            "county_fips": [county for county, _ in rows],
            "place_fips": [place for _, place in rows],
        }
    )


@settings(max_examples=200, deadline=None)
@given(_GEOGRAPHY_ROWS)
def test_flag_is_true_iff_county_or_state_and_place_match(rows) -> None:
    household = _household_table(rows)

    values = us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)

    assert set(values) == set(EXPECTED_RULES)
    for flag, observed in values.items():
        assert observed.dtype == bool
        expected = [_oracle(flag, county, place) for county, place in rows]
        assert observed.tolist() == expected


@settings(max_examples=100, deadline=None)
@given(_GEOGRAPHY_ROWS, st.randoms(use_true_random=False))
def test_derivation_commutes_with_household_row_order(rows, random) -> None:
    household = _household_table(rows)
    order = list(range(len(rows)))
    random.shuffle(order)
    shuffled = household.iloc[order].reset_index(drop=True)

    original = us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)
    reordered = us_local_jurisdiction_flag_values(shuffled, rungs=BLOCK_GEOGRAPHY_RUNGS)
    repeated = us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)

    for flag in original:
        np.testing.assert_array_equal(reordered[flag], original[flag][order])
        np.testing.assert_array_equal(repeated[flag], original[flag])


@settings(max_examples=100, deadline=None)
@given(
    _GEOGRAPHY_ROWS.flatmap(
        lambda rows: st.tuples(
            st.just(rows),
            st.lists(
                st.integers(min_value=1, max_value=len(rows)), min_size=1, max_size=60
            ),
        )
    ),
    st.randoms(use_true_random=False),
)
def test_person_flags_carry_their_household_value_in_any_person_order(
    case, random
) -> None:
    rows, memberships = case
    random.shuffle(memberships)
    household = _household_table(rows)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(memberships) + 1),
            "person_household_id": memberships,
        }
    )

    tables, _ = apply_us_local_jurisdiction_flags(
        {"household": household, "person": person}, rungs=BLOCK_GEOGRAPHY_RUNGS
    )

    county = dict(zip(household["household_id"], household["county_fips"], strict=True))
    place = dict(zip(household["household_id"], household["place_fips"], strict=True))
    for flag in PERSON_FLAGS:
        expected = [
            _oracle(flag, county[member], place[member]) for member in memberships
        ]
        assert tables["person"][flag].tolist() == expected
        assert flag not in tables["household"]
    for flag in HOUSEHOLD_FLAGS:
        assert flag not in tables["person"]
    assert "pa_philadelphia_wage_tax_resident" not in person  # input untouched


@settings(max_examples=100, deadline=None)
@given(_GEOGRAPHY_ROWS)
def test_a_line_without_a_place_rung_never_writes_a_place_flag(rows) -> None:
    household = _household_table(rows).drop(columns=["place_fips"])
    household["in_yonkers"] = False  # a stale column from an earlier line
    household["in_wilmington"] = np.nan
    person = pd.DataFrame(
        {
            "person_id": household["household_id"],
            "person_household_id": household["household_id"],
        }
    )

    tables, receipt = apply_us_local_jurisdiction_flags(
        {"household": household, "person": person}, rungs=PUMA_GEOGRAPHY_RUNGS
    )

    for flag in PLACE_FLAGS:
        assert flag not in tables["household"]
        assert receipt["flags"][flag]["status"] == JURISDICTION_FLAG_UNAVAILABLE
        assert "no place rung" in receipt["flags"][flag]["reason"]
    assert tables["household"]["in_denver"].tolist() == [
        county == "08031" for county, _ in rows
    ]
    assert receipt["flags"]["in_denver"]["status"] == JURISDICTION_FLAG_DERIVED
    assert receipt["rungs"] == ["county"]
    assert receipt["registry_sha256"] == US_LOCAL_JURISDICTION_REGISTRY_SHA256


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("county_fips", None, "county_fips must hold 5-digit"),
        ("county_fips", np.nan, "county_fips must hold 5-digit"),
        ("county_fips", 8031, "county_fips must hold 5-digit"),
        ("county_fips", "8031", "county_fips must hold 5-digit"),
        ("place_fips", None, "place_fips must hold empty or 5-digit"),
        ("place_fips", "8400", "place_fips must hold empty or 5-digit"),
        ("place_fips", 84000, "place_fips must hold empty or 5-digit"),
    ],
)
def test_unknown_or_malformed_geography_raises_instead_of_writing_false(
    column: str, value: object, message: str
) -> None:
    household = _household_table([("36119", "84000"), ("08031", "")])
    household[column] = household[column].astype(object)
    household.loc[1, column] = value

    with pytest.raises(ValueError, match=message):
        us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)


def test_missing_rung_column_raises() -> None:
    household = _household_table([("36119", "84000")]).drop(columns=["place_fips"])

    with pytest.raises(ValueError, match="no 'place_fips'"):
        us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)


def test_bytes_codes_from_an_h5_round_trip_are_decoded() -> None:
    household = pd.DataFrame(
        {"county_fips": [b"36119", b"10003"], "place_fips": [b"84000", b""]}
    )

    values = us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)

    assert values["in_yonkers"].tolist() == [True, False]
    assert values["in_wilmington"].tolist() == [False, False]


# --------------------------------------------------------------------------- #
# Population conservation over a block ladder
# --------------------------------------------------------------------------- #

_LADDER_BLOCK = st.tuples(
    st.sampled_from(
        ["08031", "08005", "36119", "36005", "10003", "42101", "42045", "06037"]
    ),
    st.sampled_from([0, 84000, 77580, 20000, 60000, 4000]),
    st.integers(min_value=1, max_value=5_000),
)


@settings(max_examples=60, deadline=None)
@given(st.lists(_LADDER_BLOCK, min_size=1, max_size=30))
def test_flag_weighted_block_population_equals_the_jurisdictions_blocks(rows) -> None:
    blocks = [
        FixtureBlock(
            name=f"block_{index}",
            block_geoid=int(f"{county}{index:06d}1000"),
            place_fips=place,
            congressional_district_geoid=int(county[:2]) * 100 + index % 100,
            population=float(population),
        )
        for index, (county, place, population) in enumerate(rows)
    ]
    # One household per block: the ladder's district-then-block draw is
    # deterministic when every block owns its district.
    blocks = list(
        {block.congressional_district_geoid: block for block in blocks}.values()
    )
    ladder = block_ladder(blocks)
    household = pd.DataFrame(households_on_blocks_for(blocks))

    assigned = assign_us_geography_ladder(household, ladder, seed=3)
    values = us_local_jurisdiction_flag_values(assigned, rungs=BLOCK_GEOGRAPHY_RUNGS)
    population = np.asarray([block.population for block in blocks])
    assert assigned["block_geoid"].tolist() == [
        f"{block.block_geoid:015d}" for block in blocks
    ]

    for flag in EXPECTED_RULES:
        expected = sum(
            block.population
            for block in blocks
            if _oracle(
                flag,
                block.county_fips,
                "" if block.place_fips == 0 else f"{block.place_fips:05d}",
            )
        )
        assert float(population[values[flag]].sum()) == expected


def households_on_blocks_for(blocks: list[FixtureBlock]) -> list[dict[str, int]]:
    return [
        {
            "household_id": index + 1,
            "state_fips": block.state_fips,
            "congressional_district_geoid": block.congressional_district_geoid,
        }
        for index, block in enumerate(blocks)
    ]


def _fixture_registry(tmp_path: Path, **population: int) -> object:
    jurisdictions = []
    for item in REGISTRY.jurisdictions:
        entry = {
            "flag": item.flag,
            "entity": item.entity,
            "jurisdiction": item.jurisdiction,
            "rule": item.rule,
            "state_fips": item.state_fips,
            "county_fips": item.county_fips,
            "place_fips": item.place_fips,
            "coextensive_place_fips": item.coextensive_place_fips,
            "population_2020": population[item.flag],
        }
        jurisdictions.append(entry)
    return load_us_local_jurisdiction_registry(
        _write_registry(tmp_path / "registry.json", jurisdictions)
    )


def test_block_ladder_check_passes_on_exact_populations_and_coextensive_blocks(
    tmp_path,
) -> None:
    ladder = block_ladder()
    registry = _fixture_registry(
        tmp_path,
        in_denver=700,
        in_wilmington=70,
        in_yonkers=210,
        pa_philadelphia_wage_tax_resident=1600,
    )

    report = check_us_local_jurisdictions_against_block_ladder(
        ladder.block_geoid, ladder.population, ladder.place_fips, registry=registry
    )

    assert report == {
        "in_denver": {"blocks": 1, "population": 700, "coextensive_place_blocks": 1},
        "in_wilmington": {"blocks": 1, "population": 70},
        "in_yonkers": {"blocks": 1, "population": 210},
        "pa_philadelphia_wage_tax_resident": {
            "blocks": 1,
            "population": 1600,
            "coextensive_place_blocks": 1,
        },
    }


def test_block_ladder_check_refuses_a_population_off_by_one(tmp_path) -> None:
    ladder = block_ladder()
    registry = _fixture_registry(
        tmp_path,
        in_denver=700,
        in_wilmington=70,
        in_yonkers=211,
        pa_philadelphia_wage_tax_resident=1600,
    )

    with pytest.raises(ValueError, match="in_yonkers: ladder population 210"):
        check_us_local_jurisdictions_against_block_ladder(
            ladder.block_geoid, ladder.population, ladder.place_fips, registry=registry
        )


def test_block_ladder_check_refuses_a_county_that_is_not_its_city(tmp_path) -> None:
    denver = FIXTURE_BLOCK_BY_NAME["denver"]
    outside_city = FixtureBlock("denver_outside_city", 80310002001000, 0, 802, 5.0)
    ladder = block_ladder([*FIXTURE_BLOCKS, outside_city])
    registry = _fixture_registry(
        tmp_path,
        in_denver=int(denver.population + outside_city.population),
        in_wilmington=70,
        in_yonkers=210,
        pa_philadelphia_wage_tax_resident=1600,
    )

    with pytest.raises(ValueError, match="are not the same blocks"):
        check_us_local_jurisdictions_against_block_ladder(
            ladder.block_geoid, ladder.population, ladder.place_fips, registry=registry
        )


# --------------------------------------------------------------------------- #
# Lines: block ladder, PUMA ladder, helper for the block-location draw
# --------------------------------------------------------------------------- #

_NAMES = [block.name for block in FIXTURE_BLOCKS]


def _expected_block_flags(name: str) -> dict[str, bool]:
    block = FIXTURE_BLOCK_BY_NAME[name]
    place = "" if block.place_fips == 0 else f"{block.place_fips:05d}"
    return {flag: _oracle(flag, block.county_fips, place) for flag in EXPECTED_RULES}


def test_block_line_writes_every_flag_as_an_engine_input() -> None:
    people = [[{"age": 40}, {"age": 9}] for _ in _NAMES]
    frame = us_frame(households_on_blocks(_NAMES), person_inputs=people)

    assigned = with_household_us_geography_ladder(frame, block_ladder(), seed=11)

    household = assigned.table("household")
    person = assigned.table("person")
    for position, name in enumerate(_NAMES):
        expected = _expected_block_flags(name)
        for flag in HOUSEHOLD_FLAGS:
            assert bool(household[flag].iloc[position]) is expected[flag], (name, flag)
        members = person["person_household_id"].eq(
            household["household_id"].iloc[position]
        )
        assert (
            person.loc[members, PERSON_FLAGS[0]].tolist()
            == [expected[PERSON_FLAGS[0]]] * 2
        )
    assert sorted(
        name for name in _NAMES if _expected_block_flags(name)["in_yonkers"]
    ) == ["yonkers"]
    assert sorted(
        name for name in _NAMES if _expected_block_flags(name)["in_wilmington"]
    ) == ["wilmington"]
    assert household[list(HOUSEHOLD_FLAGS)].dtypes.eq(bool).all()
    assert us_local_jurisdiction_flag_failures(household) == []


def test_block_line_is_deterministic_and_receipts_every_flag_as_derived() -> None:
    frame = us_frame(households_on_blocks(_NAMES), weights=np.arange(1, 11))

    first = with_household_us_geography_ladder(frame, block_ladder(), seed=5)
    second = with_household_us_geography_ladder(frame, block_ladder(), seed=5)
    summary = us_geography_ladder_assignment_summary(
        first.table("household"),
        block_ladder(),
        weight_values=first.weights_for("household").values,
    )

    for entity in ("household", "person"):
        pd.testing.assert_frame_equal(first.table(entity), second.table(entity))
    receipt = summary["local_jurisdiction_flags"]
    assert receipt["registry_sha256"] == US_LOCAL_JURISDICTION_REGISTRY_SHA256
    assert receipt["rungs"] == ["county", "place"]
    assert {flag: entry["status"] for flag, entry in receipt["flags"].items()} == {
        flag: JURISDICTION_FLAG_DERIVED for flag in EXPECTED_RULES
    }
    # Household weights are 1..10 in fixture order: yonkers is the third.
    assert receipt["flags"]["in_yonkers"]["weighted_households_in_jurisdiction"] == 3.0
    assert receipt["flags"]["in_denver"]["households_in_jurisdiction"] == 1


def test_frame_helper_matches_the_block_line_for_the_block_location_draw() -> None:
    frame = us_frame(households_on_blocks(_NAMES))
    assigned = with_household_us_geography_ladder(frame, block_ladder(), seed=2)
    stripped = assigned.table("household").drop(columns=list(HOUSEHOLD_FLAGS))
    tables = {entity: assigned.table(entity) for entity in assigned.entities}
    tables["household"] = stripped
    tables["person"] = tables["person"].drop(columns=list(PERSON_FLAGS))
    bare = type(assigned)(
        tables,
        assigned.schema,
        {"household": assigned.weights_for("household")},
    )

    rederived, receipt = with_us_local_jurisdiction_flags(
        bare, rungs=BLOCK_GEOGRAPHY_RUNGS
    )

    for entity in ("household", "person"):
        pd.testing.assert_frame_equal(
            rederived.table(entity), assigned.table(entity), check_like=True
        )
    assert receipt["flags"]["in_wilmington"]["status"] == JURISDICTION_FLAG_DERIVED


def _co_ny_pa_puma_ladder() -> UsPumaLadder:
    # PUMA 0800101 spans Denver (600) and Arapahoe (400); PUMA 3600101 is in
    # Westchester; PUMA 4200101 is Philadelphia.
    puma = np.asarray([800101, 3600101, 4200101], dtype=np.int64)
    return UsPumaLadder(
        puma=puma,
        puma_population=np.asarray([1000.0, 500.0, 800.0]),
        cd_overlap_puma=puma.copy(),
        cd_overlap_cd=np.asarray([801, 3616, 4203], dtype=np.int64),
        cd_overlap_population=np.asarray([1000.0, 500.0, 800.0]),
        county_overlap_puma=np.asarray(
            [800101, 800101, 3600101, 4200101], dtype=np.int64
        ),
        county_overlap_county=np.asarray([8031, 8005, 36119, 42101], dtype=np.int32),
        county_overlap_population=np.asarray([600.0, 400.0, 500.0, 800.0]),
        tract_overlap_puma=np.asarray(
            [800101, 800101, 3600101, 4200101], dtype=np.int64
        ),
        tract_overlap_tract=np.asarray(
            [8031000100, 8005000100, 36119000100, 42101000100], dtype=np.int64
        ),
        tract_overlap_population=np.asarray([600.0, 400.0, 500.0, 800.0]),
        metadata={
            "schema_version": 1,
            "kind": "us_puma_ladder",
            "puma_vintage": "2020_puma",
            "sampling_basis": "population",
            "layers": {
                "congressional_district": {"vintage": "119th_congress"},
                "county": {"vintage": "2020_census"},
                "tract": {"vintage": "2020_census"},
            },
        },
    )


def test_puma_line_derives_county_flags_and_records_place_flags_unavailable() -> None:
    households = [
        {"state_fips": 8, "puma": 800101, "in_yonkers": True, "in_wilmington": False}
        for _ in range(40)
    ] + [
        {"state_fips": 36, "puma": 3600101, "in_yonkers": True, "in_wilmington": False},
        {
            "state_fips": 42,
            "puma": 4200101,
            "in_yonkers": False,
            "in_wilmington": False,
        },
    ]
    frame = us_frame(households, person_inputs=[[{}, {}] for _ in households])

    assigned = with_household_us_puma_ladder(frame, _co_ny_pa_puma_ladder(), seed=7)
    household = assigned.table("household")
    person = assigned.table("person")
    summary = us_puma_ladder_assignment_summary(household, _co_ny_pa_puma_ladder())

    for flag in PLACE_FLAGS:
        assert flag not in household
    county = household["county_fips"]
    assert household["in_denver"].tolist() == county.eq("08031").tolist()
    assert 0 < int(household["in_denver"].sum()) < 40  # both counties drawn
    philly = household.set_index("household_id")["county_fips"].eq("42101")
    assert person["pa_philadelphia_wage_tax_resident"].tolist() == (
        philly.loc[person["person_household_id"]].tolist()
    )
    statuses = {
        flag: entry["status"]
        for flag, entry in summary["local_jurisdiction_flags"]["flags"].items()
    }
    assert statuses == {
        "in_denver": JURISDICTION_FLAG_DERIVED,
        "in_wilmington": JURISDICTION_FLAG_UNAVAILABLE,
        "in_yonkers": JURISDICTION_FLAG_UNAVAILABLE,
        "pa_philadelphia_wage_tax_resident": JURISDICTION_FLAG_DERIVED,
    }
    assert us_local_jurisdiction_flag_failures(household) == []


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def _assigned_block_household() -> pd.DataFrame:
    frame = us_frame(households_on_blocks(_NAMES))
    return with_household_us_geography_ladder(frame, block_ladder()).table("household")


def test_gate_accepts_consistent_or_absent_flags() -> None:
    household = _assigned_block_household()

    assert us_local_jurisdiction_flag_failures(household) == []
    assert (
        us_local_jurisdiction_flag_failures(
            household.drop(columns=list(HOUSEHOLD_FLAGS))
        )
        == []
    )


@pytest.mark.parametrize("flag", HOUSEHOLD_FLAGS)
def test_gate_fails_a_flag_that_disagrees_with_its_rule(flag: str) -> None:
    household = _assigned_block_household()
    household[flag] = ~household[flag]

    failures = us_local_jurisdiction_flag_failures(household)
    block_gate = us_geography_ladder_gate(household, np.ones(len(household)))

    assert failures == [
        f"{flag}: {len(household)}/{len(household)} household row(s) disagree "
        f"with the {REGISTRY_RULES[flag]} rule for {REGISTRY_GEOIDS[flag]}"
    ]
    assert failures[0] in block_gate.failures


REGISTRY_RULES = {item.flag: item.rule for item in REGISTRY.jurisdictions}
REGISTRY_GEOIDS = {item.flag: item.geoid for item in REGISTRY.jurisdictions}


def test_gate_fails_null_or_non_boolean_flags() -> None:
    household = _assigned_block_household()
    household["in_denver"] = household["in_denver"].astype(object)
    household.loc[0, "in_denver"] = None
    household["in_yonkers"] = household["in_yonkers"].map({True: "yes", False: "no"})

    failures = us_local_jurisdiction_flag_failures(household)

    assert "in_denver: 1 null row(s)" in failures
    assert "in_yonkers: holds non-boolean values" in failures


def test_puma_gate_fails_a_place_flag_with_no_place_rung() -> None:
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "state_fips": [36, 36],
            "puma": ["3600101", "3600101"],
            "congressional_district_geoid": [3616, 3616],
            "county_fips": ["36119", "36119"],
            "county_fips_2023": ["36119", "36119"],
            "in_yonkers": [False, False],
        }
    )

    gate = us_puma_ladder_gate(household, np.ones(2), nyc_share_bounds=(0, 1))

    assert any(
        failure.startswith("in_yonkers: present but its place rung cannot decide it")
        for failure in gate.failures
    )


def test_person_flag_column_requires_resolvable_households() -> None:
    household = _household_table([("42101", "60000")])
    person = pd.DataFrame({"person_id": [1], "person_household_id": [99]})

    with pytest.raises(ValueError, match="Every person must belong"):
        apply_us_local_jurisdiction_flags(
            {"household": household, "person": person}, rungs=BLOCK_GEOGRAPHY_RUNGS
        )
    assert US_SCHEMA.membership_column("household") == "person_household_id"


def _assigned_block_frame():
    frame = us_frame(
        households_on_blocks(_NAMES), person_inputs=[[{}, {}] for _ in _NAMES]
    )
    return with_household_us_geography_ladder(frame, block_ladder(), seed=3)


def test_gate_requires_every_flag_its_rungs_decide() -> None:
    assigned = _assigned_block_frame()
    household = assigned.table("household").drop(columns=list(HOUSEHOLD_FLAGS))
    person = assigned.table("person").drop(columns=list(PERSON_FLAGS))

    block = us_local_jurisdiction_flag_failures(
        household, person=person, required_rungs=BLOCK_GEOGRAPHY_RUNGS
    )
    puma = us_local_jurisdiction_flag_failures(
        household, person=person, required_rungs=PUMA_GEOGRAPHY_RUNGS
    )

    assert sorted(failure.split(":")[0] for failure in block) == sorted(EXPECTED_RULES)
    assert all("absent" in failure for failure in block)
    assert sorted(failure.split(":")[0] for failure in puma) == [
        "in_denver",
        "pa_philadelphia_wage_tax_resident",
    ]
    # Without required rungs an absent flag passes (pre-flag artifacts).
    assert us_local_jurisdiction_flag_failures(household, person=person) == []


def test_gate_checks_each_person_flag_against_its_household() -> None:
    assigned = _assigned_block_frame()
    household = assigned.table("household")
    person = assigned.table("person").copy()
    flag = PERSON_FLAGS[0]

    assert us_local_jurisdiction_flag_failures(household, person=person) == []
    person.loc[person.index[0], flag] = not person.loc[person.index[0], flag]

    assert us_local_jurisdiction_flag_failures(household, person=person) == [
        f"{flag}: 1/{len(person)} person row(s) disagree with the county rule for 42101"
    ]
    # The household-only gate cannot see person flags.
    assert us_local_jurisdiction_flag_failures(household) == []


# --------------------------------------------------------------------------- #
# Re-deriving an existing artifact (exports, release builder)
# --------------------------------------------------------------------------- #


def _block_table(**columns: list[object]) -> pd.DataFrame:
    table = {
        "household_id": [1, 2],
        "county_fips": ["36119", "08031"],
        "place_fips": ["84000", "20000"],
        "block_geoid": ["361190001001000", "080310001001000"],
    }
    table.update(columns)
    return pd.DataFrame(table)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({}, BLOCK_GEOGRAPHY_RUNGS),
        # A pooled table: ACS rows carry no place and no block.
        ({"place_fips": ["84000", None]}, PUMA_GEOGRAPHY_RUNGS),
        (
            {"place_fips": ["84000", ""], "block_geoid": ["361190001001000", ""]},
            PUMA_GEOGRAPHY_RUNGS,
        ),
        (
            {"block_geoid": ["361190001001000", None]},
            PUMA_GEOGRAPHY_RUNGS,
        ),
        ({"county_fips": ["36119", None]}, ()),
    ],
)
def test_rungs_present_reads_the_artifacts_own_geography(columns, expected) -> None:
    assert us_local_jurisdiction_rungs_present(_block_table(**columns)) == expected


def test_rungs_present_needs_the_columns_themselves() -> None:
    table = _block_table()

    assert us_local_jurisdiction_rungs_present(table.drop(columns=["block_geoid"])) == (
        PUMA_GEOGRAPHY_RUNGS
    )
    assert us_local_jurisdiction_rungs_present(table.drop(columns=["place_fips"])) == (
        PUMA_GEOGRAPHY_RUNGS
    )
    assert (
        us_local_jurisdiction_rungs_present(table.drop(columns=["county_fips"])) == ()
    )


def _without_flags(frame):
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = tables["household"].drop(columns=list(HOUSEHOLD_FLAGS))
    tables["person"] = tables["person"].drop(columns=list(PERSON_FLAGS))
    return type(frame)(
        tables, frame.schema, {"household": frame.weights_for("household")}
    )


@settings(max_examples=40, deadline=None)
@given(st.lists(st.sampled_from(_NAMES), min_size=1, max_size=12))
def test_rederiving_a_pre_flag_artifact_restores_the_block_line_flags(names) -> None:
    frame = us_frame(households_on_blocks(names), person_inputs=[[{}] for _ in names])
    assigned = with_household_us_geography_ladder(frame, block_ladder(), seed=1)

    rederived, receipt = rederive_us_local_jurisdiction_flags(_without_flags(assigned))
    again, again_receipt = rederive_us_local_jurisdiction_flags(rederived)

    for entity in ("household", "person"):
        pd.testing.assert_frame_equal(
            rederived.table(entity), assigned.table(entity), check_like=True
        )
        pd.testing.assert_frame_equal(again.table(entity), rederived.table(entity))
    assert receipt["rungs"] == ["county", "place"]
    assert receipt["carried_flags"] == []
    assert again_receipt["carried_flags"] == sorted(EXPECTED_RULES)
    assert {entry["status"] for entry in receipt["flags"].values()} == {
        JURISDICTION_FLAG_DERIVED
    }


@pytest.mark.parametrize("flag", sorted(EXPECTED_RULES))
def test_rederiving_refuses_a_carried_flag_that_disagrees(flag: str) -> None:
    assigned = _assigned_block_frame()
    tables = {entity: assigned.table(entity) for entity in assigned.entities}
    entity = "person" if flag in PERSON_FLAGS else "household"
    tables[entity] = tables[entity].assign(**{flag: ~tables[entity][flag]})
    corrupt = type(assigned)(
        tables, assigned.schema, {"household": assigned.weights_for("household")}
    )

    with pytest.raises(ValueError, match=f"disagree with its own geography: {flag}"):
        rederive_us_local_jurisdiction_flags(corrupt)


def test_rederiving_a_pooled_artifact_leaves_place_flags_unavailable() -> None:
    # A donor row on the Yonkers block and an ACS row in Denver with no block
    # or place: the place rule cannot be decided for the ACS row.
    households = [
        {
            "state_fips": 36,
            "county_fips": "36119",
            "place_fips": "84000",
            "block_geoid": "361190001001000",
        },
        {
            "state_fips": 8,
            "county_fips": "08031",
            "place_fips": None,
            "block_geoid": None,
        },
    ]
    frame = us_frame(households, person_inputs=[[{}], [{}]])

    rederived, receipt = rederive_us_local_jurisdiction_flags(frame)

    household = rederived.table("household")
    assert "in_yonkers" not in household and "in_wilmington" not in household
    assert household["in_denver"].tolist() == [False, True]
    assert receipt["rungs"] == ["county"]
    assert receipt["flags"]["in_yonkers"]["status"] == JURISDICTION_FLAG_UNAVAILABLE
    assert receipt["carried_flags"] == []


def test_rederiving_without_a_county_writes_no_flag() -> None:
    rederived, receipt = rederive_us_local_jurisdiction_flags(
        us_frame([{"state_fips": 36}])
    )

    assert receipt["rungs"] == []
    assert not set(EXPECTED_RULES) & set(rederived.table("household"))
    assert {entry["status"] for entry in receipt["flags"].values()} == {
        JURISDICTION_FLAG_UNAVAILABLE
    }
    # A carried flag that no county could have decided is refused.
    with pytest.raises(ValueError, match="in_denver: present but its county rung"):
        rederive_us_local_jurisdiction_flags(
            us_frame([{"state_fips": 36, "in_denver": False}])
        )


# --------------------------------------------------------------------------- #
# Selections of a pooled support, and artifacts a step must keep unchanged
# --------------------------------------------------------------------------- #

# ACS rows carry a county from the PUMA ladder and no place or block.
_ACS_COUNTIES = ("08031", "36119", "42101", "10003", "06037")
_POOLED_ROWS = st.one_of(
    st.tuples(st.just("block"), st.sampled_from(_NAMES)),
    st.tuples(st.just("acs"), st.sampled_from(_ACS_COUNTIES)),
)
# A row with a state and no county, place or block. A support that holds one
# supplies no rung, so a selection that drops it can gain both.
_STATES_WITHOUT_A_COUNTY = ("08", "36", "42", "10", "06")
_ROWS_WITHOUT_A_COUNTY = st.tuples(
    st.just("state"), st.sampled_from(_STATES_WITHOUT_A_COUNTY)
)


def _pooled_household(row: tuple[str, str]) -> dict[str, object]:
    kind, value = row
    if kind == "block":
        block = FIXTURE_BLOCK_BY_NAME[value]
        return {
            "state_fips": block.state_fips,
            "county_fips": block.county_fips,
            "place_fips": "" if block.place_fips == 0 else f"{block.place_fips:05d}",
            "block_geoid": f"{block.block_geoid:015d}",
        }
    if kind == "state":
        return {
            "state_fips": int(value),
            "county_fips": None,
            "place_fips": None,
            "block_geoid": None,
        }
    return {
        "state_fips": int(value[:2]),
        "county_fips": value,
        "place_fips": None,
        "block_geoid": None,
    }


def _pooled_frame(rows: list[tuple[str, str]]):
    return us_frame(
        [_pooled_household(row) for row in rows],
        person_inputs=[[{}, {}] for _ in rows],
        weights=np.arange(1, len(rows) + 1, dtype=np.float64),
    )


def _select_households(frame, positions: list[int]):
    """Keep the households at ``positions``, as a frozen or L0 support does."""

    kept = frame.table("household")["household_id"].to_numpy()[positions]
    return frame.select(frame.table("person")["person_household_id"].isin(kept))


@st.composite
def _pool_and_selection(
    draw, *, rows_without_a_county: bool = False
) -> tuple[list[tuple[str, str]], list[int]]:
    rows = draw(st.lists(_POOLED_ROWS, min_size=1, max_size=10))
    if rows_without_a_county and draw(st.booleans()):
        extra = draw(st.lists(_ROWS_WITHOUT_A_COUNTY, min_size=1, max_size=2))
        rows = draw(st.permutations(rows + extra))
    keep = draw(st.sets(st.integers(0, len(rows) - 1), min_size=1))
    return rows, sorted(keep)


def _without_carried(receipt: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in receipt.items() if key != "carried_flags"}


def _derived_flags(receipt: dict[str, object]) -> list[str]:
    return sorted(
        flag
        for flag, entry in receipt["flags"].items()
        if entry["status"] == JURISDICTION_FLAG_DERIVED
    )


@settings(max_examples=80, deadline=None)
@example(case=([("block", "yonkers"), ("acs", "08031")], [0]))
@given(case=_pool_and_selection())
def test_rederiving_after_a_selection_equals_deriving_the_kept_rows(case) -> None:
    # Invariant: for every pooled support and every non-empty selection of its
    # households, deriving the selection of a derived support gives the same
    # tables and receipt as deriving the selected rows from scratch, and the
    # result carries every flag its own geography decides. Selection alone
    # can break the last property: the rungs belong to the whole table, so
    # dropping every row without a block makes the place-rule flags decidable.
    rows, keep = case
    frame = _pooled_frame(rows)
    derived, derived_receipt = rederive_us_local_jurisdiction_flags(frame)
    selected = _select_households(derived, keep)

    rederived, receipt = rederive_us_local_jurisdiction_flags(selected)
    fresh, fresh_receipt = rederive_us_local_jurisdiction_flags(
        _select_households(frame, keep)
    )

    for entity in ("household", "person"):
        pd.testing.assert_frame_equal(
            rederived.table(entity), fresh.table(entity), check_like=True
        )
    assert _without_carried(receipt) == _without_carried(fresh_receipt)
    complete = us_local_jurisdiction_artifact_receipt(rederived)
    assert _without_carried(complete) == _without_carried(receipt)
    assert complete["carried_flags"] == _derived_flags(receipt)
    if receipt["rungs"] != derived_receipt["rungs"]:
        # Only a gain is possible: the selection turned the place rung on.
        assert derived_receipt["rungs"] == list(PUMA_GEOGRAPHY_RUNGS)
        assert receipt["rungs"] == list(BLOCK_GEOGRAPHY_RUNGS)
        with pytest.raises(ValueError, match="in_yonkers: absent"):
            us_local_jurisdiction_artifact_receipt(selected)


@settings(max_examples=60, deadline=None)
@given(case=_pool_and_selection())
def test_rederiving_a_complete_support_returns_it_unchanged(case) -> None:
    # Invariant: re-deriving a support whose flags are already complete
    # returns the same object with the receipt writing them would give.
    rows, _keep = case
    derived, _receipt = rederive_us_local_jurisdiction_flags(_pooled_frame(rows))

    again, again_receipt = rederive_us_local_jurisdiction_flags(derived)
    written, written_receipt = with_us_local_jurisdiction_flags(
        derived,
        rungs=us_local_jurisdiction_rungs_present(derived.table("household")),
    )

    assert again is derived
    for entity in ("household", "person"):
        pd.testing.assert_frame_equal(again.table(entity), written.table(entity))
    assert _without_carried(again_receipt) == written_receipt
    assert again_receipt["carried_flags"] == _derived_flags(written_receipt)


def test_rederiving_rewrites_a_carried_flag_stored_without_a_boolean_dtype() -> None:
    assigned = _assigned_block_frame()
    tables = {entity: assigned.table(entity) for entity in assigned.entities}
    tables["household"] = tables["household"].assign(
        in_denver=tables["household"]["in_denver"].astype(object)
    )
    carried = type(assigned)(
        tables, assigned.schema, {"household": assigned.weights_for("household")}
    )

    rederived, receipt = rederive_us_local_jurisdiction_flags(carried)

    assert rederived is not carried
    assert rederived.table("household")["in_denver"].dtype == bool
    assert receipt["carried_flags"] == sorted(EXPECTED_RULES)


def test_artifact_receipt_refuses_an_artifact_without_the_flags_it_decides() -> None:
    assigned = _assigned_block_frame()

    with pytest.raises(ValueError, match="lacks or contradicts") as refused:
        us_local_jurisdiction_artifact_receipt(_without_flags(assigned))

    for flag in EXPECTED_RULES:
        assert f"{flag}: absent, but this table's" in str(refused.value)


@pytest.mark.parametrize("flag", sorted(EXPECTED_RULES))
def test_artifact_receipt_refuses_a_flag_that_disagrees(flag: str) -> None:
    assigned = _assigned_block_frame()
    tables = {entity: assigned.table(entity) for entity in assigned.entities}
    entity = "person" if flag in PERSON_FLAGS else "household"
    tables[entity] = tables[entity].assign(**{flag: ~tables[entity][flag]})
    corrupt = type(assigned)(
        tables, assigned.schema, {"household": assigned.weights_for("household")}
    )

    with pytest.raises(ValueError, match=f"{flag}: [0-9]+/[0-9]+ {entity} row"):
        us_local_jurisdiction_artifact_receipt(corrupt)


def test_artifact_receipt_refuses_a_place_flag_its_rungs_cannot_decide() -> None:
    # The ACS row stores its missing place and block as empty strings: the
    # stored in_yonkers agrees with the place rule, but no place rung
    # decided it for that row.
    households = [
        {
            "state_fips": 36,
            "county_fips": "36119",
            "place_fips": "84000",
            "block_geoid": "361190001001000",
        },
        {"state_fips": 36, "county_fips": "36119", "place_fips": "", "block_geoid": ""},
    ]
    derived, _receipt = rederive_us_local_jurisdiction_flags(
        us_frame(households, person_inputs=[[{}], [{}]])
    )
    tables = {entity: derived.table(entity) for entity in derived.entities}
    tables["household"] = tables["household"].assign(in_yonkers=[True, False])
    carrying = type(derived)(
        tables, derived.schema, {"household": derived.weights_for("household")}
    )

    with pytest.raises(
        ValueError,
        match=r"in_yonkers: present, but this artifact's rungs \['county'\] do not",
    ):
        us_local_jurisdiction_artifact_receipt(carrying)
    # Re-deriving removes it instead.
    rederived, receipt = rederive_us_local_jurisdiction_flags(carrying)
    assert "in_yonkers" not in rederived.table("household")
    assert receipt["flags"]["in_yonkers"]["status"] == JURISDICTION_FLAG_UNAVAILABLE


def test_artifact_receipt_matches_the_rederived_receipt_on_a_complete_artifact() -> (
    None
):
    assigned = _assigned_block_frame()

    receipt = us_local_jurisdiction_artifact_receipt(assigned)

    assert receipt == rederive_us_local_jurisdiction_flags(assigned)[1]
    assert receipt["carried_flags"] == sorted(EXPECTED_RULES)
    # An artifact with no county decides no flag, so it carries none.
    bare = us_local_jurisdiction_artifact_receipt(us_frame([{"state_fips": 36}]))
    assert bare["rungs"] == []
    assert bare["carried_flags"] == []


# --------------------------------------------------------------------------- #
# A selection held to the rungs of the support that fixed its inputs
# (review round 2 of the adopted microcosm#1453)
# --------------------------------------------------------------------------- #


@settings(max_examples=80, deadline=None)
@example(
    case=([("block", "yonkers"), ("block", "wilmington"), ("acs", "08031")], [0, 1])
)
@example(case=([("block", "yonkers"), ("acs", "08031")], [0, 1]))
@example(case=([("block", "wilmington"), ("state", "10")], [0]))
@given(case=_pool_and_selection(rows_without_a_county=True))
def test_a_selection_held_to_its_supports_rungs_keeps_that_supports_flags(
    case,
) -> None:
    # Invariant: for every pooled support and every non-empty selection of its
    # households, the selected rows of the derived support pass the artifact
    # receipt held to the support's rungs. The receipt reads those rungs and
    # derives the same flags, and each flag the selection's own geography
    # adds is unavailable with the reason. The selection already carries
    # exactly the support's flags, so a step that holds it to those rungs
    # writes no input the support's targets were not computed with. When
    # the selection adds no rung, the receipt is the unheld one. A support
    # with a row that has no county supplies no rung, so the flags a
    # selection adds can be county-rule flags as well as place-rule ones.
    rows, keep = case
    derived, derived_receipt = rederive_us_local_jurisdiction_flags(_pooled_frame(rows))
    selected = _select_households(derived, keep)

    receipt = us_local_jurisdiction_artifact_receipt(
        selected, rungs=derived_receipt["rungs"]
    )

    assert receipt["rungs"] == derived_receipt["rungs"]
    assert _derived_flags(receipt) == _derived_flags(derived_receipt)
    assert receipt["carried_flags"] == _derived_flags(derived_receipt)
    own = rederive_us_local_jurisdiction_flags(selected)[1]
    added = set(_derived_flags(own)) - set(_derived_flags(derived_receipt))
    assert bool(added) == (own["rungs"] != derived_receipt["rungs"])
    for flag in added:
        entry = receipt["flags"][flag]
        assert entry["status"] == JURISDICTION_FLAG_UNAVAILABLE
        assert entry["reason"].startswith(
            f"this artifact's own geography (rungs {own['rungs']}) decides it"
        )
    if not added:
        assert receipt == us_local_jurisdiction_artifact_receipt(selected)


def test_a_selection_held_to_a_support_with_no_rung_says_why_for_every_flag() -> None:
    # Review round 3 of the adopted microcosm#1453. The pool's second row has
    # a state and no county, so the support supplies no rung and decides no
    # flag. The kept Wilmington donor's own block decides all four, the
    # county-rule flags too, so each of the four is unavailable with the
    # reason that its own geography decides it, not that no rung exists.
    derived, derived_receipt = rederive_us_local_jurisdiction_flags(
        _pooled_frame([("block", "wilmington"), ("state", "10")])
    )
    assert derived_receipt["rungs"] == []
    selected = _select_households(derived, [0])
    own = rederive_us_local_jurisdiction_flags(selected)[1]
    assert own["rungs"] == list(BLOCK_GEOGRAPHY_RUNGS)

    receipt = us_local_jurisdiction_artifact_receipt(selected, rungs=[])

    assert receipt["rungs"] == []
    assert receipt["carried_flags"] == []
    assert sorted(receipt["flags"]) == sorted(EXPECTED_RULES)
    for flag in EXPECTED_RULES:
        assert receipt["flags"][flag]["status"] == JURISDICTION_FLAG_UNAVAILABLE
        assert receipt["flags"][flag]["reason"] == (
            "this artifact's own geography (rungs ['county', 'place']) decides "
            "it, but its inputs were fixed on a support with the rungs [], so "
            "the flag is not written and the engine default (False) applies to "
            "every row"
        )


def test_a_selection_cannot_be_held_to_rungs_its_geography_does_not_supply() -> None:
    # The pool's ACS row has no block, so its geography supplies the county
    # rung only, and the place rung's flags cannot be required of it.
    derived, _receipt = rederive_us_local_jurisdiction_flags(
        _pooled_frame([("block", "yonkers"), ("acs", "08031")])
    )

    with pytest.raises(
        ValueError, match=r"supplies the rungs \['county'\], not \['place'\]"
    ):
        us_local_jurisdiction_artifact_receipt(derived, rungs=BLOCK_GEOGRAPHY_RUNGS)


def test_a_selection_held_to_its_supports_rungs_refuses_a_flag_beyond_them() -> None:
    derived, receipt = rederive_us_local_jurisdiction_flags(
        _pooled_frame([("block", "yonkers"), ("block", "wilmington")])
    )
    assert receipt["rungs"] == list(BLOCK_GEOGRAPHY_RUNGS)

    with pytest.raises(
        ValueError,
        match=(
            r"in_yonkers: present, but the rungs \['county'\] its inputs were "
            r"fixed on do not decide it"
        ),
    ):
        us_local_jurisdiction_artifact_receipt(derived, rungs=PUMA_GEOGRAPHY_RUNGS)


# --------------------------------------------------------------------------- #
# A block outside its household's county (review round 1 of microcosm#1453)
# --------------------------------------------------------------------------- #

_PUMA_LADDER_METADATA = {
    "schema_version": 1,
    "kind": "us_puma_ladder",
    "puma_vintage": "2020_puma",
    "sampling_basis": "population",
    "layers": {
        "congressional_district": {"vintage": "119th_congress"},
        "county": {"vintage": "2020_census"},
        "tract": {"vintage": "2020_census"},
    },
}
# Where the redraw ladder below puts each fixture state's households: a
# county that holds no registry jurisdiction. New York's is the Bronx.
_REDRAWN_COUNTY = {
    6: "06037",
    8: "08005",
    10: "10001",
    36: "36005",
    37: "37129",
    42: "42045",
}


def _redraw_puma_ladder() -> UsPumaLadder:
    """One PUMA per fixture state, wholly inside :data:`_REDRAWN_COUNTY`."""

    county = np.asarray(
        sorted(int(code) for code in _REDRAWN_COUNTY.values()), dtype=np.int64
    )
    state = county // 1000
    puma = state * 10**5 + 200
    population = np.full(len(county), 100.0)
    return UsPumaLadder(
        puma=puma,
        puma_population=population,
        cd_overlap_puma=puma,
        cd_overlap_cd=state * 100 + 1,
        cd_overlap_population=population,
        county_overlap_puma=puma,
        county_overlap_county=county.astype(np.int32),
        county_overlap_population=population,
        tract_overlap_puma=puma,
        tract_overlap_tract=county * 10**6 + 100,
        tract_overlap_population=population,
        metadata=_PUMA_LADDER_METADATA,
    )


def _puma_ladder_from_blocks(arrays: dict[str, np.ndarray]) -> UsPumaLadder:
    """The PUMA ladder a block ladder implies.

    Block population summed by PUMA, and by (PUMA, district), (PUMA, county)
    and (PUMA, tract), as ``assemble_us_puma_ladder`` builds the real one.
    """

    block = arrays["block_geoid"].astype(np.int64)
    table = pd.DataFrame(
        {
            "puma": arrays["puma"].astype(np.int64),
            "district": arrays["congressional_district_geoid"].astype(np.int64),
            "county": block // 10**10,
            "tract": block // 10**4,
            "population": arrays["population"].astype(np.float64),
        }
    )

    def overlap(layer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grouped = table.groupby(["puma", layer], sort=True)["population"].sum()
        return (
            grouped.index.get_level_values(0).to_numpy(np.int64),
            grouped.index.get_level_values(1).to_numpy(np.int64),
            grouped.to_numpy(np.float64),
        )

    anchor = table.groupby("puma", sort=True)["population"].sum()
    district_puma, district, district_population = overlap("district")
    county_puma, county, county_population = overlap("county")
    tract_puma, tract, tract_population = overlap("tract")
    return UsPumaLadder(
        puma=anchor.index.to_numpy(np.int64),
        puma_population=anchor.to_numpy(np.float64),
        cd_overlap_puma=district_puma,
        cd_overlap_cd=district,
        cd_overlap_population=district_population,
        county_overlap_puma=county_puma,
        county_overlap_county=county.astype(np.int32),
        county_overlap_population=county_population,
        tract_overlap_puma=tract_puma,
        tract_overlap_tract=tract,
        tract_overlap_population=tract_population,
        metadata=_PUMA_LADDER_METADATA,
    )


def _location_ladder(arrays: dict[str, np.ndarray]):
    with TemporaryDirectory() as directory:
        return load_us_location_ladder(
            write_location_ladder(Path(directory) / "ladder.npz", arrays)
        )


def test_a_puma_redraw_cannot_bring_a_place_flag_back() -> None:
    # The reviewer's sequence. A household on the Yonkers block is redrawn to
    # the Bronx by the PUMA wrapper, and the result is then read as an
    # artifact. At d8e1490db the redraw left the block and its place beside
    # the new county, the table still looked block-anchored, and the
    # re-derivation wrote in_yonkers True on a Bronx household.
    frame = us_frame(households_on_blocks(["yonkers"]), person_inputs=[[{}]])

    located = with_household_us_geography_ladder(frame, block_ladder(), seed=1)
    placed = located.table("household")
    assert placed[["county_fips", "block_geoid", "place_fips"]].iloc[0].tolist() == [
        "36119",
        "361190001001000",
        "84000",
    ]
    assert placed["in_yonkers"].tolist() == [True]

    redrawn = with_household_us_puma_ladder(located, _redraw_puma_ladder(), seed=1)
    moved = redrawn.table("household")
    assert moved["county_fips"].tolist() == ["36005"]
    assert "block_geoid" not in moved and "place_fips" not in moved
    assert "in_yonkers" not in moved

    rederived, receipt = rederive_us_local_jurisdiction_flags(redrawn)
    household = rederived.table("household")
    assert "in_yonkers" not in household
    assert "in_wilmington" not in household
    assert household["in_denver"].tolist() == [False]
    assert receipt["rungs"] == ["county"]
    assert receipt["flags"]["in_yonkers"]["status"] == JURISDICTION_FLAG_UNAVAILABLE
    assert us_local_jurisdiction_artifact_receipt(rederived)["rungs"] == ["county"]


@pytest.mark.parametrize("assign_tract", [False, True])
def test_a_puma_redraw_leaves_no_column_only_a_block_decides(
    assign_tract: bool,
) -> None:
    frame = us_frame(
        [{**row, "tenure_type": "RENTED"} for row in households_on_blocks(_NAMES)],
        person_inputs=[[{}, {}] for _ in _NAMES],
    )
    located = with_household_us_geography_ladder(frame, block_ladder(), seed=3)
    before = located.table("household").copy()
    assert set(US_GEOGRAPHY_LADDER_COLUMNS) <= set(before.columns)

    redrawn = with_household_us_puma_ladder(
        located, _redraw_puma_ladder(), seed=3, assign_tract=assign_tract
    )
    after = redrawn.table("household")

    rewritten = set(US_PUMA_LADDER_COLUMNS)
    if assign_tract:
        rewritten.add(US_PUMA_LADDER_TRACT_COLUMN)
    for column in US_GEOGRAPHY_LADDER_COLUMNS:
        assert (column in after.columns) == (column in rewritten), column
    assert set(US_PUMA_LADDER_BLOCK_ANCHORED_COLUMNS) == {
        "block_geoid",
        "tract_geoid",
        "place_fips",
        "sldu",
        "sldl",
        "cbsa_code",
    }
    # Only those columns and the place-rule flags went.
    gone = set(before.columns) - set(after.columns)
    assert gone == (set(US_PUMA_LADDER_BLOCK_ANCHORED_COLUMNS) - rewritten) | set(
        PLACE_FLAGS
    )
    assert after["household_id"].tolist() == before["household_id"].tolist()
    assert after["tenure_type"].tolist() == before["tenure_type"].tolist()
    # What the ladder wrote agrees with itself.
    assert after["county_fips"].tolist() == [
        _REDRAWN_COUNTY[int(state)] for state in after["state_fips"]
    ]
    if assign_tract:
        assert (after["tract_geoid"].str[:5] == after["county_fips"]).all()
    assert us_local_jurisdiction_rungs_present(after) == PUMA_GEOGRAPHY_RUNGS
    assert not after["in_denver"].any()
    # The frame handed in keeps its block geography.
    pd.testing.assert_frame_equal(located.table("household"), before)


@pytest.mark.parametrize("assign_tract", [False, True])
def test_a_puma_redraw_of_a_block_v1_frame_leaves_no_block_lookup(
    assign_tract: bool,
) -> None:
    arrays = local_jurisdiction_blocks(4)
    ladder = _location_ladder(arrays)
    ladder = attach_congressional_district_plans(
        ladder,
        block_geoid=ladder.block_geoid,
        plans={"117th_congress": ladder.block_geoid // 10**13 * 100 + 1},
        plan_sources={
            "117th_congress": {"vintage": "117th_congress", "source": "test plan"}
        },
    )
    states = [8, 10, 36, 42] * 6
    frame = us_frame(
        [{"state_fips": state} for state in states],
        person_inputs=[[{}] for _ in states],
    )
    located, _table = with_household_us_block_location(frame, ladder, seed=17)
    before = located.table("household").copy()
    columns = location_geography_columns(ladder)
    plan_column = f"{US_CD_PLAN_COLUMN_PREFIX}117th_congress"
    assert plan_column in columns and set(columns) <= set(before.columns)
    assert us_block_location_gate(before, ladder).passed

    redrawn = with_household_us_puma_ladder(
        located,
        _puma_ladder_from_blocks(arrays),
        seed=17,
        assign_tract=assign_tract,
        expected_congressional_district_vintage="119th_congress",
    )
    after = redrawn.table("household")

    # Every column the draw looked up from its block is rewritten by the
    # PUMA ladder or gone, the second plan's district included.
    rewritten = set(US_PUMA_LADDER_COLUMNS)
    if assign_tract:
        rewritten.add(US_PUMA_LADDER_TRACT_COLUMN)
    for column in columns:
        assert (column in after.columns) == (column in rewritten), column
    assert US_PUMA_LADDER_BLOCK_PLAN_COLUMN_PREFIX == US_CD_PLAN_COLUMN_PREFIX
    assert plan_column not in after
    # The draw's PUMA is a valid anchor, so the ladder keeps it.
    assert after["puma"].tolist() == before["puma"].tolist()
    for flag in PLACE_FLAGS:
        assert flag not in after
    gate = us_puma_ladder_gate(
        after,
        redrawn.weights_for("household").values,
        assign_tract=assign_tract,
        nyc_share_bounds=(0.0, 1.0),
        nyc_share_of_new_york_bounds=(0.0, 1.0),
    )
    assert gate.passed, gate.failures
    rederived, receipt = rederive_us_local_jurisdiction_flags(redrawn)
    assert rederived is redrawn
    assert receipt["rungs"] == ["county"]


def test_the_puma_kernel_keeps_block_columns_and_the_result_is_refused() -> None:
    # assign_us_puma_ladder serves the pooled ACS line, which restores a
    # block-line donor's own county after it, so the kernel keeps every
    # column. Used alone on block-located households it leaves each block
    # beside a redrawn county, and nothing reads a rung from that table.
    frame = us_frame(households_on_blocks(_NAMES))
    located = with_household_us_geography_ladder(frame, block_ladder(), seed=3)

    assigned = assign_us_puma_ladder(
        located.table("household"), _redraw_puma_ladder(), seed=3
    )

    assert set(US_PUMA_LADDER_BLOCK_ANCHORED_COLUMNS) <= set(assigned.columns)
    moved = assigned["block_geoid"].str[:5] != assigned["county_fips"]
    assert 0 < int(moved.sum()) < len(assigned)
    with pytest.raises(
        ValueError, match=f"{int(moved.sum())}/{len(assigned)} household row"
    ):
        us_local_jurisdiction_rungs_present(assigned)


@pytest.mark.parametrize(
    ("columns", "count", "examples"),
    [
        pytest.param(
            {"county_fips": ["36005", "08031"]},
            "1/2",
            "[('361190001001000', '36005')]",
            id="another-county-of-the-state",
        ),
        pytest.param(
            {"county_fips": ["36119", "08035"]},
            "1/2",
            "[('080310001001000', '08035')]",
            id="a-county-that-differs-in-its-last-digit",
        ),
        pytest.param(
            {"county_fips": ["36005", "08005"]},
            "2/2",
            "[('361190001001000', '36005'), ('080310001001000', '08005')]",
            id="every-row",
        ),
        pytest.param(
            {
                "county_fips": ["36005", "08031"],
                "place_fips": ["84000", None],
                "block_geoid": ["361190001001000", None],
            },
            "1/2",
            "[('361190001001000', '36005')]",
            id="a-pooled-table-whose-donor-block-is-stale",
        ),
        pytest.param(
            {"county_fips": ["36005", None]},
            "1/2",
            "[('361190001001000', '36005')]",
            id="beside-a-row-without-a-county",
        ),
        pytest.param(
            {
                "county_fips": [b"36005", b"08031"],
                "place_fips": [b"84000", b"20000"],
                "block_geoid": [b"361190001001000", b"080310001001000"],
            },
            "1/2",
            "[('361190001001000', '36005')]",
            id="bytes-from-an-h5-round-trip",
        ),
    ],
)
def test_rungs_present_refuses_a_block_outside_its_county(
    columns, count: str, examples: str
) -> None:
    with pytest.raises(
        ValueError, match="block_geoid must begin with the row's county_fips"
    ) as refused:
        us_local_jurisdiction_rungs_present(_block_table(**columns))

    message = str(refused.value)
    assert f"{count} household row(s) do not, e.g. {examples} " in message
    assert "(block_geoid, county_fips)" in message


@pytest.mark.parametrize(
    "block",
    [None, "", "8031000100100", 80310001001000],
    ids=["null", "empty", "thirteen-digits", "integer"],
)
def test_rungs_present_compares_only_rows_that_carry_a_text_block(block) -> None:
    # The second row's county is not its (former) block's, but the row carries
    # no 15-digit text block: nothing anchors a place there, and the table
    # supplies the county rung as before.
    table = _block_table(
        county_fips=["36119", "08005"], block_geoid=["361190001001000", block]
    )

    assert us_local_jurisdiction_rungs_present(table) == PUMA_GEOGRAPHY_RUNGS


def _frame_with_a_stale_block(**flags: list[bool]):
    """A Yonkers household whose county became the Bronx, block and place kept.

    What d8e1490db's PUMA wrapper returned, beside a Denver household that is
    still on its block.
    """

    households: list[dict[str, object]] = [
        {
            "state_fips": 36,
            "county_fips": "36005",
            "place_fips": "84000",
            "block_geoid": "361190001001000",
        },
        {
            "state_fips": 8,
            "county_fips": "08031",
            "place_fips": "20000",
            "block_geoid": "080310001001000",
        },
    ]
    for flag, values in flags.items():
        for household, value in zip(households, values, strict=True):
            household[flag] = value
    return us_frame(households, person_inputs=[[{}], [{}]])


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"in_yonkers": [True, False]},
        {
            "in_denver": [False, True],
            "in_wilmington": [False, False],
            "in_yonkers": [True, False],
        },
    ],
    ids=["no-flag", "the-stale-place-flag", "every-household-flag"],
)
@pytest.mark.parametrize(
    "step",
    [rederive_us_local_jurisdiction_flags, us_local_jurisdiction_artifact_receipt],
)
def test_artifact_steps_refuse_a_block_outside_its_county(step, flags) -> None:
    with pytest.raises(
        ValueError, match="block_geoid must begin with the row's county_fips"
    ) as refused:
        step(_frame_with_a_stale_block(**flags))

    assert "1/2 household row(s) do not" in str(refused.value)
    assert "[('361190001001000', '36005')]" in str(refused.value)


def test_no_place_flag_is_derived_beside_a_block_outside_its_county() -> None:
    frame = _frame_with_a_stale_block()
    household = frame.table("household")
    tables = {"household": household, "person": frame.table("person")}

    with pytest.raises(ValueError, match="1/2 household row"):
        us_local_jurisdiction_flag_values(household, rungs=BLOCK_GEOGRAPHY_RUNGS)
    with pytest.raises(ValueError, match="1/2 household row"):
        apply_us_local_jurisdiction_flags(tables, rungs=BLOCK_GEOGRAPHY_RUNGS)
    with pytest.raises(ValueError, match="1/2 household row"):
        with_us_local_jurisdiction_flags(frame, rungs=BLOCK_GEOGRAPHY_RUNGS)
    # A line that has just drawn the county derives from that rung alone.
    values = us_local_jurisdiction_flag_values(household, rungs=PUMA_GEOGRAPHY_RUNGS)
    assert sorted(values) == ["in_denver", "pa_philadelphia_wage_tax_resident"]
    assert values["in_denver"].tolist() == [False, True]


def test_both_ladder_gates_fail_a_place_flag_beside_a_block_outside_its_county() -> (
    None
):
    household = _assigned_block_household().copy()
    household["puma"] = [f"{int(state):02d}00200" for state in household["state_fips"]]
    weights = np.ones(len(household))
    bounds = {
        "nyc_share_bounds": (0.0, 1.0),
        "nyc_share_of_new_york_bounds": (0.0, 1.0),
    }
    assert not [
        failure
        for failure in us_puma_ladder_gate(household, weights, **bounds).failures
        if failure.startswith("in_")
    ]
    yonkers = household["block_geoid"].eq("361190001001000")
    assert household.loc[yonkers, "in_yonkers"].tolist() == [True]
    # The county moves to the Bronx; the block, its place and the flag stay.
    household.loc[yonkers, ["county_fips", "county_fips_2023"]] = "36005"

    failures = us_local_jurisdiction_flag_failures(
        household, required_rungs=BLOCK_GEOGRAPHY_RUNGS
    )
    puma_gate = us_puma_ladder_gate(household, weights, **bounds)
    block_gate = us_geography_ladder_gate(household, weights, **bounds)

    assert [failure.split(":")[0] for failure in failures] == [
        "in_wilmington",
        "in_yonkers",
    ]
    for failure in failures:
        assert "present but its place rung cannot decide it" in failure
        assert f"1/{len(household)} household row(s) do not" in failure
        assert "[('361190001001000', '36005')]" in failure
        assert failure in puma_gate.failures
        assert failure in block_gate.failures
    # The block-ladder gate also sees it without any flag.
    assert (
        f"county_fips: 1/{len(household)} row(s) disagree with the block_geoid prefix"
        in block_gate.failures
    )


# Three blocks of Fairfield County, Connecticut (2020 county 09001), each in
# a different 2022 planning region, in block order.
_CT_BLOCKS = (
    FixtureBlock("stamford", 90010201011000, 73000, 903, 10.0),
    FixtureBlock("bridgeport", 90010701001000, 8000, 901, 30.0),
    FixtureBlock("naugatuck_valley", 90011101001000, 0, 902, 20.0),
)
_CT_PLANNING_REGIONS = ["09190", "09120", "09140"]


def _with_spine(frame: Frame, spine: str) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for entity, table in tables.items():
        table[spine_column(entity)] = spine
    if "puma" not in tables["household"]:
        tables["household"]["puma"] = pd.Series(
            [None] * len(tables["household"]), dtype=object
        )
    return Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")})


def test_connecticut_blocks_sit_in_their_county_fips_on_every_line() -> None:
    # Connecticut is the one state whose county-equivalents changed after the
    # 2020 blocks: a block there begins with its 2020 county (09001), and the
    # planning region that replaced it (09190, say) goes in county_fips_2023.
    # Every line writes the 2020 county into county_fips, so comparing a block
    # with county_fips refuses no Connecticut table; comparing it with
    # county_fips_2023 would refuse them all.
    blocks = list(_CT_BLOCKS)
    frame = us_frame(
        households_on_blocks_for(blocks), person_inputs=[[{}] for _ in blocks]
    )

    # The block ladder.
    located = with_household_us_geography_ladder(frame, block_ladder(blocks), seed=5)
    household = located.table("household")
    assert household["block_geoid"].tolist() == [
        f"{block.block_geoid:015d}" for block in blocks
    ]
    assert household["county_fips"].tolist() == ["09001"] * 3
    assert household["county_fips_2023"].tolist() == _CT_PLANNING_REGIONS
    assert us_local_jurisdiction_rungs_present(household) == BLOCK_GEOGRAPHY_RUNGS
    assert rederive_us_local_jurisdiction_flags(located)[0] is located
    assert us_local_jurisdiction_artifact_receipt(located)["rungs"] == [
        "county",
        "place",
    ]

    # The block_v1 draw, each household within its county.
    arrays = {
        "block_geoid": np.asarray([b.block_geoid for b in blocks], dtype=np.int64),
        "population": np.asarray([b.population for b in blocks], dtype=np.int64),
        "congressional_district_geoid": np.asarray(
            [b.congressional_district_geoid for b in blocks], dtype=np.int64
        ),
        "puma": np.full(len(blocks), 900100, dtype=np.int64),
        "sldu": np.full(len(blocks), "001", dtype="U3"),
        "sldl": np.full(len(blocks), "001", dtype="U3"),
        "place_fips": np.asarray([b.place_fips for b in blocks], dtype=np.int32),
        "cbsa_code": np.zeros(len(blocks), dtype=np.int32),
    }
    ladder = _location_ladder(arrays)
    drawn, _table = with_household_us_block_location(
        frame, ladder, seed=5, county_fips=np.full(len(blocks), 9001)
    )
    household = drawn.table("household")
    assert us_block_location_gate(household, ladder).passed
    assert household["county_fips"].tolist() == ["09001"] * 3
    assert set(household["county_fips_2023"]) <= set(_CT_PLANNING_REGIONS)
    assert (household["county_fips_2023"] != household["county_fips"]).all()
    assert us_local_jurisdiction_rungs_present(household) == BLOCK_GEOGRAPHY_RUNGS
    assert rederive_us_local_jurisdiction_flags(drawn)[0] is drawn

    # The PUMA ladder, redrawing the block-located households.
    puma_ladder = _puma_ladder_from_blocks(arrays)
    redrawn = with_household_us_puma_ladder(drawn, puma_ladder, seed=5)
    household = redrawn.table("household")
    assert household["county_fips"].tolist() == ["09001"] * 3
    assert set(household["county_fips_2023"]) <= set(_CT_PLANNING_REGIONS)
    assert not set(US_PUMA_LADDER_BLOCK_ANCHORED_COLUMNS) & set(household.columns)
    assert us_local_jurisdiction_rungs_present(household) == PUMA_GEOGRAPHY_RUNGS
    assert rederive_us_local_jurisdiction_flags(redrawn)[0] is redrawn

    # The pooled ACS line: block-line donors keep their blocks, ACS rows get
    # the PUMA ladder's 2020 county and no block.
    acs = us_frame([{"state_fips": 9, "puma": "0900100"} for _ in range(2)])
    pooled = with_optional_acs_spine(
        _with_spine(located, ASEC_PUF_SPINE),
        _with_spine(acs, ACS_2024_1YR_SPINE),
        max_peak_bytes=None,
        puma_ladder=puma_ladder,
        geography_seed=5,
    )
    household = pooled.table("household")
    donor = household["block_geoid"].notna()
    assert int(donor.sum()) == 3 and len(household) == 5
    assert household["county_fips"].tolist() == ["09001"] * 5
    assert (
        household.loc[donor, "block_geoid"].str[:5]
        == household.loc[donor, "county_fips"]
    ).all()
    assert us_local_jurisdiction_rungs_present(household) == PUMA_GEOGRAPHY_RUNGS
    assert rederive_us_local_jurisdiction_flags(pooled)[1]["rungs"] == ["county"]

    # A table that stored the planning region as county_fips is one no line
    # writes, and its blocks do not begin with it.
    regional = located.table("household").copy()
    regional["county_fips"] = regional["county_fips_2023"]
    with pytest.raises(ValueError, match="3/3 household row"):
        us_local_jurisdiction_rungs_present(regional)


# --- Properties ------------------------------------------------------------- #

_LADDER_SEEDS = (0, 3, 4)
_LINES = (
    "block_ladder",
    "puma_ladder",
    "puma_ladder_with_tract",
    "block_v1",
    "rederive",
    "drop_flags",
    "select",
)


@cache
def _jurisdiction_ladders(seed: int):
    """The block, location and PUMA ladders of one synthetic geography.

    Also the counties each flag's jurisdiction has blocks in, read from the
    ladder's own blocks with the four rules written out here.
    """

    arrays = local_jurisdiction_blocks(seed)
    block = arrays["block_geoid"].astype(np.int64)
    county = np.array([f"{value:05d}" for value in (block // 10**10).tolist()])
    state = block // 10**13
    place = arrays["place_fips"]
    counties = {
        "in_denver": {"08031"},
        "pa_philadelphia_wage_tax_resident": {"42101"},
        "in_yonkers": set(county[(state == 36) & (place == 84000)].tolist()),
        "in_wilmington": set(county[(state == 10) & (place == 77580)].tolist()),
    }
    return arrays, _location_ladder(arrays), _puma_ladder_from_blocks(arrays), counties


def _without_any_flag(frame):
    tables = {entity: frame.table(entity) for entity in frame.entities}
    for entity in ("household", "person"):
        tables[entity] = tables[entity].drop(
            columns=[flag for flag in EXPECTED_RULES if flag in tables[entity]]
        )
    return type(frame)(
        tables, frame.schema, {"household": frame.weights_for("household")}
    )


def _apply_line(frame, line: str, seed: int, ladders):
    _arrays, location, puma_ladder, _counties = ladders
    if line == "block_ladder":
        return with_household_us_geography_ladder(frame, location.blocks, seed=seed)
    if line == "puma_ladder":
        return with_household_us_puma_ladder(frame, puma_ladder, seed=seed)
    if line == "puma_ladder_with_tract":
        return with_household_us_puma_ladder(
            frame, puma_ladder, seed=seed, assign_tract=True
        )
    if line == "block_v1":
        return with_household_us_block_location(frame, location, seed=seed)[0]
    if line == "rederive":
        return rederive_us_local_jurisdiction_flags(frame)[0]
    if line == "drop_flags":
        return _without_any_flag(frame)
    assert line == "select"
    return _select_households(frame, list(range(0, frame.n("household"), 2)))


def _assert_no_flag_or_block_outside_its_county(frame, counties) -> None:
    household = frame.table("household")
    if "county_fips" not in household:
        assert not set(EXPECTED_RULES) & set(household.columns)
        return
    county_of = dict(
        zip(household["household_id"], household["county_fips"], strict=True)
    )
    for flag, allowed in counties.items():
        entity = "person" if flag in PERSON_FLAGS else "household"
        table = frame.table(entity)
        if flag not in table:
            continue
        members = (
            table[US_SCHEMA.membership_column("household")]
            if (entity == "person")
            else table["household_id"]
        )
        flagged = {county_of[key] for key in members[table[flag].astype(bool)]}
        assert flagged <= allowed, (flag, sorted(flagged))
    if "block_geoid" in household:
        assert (household["block_geoid"].str[:5] == household["county_fips"]).all()
    # Reading the rungs never refuses what the lines wrote, and the place
    # rung is there exactly when the households sit on blocks.
    rungs = us_local_jurisdiction_rungs_present(household)
    assert rungs == (
        BLOCK_GEOGRAPHY_RUNGS if "block_geoid" in household else PUMA_GEOGRAPHY_RUNGS
    )


@st.composite
def _line_sequences(draw):
    ladder_seed = draw(st.sampled_from(_LADDER_SEEDS))
    arrays = _jurisdiction_ladders(ladder_seed)[0]
    districts = sorted(set(arrays["congressional_district_geoid"].tolist()))
    households = draw(st.lists(st.sampled_from(districts), min_size=1, max_size=8))
    steps = draw(
        st.lists(
            st.tuples(st.sampled_from(_LINES), st.integers(0, 50)),
            min_size=1,
            max_size=6,
        )
    )
    return ladder_seed, households, steps


@settings(max_examples=80, deadline=None)
# The reviewer's sequence on a geography whose New York PUMAs span three
# counties: a block line, a PUMA redraw, then the artifact re-derivation.
@example(
    case=(
        3,
        [3601, 3602, 3603, 3601, 3602, 3603],
        [("block_ladder", 1), ("puma_ladder", 2), ("rederive", 0)],
    )
)
@example(
    case=(
        3,
        [3601, 3602, 3603, 4201, 4202, 800],
        [("block_v1", 5), ("puma_ladder_with_tract", 6), ("drop_flags", 0)],
    )
)
@given(case=_line_sequences())
def test_no_sequence_of_lines_leaves_a_flag_or_block_outside_its_county(case) -> None:
    # Invariant: after any sequence of the geography lines (block ladder, PUMA
    # ladder with or without a tract, block_v1 draw) and artifact steps
    # (re-derivation, a pre-flag artifact, a selection), no flag is true on a
    # household whose county holds none of the flag's jurisdiction, no
    # household carries a block outside its county, and the artifact
    # re-derivation of the result has the same two properties and carries
    # every flag its geography decides.
    ladder_seed, districts, steps = case
    ladders = _jurisdiction_ladders(ladder_seed)
    counties = ladders[3]
    frame = us_frame(
        [
            {"state_fips": district // 100, "congressional_district_geoid": district}
            for district in districts
        ],
        person_inputs=[[{}, {}] for _ in districts],
    )

    for line, seed in steps:
        frame = _apply_line(frame, line, seed, ladders)
        household = frame.table("household")
        if line.startswith("puma_ladder"):
            kept = set(US_PUMA_LADDER_BLOCK_ANCHORED_COLUMNS) & set(household.columns)
            assert kept == (
                {US_PUMA_LADDER_TRACT_COLUMN}
                if line == "puma_ladder_with_tract"
                else set()
            )
        _assert_no_flag_or_block_outside_its_county(frame, counties)
        derived, receipt = rederive_us_local_jurisdiction_flags(frame)
        _assert_no_flag_or_block_outside_its_county(derived, counties)
        complete = us_local_jurisdiction_artifact_receipt(derived)
        assert complete["carried_flags"] == _derived_flags(receipt)
        table = derived.table("household")
        for flag in HOUSEHOLD_FLAGS:
            decided = EXPECTED_RULES[flag][0] in receipt["rungs"]
            assert (flag in table) == decided, flag
            if decided:
                places = (
                    table["place_fips"] if "place_fips" in table else [""] * len(table)
                )
                assert table[flag].tolist() == [
                    _oracle(flag, county, place)
                    for county, place in zip(table["county_fips"], places, strict=True)
                ]


# A block inside each county, so a row can carry its own county's block,
# another county's, or none.
_TABLE_BLOCKS = {
    "08031": "080310001001000",
    "08035": "080350001001000",
    "10003": "100030001001000",
    "36005": "360050001001000",
    "36119": "361190001001000",
}
_TABLE_ROWS = st.tuples(
    st.sampled_from(sorted(_TABLE_BLOCKS)),
    st.one_of(st.none(), st.just(""), st.sampled_from(sorted(_TABLE_BLOCKS.values()))),
    st.sampled_from([None, "", "84000", "77580", "20000"]),
)


@settings(max_examples=200, deadline=None)
@example(rows=[("36005", "361190001001000", "84000")])
@example(rows=[("36119", "361190001001000", "84000"), ("08035", "080310001001000", "")])
@given(rows=st.lists(_TABLE_ROWS, min_size=1, max_size=8))
def test_rungs_are_never_read_from_a_table_with_a_block_outside_its_county(
    rows,
) -> None:
    # Invariant: for every household table, if any row carries a block that
    # does not begin with the row's county, reading the rungs raises (and so
    # do both artifact steps), naming how many rows; otherwise the rungs are
    # the block rungs exactly when every row has a block and a place, and the
    # place-rule flags the re-derivation writes are the rule on each row.
    frame = us_frame(
        [
            {
                "state_fips": int(county[:2]),
                "county_fips": county,
                "block_geoid": block,
                "place_fips": place,
            }
            for county, block, place in rows
        ],
        person_inputs=[[{}] for _ in rows],
    )
    household = frame.table("household")
    outside = [
        (block, county)
        for county, block, _place in rows
        if block and block[:5] != county
    ]

    if outside:
        for read in (
            lambda: us_local_jurisdiction_rungs_present(household),
            lambda: rederive_us_local_jurisdiction_flags(frame),
            lambda: us_local_jurisdiction_artifact_receipt(frame),
        ):
            with pytest.raises(
                ValueError, match="block_geoid must begin with the row's county_fips"
            ) as refused:
                read()
            assert f"{len(outside)}/{len(rows)} household row(s) do not" in str(
                refused.value
            )
        return

    anchored = all(block and place is not None for _county, block, place in rows)
    expected = BLOCK_GEOGRAPHY_RUNGS if anchored else PUMA_GEOGRAPHY_RUNGS
    assert us_local_jurisdiction_rungs_present(household) == expected
    derived, receipt = rederive_us_local_jurisdiction_flags(frame)
    assert receipt["rungs"] == list(expected)
    table = derived.table("household")
    for flag in PLACE_FLAGS:
        assert (flag in table) == anchored
        if anchored:
            assert table[flag].tolist() == [
                _oracle(flag, county, place) for county, _block, place in rows
            ]
