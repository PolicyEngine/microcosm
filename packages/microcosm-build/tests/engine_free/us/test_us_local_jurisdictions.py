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
- deriving a support whose flags are already complete returns it unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.geography_ladder import (
    assign_us_geography_ladder,
    us_geography_ladder_assignment_summary,
    us_geography_ladder_gate,
    with_household_us_geography_ladder,
)
from microcosm.build.us_runtime.local_jurisdictions import (
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
from microcosm.build.us_runtime.puma_ladder import (
    UsPumaLadder,
    us_puma_ladder_assignment_summary,
    us_puma_ladder_gate,
    with_household_us_puma_ladder,
)
from microcosm.frame import US_SCHEMA
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
    / "src/microcosm/build/us_runtime"
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
def _pool_and_selection(draw) -> tuple[list[tuple[str, str]], list[int]]:
    rows = draw(st.lists(_POOLED_ROWS, min_size=1, max_size=10))
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
