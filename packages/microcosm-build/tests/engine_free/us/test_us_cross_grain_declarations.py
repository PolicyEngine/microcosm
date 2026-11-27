"""The US nesting declarations and the adapter's path to the operator.

``us/cross_grain_declarations.json`` holds the US choices of the shared
cross-grain operator (microcosm#1760), under the key names of the UK's file.
These tests check:

- the registers that were Python literals in the local tools: the tools read
  them from the declaration under their old names, and each one is the
  literal it replaces (``test_support/microcosm_build/
  us_target_nesting_reference.py`` is the tool code before the move);
- the declaration's shape, and that a malformed one is refused when it loads;
- the typed views of it;
- one pass on the operator, and the record of the operator's receipts.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from importlib import resources as importlib_resources

import pytest

from microcosm.build import cross_grain_controls as controls
from microcosm.build.cross_grain import (
    EXACT_SIGNATURE_CONTROL,
    CrossGrainBridgeSelector,
)
from microcosm.build.us_runtime import cross_grain_declarations as declarations
from microcosm.build.us_runtime import target_nesting
from test_support.microcosm_build import us_target_nesting_reference as reference
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

DISTRICT_REGISTERS = (
    "STATE_CD_VINTAGE_RULE",
    "STATE_CD_DEFECTIVE_CD_FILE_MEASURES",
    "STATE_CD_UNPARENTED_CD_MEASURES",
    "STATE_CD_LEVEL_BRIDGES",
    "STATE_CD_EXCLUDED_CD_STATES",
    "STATE_CD_FACTOR_BAND",
    "RECONCILIATION_RTOL",
)
COUNTY_REGISTERS = (
    "COUNTY_SOI_REQUIRED_MEASURES",
    "DECLARED_SOI_COMBINED_CELLS",
    "SOI_COMBINED_CELL_BASIS",
)
PASSES = (
    declarations.DISTRICT_SOI,
    declarations.COUNTY_SOI,
    declarations.COUNTY_ACS_HOUSEHOLDS,
    declarations.COUNTY_POPULATION,
)


def _load_tool(name: str):
    """A tool module by its own name, as the local release tool imports it."""

    tools = _TEST_PATHS.repository / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def district_tool():
    return _load_tool("us_acs_local_cd_surface")


@pytest.fixture(scope="module")
def county_tool():
    return _load_tool("us_acs_local_county_surface")


def _same(new, old) -> bool:
    """Equal in value, type and, for a mapping, order of entries."""

    if type(new) is not type(old) or new != old:
        return False
    if isinstance(old, dict):
        return list(new.items()) == list(old.items()) and all(
            type(new[key]) is type(old[key]) for key in old
        )
    return True


@pytest.mark.parametrize("name", DISTRICT_REGISTERS)
def test_a_district_register_is_the_literal_it_replaces(district_tool, name) -> None:
    assert _same(getattr(district_tool, name), getattr(reference, name))


@pytest.mark.parametrize("name", COUNTY_REGISTERS)
def test_a_county_register_is_the_literal_it_replaces(county_tool, name) -> None:
    assert _same(getattr(county_tool, name), getattr(reference, name))


def test_the_tools_hold_copies_not_the_cached_declaration(district_tool) -> None:
    declared = declarations.us_district_soi_nesting()
    assert district_tool.STATE_CD_LEVEL_BRIDGES == declared.level_bridges
    assert district_tool.STATE_CD_LEVEL_BRIDGES is not declared.level_bridges


# --- The shape of the declaration --------------------------------------------


def test_the_declaration_names_the_operators_keys_as_the_uk_file_does() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    uk = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("cross_grain_declarations.json")
        .read_text(encoding="utf-8")
    )
    # A key of the US file is a key of the UK file, or one of the general
    # constructs that the UK does not use yet, or the section that goes at
    # the feed re-pin.
    general = {"bridge_selectors", "derived_controls", "same_measurement"}
    assert set(declared) - set(uk) == general | {
        "members",
        "registers",
        "until_the_feed_repin",
    }
    assert declared["schema_version"] == uk["schema_version"] == 1
    # The US levels are the feed's own geography levels, widest first.
    assert declared["grain_precedence"] == [
        "country",
        "state",
        "congressional_district",
        "county",
    ]
    assert declared["control_grains"] == ["country", "state"]
    assert set(declared["signature_fields"]) <= set(uk["signature_fields"])
    # No bridge by id and no partition: a selector makes the US bridges.
    assert declared["bridges"] == [] and declared["partitions"] == []
    # A relation and a local route read as the UK's do.
    relation_keys = {key for relation in uk["relations"] for key in relation}
    relation_kinds = {relation["relation"] for relation in uk["relations"]}
    for relation in declared["relations"]:
        assert set(relation) <= relation_keys
        assert relation["relation"] in relation_kinds
    for route in declared["local_routes"].values():
        assert route["route"] in {
            entry["route"] for entry in uk["local_routes"].values()
        }


def test_each_pass_is_declared_once_with_its_issue_row() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    rows = {
        entry.get("bridge_id", entry.get("id")): entry["issue_row"]
        for section in ("bridge_selectors", "derived_controls", "same_measurement")
        for entry in declared[section]
    }
    assert rows == {
        declarations.DISTRICT_SOI: 5,
        declarations.COUNTY_SOI: 6,
        declarations.COUNTY_ACS_HOUSEHOLDS: 7,
        declarations.COUNTY_POPULATION: 8,
    }
    # Districts and counties both divide a state.
    assert {pass_id: declarations.us_pass_grains(pass_id) for pass_id in PASSES} == {
        declarations.DISTRICT_SOI: ("state", "congressional_district"),
        declarations.COUNTY_SOI: ("state", "county"),
        declarations.COUNTY_ACS_HOUSEHOLDS: ("state", "county"),
        declarations.COUNTY_POPULATION: ("state", "county"),
    }
    with pytest.raises(ValueError, match="Unknown US cross-grain pass"):
        declarations.us_pass_grains("state_tract_soi")


def test_the_allocators_are_declared_to_go_at_the_feed_repin() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    assert "allocator" not in json.dumps(
        {key: value for key, value in declared.items() if key != "until_the_feed_repin"}
    )
    assert {pass_id: declarations.us_allocator_name(pass_id) for pass_id in PASSES} == {
        declarations.DISTRICT_SOI: "proportional_fsum",
        declarations.COUNTY_SOI: "share_residual_to_largest",
        declarations.COUNTY_ACS_HOUSEHOLDS: "share_residual_to_largest",
        declarations.COUNTY_POPULATION: "scale_residual_to_largest",
    }
    assert set(declared["until_the_feed_repin"]["allocators"].values()) <= set(
        controls.ALLOCATORS
    )
    assert declarations.us_allocator_name("another_pass") is None


def test_a_selector_is_the_operators_selector() -> None:
    district = declarations.us_bridge_selector(declarations.DISTRICT_SOI)
    assert district == CrossGrainBridgeSelector(
        bridge_id=declarations.DISTRICT_SOI,
        concept="irs_soi",
        higher={"source": "state_surface", "grain": "state"},
        lower={"source": "lower_file", "grain": "congressional_district"},
        pair_on=("identity",),
        max_factor_shift=0.25,
        min_factor_shift=-0.2,
        shift_around="median",
        out_of_bound="unbind",
    )
    # A caller can turn the bound off. The pairing stays.
    unbounded = declarations.us_bridge_selector(
        declarations.DISTRICT_SOI, bounded=False
    )
    assert unbounded.max_factor_shift is None and unbounded.out_of_bound == "refuse"
    assert (unbounded.higher, unbounded.lower) == (district.higher, district.lower)
    # The county bridge has the bound of the district bridge.
    county = declarations.us_bridge_selector(declarations.COUNTY_SOI)
    assert county.pair_on == ("source_measure_id", "source_period")
    assert (
        county.max_factor_shift,
        county.min_factor_shift,
        county.shift_around,
        county.out_of_bound,
    ) == (0.25, -0.2, "median", "unbind")
    with pytest.raises(ValueError, match="Unknown US cross-grain bridge selector"):
        declarations.us_bridge_selector(declarations.COUNTY_POPULATION)


def test_the_typed_views_read_the_declaration() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    district = declarations.us_district_soi_nesting()
    assert district.factor_band == 1.25
    assert (district.max_factor_shift, district.min_factor_shift) == (0.25, -0.2)
    # The two-sided bound is the old band: 1 / 1.25 is 1 - 0.2 in floats too.
    assert 1.0 + district.min_factor_shift == 1.0 / district.factor_band
    assert district.level_bridges == dict(
        declared["until_the_feed_repin"]["level_bridges"]["siblings"]
    )
    county = declarations.us_county_soi_nesting()
    assert county.factor_band == district.factor_band == 1.25
    assert county.required_measures == ("return_count", "adjusted_gross_income")
    assert county.combined_cells == {15005: 15009}
    assert "fewer than 20 returns" in county.combined_cell_basis


def _tampered(change) -> dict:
    payload = copy.deepcopy(dict(declarations.load_us_cross_grain_declarations()))
    change(payload)
    return payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(control_grains=["country", "tract"]), "control_grains"),
        (
            lambda d: d["bridge_selectors"][0]["lower"].update(grain="state"),
            "a control grain above its lower grain",
        ),
        (
            lambda d: d["bridge_selectors"][0].update(pair_on=[]),
            "what its sides are paired on",
        ),
        (
            lambda d: d["bridge_selectors"][1].update(reason=" "),
            "what its sides are paired on",
        ),
        (
            lambda d: d["derived_controls"][0].update(derive="mean_of_lower_rows"),
            "unknown derivation",
        ),
        (
            lambda d: d["same_measurement"][0].update(higher_grain="county"),
            "a control grain above its lower grain",
        ),
        (
            lambda d: d["same_measurement"][0].update(
                id=d["derived_controls"][0]["bridge_id"]
            ),
            "pass ids repeat",
        ),
        (
            lambda d: d["until_the_feed_repin"]["allocators"].update(
                {declarations.COUNTY_SOI: "numpy_sum"}
            ),
            "must name a declared pass and one of",
        ),
        (
            lambda d: d["until_the_feed_repin"]["allocators"].update(
                {"another_pass": "proportional_fsum"}
            ),
            "must name a declared pass and one of",
        ),
        (lambda d: d["relations"][0].update(relation="equal"), "is malformed"),
        (lambda d: d["relations"][0].update(reason=""), "is malformed"),
        (
            lambda d: d["relations"][0].update(whole="usda_snap.snap_total@state"),
            "is malformed",
        ),
        (
            lambda d: d["relations"][0].update(
                targets=["usda_snap.snap_total@tract", "usda_snap.snap_total@country"],
                part="usda_snap.snap_total@tract",
            ),
            "family.role@level",
        ),
        (
            lambda d: d["local_routes"].update(
                {"ssa@tract": d["local_routes"]["ssa@state"]}
            ),
            "family@level",
        ),
        (
            lambda d: d["local_routes"]["ssa@state"].update(route="merged"),
            "with a reason",
        ),
        (
            lambda d: d["until_the_feed_repin"]["compile_rules"][0].update(issues=[]),
            "a reason and an issue",
        ),
        (
            lambda d: d["until_the_feed_repin"]["known_gaps"][0].update(
                levels=["tract"]
            ),
            "a reason and an issue",
        ),
        (
            lambda d: d["until_the_feed_repin"]["compile_rules"][0].pop("marker"),
            "the marker it writes",
        ),
    ],
)
def test_a_malformed_declaration_is_refused_when_the_file_loads(
    change, message
) -> None:
    with pytest.raises(ValueError, match=message):
        declarations._validate(_tampered(change))
    declarations._validate(_tampered(lambda payload: None))


# --- One pass on the operator ------------------------------------------------


def _population_rows(*states: tuple[str, float, tuple[float, ...]]):
    """A state control row and its county rows, for each state."""

    rows, legs = [], {}
    for state, control, values in states:
        area = target_nesting.geography_id("state", state)
        rows.append(controls.SurfaceRow("state", area, "population", control))
        for index, value in enumerate(values, start=1):
            county = target_nesting.geography_id("county", f"{state}{index:03d}")
            legs[county] = area
            rows.append(controls.SurfaceRow("county", county, "population", value))
    return rows, legs


def test_a_pass_runs_on_the_operator_with_its_declared_allocator() -> None:
    rows, legs = _population_rows(("06", 100.0, (30.0, 20.0)), ("36", 8.0, (1.0, 3.0)))
    outcomes, receipt = target_nesting.run_pass(
        declarations.COUNTY_POPULATION,
        rows,
        bound_higher_targets=["population"],
        legs=legs,
    )
    assert [outcome.value for outcome in outcomes] == [
        100.0,
        60.0,
        40.0,
        8.0,
        2.0,
        6.0,
    ]
    assert [(outcome.status, outcome.parent) for outcome in outcomes] == [
        (controls.UNTOUCHED, None),
        (controls.RECONCILED, 0),
        (controls.RECONCILED, 0),
        (controls.UNTOUCHED, None),
        (controls.RECONCILED, 3),
        (controls.RECONCILED, 3),
    ]
    assert {outcome.control for outcome in outcomes[1:3]} == {EXACT_SIGNATURE_CONTROL}
    # One group for the quantity, one leg for each state, and the operator's
    # receipt names the allocator that the declaration gives the pass.
    (group,) = receipt["groups"]
    assert (group["winning_grain"], group["lower_grain"]) == ("state", "county")
    assert sorted(leg["parent_geography_id"] for leg in group["legs"]) == [
        "state:06",
        "state:36",
    ]
    assert {leg["allocator"] for leg in group["legs"]} == {"scale_residual_to_largest"}
    # The operator's own refusals reach the caller.
    rows, legs = _population_rows(("06", 2.0, (0.0, 0.0)))
    with pytest.raises(ValueError, match="vanishing lower-leg total"):
        target_nesting.run_pass(
            declarations.COUNTY_POPULATION,
            rows,
            bound_higher_targets=["population"],
            legs=legs,
        )


def test_the_operators_receipts_are_recorded_only_when_asked(tmp_path) -> None:
    rows, legs = _population_rows(("06", 100.0, (30.0, 20.0)))

    def run():
        return target_nesting.run_pass(
            declarations.COUNTY_POPULATION,
            rows,
            bound_higher_targets=["population"],
            legs=legs,
        )

    recorded = target_nesting.start_nesting_receipts()
    _outcomes, receipt = run()
    assert recorded == [
        {
            "pass": declarations.COUNTY_POPULATION,
            "allocator": "scale_residual_to_largest",
            "operator_receipt": receipt,
        }
    ]

    path = target_nesting.write_nesting_receipts(tmp_path, recorded)
    assert path.name == target_nesting.NESTING_RECEIPTS_FILENAME
    written = json.loads(path.read_text())
    assert written["operator"] == "microcosm.build.cross_grain"
    assert written["declarations"] == "us/cross_grain_declarations.json"
    assert [entry["pass"] for entry in written["passes"]] == [
        declarations.COUNTY_POPULATION
    ]

    # A new record starts empty, and the earlier list gains nothing more.
    fresh = target_nesting.start_nesting_receipts()
    run()
    assert len(fresh) == 1 and len(recorded) == 1
    # After a stop, a pass records nothing.
    target_nesting.stop_nesting_receipts()
    run()
    assert len(fresh) == 1
