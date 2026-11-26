"""The US nesting declarations and the adapter's path to the operator.

``us/cross_grain_declarations.json`` holds the US choices of the shared
cross-grain operator (microcosm#1760). The registers in it were Python
literals in the local tools; the tools read them from the declaration under
their old names. These tests hold each one to the literal it replaces
(``test_support/microcosm_build/us_target_nesting_reference.py`` is the tool
code before the move), and check the declaration's shape and the adapter's
block interface.
"""

from __future__ import annotations

import importlib
import json
import sys
from importlib import resources as importlib_resources

import pytest

from microcosm.build import cross_grain_controls
from microcosm.build.us_runtime import cross_grain_declarations as declarations
from microcosm.build.us_runtime import target_nesting
from microcosm.build.us_runtime.target_nesting import NestingBlock
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
    "STATE_CD_INTERNAL_RTOL",
    "RECONCILIATION_RTOL",
)
COUNTY_REGISTERS = (
    "COUNTY_SOI_REQUIRED_MEASURES",
    "DECLARED_SOI_COMBINED_CELLS",
    "SOI_COMBINED_CELL_BASIS",
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


def test_the_declaration_names_the_operators_keys_as_the_uk_file_does() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    uk = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("cross_grain_declarations.json")
        .read_text(encoding="utf-8")
    )
    # Every US key but the US-only sections is a key of the UK file.
    us_only = {"mechanisms", "inventory"}
    assert set(declared) - us_only <= set(uk)
    assert declared["schema_version"] == uk["schema_version"] == 1
    # The US levels are the feed's own geography levels, widest first.
    assert declared["grain_precedence"] == [
        "country",
        "state",
        "congressional_district",
        "county",
    ]
    assert declared["control_grains"] == ["country", "state"]
    # No UK-style bridge or partition is in force: each mechanism ties a
    # control to its rows by an exact measurement signature.
    assert declared["bridges"] == [] and declared["partitions"] == []


def test_each_mechanism_declares_its_rule() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    mechanisms = declared["mechanisms"]
    assert list(mechanisms) == [
        declarations.DISTRICT_SOI,
        declarations.COUNTY_POPULATION,
        declarations.COUNTY_SOI,
        declarations.COUNTY_ACS_HOUSEHOLDS,
    ]
    assert {
        mechanism_id: (
            mechanism["issue_row"],
            mechanism["allocator"],
            mechanism["control"]["kind"],
        )
        for mechanism_id, mechanism in mechanisms.items()
    } == {
        declarations.DISTRICT_SOI: (
            5,
            "proportional_fsum",
            "bound_row_or_scaled_by_sibling_ratio",
        ),
        declarations.COUNTY_POPULATION: (8, "scale_residual_to_largest", "bound_row"),
        declarations.COUNTY_SOI: (6, "share_residual_to_largest", "bound_row"),
        declarations.COUNTY_ACS_HOUSEHOLDS: (
            7,
            "share_residual_to_largest",
            "sum_of_lower_rows",
        ),
    }
    for mechanism in mechanisms.values():
        assert mechanism["allocator"] in cross_grain_controls.ALLOCATORS
        assert mechanism["higher_grain"] in declared["control_grains"]
        assert mechanism["members"]["source"]
        assert mechanism["control"]["description"]
        assert "lower_rows" in mechanism["metadata_keys"]
        assert (
            _TEST_PATHS.repository / mechanism["tool_function"].split(":")[0]
        ).exists()


def test_an_unknown_mechanism_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown US cross-grain mechanism"):
        declarations.us_cross_grain_mechanism("state_tract_soi")


def _block(state: str, values: tuple[float, ...], control: float, concept="quantity"):
    return NestingBlock(
        concept=concept,
        filters=(("band", "all"),),
        control_geography_id=target_nesting.geography_id("state", state),
        control_value=control,
        lower_geography_ids=tuple(
            target_nesting.geography_id("county", f"{state}{index:03d}")
            for index in range(1, len(values) + 1)
        ),
        lower_values=values,
    )


def test_blocks_go_through_the_operator_with_the_declared_allocator() -> None:
    blocks = [
        _block("06", (30.0, 20.0), 100.0),
        _block("36", (1.0, 3.0), 8.0),
        _block("06", (5.0, 5.0), 40.0, concept="other_quantity"),
    ]
    nested, receipt = target_nesting.reconcile_blocks(
        declarations.COUNTY_POPULATION, blocks
    )

    assert [rows.values for rows in nested] == [(60.0, 40.0), (2.0, 6.0), (20.0, 20.0)]
    assert [(rows.factor, rows.lower_sum) for rows in nested] == [
        (2.0, 50.0),
        (2.0, 4.0),
        (4.0, 10.0),
    ]
    # One group for each quantity, one leg for each state, and the operator's
    # receipt names the allocator the mechanism declares.
    assert len(receipt["groups"]) == 2
    legs = [leg for group in receipt["groups"] for leg in group["legs"]]
    assert sorted(leg["parent_geography_id"] for leg in legs) == [
        "state:06",
        "state:06",
        "state:36",
    ]
    assert {leg["allocator"] for leg in legs} == {"scale_residual_to_largest"}
    assert {group["winning_grain"] for group in receipt["groups"]} == {"state"}
    assert {group["lower_grain"] for group in receipt["groups"]} == {"county"}


def test_a_block_the_operator_cannot_place_is_refused() -> None:
    mechanism = declarations.COUNTY_POPULATION
    with pytest.raises(ValueError, match="two blocks"):
        target_nesting.reconcile_blocks(
            mechanism, [_block("06", (1.0,), 2.0), _block("06", (1.0,), 2.0)]
        )
    stray = NestingBlock(
        "quantity",
        (),
        target_nesting.geography_id("state", "36"),
        2.0,
        (target_nesting.geography_id("county", "06001"),),
        (1.0,),
    )
    with pytest.raises(ValueError, match="sits under two controls"):
        target_nesting.reconcile_blocks(
            mechanism,
            [
                NestingBlock(
                    "other",
                    (),
                    target_nesting.geography_id("state", "06"),
                    2.0,
                    (target_nesting.geography_id("county", "06001"),),
                    (1.0,),
                ),
                stray,
            ],
        )
    with pytest.raises(ValueError, match="at least one row"):
        target_nesting.reconcile_blocks(mechanism, [_block("06", (), 2.0)])
    # The operator's own refusals reach the caller.
    with pytest.raises(ValueError, match="vanishing lower-leg total"):
        target_nesting.reconcile_blocks(mechanism, [_block("06", (0.0, 0.0), 2.0)])


def test_the_operators_receipts_are_recorded_only_when_asked(tmp_path) -> None:
    blocks = [_block("06", (30.0, 20.0), 100.0)]
    recorded = target_nesting.start_nesting_receipts()
    _nested, receipt = target_nesting.reconcile_blocks(
        declarations.COUNTY_POPULATION, blocks
    )
    assert [entry["mechanism"] for entry in recorded] == [
        declarations.COUNTY_POPULATION
    ]
    assert recorded[0]["operator_receipt"] == receipt
    assert (recorded[0]["issue_row"], recorded[0]["blocks"]) == (8, 1)

    path = target_nesting.write_nesting_receipts(tmp_path, recorded)
    assert path.name == target_nesting.NESTING_RECEIPTS_FILENAME
    written = json.loads(path.read_text())
    assert written["operator"] == "microcosm.build.cross_grain"
    assert [entry["mechanism"] for entry in written["passes"]] == [
        declarations.COUNTY_POPULATION
    ]

    # A new record starts empty, and the earlier list gains nothing more.
    fresh = target_nesting.start_nesting_receipts()
    target_nesting.reconcile_blocks(declarations.COUNTY_POPULATION, blocks)
    assert len(fresh) == 1 and len(recorded) == 1
    # After a stop, a pass records nothing.
    target_nesting.stop_nesting_receipts()
    target_nesting.reconcile_blocks(declarations.COUNTY_POPULATION, blocks)
    assert len(fresh) == 1
