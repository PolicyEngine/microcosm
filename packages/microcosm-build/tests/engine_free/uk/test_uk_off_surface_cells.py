"""Signed cells that are off the surface still count in their leg (#1123)."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.ledger_targets import (
    load_uk_local_target_reference_membership,
    uk_local_target_surface,
    uk_off_surface_cells,
)
from microcosm.build.uk_runtime.off_surface_cells import (
    UK_IMPLIED_SHARE_MAX_RATIO,
    uk_off_surface_frame_rows,
    uk_signed_cells_off_the_surface,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

_TARGET = "hmrc.employment_income.count"


def _leg_of_area(area: str) -> str:
    return area.split("-", 1)[0]


def _bound(grain: str, area: str, value: float, published: float | None = None):
    return {
        "target_id": _TARGET,
        "grain": grain,
        "geography_id": area,
        "value": value,
        "published_value": value if published is None else published,
    }


def _off(
    level: str,
    area: str,
    value: float | None,
    target_id: str = _TARGET,
    area_has_bound_cells: bool = True,
):
    return {
        "target_id": target_id,
        "geography_level": level,
        "geography_id": area,
        "value": value,
        "reason_id": "test",
        "area_has_bound_cells": area_has_bound_cells,
    }


def _both_tiers(leg: str, constituency: float, la: float) -> list[dict]:
    return [
        _bound("constituency", f"{leg}-c", constituency),
        _bound("la", f"{leg}-la", la),
    ]


def test_a_deferred_cell_joins_the_frame_moved_as_its_leg_was_moved():
    # The leg's bound cell was published at 100 and stands at 110 in the frame.
    rows, receipt = uk_off_surface_frame_rows(
        [_bound("la", "A-la", 110.0, published=100.0)],
        [_off("local_authority", "A-city", 20.0)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "A")},
    )
    (row,) = rows
    assert row == {
        "grain": "la",
        "geography_id": "A-city",
        "target_id": f"contract:{_TARGET}",
        "_output_position": None,
        "value": pytest.approx(22.0),
        "published_value": 20.0,
    }
    (entry,) = receipt["deferred_with_value"]
    assert entry["areas"] == ["A-city"]
    assert entry["share_of_leg"] == pytest.approx(22.0 / 132.0)


def test_an_absent_cell_is_implied_from_the_whole_tier():
    # Legs A and B carry both tiers whole, at a ratio of 0.99. Leg C's
    # local-authority tier lacks an area: 0.99 * 200 less the 150 it carries.
    bound = [
        *_both_tiers("A", 100.0, 99.0),
        *_both_tiers("B", 300.0, 297.0),
        *_both_tiers("C", 200.0, 150.0),
    ]
    rows, receipt = uk_off_surface_frame_rows(
        bound,
        [_off("local_authority", "C-y", None), _off("local_authority", "C-x", None)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "C")},
    )
    (row,) = rows
    assert row["geography_id"] == "C-x"
    assert row["grain"] == "la"
    assert row["value"] == pytest.approx(48.0)
    (entry,) = receipt["implied_absent"]
    assert entry["areas"] == ["C-x", "C-y"]
    assert entry["implied_from_grain"] == "constituency"
    assert entry["whole_legs"] == 2
    assert entry["tier_ratio"] == pytest.approx(0.99)
    assert entry["share_of_leg"] == pytest.approx(48.0 / 198.0)
    assert entry["status"] == "added"


def test_a_leg_short_at_either_tier_is_not_read_as_whole():
    # Leg B lacks a constituency, so only leg A gives the tiers' ratio.
    bound = [
        *_both_tiers("A", 100.0, 100.0),
        *_both_tiers("B", 50.0, 100.0),
        *_both_tiers("C", 200.0, 150.0),
    ]
    rows, receipt = uk_off_surface_frame_rows(
        bound,
        [_off("constituency", "B-x", None), _off("local_authority", "C-x", None)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "C"), (_TARGET, "constituency", "B")},
    )
    assert {row["geography_id"]: row["value"] for row in rows} == {
        "B-x": pytest.approx(50.0),
        "C-x": pytest.approx(50.0),
    }
    assert {entry["whole_legs"] for entry in receipt["implied_absent"]} == {1}


def _short_leg(households: dict[tuple[str, str], float] | None, **kwargs):
    # Leg A is whole at a ratio of one. Leg C's authorities hold 150 of the
    # 200 its constituencies hold, so the absent authority is implied at 50,
    # a quarter of the leg.
    return uk_off_surface_frame_rows(
        [*_both_tiers("A", 100.0, 100.0), *_both_tiers("C", 200.0, 150.0)],
        [_off("local_authority", "C-x", None)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "C")},
        share_reference=households,
        **kwargs,
    )


def test_an_implied_cell_is_bounded_by_its_areas_share_of_the_legs_households():
    # The absent authority holds 20% of the leg's households: a quarter of
    # the leg's employees is 1.25 times that, inside the bound.
    households = {("la", "C-la"): 800.0, ("la", "C-x"): 200.0, ("la", "A-la"): 500.0}
    rows, receipt = _short_leg(households)
    (entry,) = receipt["implied_absent"]
    assert entry["household_share"] == pytest.approx(0.2)
    assert entry["share_ratio"] == pytest.approx(1.25)
    assert UK_IMPLIED_SHARE_MAX_RATIO == 1.5
    assert len(rows) == 1
    # With 10% of the households, a quarter of the leg is an undercount of
    # the bound cells, not the absent authority: refused.
    households[("la", "C-la")] = 1_800.0
    with pytest.raises(ValueError, match="more than 1.5 times their 0.1000"):
        _short_leg(households)


def test_an_implied_cell_with_no_households_to_set_it_against_is_refused():
    households = {("la", "C-la"): 800.0, ("la", "A-la"): 500.0}
    with pytest.raises(ValueError, match=r"\['C-x'\] carry no census households"):
        _short_leg(households)


def test_a_surface_with_no_census_households_says_the_row_is_unbounded():
    for households in (None, {("la", "A-la"): 500.0}):
        rows, receipt = _short_leg(households)
        (entry,) = receipt["implied_absent"]
        assert entry["household_share"] is None
        assert entry["share_bound"].startswith("unbounded")
        assert len(rows) == 1


def test_a_deferred_cell_counts_in_a_leg_no_control_moves_when_tiers_are_compared():
    # Leg A's authorities hold 90 with a further 10 deferred. No control moves
    # leg A, so the deferred cell adds no row there, but leg A still reads
    # whole (ratio one) when leg C's absent authority is implied.
    rows, receipt = uk_off_surface_frame_rows(
        [*_both_tiers("A", 100.0, 90.0), *_both_tiers("C", 200.0, 150.0)],
        [_off("local_authority", "A-s", 10.0), _off("local_authority", "C-x", None)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "C")},
    )
    (row,) = rows
    assert (row["geography_id"], row["value"]) == ("C-x", pytest.approx(50.0))
    assert receipt["deferred_with_value"] == []


def test_a_deferred_cell_joins_where_a_control_moves_the_other_tier_of_its_leg():
    # A control moves leg A's authorities, whose parent is the constituency
    # total: the constituency tier's deferred cell must be in that total.
    rows, _ = uk_off_surface_frame_rows(
        _both_tiers("A", 100.0, 100.0),
        [_off("constituency", "A-s", 10.0)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "A")},
    )
    (row,) = rows
    assert (row["grain"], row["geography_id"], row["value"]) == (
        "constituency",
        "A-s",
        10.0,
    )


def test_cells_of_a_leg_no_control_moves_stay_off_the_frame():
    rows, receipt = uk_off_surface_frame_rows(
        [*_both_tiers("A", 100.0, 100.0), *_both_tiers("C", 200.0, 150.0)],
        [_off("local_authority", "C-x", None), _off("local_authority", "A-x", 5.0)],
        leg_of_area=_leg_of_area,
        controlled_legs=set(),
    )
    assert rows == []
    assert receipt == {"deferred_with_value": [], "implied_absent": [], "rows_added": 0}


def test_a_leg_with_no_bound_cell_is_left_to_its_licence():
    rows, _ = uk_off_surface_frame_rows(
        _both_tiers("A", 100.0, 100.0),
        [_off("local_authority", "N-x", None), _off("local_authority", "N-y", 5.0)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "N")},
    )
    assert rows == []


def test_an_absent_cell_inside_the_tables_rounding_adds_no_row():
    bound = [*_both_tiers("A", 100.0, 100.0), *_both_tiers("C", 200.0, 201.0)]
    rows, receipt = uk_off_surface_frame_rows(
        bound,
        [_off("local_authority", "C-x", None)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "C")},
    )
    assert rows == []
    (entry,) = receipt["implied_absent"]
    assert entry["status"] == "not_positive"
    assert entry["implied_value"] == pytest.approx(-1.0)


def test_an_absent_cell_no_tier_can_imply_is_refused_where_a_control_moves_it():
    # One local tier only: nothing says what the absent authority holds.
    bound = [_bound("la", "A-la", 100.0), _bound("la", "C-la", 150.0)]
    with pytest.raises(ValueError, match="cannot be implied.*C-x"):
        uk_off_surface_frame_rows(
            bound,
            [_off("local_authority", "C-x", None)],
            leg_of_area=_leg_of_area,
            controlled_legs={(_TARGET, "la", "C")},
        )


def test_rule_deferred_rows_already_in_the_frame_count_in_their_leg():
    # Leg A's local-authority tier is whole only with its rule-deferred cell.
    bound = [*_both_tiers("A", 100.0, 90.0), *_both_tiers("C", 200.0, 150.0)]
    rows, _ = uk_off_surface_frame_rows(
        bound,
        [_off("local_authority", "C-x", None)],
        leg_of_area=_leg_of_area,
        controlled_legs={(_TARGET, "la", "C")},
        frame_only_rows=[
            {"target_id": _TARGET, "grain": "la", "geography_id": "A-s", "value": 10.0}
        ],
    )
    (row,) = rows
    assert row["value"] == pytest.approx(50.0)


def test_the_membership_cells_off_the_surface_leave_out_the_rules_own():
    membership = {
        "targets": {
            _TARGET: {
                "geography_levels": {
                    "local_authority": {
                        "candidates": [
                            {"geography_id": "E1", "status": "active"},
                            {
                                "geography_id": "E2",
                                "status": "signed_deferred",
                                "signed_reason_id": "support_floor",
                                "deferred_value": 9000.0,
                            },
                            {
                                "geography_id": "E3",
                                "status": "no_fact_for_area",
                                "signed_reason_id": "coverage_absent",
                            },
                            {
                                "geography_id": "E4",
                                "status": "signed_deferred",
                                "signed_reason_id": "a_small_cell_rule",
                                "deferred_value": 3.0,
                            },
                        ]
                    }
                }
            }
        }
    }
    cells = uk_signed_cells_off_the_surface(membership, rule_ids={"a_small_cell_rule"})
    assert [(c["geography_id"], c["value"], c["reason_id"]) for c in cells] == [
        ("E2", 9000.0, "support_floor"),
        ("E3", None, "coverage_absent"),
    ]
    # The membership binds no cell for either area at this level.
    assert not any(cell["area_has_bound_cells"] for cell in cells)


def test_every_committed_off_surface_area_is_one_a_whole_run_binds():
    # The ladder rule leans on this: a run whose ladder lacks such an area is
    # refused on the area's bound cells before any off-surface cell is read.
    cells = uk_off_surface_cells(load_uk_local_target_reference_membership())
    assert cells
    assert all(cell["area_has_bound_cells"] for cell in cells)


def test_the_committed_membership_signs_torbay_off_the_hmrc_surface():
    cells = uk_off_surface_cells(load_uk_local_target_reference_membership())
    torbay = [
        cell
        for cell in cells
        if cell["target_id"] == _TARGET and cell["geography_id"] == "E06000027"
    ]
    assert [(c["geography_level"], c["value"]) for c in torbay] == [
        ("local_authority", None)
    ]


def _uc(name: str, level: str, area: str, value: float) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="household",
        value=value,
        measure="uc_households",
        period=2025,
        source="DWP",
        family="uc_households",
        metadata={
            "contract_target_id": name.split("@", 1)[0],
            "geography_level": level,
            "geography_id": area,
        },
    )


@pytest.fixture
def unbounded_uc_bridge(monkeypatch):
    """The UK cross-grain rule with the bridges' factor bounds lifted.

    The tests that take it put a GB caseload of 300 over 200 households, to
    show where an off-surface cell's share goes at factors far from one. The
    UC bridge refuses a factor more than 2% from one (microcosm#1300), as the
    ``test_the_uc_bridge_bound_*`` tests show on the real rule; these are
    about the shares, so they lift it.
    """

    from dataclasses import replace

    from microcosm.build.uk_runtime import ledger_targets

    rule = ledger_targets.UK_CROSS_GRAIN_RULE
    monkeypatch.setattr(
        ledger_targets,
        "UK_CROSS_GRAIN_RULE",
        replace(
            rule,
            bridges=tuple(
                replace(bridge, max_factor_shift=None) for bridge in rule.bridges
            ),
        ),
    )


def _uc_surface(
    off_surface_cells=(), area_region_codes=None, deferred_cells=(), *, gb=300.0
) -> tuple[dict[str, float], dict]:
    # A GB caseload (300 unless given) over two constituencies of 100
    # (Scotland, Wales). Wales has its one authority; Scotland's tier carries
    # one of two, at 60.
    registry = TargetRegistry(
        [
            _uc("dwp.uc.households", "country", "K03000001", gb),
            _uc(
                "dwp.uc.households_by_area@S14000001",
                "constituency",
                "S14000001",
                100.0,
            ),
            _uc(
                "dwp.uc.households_by_area@W07000081",
                "constituency",
                "W07000081",
                100.0,
            ),
            _uc(
                "dwp.uc.households_by_area@S12000033",
                "local_authority",
                "S12000033",
                60.0,
            ),
            _uc(
                "dwp.uc.households_by_area@W06000001",
                "local_authority",
                "W06000001",
                100.0,
            ),
        ],
        country="uk",
    )
    surface, receipt = uk_local_target_surface(
        registry,
        bound_national_target_ids=("dwp.uc.households",),
        period=2025,
        deferred_cells=deferred_cells,
        off_surface_cells=off_surface_cells,
        area_region_codes=area_region_codes,
    )
    return dict(zip(surface["area_code"], surface["value"], strict=True)), receipt


def test_the_uc_bridge_bound_refuses_a_caseload_far_from_its_cells():
    # A GB caseload of 300 over 200 constituency households is not one
    # quantity at a factor of 1.5: the bridge refuses (microcosm#1300).
    with pytest.raises(ValueError, match=r"the constituency rows of leg .* by 1\.5 "):
        _uc_surface()


def test_the_uc_bridge_bound_refuses_authorities_whose_gb_total_is_short():
    # The caseload matches the constituencies, but Scotland's authority tier
    # carries 60 of its 100: the authorities' GB total is 160 of 200. An
    # exact signature group rescales them onto the constituencies first, so
    # the bound reads the GB total as given (microcosm#1302 review); it bounds
    # each grain's GB total, not each nation within it.
    with pytest.raises(ValueError, match=r"the la rows of leg .* by 1\.25 "):
        _uc_surface(gb=200.0)


def test_the_uc_bridge_bound_counts_off_surface_cells_on_the_last_pass():
    # With Scotland's other authority off the surface at 40, the authorities
    # sum to the caseload. The first pass, without it, reads 160 of 200; the
    # bound is checked on the pass that carries it, so the real rule accepts
    # the surface and the bound authority keeps its 60.
    target = "dwp.uc.households_by_area"
    values, receipt = _uc_surface(
        [_off("local_authority", "S12000034", 40.0, target_id=target)], gb=200.0
    )
    assert values["S12000033"] == pytest.approx(60.0)
    assert values["W06000001"] == pytest.approx(100.0)
    assert receipt["off_surface_cells"]["rows_added"] == 1


_UC_AREA_TARGET = "dwp.uc.households_by_area"


@pytest.mark.parametrize(
    ("off_surface", "gb", "message"),
    [
        # Another target's cell moves no UC leg, so the pass that carries
        # the off-surface cells adds no row; it still checks the bound.
        (
            _off("local_authority", "E06000001", 5.0),
            300.0,
            r"the constituency rows of leg .* by 1\.5 ",
        ),
        # Scotland's other authority joins the frame at 40, and the GB
        # caseload is still 1.5 times the constituencies.
        (
            _off("local_authority", "S12000034", 40.0, target_id=_UC_AREA_TARGET),
            300.0,
            r"the constituency rows of leg .* by 1\.5 ",
        ),
        # At 10 the constituencies match the caseload, but the authorities
        # sum to 170 of 200 with the off-surface cell counted.
        (
            _off("local_authority", "S12000034", 10.0, target_id=_UC_AREA_TARGET),
            200.0,
            r"the la rows of leg .* by 1\.17",
        ),
    ],
    ids=["unrelated_cell", "uc_cell_far_caseload", "uc_cell_short_authorities"],
)
def test_the_uc_bridge_bound_refuses_on_the_pass_that_carries_off_surface_cells(
    off_surface, gb, message
):
    # Every committed surface has off-surface cells, so the first pass only
    # records the bridge's factor and the pass that carries them is the one
    # that refuses (microcosm#1302 re-review). That pass runs whether or not
    # an off-surface cell adds a row to the frame.
    with pytest.raises(ValueError, match=message):
        _uc_surface([off_surface], gb=gb)


def test_the_surface_keeps_an_off_surface_cells_share_off_its_neighbours(
    unbounded_uc_bridge,
):
    target = "dwp.uc.households_by_area"
    # With nothing declared, Scotland's one bound authority takes the whole leg.
    values, receipt = _uc_surface()
    assert values["S12000033"] == pytest.approx(150.0)
    assert receipt["off_surface_cells"]["rows_added"] == 0

    # Deferred with its value of 40: the bound authority keeps its 60%.
    values, receipt = _uc_surface(
        [_off("local_authority", "S12000034", 40.0, target_id=target)]
    )
    assert values["S12000033"] == pytest.approx(90.0)
    assert values["W06000001"] == pytest.approx(150.0)
    assert "S12000034" not in values
    assert receipt["off_surface_cells"]["rows_added"] == 1

    # Absent from the table: implied from the constituency tier (Wales is the
    # whole leg, at a ratio of one), so it holds the same 40.
    values, receipt = _uc_surface(
        [_off("local_authority", "S12000034", None, target_id=target)]
    )
    assert values["S12000033"] == pytest.approx(90.0)
    (entry,) = receipt["off_surface_cells"]["implied_absent"]
    assert entry["implied_value"] == pytest.approx(40.0)
    assert entry["leg"] == "S92000003"


def test_an_off_surface_area_the_runs_ladder_does_not_carry_is_left_out(
    unbounded_uc_bridge,
):
    target = "dwp.uc.households_by_area"
    cells = [
        _off("local_authority", "S12000034", 40.0, target_id=target),
        _off("local_authority", "E06999999", 5.0, target_id=target),
    ]
    # A run with its own ladder carries only its areas: the English authority
    # is not in it, so it is named in the receipt and left out.
    values, receipt = _uc_surface(cells, area_region_codes={"E06000001": "E12000001"})
    assert values["S12000033"] == pytest.approx(90.0)
    assert receipt["off_surface_cells"]["areas_outside_the_ladder"] == ["E06999999"]
    assert receipt["off_surface_cells"]["rows_added"] == 1
    # Under the committed crosswalk an area that cannot be placed is a defect.
    with pytest.raises(ValueError, match="E06999999.*local-area crosswalk"):
        _uc_surface(cells)
    # An area the membership binds nothing for has no bound cell to refuse a
    # ladder that lacks it, so the off-surface cell itself refuses.
    unguarded = [
        _off(
            "local_authority",
            "E06999999",
            5.0,
            target_id=target,
            area_has_bound_cells=False,
        )
    ]
    with pytest.raises(ValueError, match="E06999999.*run's ladder membership"):
        _uc_surface(unguarded, area_region_codes={"E06000001": "E12000001"})


def test_a_small_cell_deferral_the_runs_ladder_does_not_carry_is_left_out(
    unbounded_uc_bridge,
):
    # A small-cell deferral always joins the frame. Under a run's own ladder
    # it is placed as an off-surface cell is (microcosm#1278): the run's
    # ladder lacks the English authority, so it is left out and named.
    target = "dwp.uc.households_by_area"
    deferrals = [
        _off("local_authority", "S12000034", 40.0, target_id=target),
        _off("local_authority", "E06999999", 5.0, target_id=target),
    ]
    values, receipt = _uc_surface(
        deferred_cells=deferrals, area_region_codes={"E06000001": "E12000001"}
    )
    assert values["S12000033"] == pytest.approx(90.0)
    assert receipt["small_cell_deferrals"] == {
        "rows_added": 1,
        "areas_outside_the_ladder": ["E06999999"],
    }
    # Under the committed crosswalk an area that cannot be placed is a defect.
    with pytest.raises(ValueError, match="E06999999.*local-area crosswalk"):
        _uc_surface(deferred_cells=deferrals)
    # An area the membership binds nothing for refuses under a run's ladder.
    unguarded = [
        _off(
            "local_authority",
            "E06999999",
            5.0,
            target_id=target,
            area_has_bound_cells=False,
        )
    ]
    with pytest.raises(ValueError, match="E06999999.*run's ladder membership"):
        _uc_surface(
            deferred_cells=unguarded, area_region_codes={"E06000001": "E12000001"}
        )


def test_the_uc_bridge_bound_counts_a_small_cell_deferral():
    # The tests above lift the bound; this one keeps the real rule. Scotland's
    # other authority is a small-cell deferral at 40, which always joins the
    # frame, so the authorities' GB total meets the caseload of 200 and the
    # surface is accepted: the bound authority keeps its 60.
    deferrals = [_off("local_authority", "S12000034", 40.0, target_id=_UC_AREA_TARGET)]
    values, receipt = _uc_surface(deferred_cells=deferrals, gb=200.0)
    assert values["S12000033"] == pytest.approx(60.0)
    assert values["W06000001"] == pytest.approx(100.0)
    assert "S12000034" not in values
    assert receipt["small_cell_deferrals"] == {
        "rows_added": 1,
        "areas_outside_the_ladder": [],
    }


def test_the_uc_bridge_bound_refuses_authorities_short_with_a_small_cell_deferral():
    # At 10 the deferral leaves the authorities' GB total at 170 of 200.
    deferrals = [_off("local_authority", "S12000034", 10.0, target_id=_UC_AREA_TARGET)]
    with pytest.raises(ValueError, match=r"the la rows of leg .* by 1\.17"):
        _uc_surface(deferred_cells=deferrals, gb=200.0)
