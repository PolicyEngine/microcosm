"""Signed cells that are off the surface still count in their leg (#1123)."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.ledger_targets import (
    load_uk_local_target_reference_membership,
    uk_local_target_surface,
    uk_off_surface_cells,
)
from microcosm.build.uk_runtime.off_surface_cells import (
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


def _off(level: str, area: str, value: float | None, target_id: str = _TARGET):
    return {
        "target_id": target_id,
        "geography_level": level,
        "geography_id": area,
        "value": value,
        "reason_id": "test",
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


def _uc_surface(off_surface_cells=()) -> tuple[dict[str, float], dict]:
    # A GB caseload of 300 over two constituencies of 100 (Scotland, Wales).
    # Wales has its one authority; Scotland's tier carries one of two.
    registry = TargetRegistry(
        [
            _uc("dwp.uc.households", "country", "K03000001", 300.0),
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
        off_surface_cells=off_surface_cells,
    )
    return dict(zip(surface["area_code"], surface["value"], strict=True)), receipt


def test_the_surface_keeps_an_off_surface_cells_share_off_its_neighbours():
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
