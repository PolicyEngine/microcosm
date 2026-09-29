"""Examples for carrying a 2010-block plan onto 2020 blocks.

North Carolina's 117th-Congress plan exists only on 2010 blocks, so the
registry build carries it through the Census 2010-2020 block relationship
file. Properties of the same functions are in
``test_us_cd_plan_registry_properties.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.us_runtime.cd_plan_registry import (
    BlockRelationship,
    CdBlockAssignment,
    assignment_agreement,
    cd_block_assignment,
    crosswalk_plan_to_2020_blocks,
    fill_unassigned_blocks,
    parse_block_relationship_2010_2020,
    parse_cd_block_assignment,
)

HEADER = (
    "STATE_2010|COUNTY_2010|TRACT_2010|BLK_2010|BLKSF_2010|AREALAND_2010|"
    "AREAWATER_2010|BLOCK_PART_FLAG_O|STATE_2020|COUNTY_2020|TRACT_2020|BLK_2020|"
    "BLKSF_2020|AREALAND_2020|AREAWATER_2020|BLOCK_PART_FLAG_R|AREALAND_INT|"
    "AREAWATER_INT"
)


def _row(block_2010: str, block_2020: str, land: int, water: int = 0) -> str:
    def split(geoid: str) -> str:
        return f"{geoid[:2]}|{geoid[2:5]}|{geoid[5:11]}|{geoid[11:]}"

    return f"{split(block_2010)}||0|0|p|{split(block_2020)}||0|0|p|{land}|{water}"


def test_relationship_parser_builds_geoids_and_reads_intersection_areas():
    relationship = parse_block_relationship_2010_2020(
        [
            "﻿" + HEADER + "\r\n",
            _row("370010201001000", "370010201001000", 29235) + "\r\n",
            _row("131110501001001", "370399306042005", 783, 5) + "\r\n",
        ],
        label="rel",
    )
    assert relationship.block_2010.tolist() == [370010201001000, 131110501001001]
    assert relationship.block_2020.tolist() == [370010201001000, 370399306042005]
    assert relationship.area_land.tolist() == [29235, 783]
    assert relationship.area_water.tolist() == [0, 5]


def test_relationship_parser_refuses_a_missing_column():
    with pytest.raises(ValueError, match="missing columns"):
        parse_block_relationship_2010_2020(
            [HEADER.replace("AREALAND_INT", "AREALAND")], label="rel"
        )


def test_parser_reads_a_legislature_plan_block_file():
    plan = parse_cd_block_assignment(
        ['"Block","District"', '"371139706001037","11"', '"370010201001000","1"'],
        label="C-Goodwin-A-1-TC.csv",
    )
    assert plan.block_geoid.tolist() == [370010201001000, 371139706001037]
    assert plan.district_geoid.tolist() == [3701, 3711]
    keyed = parse_cd_block_assignment(
        ['"Block_Key","District"', '"370010205012023","06"'], label="2016 plan"
    )
    assert keyed.district_geoid.tolist() == [3706]


# 2010 blocks A, B (district 1) and C (district 2); 2020 blocks X, Y, Z, W.
A, B, C, OUT = 370010001001000, 370010001001001, 370010001002000, 450010001001000
X, Y, Z, W = 370010001001000, 370010001001001, 370010001001002, 370010001002001
PLAN_2010 = cd_block_assignment({A: 3701, B: 3701, C: 3702}, label="2010 plan")


def _relationship(rows):
    return BlockRelationship(
        block_2010=np.asarray([r[0] for r in rows], dtype=np.int64),
        block_2020=np.asarray([r[1] for r in rows], dtype=np.int64),
        area_land=np.asarray([r[2] for r in rows], dtype=np.int64),
        area_water=np.asarray([r[3] for r in rows], dtype=np.int64),
    )


def test_crosswalk_takes_the_largest_land_area_summed_by_district():
    relationship = _relationship(
        [
            (A, X, 100, 0),  # X wholly in district 1
            (A, Y, 30, 0),  # Y: district 1 has 30 + 30 = 60 of land...
            (B, Y, 30, 0),
            (C, Y, 50, 0),  # ...beating district 2's single largest piece
            (C, Z, 10, 0),  # Z: land tie, broken by water
            (A, Z, 10, 5),
            (OUT, W, 99, 0),  # W's only counterpart is in another state
        ]
    )
    carried, diagnostics = crosswalk_plan_to_2020_blocks(
        PLAN_2010, relationship, state_fips="37"
    )
    assert dict(
        zip(carried.block_geoid.tolist(), carried.district_geoid.tolist(), strict=True)
    ) == {X: 3701, Y: 3701, Z: 3701}
    assert diagnostics["straddling"].tolist() == [Y, Z]
    assert diagnostics["unassigned"].tolist() == [W]
    assert diagnostics["blocks_2020"].tolist() == [X, Y, Z, W]


def test_crosswalk_breaks_a_full_tie_by_the_lower_district():
    relationship = _relationship([(C, X, 10, 0), (A, X, 10, 0)])
    carried, _ = crosswalk_plan_to_2020_blocks(PLAN_2010, relationship, state_fips="37")
    assert carried.district_geoid.tolist() == [3701]


def test_crosswalk_refuses_a_state_the_file_does_not_cover():
    relationship = _relationship([(A, X, 1, 0)])
    with pytest.raises(ValueError, match="no 2020 blocks in state 45"):
        crosswalk_plan_to_2020_blocks(PLAN_2010, relationship, state_fips="45")


def test_fill_uses_the_smallest_neighbourhood_with_assigned_blocks():
    # Block group 3700100010010 holds X (district 1, 5 people) and Y (district
    # 2, 50 people); W sits alone in block group ...0020 of the same tract.
    assigned = cd_block_assignment({X: 3701, Y: 3702}, label="assigned")
    same_group = 370010001001009
    filled, fills = fill_unassigned_blocks(
        assigned,
        np.asarray([W, same_group]),
        block_geoid=np.asarray([X, Y, W]),
        population=np.asarray([5, 50, 7]),
    )
    placed = dict(
        zip(filled.block_geoid.tolist(), filled.district_geoid.tolist(), strict=True)
    )
    assert placed[same_group] == 3702  # block group majority by population
    assert placed[W] == 3702  # no assigned block in W's group: tract majority
    assert fills == [
        {
            "block": f"{same_group:015d}",
            "district": "02",
            "rule": "block_group",
            "population": 0,
        },
        {"block": f"{W:015d}", "district": "02", "rule": "tract", "population": 7},
    ]


def test_fill_refuses_a_block_with_no_assigned_county_neighbour():
    assigned = cd_block_assignment({X: 3701}, label="assigned")
    with pytest.raises(ValueError, match="shares a county"):
        fill_unassigned_blocks(
            assigned,
            np.asarray([370030001001000]),
            block_geoid=np.asarray([X]),
            population=np.asarray([1]),
        )


def test_agreement_counts_blocks_and_people_and_treats_gaps_as_disagreement():
    left = cd_block_assignment({X: 3701, Y: 3701, Z: 3702}, label="left")
    right = CdBlockAssignment(np.asarray([X, Y]), np.asarray([3701, 3702]))
    agreement = assignment_agreement(
        left,
        right,
        block_geoid=np.asarray([X, Y, Z]),
        population=np.asarray([10, 20, 30]),
    )
    assert agreement == {
        "blocks": 3,
        "blocks_agreeing": 1,
        "population": 60,
        "population_agreeing": 10,
        "population_share_agreeing": 10 / 60,
    }


def test_crosswalk_with_no_assigned_counterpart_leaves_every_block_out():
    # Minimized counterexample: the only 2010 counterpart is outside the plan.
    relationship = _relationship([(OUT, X, 5, 0)])
    carried, diagnostics = crosswalk_plan_to_2020_blocks(
        PLAN_2010, relationship, state_fips="37"
    )
    assert len(carried) == 0
    assert diagnostics["unassigned"].tolist() == [X]
    assert diagnostics["straddling"].tolist() == []
