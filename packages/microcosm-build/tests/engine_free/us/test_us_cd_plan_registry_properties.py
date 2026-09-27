"""Property-based tests for the US block -> congressional-district plan registry.

Each test draws a random multi-state geography (blocks, populations, and one
district assignment per plan that respects the real House apportionment of
each drawn state) and checks an invariant for every such input:

- conservation: for every plan, district populations sum exactly to state
  populations, which sum to the national total;
- partition: every block has exactly one district per plan, in its own state,
  and each state has exactly its apportioned number of districts;
- determinism: the artifact's bytes depend only on the inputs' contents, not
  their order;
- fail-closed: dropping, crossing or merging any district assignment is
  refused at build time, and the ladder differential refuses any change;
- crosswalk: carrying a 2010-block plan onto 2020 blocks gives each 2020
  block the district with the most intersecting land (then water, then the
  lower code), leaves out exactly the blocks with no assigned counterpart,
  and does not depend on the relationship file's row order.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.cd_plan_registry import (  # noqa: E402
    CD_PLAN_ARRAY_PREFIX,
    PRIMARY_CD_PLAN,
    BlockRelationship,
    assemble_us_cd_plan_registry,
    cd_block_assignment,
    check_cd_plan_registry_against_block_ladder,
    crosswalk_plan_to_2020_blocks,
    expected_district_counts,
    fill_unassigned_blocks,
    load_us_cd_plan_registry,
    replace_state_assignments,
    summarize_cd_plan_registry,
)

#: Plans under test and the apportionment each must satisfy.
PLANS = {
    "117th_congress": "2010_census",
    "119th_congress": "2020_census",
    "120th_congress": "2020_census",
}
#: Small states (few districts) so drawn geographies stay small. Montana and
#: West Virginia change seat counts between 2010 and 2020.
SMALL_STATES = ("10", "11", "15", "16", "23", "30", "31", "33", "35", "44", "54")
MAX_DISTRICTS = max(
    expected_district_counts(apportionment)[state]
    for apportionment in PLANS.values()
    for state in SMALL_STATES
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _spec(plan: str) -> dict:
    url = f"https://example.test/{plan}.zip"
    return {
        "vintage": plan,
        "source": f"test source for {plan}",
        "url": url,
        "apportionment": PLANS[plan],
        "source_files": {plan: {"url": url, "sha256": "0" * 64}},
    }


@st.composite
def geographies(draw):
    """Blocks, populations and a valid district map per plan."""

    states = draw(
        st.lists(st.sampled_from(SMALL_STATES), min_size=1, max_size=4, unique=True)
    )
    blocks: list[int] = []
    population: list[int] = []
    plans: dict[str, dict[int, int]] = {plan: {} for plan in PLANS}
    for state in sorted(states):
        suffixes = draw(
            st.lists(
                st.integers(min_value=0, max_value=10**13 - 1),
                min_size=MAX_DISTRICTS,
                max_size=MAX_DISTRICTS + 25,
                unique=True,
            )
        )
        state_blocks = [int(state) * 10**13 + suffix for suffix in suffixes]
        blocks.extend(state_blocks)
        population.extend(
            draw(
                st.lists(
                    st.integers(1, 5_000),
                    min_size=len(state_blocks),
                    max_size=len(state_blocks),
                )
            )
        )
        for plan, apportionment in PLANS.items():
            count = expected_district_counts(apportionment)[state]
            codes = [0] if count == 1 else list(range(1, count + 1))
            order = draw(st.permutations(state_blocks))
            # The first ``count`` blocks seed one district each, so no
            # apportioned district is empty; the rest land anywhere.
            for position, block in enumerate(order):
                code = (
                    codes[position]
                    if position < count
                    else draw(st.sampled_from(codes))
                )
                plans[plan][block] = int(state) * 100 + code
    return (
        np.asarray(blocks, dtype=np.int64),
        np.asarray(population, dtype=np.int64),
        plans,
    )


def _assemble(blocks, population, plans, **overrides):
    kwargs = {
        "block_geoid": blocks,
        "population": population,
        "plan_assignments": {
            plan: cd_block_assignment(mapping, label=plan)
            for plan, mapping in plans.items()
        },
        "plan_sources": {plan: _spec(plan) for plan in plans},
    }
    kwargs.update(overrides)
    return assemble_us_cd_plan_registry(**kwargs)


def _npz_bytes(payload) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def _load(tmp_path: Path, payload, name: str):
    path = tmp_path / f"{name}.npz"
    path.write_bytes(_npz_bytes(payload))
    return load_us_cd_plan_registry(path)


@SETTINGS
@given(geography=geographies())
def test_every_plan_partitions_and_conserves_state_population(
    tmp_path_factory, geography
):
    blocks, population, plans = geography
    registry = _load(
        tmp_path_factory.mktemp("r"), _assemble(blocks, population, plans), "r"
    )

    state_population: dict[str, int] = {}
    for block, people in zip(blocks.tolist(), population.tolist(), strict=True):
        state_population[f"{block // 10**13:02d}"] = (
            state_population.get(f"{block // 10**13:02d}", 0) + people
        )
    assert registry.state_population() == state_population
    assert int(registry.population.sum()) == int(population.sum())

    for plan, mapping in plans.items():
        # Exactly one district per block, the one the source gave it.
        assert registry.plans[plan].tolist() == [
            mapping[b] for b in registry.block_geoid.tolist()
        ]
        # Every district lies in its block's state.
        assert (registry.plans[plan] // 100 == registry.block_geoid // 10**13).all()
        districts = registry.district_population(plan)
        by_state: dict[str, int] = {}
        count_by_state: dict[str, int] = {}
        for district, people in districts.items():
            by_state[district[:2]] = by_state.get(district[:2], 0) + people
            count_by_state[district[:2]] = count_by_state.get(district[:2], 0) + 1
        assert by_state == state_population
        expected = expected_district_counts(PLANS[plan])
        assert count_by_state == {state: expected[state] for state in state_population}
        assert all(people > 0 for people in districts.values())

    summary = summarize_cd_plan_registry(registry)
    assert summary["population"] == int(population.sum())
    assert summary["blocks"] == len(blocks)


@SETTINGS
@given(geography=geographies(), seed=st.integers(0, 2**32 - 1))
def test_artifact_bytes_do_not_depend_on_input_order(geography, seed):
    blocks, population, plans = geography
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    shuffled_plans = {}
    for plan, mapping in plans.items():
        items = list(mapping.items())
        shuffled_plans[plan] = dict(items[i] for i in rng.permutation(len(items)))

    original = _npz_bytes(_assemble(blocks, population, plans))
    shuffled = _npz_bytes(_assemble(blocks[order], population[order], shuffled_plans))
    assert hashlib.sha256(original).hexdigest() == hashlib.sha256(shuffled).hexdigest()
    # And building twice from the same inputs gives the same bytes.
    assert _npz_bytes(_assemble(blocks, population, plans)) == original


@SETTINGS
@given(geography=geographies(), data=st.data())
def test_a_block_without_a_district_is_refused(geography, data):
    blocks, population, plans = geography
    plan = data.draw(st.sampled_from(sorted(plans)))
    dropped = data.draw(st.sampled_from(blocks.tolist()))
    broken = {**plans, plan: {b: d for b, d in plans[plan].items() if b != dropped}}
    with pytest.raises(ValueError, match="unassigned"):
        _assemble(blocks, population, broken)


@SETTINGS
@given(geography=geographies(), data=st.data())
def test_a_district_in_another_state_is_refused(geography, data):
    blocks, population, plans = geography
    plan = data.draw(st.sampled_from(sorted(plans)))
    block = data.draw(st.sampled_from(blocks.tolist()))
    other_state = data.draw(
        st.sampled_from([s for s in SMALL_STATES if int(s) != block // 10**13])
    )
    broken = {**plans, plan: {**plans[plan], block: int(other_state) * 100}}
    with pytest.raises(ValueError, match="another state's district"):
        _assemble(blocks, population, broken)


@SETTINGS
@given(geography=geographies(), data=st.data())
def test_a_merged_or_invented_district_is_refused(geography, data):
    blocks, population, plans = geography
    plan = data.draw(st.sampled_from(sorted(plans)))
    mapping = plans[plan]
    districts = sorted(set(mapping.values()))
    multi = [
        d
        for d in districts
        if sum(1 for other in districts if other // 100 == d // 100) > 1
    ]
    if multi and data.draw(st.booleans()):
        # Merge one district into a neighbour: one district too few.
        gone = data.draw(st.sampled_from(multi))
        keep = next(d for d in districts if d // 100 == gone // 100 and d != gone)
        broken_plan = {b: (keep if d == gone else d) for b, d in mapping.items()}
    else:
        # Move one block into a district number the state was not given.
        block = data.draw(st.sampled_from(blocks.tolist()))
        broken_plan = {**mapping, block: (block // 10**13) * 100 + 99}
    with pytest.raises(ValueError, match="apportionment"):
        _assemble(blocks, population, {**plans, plan: broken_plan})


@SETTINGS
@given(geography=geographies(), data=st.data())
def test_ladder_differential_refuses_any_single_change(
    tmp_path_factory, geography, data
):
    blocks, population, plans = geography
    registry = _load(
        tmp_path_factory.mktemp("d"), _assemble(blocks, population, plans), "d"
    )
    ladder_cd = np.asarray(
        [plans[PRIMARY_CD_PLAN][b] for b in blocks.tolist()], dtype=np.int64
    )
    receipt = check_cd_plan_registry_against_block_ladder(
        registry,
        block_geoid=blocks,
        population=population,
        congressional_district_geoid=ladder_cd,
    )
    assert receipt["agreement"] == "exact"
    assert receipt["population"] == int(population.sum())

    index = data.draw(st.integers(0, len(blocks) - 1))
    change = data.draw(st.sampled_from(["district", "population", "drop"]))
    changed_blocks, changed_population, changed_cd = (
        blocks,
        population.copy(),
        ladder_cd.copy(),
    )
    if change == "district":
        changed_cd[index] += 1  # any other district code
    elif change == "population":
        changed_population[index] += 1
    else:
        keep = np.arange(len(blocks)) != index
        changed_blocks, changed_population, changed_cd = (
            blocks[keep],
            population[keep],
            ladder_cd[keep],
        )
    with pytest.raises(ValueError):
        check_cd_plan_registry_against_block_ladder(
            registry,
            block_geoid=changed_blocks,
            population=changed_population,
            congressional_district_geoid=changed_cd,
        )


@SETTINGS
@given(geography=geographies(), data=st.data())
def test_state_substitution_changes_exactly_that_state(
    tmp_path_factory, geography, data
):
    blocks, population, plans = geography
    state = data.draw(
        st.sampled_from(sorted({f"{b // 10**13:02d}" for b in blocks.tolist()}))
    )
    base = cd_block_assignment(plans["120th_congress"], label="120")
    donor = cd_block_assignment(plans["119th_congress"], label="119").for_state(state)
    merged = replace_state_assignments(base, donor, state_fips=state)
    merged_map = dict(
        zip(merged.block_geoid.tolist(), merged.district_geoid.tolist(), strict=True)
    )
    for block, district in plans["120th_congress"].items():
        in_state = f"{block // 10**13:02d}" == state
        expected = plans["119th_congress"][block] if in_state else district
        assert merged_map[block] == expected

    # Both plans use the 2020 apportionment, so the result is a valid plan.
    payload = _assemble(
        blocks,
        population,
        {"120th_congress": merged_map, "119th_congress": plans["119th_congress"]},
    )
    registry = _load(tmp_path_factory.mktemp("s"), payload, "s")
    summary = summarize_cd_plan_registry(registry)
    differing = summary["plans"]["120th_congress"]["states_differing_from_primary"]
    assert state not in differing
    assert payload[f"{CD_PLAN_ARRAY_PREFIX}120th_congress"].dtype == np.int16


@st.composite
def relationships(draw):
    """A 2010 plan in one state and a relationship file to 2020 blocks."""

    state = 37
    blocks_2010 = draw(
        st.lists(st.integers(0, 10**6), min_size=1, max_size=30, unique=True)
    )
    blocks_2010 = [state * 10**13 + b for b in blocks_2010]
    plan = {b: state * 100 + draw(st.integers(1, 4)) for b in blocks_2010}
    # Some 2010 counterparts sit outside the plan (another state's blocks).
    outside = [
        45 * 10**13 + b
        for b in draw(st.lists(st.integers(0, 10**6), max_size=5, unique=True))
    ]
    blocks_2020 = [
        state * 10**13 + b
        for b in draw(
            st.lists(st.integers(0, 10**6), min_size=1, max_size=30, unique=True)
        )
    ]
    rows = []
    for block_2020 in blocks_2020:
        for block_2010 in draw(
            st.lists(
                st.sampled_from(blocks_2010 + outside),
                min_size=1,
                max_size=4,
                unique=True,
            )
        ):
            rows.append(
                (
                    block_2010,
                    block_2020,
                    draw(st.integers(0, 50)),
                    draw(st.integers(0, 5)),
                )
            )
    return plan, rows


def _relationship(rows):
    return BlockRelationship(
        block_2010=np.asarray([r[0] for r in rows], dtype=np.int64),
        block_2020=np.asarray([r[1] for r in rows], dtype=np.int64),
        area_land=np.asarray([r[2] for r in rows], dtype=np.int64),
        area_water=np.asarray([r[3] for r in rows], dtype=np.int64),
    )


@SETTINGS
@given(case=relationships(), seed=st.integers(0, 2**32 - 1))
def test_crosswalk_picks_the_largest_overlap_whatever_the_row_order(case, seed):
    plan, rows = case
    plan_2010 = cd_block_assignment(plan, label="2010 plan")
    carried, diagnostics = crosswalk_plan_to_2020_blocks(
        plan_2010, _relationship(rows), state_fips="37"
    )

    # Reference implementation: sum (land, water) per candidate district.
    expected: dict[int, int] = {}
    candidates: dict[int, dict[int, list[int]]] = {}
    for block_2010, block_2020, land, water in rows:
        if block_2010 not in plan:
            continue
        area = candidates.setdefault(block_2020, {}).setdefault(
            plan[block_2010], [0, 0]
        )
        area[0] += land
        area[1] += water
    for block_2020, by_district in candidates.items():
        expected[block_2020] = min(
            by_district, key=lambda d: (-by_district[d][0], -by_district[d][1], d)
        )
    got = dict(
        zip(carried.block_geoid.tolist(), carried.district_geoid.tolist(), strict=True)
    )
    assert got == expected
    all_2020 = {r[1] for r in rows}
    assert set(diagnostics["unassigned"].tolist()) == all_2020 - set(expected)
    assert set(diagnostics["straddling"].tolist()) == {
        b for b, by_district in candidates.items() if len(by_district) > 1
    }

    shuffled = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
    again, _ = crosswalk_plan_to_2020_blocks(
        plan_2010, _relationship(shuffled), state_fips="37"
    )
    assert np.array_equal(again.block_geoid, carried.block_geoid)
    assert np.array_equal(again.district_geoid, carried.district_geoid)


@SETTINGS
@given(case=relationships())
def test_fill_completes_every_block_with_a_district_already_nearby(case):
    plan, rows = case
    carried, diagnostics = crosswalk_plan_to_2020_blocks(
        cd_block_assignment(plan, label="2010 plan"),
        _relationship(rows),
        state_fips="37",
    )
    if len(carried) == 0:
        return  # nothing to borrow from; the build refuses this case
    filled, fills = fill_unassigned_blocks(
        carried,
        diagnostics["unassigned"],
        block_geoid=carried.block_geoid,
        population=np.ones(len(carried), dtype=np.int64),
    )
    assert set(filled.block_geoid.tolist()) == set(diagnostics["blocks_2020"].tolist())
    assert len(fills) == len(diagnostics["unassigned"])
    present = set(carried.district_geoid.tolist())
    for fill in fills:
        assert 3700 + int(fill["district"]) in present
    # Blocks the crosswalk assigned keep their district.
    kept = dict(
        zip(filled.block_geoid.tolist(), filled.district_geoid.tolist(), strict=True)
    )
    for block, district in zip(
        carried.block_geoid.tolist(), carried.district_geoid.tolist(), strict=True
    ):
        assert kept[block] == district
