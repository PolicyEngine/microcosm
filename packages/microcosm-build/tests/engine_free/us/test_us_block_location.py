"""Invariants of the US block-location draw (location v1, microcosm#696).

Each property holds for every input Hypothesis generates over small nested
synthetic ladders (states > counties > tracts > blocks; PUMAs are unions of
whole tracts; districts cut across tracts):

1. Consistency: every derived geography (tract, county, state, place, SLDU,
   SLDL, CBSA, PUMA, each attached CD plan) equals the ladder's lookup of the
   drawn block, and the release gate passes.
2. Containment: a household's block lies inside its source geography (its
   PUMA, its identified county, its state outside the excluded counties, or
   its state).
3. Exactness: the inverse CDF gives each block exactly its population share
   on a uniform grid; with a fixed seed the empirical block, county-within-
   PUMA and CD-within-PUMA frequencies pass goodness-of-fit tests against the
   population shares, and the keyed uniforms pass a uniformity test.
4. Determinism: same inputs and seed give identical output; permuting the
   input rows changes nothing per household; clone k's block does not depend
   on how many clones were drawn.
5. Conservation: with K clones every household appears K times with clone
   indices 0..K-1 and clone ids key*K+k, and its clone weights sum to its
   weight; frame cloning conserves every weighted entity's mass and keeps
   the frame's linkage valid.

Differential tests check the block ladder's PUMA against the PUMA ladder
built from the same sources (per-PUMA population and the (PUMA, CD) and
(PUMA, county) overlaps), and the attach path against the ladder's own
primary plan.

Hypothesis is a workspace dev dependency; the clean-wheel lane installs only
pytest, where the property tests skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from microcosm.build.us_runtime.block_location import (
    LOCATION_CLONE_ID_COLUMN,
    LOCATION_CLONE_INDEX_COLUMN,
    LOCATION_SOURCE_GEOGRAPHY_COLUMN,
    LOCATION_WEIGHT_COLUMN,
    SOURCE_GEOGRAPHY_COUNTY,
    SOURCE_GEOGRAPHY_PUMA,
    SOURCE_GEOGRAPHY_STATE,
    SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES,
    SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA,
    US_BLOCK_LOCATION_METADATA_KEY,
    US_BLOCK_LOCATION_RULE_ID,
    CpsIdentifiedCountyList,
    _BlockStrata,
    assign_us_block_location,
    attach_congressional_district_plans,
    cbsa_vintage_agreement,
    clone_frame_for_location,
    cps_county_coverage_ratios,
    cps_excluded_county_sets,
    cps_unidentified_county_shares,
    default_cps_asec_identified_counties_path,
    derive_us_block_geography,
    draw_us_block_locations,
    identified_counties_by_group,
    load_cps_asec_identified_counties,
    load_us_location_ladder,
    location_geography_columns,
    location_uniforms,
    us_block_location_gate,
    us_block_location_manifest,
    with_household_us_block_location,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_block_location import (
    ladder_metadata,
    synthetic_blocks,
    write_location_ladder,
)

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings
HealthCheck = hypothesis.HealthCheck

_PROPERTY = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _load(arrays: dict[str, np.ndarray], *, with_puma: bool = True):
    with TemporaryDirectory() as directory:
        path = write_location_ladder(
            Path(directory) / "ladder.npz", arrays, with_puma=with_puma
        )
        return load_us_location_ladder(path)


def _with_117(ladder, seed: int):
    """Attach a second plan: one district per block, drawn within the state."""

    rng = np.random.default_rng(seed)
    state = ladder.block_geoid // 10**13
    districts = state * 100 + rng.integers(1, 4, size=len(state))
    order = rng.permutation(len(state))  # the registry may come in any order
    return attach_congressional_district_plans(
        ladder,
        block_geoid=ladder.block_geoid[order],
        plans={"117th_congress": districts[order]},
        plan_sources={
            "117th_congress": {
                "vintage": "117th_congress",
                "source": "test BAF CD layer",
                "known_deviations": {"37": "test note"},
            }
        },
    )


@st.composite
def _scenario(draw):
    """A ladder plus households of every source-geography kind."""

    ladder = _with_117(
        _load(synthetic_blocks(draw(st.integers(0, 10_000)))),
        draw(st.integers(0, 10_000)),
    )
    block_state = ladder.block_geoid // 10**13
    block_county = ladder.block_geoid // 10**10
    states = np.unique(block_state)
    counties = np.unique(block_county)
    pumas = np.unique(ladder.puma)
    # Group 0 identifies one county per state (always leaving the rest);
    # group 1 identifies none.
    identified = {
        0: frozenset(
            int(np.unique(block_county[block_state == state])[0]) for state in states
        ),
        1: frozenset(),
    }
    n = draw(st.integers(1, 25))
    kinds = draw(
        st.lists(
            st.sampled_from(["puma", "county", "residual0", "residual1", "state"]),
            min_size=n,
            max_size=n,
        )
    )
    rows = []
    for kind in kinds:
        if kind == "puma":
            puma = int(draw(st.sampled_from(pumas.tolist())))
            rows.append((puma // 10**5, puma, 0, -1))
        elif kind == "county":
            county = int(draw(st.sampled_from(counties.tolist())))
            rows.append((county // 1000, 0, county, -1))
        elif kind.startswith("residual"):
            state = int(draw(st.sampled_from(states.tolist())))
            rows.append((state, 0, 0, int(kind[-1])))
        else:
            rows.append((int(draw(st.sampled_from(states.tolist()))), 0, 0, -1))
    keys = draw(st.lists(st.integers(0, 10**9), min_size=n, max_size=n, unique=True))
    household = pd.DataFrame(
        {
            "household_id": keys,
            "state_fips": [row[0] for row in rows],
            "puma": [row[1] for row in rows],
            "county": [row[2] for row in rows],
            "group": [row[3] for row in rows],
            "weight": draw(
                st.lists(st.floats(0.01, 1e4, allow_nan=False), min_size=n, max_size=n)
            ),
        }
    )
    return ladder, household, identified


def _assign(ladder, household, identified, *, seed=0, clones=1):
    return assign_us_block_location(
        household,
        ladder,
        seed=seed,
        clones=clones,
        weights=household["weight"].to_numpy(),
        puma_column="puma",
        county_column="county",
        identification_group_column="group",
        identified_counties=identified,
    )


def _block_index(ladder, table: pd.DataFrame) -> np.ndarray:
    blocks = table["block_geoid"].astype(np.int64).to_numpy()
    index = np.searchsorted(ladder.block_geoid, blocks)
    assert (ladder.block_geoid[index] == blocks).all()
    return index


# --------------------------------------------------------------------------
# 1. Consistency
# --------------------------------------------------------------------------


@_PROPERTY
@given(scenario=_scenario(), seed=st.integers(0, 2**32), clones=st.integers(1, 4))
def test_every_derived_geography_is_the_blocks_lookup(scenario, seed, clones):
    ladder, household, identified = scenario
    table = _assign(ladder, household, identified, seed=seed, clones=clones)
    index = _block_index(ladder, table)
    block = table["block_geoid"].to_numpy().astype(str)

    assert us_block_location_gate(table, ladder).passed
    assert (table["tract_geoid"].to_numpy() == np.array([b[:11] for b in block])).all()
    assert (table["county_fips"].to_numpy() == np.array([b[:5] for b in block])).all()
    assert (
        table["state_fips"].to_numpy()
        == index * 0 + ladder.block_geoid[index] // 10**13
    ).all()
    assert (
        table["puma"].to_numpy() == np.array([f"{p:07d}" for p in ladder.puma[index]])
    ).all()
    assert (
        table["congressional_district_geoid"].to_numpy()
        == ladder.blocks.congressional_district_geoid[index]
    ).all()
    assert (
        table["congressional_district_geoid__117th_congress"].to_numpy()
        == ladder.congressional_district_plans["117th_congress"][index]
    ).all()
    assert (table["sldu"].to_numpy() == ladder.blocks.sldu[index]).all()
    assert (table["sldl"].to_numpy() == ladder.blocks.sldl[index]).all()
    for column, values in (
        ("place_fips", ladder.blocks.place_fips[index]),
        ("cbsa_code", ladder.blocks.cbsa_code[index]),
    ):
        expected = np.array(["" if v == 0 else f"{v:05d}" for v in values])
        assert (table[column].to_numpy().astype(str) == expected).all()
    # Household state never changes: every drawn block is in it.
    assert (
        table["state_fips"].to_numpy()
        == np.repeat(household["state_fips"].to_numpy(), clones)
    ).all()


def test_gate_names_every_tampered_column(tmp_path) -> None:
    ladder = _with_117(
        load_us_location_ladder(write_location_ladder(tmp_path / "l.npz")), 1
    )
    household = pd.DataFrame(
        {"household_id": [1, 2, 3], "state_fips": [1, 2, 36], "weight": [1.0, 2.0, 3.0]}
    )
    table = assign_us_block_location(household, ladder, seed=3)
    assert us_block_location_gate(table, ladder).passed

    for column in location_geography_columns(ladder):
        if column == "block_geoid":
            continue
        broken = table.copy()
        value = broken.at[0, column]
        broken.at[0, column] = (
            (value + 1) if isinstance(value, (int, np.integer)) else "99999999999"
        )
        result = us_block_location_gate(broken, ladder)
        assert not result.passed
        assert any(failure.startswith(column) for failure in result.failures), column

    unknown = table.copy()
    unknown.at[0, "block_geoid"] = "019999999999999"
    assert "does not carry" in us_block_location_gate(unknown, ladder).failures[0]
    moved = table.copy()
    moved["state_fips"] = [2, 2, 36]
    assert any(
        "state_fips" in f for f in us_block_location_gate(moved, ladder).failures
    )


# --------------------------------------------------------------------------
# 2. Containment
# --------------------------------------------------------------------------


@_PROPERTY
@given(scenario=_scenario(), seed=st.integers(0, 2**32), clones=st.integers(1, 3))
def test_block_lies_inside_the_source_geography(scenario, seed, clones):
    ladder, household, identified = scenario
    table = _assign(ladder, household, identified, seed=seed, clones=clones)
    index = _block_index(ladder, table)
    source = household.loc[household.index.repeat(clones)].reset_index(drop=True)
    kind = table[LOCATION_SOURCE_GEOGRAPHY_COLUMN].to_numpy()
    block_county = ladder.block_geoid[index] // 10**10

    puma = kind == SOURCE_GEOGRAPHY_PUMA
    assert (ladder.puma[index][puma] == source["puma"].to_numpy()[puma]).all()
    county = kind == SOURCE_GEOGRAPHY_COUNTY
    assert (block_county[county] == source["county"].to_numpy()[county]).all()
    residual = kind == SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES
    for row in np.flatnonzero(residual):
        assert block_county[row] not in identified[int(source.at[row, "group"])]
    # Kinds are exactly what each household's source provides.
    expected = np.where(
        source["puma"] > 0,
        SOURCE_GEOGRAPHY_PUMA,
        np.where(
            source["county"] > 0,
            SOURCE_GEOGRAPHY_COUNTY,
            np.where(
                source["group"] >= 0,
                SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES,
                SOURCE_GEOGRAPHY_STATE,
            ),
        ),
    )
    assert (kind == expected).all()


def test_cbsa_narrowing_keeps_blocks_in_the_cbsa(tmp_path) -> None:
    arrays = synthetic_blocks(4)
    arrays["cbsa_code"] = np.where(
        arrays["block_geoid"] // 10**10 % 2 == 1, 11111, 0
    ).astype(np.int32)
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz", arrays))
    state = int(ladder.block_geoid[0] // 10**13)
    draw = draw_us_block_locations(
        ladder,
        household_key=np.arange(200),
        state_fips=np.full(200, state),
        seed=9,
        identification_group=np.zeros(200),
        identified_counties={0: frozenset()},
        cbsa_code=np.full(200, 11111),
    )
    assert set(draw.source_geography) == {
        SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA
    }
    assert (ladder.cbsa_code[draw.block_index] == 11111).all()


def test_empty_candidate_sets_fail_closed(tmp_path) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    counties = np.unique(
        ladder.block_geoid[ladder.block_geoid // 10**13 == 1] // 10**10
    )
    with pytest.raises(ValueError, match="no candidate block"):
        draw_us_block_locations(
            ladder,
            household_key=[1],
            state_fips=[1],
            seed=0,
            identification_group=[0],
            identified_counties={0: frozenset(int(c) for c in counties)},
        )
    with pytest.raises(ValueError, match="no candidate block"):
        draw_us_block_locations(
            ladder, household_key=[1], state_fips=[1], seed=0, county_fips=[1999]
        )
    with pytest.raises(ValueError, match="no candidate block"):
        draw_us_block_locations(
            ladder, household_key=[1], state_fips=[1], seed=0, puma=[100999]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"household_key": [1, 1], "state_fips": [1, 1]}, "unique"),
        ({"household_key": [-1], "state_fips": [1]}, "non-negative"),
        (
            {
                "household_key": [1],
                "state_fips": [1],
                "puma": [100100],
                "county_fips": [1001],
            },
            "at most one",
        ),
        ({"household_key": [1], "state_fips": [2], "county_fips": [1001]}, "disagrees"),
        (
            {"household_key": [1], "state_fips": [1], "identification_group": [0]},
            "identified_counties",
        ),
        ({"household_key": [1], "state_fips": [1], "clones": 0}, "clones"),
    ],
)
def test_inconsistent_inputs_are_refused(tmp_path, kwargs, message) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    with pytest.raises(ValueError, match=message):
        draw_us_block_locations(ladder, seed=0, **kwargs)


def test_puma_draws_need_a_ladder_that_carries_puma(tmp_path) -> None:
    ladder = load_us_location_ladder(
        write_location_ladder(tmp_path / "l.npz", with_puma=False)
    )
    assert ladder.puma is None
    assert "puma" not in location_geography_columns(ladder)
    with pytest.raises(ValueError, match="per-block 'puma'"):
        draw_us_block_locations(
            ladder, household_key=[1], state_fips=[1], seed=0, puma=[100100]
        )
    # State draws still work on a ladder without PUMA (the Route A artifact).
    table = assign_us_block_location(
        pd.DataFrame({"household_id": [1], "state_fips": [1]}), ladder, seed=0
    )
    assert us_block_location_gate(table, ladder).passed


# --------------------------------------------------------------------------
# 3. Exactness and convergence
# --------------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(
    populations=st.lists(st.integers(1, 5000), min_size=1, max_size=30),
    grid=st.integers(1000, 20000),
)
def test_inverse_cdf_gives_each_block_its_population_share(populations, grid):
    population = np.asarray(populations, dtype=np.float64)
    geoid = np.arange(len(population), dtype=np.int64)
    strata = _BlockStrata(np.zeros(len(population), dtype=np.int64), geoid, population)
    uniforms = (np.arange(grid) + 0.5) / grid
    drawn = strata.draw(np.zeros(grid, dtype=np.int64), uniforms)
    counts = np.bincount(drawn, minlength=len(population))
    expected = population / population.sum() * grid
    # A block's interval of width share*grid holds that many grid points +- 1.
    assert np.abs(counts - expected).max() <= 1.0 + 1e-9
    # Uniform endpoints stay inside the stratum.
    edge = strata.draw(np.zeros(2, dtype=np.int64), np.asarray([0.0, 1.0]))
    assert edge.tolist() == [0, len(population) - 1]


def test_empirical_block_shares_converge_to_population_shares() -> None:
    population = np.asarray([1, 3, 10, 30, 100, 300, 1000, 5, 50, 500], dtype=np.int64)
    blocks = {
        "block_geoid": 10010001001000 + np.arange(len(population), dtype=np.int64),
        "population": population,
        "congressional_district_geoid": np.full(len(population), 101, dtype=np.int64),
        "puma": np.full(len(population), 100100, dtype=np.int64),
        "sldu": np.full(len(population), "001", dtype="U3"),
        "sldl": np.full(len(population), "", dtype="U3"),
        "place_fips": np.zeros(len(population), dtype=np.int32),
        "cbsa_code": np.zeros(len(population), dtype=np.int32),
    }
    ladder = _load(blocks)
    n = 200_000
    draw = draw_us_block_locations(
        ladder, household_key=np.arange(n), state_fips=np.ones(n), seed=20260927
    )
    counts = np.bincount(draw.block_index, minlength=len(population))
    expected = population / population.sum() * n
    statistic, p_value = stats.chisquare(counts, expected)
    assert p_value > 1e-3, (statistic, p_value)
    assert np.abs(counts / n - population / population.sum()).max() < 0.005


def test_county_and_district_within_puma_match_the_population_overlap() -> None:
    """Differential vs. the PUMA ladder's semantics: the block draw's implied
    county and CD distribution within a PUMA is the (PUMA, layer) population
    overlap, which ``assign_us_puma_ladder`` draws from directly."""

    ladder = _load(synthetic_blocks(11, states=(6,)))
    puma = int(np.unique(ladder.puma)[0])
    n = 120_000
    draw = draw_us_block_locations(
        ladder,
        household_key=np.arange(n),
        state_fips=np.full(n, puma // 10**5),
        puma=np.full(n, puma),
        seed=7,
    )
    in_puma = ladder.puma == puma
    for layer in (
        ladder.block_geoid // 10**10,
        ladder.blocks.congressional_district_geoid,
    ):
        overlap = pd.Series(ladder.population[in_puma]).groupby(layer[in_puma]).sum()
        drawn = (
            pd.Series(layer[draw.block_index])
            .value_counts()
            .reindex(overlap.index, fill_value=0)
        )
        if len(overlap) > 1:
            _, p_value = stats.chisquare(
                drawn.to_numpy(), overlap.to_numpy() / overlap.sum() * n
            )
            assert p_value > 1e-3
        assert set(drawn[drawn > 0].index) <= set(overlap.index)


def test_keyed_uniforms_are_uniform_and_independent_across_clones() -> None:
    keys = np.arange(100_000)
    first = location_uniforms(0, keys, np.zeros_like(keys))
    second = location_uniforms(0, keys, np.ones_like(keys))
    assert stats.kstest(first, "uniform").pvalue > 1e-3
    assert abs(np.corrcoef(first, second)[0, 1]) < 0.01
    assert (first >= 0).all() and (first <= 1).all()
    assert not np.array_equal(first, location_uniforms(1, keys, np.zeros_like(keys)))


# --------------------------------------------------------------------------
# 4. Determinism
# --------------------------------------------------------------------------


@_PROPERTY
@given(scenario=_scenario(), seed=st.integers(0, 2**32), data=st.data())
def test_draw_is_deterministic_and_invariant_to_row_order(scenario, seed, data):
    ladder, household, identified = scenario
    clones = data.draw(st.integers(1, 3))
    first = _assign(ladder, household, identified, seed=seed, clones=clones)
    again = _assign(ladder, household, identified, seed=seed, clones=clones)
    pd.testing.assert_frame_equal(first, again)

    order = data.draw(st.permutations(range(len(household))))
    shuffled = _assign(
        ladder,
        household.iloc[list(order)].reset_index(drop=True),
        identified,
        seed=seed,
        clones=clones,
    )
    key = ["household_id", LOCATION_CLONE_INDEX_COLUMN]
    pd.testing.assert_frame_equal(
        first.sort_values(key).reset_index(drop=True),
        shuffled.sort_values(key).reset_index(drop=True),
    )


@_PROPERTY
@given(scenario=_scenario(), seed=st.integers(0, 2**32), clones=st.integers(2, 5))
def test_clone_k_block_does_not_depend_on_the_clone_count(scenario, seed, clones):
    ladder, household, identified = scenario
    single = _assign(ladder, household, identified, seed=seed, clones=1)
    many = _assign(ladder, household, identified, seed=seed, clones=clones)
    fewer = _assign(ladder, household, identified, seed=seed, clones=clones - 1)
    first_clone = many[many[LOCATION_CLONE_INDEX_COLUMN] == 0].reset_index(drop=True)
    assert (first_clone["block_geoid"] == single["block_geoid"]).all()
    shared = many[many[LOCATION_CLONE_INDEX_COLUMN] < clones - 1].reset_index(drop=True)
    assert (shared["block_geoid"] == fewer["block_geoid"]).all()


def test_a_different_seed_moves_draws(tmp_path) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    household = pd.DataFrame({"household_id": np.arange(500), "state_fips": 36})
    a = assign_us_block_location(household, ladder, seed=0)
    b = assign_us_block_location(household, ladder, seed=1)
    assert (a["block_geoid"] != b["block_geoid"]).any()


# --------------------------------------------------------------------------
# 5. Conservation under cloning
# --------------------------------------------------------------------------


@_PROPERTY
@given(scenario=_scenario(), seed=st.integers(0, 2**32), clones=st.integers(1, 6))
def test_cloning_conserves_weight_and_household_mass(scenario, seed, clones):
    ladder, household, identified = scenario
    table = _assign(ladder, household, identified, seed=seed, clones=clones)
    n = len(household)
    assert len(table) == n * clones
    grouped = table.groupby("household_id")
    assert (grouped.size() == clones).all()
    assert (
        grouped[LOCATION_CLONE_INDEX_COLUMN].apply(sorted).apply(list)
        == [list(range(clones))] * n
    ).all()
    assert (
        table[LOCATION_CLONE_ID_COLUMN]
        == table["household_id"] * clones + table[LOCATION_CLONE_INDEX_COLUMN]
    ).all()
    assert table[LOCATION_CLONE_ID_COLUMN].is_unique
    per_household = (
        grouped[LOCATION_WEIGHT_COLUMN].sum().reindex(household["household_id"])
    )
    np.testing.assert_allclose(
        per_household.to_numpy(), household["weight"].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        table[LOCATION_WEIGHT_COLUMN].sum(), household["weight"].sum(), rtol=1e-12
    )


def _us_frame(n_households: int, rng: np.random.Generator) -> Frame:
    sizes = rng.integers(1, 4, size=n_households)
    household_ids = np.arange(1, n_households + 1, dtype=np.int64) * 10
    person_household = np.repeat(household_ids, sizes)
    n_people = len(person_household)
    person_ids = np.arange(1, n_people + 1, dtype=np.int64)
    parent = np.where(
        rng.random(n_people) < 0.3, person_ids[rng.integers(0, n_people, n_people)], 0
    )
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": person_ids,
                "person_household_id": person_household,
                "person_tax_unit_id": person_household + 1,
                "person_spm_unit_id": person_household + 2,
                "person_family_id": person_household + 3,
                "person_marital_unit_id": person_ids + 10**6,
                "parent_1_id": parent,
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": rng.choice([1, 2, 36], size=n_households),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2}),
        "family": pd.DataFrame({"family_id": household_ids + 3}),
        "marital_unit": pd.DataFrame({"marital_unit_id": person_ids + 10**6}),
    }
    weights = {
        "household": Weights(
            values=rng.uniform(0.5, 500.0, size=n_households), kind=WeightKind.DESIGN
        )
    }
    strata = pd.Series(rng.choice(["asec_2023", "asec_2024"], size=n_people))
    return Frame(tables, US_SCHEMA, weights, strata, metadata={"stage": "test"})


@_PROPERTY
@given(
    n_households=st.integers(1, 12),
    clones=st.integers(1, 5),
    seed=st.integers(0, 2**32),
)
def test_frame_location_conserves_mass_and_linkage(n_households, clones, seed):
    rng = np.random.default_rng(seed)
    frame = _us_frame(n_households, rng)
    ladder = _load(synthetic_blocks(seed % 97))
    located, table = with_household_us_block_location(
        frame,
        ladder,
        seed=seed,
        clones=clones,
        person_reference_columns=("parent_1_id",),
    )
    located.revalidate()
    assert located.n("household") == frame.n("household") * clones
    assert located.n("person") == frame.n("person") * clones
    np.testing.assert_allclose(
        located.weights_for("household").total,
        frame.weights_for("household").total,
        rtol=1e-12,
    )
    assert located.weights_for("household").kind == frame.weights_for("household").kind
    assert located.mass_log == frame.mass_log
    household = located.table("household")
    assert us_block_location_gate(household, ladder).passed
    receipt = located.metadata[US_BLOCK_LOCATION_METADATA_KEY]
    assert receipt["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert receipt["seed"] == seed and receipt["clones"] == clones
    assert located.metadata["stage"] == "test"
    if clones == 1:
        assert (
            household["household_id"] == frame.table("household")["household_id"]
        ).all()
        assert "household_location_clone_index" not in household.columns
        return
    # Clone k of household h is household h*K+k, drawn for (h, k).
    assert (
        household["household_id"].to_numpy()
        == table[LOCATION_CLONE_ID_COLUMN].to_numpy()
    ).all()
    person = located.table("person")
    original = frame.table("person")
    clone = person["person_location_clone_index"].to_numpy()
    assert (
        person["person_household_id"].to_numpy()
        == np.repeat(original["person_household_id"].to_numpy(), clones) * clones
        + clone
    ).all()
    raw_parent = np.repeat(original["parent_1_id"].to_numpy(), clones)
    assert (
        person["parent_1_id"].to_numpy()
        == np.where(raw_parent == 0, 0, raw_parent * clones + clone)
    ).all()
    # Each person's parent is in the same clone (and so the same household copy).
    has_parent = person["parent_1_id"].to_numpy() > 0
    assert (
        person["parent_1_id"].to_numpy()[has_parent] % clones == clone[has_parent]
    ).all()
    assert (
        located.strata.to_numpy() == np.repeat(frame.strata.to_numpy(), clones)
    ).all()


def test_clone_frame_refuses_a_second_clone_pass() -> None:
    frame = clone_frame_for_location(_us_frame(3, np.random.default_rng(0)), 2)
    with pytest.raises(ValueError, match="already location-cloned"):
        clone_frame_for_location(frame, 2)
    assert clone_frame_for_location(frame, 1) is frame


# --------------------------------------------------------------------------
# CPS ASEC county identification
# --------------------------------------------------------------------------


def _official(year: int, counties) -> dict[int, CpsIdentifiedCountyList]:
    return {
        year: CpsIdentifiedCountyList(
            asec_year=year,
            counties=frozenset(counties),
            source="https://example.test/cpsmar.pdf",
            source_sha256="0" * 64,
        )
    }


@settings(max_examples=200, deadline=None)
@given(
    coded=st.sets(st.sampled_from([1001, 1003, 1005, 2013, 2020, 36061]), max_size=6),
    listed=st.sets(st.sampled_from([1001, 1003, 1005, 2013, 2020, 36061]), max_size=6),
)
def test_a_county_is_excluded_only_with_both_confirmations(coded, listed):
    excluded, record = cps_excluded_county_sets(
        {2024: coded}, _official(2025, listed), asec_year_of_group={2024: 2025}
    )
    assert excluded[2024] == frozenset(coded) & frozenset(listed)
    group = record["groups"]["2024"]
    assert group["basis"] == "official_list_and_coded"
    assert group["coded_not_listed"] == [
        f"{c:05d}" for c in sorted(set(coded) - set(listed))
    ]
    assert group["listed_not_coded"] == [
        f"{c:05d}" for c in sorted(set(listed) - set(coded))
    ]


def test_without_an_official_list_the_coded_set_is_the_fallback() -> None:
    excluded, record = cps_excluded_county_sets(
        {2022: {1001}, 2023: {1001, 1003}}, None, asec_year_of_group={2022: 2023}
    )
    assert excluded == {2022: frozenset({1001}), 2023: frozenset({1001, 1003})}
    assert record["groups"]["2022"]["basis"] == "data_derived_fallback"
    assert record["counties_whose_exclusion_varies_across_groups"] == ["01003"]


def test_identified_counties_are_kept_per_group() -> None:
    sets = identified_counties_by_group(
        state_fips=[1, 1, 1, 2],
        county_code=[1, 0, 3, 13],
        group=[2022, 2022, 2023, 2023],
    )
    assert sets == {2022: frozenset({1001}), 2023: frozenset({1003, 2013})}
    with pytest.raises(ValueError, match="3-digit"):
        identified_counties_by_group([1], [1000], [2022])


@_PROPERTY
@given(scenario=_scenario(), seed=st.integers(0, 2**32))
def test_a_cps_record_lands_in_its_county_or_an_unexcluded_county(scenario, seed):
    """The CPS form of containment: an identified county's record lands in it;
    a GTCO=0 record lands in its state outside the excluded counties."""

    ladder, household, identified = scenario
    cps = household[household["puma"] == 0].reset_index(drop=True)
    if cps.empty:
        return
    table = _assign(ladder, cps, identified, seed=seed)
    county = table["county_fips"].astype(int).to_numpy()
    for row, source in cps.iterrows():
        if source["county"] > 0:
            assert county[row] == source["county"]
        elif source["group"] >= 0:
            assert county[row] // 1000 == source["state_fips"]
            assert county[row] not in identified[int(source["group"])]


def test_partial_county_remainder_restores_expected_county_population() -> None:
    """With the opt-in remainder weighting, a partially coded county's
    expected mass (its coded share plus its expected share of the GTCO=0
    draws) equals its census population."""

    population = np.asarray([1000, 1000, 2000, 4000], dtype=np.int64)
    blocks = {
        "block_geoid": np.asarray(
            [c * 10**10 + 1001 for c in (1001, 1003, 1005, 1007)], dtype=np.int64
        ),
        "population": population,
        "congressional_district_geoid": np.full(4, 101, dtype=np.int64),
        "puma": np.full(4, 100100, dtype=np.int64),
        "sldu": np.full(4, "001", dtype="U3"),
        "sldl": np.full(4, "", dtype="U3"),
        "place_fips": np.zeros(4, dtype=np.int32),
        "cbsa_code": np.zeros(4, dtype=np.int32),
    }
    ladder = _load(blocks)
    # 1001 fully coded; 1003 coded for 25% of its people; 1005 and 1007 never.
    coverage_raw = {1001: 1000.0, 1003: 250.0}
    county = np.asarray([1001, 1003, 1005])
    weight = np.asarray([coverage_raw[1001], coverage_raw[1003], 0.0])
    ratios = cps_county_coverage_ratios(
        ladder,
        county_fips=np.where(weight > 0, county, 0),
        person_weight=weight,
        group=np.zeros(3, dtype=np.int64),
    )
    # Median-normalized: raw ratios 1.0 and 0.25 have median 0.625.
    assert ratios[0] == pytest.approx({1001: 1.6, 1003: 0.4})
    raw = {1001: 1.0, 1003: 0.25}
    shares = cps_unidentified_county_shares({0: raw}, {0: {1001}})
    assert shares == {0: {1003: 0.75}}
    uncoded_mass = population.sum() - 1000 - 250
    n = 200_000
    draw = draw_us_block_locations(
        ladder,
        household_key=np.arange(n),
        state_fips=np.ones(n),
        identification_group=np.zeros(n),
        identified_counties={0: frozenset({1001})},
        unidentified_county_share=shares,
        seed=5,
    )
    drawn = np.bincount(draw.block_index, minlength=4) / n * uncoded_mass
    expected = population - np.asarray([1000, 250, 0, 0])
    np.testing.assert_allclose(drawn, expected, rtol=0.02)
    np.testing.assert_allclose(
        drawn + np.asarray([1000, 250, 0, 0]), population, rtol=0.02
    )
    # Default (the ruled rule): plain population over the unexcluded counties.
    plain = draw_us_block_locations(
        ladder,
        household_key=np.arange(n),
        state_fips=np.ones(n),
        identification_group=np.zeros(n),
        identified_counties={0: frozenset({1001})},
        seed=5,
    )
    shares_plain = np.bincount(plain.block_index, minlength=4)[1:] / n
    np.testing.assert_allclose(
        shares_plain, population[1:] / population[1:].sum(), atol=0.01
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        draw_us_block_locations(
            ladder,
            household_key=[1],
            state_fips=[1],
            seed=0,
            identification_group=[0],
            identified_counties={0: frozenset()},
            unidentified_county_share={0: {1003: 1.5}},
        )


def test_cbsa_vintage_agreement_detects_a_different_delineation(tmp_path) -> None:
    arrays = synthetic_blocks(2)
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz", arrays))
    county = np.unique(ladder.block_geoid // 10**10)
    first = np.searchsorted(ladder.block_geoid // 10**10, county)
    cbsa = ladder.cbsa_code[first].astype(np.int64)
    agreed = cbsa_vintage_agreement(ladder, county, cbsa)
    assert agreed["agrees"] and agreed["disagreeing_pairs"] == 0
    shifted = cbsa_vintage_agreement(ladder, county, cbsa + 1)
    assert not shifted["agrees"] and shifted["disagreeing_pairs"] == len(county)
    assert not cbsa_vintage_agreement(ladder, [0], [0])["agrees"]


def test_packaged_official_lists_load_and_match_their_provenance() -> None:
    lists = load_cps_asec_identified_counties()
    assert sorted(lists) == [2023, 2024, 2025, 2026]
    assert {year: len(entry.counties) for year, entry in lists.items()} == {
        2023: 277,
        2024: 277,
        2025: 277,
        2026: 361,
    }
    path = default_cps_asec_identified_counties_path()
    provenance = json.loads(path.with_name(path.name + ".provenance.json").read_text())
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["csv_sha256"]
    for year, entry in lists.items():
        assert entry.source_sha256 == provenance["years"][str(year)]["sha256"]
        assert entry.source.endswith(f"cpsmar{year % 100:02d}.pdf")
    # Spot checks against the documentation text.
    assert 6037 in lists[2025].counties  # Los Angeles
    assert 48303 in lists[2025].counties  # Lubbock, listed though never coded
    assert 48439 not in lists[2025].counties  # Tarrant, coded only in part
    assert 48439 in lists[2026].counties  # listed from the 2026 documentation on


# --------------------------------------------------------------------------
# Ladder: PUMA, plans, manifest
# --------------------------------------------------------------------------


def test_location_ladder_validates_the_puma_array(tmp_path) -> None:
    arrays = synthetic_blocks(3)
    wrong_state = dict(arrays, puma=arrays["puma"] + 10**5)
    with pytest.raises(ValueError, match="lie in its block's state"):
        load_us_location_ladder(write_location_ladder(tmp_path / "a.npz", wrong_state))
    split = dict(arrays)
    split["puma"] = arrays["puma"].copy()
    same_tract = np.flatnonzero(
        (arrays["block_geoid"][1:] // 10**4) == (arrays["block_geoid"][:-1] // 10**4)
    )
    split["puma"][same_tract[0] + 1] = split["puma"][same_tract[0]] + 1
    with pytest.raises(ValueError, match="constant within a 2020 tract"):
        load_us_location_ladder(write_location_ladder(tmp_path / "b.npz", split))
    unrecorded = ladder_metadata(with_puma=False)
    with pytest.raises(ValueError, match="'puma' layer"):
        load_us_location_ladder(
            write_location_ladder(tmp_path / "c.npz", arrays, metadata=unrecorded)
        )
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "d.npz", arrays))
    assert ladder.layer_vintages["puma"] == "2020_puma"
    assert len(ladder.sha256) == 64


def test_attach_aligns_by_block_and_refuses_disagreement(tmp_path) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    attached = _with_117(ladder, 5)
    assert set(attached.congressional_district_plans) == {
        "119th_congress",
        "117th_congress",
    }
    assert attached.congressional_district_plan_sources["117th_congress"][
        "known_deviations"
    ] == {"37": "test note"}
    primary = ladder.blocks.congressional_district_geoid
    same = attach_congressional_district_plans(
        ladder,
        block_geoid=ladder.block_geoid[::-1],
        plans={"119th_congress": primary[::-1]},
        plan_sources={
            "119th_congress": {"vintage": "119th_congress", "source": "registry"}
        },
    )
    assert "registry" in same.congressional_district_plan_sources["119th_congress"]
    with pytest.raises(ValueError, match="disagrees"):
        attach_congressional_district_plans(
            ladder,
            block_geoid=ladder.block_geoid,
            plans={
                "119th_congress": primary // 100 * 100
                + np.where(primary % 100 == 1, 2, 1)
            },
            plan_sources={
                "119th_congress": {"vintage": "119th_congress", "source": "x"}
            },
        )
    with pytest.raises(ValueError, match="omits"):
        attach_congressional_district_plans(
            ladder,
            block_geoid=ladder.block_geoid[1:],
            plans={"p": primary[1:]},
            plan_sources={"p": {"vintage": "p", "source": "x"}},
        )
    with pytest.raises(ValueError, match="their block's state"):
        attach_congressional_district_plans(
            ladder,
            block_geoid=ladder.block_geoid,
            plans={"p": np.full(len(ladder), 5601)},
            plan_sources={"p": {"vintage": "p", "source": "x"}},
        )
    with pytest.raises(ValueError, match="vintage and source"):
        attach_congressional_district_plans(
            ladder,
            block_geoid=ladder.block_geoid,
            plans={"p": primary},
            plan_sources={},
        )


def test_manifest_records_the_rule_seed_and_sources(tmp_path) -> None:
    ladder = _with_117(
        load_us_location_ladder(write_location_ladder(tmp_path / "l.npz")), 2
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "state_fips": [1, 1, 36, 2],
            "puma": [0, int(ladder.puma[0]), 0, 0],
            "county": [0, 0, 36001, 0],
            "group": [0, -1, -1, -1],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )
    table = _assign(ladder, household, {0: frozenset()}, seed=42, clones=3)
    record = us_block_location_manifest(
        table, ladder, seed=42, clones=3, candidate_set_rule={"rule": "test"}
    )
    assert record["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert record["location_rule"] == "block_v1"
    assert (record["seed"], record["clones"]) == (42, 3)
    assert record["households"] == 4 and record["rows"] == 12
    assert record["source_geography_households"] == {
        "puma": 1,
        "county": 1,
        "state_unidentified_counties_cbsa": 0,
        "state_unidentified_counties": 1,
        "state": 1,
    }
    assert record["block_ladder"]["sha256"] == ladder.sha256
    assert set(record["block_ladder"]["congressional_district_plans"]) == {
        "117th_congress",
        "119th_congress",
    }
    assert record["candidate_set_rule"] == {"rule": "test"}
    assert record["location_weight_total"] == pytest.approx(10.0)
    json.dumps(record)  # serializable


def test_derived_columns_match_the_legacy_ladder_formats(tmp_path) -> None:
    ladder = load_us_location_ladder(write_location_ladder(tmp_path / "l.npz"))
    columns = derive_us_block_geography(ladder, np.arange(len(ladder)))
    assert all(len(value) == 15 for value in columns["block_geoid"])
    assert all(len(value) == 7 for value in columns["puma"])
    assert columns["congressional_district_geoid"].dtype == np.int64
    assert all(isinstance(value, str) for value in columns["county_fips"])
    assert set(columns["place_fips"]) <= {"", "12345", "67890"}


# --------------------------------------------------------------------------
# Real artifacts (local only: set the paths to run)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_national_ladder_differential_against_the_puma_ladder() -> None:
    """On the national artifacts: the PUMA-carrying block ladder's seven core
    arrays equal the Route A ladder byte for byte, and its per-block PUMA
    reproduces the PUMA ladder's per-PUMA population and its (PUMA, CD) and
    (PUMA, county) overlaps exactly. Set MICROCOSM_US_BLOCK_LADDER_PUMA,
    MICROCOSM_US_BLOCK_LADDER_ROUTE_A and MICROCOSM_US_PUMA_LADDER to run."""

    import os

    paths = [
        os.environ.get(name)
        for name in (
            "MICROCOSM_US_BLOCK_LADDER_PUMA",
            "MICROCOSM_US_BLOCK_LADDER_ROUTE_A",
            "MICROCOSM_US_PUMA_LADDER",
        )
    ]
    if not all(path and Path(path).exists() for path in paths):
        pytest.skip("national ladder artifacts not configured")
    new, route_a, puma_ladder = (np.load(path) for path in paths)
    for key in route_a.files:
        if key != "metadata_json":
            assert np.array_equal(new[key], route_a[key]), key
    population = pd.Series(new["population"].astype(np.int64))
    puma = new["puma"]
    by_puma = population.groupby(puma).sum()
    assert by_puma.index.tolist() == puma_ladder["puma"].tolist()
    assert by_puma.tolist() == puma_ladder["puma_population"].astype(np.int64).tolist()
    for layer, values in (
        ("cd", new["congressional_district_geoid"]),
        ("county", new["block_geoid"] // 10**10),
    ):
        overlap = population.groupby([puma, values]).sum().sort_index()
        assert overlap.tolist() == (
            puma_ladder[f"{layer}_overlap_population"].astype(np.int64).tolist()
        )
        assert overlap.index.get_level_values(1).tolist() == (
            puma_ladder[f"{layer}_overlap_{layer}"].astype(np.int64).tolist()
        )
