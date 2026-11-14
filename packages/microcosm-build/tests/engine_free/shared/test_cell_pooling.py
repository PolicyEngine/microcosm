"""Pooling units into cells, with fallbacks for thin cells (microcosm#1524)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.cell_pooling import (
    PoolingDeclaration,
    allocation_shares,
    apply_fallback,
    assign_levels_by_rank,
    cumulative_quota_shares,
    pool,
    stratum_weights,
)

_SETTINGS = settings(max_examples=80, deadline=None)


@st.composite
def _ranked(draw):
    """Units in groups with quotas; some levels have no quota."""
    n_groups = draw(st.integers(min_value=1, max_value=4))
    n_levels = draw(st.integers(min_value=1, max_value=6))
    quotas = np.array(
        [
            draw(
                st.lists(
                    st.one_of(st.just(0.0), st.floats(min_value=0.5, max_value=500)),
                    min_size=n_levels,
                    max_size=n_levels,
                )
            )
            for _ in range(n_groups)
        ]
    )
    for g in range(n_groups):  # every group needs some quota
        if quotas[g].sum() <= 0:
            quotas[g, draw(st.integers(0, n_levels - 1))] = 1.0
    count = draw(st.integers(min_value=1, max_value=80))
    values = np.array(
        draw(
            st.lists(
                st.sampled_from([1e5, 2e5, 2e5, 3.5e5, 9e5, 1e6]),
                min_size=count,
                max_size=count,
            )
        )
    )
    groups = np.array(
        draw(st.lists(st.integers(0, n_groups - 1), min_size=count, max_size=count))
    )
    tie_break = np.arange(count, dtype=np.uint64)[::-1] * 7919 % 104729
    return values, tie_break.astype(np.uint64), groups, quotas


@_SETTINGS
@given(_ranked())
def test_levels_rise_with_value_within_each_group(case) -> None:
    values, tie_break, groups, quotas = case
    levels = assign_levels_by_rank(values, tie_break, groups, quotas)
    order = np.lexsort((tie_break, values, groups))
    for g in np.unique(groups):
        in_group = order[groups[order] == g]
        assert np.all(np.diff(levels[in_group]) >= 0)


@_SETTINGS
@given(_ranked(), st.randoms(use_true_random=False))
def test_levels_do_not_depend_on_input_order(case, random) -> None:
    values, tie_break, groups, quotas = case
    levels = assign_levels_by_rank(values, tie_break, groups, quotas)
    permutation = list(range(len(values)))
    random.shuffle(permutation)
    permutation = np.array(permutation)
    shuffled = assign_levels_by_rank(
        values[permutation], tie_break[permutation], groups[permutation], quotas
    )
    assert np.array_equal(shuffled, levels[permutation])


@_SETTINGS
@given(_ranked(), st.integers(min_value=-6, max_value=6))
def test_only_quota_shares_matter(case, power) -> None:
    values, tie_break, groups, quotas = case
    levels = assign_levels_by_rank(values, tie_break, groups, quotas)
    scaled = assign_levels_by_rank(values, tie_break, groups, quotas * 2.0**power)
    assert np.array_equal(levels, scaled)


@_SETTINGS
@given(_ranked())
def test_each_level_holds_its_quota_share_within_one_unit(case) -> None:
    values, tie_break, groups, quotas = case
    levels = assign_levels_by_rank(values, tie_break, groups, quotas)
    for g in np.unique(groups):
        members = levels[groups == g]
        shares = quotas[g] / quotas[g].sum()
        held = np.bincount(members, minlength=quotas.shape[1])
        assert np.all(np.abs(held - len(members) * shares) < 1 + 1e-9)
        # A level with no quota never receives a unit.
        assert np.all(held[quotas[g] == 0] == 0)


def test_a_group_with_units_must_have_a_quota() -> None:
    with pytest.raises(ValueError, match="no quota"):
        assign_levels_by_rank(
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([[1.0, 1.0], [0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="integer"):
        assign_levels_by_rank(
            np.array([1.0]), np.array([0]), np.array([0.0]), np.array([[1.0]])
        )


def test_cumulative_quota_shares_close_at_one() -> None:
    shares = cumulative_quota_shares(np.array([[0.0, 1.0, 0.0, 2.0, 0.0], [0.0] * 5]))
    assert shares[0].tolist() == [0.0, 1 / 3, 1 / 3, 1.0, 1.0]
    assert shares[1].tolist() == [0.0] * 5


@_SETTINGS
@given(_ranked())
def test_weights_add_up_to_each_represented_stratum(case) -> None:
    values, tie_break, groups, quotas = case
    levels = assign_levels_by_rank(values, tie_break, groups, quotas)
    weights, unrepresented = stratum_weights(groups, levels, quotas)
    assert np.all(weights > 0)
    represented = quotas - unrepresented
    totals = np.zeros_like(quotas)
    np.add.at(totals, (groups, levels), weights)
    assert np.allclose(totals, represented, rtol=1e-12, atol=1e-9)
    assert weights.sum() == pytest.approx(represented.sum(), rel=1e-12)


def test_pool_means_are_weighted_and_empty_pools_are_nan() -> None:
    pooled = pool(
        np.array([0, 0, 2]),
        np.array([1.0, 3.0, 2.0]),
        {"value": np.array([10.0, 20.0, 5.0])},
        n_pools=3,
    )
    assert pooled.sampled.tolist() == [2, 0, 1]
    assert pooled.weight.tolist() == [4.0, 0.0, 2.0]
    assert pooled.means["value"][0] == pytest.approx(17.5)
    assert np.isnan(pooled.means["value"][1])
    assert pooled.means["value"][2] == 5.0


@st.composite
def _hierarchy(draw):
    """Cells under authorities under regions under one nation."""
    regions = draw(st.integers(1, 3))
    authorities = draw(st.integers(regions, 6))
    cells = draw(st.integers(authorities, 20))
    authority_region = np.array(
        list(range(regions))
        + draw(
            st.lists(
                st.integers(0, regions - 1),
                min_size=authorities - regions,
                max_size=authorities - regions,
            )
        )
    )
    cell_authority = np.array(
        list(range(authorities))
        + draw(
            st.lists(
                st.integers(0, authorities - 1),
                min_size=cells - authorities,
                max_size=cells - authorities,
            )
        )
    )
    sampled = np.array(
        draw(st.lists(st.integers(0, 12), min_size=cells, max_size=cells))
    )
    return sampled, cell_authority, authority_region


@_SETTINGS
@given(_hierarchy(), st.integers(1, 25))
def test_a_fallback_never_crosses_its_parent(case, minimum) -> None:
    sampled, cell_authority, authority_region = case
    authority_sampled = np.bincount(
        cell_authority, weights=sampled, minlength=len(authority_region)
    ).astype(int)
    region_sampled = np.bincount(
        authority_region,
        weights=authority_sampled,
        minlength=authority_region.max() + 1,
    ).astype(int)
    nation_sampled = np.array([region_sampled.sum()])
    levels = [sampled, authority_sampled, region_sampled, nation_sampled]
    parents = [cell_authority, authority_region, np.zeros(len(region_sampled), int)]
    level, index, below = apply_fallback(levels, parents, minimum=minimum)
    for cell in range(len(sampled)):
        ancestors = [cell]
        for parent in parents:
            ancestors.append(int(parent[ancestors[-1]]))
        # The pool read is the cell's own ancestor at that level ...
        assert index[cell] == ancestors[level[cell]]
        # ... the first one with enough units, or the top, flagged.
        enough = [levels[k][ancestors[k]] >= minimum for k in range(4)]
        if any(enough):
            assert level[cell] == enough.index(True) and not below[cell]
        else:
            assert level[cell] == 3 and below[cell]


def test_apply_fallback_refuses_a_broken_hierarchy() -> None:
    with pytest.raises(ValueError, match="parent map"):
        apply_fallback([np.array([1, 2])], [np.array([0, 0])], minimum=1)
    with pytest.raises(ValueError, match="points outside"):
        apply_fallback([np.array([1, 2]), np.array([3])], [np.array([0, 1])], minimum=1)


def test_allocation_shares_split_cells_by_their_weight() -> None:
    shares = allocation_shares(
        np.array([0, 0, 0, 2]),
        np.array(["S1", "S2", "S2", "S3"]),
        np.array([1.0, 1.0, 2.0, 5.0]),
        n_cells=3,
        default=pd.DataFrame(
            {"cell": [1, 1, 2], "target": ["S1", "S3", "S9"], "share": [0.4, 0.6, 1.0]}
        ),
    )
    assert shares.to_dict("list") == {
        "cell": [0, 0, 1, 1, 2],
        "target": ["S1", "S2", "S1", "S3", "S3"],
        "share": [0.25, 0.75, 0.4, 0.6, 1.0],
    }
    assert np.allclose(shares.groupby("cell")["share"].sum(), 1.0)
    with pytest.raises(ValueError, match="no default"):
        allocation_shares(np.array([0]), np.array(["S1"]), np.array([1.0]), n_cells=2)


def test_pooling_declaration_validates_its_levels_and_minimums() -> None:
    declaration = PoolingDeclaration(
        levels=("msoa_band", "authority_band"), minimums={"means": 30, "knots": 100}
    )
    assert declaration.minimum("knots") == 100
    with pytest.raises(KeyError):
        declaration.minimum("land")
    with pytest.raises(TypeError):
        declaration.minimums["means"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="distinct"):
        PoolingDeclaration(levels=("a", "a"))
    with pytest.raises(ValueError, match="positive integer"):
        PoolingDeclaration(levels=("a",), minimums={"means": 0})
