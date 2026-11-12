"""Land as the residual of value less depreciated building cost (microcosm#1524)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from microcosm.build.residual_land import (
    cell_residual,
    convex_mean,
    cost_level_for_ratio,
    cost_level_for_total,
    depreciation_scale,
    land_per_unit,
    log_ratio_flags,
    rank_correlation,
    residual_along_rank,
    scale_cap_rescale,
    scale_retained_share,
    unit_cost,
)

_SETTINGS = settings(max_examples=80, deadline=None)

_positive = st.floats(min_value=1.0, max_value=2e6, allow_nan=False)
_SIZES = st.integers(1, 30)


@st.composite
def _cells(draw, size=_SIZES):
    n = draw(size)
    values = np.array(draw(st.lists(_positive, min_size=n, max_size=n)))
    costs = np.array(
        draw(st.lists(st.floats(0.0, 1e6, allow_nan=False), min_size=n, max_size=n))
    )
    counts = np.array(draw(st.lists(st.integers(1, 500), min_size=n, max_size=n)))
    return values, costs, counts.astype(float)


@_SETTINGS
@given(_cells(), st.floats(0.0, 3.0), st.floats(0.0, 3.0))
def test_land_lies_in_zero_to_value_and_never_rises_with_kappa(cells, k1, k2) -> None:
    values, costs, counts = cells
    low, high = sorted((k1, k2))
    land_low = land_per_unit(values, costs, low)
    land_high = land_per_unit(values, costs, high)
    assert np.all(land_low >= 0) and np.all(land_low <= values)
    assert np.all(land_high <= land_low)
    totals = cell_residual(values, costs, counts, high)
    assert np.allclose(totals, counts * land_high)


@_SETTINGS
@given(_cells(), st.floats(0.0, 3.0))
def test_land_at_the_rank_is_never_below_the_pooled_residual(cells, kappa) -> None:
    values, costs, _ = cells
    # Each cell is split into two bins around its mean, with equal weight.
    spread = np.minimum(values, costs) * 0.5
    value_bins = np.stack([values - spread, values + spread], axis=1)
    cost_bins = np.stack([costs + spread * 0.1, costs - spread * 0.1], axis=1)
    cost_bins = np.maximum(cost_bins, 0)
    weights = np.ones_like(value_bins)
    along = residual_along_rank(value_bins, cost_bins, weights, kappa)
    pooled = land_per_unit(value_bins.mean(axis=1), cost_bins.mean(axis=1), kappa)
    assert np.all(along >= pooled - 1e-6 * np.maximum(values, 1))


@_SETTINGS
@given(_cells(size=st.integers(2, 40)), st.floats(0.02, 0.98))
def test_the_ratio_solve_reproduces_its_target(cells, ratio) -> None:
    values, costs, counts = cells
    costs = np.maximum(costs, 1.0)  # every cell has a building
    kappa = cost_level_for_ratio(values, costs, counts, ratio)
    land = cell_residual(values, costs, counts, kappa).sum()
    assert land == pytest.approx(ratio * (counts * values).sum(), rel=1e-9)
    # The root is where land falls through the target, so a step either side
    # lands on the other side of it.
    stock = (counts * values).sum()
    step = 1e-6 * max(kappa, 1e-9)
    assert cell_residual(values, costs, counts, kappa + step).sum() <= ratio * stock
    if kappa > step:
        assert cell_residual(
            values, costs, counts, kappa - step
        ).sum() >= ratio * stock * (1 - 1e-12)


def test_the_ratio_solve_matches_bisection_on_a_small_case() -> None:
    values = np.array([100.0, 200.0, 50.0, 400.0])
    costs = np.array([80.0, 60.0, 70.0, 100.0])
    counts = np.array([3.0, 1.0, 5.0, 2.0])
    ratio = 0.4
    target = ratio * (counts * values).sum()
    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2
        if cell_residual(values, costs, counts, middle).sum() > target:
            low = middle
        else:
            high = middle
    assert cost_level_for_ratio(values, costs, counts, ratio) == pytest.approx(
        low, rel=1e-12
    )


def test_the_ratio_solve_refuses_an_unreachable_ratio() -> None:
    values = np.array([100.0, 100.0])
    costs = np.array([0.0, 50.0])
    counts = np.array([1.0, 1.0])
    # Half the value has no building, so land never falls below half.
    with pytest.raises(ValueError, match="No cost level"):
        cost_level_for_ratio(values, costs, counts, 0.4)
    assert cost_level_for_ratio(values, costs, counts, 0.75) == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        cost_level_for_ratio(values, costs, counts, 1.0)


@_SETTINGS
@given(_cells(), st.floats(1e3, 1e12))
def test_the_total_solve_reproduces_its_target(cells, target) -> None:
    _, costs, counts = cells
    costs = np.maximum(costs, 1.0)
    kappa = cost_level_for_total(target, counts * costs)
    assert (kappa * counts * costs).sum() == pytest.approx(target, rel=1e-12)


@st.composite
def _profiles(draw):
    n = draw(st.integers(1, 30))
    # A profile is 0 for new buildings and a moderate number otherwise.
    shape = np.array(
        draw(
            st.lists(
                st.one_of(st.just(0.0), st.floats(1e-3, 3.0, allow_nan=False)),
                min_size=n,
                max_size=n,
            )
        )
    )
    weights = np.array(
        draw(st.lists(st.floats(0.1, 100.0, allow_nan=False), min_size=n, max_size=n))
    )
    cap = draw(st.floats(0.2, 1.0))
    return shape, weights, cap


@_SETTINGS
@given(_profiles(), st.floats(0.0, 1.0))
def test_depreciation_reaches_its_mean_and_respects_its_cap(profile, fraction) -> None:
    shape, weights, cap = profile
    reachable = cap * weights[shape > 0].sum() / weights.sum()
    target = fraction * reachable
    retained, theta = scale_retained_share(shape, weights, target, cap=cap)
    depreciation = 1 - retained
    assert theta >= 0
    assert np.all(depreciation >= 0) and np.all(depreciation <= cap + 1e-12)
    assert (weights * depreciation).sum() / weights.sum() == pytest.approx(
        target, rel=1e-9, abs=1e-12
    )


def test_depreciation_refuses_a_mean_the_cap_forbids() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        depreciation_scale(np.array([1.0, 0.0]), np.array([1.0, 1.0]), 0.6, cap=1.0)
    assert depreciation_scale(np.array([1.0]), np.array([1.0]), 0.0, cap=0.8) == 0.0


def test_unit_cost_multiplies_its_parts() -> None:
    cost = unit_cost(
        np.array([80.0, 100.0]), np.array([1900.0, 2400.0]), 1.1, np.array([0.5, 1.0])
    )
    assert cost.tolist() == pytest.approx([80 * 1900 * 1.1 * 0.5, 100 * 2400 * 1.1])
    with pytest.raises(ValueError, match="retained share"):
        unit_cost(np.array([1.0]), np.array([1.0]), 1.0, np.array([1.5]))
    with pytest.raises(ValueError, match="Floor area"):
        unit_cost(np.array([-1.0]), np.array([1.0]), 1.0, np.array([1.0]))


@st.composite
def _patterns(draw):
    n = draw(st.integers(1, 25))
    raw = np.array(draw(st.lists(st.floats(0.0, 1e6), min_size=n, max_size=n)))
    cap = np.array(draw(st.lists(st.floats(1.0, 1e6), min_size=n, max_size=n)))
    raw[draw(st.integers(0, n - 1))] += 1.0  # some mass to scale
    fraction = draw(st.floats(0.0, 1.0))
    return raw, cap, fraction


@_SETTINGS
@given(_patterns())
def test_scale_cap_rescale_holds_the_total(pattern) -> None:
    raw, cap, fraction = pattern
    reachable = cap[raw > 0].sum()
    total = fraction * reachable
    out, capped = scale_cap_rescale(raw, cap, total)
    assert out.sum() == pytest.approx(total, rel=1e-9, abs=1e-6)
    assert np.all(out <= cap * (1 + 1e-12))
    assert np.array_equal(out[capped], cap[capped])
    free = ~capped & (raw > 0)
    if free.sum() >= 2:
        ratios = out[free] / raw[free]
        assert np.allclose(ratios, ratios[0], rtol=1e-9)


def test_scale_cap_rescale_refuses_a_total_above_the_caps() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        scale_cap_rescale(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 3.0)
    with pytest.raises(ValueError, match="No uncapped"):
        scale_cap_rescale(np.array([1.0, 0.0]), np.array([1.0, 5.0]), 3.0)


@_SETTINGS
@given(
    st.lists(st.integers(-1000, 1000), min_size=3, max_size=30).filter(
        lambda xs: len(set(xs)) > 1
    ),
    st.randoms(use_true_random=False),
)
def test_rank_correlation_is_spearmans(values, random) -> None:
    first = np.array(values, dtype=float)
    second = first.copy()
    random.shuffle(second)
    if len(set(second.tolist())) > 1:
        expected = stats.spearmanr(first, second).statistic
        assert rank_correlation(first, second) == pytest.approx(expected, abs=1e-12)
    assert rank_correlation(first, 2 * first + 1) == pytest.approx(1.0)


def test_rank_correlation_refuses_a_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        rank_correlation(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_log_ratio_flags_need_enough_references() -> None:
    values = {"A": 100.0, "B": 100.0, "C": 0.0}
    references = {
        "first": {"A": 50.0, "B": 90.0, "C": 10.0},
        "second": {"A": 200.0, "B": 40.0},
    }
    flagged = log_ratio_flags(values, references, threshold=0.4, minimum_references=2)
    assert flagged == {"A": ["first", "second"]}
    assert abs(math.log(100 / 40)) > 0.4  # B breaches one reference only
    assert log_ratio_flags(values, references, threshold=0.4, minimum_references=1) == {
        "A": ["first", "second"],
        "B": ["second"],
    }


@_SETTINGS
@given(
    st.lists(
        st.tuples(st.floats(-1e6, 1e6, allow_nan=False), st.floats(0.0, 1e4)),
        min_size=1,
        max_size=30,
    ).filter(lambda pairs: sum(w for _, w in pairs) > 0)
)
def test_convex_means_lie_between_their_values(pairs) -> None:
    values = np.array([v for v, _ in pairs])
    weights = np.array([w for _, w in pairs])
    mean = convex_mean(values, weights)
    carried = values[weights > 0]
    assert carried.min() <= mean <= carried.max()
