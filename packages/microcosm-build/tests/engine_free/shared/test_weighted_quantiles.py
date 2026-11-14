"""Weighted quantile knots (microcosm#1524).

The reference functions below are the UK high-value home stock library's own
code as it stood on main at a575057bb, before it moved here. Matching them
bit for bit is what keeps the committed knots unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.weighted_quantiles import (
    check_knots,
    grouped_knots,
    knot_mean,
    weighted_quantiles,
)

_SETTINGS = settings(max_examples=80, deadline=None)
_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def _reference_weighted_quantiles(values, weights, grid):
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    midpoints = (cumulative - weights / 2.0) / total
    return np.interp(
        np.asarray(grid, dtype=float),
        midpoints,
        values,
        left=values[0],
        right=values[-1],
    )


def _reference_value_knots(values, weight, codes, *, cap, minimum_count, grid):
    values = np.minimum(np.asarray(values, dtype=float), float(cap))
    weight = np.asarray(weight, dtype=float)
    codes = np.asarray(codes).astype(str)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True]
    )
    knots, suppressed = [], []
    for begin, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        code = sorted_codes[begin]
        rows = order[begin:stop]
        count = int(stop - begin)
        if count < minimum_count:
            suppressed.append(str(code))
            continue
        quantiles = _reference_weighted_quantiles(values[rows], weight[rows], grid)
        knots.append(
            (
                str(code),
                count,
                float(weight[rows].sum()),
                tuple(float(round(q, 2)) for q in quantiles),
            )
        )
    return knots, suppressed


_values = st.lists(
    st.floats(min_value=0, max_value=5e7, allow_nan=False), min_size=1, max_size=60
)


@st.composite
def _samples(draw):
    values = draw(_values)
    weights = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=50, allow_nan=False),
            min_size=len(values),
            max_size=len(values),
        )
    )
    codes = draw(
        st.lists(
            st.sampled_from(["E1", "E2", "E3", "W1"]),
            min_size=len(values),
            max_size=len(values),
        )
    )
    return np.array(values), np.array(weights), np.array(codes)


@_SETTINGS
@given(_samples())
def test_weighted_quantiles_match_the_reference_bit_for_bit(sample) -> None:
    values, weights, _ = sample
    ours = weighted_quantiles(values, weights, _GRID)
    reference = _reference_weighted_quantiles(values, weights, _GRID)
    assert ours.tobytes() == reference.tobytes()
    assert np.all(np.diff(ours) >= 0)
    assert values.min() <= ours.min() and ours.max() <= values.max()


@_SETTINGS
@given(_samples(), st.integers(min_value=1, max_value=8))
def test_grouped_knots_match_the_reference(sample, minimum) -> None:
    values, weights, codes = sample
    knots, suppressed = grouped_knots(
        values, weights, codes, grid=_GRID, cap=2e7, minimum_count=minimum
    )
    reference, reference_suppressed = _reference_value_knots(
        values, weights, codes, cap=2e7, minimum_count=minimum, grid=_GRID
    )
    assert [(k.code, k.count, k.weight, k.quantiles) for k in knots] == reference
    assert suppressed == reference_suppressed
    for k in knots:
        check_knots(_GRID, k.quantiles, cap=2e7)


def test_weighted_quantiles_refuse_what_they_cannot_weigh() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        weighted_quantiles(np.array([]), np.array([]), _GRID)
    with pytest.raises(ValueError, match="positive total"):
        weighted_quantiles(np.array([1.0, 2.0]), np.array([0.0, 0.0]), _GRID)


def test_knot_mean_is_exact_for_a_uniform_distribution() -> None:
    # The quantile function of a uniform on [a, b] is linear, so the knots
    # carry its mean exactly on any grid.
    grid = np.array(_GRID)
    assert knot_mean(grid, 100 + 50 * grid) == pytest.approx(125.0, abs=1e-12)
    with pytest.raises(ValueError, match="match"):
        knot_mean(grid, grid[:-1])


@pytest.mark.parametrize(
    ("grid", "quantiles", "message"),
    [
        ((0.1, 1.0), (1.0, 2.0), "run from 0 to 1"),
        ((0.0, 0.5, 0.5, 1.0), (1.0, 2.0, 3.0, 4.0), "rise strictly"),
        ((0.0, 1.0), (1.0,), "match"),
        ((0.0, 1.0), (1.0, float("nan")), "finite"),
        ((0.0, 1.0), (-1.0, 2.0), r"\[0, cap\]"),
        ((0.0, 1.0), (1.0, 30.0), r"\[0, cap\]"),
        ((0.0, 1.0), (2.0, 1.0), "never fall"),
    ],
)
def test_check_knots_refuses_unusable_knots(grid, quantiles, message) -> None:
    with pytest.raises(ValueError, match=message):
        check_knots(grid, quantiles, cap=20.0)
