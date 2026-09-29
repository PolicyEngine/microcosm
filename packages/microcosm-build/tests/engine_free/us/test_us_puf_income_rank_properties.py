"""Property tests for the PUF income-rank and participation predictors (#982).

The rank share decides which PUF return's income vector each survey unit can
draw, so its invariants are stated and checked for arbitrary inputs:

- every share lies strictly between 0 and 1 when weights are positive;
- the weighted mean share is exactly one half (the midpoint convention);
- a higher value never gets a larger share, and tied values share one;
- reordering the rows or rescaling the weights changes nothing;
- rows with a missing value are excluded and do not move anyone else's rank;
- the vectorized implementation equals the definition computed pairwise.

Hypothesis is a workspace dev dependency; the clean-wheel lane installs only
pytest, where this module skips.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings

from microcosm.build.us_runtime import puf_support  # noqa: E402

_rank = puf_support._weighted_top_rank_share

# Few distinct values so ties are common; weights span survey-like magnitudes.
_values = st.lists(
    st.sampled_from([0.0, 1.0, 2.5, 10.0, 1e6, -3.0]), min_size=1, max_size=40
)


@st.composite
def _values_and_weights(draw, *, allow_zero_weight: bool = False):
    values = draw(_values)
    low = 0.0 if allow_zero_weight else 1e-3
    weights = draw(
        st.lists(
            st.floats(min_value=low, max_value=5e3, allow_nan=False),
            min_size=len(values),
            max_size=len(values),
        )
    )
    if sum(weights) <= 0:
        weights[0] = 1.0
    return np.asarray(values), np.asarray(weights)


def _reference(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The definition, pairwise: weight above plus half the tie group, over W."""

    total = weights.sum()
    return np.asarray(
        [
            (weights[values > value].sum() + weights[values == value].sum() / 2) / total
            for value in values
        ]
    )


@settings(max_examples=300, deadline=None)
@given(_values_and_weights(allow_zero_weight=True))
def test_rank_equals_the_pairwise_definition(case) -> None:
    values, weights = case
    np.testing.assert_allclose(_rank(values, weights), _reference(values, weights))


@settings(max_examples=300, deadline=None)
@given(_values_and_weights())
def test_rank_is_bounded_centred_monotone_and_tie_consistent(case) -> None:
    values, weights = case
    shares = _rank(values, weights)
    assert np.all((shares > 0) & (shares < 1))
    assert np.average(shares, weights=weights) == pytest.approx(0.5, abs=1e-9)
    for i in range(len(values)):
        for j in range(len(values)):
            if values[i] > values[j]:
                assert shares[i] < shares[j]
            elif values[i] == values[j]:
                assert shares[i] == shares[j]


@settings(max_examples=200, deadline=None)
@given(_values_and_weights(), st.randoms(use_true_random=False))
def test_rank_ignores_row_order_and_weight_scale(case, random) -> None:
    values, weights = case
    shares = _rank(values, weights)
    order = list(range(len(values)))
    random.shuffle(order)
    np.testing.assert_allclose(_rank(values[order], weights[order]), shares[order])
    np.testing.assert_allclose(_rank(values, weights * 0.5), shares)
    np.testing.assert_allclose(_rank(values, weights * 1_000.0), shares)


@settings(max_examples=200, deadline=None)
@given(_values_and_weights(), st.data())
def test_missing_rows_are_excluded_without_moving_the_rest(case, data) -> None:
    values, weights = case
    missing = np.asarray(
        data.draw(st.lists(st.booleans(), min_size=len(values), max_size=len(values)))
    )
    if missing.all():
        missing[0] = False
    with_gaps = np.where(missing, np.nan, values)
    shares = _rank(with_gaps, weights)
    assert np.isnan(shares[missing]).all()
    np.testing.assert_allclose(
        shares[~missing], _rank(values[~missing], weights[~missing])
    )


_amounts = st.one_of(
    st.just(0.0),
    st.floats(min_value=-1e7, max_value=1e7, allow_nan=False),
    st.just(float("nan")),
)


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(_amounts, _amounts), min_size=1, max_size=30))
def test_earnings_flag_is_any_nonzero_and_keeps_missing(rows) -> None:
    wages = np.asarray([row[0] for row in rows])
    self_employment = np.asarray([row[1] for row in rows])
    flag = puf_support._earnings_indicator([wages, self_employment])
    missing = np.isnan(wages) | np.isnan(self_employment)
    assert np.isnan(flag[missing]).all()
    expected = ((wages != 0) | (self_employment != 0)).astype(float)
    np.testing.assert_array_equal(flag[~missing], expected[~missing])
