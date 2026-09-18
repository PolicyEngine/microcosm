"""Invented signed-total controls and an independent exhaustive KKT oracle."""

import itertools
from math import fsum

import numpy as np
import pytest

from microcosm.fit.signed_reconciliation import reconcile_signed_total


def project(q, anchor, **options):
    parameters = dict(
        components=("interest", "dividends", "property"),
        nonnegative=[True, True, False],
        scales=[1.0, 1.0, 1.0],
        atol=1e-10,
        rtol=1e-12,
    )
    parameters.update(options)
    return reconcile_signed_total(q, anchor, **parameters)


@pytest.mark.parametrize(
    "q,anchor,expected",
    [
        ([10.0, 4.0, 2.0], -6.0, [1.0, 0.0, -7.0]),
        ([8.0, 4.0, -3.0], 0.0, [5.0, 1.0, -6.0]),
        ([8.0, 4.0, -9.0], 6.0, [9.0, 5.0, -8.0]),
        ([-3.0, 2.0, 1.0], 5.0, [0.0, 3.0, 2.0]),
        ([1.0, 2.0, 8.0], -10.0, [0.0, 0.0, -10.0]),
    ],
)
def test_signed_and_zero_net_anchors_preserve_offsets(q, anchor, expected):
    result = project(q, anchor)
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.raw_draws, q)
    np.testing.assert_array_equal(result.adjustments, result.values - q)
    assert fsum(result.values) == anchor
    assert result.residuals == 0
    np.testing.assert_array_equal(result.active_bounds, result.values == 0)


def test_unequal_scales_allocate_adjustment_by_squared_scale():
    result = project([10, 10, 10], 44.0, scales=[1, 2, 3])
    np.testing.assert_allclose(result.values, [11, 14, 19], atol=1e-12)
    assert result.objective == pytest.approx(14.0)
    assert result.normalized_shift == pytest.approx(9.0)


def test_exactly_feasible_input_is_bitwise_identity_and_detached():
    q = np.array([-0.0, 2.0, -1.0])
    a, s, bound = (
        np.array(1.0),
        np.array([1.0, 2.0, 3.0]),
        np.array([True, True, False]),
    )
    original = [v.copy() for v in (q, a, s, bound)]
    result = project(q, a, scales=s, nonnegative=bound, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(result.values.view("uint64"), q.view("uint64"))
    assert result.objective == 0 and result.residuals == 0
    for before, after in zip(original, (q, a, s, bound), strict=True):
        np.testing.assert_array_equal(before, after)
    for returned, supplied in (
        (result.raw_draws, q),
        (result.values, q),
        (result.anchors, a),
        (result.scales, s),
        (result.nonnegative, bound),
    ):
        assert not np.shares_memory(returned, supplied)
    result.raw_draws[0] = 50
    result.nonnegative[0] = False
    assert q[0] == 0 and bound[0]


def test_roster_is_explicit_and_can_split_interest_without_tax_assumptions():
    result = project(
        [4, 6, 2, -9],
        3,
        components=("ordinary_interest", "account_interest", "dividend", "property"),
        nonnegative=[True, True, True, False],
        scales=[1, 2, 3, 4],
    )
    np.testing.assert_array_equal(result.values, [4, 6, 2, -9])
    assert result.components[1] == "account_interest"


def test_batches_match_individual_rows_including_multidimensional_batch():
    q = np.array([[[8, 4, 2], [-3, 2, 1]], [[8, 4, -3], [1, 2, 8]]], dtype=float)
    anchors = np.array([[-6, 5], [0, -10]], dtype=float)
    batch = project(q, anchors)
    assert batch.values.shape == (2, 2, 3) and batch.residuals.shape == (2, 2)
    for index in np.ndindex(anchors.shape):
        row = project(q[index], anchors[index])
        np.testing.assert_array_equal(batch.values[index], row.values)
        assert batch.objective[index] == row.objective


@pytest.mark.parametrize("shape", [(0, 3), (2, 0, 3)])
def test_empty_batches_preserve_shapes(shape):
    result = project(np.empty(shape), np.empty(shape[:-1]))
    for value in (
        result.values,
        result.raw_draws,
        result.adjustments,
        result.active_bounds,
    ):
        assert value.shape == shape
    for value in (result.residuals, result.objective, result.normalized_shift):
        assert value.shape == shape[:-1]


def test_any_declared_coordinate_can_be_signed_and_all_can_be_unrestricted():
    permuted = project(
        [2, 10, 4],
        -6,
        nonnegative=[False, True, True],
        components=("property", "interest", "dividends"),
    )
    np.testing.assert_array_equal(permuted.values, [-7, 1, 0])
    free = project([3, -7, 1], 0, nonnegative=[False, False, False])
    np.testing.assert_array_equal(free.values, [4, -6, 2])


def test_rescaling_all_scales_does_not_change_the_solution():
    base = project([8, 2, 3], -6, scales=[1, 2, 3])
    large = project([8, 2, 3], -6, scales=np.array([1, 2, 3]) * 1e100)
    np.testing.assert_allclose(base.values, large.values, atol=1e-12, rtol=1e-12)
    assert large.objective == pytest.approx(base.objective / 1e200, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    "q,anchor,options",
    [
        ([np.nan, 1, 1], 0, {}),
        ([np.inf, 1, 1], 0, {}),
        ([1, 1, 1], np.inf, {}),
        ([1, 1, 1], np.nan, {}),
        ([1, 1, 1], 0, {"scales": [0, 1, 1]}),
        ([1, 1, 1], 0, {"scales": [-1, 1, 1]}),
        ([1, 1, 1], 0, {"scales": [np.inf, 1, 1]}),
        ([1, 1, 1], 0, {"scales": [1, 1]}),
        ([1, 1, 1], 0, {"scales": [True, True, True]}),
        ([1, 1, 1], 0, {"nonnegative": [1, 1, 0]}),
        ([1, 1, 1], 0, {"nonnegative": [True, True, True]}),
        ([1, 1, 1], 0, {"components": ("i", "i", "p")}),
        ([1, 1, 1], 0, {"components": ("i", "d", " ")}),
        ([1, 1, 1], 0, {"components": ["i", "d", "p"]}),
        ([1, 1, 1], 0, {"atol": -1}),
        ([1, 1, 1], 0, {"rtol": np.inf}),
        ([1, 1, 1], 0, {"atol": [1e-6]}),
        ([1, 1, 1], [0], {}),
        ([[1, 1, 1]], 0, {}),
        ([], 0, {}),
        (1, 0, {}),
        (["1", "2", "3"], 0, {}),
        ([1j, 1, 1], 0, {}),
        ([True, False, True], 0, {}),
        (np.ma.array([1, 2, 3], mask=[False, True, False]), 0, {}),
        ([1, 2, 3], 0, {"nonnegative": np.ma.array([True, True, False])}),
    ],
)
def test_malformed_nonfinite_and_undeclared_inputs_refuse(q, anchor, options):
    with pytest.raises(ValueError, match="SIGNED_RECONCILIATION_"):
        project(q, anchor, **options)


@pytest.mark.parametrize(
    "q,anchor,options",
    [
        ([1e308, 1e308, -1e308], 1, {}),
        ([1, 2, 3], 0, {"scales": [1e-200, 1, 1]}),
        ([1, 2, 3], 0, {"scales": [1e-200, 1e-200, 1e-200]}),
        ([1, 2, 3], 10, {"rtol": 1e308}),
        ([1e16, 1e16, -1e16], 1, {"atol": 1e-12, "rtol": 0}),
    ],
)
def test_unstable_overflow_or_unrepresentable_sum_refuses(q, anchor, options):
    with pytest.raises(ValueError, match="SIGNED_RECONCILIATION_"):
        project(q, anchor, **options)


def _exhaustive_kkt_oracle(q, anchor, scales, bound):
    """Enumerate bound faces and solve each equality-constrained quadratic.

    This uses dense block KKT solves and objective comparison, independently of
    the production active-set order and common-shift arithmetic.
    """
    best = None
    bounded = np.flatnonzero(bound)
    for fixed_values in itertools.product((False, True), repeat=len(bounded)):
        fixed = np.zeros(len(q), dtype=bool)
        fixed[bounded] = fixed_values
        free = ~fixed
        hessian = np.diag(1 / scales[free] ** 2)
        count = int(free.sum())
        matrix = np.block(
            [[hessian, np.ones((count, 1))], [np.ones((1, count)), np.zeros((1, 1))]]
        )
        rhs = np.concatenate([hessian @ q[free], [anchor]])
        solved = np.linalg.solve(matrix, rhs)
        z = np.zeros(len(q))
        z[free] = solved[:-1]
        if np.any(z[bound] < -1e-9):
            continue
        objective = fsum(((z - q) / scales) ** 2)
        if best is None or objective < best[0]:
            best = objective, z, solved[-1]
    assert best is not None
    return best


def test_random_cases_match_independent_convex_kkt_oracle():
    rng = np.random.default_rng(8471)
    for _ in range(160):
        count = int(rng.integers(2, 6))
        q = rng.normal(0, 40, size=count)
        anchor = float(rng.normal(0, 50))
        scales = np.exp(rng.uniform(-2, 2, size=count))
        bound = rng.uniform(size=count) > 0.25
        bound[int(rng.integers(count))] = False
        result = reconcile_signed_total(
            q,
            anchor,
            components=tuple(f"part_{i}" for i in range(count)),
            nonnegative=bound,
            scales=scales,
            atol=1e-10,
            rtol=1e-12,
        )
        objective, expected, multiplier = _exhaustive_kkt_oracle(
            q, anchor, scales, bound
        )
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=2e-10)
        assert float(result.objective) == pytest.approx(objective, rel=2e-10, abs=1e-8)
        gradient = (result.values - q) / scales**2 + multiplier
        np.testing.assert_allclose(gradient[~result.active_bounds], 0, atol=1e-8)
        assert (gradient[result.active_bounds] >= -1e-8).all()
        assert abs(fsum(result.values) - anchor) <= 1e-10 + 1e-12 * abs(anchor)
