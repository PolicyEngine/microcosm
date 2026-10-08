"""Declared weighted transport distributions, without country engines."""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum, log10

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.frame import transport
from microcosm.frame.transport import _pareto_inverse, quantile_map


def _bands():
    # Synthetic test distribution; these are not country target values.
    return [
        {"lower": 0.0, "upper": 10.0, "share": 0.25},
        {"lower": 10.0, "upper": 30.0, "share": 0.5},
        {"lower": 30.0, "upper": 90.0, "share": 0.25},
    ]


def _one_entity(values, total):
    # One entity whose total maps to the given value (the band's midpoint).
    bands = [{"lower": 0.0, "upper": 2 * abs(total), "share": 1}]
    return quantile_map(
        values,
        np.ones(len(values)),
        {"positive": bands, "negative": bands},
        interpolation="uniform",
        group={"entity_ids": np.zeros(len(values), dtype=int)},
    )


_samples = st.lists(
    st.tuples(st.integers(-100, 100), st.integers(1, 40), st.integers(0, 2)),
    min_size=1,
    max_size=40,
)


def _reference_uniform(values, weights, bands, components):
    """Deliberately quadratic independent definition of pooled weighted ranks."""
    result = []
    for value, component in zip(values, components, strict=True):
        if value == 0:
            result.append(0.0)
            continue
        peers = [
            (abs(peer), weight)
            for peer, weight, peer_component in zip(
                values, weights, components, strict=True
            )
            if peer_component == component and peer * value > 0
        ]
        rank = (
            fsum(weight for peer, weight in peers if peer < abs(value))
            + fsum(weight for peer, weight in peers if peer == abs(value)) / 2
        ) / fsum(weight for _, weight in peers)
        start = 0.0
        for band in bands:
            stop = start + band["share"]
            if rank < stop:
                within = (rank - start) / band["share"]
                result.append(
                    np.sign(value)
                    * (band["lower"] + within * (band["upper"] - band["lower"]))
                )
                break
            start = stop
    return np.array(result)


def _reference_truncated_pareto(lower, upper, alpha, rank):
    """Evaluate the declared CDF directly, independently of float64 arithmetic."""
    with localcontext() as context:
        context.prec = 100
        lower, upper, alpha = map(Decimal.from_float, (lower, upper, alpha))
        rank = Decimal(rank.numerator) / Decimal(rank.denominator)
        tail = (lower / upper) ** alpha
        return float(lower * (1 - rank * (1 - tail)) ** (-1 / alpha))


def _rank_condition(lower, upper, alpha, rank):
    """d log(x) / d log(rank) of the truncated inverse at ``rank``.

    ``quantile_map`` rounds each rank to float64, so its output can differ from
    the exact-rank reference by about this factor times 2**-53.
    """
    with localcontext() as context:
        context.prec = 100
        lower, upper, alpha = map(Decimal.from_float, (lower, upper, alpha))
        rank = Decimal(rank.numerator) / Decimal(rank.denominator)
        mass = 1 - (lower / upper) ** alpha
        return float(rank * mass / (alpha * (1 - rank * mass)))


@st.composite
def _truncated_pareto_cases(draw):
    lower_exponent = draw(st.integers(-200, 200))
    width_exponent = draw(st.integers(1, min(400, 300 - lower_exponent)))
    alpha = draw(
        st.one_of(
            st.sampled_from([1e-18, 1e-16, 1e-12, 0.1, 1.0, 50.0]),
            st.floats(-18, log10(50), allow_nan=False, allow_infinity=False).map(
                lambda exponent: min(50.0, 10**exponent)
            ),
        )
    )
    weights = draw(st.tuples(st.integers(1, 100), st.integers(1, 100)))
    return (
        10.0**lower_exponent,
        10.0 ** (lower_exponent + width_exponent),
        alpha,
        weights,
    )


def test_synthetic_distribution_round_trip():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mapped = quantile_map(values, np.ones(4), _bands(), interpolation="uniform")
    np.testing.assert_array_equal(mapped, [5.0, 15.0, 25.0, 60.0])
    np.testing.assert_array_equal(
        quantile_map(mapped, np.ones(4), _bands(), interpolation="uniform"), mapped
    )
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0])


def test_tied_values_share_a_pooled_midpoint_rank():
    values = [2.0, 1.0, 2.0]
    weights = [1.0, 1.0, 2.0]
    mapped = quantile_map(values, weights, _bands(), interpolation="uniform")
    np.testing.assert_array_equal(mapped, [25.0, 5.0, 25.0])
    np.testing.assert_array_equal(
        quantile_map(mapped, weights, _bands(), interpolation="uniform"), mapped
    )


def test_zero_and_sign_distributions_are_independent():
    bands = {
        "positive": [{"lower": 10.0, "upper": 20.0, "share": 1.0}],
        "negative": [{"lower": 30.0, "upper": 50.0, "share": 1.0}],
    }
    mapped = quantile_map(
        [-1, -2, 0, 1, 2], [1, 1, 100, 1, 1], bands, interpolation="uniform"
    )
    np.testing.assert_array_equal(mapped, [-35.0, -45.0, 0.0, 12.5, 17.5])


def test_components_are_ranked_independently():
    mapped = quantile_map(
        [1, 2, 100, 200],
        [1, 1, 3, 3],
        _bands(),
        interpolation="uniform",
        group=["a", "a", "b", "b"],
    )
    np.testing.assert_array_equal(mapped, [10.0, 30.0, 10.0, 30.0])


def test_entity_aggregation_maps_one_entity_weight_then_rescales_members():
    mapped = quantile_map(
        [1, 3, 2, 0, 0],
        [2, 2, 1, 1, 4],
        [{"lower": 0, "upper": 90, "share": 1}],
        interpolation="uniform",
        group={"entity_ids": ["h1", "h1", "h2", "h2", "h3"]},
    )
    np.testing.assert_allclose(mapped, [15.0, 45.0, 15.0, 0.0, 0.0])
    assert mapped[1] / mapped[0] == 3


def test_aggregation_accepts_components_and_preserves_cancelling_members():
    mapped = quantile_map(
        [3, -1, 2, -2, 5, 0],
        np.ones(6),
        [{"lower": 10, "upper": 30, "share": 1}],
        interpolation="uniform",
        group={
            "entity_ids": ["a", "a", "b", "b", "c", "c"],
            "components": [0, 0, 0, 0, 1, 1],
        },
    )
    np.testing.assert_array_equal(mapped, [30.0, -10.0, 2.0, -2.0, 20.0, 0.0])


@pytest.mark.parametrize("upper", [None, 100.0])
def test_pareto_top_band_uses_explicit_alpha(upper):
    bands = [{"lower": 10.0, "upper": upper, "share": 1.0}]
    mapped = quantile_map(
        [1.0, 2.0], [1, 1], bands, interpolation={"method": "pareto", "alpha": 2}
    )
    tail = 0.0 if upper is None else (10 / upper) ** 2
    expected = 10 / np.sqrt(1 - np.array([0.25, 0.75]) * (1 - tail))
    np.testing.assert_allclose(mapped, expected)
    np.testing.assert_array_equal(
        quantile_map(
            mapped, [1, 1], bands, interpolation={"method": "pareto", "alpha": 2}
        ),
        mapped,
    )


@pytest.mark.parametrize("alpha", [1e-18, 1e-16])
def test_truncated_pareto_small_alpha_midpoint_matches_decimal(alpha):
    mapped = quantile_map(
        [1],
        [1],
        [{"lower": 10, "upper": 100, "share": 1}],
        interpolation={"method": "pareto", "alpha": alpha},
    )
    expected = _reference_truncated_pareto(10.0, 100.0, alpha, Fraction(1, 2))
    np.testing.assert_allclose(mapped, [expected], rtol=1e-12, atol=0)


@given(_truncated_pareto_cases())
@example(case=(10.0, 100.0, 1e-18, (1, 1)))
@example(case=(1e-200, 1e200, 1e-18, (1, 3)))
@example(case=(1e100, 1e300, 50.0, (100, 1)))
@example(case=(1e-200, 1e200, 10**-2.1, (99, 1)))
@settings(max_examples=100, deadline=None)
def test_truncated_pareto_matches_high_precision_reference(case):
    lower, upper, alpha, weights = case
    mapped = quantile_map(
        [1, 2],
        weights,
        [{"lower": lower, "upper": upper, "share": 1}],
        interpolation={"method": "pareto", "alpha": alpha},
    )
    total = sum(weights)
    ranks = (
        Fraction(weights[0], 2 * total),
        Fraction(2 * weights[0] + weights[1], 2 * total),
    )
    expected = [
        _reference_truncated_pareto(lower, upper, alpha, rank) for rank in ranks
    ]
    for value, reference, rank in zip(mapped, expected, ranks, strict=True):
        tolerance = 1e-12 + 4 * _rank_condition(lower, upper, alpha, rank) * 2.0**-53
        np.testing.assert_allclose(value, reference, rtol=tolerance, atol=0)


@pytest.mark.parametrize(
    "alpha", [1e-18, 1e-24, 1e-100, 1e-310, 1e-315, 1e-320, 1e-322, 5e-324]
)
def test_truncated_pareto_tends_to_log_uniform_as_alpha_tends_to_zero(alpha):
    mapped = quantile_map(
        [1],
        [1],
        [{"lower": 10, "upper": 100, "share": 1}],
        interpolation={"method": "pareto", "alpha": alpha},
    )
    np.testing.assert_allclose(mapped, [10 * (100 / 10) ** 0.5], rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha", [1e-310, 1e-320, 5e-324])
def test_subnormal_alpha_quartiles_follow_the_log_uniform_limit(alpha):
    # Below alpha * w = 2**-60 the inverse is the closed-form limit
    # lower * (upper / lower)**rank; expm1 of a subnormal argument is not.
    mapped = quantile_map(
        [1, 2, 3, 4],
        [1, 1, 1, 1],
        [{"lower": 10, "upper": 100, "share": 1}],
        interpolation={"method": "pareto", "alpha": alpha},
    )
    expected = [10 * 10 ** (rank / 8) for rank in (1, 3, 5, 7)]
    np.testing.assert_allclose(mapped, expected, rtol=1e-12, atol=0)


def test_ranks_near_one_keep_their_accuracy():
    # 1 - fraction * mass cancels as fraction -> 1 with mass near 1; the inverse
    # computes log((1 - fraction) + fraction * exp(-alpha * w)) there instead.
    fraction = 1 - 2.0**-53
    lower, upper, alpha = 1e-10, 1e20, 0.5
    mapped = _pareto_inverse(np.array([fraction]), lower, upper, alpha)
    with localcontext() as context:
        context.prec = 100
        lo, hi, a, f = map(Decimal.from_float, (lower, upper, alpha, fraction))
        expected = float(lo * (1 - f * (1 - (lo / hi) ** a)) ** (-1 / a))
    np.testing.assert_allclose(mapped, [expected], rtol=1e-13, atol=0)


def test_truncated_values_never_exceed_the_band():
    mapped = quantile_map(
        [1, 2],
        [1e16, 1],
        [{"lower": 1e-200, "upper": 1e100, "share": 1}],
        interpolation={"method": "pareto", "alpha": 1e-30},
    )
    assert (mapped <= 1e100).all() and (mapped >= 1e-200).all()


def test_untruncated_band_reaches_finite_values_through_log_space():
    # lower * exp(offset) overflows here although the true quantile is finite.
    mapped = quantile_map(
        [1, 2],
        [1, 1],
        [{"lower": 1e-300, "upper": None, "share": 1}],
        interpolation={"method": "pareto", "alpha": 1e-3},
    )
    expected = [
        np.exp(np.log(1e-300) - 1000 * np.log1p(-rank)) for rank in (0.25, 0.75)
    ]
    np.testing.assert_allclose(mapped, expected, rtol=1e-12, atol=0)


def test_pareto_only_changes_the_top_band():
    bands = [
        {"lower": 0, "upper": 10, "share": 0.5},
        {"lower": 10, "upper": None, "share": 0.5},
    ]
    mapped = quantile_map(
        [1, 2], [1, 1], bands, interpolation={"method": "pareto", "alpha": 1}
    )
    np.testing.assert_array_equal(mapped, [5.0, 20.0])


@given(_samples)
@settings(max_examples=100, deadline=None)
def test_rank_sign_zero_determinism_and_idempotence_properties(samples):
    values, weights, components = map(np.array, zip(*samples, strict=True))
    mapped = quantile_map(
        values, weights, _bands(), interpolation="uniform", group=components
    )
    np.testing.assert_array_equal(np.sign(mapped), np.sign(values))
    np.testing.assert_array_equal(mapped[values == 0], values[values == 0])
    np.testing.assert_array_equal(
        quantile_map(
            values, weights, _bands(), interpolation="uniform", group=components
        ),
        mapped,
    )
    np.testing.assert_array_equal(
        quantile_map(
            mapped, weights, _bands(), interpolation="uniform", group=components
        ),
        mapped,
    )
    for component in set(components):
        for sign in (-1, 1):
            selected = (components == component) & (values * sign > 0)
            order = np.argsort(values[selected], kind="stable")
            assert np.all(np.diff(mapped[selected][order]) >= 0)


@given(_samples)
@settings(max_examples=100, deadline=None)
def test_weighted_band_shares_within_indivisible_tie_tolerance(samples):
    values, weights, components = map(np.array, zip(*samples, strict=True))
    mapped = quantile_map(
        values, weights, _bands(), interpolation="uniform", group=components
    )
    for component in set(components):
        for sign in (-1, 1):
            selected = (components == component) & (values * sign > 0)
            if not selected.any():
                continue
            originals, magnitudes, mass = (
                abs(values[selected]),
                abs(mapped[selected]),
                weights[selected],
            )
            total = mass.sum()
            tolerance = (
                max(mass[originals == value].sum() for value in set(originals)) / total
            )
            for position, band in enumerate(_bands()):
                inside = (magnitudes >= band["lower"]) & (
                    magnitudes <= band["upper"]
                    if position == 2
                    else magnitudes < band["upper"]
                )
                share = mass[inside].sum() / total
                assert abs(share - band["share"]) <= tolerance + 1e-12


@given(_samples)
@settings(max_examples=100, deadline=None)
def test_uniform_map_matches_independent_reference(samples):
    values, weights, components = map(np.array, zip(*samples, strict=True))
    mapped = quantile_map(
        values, weights, _bands(), interpolation="uniform", group=components
    )
    np.testing.assert_allclose(
        mapped,
        _reference_uniform(values, weights, _bands(), components),
        rtol=1e-12,
        atol=1e-12,
    )


@given(_samples, st.integers(1, 100))
@settings(max_examples=80, deadline=None)
def test_mapping_preserves_weights_and_is_invariant_to_their_common_scale(
    samples, factor
):
    values, weights, components = map(np.array, zip(*samples, strict=True))
    original_weights = weights.copy()
    first = quantile_map(
        values, weights, _bands(), interpolation="uniform", group=components
    )
    second = quantile_map(
        values, weights * factor, _bands(), interpolation="uniform", group=components
    )
    np.testing.assert_allclose(second, first, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(weights, original_weights)


@given(_samples)
@settings(max_examples=80, deadline=None)
def test_permuting_members_preserves_mapped_values(samples):
    values, weights, components = map(np.array, zip(*samples, strict=True))
    first = quantile_map(
        values, weights, _bands(), interpolation="uniform", group=components
    )
    second = quantile_map(
        values[::-1],
        weights[::-1],
        _bands(),
        interpolation="uniform",
        group=components[::-1],
    )
    np.testing.assert_allclose(second[::-1], first, rtol=1e-12, atol=1e-12)


@given(
    st.lists(
        st.lists(st.integers(0, 50), min_size=1, max_size=5), min_size=1, max_size=12
    )
)
# Members [1, 2] whose total maps to a total the largest member alone cannot
# reach; the second is the counterexample #1130's engine-free CI job found.
@example(entities=[[1], [1], [1, 2]])
@example(entities=[[1], [1], [1], [1], [1], [1], [1], [4], [1, 2]])
@settings(max_examples=100, deadline=None)
def test_aggregate_pro_rata_totals_and_idempotence(entities):
    values = np.array([value for members in entities for value in members], dtype=float)
    entity_ids = np.array(
        [entity for entity, members in enumerate(entities) for _ in members]
    )
    group = {"entity_ids": entity_ids}
    mapped = quantile_map(
        values, np.ones(len(values)), _bands(), interpolation="uniform", group=group
    )
    totals = np.array([fsum(members) for members in entities])
    expected = quantile_map(
        totals, np.ones(len(entities)), _bands(), interpolation="uniform"
    )
    actual = np.array(
        [fsum(mapped[entity_ids == entity]) for entity in range(len(entities))]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        quantile_map(
            mapped, np.ones(len(values)), _bands(), interpolation="uniform", group=group
        ),
        mapped,
        rtol=1e-12,
        atol=1e-12,
    )
    for entity, total in enumerate(totals):
        if total:
            selected = (entity_ids == entity) & (values != 0)
            np.testing.assert_allclose(
                mapped[selected] / values[selected], expected[entity] / total
            )


@given(st.integers(2, 10000), st.integers(1, 9))
@settings(max_examples=100, deadline=None)
def test_tied_entity_totals_remain_tied_on_repeated_mapping(total, divider):
    split = max(1, total // divider)
    values = [float(split), float(total - split), float(total)]
    group = {"entity_ids": [0, 0, 1]}
    mapped = quantile_map(
        values, [1, 1, 1], _bands(), interpolation="uniform", group=group
    )
    np.testing.assert_allclose(
        quantile_map(mapped, [1, 1, 1], _bands(), interpolation="uniform", group=group),
        mapped,
        rtol=1e-12,
        atol=1e-12,
    )


@given(
    st.lists(st.integers(1, 100), min_size=1, max_size=30),
    st.floats(0.5, 5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=80, deadline=None)
def test_pareto_rank_and_idempotence_properties(values, alpha):
    bands = [{"lower": 10, "upper": None, "share": 1}]
    interpolation = {"method": "pareto", "alpha": alpha}
    mapped = quantile_map(
        values, np.ones(len(values)), bands, interpolation=interpolation
    )
    order = np.argsort(values)
    assert np.all(np.diff(mapped[order]) >= 0)
    np.testing.assert_array_equal(
        quantile_map(mapped, np.ones(len(values)), bands, interpolation=interpolation),
        mapped,
    )


@pytest.mark.parametrize(
    "values, weights",
    [
        ([1, np.nan], [1, 1]),
        ([np.inf], [1]),
        ([1], [0]),
        ([1], [-1]),
        ([1], [np.inf]),
        ([[1]], [[1]]),
        ([1, 2], [1]),
    ],
)
def test_invalid_values_and_weights_fail_closed(values, weights):
    with pytest.raises(ValueError):
        quantile_map(values, weights, _bands(), interpolation="uniform")


@pytest.mark.parametrize(
    "bands",
    [
        [],
        [{"lower": -1, "upper": 1, "share": 1}],
        [{"lower": 1, "upper": 1, "share": 1}],
        [{"lower": 0, "upper": 1, "share": 0}],
        [{"lower": 0, "upper": 1, "share": 0.5}],
        [
            {"lower": 0, "upper": 1, "share": 0.5},
            {"lower": 2, "upper": 3, "share": 0.5},
        ],
        [{"lower": 0, "upper": None, "share": 1}],
        {"positive": _bands()},
        [{"lower": 0, "upper": 1, "share": 1, "other": 3}],
    ],
)
def test_invalid_target_band_contract_fails_closed(bands):
    with pytest.raises(ValueError):
        quantile_map([1], [1], bands, interpolation="uniform")


@pytest.mark.parametrize(
    "interpolation",
    [
        "pareto",
        "normal",
        {"method": "pareto", "alpha": 0},
        {"method": "pareto", "alpha": np.inf},
        {"method": "uniform", "alpha": 2},
        {"method": "pareto"},
    ],
)
def test_invalid_interpolation_fails_closed(interpolation):
    with pytest.raises(ValueError):
        quantile_map([1], [1], _bands(), interpolation=interpolation)


def test_pareto_requires_a_positive_top_lower_bound():
    with pytest.raises(ValueError, match="positive lower"):
        quantile_map(
            [1],
            [1],
            [{"lower": 0, "upper": None, "share": 1}],
            interpolation={"method": "pareto", "alpha": 2},
        )


@pytest.mark.parametrize(
    "group",
    [
        [1],
        [None, 1],
        [[1], [2]],
        {"components": [1, 2]},
        {"entity_ids": [1, 1], "components": [1, 2]},
        {"entity_ids": [1, 2], "other": [1, 2]},
    ],
)
def test_invalid_group_contract_fails_closed(group):
    with pytest.raises(ValueError):
        quantile_map([1, 2], [1, 1], _bands(), interpolation="uniform", group=group)


def test_aggregate_rejects_multiple_weights_for_one_entity():
    with pytest.raises(ValueError, match="one entity weight"):
        quantile_map(
            [1, 2],
            [1, 2],
            _bands(),
            interpolation="uniform",
            group={"entity_ids": [1, 1]},
        )


def test_empty_population_is_supported():
    assert quantile_map([], [], _bands(), interpolation="uniform").shape == (0,)
    assert quantile_map(
        [], [], _bands(), interpolation="uniform", group={"entity_ids": []}
    ).shape == (0,)


def test_aggregate_refuses_severe_cancellation_instead_of_distorting_small_member():
    values = [1e16, -(1e16 - 2), 1]
    bands = [{"lower": 0, "upper": 36, "share": 1}]
    with pytest.raises(ValueError, match="float64"):
        quantile_map(
            values,
            [1, 1, 1],
            bands,
            interpolation="uniform",
            group={"entity_ids": [0, 0, 0]},
        )


def test_float64_rank_collapse_is_refused():
    bands = [{"lower": 1e16, "upper": np.nextafter(1e16, np.inf), "share": 1}]
    with pytest.raises(ValueError, match="distinct ranks"):
        quantile_map([1, 2, 3], [1, 1, 1], bands, interpolation="uniform")


def test_unrepresentable_weight_ratio_is_refused():
    with pytest.raises(ValueError, match="Weight ratios"):
        quantile_map([1, 2], [1e308, 1e-308], _bands(), interpolation="uniform")


def test_pro_rata_mapping_avoids_overflow_in_the_common_scale_factor():
    bands = [{"lower": 0, "upper": 1e308, "share": 1}]
    mapped = quantile_map(
        [1e-308, 1e-308],
        [1, 1],
        bands,
        interpolation="uniform",
        group={"entity_ids": [0, 0]},
    )
    np.testing.assert_array_equal(mapped, [2.5e307, 2.5e307])


def test_pro_rata_correction_beyond_float64_rounding_is_refused(monkeypatch):
    # No correctly rounded entity total has been found to move the anchor past
    # the bound, so a mis-rounded total stands in for a pro rata step that
    # loses more than rounding. Correcting the anchor still restores the
    # exact total, so only the correction bound refuses the result.
    members = [1.0, 3.0]

    def misrounded(items):
        items = list(items)
        total = fsum(items)
        return total * (1 + 1e-9) if items == members else total

    monkeypatch.setattr(transport, "fsum", misrounded)
    with pytest.raises(ValueError, match="correction exceeds float64 rounding"):
        quantile_map(
            members,
            [1, 1],
            [{"lower": 0, "upper": 120, "share": 1}],
            interpolation="uniform",
            group={"entity_ids": [0, 0]},
        )


def test_pro_rata_moves_one_other_member_when_the_largest_cannot_reach_the_total():
    # [1, 2] scaled to a total of 0.21: every value of the larger member
    # leaves the sum halfway between 0.21 and a float64 neighbour, and
    # rounding to even never lands on 0.21, so the smaller member moves by
    # one unit in the last place.
    values = np.array([1.0, 2.0])
    mapped = quantile_map(
        values,
        [1, 1],
        [{"lower": 0, "upper": 0.42, "share": 1}],
        interpolation="uniform",
        group={"entity_ids": [0, 0]},
    )
    pro_rata = values / 3 * 0.21
    assert fsum(mapped) == 0.21
    assert mapped[0] == np.nextafter(pro_rata[0], np.inf)
    assert mapped[1] == pro_rata[1]


_amounts = st.one_of(
    st.integers(1, 50).map(float), st.integers(1, 10**6).map(lambda cents: cents / 100)
)


@given(
    # Whole amounts and amounts in cents. Entities of one to three members
    # produce the halfway ties most often; larger ones cover household sizes.
    st.lists(
        st.one_of(
            st.lists(_amounts, min_size=1, max_size=3),
            st.lists(_amounts, min_size=4, max_size=16),
        ),
        min_size=1,
        max_size=20,
    ),
    st.sampled_from([1.0, -1.0]),
)
@example(entities=[[1.0], [1.0], [1.0, 2.0]], sign=1.0)
@example(entities=[[1.0], [1.0], [1.0, 2.0]], sign=-1.0)
@example(
    entities=[[1.0], [2.0], [3.0], [1.62, 1.53], [5.0], [6.0], [7.0], [8.0], [9.0]],
    sign=1.0,
)
@settings(max_examples=100, deadline=None)
def test_same_sign_members_always_retain_exact_entity_totals(entities, sign):
    values = sign * np.array([value for members in entities for value in members])
    entity_ids = np.array(
        [entity for entity, members in enumerate(entities) for _ in members]
    )
    mapped = quantile_map(
        values,
        np.ones(len(values)),
        _bands(),
        interpolation="uniform",
        group={"entity_ids": entity_ids},
    )
    totals = np.array([fsum(values[entity_ids == e]) for e in range(len(entities))])
    expected = quantile_map(
        totals, np.ones(len(entities)), _bands(), interpolation="uniform"
    )
    for entity, total in enumerate(totals):
        selected = entity_ids == entity
        assert fsum(mapped[selected]) == expected[entity]
        pro_rata = values[selected] / total * expected[entity]
        moved = np.abs(mapped[selected] - pro_rata) / np.spacing(np.abs(pro_rata))
        # Only the largest member absorbs more than one unit in the last place.
        assert (np.delete(moved, np.argmax(np.abs(pro_rata))) <= 1).all()


def test_pro_rata_reaches_a_power_of_two_total_far_from_the_residual_step():
    # Thirteen members scaled to 8192: the total's spacing halves below it,
    # and the residual step from the rounded sum misses by more than the two
    # one-unit steps can recover. The largest member alone can still reach
    # the total, so every other member keeps its exact pro rata value.
    values = [575.99, 442.46, 26.4, 365.04, 974.36, 860.0, 783.83]
    values += [715.86, 762.91, 803.26, 810.82, 704.65, 512.47]
    mapped = _one_entity(values, 8192.0)
    pro_rata = np.array(values) / fsum(values) * 8192.0
    largest = np.argmax(values)
    assert fsum(mapped) == 8192.0
    np.testing.assert_array_equal(
        np.delete(mapped, largest), np.delete(pro_rata, largest)
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_pro_rata_moves_the_smallest_member_when_the_tie_sits_a_binade_up(sign):
    # [7, 3.57] scaled to 1.51: the larger member falls just below 1, but the
    # anchor the total needs sits halfway between two values above 1, where
    # the other member's spacing is no finer than the larger member's own.
    values = sign * np.array([7.0, 3.57])
    mapped = _one_entity(values, sign * 1.51)
    pro_rata = values / fsum(values) * (sign * 1.51)
    assert fsum(mapped) == sign * 1.51
    assert (np.abs(mapped - pro_rata) <= np.spacing(np.abs(pro_rata))).all()


def test_mixed_sign_members_mapped_once_map_to_themselves_again():
    first = _one_entity([30.59, -90.38], -763.16)
    assert fsum(first) == -763.16
    np.testing.assert_array_equal(_one_entity(first, -763.16), first)


def test_cancelling_members_refuse_an_unrepresentable_aggregate_total():
    bands = [{"lower": 10, "upper": 30, "share": 1}]
    with pytest.raises(ValueError, match="exact totals"):
        quantile_map(
            [1e16, -9999999999999998.0],
            [1, 1],
            bands,
            interpolation="uniform",
            group={"entity_ids": [0, 0]},
        )
