"""``single_record_concentration``: one column's largest record, as a share.

Invariants pinned here, each for every input the strategies reach:

1. **Share chain.** A measured column has
   ``0 < top_1_share <= top_k_share <= 1``. With ``top_k = 1`` the two shares
   are the same number, and with at most ``top_k`` carriers ``top_k_share`` is
   exactly 1.
2. **Pigeonhole floor.** The largest of ``n`` positive masses is at least
   their mean, so ``top_1_share >= 1 / carriers``, and
   ``top_k_share >= min(top_k, carriers) / carriers``.
3. **Scale-free in the weights.** Multiplying every weight by one positive
   factor leaves the carrier count and both shares unchanged and multiplies
   ``top_1_weight`` by the factor: exactly for a power of two, to float
   rounding otherwise. Scaling or negating every value changes nothing either,
   since a signed column is measured on absolute mass.
4. **Total.** Any float64 values and weights (NaN, infinities, zeros,
   negative weights, products past the float64 range) give a report that is
   valid JSON without NaN. A column with no carriers, or with a total mass
   that is not finite, reports its three measurements as ``None``: zero mass
   is reported, never divided by.
5. **Order-free.** Permuting the records leaves ``carriers`` and
   ``top_1_weight`` unchanged and the shares unchanged to float rounding.
   Records tied for the largest mass report the largest of their weights.
6. **Monotone.** Raising the largest record's value never lowers
   ``top_1_share``; raising ``top_k`` never lowers ``top_k_share``.
7. **Differential, against the gate.** On every column
   ``tail_concentration_gate`` checks, ``top_k_share`` is the gate's
   ``top_share`` bit for bit and ``carriers`` its carrier count; on a thin
   column ``carriers`` is the gate's thin count.
8. **Differential, against exact arithmetic.** Both shares equal the ratios of
   the same float masses summed as exact rationals, to float rounding.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.gates import (
    single_record_concentration,
    tail_concentration_gate,
)

PROPERTY = settings(max_examples=300, deadline=None, database=None)
_ROUNDING = 1e-12

# Dollar-like magnitudes: no product or sum leaves the float64 range, and no
# positive mass rounds to zero. The totality property drops these bounds.
_VALUE = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-3, max_value=1e12),
    st.floats(min_value=-1e12, max_value=-1e-3),
)
_WEIGHT = st.one_of(st.just(0.0), st.floats(min_value=1e-3, max_value=1e6))
_RECORDS = st.lists(st.tuples(_VALUE, _WEIGHT), min_size=0, max_size=40)
_TOP_K = st.integers(min_value=1, max_value=12)
_ANY_FLOAT = st.floats(allow_nan=True, allow_infinity=True)
_ANY_RECORDS = st.lists(st.tuples(_ANY_FLOAT, _ANY_FLOAT), min_size=0, max_size=40)
_MEASUREMENTS = ("top_1_share", "top_1_weight", "top_k_share")


def _split(records) -> tuple[list[float], list[float]]:
    return [value for value, _ in records], [weight for _, weight in records]


def _measure(records, top_k: int = 100) -> dict[str, float | int | None]:
    values, weights = _split(records)
    return single_record_concentration(values, weights, top_k=top_k)


def _carrying(records) -> list[tuple[float, float]]:
    return [(value, weight) for value, weight in records if abs(value) * weight > 0.0]


@PROPERTY
@given(records=_RECORDS, top_k=_TOP_K)
def test_share_chain(records, top_k) -> None:
    result = _measure(records, top_k)
    carriers = len(_carrying(records))

    assert result["carriers"] == carriers
    if not carriers:
        assert all(result[key] is None for key in _MEASUREMENTS)
        return
    assert 0.0 < result["top_1_share"] <= result["top_k_share"] <= 1.0
    assert result["top_1_weight"] > 0.0
    if carriers <= top_k:
        assert result["top_k_share"] == 1.0
    # The pigeonhole floors, to the rounding of one float sum.
    assert result["top_1_share"] >= (1.0 / carriers) * (1.0 - _ROUNDING)
    assert result["top_k_share"] >= (min(top_k, carriers) / carriers) * (
        1.0 - _ROUNDING
    )


@PROPERTY
@given(records=_RECORDS)
def test_top_k_of_one_is_the_top_1_share(records) -> None:
    result = _measure(records, top_k=1)

    assert result["top_k_share"] == result["top_1_share"]


@PROPERTY
@given(records=_RECORDS, top_k=_TOP_K, exponent=st.integers(-20, 20))
def test_scaling_weights_by_a_power_of_two_is_exact(records, top_k, exponent) -> None:
    factor = math.ldexp(1.0, exponent)
    values, weights = _split(records)
    base = single_record_concentration(values, weights, top_k=top_k)

    scaled = single_record_concentration(
        values, [weight * factor for weight in weights], top_k=top_k
    )

    assert scaled["carriers"] == base["carriers"]
    assert scaled["top_1_share"] == base["top_1_share"]
    assert scaled["top_k_share"] == base["top_k_share"]
    if base["top_1_weight"] is None:
        assert scaled["top_1_weight"] is None
    else:
        assert scaled["top_1_weight"] == base["top_1_weight"] * factor


@PROPERTY
@given(
    records=_RECORDS,
    top_k=_TOP_K,
    factor=st.floats(min_value=1e-3, max_value=1e3),
)
def test_shares_are_scale_free_in_the_weights(records, top_k, factor) -> None:
    values, weights = _split(records)
    base = single_record_concentration(values, weights, top_k=top_k)

    scaled = single_record_concentration(
        values, [weight * factor for weight in weights], top_k=top_k
    )

    assert scaled["carriers"] == base["carriers"]
    for key in ("top_1_share", "top_k_share"):
        if base[key] is None:
            assert scaled[key] is None
        else:
            assert scaled[key] == pytest.approx(base[key], rel=1e-9, abs=0.0)


@PROPERTY
@given(
    records=_RECORDS,
    top_k=_TOP_K,
    exponent=st.integers(-10, 10),
    negate=st.booleans(),
)
def test_scaling_or_negating_values_changes_nothing(
    records, top_k, exponent, negate
) -> None:
    factor = math.ldexp(-1.0 if negate else 1.0, exponent)
    values, weights = _split(records)

    assert single_record_concentration(
        [value * factor for value in values], weights, top_k=top_k
    ) == single_record_concentration(values, weights, top_k=top_k)


@PROPERTY
@given(records=_ANY_RECORDS, top_k=_TOP_K)
def test_any_floats_give_a_finite_report(records, top_k) -> None:
    result = _measure(records, top_k)

    # allow_nan=False refuses NaN and both infinities.
    json.dumps(result, allow_nan=False)
    assert set(result) == {"carriers", *_MEASUREMENTS}
    assert isinstance(result["carriers"], int) and result["carriers"] >= 0
    measured = [result[key] is not None for key in _MEASUREMENTS]
    assert all(measured) or not any(measured)
    if not result["carriers"]:
        assert not any(measured)
    if all(measured):
        assert 0.0 < result["top_1_share"] <= result["top_k_share"] <= 1.0
        assert result["top_1_weight"] > 0.0


@PROPERTY
@given(records=_RECORDS, top_k=_TOP_K, seed=st.integers(0, 2**32 - 1))
def test_record_order_does_not_matter(records, top_k, seed) -> None:
    order = np.random.default_rng(seed).permutation(len(records))
    base = _measure(records, top_k)

    shuffled = _measure([records[index] for index in order], top_k)

    assert shuffled["carriers"] == base["carriers"]
    assert shuffled["top_1_weight"] == base["top_1_weight"]
    for key in ("top_1_share", "top_k_share"):
        if base[key] is None:
            assert shuffled[key] is None
        else:
            assert shuffled[key] == pytest.approx(base[key], rel=_ROUNDING, abs=0.0)


@PROPERTY
@given(
    records=_RECORDS,
    top_k=_TOP_K,
    lift=st.floats(min_value=1.0, max_value=1e3),
)
def test_a_larger_top_record_never_lowers_the_top_1_share(records, top_k, lift) -> None:
    carrying = _carrying(records)
    assume(carrying)
    top = max(range(len(records)), key=lambda i: abs(records[i][0]) * records[i][1])
    lifted = list(records)
    lifted[top] = (records[top][0] * lift, records[top][1])

    assert _measure(lifted, top_k)["top_1_share"] >= _measure(records, top_k)[
        "top_1_share"
    ] * (1.0 - _ROUNDING)


@PROPERTY
@given(records=_RECORDS, top_k=_TOP_K)
def test_a_larger_top_k_never_lowers_the_top_k_share(records, top_k) -> None:
    assume(_carrying(records))

    assert _measure(records, top_k + 1)["top_k_share"] >= _measure(records, top_k)[
        "top_k_share"
    ] * (1.0 - _ROUNDING)


@PROPERTY
@given(
    columns=st.dictionaries(
        st.sampled_from(["a", "b", "c", "d"]), _RECORDS, min_size=1, max_size=4
    ),
    top_k=st.integers(min_value=1, max_value=6),
    extra=st.integers(min_value=1, max_value=6),
)
def test_agrees_with_the_tail_concentration_gate(columns, top_k, extra) -> None:
    """The gate's own share and counts, on every column it checks or calls thin."""
    values = {column: _split(records)[0] for column, records in columns.items()}
    weights = {column: _split(records)[1] for column, records in columns.items()}

    gate = tail_concentration_gate(
        values, weights, top_k=top_k, min_nonzero_records=top_k + extra
    )

    checked = gate.details["top_share"]
    thin = gate.details["thin_columns"]
    assert set(checked) | set(thin) == set(columns)
    for column in columns:
        result = single_record_concentration(
            values[column], weights[column], top_k=top_k
        )
        if column in thin:
            assert result["carriers"] == thin[column]
            continue
        assert result["carriers"] == gate.details["carrier_counts"][column]
        # The gate sums its tail and its total in different orders, so its
        # share can pass 1 by a rounding step; the measurement caps it.
        assert result["top_k_share"] == min(checked[column], 1.0)


@PROPERTY
@given(records=_RECORDS, top_k=_TOP_K)
def test_agrees_with_exact_rational_arithmetic(records, top_k) -> None:
    masses = sorted(
        (Fraction(abs(value) * weight) for value, weight in _carrying(records)),
        reverse=True,
    )
    assume(masses)
    total = sum(masses)
    result = _measure(records, top_k)

    assert result["top_1_share"] == pytest.approx(
        float(masses[0] / total), rel=_ROUNDING, abs=0.0
    )
    assert result["top_k_share"] == pytest.approx(
        float(sum(masses[:top_k]) / total), rel=_ROUNDING, abs=0.0
    )
    top_weights = [
        weight
        for value, weight in _carrying(records)
        if Fraction(abs(value) * weight) == masses[0]
    ]
    assert result["top_1_weight"] == max(top_weights)


class TestExamples:
    def test_the_1251_line_32_draw(self) -> None:
        """One weight-644.7 tax unit draws $218.9M: $141B, 21.7% of the column.

        The other 14,597 carriers are spread so the top 100 hold about 0.39,
        near the rebuild's 0.399 and well under the gate's 0.75: the gate
        passes, and the single-record share is what shows the draw.
        """
        rng = np.random.default_rng(1251)
        body_values = rng.lognormal(mean=10.0, sigma=1.6, size=14_597)
        body_weights = rng.uniform(100.0, 900.0, size=14_597)
        draw_mass = 218.9e6 * 644.7
        # Scale the body so the draw is 21.7% of the column's mass.
        body_values *= (draw_mass * (1 - 0.217) / 0.217) / float(
            body_values @ body_weights
        )
        values = np.concatenate([body_values, [218.9e6]])
        weights = np.concatenate([body_weights, [644.7]])

        result = single_record_concentration(values, weights)
        gate = tail_concentration_gate(
            {"partnership_income": values}, {"partnership_income": weights}
        )

        assert gate.passed
        assert result["carriers"] == 14_598
        assert result["top_1_share"] == pytest.approx(0.217, abs=1e-9)
        assert result["top_1_weight"] == 644.7
        assert result["top_k_share"] == gate.details["top_share"]["partnership_income"]
        assert 0.35 < result["top_k_share"] < 0.45
        assert result["top_1_share"] < result["top_k_share"] < 0.75

    def test_a_signed_column_is_measured_on_absolute_mass(self) -> None:
        result = single_record_concentration([-60.0, 30.0, 10.0], [1.0, 1.0, 1.0])

        assert result == {
            "carriers": 3,
            "top_1_share": 0.6,
            "top_1_weight": 1.0,
            "top_k_share": 1.0,
        }

    def test_the_weight_decides_the_largest_record(self) -> None:
        # The smaller value on the heavier record carries more mass.
        result = single_record_concentration([100.0, 30.0], [1.0, 10.0])

        assert result["top_1_share"] == pytest.approx(300.0 / 400.0)
        assert result["top_1_weight"] == 10.0

    def test_tied_records_report_the_largest_weight(self) -> None:
        # Masses 6, 6 and 6 from weights 3, 2 and 6, in any order.
        for values, weights in (
            ([2.0, 3.0, 1.0], [3.0, 2.0, 6.0]),
            ([1.0, 2.0, 3.0], [6.0, 3.0, 2.0]),
        ):
            result = single_record_concentration(values, weights)

            assert result["top_1_share"] == pytest.approx(1.0 / 3.0)
            assert result["top_1_weight"] == 6.0

    def test_top_k_share_counts_only_the_top_k(self) -> None:
        values = np.concatenate([np.full(4, 10.0), np.full(6, 1.0)])

        result = single_record_concentration(values, np.ones(10), top_k=4)

        assert result["top_1_share"] == pytest.approx(10.0 / 46.0)
        assert result["top_k_share"] == pytest.approx(40.0 / 46.0)

    @pytest.mark.parametrize(
        ("values", "weights"),
        [
            ([], []),
            ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            ([5.0, 7.0], [0.0, 0.0]),
            ([float("nan"), float("inf")], [1.0, 1.0]),
            ([5.0], [float("nan")]),
            ([5.0, 7.0], [-1.0, -2.0]),
        ],
    )
    def test_a_zero_mass_column_is_reported_not_divided(self, values, weights) -> None:
        assert single_record_concentration(values, weights) == {
            "carriers": 0,
            "top_1_share": None,
            "top_1_weight": None,
            "top_k_share": None,
        }

    @pytest.mark.parametrize(
        ("values", "weights"),
        [
            # One product past the float64 range.
            ([1e200, 1.0], [1e200, 1.0]),
            # Finite products whose sum is past it.
            ([1.5e308, 1.5e308], [1.0, 1.0]),
        ],
    )
    def test_a_mass_past_the_float_range_is_reported_without_a_share(
        self, values, weights
    ) -> None:
        with np.errstate(over="raise"):
            result = single_record_concentration(values, weights)

        assert result == {
            "carriers": 2,
            "top_1_share": None,
            "top_1_weight": None,
            "top_k_share": None,
        }

    def test_non_finite_records_are_dropped_like_the_gate_drops_them(self) -> None:
        result = single_record_concentration(
            [float("nan"), float("inf"), 3.0, 1.0], [1.0, 1.0, 2.0, float("inf")]
        )

        assert result == {
            "carriers": 1,
            "top_1_share": 1.0,
            "top_1_weight": 2.0,
            "top_k_share": 1.0,
        }

    def test_a_rounding_step_never_lifts_the_top_k_share_past_1(self) -> None:
        """The gate sums its tail in partition order and its total in array
        order; here the tail comes out one rounding step above the total."""
        values = [2.0, 3.0, 2.0, 1.0, 1e16, 7.0, 1e15, 1.0, 0.5]
        weights = [1.0] * len(values)
        gate = tail_concentration_gate(
            {"column": values}, {"column": weights}, top_k=8, min_nonzero_records=9
        )

        result = single_record_concentration(values, weights, top_k=8)

        assert gate.details["top_share"]["column"] > 1.0
        assert result["top_k_share"] == 1.0

    def test_top_k_share_is_exactly_1_when_every_carrier_is_in_the_top_k(
        self,
    ) -> None:
        # Summed in partition order these masses give 0.9999999999999998.
        values = [2.0, 1e15, 4.5e15, 1e16, 1.0, 1e15, 4.5e15]

        result = single_record_concentration(values, [1.0] * 7, top_k=7)

        assert result["top_k_share"] == 1.0

    def test_invalid_arguments_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            single_record_concentration([1.0], [1.0], top_k=0)
        with pytest.raises(ValueError, match="align"):
            single_record_concentration([1.0, 2.0], [1.0])
