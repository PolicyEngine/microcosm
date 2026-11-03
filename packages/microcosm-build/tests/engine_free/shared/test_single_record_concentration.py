"""``single_record_concentration``: one column's largest record, as a share.

These tests pin eleven invariants, each for every input the strategies
reach. "Exactly" means in float64, with no tolerance.

1. **Bounds.** A measured column has ``0 < top_1_share <= top_k_share <= 1``
   exactly. Every carrier's mass is positive, and a rounded sum of
   non-negative masses never falls below any one of them, so the total and
   the tail sum each reach at least the largest mass. The cap keeps
   ``top_k_share`` at most 1.
2. **No rounding step lifts a share past 1.** The measurement sums the tail
   in the order a partition leaves it and the total in record order, so their
   ratio can pass 1 when the records outside the tail hold less mass than one
   rounding step: for ``[2**53 + 2, 3, 1, 0.5]`` at ``top_k = 3`` the gate
   reports 1.0000000000000002 or 1.0000000000000004, whatever the platform.
   ``top_k_share`` is ``min(ratio, 1)`` under every arrangement numpy may
   return.
3. **One record.** With ``top_k = 1``, ``top_k_share`` is ``top_1_share``,
   and both are the largest mass over the total mass, exactly.
4. **Monotone in ``top_k``, to rounding.** A larger ``top_k`` never lowers
   ``top_k_share`` by more than float rounding. It can lower it by a
   rounding step, because each ``top_k`` sums its own partition:
   ``[0.5, 1, 2**53, 0.5, 0.5]`` gives 1.0 at ``top_k = 3`` and
   0.9999999999999998 at ``top_k = 4`` on macOS arm64. The measurement
   accepts that drop to keep ``top_k_share`` the gate's own number
   (invariant 6). With at most ``top_k`` carriers, ``top_k_share`` is
   exactly 1.
5. **Pigeonhole floor.** The largest of ``n`` positive masses is at least
   their mean, so ``top_1_share >= 1 / carriers``, and
   ``top_k_share >= min(top_k, carriers) / carriers``, to float rounding.
6. **Differential, against the gate.** On every column
   ``tail_concentration_gate`` checks whose total mass is finite,
   ``top_k_share`` is ``min(top_share, 1)`` for the gate's ``top_share``,
   exactly, under every partition arrangement, and ``carriers`` is its
   carrier count; on a thin column ``carriers`` is the gate's thin count.
7. **Differential, against exact arithmetic.** Both shares equal the ratios of
   the same float masses summed as exact rationals, to float rounding.
8. **Scale-free in the weights.** Multiplying every weight by one positive
   factor leaves the carrier count and both shares unchanged and multiplies
   ``top_1_weight`` by the factor: exactly for a power of two, to float
   rounding otherwise. Scaling or negating every value changes nothing either,
   since the measurement takes a signed column's absolute mass.
9. **Total.** Any float64 values and weights (NaN, infinities, zeros,
   negative weights, products past the float64 range) give a report that is
   valid JSON without NaN, and numpy warns about nothing. A column with no
   carriers, or with a total mass that is not finite, reports its three
   measurements as ``None``: the measurement reports zero mass and never
   divides by it.
10. **Order-free.** Permuting the records leaves ``carriers`` and
    ``top_1_weight`` unchanged and the shares unchanged to float rounding.
    Records tied for the largest mass report the largest of their weights.
11. **Monotone in the largest record.** Raising the largest record's value
    never lowers ``top_1_share``.

numpy leaves the order of the elements on each side of a partition undefined,
and the order differs across CPUs. numpy 2.4 partitions float64 with
x86-simd-sort on x86-64 CPUs at the X86_V3 (AVX2) level or above, and that
library sorts a small array outright. Elsewhere, macOS arm64 among them,
numpy uses introselect. Tests whose sums depend on that order fix it with
:func:`_partition_arranged`, so they pass the same way on every platform.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import operator
import warnings
from collections.abc import Iterator, Sequence
from fractions import Fraction
from unittest import mock

import numpy as np
import pytest
from hypothesis import assume, given, settings, target
from hypothesis import strategies as st

import microcosm.build.gates as gates
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

# Values near 2**53, where float64's spacing is 2, beside small multiples of
# one half, at power-of-two weights so every product is exact. The records
# outside a tail can then hold less mass than one rounding step, so the
# tail's sum and the total can round past each other: the inputs that reach
# the cap in invariant 2. Every record carries mass.
_NEAR_TIE_VALUE = st.one_of(
    st.integers(min_value=2**53, max_value=2**53 + 64).map(float),
    st.integers(min_value=1, max_value=16).map(lambda halves: halves / 2.0),
)
_NEAR_TIE_RECORDS = st.lists(
    st.tuples(_NEAR_TIE_VALUE, st.sampled_from([0.5, 1.0, 2.0])),
    min_size=2,
    max_size=16,
)


class _NumpyWithLegalPartition:
    """numpy as ``microcosm.build.gates`` sees it, with one chosen partition.

    ``partition(array, kth)`` returns a legal partition: the kth smallest
    value at ``kth``, the smaller values sorted before it, and the larger
    values after it in ``tail_order``, an order of ``range`` of their count.
    Every other attribute is numpy's.
    """

    def __init__(self, tail_order: Sequence[int]) -> None:
        self._tail_order = list(tail_order)

    def __getattr__(self, name: str):
        return getattr(np, name)

    def partition(self, array, kth) -> np.ndarray:
        ordered = np.sort(np.asarray(array))
        kth = operator.index(kth) % ordered.size
        larger = ordered[kth + 1 :]
        assert sorted(self._tail_order) == list(range(larger.size))
        return np.concatenate([ordered[: kth + 1], larger[self._tail_order]])


@contextlib.contextmanager
def _partition_arranged(tail_order: Sequence[int]) -> Iterator[None]:
    """Run the gate and the measurement on one legal partition arrangement."""
    with mock.patch.object(gates, "np", _NumpyWithLegalPartition(tail_order)):
        yield


def _split(records) -> tuple[list[float], list[float]]:
    return [value for value, _ in records], [weight for _, weight in records]


def _measure(records, top_k: int = 100) -> dict[str, float | int | None]:
    values, weights = _split(records)
    return single_record_concentration(values, weights, top_k=top_k)


def _carrying(records) -> list[tuple[float, float]]:
    return [(value, weight) for value, weight in records if abs(value) * weight > 0.0]


def _masses(records) -> np.ndarray:
    """The gate's population: positive ``|value| * weight`` of finite records,
    in record order."""
    values, weights = _split(records)
    value_vector = np.asarray(values, dtype=np.float64)
    weight_vector = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(value_vector) & np.isfinite(weight_vector)
    with np.errstate(over="ignore"):
        mass = np.abs(value_vector[finite]) * weight_vector[finite]
    return mass[mass > 0.0]


def _exact_top_k_share(records, top_k: int) -> float:
    masses = sorted((Fraction(mass) for mass in _masses(records)), reverse=True)
    return float(sum(masses[:top_k]) / sum(masses))


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
@given(records=_NEAR_TIE_RECORDS, data=st.data())
def test_no_legal_partition_lifts_a_share_past_1(records, data) -> None:
    """Invariants 1, 2 and 6, in any arrangement numpy may return."""
    values, weights = _split(records)
    top_k = data.draw(st.integers(min_value=1, max_value=len(records) - 1))
    tail_order = data.draw(st.permutations(range(top_k - 1)))

    with _partition_arranged(tail_order):
        result = single_record_concentration(values, weights, top_k=top_k)
        gate = tail_concentration_gate(
            {"column": values},
            {"column": weights},
            top_k=top_k,
            min_nonzero_records=top_k + 1,
        )

    ratio = gate.details["top_share"]["column"]
    target(ratio, label="the gate's uncapped top-k share")
    assert result["carriers"] == gate.details["carrier_counts"]["column"]
    assert 0.0 < result["top_1_share"] <= result["top_k_share"] <= 1.0
    assert result["top_k_share"] == min(ratio, 1.0)
    assert result["top_k_share"] == pytest.approx(
        _exact_top_k_share(records, top_k), rel=_ROUNDING, abs=0.0
    )


@PROPERTY
@given(records=st.one_of(_RECORDS, _NEAR_TIE_RECORDS, _ANY_RECORDS))
def test_top_k_of_one_is_the_largest_mass_over_the_total(records) -> None:
    result = _measure(records, top_k=1)

    assert result["top_k_share"] == result["top_1_share"]
    if result["top_1_share"] is not None:
        masses = _masses(records)
        assert result["top_1_share"] == float(masses.max()) / float(masses.sum())


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
    # Any numpy warning (overflow, invalid, divide) fails the test.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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
@given(
    records=st.one_of(_RECORDS, _NEAR_TIE_RECORDS),
    top_k=_TOP_K,
    data=st.data(),
)
def test_a_larger_top_k_never_lowers_the_top_k_share_past_rounding(
    records, top_k, data
) -> None:
    """Invariant 4, with each ``top_k`` in any arrangement numpy may return."""
    assume(_carrying(records))
    smaller_order = data.draw(st.permutations(range(top_k - 1)))
    larger_order = data.draw(st.permutations(range(top_k)))

    with _partition_arranged(smaller_order):
        smaller = _measure(records, top_k)["top_k_share"]
    with _partition_arranged(larger_order):
        larger = _measure(records, top_k + 1)["top_k_share"]

    assert larger >= smaller * (1.0 - _ROUNDING)


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
        """One weight-644.7 tax unit draws $218.9M: $141.1B, 21.7% of the column.

        The test spreads the other 16,000 carriers so the top 100 hold about
        0.43, near microcosm#1259's seed-0 rerun of the #1251 chain at initial
        weights (16,001 carriers, top 100 at 0.451) and well under the gate's
        0.75: the gate passes, and only the single-record share shows the draw.
        """
        rng = np.random.default_rng(1251)
        body_values = rng.lognormal(mean=10.0, sigma=1.8, size=16_000)
        body_weights = rng.uniform(100.0, 900.0, size=16_000)
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
        assert result["carriers"] == 16_001
        assert result["top_1_share"] == pytest.approx(0.217, abs=1e-9)
        assert result["top_1_weight"] == 644.7
        assert result["top_k_share"] == gate.details["top_share"]["partnership_income"]
        assert 0.40 < result["top_k_share"] < 0.50
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
        """The smallest input whose gate share passes 1 on every platform.

        ``[2**53 + 2, 3, 1, 0.5]`` at ``top_k = 3``, where float64's spacing
        is 2. Summed in record order, all three additions round down and the
        total comes to ``2**53 + 4``. The two legal tails,
        ``[1, 3, 2**53 + 2]`` and ``[1, 2**53 + 2, 3]``, sum to ``2**53 + 6``
        and ``2**53 + 8``. The exact share, ``(2**53 + 6) / (2**53 + 6.5)``,
        rounds to 1. Rounding is monotone, so a record-order sum of
        non-negative masses never falls below the rounded sum of any two of
        them, and a tail of one or two records cannot pass 1. The gate checks
        only columns with more carriers than ``top_k``, so four records at
        ``top_k = 3`` make the smallest input whose gate share can pass 1.
        """
        values = [2.0**53 + 2, 3.0, 1.0, 0.5]
        weights = [1.0] * len(values)

        def measure() -> tuple[float, dict[str, float | int | None]]:
            gate = tail_concentration_gate(
                {"column": values},
                {"column": weights},
                top_k=3,
                min_nonzero_records=4,
            )
            result = single_record_concentration(values, weights, top_k=3)
            return gate.details["top_share"]["column"], result

        # numpy's own partition, in whatever order this platform leaves it.
        ratio, result = measure()
        assert ratio > 1.0
        assert result["top_k_share"] == 1.0

        ratios = set()
        for tail_order in itertools.permutations(range(2)):
            with _partition_arranged(tail_order):
                ratio, result = measure()
            ratios.add(ratio)
            assert result["top_k_share"] == 1.0
        assert ratios == {1.0000000000000002, 1.0000000000000004}
        assert _exact_top_k_share(list(zip(values, weights, strict=True)), 3) == 1.0

    def test_the_partition_order_moves_the_gate_share_and_not_past_the_cap(
        self,
    ) -> None:
        """The input CI failed on, in every arrangement numpy may return.

        At ``top_k = 8`` the record outside the tail holds 0.5 of a total near
        1.1e16, below one rounding step, so the exact share rounds to 1. The
        tail's order decides the gate's share: 1.0000000000000002 in the
        order introselect returns (macOS arm64); exactly 1.0 sorted, the order
        x86-simd-sort's base case leaves and the share CI's x86-64 runners
        reported; and 0.9999999999999998 in some other legal orders. The
        measurement reports ``min(share, 1)`` in each.
        """
        values = [2.0, 3.0, 2.0, 1.0, 1e16, 7.0, 1e15, 1.0, 0.5]
        weights = [1.0] * len(values)

        def measure(tail_order) -> tuple[float, float]:
            with _partition_arranged(tail_order):
                gate = tail_concentration_gate(
                    {"column": values},
                    {"column": weights},
                    top_k=8,
                    min_nonzero_records=9,
                )
                result = single_record_concentration(values, weights, top_k=8)
            return gate.details["top_share"]["column"], result["top_k_share"]

        # The tail's seven values after its smallest, 1.0, sorted: 1.0, 2.0,
        # 2.0, 3.0, 7.0, 1e15 and 1e16. Introselect leaves the tail as 1.0,
        # 2.0, 3.0, 1e16, 7.0, 1e15, 1.0, 2.0.
        introselect = (1, 3, 6, 4, 5, 0, 2)
        assert measure(introselect) == (1.0000000000000002, 1.0)
        assert measure(range(7)) == (1.0, 1.0)

        ratios = set()
        for tail_order in itertools.permutations(range(7)):
            ratio, share = measure(tail_order)
            ratios.add(ratio)
            assert share == min(ratio, 1.0)
        assert ratios == {0.9999999999999998, 1.0, 1.0000000000000002}
        assert _exact_top_k_share(list(zip(values, weights, strict=True)), 8) == 1.0

    def test_a_larger_top_k_can_lower_the_share_by_a_rounding_step(self) -> None:
        """Intended: each ``top_k`` sums its own partition, as the gate does.

        ``[0.5, 1, 2**53, 0.5, 0.5]`` sums to ``2**53 + 2`` in record order.
        The top-3 tail ``[0.5, 1, 2**53]`` sums to that too, a share of 1.0;
        the top-4 tail ``[0.5, 2**53, 1, 0.5]`` loses each value after
        ``2**53`` to rounding and sums to ``2**53``, a share of
        0.9999999999999998. numpy's own partition leaves these two orders on
        macOS arm64. Both shares are within a rounding step of the exact ones.
        """
        records = [(value, 1.0) for value in [0.5, 1.0, 2.0**53, 0.5, 0.5]]

        with _partition_arranged((0, 1)):
            top_3 = _measure(records, top_k=3)["top_k_share"]
        with _partition_arranged((2, 1, 0)):
            top_4 = _measure(records, top_k=4)["top_k_share"]

        assert (top_3, top_4) == (1.0, 0.9999999999999998)
        for top_k, share in ((3, top_3), (4, top_4)):
            assert share == pytest.approx(
                _exact_top_k_share(records, top_k), rel=_ROUNDING, abs=0.0
            )

    def test_top_k_share_is_exactly_1_when_every_carrier_is_in_the_top_k(
        self,
    ) -> None:
        # Summed in the order introselect leaves them (macOS arm64), these
        # masses give 0.9999999999999998 of their record-order total. With
        # every carrier in the tail, the measurement sums no partition.
        values = [2.0, 1e15, 4.5e15, 1e16, 1.0, 1e15, 4.5e15]
        introselect = [1.0, 1e15, 4.5e15, 1e16, 2.0, 1e15, 4.5e15]
        assert float(np.sum(introselect)) / float(np.sum(values)) == (
            0.9999999999999998
        )

        # The six values above the smallest, sorted: 2.0, 1e15, 1e15, 4.5e15,
        # 4.5e15 and 1e16. Introselect leaves them as 1e15, 4.5e15, 1e16,
        # 2.0, 1e15, 4.5e15.
        with _partition_arranged((1, 3, 5, 0, 2, 4)):
            result = single_record_concentration(values, [1.0] * 7, top_k=7)

        assert result["top_k_share"] == 1.0

    def test_invalid_arguments_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            single_record_concentration([1.0], [1.0], top_k=0)
        with pytest.raises(ValueError, match="align"):
            single_record_concentration([1.0, 2.0], [1.0])
