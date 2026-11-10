"""Differential: the neutral transport weight gates against the UK runtime's.

``transport/gate_bindings.py`` defines country-neutral ``weight_ess`` and
``weight_ratio`` comparisons because :mod:`microcosm.build.gates` has none
and the transport bindings may not import a country runtime. The UK runtime
already implements the same two gates (``uk_runtime/terminal_gates.py``,
over ``uk_runtime/diagnostics.uk_weight_summary``). Compare concentration on
the same relative row weights, normalized for the UK helper so its raw-weight
arithmetic does not become the numerical oracle. Original-unit metadata still
agrees on the original weights. Unrepresentable concentration must fail closed
in the neutral gates. A second set of properties draws weights from the whole
finite non-negative float64 range, subnormals included: there the reported
positive-weight median must equal the exact median rounded once, and the UK
helper's own float median and total wherever those do not overflow. This file
holds those checks beside the UK code it reads. The
neutral module's own tests live in
``engine_free/shared/test_transport_gate_bindings.py``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.gate_bindings import (
    weight_ess_gate,
    weight_ratio_gate,
    weight_summary,
)
from microcosm.build.uk_runtime.diagnostics import uk_weight_summary
from microcosm.build.uk_runtime.terminal_gates import (
    uk_weight_ess_gate,
    uk_weight_ratio_gate,
)

PROPERTY = settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
_WEIGHTS = st.lists(
    st.floats(min_value=0.0, max_value=1e7, allow_nan=False), min_size=1, max_size=60
)
#: The summary fields both implementations report.
_SHARED = (
    "n_records",
    "positive_weight_records",
    "zero_weight_records",
    "total_weight",
    "effective_sample_size",
    "ess_fraction",
    "median_positive_weight",
    "max_weight",
    "max_to_median_positive_weight",
)
#: Fields reported in the weights' original units, not after normalization.
_ORIGINAL_UNITS = ("total_weight", "median_positive_weight", "max_weight")

_FLOAT64_MAX = float(np.finfo(np.float64).max)
#: Every finite non-negative float64, subnormals included.
_ANY_WEIGHT = st.floats(
    min_value=0.0, max_value=_FLOAT64_MAX, allow_nan=False, allow_infinity=False
)
#: Integer multiples of one power of two, from the smallest subnormal step up
#: to the float64 maximum. Every value is exact and the ratio stays below
#: 2**53, so these vectors always summarize, at every binary scale.
_SCALED_UNITS = st.builds(
    lambda units, exponent: [math.ldexp(unit, exponent) for unit in units],
    st.lists(st.integers(min_value=0, max_value=2**53 - 1), min_size=1, max_size=60),
    st.integers(min_value=-1074, max_value=971),
)
_FULL_RANGE_WEIGHTS = st.one_of(
    st.lists(_ANY_WEIGHT, min_size=1, max_size=60), _SCALED_UNITS
)
#: The review's subnormal pair: its exact midpoint, 3.5 steps of 2**-1074,
#: rounds to 2e-323; rescaling the normalized median gave 1.5e-323.
_SUBNORMAL_PAIR = [1e-323, 2.5e-323]
#: Two positive weights that normalize below the smallest subnormal share the
#: median with 2**-1021 + 2**-1072. Normalized to zero, they push the median's
#: reciprocal past the float64 maximum; the exact ratio is finite.
_UNDERFLOWING_MEDIAN = [
    5e-324,
    5e-324,
    math.ldexp(1, -1021) + math.ldexp(1, -1072),
    4.0,
]


def _exact_positive_median(weights) -> float | None:
    positive = sorted(Fraction(value) for value in weights if value > 0)
    if not positive:
        return None
    middle = len(positive) // 2
    if len(positive) % 2:
        return float(positive[middle])
    return float((positive[middle - 1] + positive[middle]) / 2)


def _same(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0.0)


def _uk_weights(weights) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(weights)
    maximum = float(values.max())
    normalized = values / maximum if maximum else values.copy()
    # Preserve positive support even when a normalized positive rounds to zero.
    normalized[(values > 0) & (normalized == 0)] = np.nextafter(0.0, 1.0)
    return values, normalized


def _invalid_ratio(summary) -> bool:
    ratio = summary["max_to_median_positive_weight"]
    return ratio is not None and not math.isfinite(ratio)


@PROPERTY
@example(weights=_SUBNORMAL_PAIR)
@example(weights=_UNDERFLOWING_MEDIAN)
@given(weights=_WEIGHTS)
def test_property_summaries_agree(weights) -> None:
    values, normalized = _uk_weights(weights)
    uk = uk_weight_summary(normalized)
    if _invalid_ratio(uk):
        with pytest.raises(ValueError, match="Computed"):
            weight_summary(values)
        assert not weight_ess_gate(values, minimum_ess_fraction=1.0).passed
        assert not weight_ratio_gate(values, maximum_max_to_median_ratio=1.0).passed
        return
    neutral = weight_summary(values)
    original_uk = uk_weight_summary(values)
    for field in _SHARED:
        expected = (
            original_uk[field]
            if field in ("total_weight", "median_positive_weight", "max_weight")
            else uk[field]
        )
        assert _same(neutral[field], expected), field


@PROPERTY
@example(weights=_UNDERFLOWING_MEDIAN, minimum=0.2)
@given(weights=_WEIGHTS, minimum=st.floats(min_value=1e-6, max_value=1.0))
def test_property_ess_verdicts_agree(weights, minimum) -> None:
    _, normalized = _uk_weights(weights)
    neutral = weight_ess_gate(weights, minimum_ess_fraction=minimum)
    uk = uk_weight_ess_gate(normalized, minimum_ess_fraction=minimum)
    assert neutral.name == uk.name == "weight_ess"
    assert neutral.details["minimum_ess_fraction"] == uk.details["minimum_ess_fraction"]
    if _invalid_ratio(uk_weight_summary(normalized)):
        with pytest.raises(ValueError, match="Computed"):
            weight_summary(weights)
        assert neutral.passed is False
        assert neutral.failures
    else:
        assert neutral.passed is uk.passed


@PROPERTY
@given(weights=_WEIGHTS, maximum=st.floats(min_value=1e-3, max_value=1e4))
def test_property_ratio_verdicts_agree(weights, maximum) -> None:
    _, normalized = _uk_weights(weights)
    neutral = weight_ratio_gate(weights, maximum_max_to_median_ratio=maximum)
    uk = uk_weight_ratio_gate(normalized, maximum_max_to_median_ratio=maximum)
    assert neutral.name == uk.name == "weight_ratio"
    assert neutral.passed is uk.passed
    assert (
        neutral.details["maximum_max_to_median_ratio"]
        == uk.details["maximum_max_to_median_ratio"]
    )
    if _invalid_ratio(uk_weight_summary(normalized)):
        with pytest.raises(ValueError, match="Computed"):
            weight_summary(weights)
        assert neutral.passed is False
        assert neutral.failures


def test_subnormal_median_is_rounded_once() -> None:
    neutral = weight_summary(_SUBNORMAL_PAIR)
    assert neutral["median_positive_weight"] == 2e-323
    assert (
        neutral["median_positive_weight"]
        == uk_weight_summary(_SUBNORMAL_PAIR)["median_positive_weight"]
    )


def test_positive_support_survives_normalization() -> None:
    _, normalized = _uk_weights(_UNDERFLOWING_MEDIAN)
    neutral = weight_summary(_UNDERFLOWING_MEDIAN)
    uk = uk_weight_summary(normalized)
    assert math.isfinite(uk["max_to_median_positive_weight"])
    assert (
        neutral["max_to_median_positive_weight"] == uk["max_to_median_positive_weight"]
    )
    assert weight_ess_gate(_UNDERFLOWING_MEDIAN, minimum_ess_fraction=0.2).passed
    assert uk_weight_ess_gate(normalized, minimum_ess_fraction=0.2).passed


@PROPERTY
@example(weights=_SUBNORMAL_PAIR)
@example(weights=_UNDERFLOWING_MEDIAN)
@example(weights=[_FLOAT64_MAX, _FLOAT64_MAX])
@example(weights=[0.0, 5e-324, _FLOAT64_MAX])
@given(weights=_FULL_RANGE_WEIGHTS)
def test_property_summaries_agree_over_the_full_float64_range(weights) -> None:
    values, normalized = _uk_weights(weights)
    uk = uk_weight_summary(normalized)
    if _invalid_ratio(uk):
        with pytest.raises(ValueError, match="Computed"):
            weight_summary(values)
        assert not weight_ess_gate(values, minimum_ess_fraction=1.0).passed
        assert not weight_ratio_gate(values, maximum_max_to_median_ratio=1.0).passed
        return
    neutral = weight_summary(values)
    for field in _SHARED:
        if field not in _ORIGINAL_UNITS:
            assert _same(neutral[field], uk[field]), field
    median = neutral["median_positive_weight"]
    assert median == _exact_positive_median(weights)
    # The UK helper's raw float sum and midpoint overflow near the float64
    # maximum; wherever they stay finite the original units agree exactly.
    with np.errstate(over="ignore", invalid="ignore"):
        original_uk = uk_weight_summary(values)
    assert neutral["max_weight"] == original_uk["max_weight"]
    total = original_uk["total_weight"]
    assert neutral["total_weight"] == (total if math.isfinite(total) else None)
    uk_median = original_uk["median_positive_weight"]
    if uk_median is None or math.isfinite(uk_median):
        assert median == uk_median


@PROPERTY
@example(weights=_UNDERFLOWING_MEDIAN, minimum=0.2, maximum=1e4)
@given(
    weights=_FULL_RANGE_WEIGHTS,
    minimum=st.floats(min_value=1e-6, max_value=1.0),
    maximum=st.floats(min_value=1e-3, max_value=1e4),
)
def test_property_verdicts_agree_over_the_full_float64_range(
    weights, minimum, maximum
) -> None:
    _, normalized = _uk_weights(weights)
    ess = weight_ess_gate(weights, minimum_ess_fraction=minimum)
    ratio = weight_ratio_gate(weights, maximum_max_to_median_ratio=maximum)
    assert (
        ratio.passed
        is uk_weight_ratio_gate(normalized, maximum_max_to_median_ratio=maximum).passed
    )
    if _invalid_ratio(uk_weight_summary(normalized)):
        # An unrepresentable ratio refuses the summary: both gates fail closed.
        assert ess.passed is False
        assert ess.failures
        assert ratio.failures
    else:
        assert (
            ess.passed
            is uk_weight_ess_gate(normalized, minimum_ess_fraction=minimum).passed
        )
