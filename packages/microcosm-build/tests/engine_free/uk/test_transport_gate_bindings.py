"""Differential: the neutral transport weight gates against the UK runtime's.

``transport/gate_bindings.py`` defines country-neutral ``weight_ess`` and
``weight_ratio`` comparisons because :mod:`microcosm.build.gates` has none
and the transport bindings may not import a country runtime. The UK runtime
already implements the same two gates (``uk_runtime/terminal_gates.py``,
over ``uk_runtime/diagnostics.uk_weight_summary``). Compare concentration on
the same relative row weights, normalized for the UK helper so its raw-weight
arithmetic does not become the numerical oracle. Original-unit metadata still
agrees on the original weights. Unrepresentable concentration must fail closed
in the neutral gates. This file holds those checks beside the UK code it reads. The
neutral module's own tests live in
``engine_free/shared/test_transport_gate_bindings.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
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
