"""Differential: the neutral transport weight gates against the UK runtime's.

``transport/gate_bindings.py`` defines country-neutral ``weight_ess`` and
``weight_ratio`` comparisons because :mod:`microcosm.build.gates` has none
and the transport bindings may not import a country runtime. The UK runtime
already implements the same two gates (``uk_runtime/terminal_gates.py``,
over ``uk_runtime/diagnostics.uk_weight_summary``). On the same row weights
and threshold the two must agree on the verdict and on every summary number
they share; this file holds that check, beside the UK code it reads. The
neutral module's own tests live in
``engine_free/shared/test_transport_gate_bindings.py``.
"""

from __future__ import annotations

import math

import numpy as np
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


@PROPERTY
@given(weights=_WEIGHTS)
def test_property_summaries_agree(weights) -> None:
    neutral = weight_summary(weights)
    uk = uk_weight_summary(np.asarray(weights))
    for field in _SHARED:
        assert _same(neutral[field], uk[field]), field


@PROPERTY
@given(weights=_WEIGHTS, minimum=st.floats(min_value=1e-6, max_value=1.0))
def test_property_ess_verdicts_agree(weights, minimum) -> None:
    neutral = weight_ess_gate(weights, minimum_ess_fraction=minimum)
    uk = uk_weight_ess_gate(np.asarray(weights), minimum_ess_fraction=minimum)
    assert neutral.name == uk.name == "weight_ess"
    assert neutral.passed is uk.passed
    assert neutral.details["minimum_ess_fraction"] == uk.details["minimum_ess_fraction"]


@PROPERTY
@given(weights=_WEIGHTS, maximum=st.floats(min_value=1e-3, max_value=1e4))
def test_property_ratio_verdicts_agree(weights, maximum) -> None:
    neutral = weight_ratio_gate(weights, maximum_max_to_median_ratio=maximum)
    uk = uk_weight_ratio_gate(np.asarray(weights), maximum_max_to_median_ratio=maximum)
    assert neutral.name == uk.name == "weight_ratio"
    assert neutral.passed is uk.passed
    assert (
        neutral.details["maximum_max_to_median_ratio"]
        == uk.details["maximum_max_to_median_ratio"]
    )
