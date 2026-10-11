"""Property: a release-time QRF re-run ignores a moderate rescale of weights.

The release's population mass repair multiplies every household weight by one
factor before it re-runs the base stages. Each stage resolves person weights
from those household weights and uses them as relative probabilities (the
forests' weighted bootstrap) and as sample weights (the sign gate), so a
uniform factor should not change a draw. This module samples three factors in
[0.25, 4] for each of childcare, energy subsidy and retirement contributions
and requires the re-run to reproduce the base output bit for bit.

The range is deliberate. The sign gate is scikit-learn's histogram
gradient-boosting classifier, which discards a split whose Hessian sum is below
an absolute floor (``min_hessian_to_split``, 1e-3) and is given the weights
unnormalized, so a factor small enough to push weighted sums under that floor
would change its splits. The release's factor was 1.0253 on the Route A build,
where it changed none of 907,863 retirement-contribution rows
(docs/evidence/us-release-reimputation-1191).

Hypothesis is a workspace dev dependency; the clean-wheel lane installs only
pytest, where this module skips.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")

from test_support.microcosm_build.us_release_base_input import (  # noqa: E402
    STAGES,
    assert_identical,
    build_base,
    outputs,
    release_stage,
    with_scaled_weights,
)

_BASE = None


def _base():
    global _BASE
    if _BASE is None:
        _BASE = build_base(n=160, seed=5)
    return _BASE


@pytest.mark.parametrize("stage", sorted(STAGES))
@hypothesis.settings(max_examples=3, deadline=None, derandomize=True)
@hypothesis.given(factor=st.floats(min_value=0.25, max_value=4.0))
def test_rerun_is_invariant_to_a_uniform_weight_scale(
    stage: str, factor: float
) -> None:
    base = _base()
    rerun = release_stage(stage)(with_scaled_weights(base, factor))

    assert_identical(outputs(rerun, stage), outputs(base, stage))
