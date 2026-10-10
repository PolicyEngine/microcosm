"""Property: a release-time QRF re-run ignores a uniform rescale of weights.

The release's population mass repair multiplies every household weight by one
factor before it re-runs the base stages. The retirement-contribution stage
resolves person weights from those household weights and uses them only as
relative probabilities (the forests' weighted bootstrap) and as sample weights
(the sign gate), so the re-run must reproduce the base output exactly for any
positive factor. Real data agree: on the Route A base the mass-repair factor
1.0253 changed none of 907,863 rows (docs/evidence/us-release-reimputation-1191).

Hypothesis is a workspace dev dependency; the clean-wheel lane installs only
pytest, where this module skips.
"""

from __future__ import annotations

import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")

from test_support.microcosm_build.us_release_base_input import (  # noqa: E402
    build_base,
    outputs,
    retirement_contributions,
    with_scaled_weights,
)

_BASE = None


def _base():
    global _BASE
    if _BASE is None:
        _BASE = build_base(n=160, seed=5)
    return _BASE


@hypothesis.settings(max_examples=4, deadline=None, derandomize=True)
@hypothesis.given(factor=st.floats(min_value=0.25, max_value=4.0))
def test_retirement_rerun_is_invariant_to_any_uniform_weight_scale(
    factor: float,
) -> None:
    base = _base()
    rerun = retirement_contributions(with_scaled_weights(base, factor))

    pd.testing.assert_frame_equal(
        outputs(rerun, "retirement_contributions"),
        outputs(base, "retirement_contributions"),
    )
