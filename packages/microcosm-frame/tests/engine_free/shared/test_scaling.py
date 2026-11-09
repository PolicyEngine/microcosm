"""Scaling contracts that do not depend on a country engine."""

import numpy as np
import pytest

from microcosm.frame import SignedScale, apply_scale


@pytest.mark.parametrize("component", ["positive", "negative"])
@pytest.mark.parametrize("factor", [0.0, -1.0, np.inf, np.nan])
def test_signed_scale_rejects_invalid_components(component, factor) -> None:
    factors = {"positive": 1.0, "negative": 1.0, component: factor}
    with pytest.raises(ValueError, match="finite and positive"):
        SignedScale(**factors)


def test_apply_scale_returns_float64_and_rejects_signed_overflow() -> None:
    values = np.array([2, -2, 0], dtype=np.int64)
    scaled = apply_scale(values, SignedScale(1.25, 1.5))
    assert scaled.dtype == np.float64
    assert scaled.tolist() == [2.5, -3.0, 0.0]
    assert values.tolist() == [2, -2, 0]
    with pytest.raises(ValueError, match="finite"):
        apply_scale(values, SignedScale(1e308, 1.0))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("factor", [1.1, SignedScale(1.1, 1.2)])
def test_apply_scale_rejects_nonfinite_inputs(value, factor) -> None:
    with pytest.raises(ValueError, match="Values to scale must be finite"):
        apply_scale(np.array([1.0, value]), factor)


@pytest.mark.parametrize("factor", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_apply_scale_rejects_invalid_scalar_factors(factor) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        apply_scale(np.array([2.0, -2.0, 0.0]), factor)
