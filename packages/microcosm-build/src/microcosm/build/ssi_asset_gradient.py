"""Policy-invariant SSI claiming gradient and weighted intercept solve.

Assets are dollars held by the person, not a policy's resource-test result.
The transform is natural ``log1p(assets)`` (zero maps continuously to zero).
The slope is estimated outside the build; only the intercept is count-solved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files

import numpy as np


def ssi_asset_propensity(
    assets: np.ndarray, intercept: float, slope: float
) -> np.ndarray:
    """Return logistic(intercept + slope * log1p(own liquid assets)).

    Finite inputs produce probabilities strictly inside (0, 1), including
    floating-point extremes. Reporter pinning is performed by the caller.
    """

    assets = np.asarray(assets, dtype=np.float64)
    if not np.isfinite(assets).all() or (assets < 0).any():
        raise ValueError("SSI gradient assets must be finite and nonnegative.")
    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ValueError("SSI gradient coefficients must be finite.")
    index = intercept + slope * np.log1p(assets)
    probability = np.exp(-np.logaddexp(0.0, -index))
    return np.clip(probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))


def solve_ssi_asset_intercept(
    assets: np.ndarray, weights: np.ndarray, target_mass: float, slope: float
) -> float:
    """Solve sum(weights * propensity) = target_mass by bounded bisection.

    The caller subtracts anchored mass before this solve. An unreachable
    target is an eligibility/capacity defect, not a reason to change a slope.
    """

    assets = np.asarray(assets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if assets.ndim != 1 or assets.shape != weights.shape:
        raise ValueError("SSI gradient assets and weights must be aligned vectors.")
    ssi_asset_propensity(assets, 0.0, slope)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("SSI gradient weights must be finite and nonnegative.")
    mass = float(weights.sum())
    if not np.isfinite(target_mass) or not 0 < target_mass < mass:
        raise ValueError("SSI gradient target must lie strictly within capacity.")
    positive = weights > 0
    offsets = slope * np.log1p(assets[positive])
    fraction = target_mass / mass
    logit = float(np.log(fraction) - np.log1p(-fraction))
    lower = logit - float(offsets.max())
    upper = logit - float(offsets.min())
    for _ in range(96):
        midpoint = (lower + upper) / 2
        expected = float(np.dot(weights, ssi_asset_propensity(assets, midpoint, slope)))
        if expected < target_mass:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def load_ssi_asset_slopes() -> dict[str, float]:
    """Read the reviewed SIPP estimate, packaged with its sampling provenance."""

    payload = json.loads(
        files("microcosm.build.us").joinpath("ssi_asset_gradient.json").read_text()
    )
    slopes: Mapping[str, object] = payload["slopes"]
    return {str(key): float(value) for key, value in slopes.items()}
