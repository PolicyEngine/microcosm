"""Weighted quantile knots for values pooled by group.

A knot array describes one group's value distribution: the weighted quantile
at each point of a shared grid that runs from 0 to 1. A consumer reads a
value at a rank by linear interpolation between knots, so the knots also
imply a mean (:func:`knot_mean`). The exact weighted mean is kept beside the
knots when totals must not depend on the grid.

The arithmetic moved here unchanged from the UK high-value home stock
library (microcosm#1524), which delegates to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GroupKnots",
    "check_knots",
    "grouped_knots",
    "knot_mean",
    "weighted_quantiles",
]


@dataclass(frozen=True)
class GroupKnots:
    """One group's knots: its code, row count, weight and quantiles."""

    code: str
    count: int
    weight: float
    quantiles: tuple[float, ...]


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, grid: Sequence[float]
) -> np.ndarray:
    """Weighted quantiles of ``values`` at each grid point.

    Each value sits at the midpoint of its weight's span of the cumulative
    weight; quantiles interpolate linearly between those midpoints and are
    held at the smallest and largest value outside them. A stable sort
    keeps the result independent of the order of tied values.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    if values.ndim != 1 or values.shape != weights.shape or not len(values):
        raise ValueError("Weighted quantiles need matching, nonempty 1-D arrays.")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if not total > 0:
        raise ValueError("Weighted quantiles need a positive total weight.")
    midpoints = (cumulative - weights / 2.0) / total
    return np.interp(
        np.asarray(grid, dtype=float),
        midpoints,
        values,
        left=values[0],
        right=values[-1],
    )


def grouped_knots(
    values: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    *,
    grid: Sequence[float],
    cap: float,
    minimum_count: int,
    decimals: int = 2,
) -> tuple[list[GroupKnots], list[str]]:
    """Weighted quantile knots per group, winsorised at ``cap``.

    Groups come out in code order. A group with fewer than ``minimum_count``
    rows gets no knots; its code is returned in the second element. Each
    quantile is rounded to ``decimals`` places.
    """
    values = np.minimum(np.asarray(values, dtype=float), float(cap))
    weights = np.asarray(weights, dtype=float)
    codes = np.asarray(codes).astype(str)
    if not (values.shape == weights.shape == codes.shape):
        raise ValueError("Values, weights and codes must have one shape.")
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True]
    )
    knots: list[GroupKnots] = []
    suppressed: list[str] = []
    for begin, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        code = sorted_codes[begin]
        rows = order[begin:stop]
        count = int(stop - begin)
        if count < minimum_count:
            suppressed.append(str(code))
            continue
        quantiles = weighted_quantiles(values[rows], weights[rows], grid)
        knots.append(
            GroupKnots(
                code=str(code),
                count=count,
                weight=float(weights[rows].sum()),
                quantiles=tuple(float(round(q, decimals)) for q in quantiles),
            )
        )
    return knots, suppressed


def knot_mean(grid: Sequence[float], quantiles: Sequence[float]) -> float:
    """The mean of the quantile function that the knots interpolate linearly.

    It is the area under the piecewise-linear quantile function over the
    grid, which must run from 0 to 1 (:func:`check_knots`).
    """
    grid = np.asarray(grid, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if grid.shape != quantiles.shape:
        raise ValueError("Knots must match the grid.")
    return float(np.sum(np.diff(grid) * (quantiles[1:] + quantiles[:-1]) / 2.0))


def check_knots(
    grid: Sequence[float], quantiles: Sequence[float], *, cap: float
) -> None:
    """Refuse knots a quantile reader cannot use.

    The grid must run from 0 to 1 and rise strictly. The quantiles must match
    it, be finite, lie in [0, cap] and never fall.
    """
    grid = np.asarray(grid, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("The knot grid must run from 0 to 1.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("The knot grid must rise strictly.")
    if quantiles.shape != grid.shape:
        raise ValueError("The knots do not match the grid.")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("The knots must be finite.")
    if np.any(quantiles < 0) or np.any(quantiles > cap):
        raise ValueError("The knots must lie in [0, cap].")
    if np.any(np.diff(quantiles) < 0):
        raise ValueError("The knots must never fall.")
