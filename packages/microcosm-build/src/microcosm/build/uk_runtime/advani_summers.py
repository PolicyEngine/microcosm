"""The Advani and Summers capital-gains surface, shared by the UK CGT stages.

Advani and Summers (2020, CAGE Working Paper 465, Table A1) publish, for
sixty-one total-income bands, the share of individuals reporting any gain
and the 5th to 95th percentiles of gains among those who do. The surface
comes from all individuals who reported taxable gains on the SA108 form for
tax year 2017-18, net of in-year losses and excluding trusts; the amounts
are nominal pounds of that year and are used here as published, without
uprating (microcosm#970).

Three stages read it: the incidence clone draws each carrier's prior gain
from the band's quantile function, the band-donor stage samples donors by
the band's incidence, and the Table 3 amounts stage places the sub-exempt
remainder on the quantiles between the band's zero crossing and its
crossing of the annual exempt amount. This module is the one register of
the resource, its validation, the band lookup and the degree-1 quantile
function so the three stages cannot drift apart. It imports nothing from
the stage modules, which is what lets the amounts stage use it without a
cycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any

import numpy as np
from scipy.interpolate import UnivariateSpline

__all__ = [
    "ADVANI_SUMMERS_RESOURCE",
    "ADVANI_SUMMERS_VINTAGE",
    "CGT_PRIOR_PERCENTILE_COLUMNS",
    "CGT_QUANTILE_POINTS",
    "advani_summers_band_index",
    "advani_summers_knots",
    "advani_summers_quantile_function",
    "advani_summers_resource_sha256",
    "advani_summers_rows",
    "draw_banded_priors",
    "exempt_range_quantiles",
    "load_advani_summers_distribution",
    "quantile_for_amount",
    "stratified_amounts_within_range",
]

ADVANI_SUMMERS_RESOURCE = "advani_summers_capital_gains_distribution.json"
#: Tax year of the HMRC data behind CAGE WP 465 Table A1. Amounts are nominal
#: pounds of that year and are used as published; no uprating is applied.
ADVANI_SUMMERS_VINTAGE = "2017-18"
CGT_QUANTILE_POINTS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
CGT_PRIOR_PERCENTILE_COLUMNS = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")


def load_advani_summers_distribution() -> Mapping[str, Any]:
    """Load the committed Advani-Summers incidence and quantile surface."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath(ADVANI_SUMMERS_RESOURCE)
        .read_text(encoding="utf-8")
    )


def advani_summers_resource_sha256() -> str:
    """Digest of the committed resource bytes, for receipts."""

    return hashlib.sha256(
        files("microcosm.build.uk").joinpath(ADVANI_SUMMERS_RESOURCE).read_bytes()
    ).hexdigest()


def advani_summers_rows(resource: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The validated band rows of the surface, in ascending income order."""

    rows = resource.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Advani-Summers resource must contain a non-empty rows list.")
    minimums = [float(row["minimum_total_income"]) for row in rows]
    if minimums != sorted(minimums) or minimums[0] != 0.0:
        raise ValueError("Advani-Summers income bands must be sorted and start at 0.")
    # Fail closed on non-monotone quantile rows: the prior draw interpolates
    # and extrapolates these values as a quantile function, so a malformed
    # row (the class the corrected percentile-69 dropped digit belonged to)
    # would silently fabricate loss-makers instead of failing the build.
    for row in rows:
        knots = [float(row[column]) for column in CGT_PRIOR_PERCENTILE_COLUMNS]
        if any(late < early for early, late in zip(knots, knots[1:], strict=False)):
            raise ValueError(
                "Advani-Summers quantile columns must be non-decreasing; "
                f"row with minimum_total_income {row['minimum_total_income']!r} "
                "is not a valid quantile function."
            )
    return rows


def advani_summers_band_index(
    rows: Sequence[Mapping[str, Any]], income: np.ndarray
) -> np.ndarray:
    """Row index of each income's total-income band (last minimum not above it)."""

    minimums = np.asarray([row["minimum_total_income"] for row in rows], dtype=float)
    return np.clip(
        np.searchsorted(minimums, np.asarray(income, dtype=float), side="right") - 1,
        0,
        len(rows) - 1,
    )


def advani_summers_knots(row: Mapping[str, Any]) -> np.ndarray:
    """The row's published percentiles as the knots of its quantile function."""

    return np.asarray(
        [row[column] for column in CGT_PRIOR_PERCENTILE_COLUMNS], dtype=float
    )


def advani_summers_quantile_function(knots: np.ndarray) -> UnivariateSpline:
    """The degree-1 interpolating spline through the published percentiles.

    ``ext=0`` extrapolates linearly below the 5th and above the 95th
    percentile, which is how the incidence clone's prior draw has always
    read the surface.
    """

    return UnivariateSpline(CGT_QUANTILE_POINTS, knots, k=1, s=0, ext=0)


def quantile_for_amount(knots: np.ndarray, amount: float) -> float:
    """Invert the degree-1 quantile spline at ``amount``.

    Linear between knots and linear on the end segments beyond them, which
    is what ``UnivariateSpline(k=1, s=0, ext=0)`` evaluates, so
    ``spline(quantile_for_amount(knots, a)) == a`` to rounding. The segment
    holding ``amount`` must rise strictly: a flat one has no unique
    crossing, and the caller must not pretend it does.
    """

    points = np.asarray(CGT_QUANTILE_POINTS, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.shape != points.shape:
        raise ValueError(
            "Quantile knots must align with the published quantile points."
        )
    if amount <= knots[0]:
        segment = 0
    elif amount >= knots[-1]:
        segment = len(knots) - 2
    else:
        segment = int(
            np.clip(np.searchsorted(knots, amount, side="right") - 1, 0, len(knots) - 2)
        )
    y0, y1 = float(knots[segment]), float(knots[segment + 1])
    if not y1 > y0:
        raise ValueError(
            f"Quantile knots are flat on the segment holding {amount}; "
            "no unique quantile."
        )
    x0, x1 = float(points[segment]), float(points[segment + 1])
    return x0 + (float(amount) - y0) * (x1 - x0) / (y1 - y0)


def exempt_range_quantiles(
    knots: np.ndarray, annual_exempt_amount: float
) -> tuple[float, float]:
    """``(q0, q_aea)``: the spline's zero crossing and its crossing of the AEA."""

    if not annual_exempt_amount > 0.0:
        raise ValueError("The annual exempt amount must be positive.")
    lower = quantile_for_amount(knots, 0.0)
    upper = quantile_for_amount(knots, float(annual_exempt_amount))
    if not lower < upper:
        raise ValueError(
            "The band's quantile function does not rise between zero and the "
            "annual exempt amount."
        )
    return lower, upper


def stratified_amounts_within_range(
    knots: np.ndarray,
    *,
    weights: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    """Amounts at the midpoints of weight-proportional strata of the range.

    The persons arrive in the order their amounts must rise; the strata
    partition ``(lower_quantile, upper_quantile)`` in proportion to the
    household weights, and each person takes the spline at their stratum's
    midpoint, so the weighted set reproduces the published conditional shape
    on the range rather than an independent draw per person. Zero weights
    take a nominal stratum of the smallest positive weight (or one when no
    weight is positive): they carry no mass, and the nominal width only keeps
    them strictly inside the range.
    """

    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return np.zeros(0, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("Stratum weights must be finite and non-negative.")
    if not lower_quantile < upper_quantile:
        raise ValueError("The stratified range must have a positive width.")
    positive = weights[weights > 0.0]
    nominal = float(positive.min()) if positive.size else 1.0
    stratum = np.where(weights > 0.0, weights, nominal)
    cumulative = np.cumsum(stratum)
    upper = cumulative / float(cumulative[-1])
    lower = np.concatenate(([0.0], upper[:-1]))
    quantiles = lower_quantile + 0.5 * (lower + upper) * (
        upper_quantile - lower_quantile
    )
    return np.asarray(advani_summers_quantile_function(knots)(quantiles), dtype=float)


def draw_banded_priors(
    income: np.ndarray,
    draws: np.ndarray,
    *,
    distribution: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate each person's band quantile function at their seeded draw."""

    rows = advani_summers_rows(distribution)
    indexes = advani_summers_band_index(rows, income)
    values = np.zeros(len(income), dtype=float)
    for index, row in enumerate(rows):
        mask = indexes == index
        if not mask.any():
            continue
        spline = advani_summers_quantile_function(advani_summers_knots(row))
        values[mask] = spline(draws[mask])
    return values
