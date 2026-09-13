"""Weighted projection onto an observed signed total and declared lower bounds.

This numeric operation has no source, model, tax or execution authority. The
caller chooses the component meanings and scales and qualifies the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np


@dataclass(frozen=True)
class SignedTotalReconciliation:
    """Detached numeric results and diagnostics, with no input storage aliases."""

    components: tuple[str, ...]
    nonnegative: np.ndarray
    scales: np.ndarray
    raw_draws: np.ndarray
    anchors: np.ndarray
    values: np.ndarray
    adjustments: np.ndarray
    residuals: np.ndarray
    active_bounds: np.ndarray
    normalized_shift: np.ndarray
    objective: np.ndarray
    stationarity_residuals: np.ndarray
    active_dual_violations: np.ndarray
    atol: float
    rtol: float


def _require(condition, reason):
    if not condition:
        raise ValueError("SIGNED_RECONCILIATION_" + reason)


def _numbers(value, name):
    _require(not np.ma.isMaskedArray(value), name + "_MASKED")
    array = np.asarray(value)
    _require(array.dtype.kind in "fiu", name + "_NUMERIC")
    with np.errstate(over="raise", invalid="raise"):
        result = np.array(array, dtype=np.float64, copy=True)
    _require(np.isfinite(result).all(), name + "_FINITE")
    return result


def _tolerance(atol, rtol, magnitude):
    result = atol + rtol * magnitude
    _require(np.isfinite(result).all(), "TOLERANCE_OVERFLOW")
    return result


def _row(q, anchor, weights, nonnegative):
    active = np.zeros(len(q), dtype=bool)
    # Removing a negative bounded candidate can only lower the common shift.
    # A removed coordinate therefore cannot need releasing later. At least one
    # unrestricted coordinate remains free for every signed anchor.
    for _ in range(int(nonnegative.sum()) + 1):
        free = ~active
        shift = (anchor - fsum(q[free])) / fsum(weights[free])
        _require(np.isfinite(shift), "SHIFT_OVERFLOW")
        candidate = q.copy() if shift == 0 else q + shift * weights
        _require(np.isfinite(candidate).all(), "VALUE_OVERFLOW")
        newly_active = free & nonnegative & (candidate < 0)
        if not newly_active.any():
            return np.where(active, 0.0, candidate), active, shift
        active |= newly_active
    raise ValueError("SIGNED_RECONCILIATION_ACTIVE_SET_FAILED")


def reconcile_signed_total(
    draws,
    anchors,
    *,
    components: tuple[str, ...],
    nonnegative,
    scales,
    atol: float,
    rtol: float,
) -> SignedTotalReconciliation:
    """Minimize ``sum(((z - q) / scales)**2)`` with ``sum(z) = anchor``.

    ``draws`` has shape ``(..., k)`` and ``anchors`` exactly ``draws.shape[:-1]``;
    one vector and scalar anchor, multidimensional batches and empty batches
    are supported. Scales and the boolean lower-bound mask each have shape
    ``(k,)``. The unique component roster is explicit. At least one coordinate
    must be unrestricted; no nonzero-presence or sign restriction is inferred
    from a draw or an anchor. Scales must be finite and strictly positive.

    The active-set solve uses ``weights = (scales / max(scales))**2``. This
    common normalization preserves the objective's minimizer without squaring
    very large scales. A free coordinate is ``q + normalized_shift * weights``;
    a coordinate whose declared zero bound is active is zero. The signed anchor
    is never clamped or used as a divisor. Exact feasible inputs are unchanged.

    Float64 results must pass the explicit ``atol + rtol * magnitude`` checks:
    ``abs(anchor)`` for the sum residual; the maximum absolute draw, result and
    proposed adjustment for component stationarity and active dual feasibility.
    Nonfinite inputs, overflowing intermediates/objectives, weights lost to
    underflow, or results outside these tolerances are refused. No last-coordinate
    residual patch or rounding changes the computed optimum. Returned arrays
    are detached diagnostics, not immutable authority or source observations.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _reconcile(
                draws, anchors, components, nonnegative, scales, atol, rtol
            )
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("SIGNED_RECONCILIATION_NUMERIC_OVERFLOW") from error


def _reconcile(draws, anchors, components, nonnegative, scales, atol, rtol):
    q = _numbers(draws, "DRAWS")
    a = _numbers(anchors, "ANCHORS")
    s = _numbers(scales, "SCALES")
    _require(q.ndim >= 1 and q.shape[-1] > 0, "DRAW_SHAPE")
    k = q.shape[-1]
    _require(a.shape == q.shape[:-1], "ANCHOR_SHAPE")
    _require(s.shape == (k,) and (s > 0).all(), "SCALE_SHAPE_OR_SIGN")
    _require(
        type(components) is tuple
        and len(components) == k
        and all(type(c) is str and c and c.strip() == c for c in components)
        and len(set(components)) == k,
        "COMPONENT_ROSTER",
    )
    _require(not np.ma.isMaskedArray(nonnegative), "BOUND_MASKED")
    bound = np.array(nonnegative, copy=True)
    _require(bound.shape == (k,) and bound.dtype == np.dtype(bool), "BOUND_MASK")
    _require((~bound).any(), "SIGNED_COMPONENT_REQUIRED")
    tolerances = [_numbers(v, "TOLERANCE") for v in (atol, rtol)]
    _require(all(v.shape == () and v >= 0 for v in tolerances), "TOLERANCE")
    atol, rtol = (float(v) for v in tolerances)
    weights = (s / s.max()) ** 2
    _require((weights > 0).all(), "SCALE_RATIO_UNDERFLOW")
    batch_shape = q.shape[:-1]
    rows, totals = q.reshape(-1, k), a.reshape(-1)
    values = np.empty_like(rows)
    active = np.empty(rows.shape, dtype=bool)
    shifts = np.empty(len(rows))
    residuals = np.empty(len(rows))
    stationarity = np.empty(len(rows))
    dual_violation = np.empty(len(rows))
    for i, (row, anchor) in enumerate(zip(rows, totals, strict=True)):
        z, fixed, shift = _row(row, anchor, weights, bound)
        residual = fsum(z) - anchor
        _require(np.isfinite(residual), "RESIDUAL_OVERFLOW")
        _require(abs(residual) <= _tolerance(atol, rtol, abs(anchor)), "SUM_TOLERANCE")
        proposed = shift * weights
        magnitude = np.maximum.reduce([abs(row), abs(z), abs(proposed)])
        limit = _tolerance(atol, rtol, magnitude)
        free_error = abs((z - row) - proposed)
        active_error = np.maximum(row + proposed, 0.0)
        _require(
            (z[bound] >= 0).all()
            and (free_error[~fixed] <= limit[~fixed]).all()
            and (active_error[fixed] <= limit[fixed]).all(),
            "KKT_TOLERANCE",
        )
        values[i], active[i], shifts[i], residuals[i] = z, fixed, shift, residual
        stationarity[i] = np.max(free_error[~fixed], initial=0.0)
        dual_violation[i] = np.max(active_error[fixed], initial=0.0)
    values = values.reshape(q.shape)
    adjustment = values - q
    objective = np.sum((adjustment / s) ** 2, axis=-1)
    _require(np.isfinite(objective).all(), "OBJECTIVE_OVERFLOW")
    return SignedTotalReconciliation(
        components=components,
        nonnegative=bound,
        scales=s,
        raw_draws=q,
        anchors=a,
        values=values,
        adjustments=adjustment,
        residuals=residuals.reshape(batch_shape),
        active_bounds=active.reshape(q.shape),
        normalized_shift=shifts.reshape(batch_shape),
        objective=objective,
        stationarity_residuals=stationarity.reshape(batch_shape),
        active_dual_violations=dual_violation.reshape(batch_shape),
        atol=atol,
        rtol=rtol,
    )
