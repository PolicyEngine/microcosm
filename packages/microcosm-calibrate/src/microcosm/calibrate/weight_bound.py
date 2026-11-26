"""The weight bound: one contract, declared references, and a measurement.

**The contract** (microcosm#493, microcosm#1729). ``max_weight_ratio`` is a
hard bound on each record:

    no returned weight exceeds ``max_weight_ratio x reference``,

where a record's *reference* is its weight on the frame the solve was given,
which is its starting weight. Three things follow, and they hold for every
caller of :func:`microcosm.calibrate.calibrate`:

1. **The reference is the frame, never the warm start.** A solve that
   continues from ``warm_start_weights`` (an epoch-batched solve does so on
   every batch) keeps the bound against the frame's weights. Batches do not
   compound the bound.
2. **A new frame is a new reference.** A stage whose frame carries an earlier
   stage's calibrated weights bounds against those. Two stages at ratio ``R``
   then admit ``R ** 2`` against the first stage's starting weights. A build
   that chains stages must say which weights its bound is against, and should
   record the realized ratio against the first stage's weights too
   (:func:`weight_bound_report` with ``reference`` set to them).
3. **The bound is relative, so a heavy record gets a larger absolute bound.**
   In a pooled file where one set of rows starts ``k`` times heavier than
   another, its rows may reach ``k`` times the absolute weight of the others
   under the same ratio.

**A declared reference.** Point 3 is a choice, not a law. A build may want
the bound against a weight it declares instead of each record's own start:
one shared weight, or the mean weight of the record's area
(:func:`weight_bound_reference`). ``calibrate(max_weight_reference=...)``
enforces it: no returned weight exceeds ``max_weight_ratio x`` the declared
reference. Points 1 and 2 hold for a declared reference too. The frame's
weights still set the starting point, the mass and the L2 anchor; a record
that starts above its bound starts from the bound, with the mass it gives up
spread below the bounds. It is opt-in: without it the bound is against the
frame's weights, as before.

:func:`weight_bound_report` measures any weight vector against a declared
reference, report-only, so the effect of such a bound can be read from a
finished solve before a solve enforces it.

This module solves nothing and imports no solver code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "WEIGHT_BOUND_REFERENCES",
    "weight_bound_reference",
    "weight_bound_report",
]

#: The declared references, by name.
#:
#: ``own``: each record's own starting weight (the solver's bound).
#: ``mean``: the mean starting weight, one number for every record.
#: ``group_mean``: the mean starting weight of the record's group (an area).
WEIGHT_BOUND_REFERENCES = ("own", "mean", "group_mean")


def _checked(values, label: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if not np.isfinite(array).all() or (
        (array <= 0).any() if positive else (array < 0).any()
    ):
        raise ValueError(
            f"{label} must be finite and "
            f"{'strictly positive' if positive else 'non-negative'}."
        )
    return array


def weight_bound_reference(
    starting_weights: Sequence[float] | np.ndarray,
    kind: str = "own",
    *,
    groups: Sequence | np.ndarray | None = None,
) -> np.ndarray:
    """The reference weight of each record under a declared rule.

    ``own`` returns the starting weights. ``mean`` returns their mean for
    every record. ``group_mean`` returns, for each record, the mean starting
    weight of the records that share its ``groups`` label. Every reference is
    positive, so every starting weight must be.
    """

    if kind not in WEIGHT_BOUND_REFERENCES:
        raise ValueError(
            f"the weight bound reference must be one of {WEIGHT_BOUND_REFERENCES}, "
            f"got {kind!r}."
        )
    starting = _checked(starting_weights, "starting_weights", positive=True)
    if kind == "own":
        return starting.copy()
    if kind == "mean":
        return np.full_like(starting, starting.mean())
    if groups is None:
        raise ValueError("a group_mean reference needs each record's group.")
    labels = np.asarray(groups)
    if labels.shape != starting.shape:
        raise ValueError(
            f"{labels.shape} groups do not align with {starting.shape} weights."
        )
    return (
        pd.Series(starting)
        .groupby(labels.astype(str), sort=False)
        .transform("mean")
        .to_numpy(dtype=np.float64)
    )


def weight_bound_report(
    weights: Sequence[float] | np.ndarray,
    starting_weights: Sequence[float] | np.ndarray,
    *,
    max_weight_ratio: float,
    reference: Sequence[float] | np.ndarray | None = None,
) -> dict[str, object]:
    """How a weight vector sits against the bound, under one reference.

    ``reference`` defaults to the starting weights (the solver's bound). With
    a declared reference the report answers what a bound against it would
    mean, from weights a solve already returned:

    - ``realized_max_ratio``: the largest ``weight / reference``,
    - ``records_above_bound`` and ``weight_share_above_bound``: the records
      whose weight exceeds ``max_weight_ratio x reference``, and the share of
      the total weight they hold,
    - ``records_starting_above_bound``: the records whose *starting* weight
      already exceeds the bound, which a solve under that bound would have to
      pull down before it fits anything,
    - ``admits_the_starting_mass``: whether the bounds sum to at least the
      starting total, which a mass-conserving solve needs.

    Report-only: it measures and changes nothing.
    """

    if (
        isinstance(max_weight_ratio, bool)
        or not isinstance(max_weight_ratio, int | float)
        or not math.isfinite(max_weight_ratio)
        or max_weight_ratio <= 0
    ):
        raise ValueError(
            f"max_weight_ratio must be positive and finite, got {max_weight_ratio!r}."
        )
    weights = _checked(weights, "weights")
    starting = _checked(starting_weights, "starting_weights", positive=True)
    if weights.shape != starting.shape:
        raise ValueError(
            f"{weights.shape} weights do not align with {starting.shape} "
            "starting weights."
        )
    declared = reference is not None
    bound_reference = (
        _checked(reference, "reference", positive=True) if declared else starting
    )
    if bound_reference.shape != starting.shape:
        raise ValueError(
            f"{bound_reference.shape} reference weights do not align with "
            f"{starting.shape} starting weights."
        )
    bound = float(max_weight_ratio) * bound_reference
    total = float(weights.sum())
    above = weights > bound
    return {
        "reference": "declared" if declared else "own_starting_weight",
        "max_weight_ratio": float(max_weight_ratio),
        "records": int(weights.size),
        "realized_max_ratio": float((weights / bound_reference).max()),
        "records_above_bound": int(above.sum()),
        "weight_share_above_bound": (
            float(weights[above].sum() / total) if total > 0 else 0.0
        ),
        "records_starting_above_bound": int((starting > bound).sum()),
        "admits_the_starting_mass": bool(bound.sum() >= starting.sum()),
    }
