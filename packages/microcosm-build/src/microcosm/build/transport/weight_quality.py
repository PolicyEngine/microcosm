"""Original-source aggregation for weight concentration diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def aggregate_source_weights(
    weights: Sequence[float] | np.ndarray,
    source_household_ids: Sequence[object] | np.ndarray,
) -> np.ndarray:
    """Sum clone weights by explicit source identity, preserving first-seen order.

    A unique identity vector returns the original vector without arithmetic,
    so concentration summaries are exactly equal when there are no clones.
    Empty, null, unhashable or misaligned identities fail closed.
    """

    values = np.asarray(weights, dtype=np.float64)
    identities = np.asarray(source_household_ids, dtype=object)
    if values.ndim != 1 or not values.size:
        raise ValueError("Weights must be a non-empty vector.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Weights must be finite and non-negative.")
    if identities.ndim != 1 or identities.size != values.size:
        raise ValueError("Source-household lineage must align with the weight vector.")
    if pd.isna(identities).any() or any(
        isinstance(value, str) and not value.strip() for value in identities
    ):
        raise ValueError(
            "Source-household lineage must not contain missing identities."
        )
    try:
        codes, unique = pd.factorize(identities, sort=False)
    except TypeError as exc:
        raise ValueError(
            "Source-household identities must be hashable scalars."
        ) from exc
    if unique.size == values.size:
        return values
    sums = np.zeros(unique.size, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        np.add.at(sums, codes, values)
    if not np.isfinite(sums).all():
        raise ValueError("Aggregated source-household weights must be finite.")
    return sums
