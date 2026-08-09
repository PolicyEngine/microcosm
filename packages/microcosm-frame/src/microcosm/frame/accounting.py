"""Weighted accounting on a bundle: sums, means, quantiles, Gini, group sums.

Every function resolves the column's owning entity through the bundle (column
names are globally unique) and weights it with that entity's effective
weights: explicit entity weights when stored, the weighted group entity's
weights broadcast to persons for person-level columns, and member-constant
person-level weights collapsed per group for unweighted group entities.

Quantiles use the inverse-CDF method (the smallest value whose cumulative
weight proportion reaches ``q``, zero-weight rows excluded), and the Gini
index uses a trapezoidal Lorenz-curve approximation. NaN values propagate:
an aggregate over data containing NaN is NaN, never a silently partial
number.
"""

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from microcosm.frame.bundle import Frame

__all__ = ["wsum", "wmean", "wquantile", "wmedian", "gini", "groupby_wsum"]


def _resolve(
    bundle: Frame,
    column: str,
    entity: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(values, weights, entity)`` for a column.

    Args:
        bundle: The bundle to read from.
        column: Column to aggregate.
        entity: Owning entity, or ``None`` to infer it from the column name.

    Raises:
        ValueError: If the column is unknown, lives on a different entity
            than requested, or is not numeric/boolean.
    """
    owner = bundle.column_entity(column)
    if entity is not None and entity != owner:
        bundle.table(entity)  # validates the entity name
        raise ValueError(
            f"Column {column!r} lives on entity {owner!r}, not {entity!r}."
        )
    series = bundle.table(owner)[column]
    if not (is_numeric_dtype(series) or is_bool_dtype(series)):
        raise ValueError(
            f"Column {column!r} has non-numeric dtype {series.dtype}; "
            "weighted accounting requires numeric or boolean values."
        )
    values = series.to_numpy(dtype=np.float64)
    weights = bundle.resolve_weights(owner).values
    return values, weights, owner


def wsum(bundle: Frame, column: str, entity: str | None = None) -> float:
    """Weighted sum of ``column``: ``sum(values * weights)``.

    Args:
        bundle: The bundle to read from.
        column: Column to sum (numeric or boolean; booleans count weighted
            records).
        entity: Owning entity, or ``None`` to infer it.

    Returns:
        The weighted sum. NaN if the column contains NaN.
    """
    values, weights, _ = _resolve(bundle, column, entity)
    return float((values * weights).sum())


def wmean(bundle: Frame, column: str, entity: str | None = None) -> float:
    """Weighted mean of ``column``: ``sum(values * weights) / sum(weights)``.

    Args:
        bundle: The bundle to read from.
        column: Column to average.
        entity: Owning entity, or ``None`` to infer it.

    Returns:
        The weighted mean. NaN if the column contains NaN.
    """
    values, weights, _ = _resolve(bundle, column, entity)
    return float((values * weights).sum() / weights.sum())


def wquantile(
    bundle: Frame,
    column: str,
    q: float | np.ndarray,
    entity: str | None = None,
) -> float | pd.Series:
    """Weighted quantile(s) of ``column`` by the inverse-CDF method.

    The ``q``-th quantile is the smallest value whose cumulative weight
    proportion is at least ``q``. Zero-weight rows are excluded before
    sorting, so they can never be selected.

    Args:
        bundle: The bundle to read from.
        column: Column to take quantiles of.
        q: Quantile(s) in ``[0, 1]``; a scalar returns a float, an array
            returns a Series indexed by ``q``.
        entity: Owning entity, or ``None`` to infer it.

    Returns:
        The weighted quantile value(s).

    Raises:
        ValueError: If any requested quantile lies outside ``[0, 1]``.
    """
    values, weights, _ = _resolve(bundle, column, entity)
    quantiles = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if (quantiles < 0).any() or (quantiles > 1).any():
        raise ValueError(f"Quantiles must lie in [0, 1], got {quantiles.tolist()}.")
    scalar = np.asarray(q).shape == ()

    # NaN propagates, as the module promises: a quantile over data containing
    # a NaN is NaN for every q, never a silently partial number (NaN sorts
    # last, so without this guard it would be treated as +inf and the missing
    # value's weight would still inflate the denominator).
    if np.isnan(values).any():
        result = np.full(quantiles.shape, np.nan)
        if scalar:
            return float(result[0])
        return pd.Series(result, index=quantiles)

    nonzero = weights > 0
    values = values[nonzero]
    weights = weights[nonzero]
    sorter = np.argsort(values)
    values = values[sorter]
    cumulative = np.cumsum(weights[sorter])
    proportions = cumulative / cumulative[-1]
    positions = np.minimum(np.searchsorted(proportions, quantiles), len(values) - 1)
    result = values[positions]
    if scalar:
        return float(result[0])
    return pd.Series(result, index=quantiles)


def wmedian(bundle: Frame, column: str, entity: str | None = None) -> float:
    """Weighted median of ``column`` (the 0.5 weighted quantile)."""
    result = wquantile(bundle, column, 0.5, entity)
    return float(result)


def gini(
    bundle: Frame,
    column: str,
    entity: str | None = None,
    negatives: str | None = None,
) -> float:
    """Weighted Gini index of ``column``.

    Computed as a trapezoidal approximation of the area under the Lorenz
    curve. Degenerate cases short-circuit: a zero total returns ``0.0``.

    Args:
        bundle: The bundle to read from.
        column: Column to compute inequality of.
        entity: Owning entity, or ``None`` to infer it.
        negatives: How to treat negative values: ``"zero"`` replaces them
            with zero, ``"shift"`` adds the absolute minimum to every value
            when the minimum is negative, ``None`` leaves them (the
            Lorenz-based formula then is not guaranteed to lie in ``[0, 1]``
            and a ``UserWarning`` is emitted).

    Returns:
        The Gini index: ``0.0`` for equal values, approaching ``1.0`` as one
        record holds everything.

    Raises:
        ValueError: If ``negatives`` is not ``"zero"``, ``"shift"``, or
            ``None``.
    """
    values, weights, _ = _resolve(bundle, column, entity)
    if negatives == "zero":
        values = np.where(values < 0, 0.0, values)
    elif negatives == "shift" and values.size > 0 and np.amin(values) < 0:
        values = values - np.amin(values)
    elif negatives not in (None, "shift"):
        raise ValueError(
            f"Unknown negatives option {negatives!r}; expected 'zero', "
            "'shift', or None."
        )
    if (values < 0).any():
        warnings.warn(
            "gini() called on data containing negative values; the result is "
            "not guaranteed to lie in [0, 1]. Pass negatives='zero' or "
            "negatives='shift' to handle them.",
            UserWarning,
            stacklevel=2,
        )

    total = float((values * weights).sum())
    if total == 0:
        return 0.0
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative_weight = np.cumsum(sorted_weights)
    cumulative_mass = np.cumsum(sorted_values * sorted_weights)
    return float(
        np.sum(
            cumulative_mass[1:] * cumulative_weight[:-1]
            - cumulative_mass[:-1] * cumulative_weight[1:]
        )
        / (cumulative_mass[-1] * cumulative_weight[-1])
    )


def groupby_wsum(
    bundle: Frame,
    column: str,
    by: str,
    entity: str | None = None,
) -> pd.Series:
    """Weighted sum of ``column`` within each group of ``by``.

    The grouping column must live on the same entity as ``column``, or — when
    ``column`` is person-level — on a group entity, in which case it is
    broadcast onto persons through membership.

    Args:
        bundle: The bundle to read from.
        column: Column to sum.
        by: Grouping column.
        entity: Owning entity of ``column``, or ``None`` to infer it.

    Returns:
        Weighted sums indexed by the sorted distinct ``by`` values, named
        ``column``. A group containing NaN sums to NaN.

    Raises:
        ValueError: If ``by`` cannot be aligned to ``column``'s entity.
    """
    values, weights, owner = _resolve(bundle, column, entity)
    by_owner = bundle.column_entity(by)
    person_entity = bundle.schema.person_entity
    if by_owner == owner:
        keys = bundle.table(owner)[by].to_numpy()
    elif owner == person_entity:
        keys = bundle.broadcast(by).to_numpy()
    else:
        raise ValueError(
            f"Cannot group {column!r} on entity {owner!r} by {by!r} on entity "
            f"{by_owner!r}; the grouping column must live on the same entity "
            "or broadcast group -> person."
        )
    labels, codes = np.unique(keys, return_inverse=True)
    sums = np.zeros(labels.size, dtype=np.float64)
    np.add.at(sums, codes, values * weights)
    result = pd.Series(sums, index=labels, name=column)
    result.index.name = by
    return result
