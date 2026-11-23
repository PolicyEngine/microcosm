"""Pure helpers for identity-keyed stochastic source assignments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import norm

__all__ = [
    "assign_binary_from_rate",
    "assign_binary_with_anchored_residual",
    "clipped_normal_from_uniforms",
    "integer_record_keys",
    "sample_categorical_from_counts",
    "stable_identity_uniforms",
    "weight_balanced_uniforms",
]


def stable_identity_uniforms(
    ids: Sequence[object] | np.ndarray,
    *,
    seed: int,
    salt: str,
) -> np.ndarray:
    """Return deterministic U[0,1) draws keyed by ``seed:salt:id``."""

    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{int(seed)}:{salt}:{value}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for value in ids
        ],
        dtype=np.float64,
    )


def weight_balanced_uniforms(
    strata: Sequence[object] | np.ndarray,
    draws: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    starts: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Spread records evenly over [0, 1) within each stratum, by weight.

    Records in a stratum are ordered by their draw, and each is placed at the
    midpoint of its slice of the stratum's cumulative weight, as
    policyengine-uk's ``stratified_uniform`` does (``utils/stochastic.py``).
    A rate then takes its share of every stratum's weight to within one
    record, where independent draws take it only in expectation. The result
    does not depend on the order of the records, and with identity-keyed
    draws it is as reproducible as they are. A stratum with no positive
    weight keeps its draws.

    Placed at midpoints alone, a stratum of one record always sits at 0.5, so
    a rate above a half always takes it and one below never does. ``starts``
    (aligned with the records, one value in [0, 1) per stratum) rotates each
    stratum's positions by its own start, systematic sampling with a random
    start: every record is then taken with probability equal to the rate, and
    a stratum's taken weight stays within one record of its share, half a
    record at each end of its arc (the independent review of
    microcosm#1420, finding 4).
    """

    strata = np.asarray(strata)
    draws = np.asarray(draws, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (strata.shape == draws.shape == weights.shape) or draws.ndim != 1:
        raise ValueError("strata, draws and weights must be aligned 1-D arrays.")
    if not np.isfinite(weights).all():
        raise ValueError("weights must be finite.")
    weights = np.maximum(weights, 0.0)
    if starts is not None:
        starts = np.asarray(starts, dtype=np.float64)
        if starts.shape != draws.shape:
            raise ValueError("starts must be aligned with the draws.")
        if not np.all((starts >= 0.0) & (starts < 1.0)):
            raise ValueError("starts must lie in [0, 1).")
    if draws.size == 0:
        return draws.copy()
    order = np.lexsort((draws, strata))
    _, group = np.unique(strata[order], return_inverse=True)
    sorted_weights = weights[order]
    totals = np.bincount(group, weights=sorted_weights)
    offsets = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
    within = np.cumsum(sorted_weights) - offsets[group] - sorted_weights / 2
    group_totals = totals[group]
    midpoints = within / np.where(group_totals > 0, group_totals, 1.0)
    if starts is not None:
        midpoints = np.mod(midpoints + starts[order], 1.0)
    positions = np.where(group_totals > 0, midpoints, draws[order])
    result = np.empty_like(positions)
    result[order] = positions
    return np.clip(result, 0.0, 1.0 - 2.0**-24)


def integer_record_keys(
    ids: Sequence[object] | np.ndarray, *, label: str
) -> np.ndarray:
    """Record ids to key ``stable_identity_uniforms`` on: integers, each once.

    The key string formats each id, so the integer ``1001`` and the float
    ``1001.0`` draw differently, and a merge that turned an id column into
    floats would silently redraw every record. A repeated id would give two
    records the same uniforms.
    """

    keys = np.asarray(ids)
    if keys.ndim != 1 or not np.issubdtype(keys.dtype, np.integer):
        raise TypeError(f"{label} must be a one-dimensional array of integer ids.")
    if np.unique(keys).size != keys.size:
        raise ValueError(f"{label} must name each record once.")
    return keys


def assign_binary_from_rate(
    draws: Sequence[float] | np.ndarray,
    rate: float,
) -> np.ndarray:
    """Assign a boolean flag from uniform draws and a scalar rate."""

    rate = _validate_rate(rate)
    return np.asarray(draws, dtype=np.float64) < rate


def assign_binary_with_anchored_residual(
    draws: Sequence[float] | np.ndarray,
    rate: float,
    anchor: Sequence[bool] | np.ndarray | None = None,
    population: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Assign a flag while forcing reported-recipient anchors to true.

    The target count is ``int(rate * n_units)`` over the unweighted
    population; ``population`` restricts that population to the rows the
    programme can reach (a benefit unit with no claimant or partner of working
    age cannot claim Universal Credit), so the rate is a share of the reachable
    rows and rows outside it are never drawn. Anchored overshoot is accepted,
    an anchor outside the population stays true (reported receipt is a fact),
    and the residual fills only non-anchored rows inside the population.
    """

    draws = np.asarray(draws, dtype=np.float64)
    rate = _validate_rate(rate)
    if population is None:
        population = np.ones(draws.shape, dtype=bool)
    else:
        population = np.asarray(population, dtype=bool)
        if population.shape != draws.shape:
            raise ValueError("population and draws must align")
    if anchor is None:
        return population & (draws < rate)
    anchor = np.asarray(anchor, dtype=bool)
    if anchor.shape != draws.shape:
        raise ValueError("anchor and draws must align")
    result = anchor.copy()
    target = int(rate * int(population.sum()))
    remaining_needed = max(0, target - int((anchor & population).sum()))
    non_anchored = ~anchor & population
    if remaining_needed == 0 or not non_anchored.any():
        return result
    adjusted = remaining_needed / int(non_anchored.sum())
    result |= non_anchored & (draws < adjusted)
    return result


def clipped_normal_from_uniforms(
    draws: Sequence[float] | np.ndarray,
    *,
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Map U[0,1) draws through a clipped normal inverse CDF."""

    if sd <= 0:
        raise ValueError("sd must be positive")
    values = norm.ppf(np.asarray(draws, dtype=np.float64), loc=mean, scale=sd)
    return np.clip(values, lower, upper)


def sample_categorical_from_counts(
    draws: Sequence[float] | np.ndarray,
    *,
    counts: Mapping[str, int | float],
) -> np.ndarray:
    """Sample category names by inverse CDF from a count mapping."""

    draws = np.asarray(draws, dtype=np.float64)
    if not counts:
        raise ValueError("count table cell is empty")
    categories: list[str] = []
    weights: list[float] = []
    for category, count in sorted(counts.items()):
        weight = float(count)
        if weight < 0:
            raise ValueError(f"{category!r} has a negative count")
        if weight > 0:
            categories.append(str(category))
            weights.append(weight)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("count table cell has no positive counts")
    cdf = np.cumsum(np.asarray(weights, dtype=np.float64)) / total
    indexes = np.searchsorted(cdf, draws, side="right")
    indexes = np.minimum(indexes, len(categories) - 1)
    return np.asarray([categories[index] for index in indexes], dtype=object)


def _validate_rate(rate: Any) -> float:
    value = float(rate)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {value}.")
    return value
