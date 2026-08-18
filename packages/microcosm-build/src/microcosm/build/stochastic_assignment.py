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
    "sample_categorical_from_counts",
    "stable_identity_uniforms",
]


def stable_identity_uniforms(
    ids: Sequence[object] | np.ndarray,
    *,
    seed: int,
    salt: str,
) -> np.ndarray:
    """Return deterministic U[0,1) draws keyed by ``seed:salt:id``.

    Known hot spot: this is a per-id Python loop over blake2b, measured at
    ~1.8M ids/s. That is negligible at FRS spine scale (36k persons x 17
    columns is ~0.35s) and bounded but visible for a country-agnostic caller
    at local-area scale (1.6M ids x 17 columns is ~15s). If a consumer ever
    needs it faster, the fix is to hash into one contiguous bytes buffer and
    reinterpret it (``np.frombuffer(buf, dtype=">u8") / 2**64``) rather than
    building a Python list of floats; the per-id key string and digest size
    must stay byte-identical or every committed draw moves.
    """

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
) -> np.ndarray:
    """Assign a flag while forcing reported-recipient anchors to true.

    The target count is ``int(rate * n_units)`` over the full unweighted
    population. Anchored overshoot is accepted; the residual fills only
    non-anchored rows.
    """

    draws = np.asarray(draws, dtype=np.float64)
    rate = _validate_rate(rate)
    if anchor is None:
        return draws < rate
    anchor = np.asarray(anchor, dtype=bool)
    if anchor.shape != draws.shape:
        raise ValueError("anchor and draws must align")
    result = anchor.copy()
    target = int(rate * len(draws))
    remaining_needed = max(0, target - int(anchor.sum()))
    non_anchored = ~anchor
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
