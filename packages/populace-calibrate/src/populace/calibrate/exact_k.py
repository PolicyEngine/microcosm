"""Deterministic exact-cardinality probability-proportional selection.

The public :func:`select_exact_k` seam separates threshold certainties from a
fixed-size boundary draw. Records with hard-concrete open probability
``pi_i >= pi_hi`` are included first. If ``m`` places remain, positive boundary
scores are converted to target first-order inclusion probabilities
``q_i = m * pi_i / sum(pi_boundary)``. A boundary vector that would require
any ``q_i >= 1`` fails as degenerate rather than silently clamping or creating
certainty units outside the caller's declared ``pi_hi`` rule. The census and
only-positive-support cases are deterministic special cases.

The boundary design is Sampford probability-proportional-to-size sampling
without replacement. Sampford is used instead of naively conditioning
Bernoulli draws because it preserves the feasible target first-order inclusion
probabilities ``q`` exactly while fixing the draw size. The implementation uses
the equivalent Sampford subset law

``P(S) proportional to sum(1 - q_i, i in S) * product(q_i / (1 - q_i), i in S)``.

It generates Bernoulli ``q`` subsets, conditions on the requested count, and
accepts a candidate with probability ``sum(1 - q_i, i in S) / m``. For a
majority draw it samples the complementary Sampford design with targets
``1 - q``. See M. R. Sampford, "On sampling without replacement with unequal
probabilities of selection", *Biometrika* 54 (1967), 499--513,
doi:10.1093/biomet/54.3-4.499.

``group_ids`` is deliberately only a seam in this PR. When supplied it must be
one-dimensional, aligned with ``pi``, and contain one unique id per record;
duplicates fail closed. The spine-aware near-duplicate grouping policy belongs
to the increment-2 pool and lands there. ``group_ids=None`` means no grouping.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

__all__ = ["assert_exact_k_support", "select_exact_k"]

SelectionReceipt = dict[str, int | float | str]


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None
    result = int(result)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result!r}.")
    return result


def assert_exact_k_support(
    support: Sequence[int] | np.ndarray,
    k: int,
    *,
    pool_size: int | None = None,
) -> np.ndarray:
    """Fail closed unless ``support`` is a valid, unique exact-``k`` index set.

    This is the named cardinality gate shared by selection and frozen-support
    refit. It raises explicitly rather than relying on Python ``assert``, which
    can be disabled at runtime.
    """
    expected = _integer(k, name="k")
    indices = np.asarray(support)
    if indices.ndim != 1:
        raise ValueError(
            "exact-k cardinality gate requires one-dimensional support indices, "
            f"got shape {indices.shape}."
        )
    if len(indices) != expected:
        raise ValueError(
            "exact-k cardinality gate failed: "
            f"len(support)={len(indices)} != k={expected}."
        )
    if indices.size == 0:
        normalized = np.empty(0, dtype=np.int64)
    else:
        if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
            indices.dtype, np.bool_
        ):
            raise ValueError("exact-k support indices must be integers.")
        normalized = np.asarray(indices, dtype=np.int64).copy()
        if np.unique(normalized).size != normalized.size:
            raise ValueError("exact-k support indices must be unique.")

    if pool_size is not None:
        size = _integer(pool_size, name="pool_size")
        if normalized.size and ((normalized < 0).any() or (normalized >= size).any()):
            raise ValueError(
                "exact-k support indices must lie within the pool: "
                f"valid range is [0, {size}), got "
                f"min={int(normalized.min())}, max={int(normalized.max())}."
            )
    normalized.sort()
    return normalized


def _validate_group_ids(group_ids: np.ndarray | None, shape: tuple[int, ...]) -> None:
    if group_ids is None:
        return
    groups = np.asarray(group_ids)
    if groups.shape != shape:
        raise ValueError(
            "group_ids must be one-dimensional and aligned with pi: "
            f"got shape {groups.shape}, expected {shape}."
        )
    try:
        unique_count = np.unique(groups).size
    except TypeError:
        raise ValueError(
            "group_ids must contain mutually comparable scalar identifiers."
        ) from None
    if unique_count != groups.size:
        raise ValueError(
            "duplicate group_ids require the spine-aware near-duplicate policy; "
            "that policy is deferred to the increment-2 pool."
        )


def _categorical(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    draw = float(rng.random())
    index = int(np.searchsorted(cumulative, draw, side="right"))
    return min(index, len(probabilities) - 1)


def _sampford_core(
    probabilities: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw local indices from a feasible, non-certain Sampford design."""
    pool_size = len(probabilities)
    if sample_size == 0:
        return np.empty(0, dtype=np.int64)
    if sample_size == pool_size:
        return np.arange(pool_size, dtype=np.int64)
    if sample_size == 1:
        return np.asarray([_categorical(probabilities, rng)], dtype=np.int64)

    if sample_size > pool_size // 2:
        excluded = _sampford_core(
            1.0 - probabilities,
            pool_size - sample_size,
            rng,
        )
        included = np.ones(pool_size, dtype=bool)
        included[excluded] = False
        return np.flatnonzero(included).astype(np.int64, copy=False)

    while True:
        selected = rng.random(pool_size) < probabilities
        if int(selected.sum()) != sample_size:
            continue
        selected_probabilities = probabilities[selected]
        acceptance_probability = float(
            np.sum(1.0 - selected_probabilities, dtype=np.float64) / sample_size
        )
        if float(rng.random()) < acceptance_probability:
            return np.flatnonzero(selected).astype(np.int64, copy=False)


def _draw_boundary(
    boundary_probabilities: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positive = boundary_probabilities > 0.0
    positive_indices = np.flatnonzero(positive)
    if sample_size > len(positive_indices):
        raise ValueError(
            "degenerate boundary mass: fewer positive pi values than the "
            f"remaining draw size ({len(positive_indices)} < {sample_size})."
        )
    if sample_size == len(boundary_probabilities):
        return np.arange(len(boundary_probabilities), dtype=np.int64)
    if sample_size == len(positive_indices):
        return positive_indices.astype(np.int64, copy=False)

    positive_probabilities = boundary_probabilities[positive]
    total = float(np.sum(positive_probabilities, dtype=np.float64))
    if total <= 0.0:
        raise ValueError(
            "degenerate boundary mass: a positive remaining draw requires "
            "positive boundary pi mass."
        )
    target_probabilities = positive_probabilities * (sample_size / total)
    if (target_probabilities >= 1.0).any():
        raise ValueError(
            "degenerate boundary mass: proportional normalization would require "
            "a boundary inclusion probability greater than or equal to one; "
            "adjust pi_hi or k."
        )

    # Close the float64 summation residual at a deterministic pivot. This is a
    # one-ulp normalization correction, not probability clipping.
    pivot = int(np.argmax(target_probabilities))
    target_probabilities[pivot] += sample_size - float(
        np.sum(target_probabilities, dtype=np.float64)
    )
    if (
        not np.isfinite(target_probabilities).all()
        or (target_probabilities <= 0.0).any()
        or (target_probabilities >= 1.0).any()
    ):
        raise ValueError(
            "degenerate boundary mass: no feasible strictly between-zero-and-one "
            "Sampford inclusion probabilities exist."
        )

    selected_positive = _sampford_core(
        target_probabilities,
        sample_size,
        rng,
    )
    return positive_indices[selected_positive].astype(np.int64, copy=False)


def select_exact_k(
    pi: Sequence[float] | np.ndarray,
    k: int,
    pi_hi: float,
    seed: int,
    *,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, SelectionReceipt]:
    """Select exactly ``k`` records from hard-concrete open probabilities.

    The result is deterministic for identical ``(pi, k, pi_hi, seed,
    group_ids)`` inputs and uses a fresh NumPy ``Generator(PCG64(seed))``; global
    random state is never read or changed. Returned indices are sorted and the
    receipt contains JSON-safe scalar values suitable for a release manifest.

    See the module docstring for the Sampford design, boundary normalization,
    and the deliberately minimal ``group_ids`` contract.
    """
    try:
        probabilities = np.asarray(pi, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("pi must be a one-dimensional numeric vector.") from None
    if probabilities.ndim != 1:
        raise ValueError(
            f"pi must be one-dimensional, got shape {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("pi values must be finite.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("pi values must lie in [0, 1].")

    target = _integer(k, name="k")
    pool_size = len(probabilities)
    if target > pool_size:
        raise ValueError(
            f"k={target} exceeds the pool size {pool_size}; exact-k selection "
            "never clamps the requested cardinality."
        )
    try:
        certainty_threshold = float(pi_hi)
    except (TypeError, ValueError):
        raise ValueError(
            f"pi_hi must be a finite value in [0, 1], got {pi_hi!r}."
        ) from None
    if not np.isfinite(certainty_threshold) or not (0.0 <= certainty_threshold <= 1.0):
        raise ValueError(f"pi_hi must be a finite value in [0, 1], got {pi_hi!r}.")
    random_seed = _integer(seed, name="seed")
    _validate_group_ids(group_ids, probabilities.shape)

    certainty_mask = probabilities >= certainty_threshold
    certainty_indices = np.flatnonzero(certainty_mask).astype(np.int64, copy=False)
    certainty_count = len(certainty_indices)
    if target < certainty_count:
        raise ValueError(
            f"k={target} is smaller than the {certainty_count} certainty units "
            f"defined by pi_hi={certainty_threshold!r}; exact-k selection never "
            "drops declared certainty units."
        )

    boundary_indices = np.flatnonzero(~certainty_mask).astype(np.int64, copy=False)
    boundary_draw_size = target - certainty_count
    if boundary_draw_size == 0:
        support = certainty_indices.copy()
    elif boundary_draw_size == len(boundary_indices):
        support = np.concatenate((certainty_indices, boundary_indices))
    else:
        rng = np.random.Generator(np.random.PCG64(random_seed))
        selected_boundary = _draw_boundary(
            probabilities[boundary_indices],
            boundary_draw_size,
            rng,
        )
        support = np.concatenate(
            (certainty_indices, boundary_indices[selected_boundary])
        )

    support = assert_exact_k_support(support, target, pool_size=pool_size)
    receipt: SelectionReceipt = {
        "k": target,
        "pi_hi": certainty_threshold,
        "seed": random_seed,
        "certainty_count": certainty_count,
        "boundary_pool_size": len(boundary_indices),
        "design": "sampford",
    }
    return support, receipt
