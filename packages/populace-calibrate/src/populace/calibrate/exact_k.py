"""Deterministic exact-cardinality probability-proportional selection.

The public :func:`select_exact_k` seam separates threshold certainties from a
fixed-size boundary draw. Records with hard-concrete open probability
``pi_i >= pi_hi`` are included first. If ``m`` places remain, positive boundary
scores are converted to target first-order inclusion probabilities
``q_i = m * pi_i / sum(pi_boundary)``. A boundary vector that would require
any ``q_i > 1`` fails as degenerate rather than silently clamping. A ``q_i``
within a few float64 ulps of one is normalized to a deterministic take-all
implied by the proportional design, distinct from the caller's declared
``pi_hi`` certainty split. The normalization keeps ``sum(q) = m`` to an
ulp-scale bound; it does not promise a bit-exact floating-point identity. The
full-pool census is also a deterministic special case.

The boundary design is Sampford probability-proportional-to-size sampling
without replacement. Sampford is used instead of naively conditioning
Bernoulli draws because the mathematical design preserves the feasible target
first-order inclusion probabilities ``q`` while fixing the draw size. The
implementation uses the equivalent Sampford subset law

``P(S) proportional to sum(1 - q_i, i in S) * product(q_i / (1 - q_i), i in S)``.

For bounded problems an exact suffix dynamic program draws directly from that
subset law, including numerically near-deterministic designs. Its two float64
tables cost ``16 * (N + 1) * (m + 1)`` bytes; the hard 256 MiB bound includes
at most that many table cells. Larger problems use the equivalent accept/reject
construction: generate Bernoulli ``q`` subsets, condition on the requested
count, and accept a candidate with probability
``sum(1 - q_i, i in S) / m``. Dispatch estimates both the Poisson-binomial
count probability and this Sampford acceptance tilt, using the dynamic program
for ill-conditioned designs when its memory bound permits. Rejection has a
deterministic attempt limit and fails closed rather than hanging. For a
majority draw the algorithm samples the complementary Sampford design with
targets ``1 - q``. See M. R. Sampford, "On sampling without replacement with
unequal probabilities of selection", *Biometrika* 54 (1967), 499--513,
doi:10.1093/biomet/54.3-4.499.

``group_ids`` is deliberately only a seam in this PR. When supplied it must be
one-dimensional, aligned with ``pi``, and contain one unique id per record;
duplicates fail closed. The spine-aware near-duplicate grouping policy belongs
to the increment-2 pool and lands there. ``group_ids=None`` means no grouping.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import numpy as np

__all__ = ["assert_exact_k_support", "select_exact_k"]

SelectionReceipt = dict[str, int | float | str]
_INDEX_DTYPE = np.dtype("<i8")

# Exact dynamic programming is robust for numerically concentrated designs but
# costs O(pool_size * sample_size) time and memory. Use it unconditionally for
# small tables. For a poorly conditioned rejection design it may grow to the
# hard memory ceiling: two float64 tables cost 16 bytes per suffix/sample-size
# cell, so 256 MiB permits 16,777,216 cells (including the extra suffix row).
_SAMPFORD_DP_ALWAYS_MAX_CELLS = 2_000_000
_SAMPFORD_DP_MAX_BYTES = 256 * 1024 * 1024
_SAMPFORD_DP_MAX_CELLS = _SAMPFORD_DP_MAX_BYTES // (2 * np.dtype(np.float64).itemsize)
_SAMPFORD_MAX_REJECTION_ATTEMPTS = 8_192
_SAMPFORD_REJECTION_SAFETY_FACTOR = 32.0
_PROBABILITY_ONE_TOLERANCE = 8.0 * np.finfo(np.float64).eps


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
        normalized = np.empty(0, dtype=_INDEX_DTYPE)
    else:
        if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
            indices.dtype, np.bool_
        ):
            raise ValueError("exact-k support indices must be integers.")
        normalized = np.asarray(indices, dtype=_INDEX_DTYPE).copy()
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
    if groups.dtype.kind in "fc" and np.isnan(groups).any():
        raise ValueError("group_ids cannot contain missing identifiers.")
    if groups.dtype.kind in "Mm" and np.isnat(groups).any():
        raise ValueError("group_ids cannot contain missing identifiers.")
    if groups.dtype.kind == "O":
        for group_id in groups:
            if group_id is None:
                raise ValueError("group_ids cannot contain missing identifiers.")
            try:
                reflexive = group_id == group_id
            except Exception:
                reflexive = False
            if not isinstance(reflexive, (bool, np.bool_)) or not bool(reflexive):
                raise ValueError(
                    "group_ids cannot contain missing or non-reflexive identifiers."
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

    dp_cells = (pool_size + 1) * (sample_size + 1)
    if dp_cells <= min(_SAMPFORD_DP_ALWAYS_MAX_CELLS, _SAMPFORD_DP_MAX_CELLS):
        return _sampford_dynamic_programming(probabilities, sample_size, rng)

    variance = float(np.sum(probabilities * (1.0 - probabilities), dtype=np.float64))
    # The local central-limit estimate for P(sum(Bernoulli(q)) == m) is
    # 1 / sqrt(2*pi*variance), capped at one for concentrated designs. The
    # unconditional expected numerator of Sampford's acceptance tilt is the
    # same variance, so variance / m is a useful conditioning estimate for the
    # second factor. It deliberately recognizes the case where the requested
    # count is likely but every likely subset has a vanishing acceptance tilt.
    if variance > 0.0:
        estimated_count_probability = min(
            1.0,
            1.0 / math.sqrt(2.0 * math.pi * variance),
        )
        estimated_tilt = min(1.0, variance / sample_size)
        estimated_acceptance = estimated_count_probability * estimated_tilt
    else:  # pragma: no cover - feasible interior probabilities have variance
        estimated_acceptance = 0.0
    expected_attempts = (
        math.inf if estimated_acceptance <= 0.0 else 1.0 / estimated_acceptance
    )

    if expected_attempts > _SAMPFORD_MAX_REJECTION_ATTEMPTS:
        if dp_cells <= _SAMPFORD_DP_MAX_CELLS:
            return _sampford_dynamic_programming(probabilities, sample_size, rng)
        dp_bytes = 2 * np.dtype(np.float64).itemsize * dp_cells
        raise RuntimeError(
            "Sampford sampling has no viable execution path: estimated "
            f"rejection work ({expected_attempts:.6g} attempts) exceeds the "
            f"{_SAMPFORD_MAX_REJECTION_ATTEMPTS}-attempt budget, and dynamic "
            f"programming requires {dp_cells} cells ({dp_bytes} bytes), "
            f"exceeding the {_SAMPFORD_DP_MAX_CELLS}-cell "
            f"({_SAMPFORD_DP_MAX_BYTES}-byte) memory bound."
        )

    max_attempts = min(
        _SAMPFORD_MAX_REJECTION_ATTEMPTS,
        max(256, math.ceil(_SAMPFORD_REJECTION_SAFETY_FACTOR * expected_attempts)),
    )
    for _ in range(max_attempts):
        selected = rng.random(pool_size) < probabilities
        if int(selected.sum()) != sample_size:
            continue
        selected_probabilities = probabilities[selected]
        acceptance_probability = float(
            np.sum(1.0 - selected_probabilities, dtype=np.float64) / sample_size
        )
        if float(rng.random()) < acceptance_probability:
            return np.flatnonzero(selected).astype(np.int64, copy=False)
    if dp_cells <= _SAMPFORD_DP_MAX_CELLS:
        return _sampford_dynamic_programming(probabilities, sample_size, rng)
    dp_bytes = 2 * np.dtype(np.float64).itemsize * dp_cells
    raise RuntimeError(
        "Sampford rejection sampling exhausted its "
        f"{max_attempts}-attempt budget (estimated acceptance probability "
        f"{estimated_acceptance:.6g}); dynamic programming cannot recover "
        f"because its {dp_cells} cells ({dp_bytes} bytes) exceed the "
        f"{_SAMPFORD_DP_MAX_CELLS}-cell ({_SAMPFORD_DP_MAX_BYTES}-byte) "
        "memory bound."
    )


def _sampford_dynamic_programming(
    probabilities: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw exactly from Sampford subset masses using suffix recurrences.

    For a suffix and subset size ``r``, ``elementary`` stores the sum of odds
    products and ``weighted`` additionally multiplies each product by its
    subset's ``sum(1 - q_i)`` factor. Rows are normalized together to avoid
    overflow; each inclusion decision compares entries from the same suffix
    row, so the common scale cancels.
    """
    pool_size = len(probabilities)
    odds = probabilities / (1.0 - probabilities)
    complement = 1.0 - probabilities
    shape = (pool_size + 1, sample_size + 1)
    elementary = np.zeros(shape, dtype=np.float64)
    weighted = np.zeros(shape, dtype=np.float64)
    elementary[pool_size, 0] = 1.0

    for index in range(pool_size - 1, -1, -1):
        max_size = min(sample_size, pool_size - index)
        elementary[index, 0] = elementary[index + 1, 0]
        weighted[index, 0] = weighted[index + 1, 0]
        for size in range(1, max_size + 1):
            elementary[index, size] = (
                elementary[index + 1, size]
                + odds[index] * elementary[index + 1, size - 1]
            )
            weighted[index, size] = weighted[index + 1, size] + odds[index] * (
                weighted[index + 1, size - 1]
                + complement[index] * elementary[index + 1, size - 1]
            )
        scale = max(
            float(np.max(elementary[index, : max_size + 1])),
            float(np.max(weighted[index, : max_size + 1])),
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(
                "Sampford dynamic program failed closed on a numerically "
                "ill-conditioned boundary design."
            )
        elementary[index, : max_size + 1] /= scale
        weighted[index, : max_size + 1] /= scale

    selected: list[int] = []
    accumulated_complement = 0.0
    for index in range(pool_size):
        remaining = sample_size - len(selected)
        if remaining == 0:
            break
        if pool_size - index == remaining:
            choose = True
        else:
            exclude_mass = (
                accumulated_complement * elementary[index + 1, remaining]
                + weighted[index + 1, remaining]
            )
            include_mass = odds[index] * (
                (accumulated_complement + complement[index])
                * elementary[index + 1, remaining - 1]
                + weighted[index + 1, remaining - 1]
            )
            total_mass = include_mass + exclude_mass
            if not math.isfinite(total_mass) or total_mass <= 0.0:
                raise RuntimeError(
                    "Sampford dynamic program failed closed while drawing from "
                    "a numerically ill-conditioned boundary design."
                )
            choose = float(rng.random()) < include_mass / total_mass
        if choose:
            selected.append(index)
            accumulated_complement += float(complement[index])

    if len(selected) != sample_size:  # pragma: no cover - recurrence invariant
        raise RuntimeError(
            "Sampford dynamic program violated its fixed-size invariant: "
            f"selected {len(selected)} != {sample_size}."
        )
    return np.asarray(selected, dtype=np.int64)


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
    positive_probabilities = boundary_probabilities[positive]
    total = float(np.sum(positive_probabilities, dtype=np.float64))
    if total <= 0.0:
        raise ValueError(
            "degenerate boundary mass: a positive remaining draw requires "
            "positive boundary pi mass."
        )
    target_probabilities = positive_probabilities * (sample_size / total)
    if (target_probabilities > 1.0 + _PROBABILITY_ONE_TOLERANCE).any():
        raise ValueError(
            "degenerate boundary mass: proportional normalization would require "
            "a boundary inclusion probability greater than one; "
            "adjust pi_hi or k."
        )

    # Extract mathematical exact-one entries before correcting the float64 sum.
    # Otherwise a positive summation residual can push the largest q just over
    # one and falsely reject a feasible design. This tolerance only resolves
    # ulp-scale representations of one; materially infeasible q values failed
    # above. Apply the residual to an interior entry with room in its direction.
    take_all = target_probabilities >= 1.0 - _PROBABILITY_ONE_TOLERANCE
    target_probabilities[take_all] = 1.0
    fractional_indices = np.flatnonzero(~take_all)
    remaining_draw = sample_size - int(np.count_nonzero(take_all))
    if remaining_draw and fractional_indices.size:
        fractional = target_probabilities[fractional_indices]
        residual = remaining_draw - math.fsum(float(value) for value in fractional)
        if residual >= 0.0:
            pivot_offset = int(np.argmin(fractional))
        else:
            pivot_offset = int(np.argmax(fractional))
        pivot = int(fractional_indices[pivot_offset])
        target_probabilities[pivot] += residual
    if (
        not np.isfinite(target_probabilities).all()
        or (target_probabilities <= 0.0).any()
        or (target_probabilities > 1.0).any()
    ):
        raise ValueError(
            "degenerate boundary mass: no feasible Sampford inclusion "
            "probabilities exist."
        )

    take_all_indices = np.flatnonzero(take_all)
    if remaining_draw == 0:
        selected_positive = take_all_indices
    else:
        selected_fractional = _sampford_core(
            target_probabilities[fractional_indices],
            remaining_draw,
            rng,
        )
        selected_positive = np.concatenate(
            (take_all_indices, fractional_indices[selected_fractional])
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
