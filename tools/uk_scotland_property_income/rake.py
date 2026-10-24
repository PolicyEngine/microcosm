"""Bounded reweighting of household weights to a handful of facts (microcosm#1326 A4).

Each household's weight is scaled by a ratio ``g`` in ``[lower, upper]``. The
ratios minimise the squared relative misses on the facts plus a small
divergence from the starting weights,
``sum(w0 * (g log g - g + 1)) / sum(w0)``. When the facts can be met, the
divergence picks the solution closest to the starting weights. When they
can't be met within the bounds, it gives the least distorting compromise
instead of failing. L-BFGS-B searches over ``log g``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, minimize


@dataclass(frozen=True)
class RakeResult:
    weights: np.ndarray
    ratio: np.ndarray
    targets: np.ndarray
    before: np.ndarray
    after: np.ndarray
    converged: bool
    message: str
    #: L-BFGS-B's status: 0 converged, 1 stopped at the iteration limit, 2 stopped
    #: abnormally (a failed line search, for instance).
    status: int

    @property
    def stopped_at_iteration_limit(self) -> bool:
        return self.status == 1

    @property
    def relative_error_before(self) -> np.ndarray:
        return self.before / self.targets - 1.0

    @property
    def relative_error_after(self) -> np.ndarray:
        return self.after / self.targets - 1.0

    @property
    def effective_sample_size_ratio(self) -> float:
        """Kish effective sample size after, over before."""

        def kish(weights: np.ndarray) -> float:
            return float(weights.sum() ** 2 / np.square(weights).sum())

        return kish(self.weights) / kish(self.weights / self.ratio)


def rake(
    matrix: sparse.spmatrix,
    initial_weights: object,
    targets: object,
    *,
    lower: float = 0.25,
    upper: float = 4.0,
    divergence_weight: float = 1e-3,
    max_iterations: int = 20_000,
) -> RakeResult:
    """Ratios on ``initial_weights`` so that ``matrix.T @ weights`` meets ``targets``.

    ``matrix`` is households by facts. With few facts the misses reach zero
    long before the divergence stops shrinking, so the search can end at
    ``max_iterations`` with the facts met; the caller judges the misses.
    """

    a = sparse.csr_matrix(matrix, dtype=float)
    w0 = np.asarray(initial_weights, dtype=float)
    t = np.asarray(targets, dtype=float)
    if a.shape != (w0.size, t.size):
        raise ValueError(
            f"matrix {a.shape} does not match {w0.size} weights and {t.size} facts."
        )
    if not 0 < lower <= 1 <= upper:
        raise ValueError(f"Ratio bounds must contain 1: [{lower}, {upper}].")
    if np.any(w0 <= 0):
        raise ValueError("Starting weights must be positive.")
    if np.any(t <= 0):
        raise ValueError("Every fact must be positive.")

    scale = 1.0 / t
    total = w0.sum()
    a_t = a.T.tocsr()

    def objective(log_ratio: np.ndarray) -> tuple[float, np.ndarray]:
        ratio = np.exp(log_ratio)
        weights = w0 * ratio
        miss = (a_t @ weights - t) * scale
        divergence = np.sum(w0 * (ratio * log_ratio - ratio + 1.0)) / total
        value = float(miss @ miss + divergence_weight * divergence)
        gradient = weights * (a @ (2.0 * miss * scale))
        gradient += divergence_weight * w0 * ratio * log_ratio / total
        return value, gradient

    solution = minimize(
        objective,
        np.zeros(w0.size),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(np.full(w0.size, np.log(lower)), np.full(w0.size, np.log(upper))),
        options={"maxiter": max_iterations, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-12},
    )
    ratio = np.clip(np.exp(solution.x), lower, upper)
    weights = w0 * ratio
    return RakeResult(
        weights=weights,
        ratio=ratio,
        targets=t,
        before=a_t @ w0,
        after=a_t @ weights,
        converged=bool(solution.success),
        message=str(solution.message),
        status=int(solution.status),
    )
