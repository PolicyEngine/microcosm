"""The calibration solver: targets -> calibrated weights on a Frame.

:func:`calibrate` is the representation operator. It compiles a
:class:`~populace.calibrate.target.TargetSet` against a
:class:`~populace.frame.Frame` (:mod:`populace.calibrate.matrix`), then optimizes
the weight vector of ``weight_entity`` to minimize the bounded relative-error loss

    ``mean(((A @ w - b) / (b + 1))**2)``

with torch's Adam over the **log-weights** (so weights stay strictly positive by
construction). It returns a
:class:`CalibrationResult` carrying a new frame whose ``weight_entity`` weights
are :class:`~populace.frame.WeightKind.CALIBRATED`, per-target diagnostics, and
the loss trajectory.

Four declared options, each a real feature (and each its own test):

- ``mass="free"`` (default) lets the total weight move to fit the targets;
  ``mass="conserve"`` projects every step's weights back to the input total, so
  the calibrated population conserves the starting mass exactly.
- ``max_weight_ratio`` is a **hard** per-record bound: no calibrated weight may
  exceed ``max_weight_ratio * initial_weight``. It is clamped after every step —
  the documented guard against the tail "landmine" (a rare high-value,
  near-zero-weight donor whose weight detonates on reweight and blows up an
  aggregate).
- ``target_records`` turns on hard-concrete L0 gates
  (:mod:`populace.calibrate.gates`) with **budget control**: the solver searches
  ``l0_lambda`` (a bisection on its log) so the achieved non-zero count tracks
  ``target_records``, and reports the penalty it settled on — the
  generate-big-then-prune path. A supplied ``l0_lambda`` is the search's warm
  start.
- ``l0_lambda`` alone (no ``target_records``) prunes at a fixed penalty: ``> 0``
  gates the pool, ``0.0`` keeps every record. It is the sole control when no
  budget is given.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from populace.calibrate.gates import HardConcrete
from populace.calibrate.matrix import (
    CalibrationProblem,
    SkippedTarget,
    _linearization_weights,
    build_constraint_matrix,
)
from populace.calibrate.target import TargetSet
from populace.frame import Frame, MassChange, WeightKind, Weights

__all__ = [
    "calibrate",
    "relative_error_loss",
    "CalibrationResult",
    "TargetDiagnostic",
    "FREE_MASS",
    "CONSERVE_MASS",
]

#: ``mass="free"`` — the total weight may move to fit the targets (the default).
FREE_MASS = "free"

#: ``mass="conserve"`` — project the weights to the input total every step, so
#: the calibrated population conserves the starting mass exactly.
CONSERVE_MASS = "conserve"

#: Threshold below which a weight counts as pruned (a "zero") when reporting the
#: non-zero record count for the L0 path. Relative to the *initial* mean weight.
_PRUNE_REL_ATOL = 1e-6

#: Bracket for the ``l0_lambda`` budget search (Finding 3). The achieved non-zero
#: count is monotone decreasing in ``l0_lambda``; this bracket spans
#: "essentially no pruning" to "prune almost everything" across sparse-weight
#: regimes. The search bisects on ``log10(l0_lambda)`` inside it.
_L0_SEARCH_LO = 1e-7
_L0_SEARCH_HI = 1e1

#: Number of outer iterations (full optimizations) the budget search may spend
#: bisecting ``l0_lambda``. Each iteration is one ``_optimize`` run, so this caps
#: the search's cost at ``budget_iters`` optimizations; ~10 bisection steps cut
#: the ``log10`` bracket by 2^10, far finer than the count is resolvable.
_DEFAULT_BUDGET_ITERS = 10


@dataclass(frozen=True)
class TargetDiagnostic:
    """Per-target calibration diagnostics.

    Attributes:
        name: The target's ``"name@period"`` row label.
        target: The target value aimed at. For ``sum``/``count`` this is the
            compiled right-hand side ``b``; for ``mean`` it is the user's declared
            target mean (not the linearization's shifted right-hand side).
        initial_estimate: The achieved aggregate under the input weights —
            ``row @ w0`` for ``sum``/``count``, the true ratio
            ``sum(measure*filter*w0)/sum(filter*w0)`` for ``mean``.
        final_estimate: The achieved aggregate under the calibrated weights —
            ``row @ w`` for ``sum``/``count``, the true ratio under ``w`` for
            ``mean`` (the achieved mean, not the linearized row value).
        relative_error: ``(final_estimate - target) / target`` (or
            ``final_estimate - target`` when ``target`` is zero, since the
            relative form is undefined there). For ``mean`` this is the true
            ratio's relative miss.
        within_tolerance: Whether ``|final_estimate - target|`` is within the
            target's declared tolerance (the true achieved value for ``mean``).
            ``None`` when the target declared no tolerance.
    """

    name: str
    target: float
    initial_estimate: float
    final_estimate: float
    relative_error: float
    within_tolerance: bool | None


@dataclass(frozen=True)
class CalibrationResult:
    """The output of :func:`calibrate`.

    Attributes:
        frame: A new :class:`~populace.frame.Frame` whose ``weight_entity``
            weights are :class:`~populace.frame.WeightKind.CALIBRATED`.
        weight_entity: The entity whose weights were calibrated.
        weights: The calibrated weight values (also on :attr:`frame`).
        initial_weights: The input weight values (the starting point).
        diagnostics: Per-target :class:`TargetDiagnostic`, aligned to the
            compiled problem rows.
        loss_trajectory: The loss at each epoch (length ``epochs``).
        skipped: Targets that could not be compiled (carried through from the
            matrix build), each with its reason.
        problem: The compiled :class:`CalibrationProblem` (matrix, b, names).
        l0_lambda: The L0 penalty actually applied (0.0 when no pruning). When a
            ``target_records`` budget was set, this is the penalty the budget
            search settled on, not the value passed in.
        n_nonzero: Number of calibrated weights above the prune threshold. With a
            ``target_records`` budget, the quantity the search drives toward it.
        closing_loss: The bounded relative-error loss evaluated once on the
            *returned* weights (after the closing mass/cap projections). Exposed
            as :attr:`final_loss`; recorded separately from the trajectory, whose
            tail is a pre-step/pre-projection value.
        options: The solver configuration as passed (method, epochs,
            learning_rate, mass, max_weight_ratio, target_records, seed) plus
            the realized ``matrix_format`` (``"dense"`` or ``"sparse_csr"``).
            This is what a build records in its release manifest — the
            max_weight_ratio bound is part of the dataset's provenance, not a
            local solver detail.

    The result is a frozen record; the calibrated frame is the primary product
    and every operator downstream consumes :attr:`frame`.
    """

    frame: Frame
    weight_entity: str
    weights: np.ndarray
    initial_weights: np.ndarray
    diagnostics: tuple[TargetDiagnostic, ...]
    loss_trajectory: np.ndarray
    skipped: tuple[SkippedTarget, ...]
    problem: CalibrationProblem
    l0_lambda: float
    n_nonzero: int
    closing_loss: float
    options: Mapping[str, object] = field(default_factory=dict)

    @property
    def initial_loss(self) -> float:
        """The bounded relative-error loss under the input weights."""
        return float(self.loss_trajectory[0])

    @property
    def final_loss(self) -> float:
        """The bounded relative-error loss of the *returned* weights.

        This is a single eval-mode evaluation on the weights actually returned —
        after the closing mass/cap projections — so it describes the calibrated
        vector. It is **not** ``loss_trajectory[-1]``: the trajectory records each
        epoch's loss before that epoch's step and before the closing projections,
        so its tail can differ (e.g. under ``mass="conserve"`` with a cap).
        """
        return self.closing_loss

    @property
    def fraction_within_10pct(self) -> float:
        """Share of targets whose final relative error is within 10%.

        A summary of representation quality: the fraction of compiled targets
        the calibrated weights reproduce to within 10% (in relative terms, or
        absolute terms for a zero-valued target).
        """
        if not self.diagnostics:
            return 0.0
        hits = sum(abs(d.relative_error) <= 0.10 for d in self.diagnostics)
        return hits / len(self.diagnostics)


#: Density above which a sparse matrix gains nothing over dense compute.
_SPARSE_DENSITY_CUTOFF = 0.25
#: Matrices smaller than this (cells) stay dense; sparse kernels have
#: per-call overhead that only pays off at scale.
_SPARSE_MIN_CELLS = 1_000_000


def _torch_constraint_matrix(matrix) -> torch.Tensor:
    """Torch operator for ``A`` (targets x records): sparse CSR when it pays.

    The dense path materializes ``A`` as a float32 tensor — fine for small
    problems, fatal at national scale (3,704 x 75k is ~1.1 GB; a 3M-record
    pool would need ~44 GB). Above :data:`_SPARSE_MIN_CELLS` cells and below
    :data:`_SPARSE_DENSITY_CUTOFF` density, the scipy CSR converts directly
    to a torch sparse-CSR tensor and every epoch runs SpMM instead; autograd
    flows to the dense weight operand.
    """
    cells = int(matrix.shape[0]) * int(matrix.shape[1])
    density = (matrix.nnz / cells) if cells else 1.0
    if cells >= _SPARSE_MIN_CELLS and density <= _SPARSE_DENSITY_CUTOFF:
        as_f32 = matrix.astype(np.float32)
        return torch.sparse_csr_tensor(
            torch.from_numpy(np.asarray(as_f32.indptr, dtype=np.int64)),
            torch.from_numpy(np.asarray(as_f32.indices, dtype=np.int64)),
            torch.from_numpy(as_f32.data),
            size=as_f32.shape,
        )
    return torch.tensor(matrix.toarray(), dtype=torch.float32)


def _apply_constraint(matrix: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``A @ w`` for a dense or sparse-CSR ``A`` (targets x records)."""
    if matrix.layout == torch.sparse_csr:
        # SpMM needs a 2-D dense operand; SpMV is not exposed with autograd.
        return (matrix @ weights.unsqueeze(1)).squeeze(1)
    return matrix @ weights


def relative_error_loss(
    estimates: np.ndarray,
    targets: np.ndarray,
    *,
    target_loss_weights: np.ndarray | None = None,
) -> float:
    """THE loss, in numpy: weighted ``((est - tgt)/(tgt + 1))**2``.

    The single canonical definition every measurement imports — the solver's
    closing loss, the acceptance gates, and scorers all call this function
    (the torch twin below is the autograd path of the same formula). Refuses
    non-finite inputs: a NaN estimate is a harness bug, not a large miss.
    When ``target_loss_weights`` is omitted, this is the historical unweighted
    mean. When supplied, weights must align to targets and are normalized by
    their own sum, so multiplying all weights by a constant does not change the
    objective.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if estimates.shape != targets.shape:
        raise ValueError(
            f"estimates and targets must align, got shapes "
            f"{estimates.shape} vs {targets.shape}."
        )
    if not (np.isfinite(estimates).all() and np.isfinite(targets).all()):
        raise ValueError(
            "relative_error_loss requires finite inputs; got non-finite "
            "estimate or target values."
        )
    rel = (estimates - targets) / (targets + 1.0)
    loss = rel**2
    weights = _validate_target_loss_weights(target_loss_weights, targets.shape)
    if weights is None:
        return float(loss.mean())
    return float(np.average(loss, weights=weights))


def _relative_error_loss(
    estimate: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
) -> torch.Tensor:
    """The relative-error loss, optionally averaged with target row weights.

    The ``+1`` in the *denominator* is the regularizer: it keeps the loss finite
    and well-scaled for targets near zero (a zero-valued count target then
    contributes ``est**2`` rather than dividing by zero). The numerator is the
    raw residual ``est - tgt``, so the loss is minimized exactly at ``est = tgt``.

    Some older reweighting formulas carried a ``+1`` in the numerator too. That
    biases the optimum to ``est = tgt - 1`` and is fatal for small-valued
    targets (a count of 5 converges to 4). We use the raw residual — this is
    also the loss this docstring has always described.
    """
    rel_error = (estimate - targets) / (targets + 1.0)
    loss = rel_error**2
    if target_loss_weights is None:
        return loss.mean()
    return (loss * target_loss_weights).sum() / target_loss_weights.sum()


def _validate_target_loss_weights(
    target_loss_weights: np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if target_loss_weights is None:
        return None
    weights = np.asarray(target_loss_weights, dtype=np.float64)
    if weights.shape != shape:
        raise ValueError(
            "target_loss_weights must align with targets, got shapes "
            f"{weights.shape} vs {shape}."
        )
    if not np.isfinite(weights).all():
        raise ValueError("target_loss_weights must be finite.")
    if (weights < 0).any():
        raise ValueError("target_loss_weights must be non-negative.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("target_loss_weights must have positive total weight.")
    return weights


def _target_loss_weight_options(
    target_loss_weights: np.ndarray | None,
) -> Mapping[str, object]:
    if target_loss_weights is None:
        return {"kind": "uniform"}
    weights = np.asarray(target_loss_weights, dtype=np.float64)
    return {
        "kind": "provided",
        "n": int(weights.shape[0]),
        "sum": float(weights.sum()),
        "min": float(weights.min()),
        "max": float(weights.max()),
    }


def _build_diagnostics(
    problem: CalibrationProblem,
    frame: Frame,
    initial_weights: np.ndarray,
    final_weights: np.ndarray,
) -> tuple[TargetDiagnostic, ...]:
    """Assemble per-target diagnostics from the problem and both weight vectors.

    ``sum``/``count`` rows are exactly ``A @ w``, so their estimates and target
    come straight from the compiled system. A ``mean`` row is only the *linearized*
    value about the input weights — reporting ``A @ w`` for it after a large mass
    move reads as a perfect hit even when the achieved ratio missed (Finding 6).
    For ``mean`` targets the diagnostic instead reports the true ratio
    ``sum(measure*filter*w)/sum(filter*w)`` under each weight vector, against the
    user's declared target value (not the linearization's shifted right-hand
    side).
    """
    initial_est = problem.estimates(initial_weights)
    final_est = problem.estimates(final_weights)
    weight_entity = problem.weight_entity
    diagnostics: list[TargetDiagnostic] = []
    for i, target in enumerate(problem.targets):
        if target.aggregation == "mean":
            # True achieved ratio under the (entity-aligned) weights, against the
            # user's declared mean value.
            tgt = float(target.value)
            w0_aligned = _linearization_weights(
                target, frame, initial_weights, weight_entity
            )
            w_aligned = _linearization_weights(
                target, frame, final_weights, weight_entity
            )
            initial_value = target.achieved_value(frame, w0_aligned)
            final = target.achieved_value(frame, w_aligned)
        else:
            tgt = float(problem.target_vector[i])
            initial_value = float(initial_est[i])
            final = float(final_est[i])
        if tgt != 0.0:
            rel = (final - tgt) / tgt
        else:
            rel = final - tgt
        within: bool | None
        if target.tolerance is None:
            within = None
        else:
            within = abs(final - tgt) <= target.tolerance
        diagnostics.append(
            TargetDiagnostic(
                name=problem.names[i],
                target=tgt,
                initial_estimate=float(initial_value),
                final_estimate=float(final),
                relative_error=float(rel),
                within_tolerance=within,
            )
        )
    return tuple(diagnostics)


def _optimize(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    initial_weights: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    l0_lambda: float,
    target_records: int | None,
    init_mean: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the torch optimization and return ``(final_weights, loss_trajectory)``.

    Optimizes the log-weights with Adam against the bounded relative-error loss.
    Positivity is by construction (``w = exp(log_w)`` times optional gates). The
    hard constraints — mass conservation and ``max_weight_ratio`` — are applied
    by projecting the realized weights after each step, so they hold on the
    returned vector exactly, not merely in expectation.
    """
    w0 = np.asarray(initial_weights, dtype=np.float64)
    total0 = float(w0.sum())
    # Same prune threshold the result's n_nonzero uses, so "pruned" here means
    # exactly what the reported non-zero count means.
    prune_atol = _PRUNE_REL_ATOL * float(np.mean(w0))
    log_w = torch.tensor(np.log(w0), dtype=torch.float32, requires_grad=True)

    gates: HardConcrete | None = None
    params: list[torch.Tensor] = [log_w]
    if l0_lambda > 0.0 or target_records is not None:
        gates = HardConcrete(len(w0), init_mean=init_mean, temperature=temperature)
        params = [log_w, *gates.parameters()]

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    upper = (
        torch.tensor(max_weight_ratio * w0, dtype=torch.float32)
        if max_weight_ratio is not None
        else None
    )

    trajectory = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = torch.exp(log_w)
        if gates is not None:
            weights = weights * gates()
        estimate = _apply_constraint(matrix, weights)
        loss = _relative_error_loss(estimate, targets, target_loss_weights)
        penalty = (
            l0_lambda * gates.get_penalty()
            if (gates is not None and l0_lambda > 0.0)
            else torch.zeros((), dtype=torch.float32)
        )
        total_loss = loss + penalty
        trajectory[epoch] = float(loss.item())
        total_loss.backward()
        optimizer.step()

        # Hard projections, applied to the realized weights every step so the
        # guarantees hold on the returned vector, not just in expectation.
        with torch.no_grad():
            if upper is not None:
                # Clamp log-weights so exp(log_w) <= max_weight_ratio*w0. This is
                # the landmine guard: a rare high-value near-zero-weight record
                # can never be inflated past its bound.
                log_w.clamp_(max=torch.log(upper))
            if conserve_mass:
                realized = torch.exp(log_w)
                if gates is not None:
                    realized = realized * gates()
                current_total = float(realized.sum().item())
                if current_total > 0:
                    log_w.add_(float(np.log(total0 / current_total)))
                    if upper is not None:
                        # Re-clamp: the rescale may have pushed a record over its
                        # bound. The mass invariant is held within rtol by the
                        # final-vector rescale below; per-step we keep the bound
                        # hard so it can never be violated mid-run.
                        log_w.clamp_(max=torch.log(upper))

    with torch.no_grad():
        weights = torch.exp(log_w)
        if gates is not None:
            gates.eval()
            weights = weights * gates()
        final = weights.detach().numpy().astype(np.float64)

    # Make the hard ratio bound exact on the returned vector. The per-step
    # clamp is in float32, so exp() can overshoot the bound by a float epsilon
    # (~1e-7 relative); a closing float64 cap guarantees no returned weight
    # exceeds max_weight_ratio * w0, which downstream code may assert.
    if max_weight_ratio is not None:
        final = np.minimum(final, max_weight_ratio * np.asarray(w0, dtype=np.float64))

    # Exact mass conservation on the returned vector: a single closing rescale
    # to the input total. When a max_weight_ratio is also set, the rescale is
    # capped at the bound and the residual is redistributed below the bound, so
    # both invariants hold together. When L0 pruning is active, the deficit is
    # redistributed only over surviving (gate-open) records so the cap fill never
    # resurrects a pruned one.
    if conserve_mass:
        pruned = (
            final <= prune_atol if (gates is not None and l0_lambda > 0.0) else None
        )
        final = _project_to_total(final, total0, max_weight_ratio, w0, pruned=pruned)
    return final, trajectory


def _search_l0_lambda_for_budget(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    initial_weights: np.ndarray,
    *,
    target_records: int,
    epochs: int,
    learning_rate: float,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    init_mean: float,
    temperature: float,
    seed: int,
    prune_atol: float,
    initial_lambda: float | None,
    budget_iters: int = _DEFAULT_BUDGET_ITERS,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Search ``l0_lambda`` so the achieved non-zero count tracks the budget.

    The realized non-zero count is monotone *decreasing* in ``l0_lambda`` (a
    stronger penalty closes more gates), so a bisection on ``log10(l0_lambda)``
    drives the count toward ``target_records``. Each evaluation is a full
    :func:`_optimize` run reseeded to ``seed`` (so the count-vs-lambda response is
    a deterministic function the bisection can trust). This is the budget control
    Finding 3 requires: the *number* of records now enters the optimization, not
    just the penalty.

    The search keeps the bracket ``[_L0_SEARCH_LO, _L0_SEARCH_HI]`` and tracks the
    best run seen (the one whose non-zero count is closest to the budget),
    returning it even if the bracket never pins the budget exactly — the count is
    a noisy discrete function, so "closest within ``budget_iters`` steps" is the
    honest contract. Stops early once the achieved count is within ``tol`` of the
    budget, where ``tol = max(1, round(0.05 * target_records))``.

    Args:
        matrix: The constraint matrix ``A`` (dense or sparse-CSR torch tensor), as
            :func:`_optimize` consumes it.
        targets: The target vector tensor.
        initial_weights: The starting weights.
        target_records: The non-zero budget to hit.
        epochs, learning_rate, conserve_mass, max_weight_ratio, init_mean,
            temperature: Passed through to :func:`_optimize`.
        seed: Reseeded before every evaluation for a deterministic response.
        prune_atol: Threshold counting a weight as non-zero (a survivor).
        initial_lambda: A user-supplied ``l0_lambda`` to evaluate first as a warm
            start (clamped into the bracket); ``None`` starts at the bracket
            mid-point.
        budget_iters: Maximum number of optimizations the search may spend.

    Returns:
        ``(weights, trajectory, l0_lambda, n_nonzero)`` of the best run found.
    """
    lo_u, hi_u = math.log10(_L0_SEARCH_LO), math.log10(_L0_SEARCH_HI)
    tol = max(1, round(0.05 * target_records))

    def evaluate(lam: float) -> tuple[np.ndarray, np.ndarray, int]:
        torch.manual_seed(seed)
        weights, trajectory = _optimize(
            matrix,
            targets,
            target_loss_weights,
            initial_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            conserve_mass=conserve_mass,
            max_weight_ratio=max_weight_ratio,
            l0_lambda=lam,
            target_records=target_records,
            init_mean=init_mean,
            temperature=temperature,
        )
        n_nonzero = int((weights > prune_atol).sum())
        return weights, trajectory, n_nonzero

    best: tuple[np.ndarray, np.ndarray, float, int] | None = None
    # Sentinel: a probe whose penalty over-pruned past the cap-feasible floor
    # (the conserve+cap projection raised). It is *more* pruning than feasible,
    # so it steers the search the same way "too few survivors" does — toward a
    # smaller penalty — without crashing the whole search or polluting ``best``.
    _over_pruned = -1

    def consider(lam: float) -> int:
        nonlocal best
        try:
            weights, trajectory, n_nonzero = evaluate(lam)
        except ValueError as exc:
            if "Infeasible combination" in str(exc):
                return _over_pruned
            raise
        if best is None or abs(n_nonzero - target_records) < abs(
            best[3] - target_records
        ):
            best = (weights, trajectory, lam, n_nonzero)
        return n_nonzero

    # Warm start: evaluate the user's lambda (or the bracket mid-point) first.
    if initial_lambda is not None and initial_lambda > 0:
        first_u = min(max(math.log10(initial_lambda), lo_u), hi_u)
    else:
        first_u = (lo_u + hi_u) / 2.0
    iters_left = budget_iters
    n_nonzero = consider(10.0**first_u)
    iters_left -= 1
    # Seed the bracket so the side the warm start landed on is tightened. An
    # over-pruned (infeasible) probe groups with "too few survivors".
    if n_nonzero > target_records:
        lo_u = first_u  # too many survivors -> need a larger penalty
    else:
        hi_u = first_u  # too few survivors / over-pruned -> need a smaller penalty

    # Keep searching while no feasible run is known yet, or the best is outside
    # tolerance, until the iteration budget is spent.
    while iters_left > 0 and (best is None or abs(best[3] - target_records) > tol):
        mid_u = (lo_u + hi_u) / 2.0
        n_nonzero = consider(10.0**mid_u)
        iters_left -= 1
        if n_nonzero == _over_pruned or n_nonzero < target_records:
            hi_u = mid_u  # over-pruned / too few survivors -> smaller penalty
        elif n_nonzero > target_records:
            lo_u = mid_u  # too many survivors -> larger penalty
        else:
            break

    if best is None:
        # Every penalty tried over-pruned past the cap-feasible floor: the budget
        # cannot be met under this conservation + cap. Name the three causes.
        raise ValueError(
            f"Cannot meet target_records={target_records} with mass='conserve' "
            f"and max_weight_ratio={max_weight_ratio}: every L0 penalty searched "
            "over-pruned past the mass the surviving records can carry under the "
            "cap. Loosen max_weight_ratio, relax mass conservation, or raise the "
            "record budget."
        )
    return best


def _project_to_total(
    weights: np.ndarray,
    total: float,
    max_weight_ratio: float | None,
    initial_weights: np.ndarray,
    *,
    pruned: np.ndarray | None = None,
) -> np.ndarray:
    """Scale ``weights`` to sum to ``total`` while respecting an optional cap.

    Without a cap this is a single multiplicative rescale (which preserves
    zeros, so pruned records stay pruned). With a cap, records are scaled up only
    to their bound and any shortfall is spread over the records still below their
    bound, iterating until the total is met or no headroom remains.

    When ``pruned`` is given (L0 pruning is active), the gate-closed records it
    marks are held at their pruned value and are *never* refilled: the cap-fill
    deficit is redistributed only over surviving records. This is the guard
    against the cap fill resurrecting pruned records (Finding 4) — additive
    redistribution over *all* records with headroom would re-open every gate.

    Args:
        weights: The realized weights to rescale (capped already or not).
        total: The mass to restore (the input total).
        max_weight_ratio: The per-record cap multiplier, or ``None``.
        initial_weights: The initial weights the cap multiplies.
        pruned: Optional boolean mask of gate-closed records to hold at zero and
            exclude from deficit redistribution. ``None`` redistributes over
            every record with headroom (the no-pruning case).

    Raises:
        ValueError: If pruning is active and the surviving (non-pruned) records
            lack the headroom to absorb the freed mass under the cap — pruning +
            conserve + cap are then jointly infeasible.
    """
    weights = weights.astype(np.float64).copy()
    if max_weight_ratio is None:
        current = weights.sum()
        if current > 0:
            weights *= total / current
        return weights

    cap = max_weight_ratio * np.asarray(initial_weights, dtype=np.float64)
    weights = np.minimum(weights, cap)
    # Survivors are the records eligible to absorb the deficit: below their cap
    # and, when pruning is active, not gate-closed.
    eligible = np.ones(len(weights), dtype=bool) if pruned is None else ~pruned
    for _ in range(64):
        current = weights.sum()
        if current <= 0 or np.isclose(current, total, rtol=1e-12):
            break
        if current > total:
            weights *= total / current
            continue
        headroom = cap - weights
        free = (headroom > 0) & eligible
        if not free.any():
            if pruned is not None and pruned.any():
                # The survivors are pinned at their caps yet the mass is still
                # short: the only way to close it would be to refill pruned
                # records, which we refuse. Surface the joint infeasibility.
                raise ValueError(
                    "Infeasible combination: L0 pruning + mass='conserve' + "
                    f"max_weight_ratio={max_weight_ratio!r}. After pruning "
                    f"{int(pruned.sum())} record(s), the {int(eligible.sum())} "
                    "surviving record(s) cannot absorb the freed mass under the "
                    "cap (sum of survivor caps < input total). Raise "
                    "max_weight_ratio, loosen the record budget (smaller "
                    "l0_lambda), or use mass='free'."
                )
            break  # no pruning: cap binds everywhere; total is the maximum.
        deficit = total - current
        share = headroom[free] / headroom[free].sum()
        weights[free] = np.minimum(weights[free] + deficit * share, cap[free])
    return weights


def calibrate(
    frame: Frame,
    targets: TargetSet,
    *,
    weight_entity: str = "household",
    method: str = "apg",
    epochs: int = 256,
    learning_rate: float = 0.1,
    mass: str = FREE_MASS,
    max_weight_ratio: float | None = None,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    init_mean: float = 0.999,
    temperature: float = 0.25,
    budget_iters: int = _DEFAULT_BUDGET_ITERS,
    seed: int = 0,
    target_loss_weights: np.ndarray | None = None,
) -> CalibrationResult:
    """Calibrate ``weight_entity``'s weights to ``targets`` over ``frame``.

    Compiles the targets into a sparse system and optimizes the log-weights with
    Adam to minimize the bounded relative-error loss
    ``mean(((A @ w - b)/(b + 1))**2)``. Returns a new frame whose
    ``weight_entity`` weights are :class:`~populace.frame.WeightKind.CALIBRATED`.

    Args:
        frame: The frame to calibrate.
        targets: The :class:`~populace.calibrate.target.TargetSet` of facts.
        weight_entity: Entity whose weights to calibrate (default
            ``"household"``).
        method: Optimization method label. ``"apg"`` (accelerated proximal
            gradient, the charter's named core) and ``"adam"`` both run the
            torch Adam optimizer described above — Adam *is* the accelerated
            first-order method here; the label is carried for the manifest and
            future solver swaps. Any other value is rejected.
        epochs: Number of optimization steps.
        learning_rate: Adam learning rate on the log-weights. The reference
            ``reweight`` is sensitive to this — it does well above ~0.1 and
            breaks down near ~0.005 — so the default matches that regime.
        mass: :data:`FREE_MASS` (default) to let the total move, or
            :data:`CONSERVE_MASS` to hold it to the input total.
        max_weight_ratio: If given, a hard per-record cap: no calibrated weight
            exceeds ``max_weight_ratio * initial_weight``. The landmine guard.
        target_records: If given, enable L0 pruning with **budget control**: the
            solver searches ``l0_lambda`` (a bisection on its log, ``budget_iters``
            optimizations) so the achieved non-zero count tracks this budget, and
            reports the penalty it settled on as
            :attr:`CalibrationResult.l0_lambda`. A supplied ``l0_lambda`` is the
            search's warm start. The achieved count tracks the budget within a
            tolerance (the count is a noisy discrete function of the penalty), not
            exactly.
        l0_lambda: L0 penalty strength. Used directly when ``target_records`` is
            ``None``: ``> 0`` enables hard-concrete gates that prune the pool,
            ``0.0`` (default) keeps every record. When ``target_records`` is set,
            this is only the budget search's warm start (the search overrides it).
        init_mean: Initial expected open-probability of the L0 gates (only used
            when pruning).
        temperature: Hard-concrete temperature (only used when pruning).
        budget_iters: Maximum optimizations the ``target_records`` budget search
            may spend bisecting ``l0_lambda`` (only used when ``target_records``
            is set). Higher resolves the budget finer at a proportional cost.
        seed: Seed for torch's RNG (the gate sampling), for reproducibility.
        target_loss_weights: Optional non-negative row weights aligned to the
            supplied :class:`TargetSet`. When omitted, every compiled target row
            contributes equally (historical behavior). When supplied, the weights
            for skipped targets are dropped with those targets, and the squared
            bounded relative errors for compiled rows are averaged with the
            remaining weights, normalized by their sum.

    Returns:
        A :class:`CalibrationResult` with the calibrated frame, per-target
        diagnostics, loss trajectory, and any skipped targets.

    Raises:
        ValueError: If ``method`` is unknown, ``mass`` is not ``"free"`` or
            ``"conserve"``, ``epochs`` is not positive, ``max_weight_ratio`` is
            given and is not ``> 0``, or ``target_records`` is given and is not
            a positive integer; or if no targets compile (from the matrix
            build).
    """
    if method not in ("apg", "adam"):
        raise ValueError(
            f"Unknown method {method!r}; supported: 'apg', 'adam' (both run the "
            "torch Adam first-order optimizer on log-weights)."
        )
    if mass not in (FREE_MASS, CONSERVE_MASS):
        raise ValueError(
            f"mass must be {FREE_MASS!r} or {CONSERVE_MASS!r}, got {mass!r}."
        )
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs!r}.")
    if max_weight_ratio is not None and not (max_weight_ratio > 0):
        raise ValueError(
            f"max_weight_ratio must be positive, got {max_weight_ratio!r}."
        )
    if max_weight_ratio is not None and max_weight_ratio < 1 and mass == CONSERVE_MASS:
        # Every capped weight is below its initial, so sum(cap) < input total:
        # mass conservation is infeasible a priori (Finding 7). Reject it here
        # with a named error rather than letting it surface later as the kernel's
        # opaque mass-conservation failure.
        raise ValueError(
            f"max_weight_ratio={max_weight_ratio!r} < 1 with mass={mass!r} is "
            "infeasible: every weight is capped below its initial value, so the "
            "total cannot be conserved (sum of caps < input total). Use "
            "max_weight_ratio >= 1, or mass='free'."
        )
    if target_records is not None and (
        not isinstance(target_records, int) or target_records <= 0
    ):
        raise ValueError(
            f"target_records must be a positive integer, got {target_records!r}."
        )
    if budget_iters <= 0:
        raise ValueError(f"budget_iters must be positive, got {budget_iters!r}.")

    target_loss_weights_input = _validate_target_loss_weights(
        target_loss_weights,
        (len(targets),),
    )
    problem = build_constraint_matrix(frame, targets, weight_entity)
    initial = problem.initial_weights
    w0 = initial.values
    prune_atol = _PRUNE_REL_ATOL * float(np.mean(w0))

    torch.manual_seed(seed)
    matrix_t = _torch_constraint_matrix(problem.matrix)
    targets_t = torch.tensor(problem.target_vector, dtype=torch.float32)
    target_loss_weights_np: np.ndarray | None = None
    if target_loss_weights_input is not None:
        weights_by_key = {
            target.key: weight
            for target, weight in zip(targets, target_loss_weights_input, strict=True)
        }
        target_loss_weights_np = np.asarray(
            [weights_by_key[target.key] for target in problem.targets],
            dtype=np.float64,
        )
        target_loss_weights_np = _validate_target_loss_weights(
            target_loss_weights_np,
            problem.target_vector.shape,
        )
    target_loss_weights_t = (
        torch.tensor(target_loss_weights_np, dtype=torch.float32)
        if target_loss_weights_np is not None
        else None
    )

    if target_records is not None:
        # Budget control (Finding 3): search l0_lambda so the achieved non-zero
        # count tracks target_records. The supplied l0_lambda (if any) is the
        # warm start; the search reports the penalty it settled on.
        final_weights, trajectory, effective_l0, n_nonzero = (
            _search_l0_lambda_for_budget(
                matrix_t,
                targets_t,
                target_loss_weights_t,
                w0,
                target_records=target_records,
                epochs=epochs,
                learning_rate=learning_rate,
                conserve_mass=(mass == CONSERVE_MASS),
                max_weight_ratio=max_weight_ratio,
                init_mean=init_mean,
                temperature=temperature,
                seed=seed,
                prune_atol=prune_atol,
                initial_lambda=(l0_lambda if l0_lambda > 0.0 else None),
                budget_iters=budget_iters,
            )
        )
    else:
        effective_l0 = l0_lambda
        final_weights, trajectory = _optimize(
            matrix_t,
            targets_t,
            target_loss_weights_t,
            w0,
            epochs=epochs,
            learning_rate=learning_rate,
            conserve_mass=(mass == CONSERVE_MASS),
            max_weight_ratio=max_weight_ratio,
            l0_lambda=effective_l0,
            target_records=target_records,
            init_mean=init_mean,
            temperature=temperature,
        )
        n_nonzero = int((final_weights > prune_atol).sum())
        if effective_l0 > 0.0 and n_nonzero == 0:
            # Every gate closed: the penalty overwhelmed the fit loss (under
            # Adam the tug-of-war is about gradient sign, so a penalty far
            # above the loss marches every gate logit shut). The kernel would
            # reject the all-zero vector with an opaque "Weights cannot be all
            # zero" — name the cause and the remedies here instead.
            raise ValueError(
                f"L0 pruning closed every gate: l0_lambda={effective_l0!r} "
                "overwhelmed the fit loss and every calibrated weight is "
                "zero. Lower l0_lambda, or use target_records= budget "
                "control, which adapts the penalty to a survivor count."
            )

    calibrated = initial.with_values(final_weights, kind=WeightKind.CALIBRATED)
    new_frame = _apply_weights(frame, weight_entity, initial, calibrated, mass, targets)

    diagnostics = _build_diagnostics(problem, frame, w0, final_weights)
    # One closing eval-mode loss on the RETURNED weights (Finding 8): the same
    # bounded relative-error loss the optimizer minimizes,
    # mean(((A@w - b)/(b+1))**2),
    # evaluated after the closing mass/cap projections — so final_loss describes
    # what calibrate returns, not the trajectory's pre-projection tail.
    closing_loss = relative_error_loss(
        problem.estimates(final_weights),
        problem.target_vector,
        target_loss_weights=target_loss_weights_np,
    )

    return CalibrationResult(
        frame=new_frame,
        weight_entity=weight_entity,
        weights=final_weights,
        initial_weights=w0.copy(),
        diagnostics=diagnostics,
        loss_trajectory=trajectory,
        skipped=problem.skipped,
        problem=problem,
        l0_lambda=effective_l0,
        n_nonzero=n_nonzero,
        closing_loss=closing_loss,
        options={
            "method": method,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "mass": mass,
            "max_weight_ratio": max_weight_ratio,
            "target_records": target_records,
            "seed": seed,
            "target_loss_weights": _target_loss_weight_options(target_loss_weights_np),
            "matrix_format": (
                "sparse_csr" if matrix_t.layout == torch.sparse_csr else "dense"
            ),
        },
    )


def _apply_weights(
    frame: Frame,
    weight_entity: str,
    initial: Weights,
    calibrated: Weights,
    mass: str,
    targets: TargetSet,
) -> Frame:
    """Place the calibrated weights on the frame with the right mass policy.

    ``mass="conserve"`` uses the kernel's ``CONSERVE_MASS`` (the total is held
    to the input within rtol, so the kernel's conservation check passes).
    ``mass="free"`` declares a :class:`~populace.frame.MassChange`: the total
    moved on purpose to fit the targets, and the change is recorded on the
    frame's mass log with a reason naming the calibration.
    """
    if mass == CONSERVE_MASS:
        from populace.frame import CONSERVE_MASS as FRAME_CONSERVE

        return frame.with_weights(weight_entity, calibrated, mass=FRAME_CONSERVE)
    factor = calibrated.total / initial.total if initial.total != 0 else None
    reason = (
        f"calibrated {weight_entity!r} weights to {len(targets)} target(s) "
        "(bounded relative-error loss); total mass free to move"
    )
    return frame.with_weights(
        weight_entity,
        calibrated,
        mass=MassChange(factor=factor, reason=reason),
    )
