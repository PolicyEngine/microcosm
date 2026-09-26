"""The calibration solver: targets -> calibrated weights on a Frame.

:func:`calibrate` is the representation operator. It compiles a
:class:`~microcosm.calibrate.target.TargetSet` against a
:class:`~microcosm.frame.Frame` (:mod:`microcosm.calibrate.matrix`), then optimizes
the weight vector of ``weight_entity`` to minimize fixed-scale weighted MAPE

    ``weighted_mean(abs((A @ w - b) / s))``

where ``s`` is a per-target scale fixed before optimization. By default,
``s = max(abs(b), 1)``: the loss is measured against the administrative target
value, while zero-valued targets use one unit in their measure basis.

The default ``method="adam"`` optimizes with torch's Adam over the
**log-weights** (so weights stay strictly positive by construction).
``method="prox"`` uses proximal gradient on raw non-negative weight ratios for
the nonsmooth L1 selection path. It returns a
:class:`CalibrationResult` carrying a new frame whose ``weight_entity`` weights
are :class:`~microcosm.frame.WeightKind.CALIBRATED`, per-target diagnostics, and
the loss trajectory.

Six declared options, each a real feature (and each its own test):

- ``mass="free"`` (default) lets the total weight move to fit the targets;
  ``mass="conserve"`` projects every step's weights back to the input total, so
  the calibrated population conserves the starting mass exactly.
- ``max_weight_ratio`` is a **hard** per-record bound: no calibrated weight may
  exceed ``max_weight_ratio * initial_weight``. It is clamped after every step —
  the documented guard against the tail "landmine" (a rare high-value,
  near-zero-weight donor whose weight detonates on reweight and blows up an
  aggregate).
- ``target_records`` turns on hard-concrete L0 gates
  (:mod:`microcosm.calibrate.gates`) with **budget control**: the solver searches
  ``l0_lambda`` (a bisection on its log) so the achieved non-zero count tracks
  ``target_records``, and reports the penalty it settled on — the
  generate-big-then-prune path. A supplied ``l0_lambda`` is the search's warm
  start.
- ``l0_lambda`` alone (no ``target_records``) prunes at a fixed penalty: ``> 0``
  gates the pool, ``0.0`` keeps every record. It is the sole control when no
  budget is given.
- ``l1_lambda`` adds a proximal L1 penalty on
  ``mean(weight / initial_weight)`` under ``method="prox"``. The soft-threshold
  step can send unneeded records to exact zero, so L1 is a sparse selection path
  with a clear objective coefficient.
- ``l2_lambda`` adds an experimental soft concentration penalty on
  ``mean((pre_gate_weight / initial_weight) ** 2)``. With no L0 gates this is
  the realized weight ratio; with L0 gates it intentionally penalizes the
  latent pre-gate weight so a nearly closed gate cannot hide an exploding
  ``log_w``. It is cleanest as an ESS/design-effect knob under
  ``mass="conserve"``; ``max_weight_ratio`` remains the hard safety bound.
  This penalty is only implemented by ``method="adam"``. In the two-stage
  :func:`calibrate_l0_refit` path it applies to both stages unless
  ``refit_l2_lambda`` overrides the refit stage — the stage whose weights ship.

An explicit exact-k frozen refit requires the selected records' aligned marginal
inclusion probabilities ``q_i`` and benchmarks them with Horvitz--Thompson
weights ``w_i / q_i``. For selection indicator ``I_i``,
``E[I_i * w_i / q_i] = w_i``, so their sum is design-unbiased for full-pool
mass. Because that mass is known, the realized weights are subsequently
projected to the exact control total while preserving the relative allocation
defined by ``w_i / q_i``. This normalized benchmark is the optimizer's initial
weight, so ``max_weight_ratio`` scales its inclusion-aware allocation. Constant
``q`` reduces algebraically to the previous known-total ratio rescale. The
legacy thresholded L0 refit does not perform this normalization and remains
unchanged.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from microcosm.calibrate.exact_k import (
    BOUNDARY_MASS_SHORT,
    BOUNDARY_SHORT_OF_DRAW,
    CERTAINTIES_EXCEED_K,
    FEASIBLE,
    assert_exact_k_support,
    exact_k_design_feasibility,
)
from microcosm.calibrate.gates import HardConcrete
from microcosm.calibrate.group_bounds import GroupedUpperBounds
from microcosm.calibrate.initialization import GateInitialization
from microcosm.calibrate.matrix import (
    CalibrationProblem,
    SkippedTarget,
    build_constraint_matrix,
)
from microcosm.calibrate.target import TargetSet
from microcosm.calibrate.target_snapshots import (
    ITERATE_BEST_RETAINED,
    ITERATE_CURRENT,
    ITERATE_SELECTED,
    BoundTargetSnapshots,
    TargetSnapshotObserver,
)
from microcosm.frame import Frame, MassChange, WeightKind, Weights

__all__ = [
    "calibrate",
    "calibrate_l0_refit",
    "rebuild_calibration_result",
    "refit_l0_selection",
    "default_target_loss_scales",
    "effective_sample_size",
    "relative_error_loss",
    "CalibrationResult",
    "L0RefitResult",
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
#: What the ``target_records`` budget search measures against the budget.
BUDGET_BASIS_NONZERO_COUNT = "nonzero_count"
BUDGET_BASIS_OPEN_PROBABILITY_MASS = "open_probability_mass"
BUDGET_BASES = frozenset(
    {BUDGET_BASIS_NONZERO_COUNT, BUDGET_BASIS_OPEN_PROBABILITY_MASS}
)

# Per-target contribution cap for weighted MAPE. A target can contribute at most
# a 1000% scaled miss to the objective.
_DEFAULT_TARGET_LOSS_CAP = 10.0


@dataclass(frozen=True)
class TargetDiagnostic:
    """Per-target calibration diagnostics.

    Attributes:
        name: The target's ``"name@period"`` row label.
        target: The sum target value aimed at; this is the compiled right-hand
            side ``b``.
        initial_estimate: The achieved aggregate under the input weights —
            ``row @ w0``.
        final_estimate: The achieved aggregate under the calibrated weights —
            ``row @ w``.
        relative_error: ``(final_estimate - target) / target`` (or
            ``final_estimate - target`` when ``target`` is zero, since the
            relative form is undefined there).
        within_tolerance: Whether ``|final_estimate - target|`` is within the
            target's declared tolerance. ``None`` when the target declared no
            tolerance.
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
        frame: A new :class:`~microcosm.frame.Frame` whose ``weight_entity``
            weights are :class:`~microcosm.frame.WeightKind.CALIBRATED`.
        weight_entity: The entity whose weights were calibrated.
        weights: The calibrated weight values (also on :attr:`frame`).
        initial_weights: The input weight values. These remain the frame's
            original weights even when the optimizer is initialized from
            ``warm_start_weights``.
        diagnostics: Per-target :class:`TargetDiagnostic`, aligned to the
            compiled problem rows.
        loss_trajectory: The optimizer-start loss at each epoch (length
            ``epochs``). With no warm start, the first value is the input-weight
            loss; with a warm start, the first value is the warm-start loss.
            Best-iterate selection does not truncate this history.
        skipped: Targets that could not be compiled (carried through from the
            matrix build), each with its reason.
        problem: The compiled :class:`CalibrationProblem` (matrix, b, names).
        l0_lambda: The L0 penalty actually applied (0.0 when no pruning). When a
            ``target_records`` budget was set, this is the penalty the budget
            search settled on, not the value passed in.
        n_nonzero: Number of calibrated weights above the prune threshold. With a
            ``target_records`` budget, the quantity the search drives toward it.
        closing_loss: The capped weighted-MAPE calibration loss evaluated once on the
            *returned* weights (after the closing mass/cap projections). Exposed
            as :attr:`final_loss`; recorded separately from the trajectory, whose
            tail is a pre-step/pre-projection value and may follow an earlier
            selected best iterate.
        target_loss_weights: The effective non-negative target-importance vector,
            aligned to :attr:`diagnostics`. Uniform defaults are materialized as
            ones so diagnostics can publish the exact final loss basis.
        target_loss_scales: The effective positive target-loss denominator vector,
            aligned to :attr:`diagnostics` after skipped targets are removed.
        target_loss_cap: The positive per-target scaled-error cap used by the final
            loss evaluation.
        options: The solver configuration as passed (method, epochs,
            learning_rate, mass, max_weight_ratio, target_records, seed,
            l1_lambda, l2_lambda) plus the realized ``matrix_format``
            (``"dense"`` or ``"sparse_csr"``). This is what a build records in its release
            manifest — the max_weight_ratio bound is part of the dataset's
            provenance, not a local solver detail.
        gate_open_probabilities: Per-record hard-concrete open probabilities
            ``pi_i``, aligned to :attr:`weights`, for an L0 solve. ``None`` for
            ordinary Adam, proximal, and score-only results, which have no
            learned gate state.

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
    target_loss_weights: np.ndarray
    target_loss_scales: np.ndarray
    target_loss_cap: float
    options: Mapping[str, object] = field(default_factory=dict)
    gate_open_probabilities: np.ndarray | None = None

    @property
    def initial_loss(self) -> float:
        """The capped weighted-MAPE loss at the optimizer's starting weights.

        Without ``warm_start_weights`` this is the input-weight loss. With a warm
        start this is the supplied starting vector's loss; per-target diagnostic
        ``initial_estimate`` values still describe the original input weights.
        """
        return float(self.loss_trajectory[0])

    @property
    def final_loss(self) -> float:
        """The capped weighted-MAPE calibration loss of the *returned* weights.

        This is a single eval-mode evaluation on the weights actually returned —
        after the closing mass/cap projections — so it describes the calibrated
        vector. It is **not** ``loss_trajectory[-1]``: the trajectory records each
        epoch's loss before that epoch's step and before the closing projections,
        so its tail can differ (e.g. under ``mass="conserve"`` with a cap or when
        an earlier feasible iterate has a lower loss).
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

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size of the calibrated weights.

        ``(sum w)^2 / sum(w^2)`` — the number of equal-weight records carrying
        the same information as the calibrated vector. Uniform weights maximize
        it at the record count; concentrating mass on few records lowers it.
        Pruned (zero) weights contribute nothing, so a sparse L0 result's ESS
        describes its surviving support.
        """
        return effective_sample_size(self.weights)

    @property
    def realized_max_weight_ratio(self) -> float:
        """The largest calibrated-to-initial weight ratio actually realized.

        The realized counterpart of the ``max_weight_ratio`` *bound* in
        :attr:`options`: how far calibration actually inflated its most
        inflated record, whether or not a bound was set.
        """
        weights = np.asarray(self.weights, dtype=np.float64)
        initial = np.asarray(self.initial_weights, dtype=np.float64)
        if not self.options.get("grouped_preserve_zeros", {}).get("enabled", False):
            return float((weights / initial).max())
        if np.any((initial == 0) & (weights > 0)):
            return float("inf")
        ratios = np.zeros_like(weights)
        np.divide(
            weights, initial, out=ratios, where=~((initial == 0) & (weights == 0))
        )
        return float(ratios.max())

    @property
    def top_1pct_weight_share(self) -> float:
        """Share of total calibrated weight carried by the heaviest 1% of records.

        A tail-concentration summary to read alongside
        :attr:`effective_sample_size`: with uniform weights the heaviest 1% of
        records carry 1% of the weight; values far above that mean few records
        represent much of the population.
        """
        weights = np.asarray(self.weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return 0.0
        k = max(1, math.ceil(0.01 * weights.size))
        return float(np.sort(weights)[-k:].sum() / total)


@dataclass(frozen=True)
class L0RefitResult:
    """L0 probability/support selection followed by an ordinary frozen refit.

    The :attr:`selection` stage is a normal :func:`calibrate` call with
    hard-concrete L0 gates. The :attr:`refit` stage keeps either the legacy
    weight-threshold support or a caller-supplied exact-k support, removes the
    gates and L0 penalty, and runs ordinary calibration on the resulting frame.
    Convenience properties delegate to :attr:`refit`, because the refit frame
    and weights are the production artifact.
    """

    selection: CalibrationResult
    refit: CalibrationResult
    selected_entity_ids: np.ndarray
    selected_mask: np.ndarray

    @property
    def frame(self) -> Frame:
        """The post-L0-refit sparse frame."""
        return self.refit.frame

    @property
    def weight_entity(self) -> str:
        """The entity whose weights were calibrated."""
        return self.refit.weight_entity

    @property
    def weights(self) -> np.ndarray:
        """The post-L0-refit weights on the sparse frame."""
        return self.refit.weights

    @property
    def initial_weights(self) -> np.ndarray:
        """The refit's starting weights on its frozen support.

        The legacy path inherits surviving L0 weights. The explicit exact-k
        path starts from original-frame ``w_i / q_i`` design weights and
        projects them to the known full-pool total, so a positive-probability
        record whose deterministic gate closed remains refittable without
        losing population mass.
        """
        return self.refit.initial_weights

    @property
    def diagnostics(self) -> tuple[TargetDiagnostic, ...]:
        """Per-target diagnostics from the post-L0 refit."""
        return self.refit.diagnostics

    @property
    def loss_trajectory(self) -> np.ndarray:
        """The post-L0 refit's loss trajectory."""
        return self.refit.loss_trajectory

    @property
    def skipped(self) -> tuple[SkippedTarget, ...]:
        """Targets skipped by the post-L0 refit."""
        return self.refit.skipped

    @property
    def problem(self) -> CalibrationProblem:
        """The post-L0 refit's compiled calibration problem."""
        return self.refit.problem

    @property
    def l0_lambda(self) -> float:
        """The L0 penalty used by the selection stage."""
        return self.selection.l0_lambda

    @property
    def n_nonzero(self) -> int:
        """Number of positive weights in the post-L0 refit."""
        return self.refit.n_nonzero

    @property
    def initial_loss(self) -> float:
        """The refit's initial loss on the selected support."""
        return self.refit.initial_loss

    @property
    def final_loss(self) -> float:
        """The post-L0 refit's final penalty-free target loss."""
        return self.refit.final_loss

    @property
    def target_loss_weights(self) -> np.ndarray:
        """The final refit's aligned target-importance vector."""
        return self.refit.target_loss_weights

    @property
    def target_loss_scales(self) -> np.ndarray:
        """The final refit's aligned target-loss scale vector."""
        return self.refit.target_loss_scales

    @property
    def target_loss_cap(self) -> float:
        """The final refit's per-target scaled-error cap."""
        return self.refit.target_loss_cap

    @property
    def fraction_within_10pct(self) -> float:
        """Share of targets the post-L0 refit reproduces within 10%."""
        return self.refit.fraction_within_10pct

    @property
    def effective_sample_size(self) -> float:
        """Kish ESS of the post-L0 refit weights (the shipped vector)."""
        return self.refit.effective_sample_size

    @property
    def realized_max_weight_ratio(self) -> float:
        """The post-L0 refit's largest realized calibrated-to-initial ratio."""
        return self.refit.realized_max_weight_ratio

    @property
    def top_1pct_weight_share(self) -> float:
        """Weight share of the post-L0 refit's heaviest 1% of records."""
        return self.refit.top_1pct_weight_share

    @property
    def options(self) -> Mapping[str, object]:
        """Combined production options and selection provenance."""
        return {
            **dict(self.refit.options),
            "post_l0_refit": True,
            "selection_l0_lambda": self.selection.l0_lambda,
            "selection_n_nonzero": self.selection.n_nonzero,
            "selection_final_loss": self.selection.final_loss,
            "selection_options": dict(self.selection.options),
        }


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
    target_loss_scales: np.ndarray | None = None,
    target_loss_cap: float = _DEFAULT_TARGET_LOSS_CAP,
) -> float:
    """THE loss, in numpy: capped weighted MAPE on fixed row scales.

    The single canonical definition every measurement imports — the solver's
    closing loss, the acceptance gates, and scorers all call this function
    (the torch twin below is the autograd path of the same formula). Refuses
    non-finite inputs: a NaN estimate is a harness bug, not a large miss.

    ``target_loss_scales`` is the row denominator ``s`` in
    ``abs((estimate - target) / s)``. If omitted, rows use
    ``max(abs(target), 1)`` so the target surface, not the starting estimate,
    defines the permanent scale. Target weights are normalized by their own
    sum, so multiplying all weights by a constant does not change the
    objective. Each row's scaled absolute miss is capped by
    ``target_loss_cap``.
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
    scales = _validate_target_loss_scales(
        target_loss_scales,
        targets.shape,
        targets=targets,
    )
    cap = _validate_target_loss_cap(target_loss_cap)
    loss = np.minimum(np.abs((estimates - targets) / scales), cap)
    weights = _validate_target_loss_weights(target_loss_weights, targets.shape)
    if weights is None:
        return float(loss.mean())
    return float(np.average(loss, weights=weights))


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: ``(sum w)^2 / sum(w^2)``.

    The number of equal-weight records carrying the same information as the
    weighted vector — the canonical weight-concentration diagnostic (the
    design-effect denominator). Uniform weights maximize it at the record
    count; piling mass onto few records drives it down. Usable on any
    non-negative weight vector, e.g. to score a published artifact's weights
    without re-running calibration.

    Args:
        weights: Non-negative, finite weight values.

    Returns:
        The ESS; ``0.0`` for an all-zero vector.

    Raises:
        ValueError: If any weight is negative or non-finite.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("effective_sample_size requires finite, non-negative weights.")
    denominator = float(np.square(weights).sum())
    if denominator == 0.0:
        return 0.0
    return float(weights.sum() ** 2 / denominator)


def _relative_error_loss(
    estimate: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    target_loss_scales: torch.Tensor,
    target_loss_cap: float,
) -> torch.Tensor:
    """The capped weighted-MAPE target loss, optionally averaged with row weights."""
    scaled_error = (estimate - targets) / target_loss_scales
    loss = torch.clamp(
        torch.abs(scaled_error),
        max=_validate_target_loss_cap(target_loss_cap),
    )
    if target_loss_weights is None:
        return loss.mean()
    return (loss * target_loss_weights).sum() / target_loss_weights.sum()


def _validate_target_loss_cap(target_loss_cap: float) -> float:
    cap = float(target_loss_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError(
            f"target_loss_cap must be positive and finite, got {target_loss_cap!r}."
        )
    return cap


def default_target_loss_scales(
    targets: np.ndarray,
    initial_estimates: np.ndarray | None = None,
) -> np.ndarray:
    """Default fixed row scales for calibration.

    The old objective used ``target + 1`` as the denominator. That makes a
    zero-valued target with a large starting estimate dominate the loss by many
    orders of magnitude. The production scale is instead target-defined:
    the absolute target or one unit in the row's measure basis. Starting
    estimates are deliberately excluded; they describe the baseline, not the
    administrative fact we are trying to hit. ``initial_estimates`` is accepted
    only for compatibility with callers of the earlier helper signature.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if initial_estimates is not None:
        initial_estimates = np.asarray(initial_estimates, dtype=np.float64)
        if targets.shape != initial_estimates.shape:
            raise ValueError(
                "targets and initial_estimates must align, got shapes "
                f"{targets.shape} vs {initial_estimates.shape}."
            )
    if not np.isfinite(targets).all():
        raise ValueError("default_target_loss_scales requires finite targets.")
    return np.maximum(np.abs(targets), np.ones_like(targets, dtype=np.float64))


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


def _validate_target_loss_scales(
    target_loss_scales: np.ndarray | None,
    shape: tuple[int, ...],
    *,
    targets: np.ndarray,
) -> np.ndarray:
    if target_loss_scales is None:
        scales = np.maximum(np.abs(np.asarray(targets, dtype=np.float64)), 1.0)
    else:
        scales = np.asarray(target_loss_scales, dtype=np.float64)
    if scales.shape != shape:
        raise ValueError(
            "target_loss_scales must align with targets, got shapes "
            f"{scales.shape} vs {shape}."
        )
    if not np.isfinite(scales).all():
        raise ValueError("target_loss_scales must be finite.")
    if (scales <= 0).any():
        raise ValueError("target_loss_scales must be positive.")
    return scales


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


def _target_loss_scale_options(
    target_loss_scales: np.ndarray,
    *,
    kind: str,
    target_loss_cap: float,
) -> Mapping[str, object]:
    scales = np.asarray(target_loss_scales, dtype=np.float64)
    return {
        "kind": kind,
        "formula": "weighted_mean(min(abs((estimate - target) / scale), cap))",
        "cap": float(target_loss_cap),
        "n": int(scales.shape[0]),
        "min": float(scales.min()),
        "median": float(np.median(scales)),
        "max": float(scales.max()),
    }


def _build_diagnostics(
    problem: CalibrationProblem,
    frame: Frame,
    initial_weights: np.ndarray,
    final_weights: np.ndarray,
) -> tuple[TargetDiagnostic, ...]:
    """Assemble per-target diagnostics from the problem and both weight vectors.

    Sum rows are exactly ``A @ w``, so their estimates and target come straight
    from the compiled system.
    """
    initial_est = problem.estimates(initial_weights)
    final_est = problem.estimates(final_weights)
    diagnostics: list[TargetDiagnostic] = []
    for i, target in enumerate(problem.targets):
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


def _prepare_warm_start_weights(
    initial_weights: np.ndarray,
    warm_start_weights: np.ndarray | None,
    *,
    conserve_mass: bool,
    max_weight_ratio: float | None,
) -> np.ndarray:
    w0 = np.asarray(initial_weights, dtype=np.float64)
    if warm_start_weights is None:
        return w0.copy()
    start = np.asarray(warm_start_weights, dtype=np.float64)
    if start.shape != w0.shape:
        raise ValueError(
            "warm_start_weights shape must match the calibration weights: "
            f"got {start.shape}, expected {w0.shape}."
        )
    if not np.isfinite(start).all():
        raise ValueError("warm_start_weights must be finite.")
    if (start <= 0.0).any():
        raise ValueError(
            "warm_start_weights must be strictly positive for log-weight optimization."
        )
    prepared = start.copy()
    if max_weight_ratio is not None:
        prepared = np.minimum(prepared, max_weight_ratio * w0)
    if conserve_mass:
        prepared = _project_to_total(prepared, float(w0.sum()), max_weight_ratio, w0)
    return prepared


def _optimize(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    target_loss_scales: torch.Tensor,
    target_loss_cap: float,
    initial_weights: np.ndarray,
    *,
    gate_initialization: GateInitialization | None = None,
    warm_start_weights: np.ndarray | None = None,
    epochs: int,
    learning_rate: float,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    l0_lambda: float,
    l2_lambda: float,
    l2_anchor: str = "initial",
    l2_anchor_weights: np.ndarray | None = None,
    target_records: int | None,
    init_mean: float,
    temperature: float,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_context: Mapping[str, object] | None = None,
    snapshots: BoundTargetSnapshots | None = None,
    return_gate_open_probabilities: bool = False,
    grouped_upper_bounds: GroupedUpperBounds | None = None,
    grouped_preserve_zeros: bool = False,
    _post_projection_observer: Callable[[dict[str, object]], None] | None = None,
    selection_receipt: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run the torch optimization and return weights plus its trajectory.

    Optimizes the log-weights with Adam against capped weighted MAPE.
    Positivity is by construction (``w = exp(log_w)`` times optional gates). The
    hard constraints — mass conservation and ``max_weight_ratio`` — are applied
    by projecting the realized weights after each step, so they hold on the
    returned vector exactly, not merely in expectation.

    For ungrouped, deterministic, unregularized free-mass runs, return the best feasible
    iterate seen within the requested budget, including the warm start and the
    final update. Constant-step Adam can otherwise discard an earlier better
    fit when it oscillates across the absolute-error kink. Gated, regularized,
    and conserved-mass runs keep their closing-state selection: their stochastic
    penalties or closing mass projection need a different selection criterion.

    The private opt-in ``return_gate_open_probabilities`` adds a third return
    value without changing the two-item tuple used by existing workspace
    callers. The third value is aligned per-record ``pi_i`` for an L0 run and
    ``None`` when no gates were active.
    """
    _validate_grouped_zero_option(grouped_upper_bounds, grouped_preserve_zeros)
    if grouped_upper_bounds is not None:
        if gate_initialization is not None:
            raise ValueError("grouped upper bounds do not support gate initialization")
        _validate_grouped_mode(
            grouped_upper_bounds,
            method="adam",
            conserve_mass=conserve_mass,
            max_weight_ratio=max_weight_ratio,
            l0_lambda=l0_lambda,
            l1_lambda=0.0,
            target_records=target_records,
        )
        final, trajectory = _optimize_grouped(
            matrix,
            targets,
            target_loss_weights,
            target_loss_scales,
            target_loss_cap,
            initial_weights,
            grouped_upper_bounds,
            warm_start_weights=warm_start_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            l2_anchor=l2_anchor,
            l2_anchor_weights=l2_anchor_weights,
            progress_callback=progress_callback,
            progress_context=progress_context,
            observer=_post_projection_observer,
            snapshots=snapshots,
            preserve_zeros=grouped_preserve_zeros,
        )
        if return_gate_open_probabilities:
            return final, trajectory, None
        return final, trajectory
    if _post_projection_observer is not None:
        raise ValueError("post-projection observer requires grouped upper bounds")
    w0 = np.asarray(initial_weights, dtype=np.float64)
    start = _prepare_warm_start_weights(
        w0,
        warm_start_weights,
        conserve_mass=conserve_mass,
        max_weight_ratio=max_weight_ratio,
    )
    total0 = float(w0.sum())
    # Same prune threshold the result's n_nonzero uses, so "pruned" here means
    # exactly what the reported non-zero count means.
    prune_atol = _PRUNE_REL_ATOL * float(np.mean(w0))
    log_w = torch.tensor(np.log(start), dtype=torch.float32, requires_grad=True)

    gates: HardConcrete | None = None
    params: list[torch.Tensor] = [log_w]
    if l0_lambda > 0.0 or target_records is not None:
        gates = HardConcrete(
            len(w0),
            init_mean=init_mean,
            temperature=temperature,
            **(
                {}
                if gate_initialization is None
                else {
                    "initial_probabilities": gate_initialization.probabilities,
                    "protected_mask": gate_initialization.protected,
                }
            ),
        )
        params = [log_w, *gates.parameters()]

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    upper = (
        torch.tensor(max_weight_ratio * w0, dtype=torch.float32)
        if max_weight_ratio is not None
        else None
    )
    if l2_lambda > 0.0:
        # The L2 penalty's reference vector. "initial" divides by each record's
        # own starting weight; "uniform" divides by the shared mean weight; an
        # explicit vector (e.g. the pre-selection design weights during a
        # refit) overrides both. Under mass conservation the penalty's
        # constrained optimum is w_i ∝ anchor_i², so anchoring on
        # heterogeneous starting weights (e.g. a refit whose start is a
        # concentrated selection vector) pulls toward MORE concentration; the
        # uniform anchor makes the penalty a direct 1/ESS control regardless
        # of the starting distribution.
        if l2_anchor_weights is not None:
            anchor = np.asarray(l2_anchor_weights, dtype=np.float64)
        elif l2_anchor == "initial":
            anchor = w0
        else:
            anchor = np.full_like(w0, w0.mean())
        w0_t = torch.tensor(anchor, dtype=torch.float32)
    else:
        w0_t = None

    retain_best = gates is None and not conserve_mass and l2_lambda == 0.0
    best_loss = float("inf")
    best_epoch = 0
    best_log_w: torch.Tensor | None = None
    trajectory = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = torch.exp(log_w)
        if gates is not None:
            weights = weights * gates()
        estimate = _apply_constraint(matrix, weights)
        loss = _relative_error_loss(
            estimate,
            targets,
            target_loss_weights,
            target_loss_scales,
            target_loss_cap,
        )
        penalty = (
            l0_lambda * gates.get_penalty()
            if (gates is not None and l0_lambda > 0.0)
            else torch.zeros((), dtype=torch.float32)
        )
        # Penalize latent pre-gate weights. Under L0, a nearly closed gate
        # should not be able to hide a very large exp(log_w).
        l2_penalty = (
            ((torch.exp(log_w) / w0_t) ** 2).mean()
            if w0_t is not None
            else torch.zeros((), dtype=torch.float32)
        )
        total_loss = loss + penalty + l2_lambda * l2_penalty
        trajectory[epoch] = float(loss.item())
        if retain_best and trajectory[epoch] < best_loss:
            best_loss = trajectory[epoch]
            best_epoch = epoch
            best_log_w = log_w.detach().clone()
        if progress_callback is not None:
            progress_callback(
                {
                    **dict(progress_context or {}),
                    "kind": "calibration_epoch",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": trajectory[epoch],
                }
            )
        if snapshots is not None and snapshots.emits(epoch + 1, epochs):
            # The estimate tensor this epoch's loss was computed from, detached
            # rather than recomputed: re-running the forward pass would redraw
            # the hard-concrete gates' noise and move the RNG stream, changing
            # the run. The current values are labelled best_retained only when
            # they ARE the incumbent best iterate.
            retained = retain_best and best_log_w is not None
            snapshots.emit(
                estimate.detach().numpy(),
                # Number of completed updates, matching the selection receipt.
                # Cadence still uses the 1-based evaluation ordinal above.
                epoch=epoch,
                epochs=epochs,
                iterate=(
                    ITERATE_BEST_RETAINED
                    if retained and best_epoch == epoch
                    else ITERATE_CURRENT
                ),
                precision="float32",
                loss=trajectory[epoch],
                best_retained={
                    "available": bool(retain_best),
                    "epoch": best_epoch if retained else None,
                    "loss": best_loss if retained else None,
                },
            )
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

    gate_open_probabilities: np.ndarray | None = None
    with torch.no_grad():
        weights = torch.exp(log_w)
        if best_log_w is not None:
            closing_loss = _relative_error_loss(
                _apply_constraint(matrix, weights),
                targets,
                target_loss_weights,
                target_loss_scales,
                target_loss_cap,
            )
            if float(closing_loss.item()) > best_loss:
                weights = torch.exp(best_log_w)
                selected_epoch = best_epoch
                selected_loss = best_loss
            else:
                selected_epoch = epochs
                selected_loss = float(closing_loss.item())
            if selection_receipt is not None:
                selection_receipt.update(
                    {
                        "rule": "best_feasible_loss",
                        "selected_epoch": selected_epoch,
                        "epochs_executed": epochs,
                        "epoch_convention": "completed_optimizer_updates; zero is start",
                        "selected_loss_float32": selected_loss,
                        "closing_iterate_loss_float32": float(closing_loss.item()),
                    }
                )
        if gates is not None:
            gates.eval()
            weights = weights * gates()
            if return_gate_open_probabilities:
                gate_open_probabilities = (
                    gates.get_active_prob()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
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
    if return_gate_open_probabilities:
        return final, trajectory, gate_open_probabilities
    return final, trajectory


def _validate_grouped_mode(
    groups: GroupedUpperBounds,
    *,
    method: str,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    l0_lambda: float,
    l1_lambda: float,
    target_records: int | None,
) -> None:
    if not isinstance(groups, GroupedUpperBounds):
        raise ValueError("grouped_upper_bounds must be GroupedUpperBounds")
    if max_weight_ratio is not None:
        raise ValueError(
            "grouped upper bounds refuse simultaneous scalar max_weight_ratio"
        )
    if (
        method != "adam"
        or conserve_mass
        or l0_lambda != 0
        or l1_lambda != 0
        or target_records is not None
    ):
        raise ValueError(
            "grouped upper bounds require Adam free mass without prox, L0, or exact-k selection"
        )


def _validate_grouped_zero_option(
    groups: GroupedUpperBounds | None, preserve_zeros: bool
) -> None:
    if type(preserve_zeros) is not bool:
        raise ValueError("grouped_preserve_zeros must be a boolean")
    if preserve_zeros and groups is None:
        raise ValueError("grouped_preserve_zeros requires grouped upper bounds")


def _check_fixed_support(
    weights: np.ndarray, initial: np.ndarray, groups: GroupedUpperBounds
) -> np.ndarray:
    """Validate the complete vector, preserving the initial zero coordinates."""
    totals = groups.check(weights)
    active = initial > 0
    if not active.any():
        raise ValueError("grouped fixed support requires a positive initial weight")
    if weights[~active].tobytes() != initial[~active].tobytes():
        raise ValueError("grouped weights must preserve the frozen zero support")
    if (weights[active] <= 0).any():
        raise ValueError("grouped active support must remain strictly positive")
    return totals


def _check_grouped_household_ids(frame: Frame, expected: tuple[int | str, ...]) -> None:
    actual = tuple(
        frame.table("household")[frame.schema.entity_id_column("household")].tolist()
    )
    if actual != expected:
        raise ValueError("stored group household IDs must preserve the ordered IDs")


def _grouped_snapshot_selection(preserve_zeros: bool) -> dict[str, object]:
    """The bounded aggregate labels identifying a grouped snapshot's solver.

    Grouped Adam is a closing-state algorithm: it never runs the retain-best
    rule, so every grouped snapshot carries ``best_retained.available: False``.
    Without a label saying which solver produced it, that is indistinguishable
    from an ordinary run whose retain-best rule happened to be off, and a
    consumer would read the two the same way. These are three JSON scalars
    describing the run's mode -- no household IDs, no group membership, no
    bounds vector, and nothing of record length.
    """
    return {
        "rule": "closing_state",
        "constraint_mode": "grouped_upper_bounds",
        "grouped_preserve_zeros": bool(preserve_zeros),
    }


def _optimize_grouped(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    target_loss_scales: torch.Tensor,
    target_loss_cap: float,
    initial_weights: np.ndarray,
    groups: GroupedUpperBounds,
    *,
    warm_start_weights: np.ndarray | None,
    epochs: int,
    learning_rate: float,
    l2_lambda: float,
    l2_anchor: str,
    l2_anchor_weights: np.ndarray | None,
    progress_callback: Callable[[dict[str, object]], None] | None,
    progress_context: Mapping[str, object] | None,
    observer: Callable[[dict[str, object]], None] | None,
    snapshots: BoundTargetSnapshots | None = None,
    preserve_zeros: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-mass Adam on positive coordinates, with full float64 accepted state.

    ``snapshots`` is the public aggregate observer (microcosm#908), off by
    default and strictly separate from the private ``observer`` proof seam:
    the latter carries record-length accepted weights, household IDs, the
    group map and the bounds vector, none of which a public snapshot may ever
    hold. Snapshots read the estimate tensor this epoch's loss was already
    computed from, so enabling them adds no matrix evaluation, no model
    evaluation and no RNG advance, and the returned weights and trajectory are
    bit-identical to the same run with snapshots off.
    """
    if (
        matrix.dtype != torch.float32
        or targets.dtype != torch.float32
        or target_loss_scales.dtype != torch.float32
    ):
        raise ValueError("grouped loss matrix, targets and scales must remain float32")
    w0 = np.array(initial_weights, dtype=np.float64, copy=True)
    groups.check(w0, positive=not preserve_zeros)
    if preserve_zeros:
        _check_fixed_support(w0, w0, groups)
        active = w0 > 0
        accepted = (
            w0.copy()
            if warm_start_weights is None
            else np.array(warm_start_weights, dtype=np.float64, copy=True)
        )
        _check_fixed_support(accepted, w0, groups)
        active_indices = torch.tensor(np.flatnonzero(active), dtype=torch.int64)
        zero_template = w0.copy()
        zero_template[active] = 0.0
        zero_template_t = torch.tensor(zero_template, dtype=torch.float64)
    else:
        active = np.ones(w0.shape, dtype=bool)
        accepted = _prepare_warm_start_weights(
            w0,
            warm_start_weights,
            conserve_mass=False,
            max_weight_ratio=None,
        )
        groups.check(accepted, positive=True)

    def observe(epoch: int, corrected: int) -> None:
        totals = (
            _check_fixed_support(accepted, w0, groups)
            if preserve_zeros
            else groups.check(accepted, positive=True)
        )
        if observer is not None:
            observer(
                {
                    "kind": "grouped_accepted_weights",
                    "epoch": epoch,
                    "weights": accepted.copy(),
                    "household_ids": groups.household_ids,
                    "group_indices": groups.group_indices.copy(),
                    "absolute_bounds": groups.absolute_bounds.copy(),
                    "group_totals": totals.copy(),
                    "corrected_group_count": corrected,
                }
            )

    # Admission and initial observer precede even optimizer construction.
    observe(0, 0)
    log_w = torch.tensor(
        np.log(accepted[active]), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.Adam([log_w], lr=learning_rate)
    anchor_t = None
    if l2_lambda > 0:
        anchor = (
            l2_anchor_weights
            if l2_anchor_weights is not None
            else (w0 if l2_anchor == "initial" else np.full_like(w0, w0.mean()))
        )
        anchor_t = torch.tensor(anchor[active], dtype=torch.float64)
    trajectory = np.empty(epochs, dtype=np.float64)
    selection_labels = (
        None if snapshots is None else _grouped_snapshot_selection(preserve_zeros)
    )
    for epoch in range(epochs):
        optimizer.zero_grad()
        # This is an internal candidate, not an accepted population vector.
        active_weights = torch.exp(log_w)
        weights = (
            zero_template_t.index_copy(0, active_indices, active_weights)
            if preserve_zeros
            else active_weights
        )
        estimate = _apply_constraint(matrix, weights.to(dtype=torch.float32))
        loss = _relative_error_loss(
            estimate,
            targets,
            target_loss_weights,
            target_loss_scales,
            target_loss_cap,
        )
        trajectory[epoch] = float(loss.item())
        if anchor_t is None:
            total_loss = loss
        elif preserve_zeros:
            total_loss = loss + l2_lambda * (
                (active_weights / anchor_t) ** 2
            ).sum() / len(w0)
        else:
            total_loss = loss + l2_lambda * ((weights / anchor_t) ** 2).mean()
        if progress_callback is not None:
            progress_callback(
                {
                    **dict(progress_context or {}),
                    "kind": "calibration_epoch",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": trajectory[epoch],
                }
            )
        if snapshots is not None and snapshots.emits(epoch + 1, epochs):
            # The exact estimate tensor this epoch's loss was computed from,
            # detached rather than recomputed: recomputing would add a matrix
            # evaluation, and the truthful numerical basis for trajectory
            # [epoch] is the float32 forward pass that produced it, even
            # though the accepted state is float64.
            #
            # This is a PRE-update loss-evaluation snapshot, the same
            # convention the ordinary Adam loop above uses. It is deliberately
            # not the post-update projected accepted vector the private
            # observer reports at observe(epoch + 1): those are different
            # vectors, and producing target totals for the accepted one would
            # need the extra evaluation this design forbids.
            snapshots.emit(
                estimate.detach().cpu().numpy(),
                # Completed updates, as in ordinary Adam and proximal snapshots.
                # Cadence above retains the 1-based evaluation ordinal.
                epoch=epoch,
                epochs=epochs,
                iterate=ITERATE_CURRENT,
                precision="float32",
                loss=trajectory[epoch],
                # Grouped Adam retains no best iterate at all, so this is the
                # closed "no retain-best rule is running" triple, not a
                # not-yet-recorded incumbent.
                best_retained={"available": False, "epoch": None, "loss": None},
                selection=selection_labels,
            )
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            active_candidate = torch.exp(log_w).detach().cpu().numpy().copy()
            if preserve_zeros:
                candidate = zero_template.copy()
                candidate[active] = active_candidate
                accepted, corrected = groups.project(candidate, positive=False)
                _check_fixed_support(accepted, w0, groups)
            else:
                accepted, corrected = groups.project(active_candidate)
            # Exactly one overwrite for the next update; no reconstruction loop.
            log_w.copy_(torch.from_numpy(np.log(accepted[active])))
        observe(epoch + 1, corrected)
    closing, _ = groups.project(accepted, positive=not preserve_zeros)
    if preserve_zeros:
        _check_fixed_support(closing, w0, groups)
    if closing.tobytes() != accepted.tobytes():
        raise ValueError("final grouped projection must preserve accepted weight bytes")
    return accepted, trajectory


def _optimize_proximal(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    target_loss_scales: torch.Tensor,
    target_loss_cap: float,
    initial_weights: np.ndarray,
    *,
    warm_start_weights: np.ndarray | None = None,
    epochs: int,
    learning_rate: float,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    l1_lambda: float,
    prune_atol: float,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_context: Mapping[str, object] | None = None,
    snapshots: BoundTargetSnapshots | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Proximal gradient (ISTA-style) on weight ratios -- the L1 selection path.

    Minimizes the same capped weighted-MAPE loss as :func:`_optimize` plus an L1
    penalty ``l1_lambda * mean(w_i / w0_i)``. To keep the optimization well-scaled
    it parameterizes the ratio ``r_i = w_i / w0_i`` (which starts at 1, the same
    O(1) scale the log-weight path enjoys) rather than the raw weights, and uses
    a gradient-RMS-normalized smooth step. After each smooth step the L1 prox --
    the non-negative soft-threshold ``r <- max(r - eta * l1_lambda / n, 0)``
    -- sends unneeded records to *exact* zero, so the returned weight vector is
    sparse (the convex analog of the L0 gates; Adam on log-weights can never reach
    exact zero). The same effective step size ``eta`` used for the smooth update
    is used for the prox, so ``l1_lambda`` is the recorded mean-ratio objective
    coefficient.
    """
    w0 = np.asarray(initial_weights, dtype=np.float64)
    start = _prepare_warm_start_weights(
        w0,
        warm_start_weights,
        conserve_mass=conserve_mass,
        max_weight_ratio=max_weight_ratio,
    )
    total0 = float(w0.sum())
    n = w0.size
    w0_t = torch.tensor(w0, dtype=torch.float32)
    # r = w / w0, so a uniform start is r == 1 (O(1) scale, like the log-weight
    # path). ISTA uses a *plain* gradient step, not Adam: Adam normalizes every
    # coordinate's step to ~lr, which keeps even untargeted records alive and
    # defeats the soft-threshold. Plain gradient lets low-pull records fall to the
    # prox, which is what selects the sparse subset.
    ratio = torch.tensor(start / w0, dtype=torch.float32, requires_grad=True)
    trajectory = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        if ratio.grad is not None:
            ratio.grad = None
        weights = torch.clamp(ratio, min=0.0) * w0_t
        estimate = _apply_constraint(matrix, weights)
        loss = _relative_error_loss(
            estimate,
            targets,
            target_loss_weights,
            target_loss_scales,
            target_loss_cap,
        )
        trajectory[epoch] = float(loss.item())
        if progress_callback is not None:
            progress_callback(
                {
                    **dict(progress_context or {}),
                    "kind": "calibration_epoch",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": trajectory[epoch],
                }
            )
        if snapshots is not None and snapshots.emits(epoch + 1, epochs):
            # The proximal path retains no best iterate (its selection is the
            # closing state), so every in-loop snapshot here is `current`.
            snapshots.emit(
                estimate.detach().numpy(),
                epoch=epoch,
                epochs=epochs,
                iterate=ITERATE_CURRENT,
                precision="float32",
                loss=trajectory[epoch],
            )
        loss.backward()
        with torch.no_grad():
            grad = ratio.grad
            # Scale-robust step: normalize the gradient to unit RMS so the step is
            # ~learning_rate in ratio space regardless of the loss's gradient scale,
            # while preserving each record's *relative* pull (so low-pull records
            # still fall to the prox).
            rms = float(torch.sqrt((grad**2).mean()).item())
            step_size = learning_rate / rms if rms > 0 else learning_rate
            if rms > 0:
                ratio.add_(grad, alpha=-step_size)
            # Prox of l1_lambda * mean(r): max(r - step_size*l1_lambda/n, 0).
            thresh = step_size * l1_lambda / n
            if thresh > 0.0:
                ratio.copy_(torch.clamp(ratio - thresh, min=0.0))
            else:
                ratio.clamp_(min=0.0)
            if max_weight_ratio is not None:
                ratio.clamp_(max=max_weight_ratio)
            # Note: mass conservation is NOT enforced per step. A uniform rescale
            # would multiply shrinking ratios back up and resurrect records the prox
            # is trying to zero, defeating selection. The run optimizes at free mass
            # and conservation is applied once at the end, over the survivors only.

    final = (torch.clamp(ratio, min=0.0) * w0_t).detach().numpy().astype(np.float64)
    # Make sub-threshold residuals exact zeros so the reported n_nonzero and the
    # shipped dataset agree on which records survived.
    final[final <= prune_atol] = 0.0
    if max_weight_ratio is not None:
        final = np.minimum(final, max_weight_ratio * w0)
    if conserve_mass:
        final = _project_to_total(
            final,
            total0,
            max_weight_ratio,
            w0,
            pruned=(final <= 0.0),
            pruning_label="L1 proximal pruning",
            pruning_remedy="lower l1_lambda",
        )
    return final, trajectory


def _search_l0_lambda_for_budget(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    target_loss_scales: torch.Tensor,
    target_loss_cap: float,
    initial_weights: np.ndarray,
    *,
    gate_initialization: GateInitialization | None = None,
    target_records: int,
    epochs: int,
    learning_rate: float,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    l2_lambda: float,
    l2_anchor: str = "initial",
    l2_anchor_weights: np.ndarray | None = None,
    init_mean: float,
    temperature: float,
    seed: int,
    prune_atol: float,
    initial_lambda: float | None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    snapshots: BoundTargetSnapshots | None = None,
    budget_iters: int = _DEFAULT_BUDGET_ITERS,
    return_gate_open_probabilities: bool = False,
    budget_basis: str = BUDGET_BASIS_NONZERO_COUNT,
    feasible_draw_pi_hi: float | None = None,
    search_receipt: dict[str, object] | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, float, int]
    | tuple[np.ndarray, np.ndarray, float, int, np.ndarray]
):
    """Search ``l0_lambda`` so the achieved budget measure tracks the budget.

    ``budget_basis`` names what the search measures against
    ``target_records``: the historical non-zero *count* of surviving weights
    (``BUDGET_BASIS_NONZERO_COUNT``), or the gates' expected open count, the
    sum of their open probabilities (``BUDGET_BASIS_OPEN_PROBABILITY_MASS``).
    An exact-count draw on the learned probabilities can only draw from that
    mass, so a selection that targets a count with the mass basis is feasible
    by construction; the count basis can leave partly polarised gates counted
    as survivors while carrying a fraction of a record each.

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

    ``feasible_draw_pi_hi`` (mass basis only) adds the exact-count draw's own
    inequality to the stopping rule: a probe counts as acceptable only when
    :func:`~microcosm.calibrate.exact_k.exact_k_design_feasibility` says a
    draw of ``target_records`` at that certainty threshold is feasible on its
    gate probabilities. The band is two-sided but the draw's condition is
    one-sided (roughly "open mass at least the budget"), so without it a
    search can stop inside the band on the wrong side and the draw refuses
    (microcosm#355, 2026-09-08: 54,834 open mass for 55,000 requested). An
    infeasible probe steers the bisection like a count miss would: short
    boundary mass toward a smaller penalty, too many certainties toward a
    larger one. Feasible probes are preferred over infeasible ones when the
    best run is chosen; if no probe is feasible within the iteration budget the
    closest run is still returned and the draw refuses with its measurement.
    ``search_receipt``, when supplied, is filled in place with every probe's
    penalty, measure and verdict and the reason the search stopped. Beside the
    per-epoch ``calibration_epoch`` events, ``progress_callback`` receives one
    ``budget_probe`` event per finished probe (the same fields as the receipt's
    probe entry) and a closing ``budget_search_done`` event.

    Args:
        matrix: The constraint matrix ``A`` (dense or sparse-CSR torch tensor), as
            :func:`_optimize` consumes it.
        targets: The target vector tensor.
        initial_weights: The starting weights.
        target_records: The non-zero budget to hit.
        epochs, learning_rate, conserve_mass, max_weight_ratio, init_mean,
            temperature: Passed through to :func:`_optimize`.
        l2_lambda: Fixed soft concentration penalty passed through to
            :func:`_optimize`; the budget search varies only ``l0_lambda``.
        seed: Reseeded before every evaluation for a deterministic response.
        prune_atol: Threshold counting a weight as non-zero (a survivor).
        initial_lambda: A user-supplied ``l0_lambda`` to evaluate first as a warm
            start (clamped into the bracket); ``None`` starts at the bracket
            mid-point.
        budget_iters: Maximum number of optimizations the search may spend.

    Returns:
        ``(weights, trajectory, l0_lambda, n_nonzero)`` of the best run found.
        When ``return_gate_open_probabilities`` is true, the aligned ``pi_i``
        vector is appended. The default four-item tuple is retained for
        existing workspace callers.
    """
    lo_u, hi_u = math.log10(_L0_SEARCH_LO), math.log10(_L0_SEARCH_HI)
    if budget_basis not in BUDGET_BASES:
        raise ValueError(
            f"budget_basis must be one of {sorted(BUDGET_BASES)}, got {budget_basis!r}."
        )
    if budget_basis == BUDGET_BASIS_OPEN_PROBABILITY_MASS:
        return_gate_open_probabilities = True
    if feasible_draw_pi_hi is not None:
        if budget_basis != BUDGET_BASIS_OPEN_PROBABILITY_MASS:
            raise ValueError(
                "feasible_draw_pi_hi requires the open-probability-mass budget basis."
            )
        if (
            isinstance(feasible_draw_pi_hi, bool)
            or not isinstance(feasible_draw_pi_hi, int | float)
            or not math.isfinite(feasible_draw_pi_hi)
            or not (0.0 < float(feasible_draw_pi_hi) <= 1.0)
        ):
            raise ValueError(
                "feasible_draw_pi_hi must be a finite value in (0, 1], got "
                f"{feasible_draw_pi_hi!r}."
            )
        feasible_draw_pi_hi = float(feasible_draw_pi_hi)
    tol = max(1, round(0.05 * target_records))

    evaluation = 0

    def evaluate(
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None]:
        nonlocal evaluation
        evaluation += 1
        torch.manual_seed(seed)
        optimize_kwargs = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "conserve_mass": conserve_mass,
            "max_weight_ratio": max_weight_ratio,
            "l0_lambda": lam,
            "l2_lambda": l2_lambda,
            "l2_anchor": l2_anchor,
            "l2_anchor_weights": l2_anchor_weights,
            "target_records": target_records,
            "init_mean": init_mean,
            "temperature": temperature,
            "progress_callback": progress_callback,
            "progress_context": {
                "budget_search": True,
                "budget_iteration": evaluation,
                "budget_iters": budget_iters,
                "l0_lambda": lam,
            },
            # Each probe is a full reseeded optimization whose epoch numbering
            # restarts, so its snapshots carry the probe's own identity.
            "snapshots": (
                None
                if snapshots is None
                else snapshots.with_search(
                    budget_iteration=evaluation,
                    budget_iters=budget_iters,
                    l0_lambda=lam,
                )
            ),
        }
        if gate_initialization is not None:
            optimize_kwargs["gate_initialization"] = gate_initialization
        if return_gate_open_probabilities:
            weights, trajectory, gate_open_probabilities = _optimize(
                matrix,
                targets,
                target_loss_weights,
                target_loss_scales,
                target_loss_cap,
                initial_weights,
                **optimize_kwargs,
                return_gate_open_probabilities=True,
            )
            if gate_open_probabilities is None:  # pragma: no cover - L0 invariant
                raise RuntimeError(
                    "L0 budget search did not produce gate probabilities."
                )
        else:
            # Preserve the historical call and two-item tuple for private
            # workspace consumers that do not opt into the probability seam.
            weights, trajectory = _optimize(
                matrix,
                targets,
                target_loss_weights,
                target_loss_scales,
                target_loss_cap,
                initial_weights,
                **optimize_kwargs,
            )
            gate_open_probabilities = None
        if budget_basis == BUDGET_BASIS_OPEN_PROBABILITY_MASS:
            if gate_open_probabilities is None:  # pragma: no cover - guarded above
                raise RuntimeError("mass-basis budget search needs gate probabilities.")
            n_nonzero = int(round(float(np.sum(gate_open_probabilities))))
        else:
            n_nonzero = int((weights > prune_atol).sum())
        return weights, trajectory, n_nonzero, gate_open_probabilities

    best: tuple[np.ndarray, np.ndarray, float, int, np.ndarray | None] | None = None
    # (0 for a feasible/unconstrained probe, 1 for an infeasible one; distance
    # to the budget): feasible probes always beat infeasible ones.
    best_key: tuple[int, int] | None = None
    best_evaluation: int | None = None
    probes: list[dict[str, object]] = []
    # Sentinel: a probe whose penalty over-pruned past the cap-feasible floor
    # (the conserve+cap projection raised). It is *more* pruning than feasible,
    # so it steers the search the same way "too few survivors" does — toward a
    # smaller penalty — without crashing the whole search or polluting ``best``.
    _over_pruned = -1
    _not_required = "not_required"
    _steer_smaller = "smaller_penalty"
    _steer_larger = "larger_penalty"
    _steer_stop = "stop"

    def consider(lam: float) -> tuple[int, str]:
        nonlocal best, best_key, best_evaluation
        try:
            weights, trajectory, n_nonzero, gate_open_probabilities = evaluate(lam)
        except ValueError as exc:
            if "Infeasible combination" in str(exc):
                probes.append(
                    {"l0_lambda": lam, "measure": None, "verdict": "over_pruned"}
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "kind": "budget_probe",
                            "budget_iteration": evaluation,
                            "budget_iters": budget_iters,
                            "target_records": target_records,
                            "budget_basis": budget_basis,
                            **probes[-1],
                        }
                    )
                return _over_pruned, "over_pruned"
            raise
        probe: dict[str, object] = {"l0_lambda": lam, "measure": int(n_nonzero)}
        verdict = _not_required
        if feasible_draw_pi_hi is not None:
            if gate_open_probabilities is None:  # pragma: no cover - guarded above
                raise RuntimeError("feasibility-aware search needs gate probabilities.")
            design = exact_k_design_feasibility(
                gate_open_probabilities, target_records, feasible_draw_pi_hi
            )
            verdict = str(design["reason"])
            probe.update(
                certainty_count=int(design["certainty_count"]),
                boundary_draw=int(design["boundary_draw"]),
                boundary_mass=float(design["boundary_mass"]),
                boundary_max=float(design["boundary_max"]),
                feasible=bool(design["feasible"]),
            )
        probe["verdict"] = verdict
        probes.append(probe)
        if progress_callback is not None:
            # One event per finished probe, beside the per-epoch stream, so a
            # long search is readable from a log: penalty, measure, verdict.
            progress_callback(
                {
                    "kind": "budget_probe",
                    "budget_iteration": evaluation,
                    "budget_iters": budget_iters,
                    "target_records": target_records,
                    "budget_basis": budget_basis,
                    **probe,
                }
            )
        key = (
            0 if verdict in (_not_required, FEASIBLE) else 1,
            abs(n_nonzero - target_records),
        )
        if best_key is None or key < best_key:
            best = (
                weights,
                trajectory,
                lam,
                n_nonzero,
                gate_open_probabilities,
            )
            best_key = key
            best_evaluation = evaluation
        return n_nonzero, verdict

    def steer(n_nonzero: int, verdict: str) -> str:
        # An infeasible draw steers like a count miss: short boundary mass needs
        # more open mass (smaller penalty); surplus certainties need less.
        if n_nonzero == _over_pruned or verdict in (
            BOUNDARY_MASS_SHORT,
            BOUNDARY_SHORT_OF_DRAW,
        ):
            return _steer_smaller
        if verdict == CERTAINTIES_EXCEED_K:
            return _steer_larger
        if n_nonzero < target_records:
            return _steer_smaller
        if n_nonzero > target_records:
            return _steer_larger
        return _steer_stop

    def settled() -> bool:
        return (
            best is not None
            and best_key is not None
            and best_key[0] == 0
            and abs(best[3] - target_records) <= tol
        )

    # Warm start: evaluate the user's lambda (or the bracket mid-point) first.
    if initial_lambda is not None and initial_lambda > 0:
        first_u = min(max(math.log10(initial_lambda), lo_u), hi_u)
    else:
        first_u = (lo_u + hi_u) / 2.0
    iters_left = budget_iters
    n_nonzero, verdict = consider(10.0**first_u)
    iters_left -= 1
    # Seed the bracket so the side the warm start landed on is tightened. An
    # over-pruned (infeasible) probe groups with "too few survivors".
    if steer(n_nonzero, verdict) == _steer_larger:
        lo_u = first_u  # too many survivors -> need a larger penalty
    else:
        hi_u = first_u  # too few survivors / over-pruned -> need a smaller penalty

    # Keep searching while no acceptable run is known yet, or the best is
    # outside tolerance, until the iteration budget is spent.
    while iters_left > 0 and not settled():
        mid_u = (lo_u + hi_u) / 2.0
        n_nonzero, verdict = consider(10.0**mid_u)
        iters_left -= 1
        direction = steer(n_nonzero, verdict)
        if direction == _steer_smaller:
            hi_u = mid_u  # over-pruned / too few / short mass -> smaller penalty
        elif direction == _steer_larger:
            lo_u = mid_u  # too many survivors or certainties -> larger penalty
        else:
            break

    if progress_callback is not None:
        progress_callback(
            {
                "kind": "budget_search_done",
                "evaluations": int(evaluation),
                "budget_iters": int(budget_iters),
                "target_records": int(target_records),
                "stopped_on": (
                    "acceptable_within_tolerance" if settled() else "budget_exhausted"
                ),
                "selected_l0_lambda": None if best is None else float(best[2]),
                "selected_budget_iteration": best_evaluation,
                "selected_measure": None if best is None else int(best[3]),
                "selected_feasible": (
                    None
                    if best_key is None or feasible_draw_pi_hi is None
                    else best_key[0] == 0
                ),
            }
        )
    if search_receipt is not None:
        search_receipt.update(
            {
                "budget_basis": budget_basis,
                "target_records": int(target_records),
                "tolerance": int(tol),
                "budget_iters": int(budget_iters),
                "feasible_draw_pi_hi": feasible_draw_pi_hi,
                "evaluations": int(evaluation),
                "probes": probes,
                "selected_l0_lambda": None if best is None else float(best[2]),
                "selected_budget_iteration": best_evaluation,
                "selected_measure": None if best is None else int(best[3]),
                "selected_feasible": (
                    None
                    if best_key is None or feasible_draw_pi_hi is None
                    else best_key[0] == 0
                ),
                "stopped_on": (
                    "acceptable_within_tolerance" if settled() else "budget_exhausted"
                ),
            }
        )

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
    if return_gate_open_probabilities:
        if best[4] is None:  # pragma: no cover - opt-in invariant
            raise RuntimeError("L0 budget search lost its gate probabilities.")
        return best
    return best[:4]


def _project_to_total(
    weights: np.ndarray,
    total: float,
    max_weight_ratio: float | None,
    initial_weights: np.ndarray,
    *,
    pruned: np.ndarray | None = None,
    pruning_label: str = "L0 pruning",
    pruning_remedy: str = "loosen the record budget (smaller l0_lambda)",
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
        pruning_label: Human-readable name of the pruning path for infeasibility
            errors.
        pruning_remedy: Path-specific remediation text for infeasibility errors.

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
                    f"Infeasible combination: {pruning_label} + mass='conserve' + "
                    f"max_weight_ratio={max_weight_ratio!r}. After pruning "
                    f"{int(pruned.sum())} record(s), the {int(eligible.sum())} "
                    "surviving record(s) cannot absorb the freed mass under the "
                    "cap (sum of survivor caps < input total). Raise "
                    f"max_weight_ratio, {pruning_remedy}, or use mass='free'."
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
    gate_initialization: GateInitialization | None = None,
    weight_entity: str = "household",
    method: str = "adam",
    epochs: int = 256,
    learning_rate: float = 0.02,
    mass: str = FREE_MASS,
    mass_reason: str | None = None,
    max_weight_ratio: float | None = None,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    l2_anchor: str = "initial",
    l2_anchor_weights: np.ndarray | None = None,
    init_mean: float = 0.999,
    temperature: float = 0.25,
    budget_iters: int = _DEFAULT_BUDGET_ITERS,
    budget_basis: str = BUDGET_BASIS_NONZERO_COUNT,
    feasible_draw_pi_hi: float | None = None,
    seed: int = 0,
    target_loss_weights: np.ndarray | None = None,
    target_loss_scales: np.ndarray | None = None,
    target_loss_cap: float = _DEFAULT_TARGET_LOSS_CAP,
    warm_start_weights: np.ndarray | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    target_snapshots: TargetSnapshotObserver | None = None,
    grouped_upper_bounds: GroupedUpperBounds | None = None,
    grouped_preserve_zeros: bool = False,
    _post_projection_observer: Callable[[dict[str, object]], None] | None = None,
) -> CalibrationResult:
    """Calibrate ``weight_entity``'s weights to ``targets`` over ``frame``.

    Compiles the targets into a sparse system and optimizes the log-weights with
    Adam to minimize capped fixed-scale weighted MAPE
    ``weighted_mean(min(abs((A @ w - b) / s), cap))``. Returns a new frame whose
    ``weight_entity`` weights are :class:`~microcosm.frame.WeightKind.CALIBRATED`.

    Args:
        frame: The frame to calibrate.
        targets: The :class:`~microcosm.calibrate.target.TargetSet` of facts.
        weight_entity: Entity whose weights to calibrate (default
            ``"household"``).
        method: Optimization method. ``"adam"`` (default) runs the torch Adam
            optimizer on the log-weights described above (weights stay strictly
            positive by construction). Ungrouped unregularized free-mass runs
            retain their best feasible fit within the requested epoch budget.
            ``"prox"`` runs proximal
            gradient (ISTA) on raw non-negative weight ratios with a
            soft-threshold step, the optimizer required for the nonsmooth
            ``l1_lambda`` penalty: it drives unneeded records to exact zero, so
            L1 selects a sparse weighted subset. ``"apg"`` is accepted only as a
            deprecated alias for ``"adam"`` and is recorded as ``"adam"`` in
            result options. Any other value is rejected.
        epochs: Number of optimization steps.
        learning_rate: Adam learning rate on the log-weights. Capped MAPE has a
            nearly constant gradient away from zero, so the default is lower
            than the old squared-error objective to avoid oscillating around
            feasible targets.
        mass: :data:`FREE_MASS` (default) to let the total move, or
            :data:`CONSERVE_MASS` to hold it to the input total.
        mass_reason: Optional caller-supplied reason recorded on the
            :class:`~microcosm.frame.MassChange` a :data:`FREE_MASS`
            calibration appends to the frame's mass log — domain context
            (which target families moved the mass) instead of the generic
            default. Must be ``None`` under :data:`CONSERVE_MASS`, which
            appends no record.
        max_weight_ratio: If given, a hard per-record cap: no calibrated weight
            exceeds ``max_weight_ratio * initial_weight``. The landmine guard.
        grouped_upper_bounds: Optional exhaustive ordered household grouping
            with frozen absolute bounds. This positive-only Adam/free-mass path
            refuses scalar row caps, prox, L0 and record selection. Both input
            and effective warm-start weights must already be feasible. Accepted
            float64 vectors use stable-ID ``math.fsum`` and direct weight-space
            projection; float32 matrix/loss arithmetic remains differentiable.
        grouped_preserve_zeros: Opt in to fixed initial zero support for grouped
            Adam. Zero rows remain in the full Frame at their original zero
            values; only positive coordinates receive log parameters. Both
            warm starts and every accepted/stored vector must preserve that
            support exactly. An entirely zero initial vector is refused. The
            L2 penalty sums positive-coordinate contributions divided by the
            full row count; zero coordinates contribute zero. The default
            retains the existing strictly positive grouped contract.
        _post_projection_observer: Private grouped proof seam, called with
            independent copies of the validated initial and every accepted
            post-update vector, IDs, group map, bounds and exact group totals.
        target_records: If given, enable L0 pruning with **budget control**: the
            solver searches ``l0_lambda`` (a bisection on its log, ``budget_iters``
            optimizations) so the achieved non-zero count tracks this budget, and
            reports the penalty it settled on as
            :attr:`CalibrationResult.l0_lambda`. A supplied ``l0_lambda`` is the
            search's warm start. The achieved count tracks the budget within a
            tolerance (the count is a noisy discrete function of the penalty), not
            exactly. ``budget_basis`` names the measure the search tracks (the
            surviving non-zero count, or the gates' open-probability mass);
            ``feasible_draw_pi_hi`` (mass basis only) makes the search stop only
            on a probe whose gate probabilities admit an exact-count draw of
            ``target_records`` at that certainty threshold, recording every probe
            under ``options["budget_search"]``.
        l0_lambda: L0 penalty strength. Used directly when ``target_records`` is
            ``None``: ``> 0`` enables hard-concrete gates that prune the pool,
            ``0.0`` (default) keeps every record. When ``target_records`` is set,
            this is only the budget search's warm start (the search overrides it).
        l1_lambda: L1 penalty strength for ``method="prox"``. Positive values
            add ``l1_lambda * mean(weight / initial_weight)`` to the objective and
            use a proximal soft-threshold step that can drive records to exact
            zero. ``l1_lambda`` must be zero unless ``method="prox"``.
        l2_lambda: Experimental soft concentration penalty strength. ``0.0``
            (default) preserves the unpenalized path. Positive values add
            ``l2_lambda * mean((pre_gate_weight / initial_weight) ** 2)`` to the
            optimization loss while leaving ``max_weight_ratio`` as the hard
            per-record cap. When L0 gates are active, this is a latent pre-gate
            penalty on ``exp(log_w)``, not the realized gated returned weight;
            that preserves the original L0 behavior of discouraging hidden
            weight explosion behind partially closed gates. Its ESS/design-effect
            interpretation is cleanest with ``mass="conserve"``; with
            ``mass="free"`` it also penalizes total weight scale.
        l2_anchor: The L2 penalty's reference weights. ``"initial"`` (default)
            divides by each record's own starting weight — a "stay near the
            start" prior. ``"uniform"`` divides by the shared mean starting
            weight, making the penalty (under ``mass="conserve"``) a direct
            1/ESS control. The distinction matters when starting weights are
            heterogeneous: the mass-constrained optimum of the penalty is
            ``w_i ∝ anchor_i²``, so an ``"initial"`` anchor on concentrated
            starting weights (e.g. a post-L0 refit) pulls toward *more*
            concentration, not less. Only consulted when ``l2_lambda > 0``.
        l2_anchor_weights: Optional explicit anchor vector (one positive
            finite value per ``weight_entity`` record), overriding the
            ``l2_anchor`` presets — the harness seam for anchors the frame
            cannot express, e.g. the pre-selection *design* weights during a
            post-L0 refit. When supplied, ``l2_anchor`` becomes a free-form
            provenance label recorded in options (e.g. ``"design"``).
        gate_initialization: Optional informed L0 prior and protected-record
            mask, aligned to the weight entity. Requires an Adam L0 solve;
            protected gates remain open throughout training and selection.
        init_mean: Initial expected open-probability of the L0 gates (only used
            when pruning).
        temperature: Hard-concrete temperature (only used when pruning).
        budget_iters: Maximum optimizations the ``target_records`` budget search
            may spend bisecting ``l0_lambda`` (only used when ``target_records``
            is set). Higher resolves the budget finer at a proportional cost.
        seed: Seed for torch's RNG (the gate sampling), for reproducibility.
        target_loss_weights: Optional non-negative row weights aligned to the
            supplied :class:`TargetSet`. When omitted, every compiled target row
            contributes equally. When supplied, the weights for skipped targets
            are dropped with those targets, and the capped scaled misses for
            compiled rows are averaged with the remaining weights, normalized by
            their sum.
        target_loss_scales: Optional positive row scales aligned to the supplied
            :class:`TargetSet`. When omitted, compiled rows use
            :func:`default_target_loss_scales`, fixed from target values only.
            Supplying scales is mainly for harnesses and specialized releases.
        target_loss_cap: Positive per-row cap on scaled absolute misses. The
            default caps each target's objective contribution at 1000%.
        warm_start_weights: Optional positive starting weights aligned to
            ``weight_entity``. These initialize the optimizer only; the frame's
            original weights still define mass conservation, hard ratio caps,
            diagnostics' initial estimates, and release provenance. Warm starts
            are not yet supported with L0 gates or record-budget search because
            that would also need persisted gate state.
        progress_callback: Optional observer called during optimization with
            JSON-serializable dictionaries such as ``{"kind":
            "calibration_epoch", "epoch": 1, "epochs": 256, "loss": ...}``.
            Build drivers use this to publish staging telemetry; calibration
            results are unchanged.
        target_snapshots: Optional
            :class:`~microcosm.calibrate.target_snapshots.TargetSnapshotObserver`
            receiving aggregate-only per-target estimate snapshots at its own
            cadence while the run is in flight (microcosm#908). Off by default;
            enabling it does not change the weights, the trajectory, or the
            RNG stream — in-loop snapshots read the estimate tensor the epoch's
            loss was already computed from. The closing ``selected`` snapshot
            describes the weights calibration actually returns.

            ``grouped_upper_bounds`` runs are instrumented the same way and at
            the same seams. Because grouped Adam is a closing-state algorithm,
            every grouped snapshot reports ``best_retained.available: False``
            and carries the bounded ``selection`` labels ``rule``,
            ``constraint_mode`` and ``grouped_preserve_zeros`` so a consumer
            can tell "this solver retains no best iterate" from "retain-best
            was switched off". The private ``_post_projection_observer`` proof
            seam stays entirely separate: its record-length weights, household
            IDs, group map and bounds never reach a snapshot.

    Returns:
        A :class:`CalibrationResult` with the calibrated frame, per-target
        diagnostics, loss trajectory, and any skipped targets.

    Raises:
        ValueError: If ``method`` is unknown, ``mass`` is not ``"free"`` or
            ``"conserve"``, ``epochs`` is not positive, ``max_weight_ratio`` is
            given and is not ``> 0``, or ``target_records`` is given and is not
            a positive integer, if ``l1_lambda`` or ``l2_lambda`` is negative or
            non-finite, if ``l1_lambda`` is used outside ``method="prox"``, if
            ``method="prox"`` is combined with L0/budget pruning or
            ``l2_lambda``, or if no targets compile (from the matrix build).
    """
    _validate_grouped_zero_option(grouped_upper_bounds, grouped_preserve_zeros)
    # Normalize and value-check method/mass before grouped-mode validation:
    # grouped bounds are documented as Adam with free mass, so the deprecated
    # 'apg' alias must reach that check as 'adam', and an invalid mass string
    # must report itself rather than be read as a mass-conserving request.
    if method == "apg":
        warnings.warn(
            "method='apg' is deprecated and now aliases method='adam'; result "
            "options record the normalized method='adam'.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "adam"
    if method not in ("adam", "prox"):
        raise ValueError(
            f"Unknown method {method!r}; supported: 'adam' (Adam on log-weights), "
            "'prox' (proximal gradient on raw weights, for the l1_lambda penalty), "
            "and deprecated alias 'apg' -> 'adam'."
        )
    if mass not in (FREE_MASS, CONSERVE_MASS):
        raise ValueError(
            f"mass must be {FREE_MASS!r} or {CONSERVE_MASS!r}, got {mass!r}."
        )
    if grouped_upper_bounds is not None:
        _validate_grouped_mode(
            grouped_upper_bounds,
            method=method,
            conserve_mass=(mass != FREE_MASS),
            max_weight_ratio=max_weight_ratio,
            l0_lambda=l0_lambda,
            l1_lambda=l1_lambda,
            target_records=target_records,
        )
        if weight_entity != "household":
            raise ValueError("grouped upper bounds require household weights")
        actual_ids = tuple(
            frame.table(weight_entity)[
                frame.schema.entity_id_column(weight_entity)
            ].tolist()
        )
        if actual_ids != grouped_upper_bounds.household_ids:
            raise ValueError(
                "group household IDs must exactly match the frame's ordered IDs"
            )
    elif _post_projection_observer is not None:
        raise ValueError("post-projection observer requires grouped upper bounds")
    last_group_correction = 0

    def grouped_observer(payload: dict[str, object]) -> None:
        nonlocal last_group_correction
        _check_grouped_household_ids(frame, actual_ids)
        last_group_correction = int(payload["corrected_group_count"])
        if _post_projection_observer is not None:
            _post_projection_observer(payload)
        _check_grouped_household_ids(frame, actual_ids)

    if mass_reason is not None:
        if mass != FREE_MASS:
            raise ValueError(
                "mass_reason requires mass='free'; a mass-conserving "
                "calibration appends no mass record to carry it."
            )
        if not isinstance(mass_reason, str) or not mass_reason.strip():
            raise ValueError("mass_reason must be a non-empty string when provided.")
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
    if gate_initialization is not None:
        if method != "adam" or (target_records is None and l0_lambda <= 0):
            raise ValueError("gate_initialization requires an Adam L0 solve.")
    if target_records is not None and (
        not isinstance(target_records, int) or target_records <= 0
    ):
        raise ValueError(
            f"target_records must be a positive integer, got {target_records!r}."
        )
    if not math.isfinite(l2_lambda) or l2_lambda < 0.0:
        raise ValueError(
            f"l2_lambda must be finite and non-negative, got {l2_lambda!r}."
        )
    if l2_anchor_weights is None:
        if l2_anchor not in ("initial", "uniform"):
            raise ValueError(
                f"l2_anchor must be 'initial' or 'uniform', got {l2_anchor!r}."
            )
    else:
        if not isinstance(l2_anchor, str) or not l2_anchor:
            raise ValueError(
                "l2_anchor must be a non-empty label when l2_anchor_weights "
                f"is supplied, got {l2_anchor!r}."
            )
        l2_anchor_weights = np.asarray(l2_anchor_weights, dtype=np.float64)
        if not np.isfinite(l2_anchor_weights).all() or (l2_anchor_weights <= 0).any():
            raise ValueError("l2_anchor_weights must be finite and strictly positive.")
    if not math.isfinite(l1_lambda) or l1_lambda < 0.0:
        raise ValueError(
            f"l1_lambda must be finite and non-negative, got {l1_lambda!r}."
        )
    if l1_lambda > 0.0 and method != "prox":
        raise ValueError(
            "l1_lambda requires method='prox': the nonsmooth L1 penalty needs the "
            "proximal soft-threshold step, which Adam cannot provide (no exact zeros)."
        )
    if method == "prox" and (l0_lambda > 0.0 or target_records is not None):
        raise ValueError(
            "method='prox' is the L1 selection path and does not use L0 gates; pass "
            "l0_lambda=0 and target_records=None (use method='adam' for L0/budget search)."
        )
    if method == "prox" and l2_lambda > 0.0:
        raise ValueError(
            "method='prox' does not implement l2_lambda; use method='adam' for the "
            "L2 concentration penalty or pass l2_lambda=0."
        )
    if warm_start_weights is not None and (
        l0_lambda > 0.0 or target_records is not None
    ):
        raise ValueError(
            "warm_start_weights is not supported with L0 pruning or target_records "
            "budget search yet; persisted gate state is needed for a faithful "
            "warm start of sparse calibration."
        )
    if budget_iters <= 0:
        raise ValueError(f"budget_iters must be positive, got {budget_iters!r}.")
    if budget_basis not in BUDGET_BASES:
        raise ValueError(
            f"budget_basis must be one of {sorted(BUDGET_BASES)}, got {budget_basis!r}."
        )
    if budget_basis != BUDGET_BASIS_NONZERO_COUNT and target_records is None:
        raise ValueError("budget_basis applies to a target_records budget search.")
    if feasible_draw_pi_hi is not None:
        if target_records is None or budget_basis != BUDGET_BASIS_OPEN_PROBABILITY_MASS:
            raise ValueError(
                "feasible_draw_pi_hi requires a target_records budget search on the "
                "open-probability-mass basis."
            )
        if (
            isinstance(feasible_draw_pi_hi, bool)
            or not isinstance(feasible_draw_pi_hi, int | float)
            or not math.isfinite(feasible_draw_pi_hi)
            or not (0.0 < float(feasible_draw_pi_hi) <= 1.0)
        ):
            raise ValueError(
                "feasible_draw_pi_hi must be a finite value in (0, 1], got "
                f"{feasible_draw_pi_hi!r}."
            )
        feasible_draw_pi_hi = float(feasible_draw_pi_hi)
    target_loss_cap = _validate_target_loss_cap(target_loss_cap)

    target_loss_weights_input = _validate_target_loss_weights(
        target_loss_weights,
        (len(targets),),
    )
    target_loss_scales_input = (
        None
        if target_loss_scales is None
        else _validate_target_loss_scales(
            target_loss_scales,
            (len(targets),),
            targets=np.asarray([target.value for target in targets], dtype=np.float64),
        )
    )
    problem = build_constraint_matrix(frame, targets, weight_entity)
    snapshots = (
        None
        if target_snapshots is None
        else target_snapshots.bind(names=problem.names, targets=problem.target_vector)
    )
    initial = problem.initial_weights
    w0 = initial.values
    prune_atol = _PRUNE_REL_ATOL * float(np.mean(w0))
    if l2_anchor_weights is not None and l2_anchor_weights.shape != w0.shape:
        raise ValueError(
            f"l2_anchor_weights shape {l2_anchor_weights.shape} must match "
            f"the calibrated weight vector shape {w0.shape}."
        )

    torch.manual_seed(seed)
    matrix_t = _torch_constraint_matrix(problem.matrix)
    targets_t = torch.tensor(problem.target_vector, dtype=torch.float32)
    target_loss_weights_np: np.ndarray | None = None
    target_loss_scales_np: np.ndarray
    target_loss_scale_kind = "default_target"
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
    if target_loss_scales_input is not None:
        scales_by_key = {
            target.key: scale
            for target, scale in zip(targets, target_loss_scales_input, strict=True)
        }
        target_loss_scales_np = np.asarray(
            [scales_by_key[target.key] for target in problem.targets],
            dtype=np.float64,
        )
        target_loss_scales_np = _validate_target_loss_scales(
            target_loss_scales_np,
            problem.target_vector.shape,
            targets=problem.target_vector,
        )
        target_loss_scale_kind = "provided"
    else:
        target_loss_scales_np = default_target_loss_scales(problem.target_vector)
    target_loss_weights_t = (
        torch.tensor(target_loss_weights_np, dtype=torch.float32)
        if target_loss_weights_np is not None
        else None
    )
    target_loss_scales_t = torch.tensor(target_loss_scales_np, dtype=torch.float32)

    iterate_selection_receipt: dict[str, object] = {}
    budget_search: dict[str, object] | None = None
    if method == "prox":
        # L1 path: proximal gradient (ISTA) on raw weights. The soft-threshold
        # drives unneeded records to exact zero, so L1 selects a sparse weighted
        # subset jointly with calibrating it (the convex analog of the L0 gates).
        effective_l0 = 0.0
        gate_open_probabilities = None
        final_weights, trajectory = _optimize_proximal(
            matrix_t,
            targets_t,
            target_loss_weights_t,
            target_loss_scales_t,
            target_loss_cap,
            w0,
            epochs=epochs,
            learning_rate=learning_rate,
            conserve_mass=(mass == CONSERVE_MASS),
            max_weight_ratio=max_weight_ratio,
            warm_start_weights=warm_start_weights,
            l1_lambda=l1_lambda,
            prune_atol=prune_atol,
            progress_callback=progress_callback,
            snapshots=snapshots,
        )
        n_nonzero = int((final_weights > prune_atol).sum())
        if n_nonzero == 0:
            if l1_lambda > 0.0:
                raise ValueError(
                    f"L1 penalty zeroed every weight: l1_lambda={l1_lambda!r} "
                    "overwhelmed the fit loss. Lower l1_lambda."
                )
            raise ValueError(
                "method='prox' returned every calibrated weight as zero. "
                "Microcosm frames require at least one positive calibrated weight; "
                "use method='adam', mass='conserve', or a target surface that "
                "does not make zero total mass optimal."
            )
    elif target_records is not None:
        # Budget control (Finding 3): search l0_lambda so the achieved non-zero
        # count tracks target_records. The supplied l0_lambda (if any) is the
        # warm start; the search reports the penalty it settled on.
        budget_search = {}
        (
            final_weights,
            trajectory,
            effective_l0,
            n_nonzero,
            gate_open_probabilities,
        ) = _search_l0_lambda_for_budget(
            matrix_t,
            targets_t,
            target_loss_weights_t,
            target_loss_scales_t,
            target_loss_cap,
            w0,
            target_records=target_records,
            epochs=epochs,
            learning_rate=learning_rate,
            conserve_mass=(mass == CONSERVE_MASS),
            max_weight_ratio=max_weight_ratio,
            l2_lambda=l2_lambda,
            l2_anchor=l2_anchor,
            l2_anchor_weights=l2_anchor_weights,
            init_mean=init_mean,
            temperature=temperature,
            seed=seed,
            prune_atol=prune_atol,
            initial_lambda=(l0_lambda if l0_lambda > 0.0 else None),
            progress_callback=progress_callback,
            snapshots=snapshots,
            budget_iters=budget_iters,
            budget_basis=budget_basis,
            feasible_draw_pi_hi=feasible_draw_pi_hi,
            search_receipt=budget_search,
            return_gate_open_probabilities=True,
            **(
                {}
                if gate_initialization is None
                else {"gate_initialization": gate_initialization}
            ),
        )
    else:
        effective_l0 = l0_lambda
        final_weights, trajectory, gate_open_probabilities = _optimize(
            matrix_t,
            targets_t,
            target_loss_weights_t,
            target_loss_scales_t,
            target_loss_cap,
            w0,
            epochs=epochs,
            learning_rate=learning_rate,
            conserve_mass=(mass == CONSERVE_MASS),
            max_weight_ratio=max_weight_ratio,
            warm_start_weights=warm_start_weights,
            l0_lambda=effective_l0,
            l2_lambda=l2_lambda,
            l2_anchor=l2_anchor,
            l2_anchor_weights=l2_anchor_weights,
            target_records=target_records,
            init_mean=init_mean,
            temperature=temperature,
            progress_callback=progress_callback,
            snapshots=snapshots,
            return_gate_open_probabilities=True,
            selection_receipt=iterate_selection_receipt,
            grouped_upper_bounds=grouped_upper_bounds,
            grouped_preserve_zeros=grouped_preserve_zeros,
            _post_projection_observer=grouped_observer
            if grouped_upper_bounds is not None
            else None,
            **(
                {}
                if gate_initialization is None
                else {"gate_initialization": gate_initialization}
            ),
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

    accepted_weight_bytes = (
        final_weights.tobytes() if grouped_upper_bounds is not None else None
    )
    calibrated = initial.with_values(final_weights, kind=WeightKind.CALIBRATED)
    new_frame = _apply_weights(
        frame,
        weight_entity,
        initial,
        calibrated,
        mass,
        targets,
        mass_reason=mass_reason,
    )
    diagnostics = _build_diagnostics(problem, frame, w0, final_weights)
    # One closing eval-mode loss on the RETURNED weights (Finding 8): the same
    # capped weighted-MAPE loss the optimizer minimizes, evaluated after the closing
    # mass/cap projections — so final_loss describes
    # what calibrate returns, not the trajectory's pre-projection tail.
    final_estimates = problem.estimates(final_weights)
    closing_loss = relative_error_loss(
        final_estimates,
        problem.target_vector,
        target_loss_weights=target_loss_weights_np,
        target_loss_scales=target_loss_scales_np,
        target_loss_cap=target_loss_cap,
    )
    if snapshots is not None:
        # The selected snapshot describes the weights calibration RETURNS,
        # read off the same float64 estimates the final diagnostics use — not
        # the last in-loop iterate, which the optimizer may have discarded in
        # favour of an earlier better one or changed by a closing projection.
        #
        # Grouped Adam is a closing-state algorithm. It returns the accepted
        # vector of its last projection and never runs the retain-best rule,
        # so its retain-best answer is read off the solver it actually used,
        # not inferred from an empty receipt. Inferring it would make a later
        # change that populated a grouped receipt silently claim the grouped
        # solver had retained a best iterate.
        grouped_run = grouped_upper_bounds is not None
        selected_epoch = (
            None if grouped_run else iterate_selection_receipt.get("selected_epoch")
        )
        # A non-empty receipt means the optimizer ran the retain-best rule. It
        # records the epoch it selected, which IS the best iterate's epoch when
        # an earlier iterate won; when the closing iterate won, no separate best
        # epoch was recorded, so the snapshot says "retained, epoch unrecorded"
        # instead of inventing one.
        retained_best = (not grouped_run) and bool(iterate_selection_receipt)
        best_is_earlier = (
            retained_best
            and isinstance(selected_epoch, int)
            and selected_epoch < epochs
        )
        selected_snapshots = snapshots
        if budget_search is not None:
            selected_snapshots = snapshots.with_search(
                budget_iteration=budget_search["selected_budget_iteration"],
                budget_iters=budget_search["budget_iters"],
                l0_lambda=budget_search["selected_l0_lambda"],
            )
        selected_snapshots.emit(
            final_estimates,
            # The epoch whose iterate was actually selected, not the last one
            # executed: a retained-best run returns an earlier iterate, and
            # stamping the closing epoch on it would misattribute the values.
            epoch=(int(selected_epoch) if isinstance(selected_epoch, int) else epochs),
            epochs=epochs,
            iterate=ITERATE_SELECTED,
            precision="float64",
            loss=float(closing_loss),
            best_retained={
                "available": retained_best,
                "epoch": int(selected_epoch) if best_is_earlier else None,
                "loss": (
                    iterate_selection_receipt.get("selected_loss_float32")
                    if best_is_earlier
                    else None
                ),
            },
            # The grouped receipt stays empty by design, so the grouped
            # selection identity is the bounded mode label instead. Both are
            # aggregate scalars; neither carries record-level content.
            selection=(
                _grouped_snapshot_selection(grouped_preserve_zeros)
                if grouped_run
                else (dict(iterate_selection_receipt) or None)
            ),
        )
    effective_target_loss_weights = (
        np.ones(problem.target_vector.shape, dtype=np.float64)
        if target_loss_weights_np is None
        else target_loss_weights_np.copy()
    )

    result = CalibrationResult(
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
        target_loss_weights=effective_target_loss_weights,
        target_loss_scales=target_loss_scales_np.copy(),
        target_loss_cap=target_loss_cap,
        options={
            "gate_initialization_supplied": gate_initialization is not None,
            "budget_basis": budget_basis,
            "feasible_draw_pi_hi": feasible_draw_pi_hi,
            "budget_search": budget_search,
            **(
                {
                    "grouped_preserve_zeros": {
                        "enabled": True,
                        "fixed_zero_count": int(np.count_nonzero(w0 == 0)),
                        "ordered_zero_mask_sha256": hashlib.sha256(
                            np.asarray(w0 == 0, dtype=np.uint8).tobytes()
                        ).hexdigest(),
                    }
                }
                if grouped_preserve_zeros
                else {}
            ),
            **(
                {
                    "grouped_upper_bounds": grouped_upper_bounds.diagnostics(
                        final_weights, last_group_correction
                    )
                }
                if grouped_upper_bounds is not None
                else {}
            ),
            "method": method,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "iterate_selection": (
                "best_feasible_loss"
                if grouped_upper_bounds is None
                and method == "adam"
                and mass == FREE_MASS
                and l0_lambda == 0.0
                and l2_lambda == 0.0
                and target_records is None
                else "closing_state"
            ),
            "iterate_selection_receipt": iterate_selection_receipt,
            "mass": mass,
            "mass_reason": mass_reason,
            "max_weight_ratio": max_weight_ratio,
            "target_records": target_records,
            "l1_lambda": l1_lambda,
            "l1_penalty": "mean_initial_weight_ratio_abs",
            "l2_lambda": l2_lambda,
            "l2_anchor": l2_anchor,
            "l2_anchor_weights_supplied": l2_anchor_weights is not None,
            "l2_penalty": (
                "mean_explicit_anchor_pre_gate_weight_ratio_squared"
                if l2_anchor_weights is not None
                else "mean_initial_pre_gate_weight_ratio_squared"
                if l2_anchor == "initial"
                else "mean_uniform_pre_gate_weight_ratio_squared"
            ),
            "seed": seed,
            "target_loss_weights": _target_loss_weight_options(target_loss_weights_np),
            "target_loss_scales": _target_loss_scale_options(
                target_loss_scales_np,
                kind=target_loss_scale_kind,
                target_loss_cap=target_loss_cap,
            ),
            "warm_start_weights": {
                "enabled": warm_start_weights is not None,
                "kind": "explicit" if warm_start_weights is not None else None,
            },
            "matrix_format": (
                "sparse_csr" if matrix_t.layout == torch.sparse_csr else "dense"
            ),
        },
        gate_open_probabilities=gate_open_probabilities,
    )
    if grouped_upper_bounds is not None:
        # Admission is last, after diagnostics/options and their possible hooks.
        # Frozen accepted bytes precede materialization, so changing two live
        # vector aliases cannot make a changed result validate itself.
        stored = result.frame.resolve_weights(weight_entity).values
        closing, _ = grouped_upper_bounds.project(
            stored, positive=not grouped_preserve_zeros
        )
        if grouped_preserve_zeros:
            _check_fixed_support(stored, w0, grouped_upper_bounds)
            _check_fixed_support(closing, w0, grouped_upper_bounds)
        if (
            result.weights.tobytes() != accepted_weight_bytes
            or stored.tobytes() != accepted_weight_bytes
            or closing.tobytes() != accepted_weight_bytes
        ):
            raise ValueError(
                "stored grouped weights must preserve exact final accepted bytes"
            )
        _check_grouped_household_ids(frame, actual_ids)
        _check_grouped_household_ids(result.frame, actual_ids)
    return result


def rebuild_calibration_result(
    frame: Frame,
    targets: TargetSet,
    *,
    weight_entity: str = "household",
    weights: np.ndarray,
    loss_trajectory: np.ndarray,
    l0_lambda: float,
    n_nonzero: int,
    target_loss_weights: np.ndarray,
    target_loss_scales: np.ndarray,
    target_loss_cap: float,
    options: Mapping[str, object],
    gate_open_probabilities: np.ndarray | None = None,
    closing_loss: float | None = None,
) -> CalibrationResult:
    """Re-assemble a :class:`CalibrationResult` from a solve's persisted outputs.

    Compiles ``targets`` against ``frame`` exactly as :func:`calibrate` does,
    places ``weights`` on the frame under the recorded mass policy
    (``options["mass"]`` / ``options["mass_reason"]``), and rebuilds the
    per-target diagnostics and the closing loss from the compiled system.
    Nothing is optimised: this is the checkpoint/resume seam for builds whose
    solve is expensive and whose later stages (an exact-count draw, a refit)
    need only the solve's outputs. ``closing_loss``, when supplied, must agree
    with the recomputed loss to a relative 1e-9; a mismatch means the frame,
    the targets or the weights are not the ones the outputs were cut from.

    Grouped solves cannot use this seam: their diagnostics do not contain the
    original aligned constraints needed to recheck group caps and fixed zeros.
    """
    if "grouped_upper_bounds" in options or "grouped_preserve_zeros" in options:
        raise ValueError(
            "cannot rebuild a result with grouped upper bounds or fixed zeros "
            "without the original aligned constraints"
        )
    problem = build_constraint_matrix(frame, targets, weight_entity)
    if problem.skipped:
        names = ", ".join(skipped.target.name for skipped in problem.skipped[:5])
        raise ValueError(
            f"cannot rebuild a result with {len(problem.skipped)} uncompilable "
            f"target(s): {names}."
        )
    initial = problem.initial_weights
    w0 = np.asarray(initial.values, dtype=np.float64)
    final_weights = np.asarray(weights, dtype=np.float64)
    if final_weights.shape != w0.shape:
        raise ValueError(
            f"weights shape {final_weights.shape} must match the {w0.shape} "
            f"{weight_entity!r} weight vector."
        )
    if not np.isfinite(final_weights).all() or (final_weights < 0.0).any():
        raise ValueError("weights must be finite and non-negative.")
    trajectory = np.asarray(loss_trajectory, dtype=np.float64)
    if trajectory.ndim != 1 or trajectory.size == 0:
        raise ValueError("loss_trajectory must be a non-empty vector.")
    loss_weights = _validate_target_loss_weights(
        np.asarray(target_loss_weights, dtype=np.float64), problem.target_vector.shape
    )
    loss_scales = _validate_target_loss_scales(
        np.asarray(target_loss_scales, dtype=np.float64),
        problem.target_vector.shape,
        targets=problem.target_vector,
    )
    loss_cap = _validate_target_loss_cap(target_loss_cap)
    if gate_open_probabilities is not None:
        gate_open_probabilities = np.asarray(gate_open_probabilities, dtype=np.float64)
        if gate_open_probabilities.shape != w0.shape:
            raise ValueError("gate_open_probabilities must align with the weights.")
    mass = str(options.get("mass", FREE_MASS))
    if mass not in (CONSERVE_MASS, FREE_MASS):
        raise ValueError(f"options['mass'] must be conserve or free, got {mass!r}.")
    mass_reason = options.get("mass_reason")
    calibrated = initial.with_values(final_weights, kind=WeightKind.CALIBRATED)
    new_frame = _apply_weights(
        frame,
        weight_entity,
        initial,
        calibrated,
        mass,
        targets,
        mass_reason=None if mass_reason is None else str(mass_reason),
    )
    diagnostics = _build_diagnostics(problem, frame, w0, final_weights)
    recomputed = relative_error_loss(
        problem.estimates(final_weights),
        problem.target_vector,
        target_loss_weights=loss_weights,
        target_loss_scales=loss_scales,
        target_loss_cap=loss_cap,
    )
    if closing_loss is not None and not math.isclose(
        float(closing_loss), float(recomputed), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"closing loss {closing_loss!r} disagrees with the loss recomputed on "
            f"the compiled system ({recomputed!r}); the frame, targets or weights "
            "are not the ones these outputs were cut from."
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
        l0_lambda=float(l0_lambda),
        n_nonzero=int(n_nonzero),
        closing_loss=float(recomputed),
        target_loss_weights=loss_weights,
        target_loss_scales=loss_scales,
        target_loss_cap=loss_cap,
        options=dict(options),
        gate_open_probabilities=gate_open_probabilities,
    )


def _apply_weights(
    frame: Frame,
    weight_entity: str,
    initial: Weights,
    calibrated: Weights,
    mass: str,
    targets: TargetSet,
    *,
    mass_reason: str | None = None,
) -> Frame:
    """Place the calibrated weights on the frame with the right mass policy.

    ``mass="conserve"`` uses the kernel's ``CONSERVE_MASS`` (the total is held
    to the input within rtol, so the kernel's conservation check passes).
    ``mass="free"`` declares a :class:`~microcosm.frame.MassChange`: the total
    moved on purpose to fit the targets, and the change is recorded on the
    frame's mass log with the caller's ``mass_reason`` when supplied, else a
    generic reason naming the calibration.
    """
    if mass == CONSERVE_MASS:
        from microcosm.frame import CONSERVE_MASS as FRAME_CONSERVE

        return frame.with_weights(weight_entity, calibrated, mass=FRAME_CONSERVE)
    factor = calibrated.total / initial.total if initial.total != 0 else None
    reason = mass_reason or (
        f"calibrated {weight_entity!r} weights to {len(targets)} target(s) "
        "(capped weighted-MAPE loss); total mass free to move"
    )
    return frame.with_weights(
        weight_entity,
        calibrated,
        mass=MassChange(factor=factor, reason=reason),
    )


def _phase_callback(
    progress_callback: Callable[[dict[str, object]], None] | None,
    phase: str,
) -> Callable[[dict[str, object]], None] | None:
    if progress_callback is None:
        return None

    def callback(event: dict[str, object]) -> None:
        progress_callback({"phase": phase, **event})

    return callback


def _phase_snapshots(
    target_snapshots: TargetSnapshotObserver | None,
    phase: str,
) -> TargetSnapshotObserver | None:
    """The ``_phase_callback`` analogue for the snapshot observer."""
    if target_snapshots is None:
        return None
    return target_snapshots.with_phase(phase)


def _selected_person_mask(
    frame: Frame,
    weight_entity: str,
    selected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a selected weight-entity row mask to person rows and entity ids."""
    schema = frame.schema
    entity_table = frame.table(weight_entity)
    entity_ids = entity_table[schema.entity_id_column(weight_entity)].to_numpy()
    if selected_mask.shape != (len(entity_ids),):
        raise ValueError(
            "selected_mask must align with the calibrated weight entity: "
            f"got {selected_mask.shape}, expected {(len(entity_ids),)}."
        )
    selected_ids = entity_ids[selected_mask]
    if weight_entity == schema.person_entity:
        return selected_mask.copy(), selected_ids.copy()
    if weight_entity not in schema.group_entities:
        raise ValueError(
            f"Post-L0 refit can only map the person entity or a group entity; "
            f"got {weight_entity!r}."
        )
    membership = frame.person[schema.membership_column(weight_entity)].to_numpy()
    person_mask = np.isin(membership, selected_ids)
    return person_mask, selected_ids.copy()


def _with_exact_k_full_pool_weights(
    subset: Frame,
    pool: Frame,
    weight_entity: str,
    support_inclusion_probabilities: np.ndarray,
) -> Frame:
    """Set the selected support's normalized Horvitz--Thompson baseline.

    For selection indicator ``I_i`` with marginal inclusion probability
    ``q_i``, ``E[I_i * w_i / q_i] = w_i``. Thus the sum of ``I_i * w_i / q_i``
    is an unbiased estimator of full-pool mass. The pool total is known here,
    so the realized Horvitz--Thompson weights are subsequently projected to
    that exact total while preserving their relative ``w_i / q_i`` allocation.
    """
    selected = subset.resolve_weights(weight_entity)
    pool_total = pool.resolve_weights(weight_entity).total
    inverse_probability_values = selected.values / support_inclusion_probabilities
    expanded_values = _project_to_total(
        inverse_probability_values,
        pool_total,
        max_weight_ratio=None,
        initial_weights=inverse_probability_values,
    )
    expanded = selected.with_values(expanded_values, kind=selected.kind)
    return subset.with_weights(
        weight_entity,
        expanded,
        mass=MassChange(
            factor=pool_total / selected.total,
            reason=(
                "exact-k selected-support Horvitz-Thompson weights normalized "
                "to the known full-pool mass"
            ),
        ),
    )


def refit_l0_selection(
    frame: Frame,
    targets: TargetSet,
    selection: CalibrationResult,
    *,
    weight_entity: str | None = None,
    support: np.ndarray | None = None,
    k: int | None = None,
    support_inclusion_probabilities: np.ndarray | None = None,
    epochs: int = 256,
    learning_rate: float = 0.02,
    mass: str = FREE_MASS,
    mass_reason: str | None = None,
    max_weight_ratio: float | None = None,
    l2_lambda: float = 0.0,
    l2_anchor: str = "initial",
    init_mean: float = 0.999,
    temperature: float = 0.25,
    budget_iters: int = _DEFAULT_BUDGET_ITERS,
    seed: int = 0,
    target_loss_weights: np.ndarray | None = None,
    target_loss_scales: np.ndarray | None = None,
    target_loss_cap: float = _DEFAULT_TARGET_LOSS_CAP,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    target_snapshots: TargetSnapshotObserver | None = None,
) -> L0RefitResult:
    """Refit ordinary calibration on a frozen support from an L0 solve.

    Use this when the caller has already run :func:`calibrate` with L0 pruning
    and wants to reuse a frozen selection. By default the refit preserves the
    existing path: it subsets to records whose L0 weights survived. A caller may
    instead pass the stage-2 exact-k ``support`` index array together with ``k``;
    :func:`~microcosm.calibrate.exact_k.assert_exact_k_support` then applies the
    named, fail-closed cardinality gate before any refit work. In either case the
    gates and L0 penalty are removed and :func:`calibrate` runs again with
    ``l0_lambda=0`` on only the selected records.

    ``mass_reason`` carries a caller's calibration provenance into the refit.

    The explicit exact-k path subsets the original ``frame`` rather than
    ``selection.frame``. A record may have positive open probability while its
    deterministic L0 gate is closed; starting from the original frame keeps
    such a valid stage-2 draw available to ordinary log-weight calibration.
    Its selected design weights become ``w_i / q_i`` using the required,
    support-aligned ``support_inclusion_probabilities``, then are normalized to
    the original frame's known full-pool mass as described in this module's
    estimator note. Omitting ``support``, ``k``, and the inclusion probabilities
    explicitly selects the existing thresholded path and leaves its starting
    weights unchanged.
    ``l2_lambda`` applies the soft concentration penalty to the refit itself —
    the stage that produces the shipped weights; the default ``0.0`` keeps the
    refit unpenalized.

    On the legacy path, the refit's starting weights are the *selection stage's
    calibrated weights*, which are already concentrated — so with
    ``l2_anchor="initial"`` a strong penalty pulls toward their square (more
    concentration, lower ESS). Pass ``l2_anchor="uniform"`` when the penalty's
    job is to spread the shipped weights, or ``l2_anchor="design"`` to anchor at
    the *pre-selection* initial weights of the surviving records — "stay near
    the survey design", the natural choice when the candidate frame carries
    real design weights rather than a uniform reset. On the explicit exact-k
    path, the starting weights and design anchor instead use the original-frame
    design weights after the same normalized Horvitz--Thompson projection.
    """
    if l2_anchor not in ("initial", "uniform", "design"):
        raise ValueError(
            "refit l2_anchor must be 'initial', 'uniform', or 'design', got "
            f"{l2_anchor!r}."
        )
    refit_entity = selection.weight_entity if weight_entity is None else weight_entity
    if selection.weight_entity != refit_entity:
        raise ValueError(
            "weight_entity must match the L0 selection weight entity: "
            f"{refit_entity!r} != {selection.weight_entity!r}."
        )
    if selection.l0_lambda <= 0.0:
        raise ValueError(
            "refit_l0_selection requires a selection result produced with "
            "positive L0 pruning."
        )

    if (support is None) != (k is None):
        raise ValueError(
            "support and k must be provided together for an exact-k frozen-support "
            "refit."
        )
    if support is None:
        if support_inclusion_probabilities is not None:
            raise ValueError(
                "support_inclusion_probabilities may only be provided with "
                "support and k for an exact-k frozen-support refit."
            )
        selected_mask = selection.weights > (
            _PRUNE_REL_ATOL * float(np.mean(selection.initial_weights))
        )
        subset_source = selection.frame
        selected_inclusion_probabilities = None
    else:
        selected_indices = assert_exact_k_support(
            support,
            k,
            pool_size=len(selection.weights),
        )
        if support_inclusion_probabilities is None:
            raise ValueError(
                "support_inclusion_probabilities are required for an exact-k "
                "frozen-support refit."
            )
        try:
            supplied_probabilities = np.asarray(support_inclusion_probabilities)
        except (TypeError, ValueError):
            raise ValueError(
                "support_inclusion_probabilities must be a one-dimensional "
                "numeric vector aligned with support."
            ) from None
        if supplied_probabilities.dtype.kind == "c":
            raise ValueError(
                "support_inclusion_probabilities must be a one-dimensional "
                "numeric vector aligned with support."
            )
        try:
            supplied_probabilities = np.asarray(
                supplied_probabilities,
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            raise ValueError(
                "support_inclusion_probabilities must be a one-dimensional "
                "numeric vector aligned with support."
            ) from None
        if supplied_probabilities.shape != selected_indices.shape:
            raise ValueError(
                "support_inclusion_probabilities must be one-dimensional and "
                "aligned with support: "
                f"got shape {supplied_probabilities.shape}, expected "
                f"{selected_indices.shape}."
            )
        if not np.isfinite(supplied_probabilities).all():
            raise ValueError("support_inclusion_probabilities must be finite.")
        if ((supplied_probabilities <= 0.0) | (supplied_probabilities > 1.0)).any():
            raise ValueError("support_inclusion_probabilities must lie in (0, 1].")
        support_order = np.argsort(np.asarray(support), kind="stable")
        selected_inclusion_probabilities = supplied_probabilities[support_order]
        selected_mask = np.zeros(len(selection.weights), dtype=bool)
        selected_mask[selected_indices] = True
        subset_source = frame
    if not selected_mask.any():
        raise ValueError("Cannot refit an L0 selection with no retained records.")
    person_mask, selected_ids = _selected_person_mask(
        frame, refit_entity, selected_mask
    )
    subset = subset_source.select(person_mask)
    if k is not None:
        realized_count = subset.n(refit_entity)
        assert_exact_k_support(
            np.arange(realized_count, dtype=np.int64),
            k,
            pool_size=realized_count,
        )
        if selected_inclusion_probabilities is None:  # pragma: no cover
            raise RuntimeError("exact-k refit lost its inclusion probabilities.")
        subset = _with_exact_k_full_pool_weights(
            subset,
            frame,
            refit_entity,
            selected_inclusion_probabilities,
        )

    refit_l2_anchor_weights = None
    if l2_anchor == "design":
        # The legacy branch uses the pre-selection design prior on survivors.
        # The exact-k branch uses that same prior after its full-pool ratio
        # normalization, keeping the anchor on the refit's expanded mass scale.
        if k is None:
            refit_l2_anchor_weights = np.asarray(
                selection.initial_weights, dtype=np.float64
            )[selected_mask]
        else:
            refit_l2_anchor_weights = subset.resolve_weights(refit_entity).values

    refit = calibrate(
        subset,
        targets,
        weight_entity=refit_entity,
        method="adam",
        epochs=epochs,
        learning_rate=learning_rate,
        mass=mass,
        mass_reason=mass_reason,
        max_weight_ratio=max_weight_ratio,
        target_records=None,
        l0_lambda=0.0,
        l2_lambda=l2_lambda,
        l2_anchor=l2_anchor,
        l2_anchor_weights=refit_l2_anchor_weights,
        init_mean=init_mean,
        temperature=temperature,
        budget_iters=budget_iters,
        seed=seed,
        target_loss_weights=target_loss_weights,
        target_loss_scales=target_loss_scales,
        target_loss_cap=target_loss_cap,
        progress_callback=progress_callback,
        target_snapshots=target_snapshots,
    )
    return L0RefitResult(
        selection=selection,
        refit=refit,
        selected_entity_ids=selected_ids,
        selected_mask=selected_mask.copy(),
    )


def calibrate_l0_refit(
    frame: Frame,
    targets: TargetSet,
    *,
    weight_entity: str = "household",
    epochs: int = 256,
    refit_epochs: int | None = None,
    learning_rate: float = 0.02,
    refit_learning_rate: float | None = None,
    mass: str = FREE_MASS,
    max_weight_ratio: float | None = None,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    refit_l2_lambda: float | None = None,
    l2_anchor: str = "initial",
    refit_l2_anchor: str | None = None,
    init_mean: float = 0.999,
    temperature: float = 0.25,
    budget_iters: int = _DEFAULT_BUDGET_ITERS,
    seed: int = 0,
    refit_seed: int | None = None,
    target_loss_weights: np.ndarray | None = None,
    target_loss_scales: np.ndarray | None = None,
    target_loss_cap: float = _DEFAULT_TARGET_LOSS_CAP,
    warm_start_weights: np.ndarray | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    target_snapshots: TargetSnapshotObserver | None = None,
) -> L0RefitResult:
    """Select a sparse support with L0 gates, then refit ordinary calibration.

    This is the production "generate big, then prune" path when the final
    artifact should be a sparse calibrated frame rather than the raw gated L0
    weights. The first stage is exactly :func:`calibrate` with L0 pruning
    enabled by ``target_records`` or a positive fixed ``l0_lambda``. The second
    stage delegates to :func:`refit_l0_selection`.

    ``l2_lambda`` applies the soft concentration penalty to both stages.
    ``refit_l2_lambda`` overrides the refit stage only (the ``refit_epochs`` /
    ``refit_learning_rate`` pattern), so selection-only
    (``refit_l2_lambda=0.0``) and refit-only (``l2_lambda=0.0`` with a positive
    override) penalties are both expressible. ``l2_anchor`` /
    ``refit_l2_anchor`` follow the same inherit-or-override pattern; note the
    refit starts from the selection stage's concentrated weights, so a
    penalized refit whose goal is spreading the shipped weights should anchor
    ``"uniform"`` (see :func:`refit_l0_selection`).
    """
    if target_records is None and not (math.isfinite(l0_lambda) and l0_lambda > 0.0):
        raise ValueError(
            "calibrate_l0_refit requires L0 pruning: pass target_records or a "
            "positive fixed l0_lambda."
        )
    if not math.isfinite(l0_lambda) or l0_lambda < 0.0:
        raise ValueError(
            f"l0_lambda must be finite and non-negative, got {l0_lambda!r}."
        )
    if refit_l2_lambda is not None and (
        not math.isfinite(refit_l2_lambda) or refit_l2_lambda < 0.0
    ):
        # Fail before the expensive selection stage, not at the refit call.
        raise ValueError(
            f"refit_l2_lambda must be finite and non-negative, got {refit_l2_lambda!r}."
        )
    if l2_anchor not in ("initial", "uniform", "design"):
        raise ValueError(
            f"l2_anchor must be 'initial', 'uniform', or 'design', got {l2_anchor!r}."
        )
    if refit_l2_anchor is not None and refit_l2_anchor not in (
        "initial",
        "uniform",
        "design",
    ):
        # Same early check: an invalid refit anchor must not cost a selection run.
        raise ValueError(
            "refit_l2_anchor must be 'initial', 'uniform', 'design', or None, "
            f"got {refit_l2_anchor!r}."
        )
    selection = calibrate(
        frame,
        targets,
        weight_entity=weight_entity,
        method="adam",
        epochs=epochs,
        learning_rate=learning_rate,
        mass=mass,
        max_weight_ratio=max_weight_ratio,
        target_records=target_records,
        l0_lambda=l0_lambda,
        l2_lambda=l2_lambda,
        # At a fresh selection the frame's initial weights ARE the design
        # prior, so the "design" anchor is exactly the "initial" anchor.
        l2_anchor="initial" if l2_anchor == "design" else l2_anchor,
        init_mean=init_mean,
        temperature=temperature,
        budget_iters=budget_iters,
        seed=seed,
        target_loss_weights=target_loss_weights,
        target_loss_scales=target_loss_scales,
        target_loss_cap=target_loss_cap,
        warm_start_weights=warm_start_weights,
        progress_callback=_phase_callback(progress_callback, "l0_selection"),
        target_snapshots=_phase_snapshots(target_snapshots, "l0_selection"),
    )
    return refit_l0_selection(
        frame,
        targets,
        selection,
        weight_entity=weight_entity,
        epochs=epochs if refit_epochs is None else refit_epochs,
        learning_rate=(
            learning_rate if refit_learning_rate is None else refit_learning_rate
        ),
        mass=mass,
        max_weight_ratio=max_weight_ratio,
        l2_lambda=l2_lambda if refit_l2_lambda is None else refit_l2_lambda,
        l2_anchor=l2_anchor if refit_l2_anchor is None else refit_l2_anchor,
        init_mean=init_mean,
        temperature=temperature,
        budget_iters=budget_iters,
        seed=seed if refit_seed is None else refit_seed,
        target_loss_weights=target_loss_weights,
        target_loss_scales=target_loss_scales,
        target_loss_cap=target_loss_cap,
        progress_callback=_phase_callback(progress_callback, "post_l0_refit"),
        target_snapshots=_phase_snapshots(target_snapshots, "post_l0_refit"),
    )
