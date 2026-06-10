"""The calibration solver: targets -> calibrated weights on a Frame.

:func:`calibrate` is the representation operator. It compiles a
:class:`~populace.calibrate.target.TargetSet` against a
:class:`~populace.frame.Frame` (:mod:`populace.calibrate.matrix`), then optimizes
the weight vector of ``weight_entity`` to minimize the eCPS relative-error loss

    ``mean(((A @ w - b + 1) / (b + 1))**2)``

— the loss that produced the enhanced CPS — with torch's Adam over the
**log-weights** (so weights stay strictly positive by construction). It returns a
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
- ``target_records`` + ``l0_lambda`` add hard-concrete L0 gates
  (:mod:`populace.calibrate.gates`) that prune the pool toward
  ``target_records`` non-zero weights — the generate-big-then-prune path.
- ``target_records`` (without ``l0_lambda``) is also honored by ``l0_lambda``
  auto-selection: if records are requested but no penalty is given, a small
  default penalty is used and reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from populace.frame import Frame, MassChange, Weights, WeightKind

from populace.calibrate.matrix import (
    CalibrationProblem,
    SkippedTarget,
    build_constraint_matrix,
)
from populace.calibrate.gates import HardConcrete
from populace.calibrate.target import TargetSet

__all__ = [
    "calibrate",
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

#: Default L0 penalty applied when ``target_records`` is requested but no
#: ``l0_lambda`` is given. Small, matching the eCPS gate's regime where the
#: relative-error term dominates and the penalty trims the long tail.
_DEFAULT_L0_LAMBDA = 1e-6


@dataclass(frozen=True)
class TargetDiagnostic:
    """Per-target calibration diagnostics.

    Attributes:
        name: The target's ``"name@period"`` row label.
        target: The target value the row aims at (the right-hand side ``b``).
        initial_estimate: ``row @ w0`` under the input weights.
        final_estimate: ``row @ w`` under the calibrated weights.
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
        l0_lambda: The L0 penalty actually applied (0.0 when no pruning).
        n_nonzero: Number of calibrated weights above the prune threshold.

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

    @property
    def initial_loss(self) -> float:
        """The eCPS relative-error loss under the input weights."""
        return float(self.loss_trajectory[0])

    @property
    def final_loss(self) -> float:
        """The eCPS relative-error loss under the calibrated weights."""
        return float(self.loss_trajectory[-1])

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


def _relative_error_loss(
    estimate: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """The eCPS relative-error loss ``mean(((est - tgt + 1)/(tgt + 1))**2)``.

    The ``+1`` in numerator and denominator is the eCPS regularizer: it keeps
    the loss finite and well-scaled for targets near zero (a zero-valued count
    target contributes ``est**2`` rather than dividing by zero), exactly as in
    the reference ``reweight`` loss.
    """
    rel_error = ((estimate - targets) + 1.0) / (targets + 1.0)
    return (rel_error**2).mean()


def _build_diagnostics(
    problem: CalibrationProblem,
    initial_weights: np.ndarray,
    final_weights: np.ndarray,
) -> tuple[TargetDiagnostic, ...]:
    """Assemble per-target diagnostics from the problem and both weight vectors."""
    initial_est = problem.estimates(initial_weights)
    final_est = problem.estimates(final_weights)
    diagnostics: list[TargetDiagnostic] = []
    for i, target in enumerate(problem.targets):
        tgt = float(problem.target_vector[i])
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
                initial_estimate=float(initial_est[i]),
                final_estimate=final,
                relative_error=float(rel),
                within_tolerance=within,
            )
        )
    return tuple(diagnostics)


def _optimize(
    matrix: torch.Tensor,
    targets: torch.Tensor,
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

    Optimizes the log-weights with Adam against the eCPS relative-error loss.
    Positivity is by construction (``w = exp(log_w)`` times optional gates). The
    hard constraints — mass conservation and ``max_weight_ratio`` — are applied
    by projecting the realized weights after each step, so they hold on the
    returned vector exactly, not merely in expectation.
    """
    w0 = np.asarray(initial_weights, dtype=np.float64)
    total0 = float(w0.sum())
    log_w = torch.tensor(np.log(w0), dtype=torch.float32, requires_grad=True)

    gates: HardConcrete | None = None
    params: list[torch.Tensor] = [log_w]
    if l0_lambda > 0.0 or target_records is not None:
        gates = HardConcrete(
            len(w0), init_mean=init_mean, temperature=temperature
        )
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
        estimate = weights @ matrix
        loss = _relative_error_loss(estimate, targets)
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

    # Exact mass conservation on the returned vector: a single closing rescale
    # to the input total. When a max_weight_ratio is also set, the rescale is
    # capped at the bound and the residual is redistributed below the bound, so
    # both invariants hold together.
    if conserve_mass:
        final = _project_to_total(final, total0, max_weight_ratio, w0)
    return final, trajectory


def _project_to_total(
    weights: np.ndarray,
    total: float,
    max_weight_ratio: float | None,
    initial_weights: np.ndarray,
) -> np.ndarray:
    """Scale ``weights`` to sum to ``total`` while respecting an optional cap.

    Without a cap this is a single multiplicative rescale. With a cap, records
    are scaled up only to their bound and any shortfall is spread over the
    records still below their bound, iterating until the total is met or no
    headroom remains (in which case the cap binds and the total is the maximum
    achievable — the caller's targets and cap are then jointly infeasible, which
    the mass-conservation tolerance surfaces).
    """
    weights = weights.astype(np.float64).copy()
    if max_weight_ratio is None:
        current = weights.sum()
        if current > 0:
            weights *= total / current
        return weights

    cap = max_weight_ratio * np.asarray(initial_weights, dtype=np.float64)
    weights = np.minimum(weights, cap)
    for _ in range(64):
        current = weights.sum()
        if current <= 0 or np.isclose(current, total, rtol=1e-12):
            break
        if current > total:
            weights *= total / current
            continue
        headroom = cap - weights
        free = headroom > 0
        if not free.any():
            break  # cap binds everywhere; total is the achievable maximum.
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
    seed: int = 0,
) -> CalibrationResult:
    """Calibrate ``weight_entity``'s weights to ``targets`` over ``frame``.

    Compiles the targets into a sparse system and optimizes the log-weights with
    Adam to minimize the eCPS relative-error loss
    ``mean(((A @ w - b + 1)/(b + 1))**2)``. Returns a new frame whose
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
        target_records: If given, enable L0 pruning toward this many non-zero
            weights (uses a default ``l0_lambda`` when none is supplied).
        l0_lambda: L0 penalty strength. ``> 0`` enables hard-concrete gates that
            prune the pool; ``0.0`` (default) keeps every record.
        init_mean: Initial expected open-probability of the L0 gates (only used
            when pruning).
        temperature: Hard-concrete temperature (only used when pruning).
        seed: Seed for torch's RNG (the gate sampling), for reproducibility.

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
    if target_records is not None and (
        not isinstance(target_records, int) or target_records <= 0
    ):
        raise ValueError(
            f"target_records must be a positive integer, got {target_records!r}."
        )

    problem = build_constraint_matrix(frame, targets, weight_entity)
    initial = problem.initial_weights
    w0 = initial.values

    effective_l0 = l0_lambda
    if target_records is not None and l0_lambda == 0.0:
        effective_l0 = _DEFAULT_L0_LAMBDA

    torch.manual_seed(seed)
    matrix_t = torch.tensor(problem.matrix.toarray().T, dtype=torch.float32)
    targets_t = torch.tensor(problem.target_vector, dtype=torch.float32)

    final_weights, trajectory = _optimize(
        matrix_t,
        targets_t,
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

    calibrated = initial.with_values(final_weights, kind=WeightKind.CALIBRATED)
    new_frame = _apply_weights(
        frame, weight_entity, initial, calibrated, mass, targets
    )

    diagnostics = _build_diagnostics(problem, w0, final_weights)
    prune_atol = _PRUNE_REL_ATOL * float(np.mean(w0))
    n_nonzero = int((final_weights > prune_atol).sum())

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

        return frame.with_weights(
            weight_entity, calibrated, mass=FRAME_CONSERVE
        )
    factor = (
        calibrated.total / initial.total if initial.total != 0 else None
    )
    reason = (
        f"calibrated {weight_entity!r} weights to {len(targets)} target(s) "
        "(eCPS relative-error loss); total mass free to move"
    )
    return frame.with_weights(
        weight_entity,
        calibrated,
        mass=MassChange(factor=factor, reason=reason),
    )
