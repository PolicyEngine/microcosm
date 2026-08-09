"""Read-only scoring of targets against an existing frame and weight vector."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.solve import (
    CalibrationResult,
    TargetDiagnostic,
    default_target_loss_scales,
    relative_error_loss,
)
from microcosm.calibrate.target import TargetSet
from microcosm.frame import Frame


def _target_diagnostics(problem, weights: np.ndarray) -> tuple[TargetDiagnostic, ...]:
    estimates = problem.estimates(weights)
    diagnostics: list[TargetDiagnostic] = []
    for index, target in enumerate(problem.targets):
        target_value = float(problem.target_vector[index])
        estimate = float(estimates[index])
        relative_error = (
            (estimate - target_value) / target_value
            if target_value != 0.0
            else estimate - target_value
        )
        within_tolerance = (
            None
            if target.tolerance is None
            else abs(estimate - target_value) <= target.tolerance
        )
        diagnostics.append(
            TargetDiagnostic(
                name=problem.names[index],
                target=target_value,
                initial_estimate=estimate,
                final_estimate=estimate,
                relative_error=float(relative_error),
                within_tolerance=within_tolerance,
            )
        )
    return tuple(diagnostics)


def score_targets(
    frame: Frame,
    targets: TargetSet,
    *,
    weight_entity: str = "household",
    weights: np.ndarray | None = None,
    target_loss_weights: np.ndarray | None = None,
    target_loss_scales: np.ndarray | None = None,
    target_loss_cap: float = 10.0,
    options: Mapping[str, object] | None = None,
) -> CalibrationResult:
    """Evaluate a target surface under an existing dataset's weights.

    This is the no-optimizer counterpart to :func:`calibrate`: it compiles the
    same sparse target matrix, evaluates ``A @ weights``, and returns a
    :class:`CalibrationResult` so diagnostics and dashboards can consume scores
    and calibrations through the same schema.
    """
    problem = build_constraint_matrix(frame, targets, weight_entity)
    initial_weights = problem.initial_weights.values.astype(np.float64, copy=True)
    score_weights = (
        initial_weights.copy()
        if weights is None
        else np.asarray(weights, dtype=np.float64).copy()
    )
    if score_weights.shape != initial_weights.shape:
        raise ValueError(
            "weights must align with the compiled weight entity, got shapes "
            f"{score_weights.shape} vs {initial_weights.shape}."
        )

    estimates = problem.estimates(score_weights)
    scales = (
        default_target_loss_scales(problem.target_vector)
        if target_loss_scales is None
        else np.asarray(target_loss_scales, dtype=np.float64)
    )
    loss = relative_error_loss(
        estimates,
        problem.target_vector,
        target_loss_weights=target_loss_weights,
        target_loss_scales=scales,
        target_loss_cap=target_loss_cap,
    )
    prune_atol = 1e-6 * float(np.mean(initial_weights))
    return CalibrationResult(
        frame=frame,
        weight_entity=problem.weight_entity,
        weights=score_weights,
        initial_weights=initial_weights,
        diagnostics=_target_diagnostics(problem, score_weights),
        loss_trajectory=np.asarray([loss], dtype=np.float64),
        skipped=problem.skipped,
        problem=problem,
        l0_lambda=0.0,
        n_nonzero=int((score_weights > prune_atol).sum()),
        closing_loss=loss,
        options={
            "method": "score_only",
            "target_loss_cap": float(target_loss_cap),
            "target_loss_weights": "provided"
            if target_loss_weights is not None
            else "uniform",
            **dict(options or {}),
        },
        gate_open_probabilities=None,
    )
