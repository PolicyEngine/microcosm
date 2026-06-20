"""Solver wrappers for UK local-geography weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from populace.build.uk.local_geography import (
    StackedLocalMatrix,
    stacked_design_weights,
)
from populace.calibrate.solve import (
    _optimize as _calibrate_optimize,
)
from populace.calibrate.solve import (
    _search_l0_lambda_for_budget as _calibrate_search_l0_lambda_for_budget,
)
from populace.calibrate.solve import (
    _torch_constraint_matrix as _calibrate_torch_constraint_matrix,
)
from populace.calibrate.solve import (
    default_target_loss_scales,
    relative_error_loss,
)


@dataclass(frozen=True)
class StackedLocalSolveResult:
    """Solved stacked local weights and diagnostics."""

    weights: np.ndarray
    initial_weights: np.ndarray
    diagnostics: pd.DataFrame
    loss_trajectory: np.ndarray
    initial_loss: float
    final_loss: float
    n_nonzero: int


def solve_stacked_local_weights(
    problem: StackedLocalMatrix,
    base_weights: Sequence[float],
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    max_weight_ratio: float | None = 100.0,
    conserve_mass: bool = False,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    min_initial_weight: float = 1e-4,
    target_loss_weights: Sequence[float] | None = None,
    target_loss_scales: Sequence[float] | None = None,
    target_loss_cap: float = 10.0,
    budget_iters: int = 10,
    seed: int = 0,
) -> StackedLocalSolveResult:
    """Solve stacked local-area weights for a Populace UK local build.

    ``base_weights`` are split evenly across each area stack. The
    ``min_initial_weight`` floor is explicit because the torch log-weight
    optimizer requires strictly positive starting weights; callers that need
    exact zero-preserving design weights can use
    :func:`populace.build.uk.local_geography.stacked_design_weights` directly.
    """

    initial_weights = stacked_design_weights(
        base_weights,
        problem.n_areas,
        min_weight=min_initial_weight,
    )
    if len(initial_weights) != problem.matrix.shape[1]:
        raise ValueError(
            "base_weights expanded to the wrong stacked length: "
            f"{len(initial_weights)} vs {problem.matrix.shape[1]}."
        )
    return _solve_local_weights(
        problem,
        initial_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=max_weight_ratio,
        conserve_mass=conserve_mass,
        target_records=target_records,
        l0_lambda=l0_lambda,
        target_loss_weights=target_loss_weights,
        target_loss_scales=target_loss_scales,
        target_loss_cap=target_loss_cap,
        budget_iters=budget_iters,
        seed=seed,
    )


def solve_assigned_local_weights(
    problem: StackedLocalMatrix,
    base_weights: Sequence[float],
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    max_weight_ratio: float | None = None,
    conserve_mass: bool = False,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    min_initial_weight: float = 1e-4,
    target_loss_weights: Sequence[float] | None = None,
    target_loss_scales: Sequence[float] | None = None,
    target_loss_cap: float = 10.0,
    budget_iters: int = 10,
    seed: int = 0,
) -> StackedLocalSolveResult:
    """Solve rowwise-assigned local weights for a Populace UK local build.

    ``base_weights`` align one-to-one with the household columns in ``problem``.
    The optional ``min_initial_weight`` floor mirrors the stacked solver and is
    required by the torch log-weight optimizer. The assigned path defaults to
    no ``max_weight_ratio`` cap so zero-weight support rows, such as synthetic
    SPI rows, can be upweighted from the optimizer floor.
    """

    weights = np.asarray(base_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("base_weights must be one-dimensional.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("base_weights must be finite and non-negative.")
    if not np.isfinite(min_initial_weight) or min_initial_weight < 0:
        raise ValueError("min_initial_weight must be finite and non-negative.")
    initial_weights = np.maximum(weights, min_initial_weight)
    if len(initial_weights) != problem.matrix.shape[1]:
        raise ValueError(
            "base_weights must align with the assigned local matrix columns: "
            f"{len(initial_weights)} vs {problem.matrix.shape[1]}."
        )
    return _solve_local_weights(
        problem,
        initial_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=max_weight_ratio,
        conserve_mass=conserve_mass,
        target_records=target_records,
        l0_lambda=l0_lambda,
        target_loss_weights=target_loss_weights,
        target_loss_scales=target_loss_scales,
        target_loss_cap=target_loss_cap,
        budget_iters=budget_iters,
        seed=seed,
    )


def _solve_local_weights(
    problem: StackedLocalMatrix,
    initial_weights: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    max_weight_ratio: float | None,
    conserve_mass: bool,
    target_records: int | None,
    l0_lambda: float,
    target_loss_weights: Sequence[float] | None,
    target_loss_scales: Sequence[float] | None,
    target_loss_cap: float,
    budget_iters: int,
    seed: int,
) -> StackedLocalSolveResult:
    targets = np.asarray(problem.targets, dtype=np.float64)
    scales = (
        default_target_loss_scales(targets)
        if target_loss_scales is None
        else np.asarray(target_loss_scales, dtype=np.float64)
    )
    if scales.shape != targets.shape:
        raise ValueError(
            "target_loss_scales must align with targets, got "
            f"{scales.shape} vs {targets.shape}."
        )
    loss_weights = (
        None
        if target_loss_weights is None
        else np.asarray(target_loss_weights, dtype=np.float64)
    )
    if loss_weights is not None and loss_weights.shape != targets.shape:
        raise ValueError(
            "target_loss_weights must align with targets, got "
            f"{loss_weights.shape} vs {targets.shape}."
        )
    if (initial_weights <= 0).any():
        raise ValueError(
            "all initial weights must be strictly positive for the "
            "log-weight optimizer; use a positive min_initial_weight or remove "
            "zero-weight records before solving."
        )

    torch.manual_seed(seed)
    matrix = _calibrate_torch_constraint_matrix(problem.matrix)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    loss_weight_tensor = (
        None
        if loss_weights is None
        else torch.tensor(loss_weights, dtype=torch.float32)
    )
    scale_tensor = torch.tensor(scales, dtype=torch.float32)
    if target_records is not None:
        if not isinstance(target_records, int) or target_records <= 0:
            raise ValueError("target_records must be a positive integer.")
        if not isinstance(budget_iters, int) or budget_iters <= 0:
            raise ValueError("budget_iters must be a positive integer.")
        prune_atol = 1e-6 * float(np.mean(initial_weights))
        weights, trajectory, _realized_l0_lambda, _realized_nonzero = (
            _calibrate_search_l0_lambda_for_budget(
                matrix,
                target_tensor,
                loss_weight_tensor,
                scale_tensor,
                target_loss_cap,
                initial_weights,
                target_records=target_records,
                epochs=epochs,
                learning_rate=learning_rate,
                conserve_mass=conserve_mass,
                max_weight_ratio=max_weight_ratio,
                init_mean=0.999,
                temperature=0.25,
                seed=seed,
                prune_atol=prune_atol,
                initial_lambda=l0_lambda if l0_lambda > 0 else None,
                budget_iters=budget_iters,
            )
        )
    else:
        weights, trajectory = _calibrate_optimize(
            matrix,
            target_tensor,
            loss_weight_tensor,
            scale_tensor,
            target_loss_cap,
            initial_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            conserve_mass=conserve_mass,
            max_weight_ratio=max_weight_ratio,
            l0_lambda=l0_lambda,
            target_records=None,
            init_mean=0.999,
            temperature=0.25,
        )
    initial_estimates = problem.matrix @ initial_weights
    final_estimates = problem.matrix @ weights
    initial_loss = relative_error_loss(
        initial_estimates,
        targets,
        target_loss_weights=loss_weights,
        target_loss_scales=scales,
        target_loss_cap=target_loss_cap,
    )
    final_loss = relative_error_loss(
        final_estimates,
        targets,
        target_loss_weights=loss_weights,
        target_loss_scales=scales,
        target_loss_cap=target_loss_cap,
    )
    diagnostics = problem.target_frame.copy()
    diagnostics["target"] = targets
    diagnostics["initial_estimate"] = initial_estimates
    diagnostics["final_estimate"] = final_estimates
    diagnostics["relative_error"] = np.divide(
        final_estimates - targets,
        scales,
        out=np.zeros_like(targets, dtype=np.float64),
        where=scales != 0,
    )
    diagnostics["abs_relative_error"] = np.abs(diagnostics["relative_error"])
    prune_atol = 1e-6 * float(np.mean(initial_weights))
    return StackedLocalSolveResult(
        weights=weights,
        initial_weights=initial_weights,
        diagnostics=diagnostics,
        loss_trajectory=trajectory,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        n_nonzero=int((weights > prune_atol).sum()),
    )
