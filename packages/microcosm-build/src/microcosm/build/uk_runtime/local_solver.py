"""Solver wrapper for UK stacked local-geography weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp

from microcosm.build.uk_runtime.local_geography import (
    StackedLocalMatrix,
    stacked_design_weights,
)
from microcosm.calibrate.solve import (
    _optimize as _calibrate_optimize,
)
from microcosm.calibrate.solve import (
    _search_l0_lambda_for_budget as _calibrate_search_l0_lambda_for_budget,
)
from microcosm.calibrate.solve import (
    _torch_constraint_matrix as _calibrate_torch_constraint_matrix,
)
from microcosm.calibrate.solve import (
    default_target_loss_scales,
    relative_error_loss,
)


@dataclass(frozen=True)
class StackedLocalSolveResult:
    """Solved stacked local weights and diagnostics.

    ``past_cap_census`` is the microcosm#492 observability diagnostic: which
    target rows sat past the loss cap at initialization and at the final
    weights, which escaped back inside, and — the silent-triage class —
    which were pushed out during the solve, listed row by row.
    """

    weights: np.ndarray
    initial_weights: np.ndarray
    diagnostics: pd.DataFrame
    loss_trajectory: np.ndarray
    initial_loss: float
    final_loss: float
    n_nonzero: int
    past_cap_census: Mapping[str, Any] | None = None


def past_cap_census(
    initial_estimates: np.ndarray,
    final_estimates: np.ndarray,
    targets: np.ndarray,
    *,
    target_loss_cap: float,
    target_loss_scales: np.ndarray | None = None,
    target_frame: pd.DataFrame | None = None,
    max_listed_rows: int | None = None,
) -> dict[str, Any]:
    """Census of target rows relative to the loss cap (microcosm#492).

    Past the cap a row's gradient is zero, so the solve can silently write
    rows off. The census counts rows past the cap at initialization and at
    the final estimates, the rows that escaped back inside, the rows frozen
    past the cap throughout, and — the dumping-ground class — the rows
    pushed out during the solve, listing every one with its before/after
    scaled absolute errors (labelled from ``target_frame`` when supplied;
    pass ``max_listed_rows`` to bound the list, flagged as truncated).
    """

    initial = np.asarray(initial_estimates, dtype=np.float64)
    final = np.asarray(final_estimates, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if not (initial.shape == final.shape == target_values.shape):
        raise ValueError(
            "initial_estimates, final_estimates, and targets must align, got "
            f"shapes {initial.shape}, {final.shape}, {target_values.shape}."
        )
    if not np.isfinite(target_loss_cap) or target_loss_cap <= 0:
        raise ValueError("target_loss_cap must be a positive finite number.")
    # Mirror the canonical loss's refusals: a NaN estimate or a degenerate
    # scale is a harness bug, never a row silently classified inside the cap.
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise ValueError("census estimates must be finite.")
    if not np.isfinite(target_values).all():
        raise ValueError("census targets must be finite.")
    scales = (
        default_target_loss_scales(target_values)
        if target_loss_scales is None
        else np.asarray(target_loss_scales, dtype=np.float64)
    )
    if scales.shape != target_values.shape:
        raise ValueError(
            "target_loss_scales must align with targets, got "
            f"{scales.shape} vs {target_values.shape}."
        )
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("target_loss_scales must be finite and positive.")
    initial_errors = np.abs((initial - target_values) / scales)
    final_errors = np.abs((final - target_values) / scales)
    past_init = initial_errors > target_loss_cap
    past_final = final_errors > target_loss_cap
    pushed_out = ~past_init & past_final
    pushed_indices = np.flatnonzero(pushed_out)

    def _row(index: int) -> dict[str, Any]:
        row: dict[str, Any] = {"target_index": int(index)}
        if target_frame is not None and index < len(target_frame):
            frame_row = target_frame.iloc[index]
            for column in ("area_type", "area_code", "metric"):
                if column in target_frame.columns:
                    row[column] = str(frame_row[column])
        row["target"] = float(target_values[index])
        row["initial_abs_relative_error"] = float(initial_errors[index])
        row["final_abs_relative_error"] = float(final_errors[index])
        return row

    return {
        "target_loss_cap": float(target_loss_cap),
        "n_targets": int(len(target_values)),
        "past_at_init": int(past_init.sum()),
        "past_at_final": int(past_final.sum()),
        "escaped": int((past_init & ~past_final).sum()),
        "frozen": int((past_init & past_final).sum()),
        "pushed_out": int(pushed_out.sum()),
        "pushed_out_rows": [
            _row(index)
            for index in (
                pushed_indices
                if max_listed_rows is None
                else pushed_indices[:max_listed_rows]
            ).tolist()
        ],
        "pushed_out_rows_truncated": bool(
            max_listed_rows is not None and len(pushed_indices) > max_listed_rows
        ),
    }


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
    """Solve stacked local-area weights for a Microcosm UK local build.

    ``base_weights`` are split evenly across each area stack. The
    ``min_initial_weight`` floor is explicit because the torch log-weight
    optimizer requires strictly positive starting weights; callers that need
    exact zero-preserving design weights can use
    :func:`microcosm.build.uk_runtime.local_geography.stacked_design_weights` directly.
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
    return solve_prepared_local_weights(
        matrix=problem.matrix,
        targets=problem.targets,
        target_frame=problem.target_frame,
        initial_weights=initial_weights,
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


def solve_prepared_local_weights(
    *,
    matrix: sp.csr_matrix,
    targets: np.ndarray,
    target_frame: pd.DataFrame,
    initial_weights: np.ndarray,
    epochs: int = 512,
    learning_rate: float = 0.15,
    max_weight_ratio: float | None = 100.0,
    conserve_mass: bool = False,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    target_loss_weights: Sequence[float] | None = None,
    target_loss_scales: Sequence[float] | None = None,
    target_loss_cap: float = 10.0,
    budget_iters: int = 10,
    seed: int = 0,
) -> StackedLocalSolveResult:
    """Shared solve core: prepared initial weights against a target matrix.

    Both local shapes route here — the stacked path after splitting base
    weights across areas, and the rowwise path with the household base
    weights directly (a rowwise household exists in exactly one area, so
    there is nothing to split). The past-cap census rides on every result.
    """

    constraint_matrix = matrix
    initial_weights = np.asarray(initial_weights, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if initial_weights.ndim != 1:
        raise ValueError(
            f"initial weights must be one-dimensional, got shape "
            f"{initial_weights.shape}."
        )
    if not np.isfinite(initial_weights).all():
        raise ValueError("initial weights must be finite.")
    if targets.ndim != 1:
        raise ValueError(f"targets must be one-dimensional, got {targets.shape}.")
    if constraint_matrix.shape[0] != len(targets):
        raise ValueError(
            "constraint matrix rows must match targets, got "
            f"{constraint_matrix.shape[0]} vs {len(targets)}."
        )
    if len(target_frame) != len(targets):
        raise ValueError(
            "target_frame must align with targets, got "
            f"{len(target_frame)} rows vs {len(targets)}."
        )
    if len(targets) == 0:
        raise ValueError("the target surface must not be empty.")
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
    if len(initial_weights) != constraint_matrix.shape[1]:
        raise ValueError(
            "initial weights have the wrong length for the constraint "
            f"matrix: {len(initial_weights)} vs {constraint_matrix.shape[1]}."
        )
    if (initial_weights <= 0).any():
        raise ValueError(
            "all expanded initial weights must be strictly positive for the "
            "log-weight optimizer; use a positive min_initial_weight or remove "
            "zero-weight records before solving."
        )

    torch.manual_seed(seed)
    matrix_tensor = _calibrate_torch_constraint_matrix(constraint_matrix)
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
                matrix_tensor,
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
                l2_lambda=0.0,
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
            matrix_tensor,
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
            l2_lambda=0.0,
            target_records=None,
            init_mean=0.999,
            temperature=0.25,
        )
    initial_estimates = constraint_matrix @ initial_weights
    final_estimates = constraint_matrix @ weights
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
    diagnostics = target_frame.copy()
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
    census = past_cap_census(
        initial_estimates,
        final_estimates,
        targets,
        target_loss_cap=target_loss_cap,
        target_loss_scales=scales,
        target_frame=target_frame,
    )
    return StackedLocalSolveResult(
        weights=weights,
        initial_weights=initial_weights,
        diagnostics=diagnostics,
        loss_trajectory=trajectory,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        n_nonzero=int((weights > prune_atol).sum()),
        past_cap_census=census,
    )
