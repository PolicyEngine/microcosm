"""Rowwise local solve surface: one weight per cloned household (#495).

The US Build-N shape for the UK dense/local arm. A rowwise household exists
in exactly one assigned area, so the constraint matrix has one column per
household (never area x household stacking) and an area's target rows draw
support only from the households assigned there. The matrix builder fails
closed when an assigned area is missing from the target surface — local
misses are support or target work, never silent exclusion — and the solve
runs under the reviewed :data:`UK_LOCAL_SOLVE_DOCTRINE` with no per-target
parameters and no doctrine injection point, mirroring
:func:`solve_uk_local_weights_under_doctrine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

from populace.build.uk_runtime.local_doctrine import (
    UK_LOCAL_SOLVE_DOCTRINE,
)
from populace.build.uk_runtime.local_solver import (
    StackedLocalSolveResult,
    solve_prepared_local_weights,
)

__all__ = [
    "UKRowwiseLocalMatrix",
    "build_uk_rowwise_local_matrix",
    "rowwise_area_support_summary",
    "solve_uk_rowwise_weights_under_doctrine",
]


@dataclass(frozen=True)
class UKRowwiseLocalMatrix:
    """Sparse rowwise calibration matrix: one column per cloned household."""

    matrix: sp.csr_matrix
    targets: np.ndarray
    target_frame: pd.DataFrame
    area_codes: tuple[str, ...]
    metric_names: tuple[str, ...]
    household_ids: tuple[Any, ...]
    assigned_areas: tuple[str, ...]

    @property
    def n_areas(self) -> int:
        return len(self.area_codes)

    @property
    def n_households(self) -> int:
        return len(self.household_ids)


def build_uk_rowwise_local_matrix(
    metrics: pd.DataFrame,
    assigned_areas: pd.Series | Sequence[str],
    targets: pd.DataFrame,
    *,
    area_codes: Sequence[str] | None = None,
    area_type: str = "constituency",
    code_column: str = "code",
) -> UKRowwiseLocalMatrix:
    """Build the rowwise local matrix from household metrics and assignments.

    Args:
        metrics: Household-grain metric columns, indexed by household id.
        assigned_areas: Each household's assigned area code. A Series must
            carry exactly the metric index; a plain sequence must match its
            length and order.
        targets: Area target frame (``code_column`` + metric columns), the
            same contract as the stacked path.
        area_codes: Canonical area order; defaults to the target frame order.
        area_type: Stored on ``target_frame`` for diagnostics.
        code_column: Target frame column holding area codes.
    """

    from populace.build.uk_runtime.local_geography import align_area_targets

    if metrics.empty:
        raise ValueError("metrics must not be empty.")
    if metrics.index.has_duplicates:
        duplicates = metrics.index[metrics.index.duplicated()].unique()
        raise ValueError(
            "metrics household index must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    values = metrics.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("metrics must be finite.")

    if isinstance(assigned_areas, pd.Series):
        if not assigned_areas.index.equals(metrics.index):
            raise ValueError(
                "assigned_areas index must align with the metrics household index."
            )
        assigned = assigned_areas.astype(str).to_numpy()
    else:
        assigned = np.asarray([str(code) for code in assigned_areas], dtype=object)
        if len(assigned) != len(metrics):
            raise ValueError(
                "assigned_areas must align with metrics rows, got "
                f"{len(assigned)} assignments for {len(metrics)} rows."
            )
    blank = np.array([not code.strip() for code in assigned.tolist()])
    if blank.any():
        raise ValueError(f"assigned_areas contains {int(blank.sum())} blank code(s).")

    if area_codes is None:
        if code_column not in targets.columns:
            raise ValueError(
                "area_codes must be supplied when targets has no "
                f"{code_column!r} column."
            )
        area_codes = targets[code_column].astype(str).tolist()
    codes = tuple(str(code) for code in area_codes)
    if len(set(codes)) != len(codes):
        raise ValueError("area_codes must be unique.")

    uncovered = sorted(set(assigned.tolist()) - set(codes))
    if uncovered:
        raise ValueError(
            "target surface does not cover assigned area(s): "
            f"{uncovered[:5]}. Every assigned area must carry targets — "
            "local misses are support or target work, never silent "
            "exclusion."
        )

    metric_names = tuple(str(column) for column in metrics.columns)
    target_values = align_area_targets(
        targets,
        codes,
        metric_names=metric_names,
        code_column=code_column,
    )

    area_index_by_code = {code: index for index, code in enumerate(codes)}
    household_area_index = np.asarray(
        [area_index_by_code[code] for code in assigned.tolist()],
        dtype=np.int64,
    )
    n_metrics = len(metric_names)
    n_households = len(metrics)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    target_rows: list[dict[str, Any]] = []
    for area_index, area_code in enumerate(codes):
        members = np.flatnonzero(household_area_index == area_index)
        for metric_index, metric_name in enumerate(metric_names):
            target_index = area_index * n_metrics + metric_index
            target_rows.append(
                {
                    "target_index": target_index,
                    "area_type": area_type,
                    "area_code": area_code,
                    "area_index": area_index,
                    "metric": metric_name,
                    "metric_index": metric_index,
                    "value": float(target_values.loc[area_code, metric_name]),
                }
            )
            if len(members) == 0:
                continue
            column_values = values[members, metric_index]
            nonzero = np.flatnonzero(column_values)
            if len(nonzero) == 0:
                continue
            rows.append(np.full(len(nonzero), target_index, dtype=np.int64))
            cols.append(members[nonzero].astype(np.int64))
            data.append(column_values[nonzero].astype(np.float64, copy=False))

    if rows:
        row_array = np.concatenate(rows)
        col_array = np.concatenate(cols)
        data_array = np.concatenate(data)
    else:
        row_array = np.array([], dtype=np.int64)
        col_array = np.array([], dtype=np.int64)
        data_array = np.array([], dtype=np.float64)
    matrix = sp.csr_matrix(
        (data_array, (row_array, col_array)),
        shape=(len(codes) * n_metrics, n_households),
        dtype=np.float64,
    )
    target_frame = pd.DataFrame(target_rows)
    return UKRowwiseLocalMatrix(
        matrix=matrix,
        targets=target_frame["value"].to_numpy(dtype=np.float64),
        target_frame=target_frame,
        area_codes=codes,
        metric_names=metric_names,
        household_ids=tuple(metrics.index.tolist()),
        assigned_areas=tuple(assigned.tolist()),
    )


def solve_uk_rowwise_weights_under_doctrine(
    problem: UKRowwiseLocalMatrix,
    base_weights: Sequence[float],
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    conserve_mass: bool = False,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    min_initial_weight: float = 1e-4,
    budget_iters: int = 10,
    seed: int = 0,
) -> StackedLocalSolveResult:
    """Solve rowwise household weights under the reviewed doctrine.

    Structurally knob-free like the stacked doctrine solve: no per-target
    parameters and no doctrine parameter — the bounds always come from
    :data:`UK_LOCAL_SOLVE_DOCTRINE`. Initial weights are the household base
    weights floored at ``min_initial_weight``; a rowwise household exists in
    exactly one area, so nothing is split.
    """

    from populace.build.uk_runtime.local_doctrine import (
        _require_uniform_target_surface,
    )

    doctrine = UK_LOCAL_SOLVE_DOCTRINE
    _require_uniform_target_surface(problem)
    weights = np.asarray(base_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("base_weights must be one-dimensional.")
    if len(weights) != problem.n_households:
        raise ValueError(
            "base_weights must align with households, got "
            f"{len(weights)} weights for {problem.n_households} households."
        )
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("base_weights must be finite and non-negative.")
    if not np.isfinite(min_initial_weight) or min_initial_weight <= 0:
        raise ValueError("min_initial_weight must be a positive finite number.")
    initial_weights = np.maximum(weights, min_initial_weight)
    return solve_prepared_local_weights(
        matrix=problem.matrix,
        targets=problem.targets,
        target_frame=problem.target_frame,
        initial_weights=initial_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=doctrine.max_weight_ratio,
        conserve_mass=conserve_mass,
        target_records=target_records,
        l0_lambda=l0_lambda,
        target_loss_cap=doctrine.target_loss_cap,
        budget_iters=budget_iters,
        seed=seed,
    )


def rowwise_area_support_summary(
    problem: UKRowwiseLocalMatrix,
    weights: Sequence[float],
    *,
    source_household_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Per-area support of a rowwise weight vector, all target areas included."""

    values = np.asarray(weights, dtype=np.float64)
    if len(values) != problem.n_households:
        raise ValueError(
            "weights must align with households, got "
            f"{len(values)} weights for {problem.n_households} households."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be finite and non-negative.")
    sources = (
        np.asarray(problem.household_ids, dtype=object)
        if source_household_ids is None
        else np.asarray(list(source_household_ids), dtype=object)
    )
    if len(sources) != problem.n_households:
        raise ValueError(
            "source_household_ids must align with households, got "
            f"{len(sources)} for {problem.n_households}."
        )
    assigned = np.asarray(problem.assigned_areas, dtype=object)
    rows: list[dict[str, Any]] = []
    for area_code in problem.area_codes:
        members = np.flatnonzero(assigned == area_code)
        member_weights = values[members]
        positive = member_weights > 0
        weight_sum = float(member_weights.sum())
        square_sum = float(np.square(member_weights).sum())
        rows.append(
            {
                "area_code": area_code,
                "assigned_households": int(len(members)),
                "nonzero_households": int(positive.sum()),
                "nonzero_source_households": int(
                    len(set(sources[members][positive].tolist()))
                ),
                "weight_sum": weight_sum,
                "max_weight": (float(member_weights.max()) if len(members) else 0.0),
                "effective_sample_size": (
                    weight_sum**2 / square_sum if square_sum > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
