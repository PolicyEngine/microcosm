"""Per-district solver for the US SLD local layer (populace#625).

Each state legislative district is an independent reweighting problem over
exactly the households assigned to it: the districts of a chamber partition
the artifact's rows, no target row spans two districts, so the chamber-level
block-diagonal problem decomposes into per-district solves with identical
semantics. One uniform operator applied per district — never a per-district
configuration.

The solve reuses the canonical calibration internals
(:mod:`populace.calibrate.solve`): log-weight Adam on the capped
relative-error loss, target-defined scales, and the hard per-record
``max_weight_ratio`` guard. The anchor is single-stage and declared
(populace#493): starting weights are the artifact's calibrated household
weights restricted to the district, and the realized max ratio vs that
anchor is recorded per district.

``past_cap_census`` mirrors
:func:`populace.build.uk_runtime.local_solver.past_cap_census`
(populace#492/#494): rows past the loss cap have zero gradient, so every
district solve records which target rows sat past the cap, escaped, froze,
or were pushed out — and the chamber roll-up names the worst districts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from populace.calibrate.solve import (
    _optimize as _calibrate_optimize,
)
from populace.calibrate.solve import (
    _torch_constraint_matrix as _calibrate_torch_constraint_matrix,
)
from populace.calibrate.solve import (
    default_target_loss_scales,
    effective_sample_size,
    relative_error_loss,
)

__all__ = [
    "SldChamberSolveResult",
    "SldDistrictProblem",
    "SldDistrictSolveResult",
    "past_cap_census",
    "solve_sld_chamber",
    "solve_sld_district_weights",
]

#: The long-format sidecar columns for per-district weights.
SLD_LONG_WEIGHT_COLUMNS = (
    "area_type",
    "area_code",
    "household_id",
    "weight",
    "weight_source",
)


@dataclass(frozen=True)
class SldDistrictProblem:
    """One district's reweighting problem over its assigned households."""

    area_type: str
    area_code: str
    state_fips: str
    district_code: str
    matrix: np.ndarray
    targets: np.ndarray
    target_frame: pd.DataFrame
    household_ids: np.ndarray
    base_weights: np.ndarray

    def __post_init__(self) -> None:
        if self.area_type not in ("sldu", "sldl"):
            raise ValueError(
                f"area_type must be 'sldu' or 'sldl', got {self.area_type!r}."
            )
        matrix = np.asarray(self.matrix, dtype=np.float64)
        targets = np.asarray(self.targets, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-dimensional (targets x rows).")
        if matrix.shape[0] != len(targets):
            raise ValueError(
                f"matrix has {matrix.shape[0]} rows but {len(targets)} targets."
            )
        if matrix.shape[1] != len(self.household_ids):
            raise ValueError(
                f"matrix has {matrix.shape[1]} columns but "
                f"{len(self.household_ids)} household ids."
            )
        if len(self.base_weights) != len(self.household_ids):
            raise ValueError(
                "base_weights must align with household_ids, got "
                f"{len(self.base_weights)} vs {len(self.household_ids)}."
            )
        if len(self.target_frame) != len(targets):
            raise ValueError(
                "target_frame must have one row per target, got "
                f"{len(self.target_frame)} vs {len(targets)}."
            )
        if "metric" not in self.target_frame.columns:
            raise ValueError("target_frame must carry a metric column.")
        if not np.isfinite(targets).all():
            raise ValueError("targets must be finite.")
        if not np.isfinite(matrix).all():
            raise ValueError("matrix must be finite.")


@dataclass(frozen=True)
class SldDistrictSolveResult:
    """Solved weights and diagnostics for one district."""

    problem: SldDistrictProblem
    weights: np.ndarray
    initial_weights: np.ndarray
    diagnostics: pd.DataFrame
    initial_loss: float
    final_loss: float
    fraction_within_10pct: float
    realized_max_weight_ratio: float
    effective_sample_size: float
    n_floored_base_weights: int
    past_cap_census: Mapping[str, Any]


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
    """Census of target rows relative to the loss cap (populace#492).

    Same semantics as
    :func:`populace.build.uk_runtime.local_solver.past_cap_census`, carried
    here so the US layer does not import the UK runtime: counts of rows past
    the cap at initialization and at the final estimates, the rows that
    escaped, the rows frozen past the cap, and the pushed-out rows listed
    with before/after scaled errors.
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


def solve_sld_district_weights(
    problem: SldDistrictProblem,
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    max_weight_ratio: float | None = 100.0,
    target_loss_cap: float = 10.0,
    min_initial_weight: float = 1e-4,
    seed: int = 0,
) -> SldDistrictSolveResult:
    """Solve one district's weights from its artifact-weight anchor.

    Mass is never conserved: the district's own household-count and
    population targets pin the level, and conserving the anchor mass would
    fight them. Zero or negative anchor weights are floored at
    ``min_initial_weight`` (the log-weight optimizer needs strictly positive
    starts) with the floored count recorded.
    """
    base = np.asarray(problem.base_weights, dtype=np.float64)
    floored = base < min_initial_weight
    initial_weights = np.where(floored, min_initial_weight, base)
    targets = np.asarray(problem.targets, dtype=np.float64)
    scales = default_target_loss_scales(targets)

    torch.manual_seed(seed)
    matrix = _calibrate_torch_constraint_matrix(
        sp.csr_matrix(np.asarray(problem.matrix, dtype=np.float64))
    )
    weights, _trajectory = _calibrate_optimize(
        matrix,
        torch.tensor(targets, dtype=torch.float32),
        None,
        torch.tensor(scales, dtype=torch.float32),
        target_loss_cap,
        initial_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        conserve_mass=False,
        max_weight_ratio=max_weight_ratio,
        l0_lambda=0.0,
        l2_lambda=0.0,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
    )
    initial_estimates = problem.matrix @ initial_weights
    final_estimates = problem.matrix @ weights
    initial_loss = relative_error_loss(
        initial_estimates,
        targets,
        target_loss_scales=scales,
        target_loss_cap=target_loss_cap,
    )
    final_loss = relative_error_loss(
        final_estimates,
        targets,
        target_loss_scales=scales,
        target_loss_cap=target_loss_cap,
    )
    relative_errors = np.divide(
        final_estimates - targets,
        scales,
        out=np.zeros_like(targets, dtype=np.float64),
        where=scales != 0,
    )
    diagnostics = problem.target_frame.copy()
    diagnostics["target"] = targets
    diagnostics["initial_estimate"] = initial_estimates
    diagnostics["final_estimate"] = final_estimates
    diagnostics["relative_error"] = relative_errors
    diagnostics["abs_relative_error"] = np.abs(relative_errors)
    census = past_cap_census(
        initial_estimates,
        final_estimates,
        targets,
        target_loss_cap=target_loss_cap,
        target_frame=problem.target_frame,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(initial_weights > 0, weights / initial_weights, np.inf)
    return SldDistrictSolveResult(
        problem=problem,
        weights=np.asarray(weights, dtype=np.float64),
        initial_weights=initial_weights,
        diagnostics=diagnostics,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        fraction_within_10pct=float((np.abs(relative_errors) <= 0.1).mean()),
        realized_max_weight_ratio=float(np.max(ratios)),
        effective_sample_size=float(effective_sample_size(weights)),
        n_floored_base_weights=int(floored.sum()),
        past_cap_census=census,
    )


@dataclass(frozen=True)
class SldChamberSolveResult:
    """All district solves of one chamber, aggregated for the sidecar."""

    area_type: str
    district_results: tuple[SldDistrictSolveResult, ...]
    long_weights: pd.DataFrame
    district_summary: pd.DataFrame
    census_rollup: dict[str, Any]


def solve_sld_chamber(
    problems: list[SldDistrictProblem],
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    max_weight_ratio: float | None = 100.0,
    target_loss_cap: float = 10.0,
    min_initial_weight: float = 1e-4,
    seed: int = 0,
    weight_source: str = "populace_us_sld_local",
    worst_districts_listed: int = 20,
) -> SldChamberSolveResult:
    """Solve every district of one chamber with the same uniform operator.

    Districts are solved in sorted ``area_code`` order with per-district
    seeds spawned deterministically from ``seed``, so the chamber result is
    reproducible and independent of input ordering.
    """
    if not problems:
        raise ValueError("solve_sld_chamber needs at least one problem.")
    area_types = {problem.area_type for problem in problems}
    if len(area_types) != 1:
        raise ValueError(
            f"all problems must share one area_type, got {sorted(area_types)}."
        )
    area_type = problems[0].area_type
    codes = [problem.area_code for problem in problems]
    if len(set(codes)) != len(codes):
        raise ValueError("district area_codes must be unique within a chamber.")

    ordered = sorted(problems, key=lambda problem: problem.area_code)
    results: list[SldDistrictSolveResult] = []
    for index, problem in enumerate(ordered):
        district_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        results.append(
            solve_sld_district_weights(
                problem,
                epochs=epochs,
                learning_rate=learning_rate,
                max_weight_ratio=max_weight_ratio,
                target_loss_cap=target_loss_cap,
                min_initial_weight=min_initial_weight,
                seed=district_seed,
            )
        )

    long_frames = [
        pd.DataFrame(
            {
                "area_type": result.problem.area_type,
                "area_code": result.problem.area_code,
                "household_id": result.problem.household_ids,
                "weight": result.weights,
                "weight_source": weight_source,
            }
        )
        for result in results
    ]
    long_weights = pd.concat(long_frames, ignore_index=True)

    district_summary = pd.DataFrame(
        {
            "area_type": [result.problem.area_type for result in results],
            "area_code": [result.problem.area_code for result in results],
            "state_fips": [result.problem.state_fips for result in results],
            "district_code": [result.problem.district_code for result in results],
            "n_households": [len(result.problem.household_ids) for result in results],
            "n_targets": [len(result.problem.targets) for result in results],
            "initial_loss": [result.initial_loss for result in results],
            "final_loss": [result.final_loss for result in results],
            "fraction_within_10pct": [
                result.fraction_within_10pct for result in results
            ],
            "realized_max_weight_ratio": [
                result.realized_max_weight_ratio for result in results
            ],
            "effective_sample_size": [
                result.effective_sample_size for result in results
            ],
            "n_floored_base_weights": [
                result.n_floored_base_weights for result in results
            ],
            "past_cap_at_final": [
                result.past_cap_census["past_at_final"] for result in results
            ],
            "pushed_out": [result.past_cap_census["pushed_out"] for result in results],
        }
    )

    by_pushed = district_summary.sort_values(
        ["pushed_out", "past_cap_at_final", "final_loss"],
        ascending=False,
    )
    census_rollup = {
        "area_type": area_type,
        "n_districts": len(results),
        "n_targets": int(district_summary["n_targets"].sum()),
        "past_at_init": int(
            sum(result.past_cap_census["past_at_init"] for result in results)
        ),
        "past_at_final": int(district_summary["past_cap_at_final"].sum()),
        "escaped": int(sum(result.past_cap_census["escaped"] for result in results)),
        "frozen": int(sum(result.past_cap_census["frozen"] for result in results)),
        "pushed_out": int(district_summary["pushed_out"].sum()),
        "worst_districts": [
            {
                "area_code": str(row.area_code),
                "pushed_out": int(row.pushed_out),
                "past_cap_at_final": int(row.past_cap_at_final),
                "final_loss": float(row.final_loss),
            }
            for row in by_pushed.head(worst_districts_listed).itertuples()
            if row.pushed_out > 0 or row.past_cap_at_final > 0
        ],
    }
    return SldChamberSolveResult(
        area_type=area_type,
        district_results=tuple(results),
        long_weights=long_weights,
        district_summary=district_summary,
        census_rollup=census_rollup,
    )
