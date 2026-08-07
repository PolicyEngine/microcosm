"""Rowwise local solve surface: one weight per cloned household (#495).

The US Build-N shape for the UK dense/local arm. A rowwise household exists
in exactly one assigned area, so an area's target rows draw support only
from the households assigned there. The matrix builder fails closed when an
assigned area is missing from the target surface — local misses are support
or target work, never silent exclusion — and the solve runs under the
reviewed :data:`UK_LOCAL_SOLVE_DOCTRINE` with no per-target parameters and
no doctrine injection point.

The solve itself goes through the public :func:`microcosm.calibrate.calibrate`
front door (#612 increment 2): the area x metric surface is expressed as a
declarative :class:`~microcosm.calibrate.target.TargetSet` (the metric vector
as the measure, area membership as the filter), the kernel enforces the
``WeightKind.CALIBRATED`` transition and mints the mass record, and the
doctrine's declared bounds ride in as explicit arguments. The hand-assembled
:class:`UKRowwiseLocalMatrix` stays as the fail-closed surface definition
and the evidence substrate (support summaries, past-cap census).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_SOLVE_DOCTRINE,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.calibrate.solve import (
    CONSERVE_MASS,
    FREE_MASS,
    calibrate,
    default_target_loss_scales,
    relative_error_loss,
)
from microcosm.calibrate.target import Target, TargetSet
from microcosm.frame import Frame, WeightKind

__all__ = [
    "UKRowwiseDoctrineSolve",
    "UKRowwiseLocalMatrix",
    "build_uk_rowwise_local_matrix",
    "past_cap_census",
    "rowwise_calibration_mass_reason",
    "rowwise_area_support_summary",
    "solve_uk_rowwise_weights_under_doctrine",
]


@dataclass(frozen=True)
class UKRowwiseLocalMatrix:
    """Sparse rowwise calibration surface: one column per cloned household.

    ``metric_values`` retains the per-household metric matrix the constraint
    rows were assembled from, so the declarative target expression the solve
    hands to ``calibrate()`` derives from exactly the same numbers as the
    hand-assembled sparse matrix.
    """

    matrix: sp.csr_matrix
    targets: np.ndarray
    target_frame: pd.DataFrame
    area_codes: tuple[str, ...]
    metric_names: tuple[str, ...]
    household_ids: tuple[Any, ...]
    assigned_areas: tuple[str, ...]
    metric_values: np.ndarray

    @property
    def n_areas(self) -> int:
        return len(self.area_codes)

    @property
    def n_households(self) -> int:
        return len(self.household_ids)


@dataclass(frozen=True)
class UKRowwiseDoctrineSolve:
    """A doctrine solve's calibrated frame and its solve evidence.

    ``frame`` is the finished calibrated UK national frame: kernel-minted
    ``CALIBRATED`` weights, the kernel's mass record on the log, and the
    persisted ``household_weight`` column refreshed to the typed vector.
    The evidence fields keep the pre-#612 vocabulary: a labelled
    diagnostics frame (scaled relative errors on the doctrine's default
    scale rule) and the microcosm#492 past-cap census.
    """

    frame: Frame
    weights: np.ndarray
    initial_weights: np.ndarray
    diagnostics: pd.DataFrame
    loss_trajectory: np.ndarray
    initial_loss: float
    final_loss: float
    n_nonzero: int
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

    from microcosm.build.uk_runtime.local_geography import align_area_targets

    if metrics.empty:
        raise ValueError("metrics must not be empty.")
    if metrics.index.has_duplicates:
        duplicates = metrics.index[metrics.index.duplicated()].unique()
        raise ValueError(
            "metrics household index must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    metric_labels = [str(column) for column in metrics.columns]
    duplicate_labels = sorted(
        {label for label in metric_labels if metric_labels.count(label) > 1}
    )
    if duplicate_labels:
        raise ValueError(f"metrics has duplicate column label(s): {duplicate_labels}.")
    from microcosm.build.uk_runtime.local_geography import _AREA_METADATA_COLUMNS

    metadata_collisions = sorted(set(metric_labels) & set(_AREA_METADATA_COLUMNS))
    if metadata_collisions:
        raise ValueError(
            "metric column(s) collide with target-frame metadata names and "
            f"would silently calibrate metadata: {metadata_collisions}."
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
    # Fail closed on unreachable rows: a nonzero target whose area has no
    # supporting entries can never be hit and would sit invisibly inside the
    # loss cap. Misses are support or target work, never silent exclusion.
    row_support = np.diff(matrix.indptr)
    unreachable = (row_support == 0) & (
        target_frame["value"].to_numpy(dtype=np.float64) != 0.0
    )
    if unreachable.any():
        examples = [
            f"{row.area_code}/{row.metric}"
            for row in target_frame.loc[unreachable].head(5).itertuples(index=False)
        ]
        raise ValueError(
            f"{int(unreachable.sum())} target row(s) have a nonzero target "
            f"but zero household support: {examples}. Add support (clones, "
            "assignment) or fix the target surface."
        )
    return UKRowwiseLocalMatrix(
        matrix=matrix,
        targets=target_frame["value"].to_numpy(dtype=np.float64),
        target_frame=target_frame,
        area_codes=codes,
        metric_names=metric_names,
        household_ids=tuple(metrics.index.tolist()),
        assigned_areas=tuple(assigned.tolist()),
        metric_values=values,
    )


def _require_uniform_target_surface(problem: UKRowwiseLocalMatrix) -> None:
    """Refuse a target surface whose rows repeat an (area, metric) cell.

    ``build_uk_rowwise_local_matrix`` constructs unique rows by design; a
    hand-built matrix that duplicates a row would double that cell's weight
    in the uniform loss — a per-target knob smuggled through the surface.
    """

    frame = problem.target_frame
    required = {"area_type", "area_code", "metric"}
    if not required <= set(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(
            f"doctrine solve requires target_frame column(s) {missing} to "
            "verify surface uniqueness."
        )
    duplicated = frame.duplicated(["area_type", "area_code", "metric"])
    if duplicated.any():
        rows = frame.loc[
            duplicated, ["area_type", "area_code", "metric"]
        ].drop_duplicates()
        examples = [tuple(map(str, row)) for row in rows.head(5).to_numpy()]
        raise ValueError(
            "doctrine solve refuses a non-uniform target surface: duplicate "
            f"(area_type, area_code, metric) row(s) {examples} would act as "
            "implicit per-target weights."
        )


def rowwise_calibration_mass_reason(bound_families: Sequence[str]) -> str:
    """The mass-record reason a rowwise doctrine calibration declares.

    Passed to ``calibrate(mass_reason=...)`` so the kernel-minted record on
    the calibrated frame names the bound target families;
    ``write_uk_rowwise_dataset``'s chain-currency fence refuses weights whose
    total disagrees with that record, so a calibration cannot ship silently.
    """

    families = [str(name) for name in bound_families]
    if not families or any(not name.strip() for name in families):
        raise ValueError("bound_families must name at least one target family.")
    return (
        "Rowwise doctrine calibration to bound target family(ies) "
        f"{', '.join(families)}; total household mass moved with the "
        "targets."
    )


def _rowwise_target_set(problem: UKRowwiseLocalMatrix) -> TargetSet:
    """Express the rowwise surface as declarative calibration targets.

    One target per (area, metric) cell, in ``target_frame`` order: the
    metric's per-household vector is the measure, the household's area
    assignment is the filter. Both close over the problem's own arrays, so
    the declarative expression derives from exactly the numbers the
    hand-assembled sparse matrix was built from.
    """

    assigned = np.asarray(problem.assigned_areas, dtype=object)
    metric_columns = {
        name: np.ascontiguousarray(problem.metric_values[:, index], dtype=np.float64)
        for index, name in enumerate(problem.metric_names)
    }
    area_masks = {
        code: (assigned == code).astype(np.float64) for code in problem.area_codes
    }

    def _measure(values: np.ndarray):
        def measure(frame: Frame) -> np.ndarray:
            return values

        return measure

    targets = []
    for row in problem.target_frame.itertuples(index=False):
        targets.append(
            Target(
                name=f"{row.area_type}/{row.area_code}/{row.metric}",
                entity="household",
                measure=_measure(metric_columns[str(row.metric)]),
                value=float(row.value),
                filter=_measure(area_masks[str(row.area_code)]),
                source="uk_rowwise_local_surface",
                metadata={
                    "area_type": str(row.area_type),
                    "area_code": str(row.area_code),
                    "metric": str(row.metric),
                },
            )
        )
    return TargetSet(targets)


def solve_uk_rowwise_weights_under_doctrine(
    frame: Frame,
    problem: UKRowwiseLocalMatrix,
    *,
    bound_families: Sequence[str],
    epochs: int = 512,
    learning_rate: float = 0.15,
    conserve_mass: bool = False,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    budget_iters: int = 10,
    seed: int = 0,
) -> UKRowwiseDoctrineSolve:
    """Solve rowwise household weights under the reviewed doctrine.

    Structurally knob-free like before the ``calibrate()`` migration: no
    per-target parameters and no doctrine parameter — the bounds always come
    from :data:`UK_LOCAL_SOLVE_DOCTRINE` and ride into the public front door
    as explicit arguments. Initial weights are the frame's typed household
    weights directly (a rowwise household exists in exactly one area, so
    nothing is split); zero weights are refused — a dead row must be dropped
    or revived upstream with a recorded mass change, never resurrected by a
    solver floor. The kernel enforces the ``CALIBRATED`` kind transition and
    mints the mass record (reason from
    :func:`rowwise_calibration_mass_reason`); the returned frame carries
    both, with the persisted ``household_weight`` column refreshed.
    """

    doctrine = UK_LOCAL_SOLVE_DOCTRINE
    _require_uniform_target_surface(problem)
    mass_reason = rowwise_calibration_mass_reason(bound_families)

    household = frame.table("household")
    frame_ids = tuple(household["household_id"].tolist())
    if frame_ids != problem.household_ids:
        raise ValueError(
            "doctrine solve requires the frame's household rows to match the "
            "problem's households exactly (same ids, same order); got "
            f"{len(frame_ids)} frame rows vs {problem.n_households} problem "
            "households."
        )
    base = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    if (base == 0).any():
        raise ValueError(
            f"base weights contain {int((base == 0).sum())} zero value(s); a "
            "doctrine solve must not resurrect dead rows — drop them or "
            "revive them upstream with a recorded mass change."
        )

    result = calibrate(
        frame,
        _rowwise_target_set(problem),
        weight_entity="household",
        epochs=epochs,
        learning_rate=learning_rate,
        mass=CONSERVE_MASS if conserve_mass else FREE_MASS,
        mass_reason=None if conserve_mass else mass_reason,
        max_weight_ratio=doctrine.max_weight_ratio,
        target_records=target_records,
        l0_lambda=l0_lambda,
        budget_iters=budget_iters,
        seed=seed,
        target_loss_cap=doctrine.target_loss_cap,
    )
    if result.skipped:
        reasons = [
            f"{skipped.target.name}: {skipped.reason}" for skipped in result.skipped[:5]
        ]
        raise ValueError(
            f"doctrine solve refuses {len(result.skipped)} uncompilable "
            f"target(s): {reasons}. The rowwise surface must compile whole — "
            "a skipped target is a harness bug, never a silent exclusion."
        )
    if len(result.diagnostics) != len(problem.target_frame):
        raise ValueError(
            "compiled diagnostics do not cover the declared target surface: "
            f"{len(result.diagnostics)} rows vs {len(problem.target_frame)}."
        )

    targets_vec = problem.targets
    initial_estimates = np.asarray(
        [diagnostic.initial_estimate for diagnostic in result.diagnostics],
        dtype=np.float64,
    )
    final_estimates = np.asarray(
        [diagnostic.final_estimate for diagnostic in result.diagnostics],
        dtype=np.float64,
    )
    compiled_targets = np.asarray(
        [diagnostic.target for diagnostic in result.diagnostics],
        dtype=np.float64,
    )
    if not np.allclose(compiled_targets, targets_vec, rtol=0, atol=0):
        raise ValueError(
            "compiled target values disagree with the declared surface; the "
            "declarative expression and the hand-assembled matrix must "
            "derive from the same numbers."
        )

    scales = default_target_loss_scales(targets_vec)
    diagnostics = problem.target_frame.copy()
    diagnostics["target"] = targets_vec
    diagnostics["initial_estimate"] = initial_estimates
    diagnostics["final_estimate"] = final_estimates
    diagnostics["relative_error"] = np.divide(
        final_estimates - targets_vec,
        scales,
        out=np.zeros_like(targets_vec, dtype=np.float64),
        where=scales != 0,
    )
    diagnostics["abs_relative_error"] = np.abs(diagnostics["relative_error"])
    census = past_cap_census(
        initial_estimates,
        final_estimates,
        targets_vec,
        target_loss_cap=doctrine.target_loss_cap,
        target_loss_scales=scales,
        target_frame=problem.target_frame,
    )
    # Reported in float64 from the compiled estimates (the trajectory head
    # is a float32 optimizer value), matching the pre-migration solver's
    # reported precision; final_loss is already the float64 closing loss.
    initial_loss = float(
        relative_error_loss(
            initial_estimates,
            targets_vec,
            target_loss_weights=None,
            target_loss_scales=scales,
            target_loss_cap=doctrine.target_loss_cap,
        )
    )

    # The kernel product carries the CALIBRATED transition and the mass
    # record; the UK carrier additionally persists the weight column, so the
    # finished frame is hard-constructed with the refreshed column and the
    # kernel's own mass log (nothing appended here).
    calibrated_household = result.frame.table("household").copy()
    calibrated_household["household_weight"] = np.asarray(
        result.weights, dtype=np.float64
    )
    finished = uk_national_frame(
        person=result.frame.table("person"),
        benunit=result.frame.table("benunit"),
        household=calibrated_household,
        time_period=uk_time_period(result.frame),
        weight_kind=WeightKind.CALIBRATED,
        mass_log=result.frame.mass_log,
    )
    validate_uk_national_frame(finished)
    return UKRowwiseDoctrineSolve(
        frame=finished,
        weights=np.asarray(result.weights, dtype=np.float64),
        initial_weights=np.asarray(result.initial_weights, dtype=np.float64),
        diagnostics=diagnostics,
        loss_trajectory=np.asarray(result.loss_trajectory, dtype=np.float64),
        initial_loss=initial_loss,
        final_loss=float(result.final_loss),
        n_nonzero=int(result.n_nonzero),
        past_cap_census=census,
    )


def rowwise_area_support_summary(
    problem: UKRowwiseLocalMatrix,
    weights: Sequence[float],
    *,
    source_household_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Per-area support of a rowwise weight vector, all target areas included."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {values.shape}.")
    if len(values) != problem.n_households:
        raise ValueError(
            "weights must align with households, got "
            f"{len(values)} weights for {problem.n_households} households."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be finite and non-negative.")
    sources_list = (
        list(problem.household_ids)
        if source_household_ids is None
        else list(source_household_ids)
    )
    if len(sources_list) != problem.n_households:
        raise ValueError(
            "source_household_ids must align with households, got "
            f"{len(sources_list)} for {problem.n_households}."
        )
    sources = np.empty(problem.n_households, dtype=object)
    for index, source in enumerate(sources_list):
        sources[index] = source
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
