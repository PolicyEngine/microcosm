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

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

from microcosm.build.uk_runtime import local_target_census
from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_SOLVE_DOCTRINE,
)
from microcosm.build.uk_runtime.local_targets import AREA_TYPES
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    coerce_reviewed_exclusions,
    exclusion_evaluation_date,
    load_uk_reviewed_exclusion_register,
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
    "UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE",
    "UKRowwiseDoctrineSolve",
    "UKRowwiseLocalMatrix",
    "build_uk_rowwise_local_matrix",
    "past_cap_census",
    "require_adjudicated_uk_local_binding",
    "uk_area_support_summary",
    "uk_ladder_area_support_summary",
    "rowwise_calibration_mass_reason",
    "rowwise_area_support_summary",
    "solve_uk_rowwise_weights_under_doctrine",
]

UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE = "local_binding_adjudications.json"


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
    binding_adjudications: Mapping[str, Any]


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


def require_adjudicated_uk_local_binding(
    bound_families: Sequence[str],
    target_frame: pd.DataFrame,
    *,
    census: Mapping[str, Any] | None = None,
    register: Mapping[str, Any] | None = None,
    now: Any = None,
) -> dict[str, Any]:
    """Require in-force review records before binding fenced UK local families."""

    census_payload = (
        local_target_census.load_uk_local_target_census()
        if census is None
        else census
    )
    family_rows = _uk_local_census_family_rows(census_payload)
    declared, parsed = _normalise_uk_local_bound_families(
        bound_families,
        family_rows=family_rows,
    )
    derived = _derive_uk_local_bound_families_from_target_frame(
        target_frame,
        family_rows=family_rows,
    )
    if sorted(declared) != derived:
        missing = sorted(set(derived) - set(declared))
        extra = sorted(set(declared) - set(derived))
        raise ValueError(
            "UK local binding declarations: declared bound families "
            f"{sorted(declared)} disagree with the families derived from "
            f"the target surface {derived}; missing {missing}, extra "
            f"{extra}; the declaration must name exactly what the matrix "
            "binds."
        )

    records = (
        load_uk_reviewed_exclusion_register(
            None,
            resource=UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
        )
        if register is None
        else coerce_reviewed_exclusions(
            register,
            label=UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
        )
    )
    evaluated_on = exclusion_evaluation_date(now)
    stood_on: dict[str, dict[str, dict[str, str]]] = {}
    expiring_soon: dict[str, str] = {}
    for declared_family in declared:
        census_family, _area_type = parsed[declared_family]
        stood_on[declared_family] = {}
        for fence_id in family_rows[census_family].get("adjudications", ()):
            fence = str(fence_id)
            record = records.get(fence)
            if record is None:
                raise ValueError(
                    f"{UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE}: "
                    f"bound family {declared_family!r} requires an in-force "
                    f"adjudication of fence {fence!r} and the committed "
                    "register records none. Record a reviewed adjudication "
                    "(reason stating the ruling and evidence, approver, "
                    "window) before binding — the fence is a reviewed "
                    "requirement, never a runtime flag."
                )
            if record.expired(evaluated_on) or record.premature(evaluated_on):
                raise ValueError(
                    f"{UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE}: "
                    f"bound family {declared_family!r} requires adjudication "
                    f"of fence {fence!r}, but the entry is outside its "
                    f"reviewed window ({record.approved_on}.."
                    f"{record.expires_on}, evaluated "
                    f"{evaluated_on.isoformat()}): correct the underlying "
                    "gap or renew the adjudication with a new approval and "
                    "expiry."
                )
            if (date.fromisoformat(record.expires_on) - evaluated_on).days <= 7:
                expiring_soon[fence] = record.expires_on
            stood_on[declared_family][fence] = record.policy_payload()

    for fence, expires_on in sorted(expiring_soon.items()):
        warnings.warn(
            f"{UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE}: adjudication "
            f"of fence {fence!r} expires {expires_on} — within one week of "
            f"the evaluation date {evaluated_on.isoformat()}; renew the "
            "adjudication or the fence closes the rowwise solve on expiry.",
            UserWarning,
            stacklevel=2,
        )

    fence_families: dict[str, set[str]] = {}
    for family, row in family_rows.items():
        for fence_id in row.get("adjudications", ()):
            fence_families.setdefault(str(fence_id), set()).add(family)
    bound_census_families = {parsed[name][0] for name in declared}
    dormant = sorted(
        fence
        for fence in records
        if not (fence_families.get(fence, set()) & bound_census_families)
    )
    return {
        "register_resource": UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
        "evaluated_on": evaluated_on.isoformat(),
        "bound_families": list(declared),
        "stood_on": stood_on,
        "dormant": dormant,
    }


def _uk_local_census_family_rows(
    census: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = census.get("families")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("UK local target census must carry a families list.")
    families: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("UK local target census family rows must be objects.")
        family = str(row.get("family", ""))
        if not family:
            raise ValueError("UK local target census family row has no family id.")
        if family in families:
            raise ValueError(
                f"UK local target census declares duplicate family {family!r}."
            )
        families[family] = row
    return families


def _normalise_uk_local_bound_families(
    bound_families: Sequence[str],
    *,
    family_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, tuple[str, str]]]:
    if isinstance(bound_families, str):
        raise ValueError(
            "UK local binding declarations: bound_families must be a "
            "sequence of family/area_type strings, not one string."
        )
    declared = tuple(str(name) for name in bound_families)
    if not declared:
        raise ValueError(
            "UK local binding declarations: bound_families must name at "
            "least one family/area_type pair."
        )
    blanks = [name for name in declared if not name.strip()]
    if blanks:
        raise ValueError(
            "UK local binding declarations: bound_families must not contain "
            "blank family names."
        )
    padded = [name for name in declared if name != name.strip()]
    if padded:
        raise ValueError(
            "UK local binding declarations: bound_families must not carry "
            f"surrounding whitespace: {padded}."
        )
    duplicates = sorted({name for name in declared if declared.count(name) > 1})
    if duplicates:
        raise ValueError(
            "UK local binding declarations: duplicate bound family(ies) "
            f"{duplicates}."
        )

    parsed: dict[str, tuple[str, str]] = {}
    for name in declared:
        if "/" not in name:
            raise ValueError(
                "UK local binding declarations: bound family "
                f"{name!r} must have form '<census_family>/<area_type>'."
            )
        family, area_type = name.rsplit("/", 1)
        if not family or not area_type:
            raise ValueError(
                "UK local binding declarations: bound family "
                f"{name!r} must have form '<census_family>/<area_type>'."
            )
        if area_type not in AREA_TYPES:
            raise ValueError(
                "UK local binding declarations: bound family "
                f"{name!r} names unsupported area_type {area_type!r}; "
                f"expected one of {list(AREA_TYPES)}."
            )
        if family not in family_rows:
            raise ValueError(
                "UK local binding declarations: bound family "
                f"{name!r} names unknown census family {family!r}; "
                f"expected one of {sorted(family_rows)}."
            )
        parsed[name] = (family, area_type)
    return declared, parsed


def _derive_uk_local_bound_families_from_target_frame(
    target_frame: pd.DataFrame,
    *,
    family_rows: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    required = {"area_type", "metric"}
    if not required <= set(target_frame.columns):
        missing = sorted(required - set(target_frame.columns))
        raise ValueError(
            "UK local binding declarations: target_frame column(s) "
            f"{missing} are required to derive bound families."
        )
    derived: set[str] = set()
    cells = target_frame[["area_type", "metric"]].drop_duplicates()
    for row in cells.itertuples(index=False):
        area_type = str(row.area_type)
        if area_type not in AREA_TYPES:
            raise ValueError(
                "UK local binding declarations: target surface names "
                f"unsupported area_type {area_type!r}; expected one of "
                f"{list(AREA_TYPES)}."
            )
        metric = str(row.metric)
        family = local_target_census.family_for_metric(metric)
        if family not in family_rows:
            raise ValueError(
                "UK local binding declarations: target surface metric "
                f"{metric!r} classified into unknown census family "
                f"{family!r}."
            )
        derived.add(f"{family}/{area_type}")
    return sorted(derived)


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

    def _constant_vector(values: np.ndarray):
        def vector(frame: Frame) -> np.ndarray:
            return values

        return vector

    targets = []
    for row in problem.target_frame.itertuples(index=False):
        targets.append(
            Target(
                name=f"{row.area_type}/{row.area_code}/{row.metric}",
                entity="household",
                measure=_constant_vector(metric_columns[str(row.metric)]),
                value=float(row.value),
                filter=_constant_vector(area_masks[str(row.area_code)]),
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
    binding_adjudications = require_adjudicated_uk_local_binding(
        bound_families,
        problem.target_frame,
    )
    mass_reason = rowwise_calibration_mass_reason(bound_families)

    household = frame.table("household")
    frame_ids = tuple(household["household_id"].tolist())
    if frame_ids != problem.household_ids:
        prefix = (
            "doctrine solve requires the frame's household rows to match the "
            "problem's households exactly (same ids, same order); "
        )
        if len(frame_ids) != problem.n_households:
            raise ValueError(
                prefix + f"got {len(frame_ids)} frame rows vs "
                f"{problem.n_households} problem households."
            )
        mismatch = next(
            index
            for index, (frame_id, problem_id) in enumerate(
                zip(frame_ids, problem.household_ids, strict=True)
            )
            if frame_id != problem_id
        )
        shape = (
            "the same households arrive in a different order"
            if set(frame_ids) == set(problem.household_ids)
            else "the id sets differ"
        )
        raise ValueError(
            prefix + f"{shape}; first mismatch at row {mismatch} "
            f"(frame id {frame_ids[mismatch]!r} vs problem id "
            f"{problem.household_ids[mismatch]!r})."
        )
    base = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    if (base == 0).any():
        raise ValueError(
            f"base weights contain {int((base == 0).sum())} zero value(s); a "
            "doctrine solve must not resurrect dead rows — drop them or "
            "revive them upstream with a recorded mass change."
        )

    target_set = _rowwise_target_set(problem)
    result = calibrate(
        frame,
        target_set,
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
    # The evidence tables consume diagnostics positionally, and diagnostics
    # order is documented only as "aligned to the compiled problem rows" —
    # not as declaration order. Assert the alignment by name (target values
    # legitimately repeat on a local surface, so value equality alone could
    # pass a reordering by coincidence).
    for index, (diagnostic, target) in enumerate(
        zip(result.diagnostics, target_set.targets, strict=True)
    ):
        if diagnostic.name != target.row_name:
            raise ValueError(
                "compiled diagnostics are not aligned with the declared "
                f"target surface: row {index} reports {diagnostic.name!r} "
                f"where the surface declares {target.row_name!r}. The "
                "evidence tables consume diagnostics positionally; a "
                "reordered front-door result must be refused, never "
                "misattributed."
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
    # Exact equality, stated plainly; a NaN on either side also refuses
    # (array_equal never treats NaN as equal to anything).
    if not np.array_equal(compiled_targets, targets_vec):
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

    # The UK carrier persists the weight column, and the kernel product is
    # immutable with no table-refresh operation, so the finished frame is
    # *rebuilt* through the canonical assembler from the kernel product's
    # tables, mass log, and period. A rebuild can silently drop kernel
    # surfaces the assembler does not carry, so the guards below refuse to
    # ship if the kernel product held strata or metadata the rebuilt frame
    # lost (both trivially equal today; the guard is the boundary marker
    # for the day they are not).
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
    if not result.frame.strata.equals(finished.strata):
        raise ValueError(
            "the calibrated frame carries strata the rebuilt UK national "
            "frame would drop; extend uk_national_frame to carry them "
            "before shipping a strata-bearing local solve."
        )
    if dict(result.frame.metadata) != dict(finished.metadata):
        raise ValueError(
            "the calibrated frame carries metadata beyond the UK time "
            "period; the rebuild would drop it, so the solve refuses "
            "instead."
        )
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
        binding_adjudications=binding_adjudications,
    )


def rowwise_area_support_summary(
    problem: UKRowwiseLocalMatrix,
    weights: Sequence[float],
    *,
    source_household_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Per-area support of a rowwise weight vector, all target areas included."""

    return uk_area_support_summary(
        problem.assigned_areas,
        weights,
        area_codes=problem.area_codes,
        source_household_ids=(
            problem.household_ids
            if source_household_ids is None
            else source_household_ids
        ),
    )


def uk_area_support_summary(
    assigned_areas: Sequence[Any],
    weights: Sequence[float],
    *,
    area_codes: Sequence[Any],
    source_household_ids: Sequence[Any],
) -> pd.DataFrame:
    """Summarize row and weighted support for a declared roster of areas."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {values.shape}.")
    assigned = np.asarray(assigned_areas, dtype=object)
    if assigned.ndim != 1:
        raise ValueError(
            f"assigned_areas must be one-dimensional, got shape {assigned.shape}."
        )
    if len(values) != len(assigned):
        raise ValueError(
            "weights must align with households, got "
            f"{len(values)} weights for {len(assigned)} households."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be finite and non-negative.")
    sources_list = list(source_household_ids)
    if len(sources_list) != len(assigned):
        raise ValueError(
            "source_household_ids must align with households, got "
            f"{len(sources_list)} for {len(assigned)}."
        )
    sources = np.empty(len(assigned), dtype=object)
    for index, source in enumerate(sources_list):
        sources[index] = source
    area_roster = np.asarray(area_codes, dtype=object)
    if area_roster.ndim != 1:
        raise ValueError(
            f"area_codes must be one-dimensional, got shape {area_roster.shape}."
        )

    source_codes, _ = pd.factorize(
        pd.Series(sources, dtype=object),
        sort=False,
        use_na_sentinel=False,
    )
    positive = values > 0
    rows = pd.DataFrame(
        {
            "area_code": assigned,
            "weight": values,
            "weight_squared": np.square(values),
            "positive": positive,
            "positive_source_code": np.where(positive, source_codes, -1),
        }
    )
    grouped = rows.groupby("area_code", sort=False, dropna=False).agg(
        assigned_households=("weight", "size"),
        nonzero_households=("positive", "sum"),
        nonzero_source_households=(
            "positive_source_code",
            lambda codes: int(codes[codes >= 0].nunique()),
        ),
        weight_sum=("weight", "sum"),
        max_weight=("weight", "max"),
        weight_square_sum=("weight_squared", "sum"),
    )
    support = pd.DataFrame({"area_code": area_roster}).merge(
        grouped,
        how="left",
        left_on="area_code",
        right_index=True,
        sort=False,
    )
    count_columns = (
        "assigned_households",
        "nonzero_households",
        "nonzero_source_households",
    )
    support[list(count_columns)] = (
        support[list(count_columns)].fillna(0).astype(np.int64)
    )
    support[["weight_sum", "max_weight", "weight_square_sum"]] = support[
        ["weight_sum", "max_weight", "weight_square_sum"]
    ].fillna(0.0)
    effective_sample_size = np.zeros(len(support), dtype=np.float64)
    np.divide(
        np.square(support["weight_sum"].to_numpy(dtype=np.float64)),
        support["weight_square_sum"].to_numpy(dtype=np.float64),
        out=effective_sample_size,
        where=support["weight_square_sum"].to_numpy(dtype=np.float64) > 0,
    )
    support["effective_sample_size"] = effective_sample_size
    return support[
        [
            "area_code",
            "assigned_households",
            "nonzero_households",
            "nonzero_source_households",
            "weight_sum",
            "max_weight",
            "effective_sample_size",
        ]
    ]


def uk_ladder_area_support_summary(
    household: pd.DataFrame,
    ladder: Any,
    *,
    weight_column: str = "household_weight",
    source_column: str = "source_household_id",
) -> dict[str, pd.DataFrame]:
    """Return matrix-free constituency and LA support on the ladder roster."""

    if source_column not in household.columns:
        raise ValueError(
            f"household table must contain source column {source_column!r}; "
            "distinct-source honesty requires it (pass source_column="
            "'household_id' explicitly for row-grain sources)."
        )
    if weight_column not in household.columns:
        raise ValueError(f"household table must contain weight column {weight_column!r}.")

    summaries: dict[str, pd.DataFrame] = {}
    for area_type, assigned_column, ladder_codes in (
        ("constituency", "constituency_code", ladder.constituency_code),
        ("la", "local_authority_code", ladder.local_authority_code),
    ):
        if assigned_column not in household.columns:
            raise ValueError(
                f"household table must contain assigned area column "
                f"{assigned_column!r}."
            )
        summaries[area_type] = uk_area_support_summary(
            household[assigned_column],
            household[weight_column],
            area_codes=np.unique(ladder_codes),
            source_household_ids=household[source_column],
        )
    return summaries
