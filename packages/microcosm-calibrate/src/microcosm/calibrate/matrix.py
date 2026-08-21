"""Compile targets + a Frame into a sparse linear constraint system.

:func:`build_constraint_matrix` turns a :class:`~microcosm.calibrate.target.TargetSet`
and a :class:`~microcosm.frame.Frame` into a :class:`CalibrationProblem`: a sparse
CSR matrix ``A`` (one row per compilable ``(target, period)``, one column per
record of the calibrated entity), the target vector ``b``, the row names, and the
initial weights resolved from the frame.

The matrix is the linear form of the charter's calibration objective: an
estimate of every target's aggregate is ``A @ w`` for a weight vector ``w``, and
calibration searches for the ``w`` that drives ``A @ w`` to ``b``. Multi-period
targets stack as extra rows over the *same* ``w`` — the "one weight per
trajectory" rule — because cross-sections are not calibrated independently.

A target the frame cannot compile (a measure column missing on the entity or a
length mismatch) is **skipped and reported**, never silently dropped: the returned
problem carries a
:class:`SkippedTarget` for each, naming the target and the reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from microcosm.calibrate.target import Target, TargetSet
from microcosm.frame import Frame, Weights

__all__ = ["CalibrationProblem", "SkippedTarget", "build_constraint_matrix"]


@dataclass(frozen=True)
class SkippedTarget:
    """A target that could not be compiled against the frame.

    Attributes:
        target: The target that was skipped.
        reason: Human-readable explanation naming the culprit, e.g. a missing
            measure or filter column.
    """

    target: Target
    reason: str


@dataclass(frozen=True)
class CalibrationProblem:
    """The sparse linear system a calibration solves.

    ``A @ w`` estimates every compiled target's aggregate; calibration searches
    for ``w`` driving ``A @ w`` toward ``b``.

    Attributes:
        matrix: The constraint matrix ``A`` as a CSR ``scipy.sparse`` array of
            shape ``(n_targets, n_weights)``: one row per compiled
            ``(target, period)``, one column per record of ``weight_entity``.
        target_vector: The right-hand side ``b`` of length ``n_targets``. Each
            row is a sum target, so the right-hand side is the target value.
        names: Row labels (``"name@period"``), aligned to ``matrix`` rows.
        initial_weights: The :class:`~microcosm.frame.Weights` resolved from the
            frame for ``weight_entity`` — the calibration's starting point and
            the mass reference.
        weight_entity: The entity whose weight vector is calibrated.
        targets: The compiled targets, aligned to ``matrix`` rows / ``names``.
        skipped: Targets that could not be compiled, each with its reason.

    Raises:
        ValueError: If row/column dimensions are inconsistent (a construction
            guard; :func:`build_constraint_matrix` never produces one).
    """

    matrix: sparse.csr_array
    target_vector: np.ndarray
    names: tuple[str, ...]
    initial_weights: Weights
    weight_entity: str
    targets: tuple[Target, ...]
    skipped: tuple[SkippedTarget, ...] = ()

    def __post_init__(self) -> None:
        n_rows = self.matrix.shape[0]
        if not (
            len(self.target_vector) == n_rows == len(self.names) == len(self.targets)
        ):
            raise ValueError(
                "CalibrationProblem row dimensions disagree: matrix has "
                f"{n_rows} rows, target_vector {len(self.target_vector)}, "
                f"names {len(self.names)}, targets {len(self.targets)}."
            )
        if self.matrix.shape[1] != len(self.initial_weights):
            raise ValueError(
                f"CalibrationProblem column count {self.matrix.shape[1]} does "
                f"not match the {len(self.initial_weights)} initial weights of "
                f"entity {self.weight_entity!r}."
            )

    @property
    def n_targets(self) -> int:
        """Number of compiled constraint rows."""
        return self.matrix.shape[0]

    @property
    def n_weights(self) -> int:
        """Number of weight columns (records of ``weight_entity``)."""
        return self.matrix.shape[1]

    def estimates(self, weights: np.ndarray) -> np.ndarray:
        """Aggregate estimates ``A @ weights`` for every compiled target.

        Args:
            weights: A weight vector of length :attr:`n_weights`.

        Returns:
            The estimated aggregate per target row (length :attr:`n_targets`).
        """
        return np.asarray(self.matrix @ np.asarray(weights, dtype=np.float64))


def _entity_row(
    target: Target,
    frame: Frame,
    weight_entity: str,
) -> np.ndarray:
    """Build ``target``'s row aligned to ``weight_entity``'s weight vector.

    A target measured on the calibrated entity contributes its constraint row
    directly. A target measured on a *different* entity nested under the
    calibrated one (e.g. a person-level count while calibrating household
    weights) is collapsed onto the calibrated entity: the per-record values are
    summed within each calibrated-entity group, because the calibrated weight is
    shared across that group's members.

    Args:
        target: The target to compile.
        frame: The frame to read from.
        weight_entity: The entity being calibrated.

    Returns:
        The constraint row aligned to ``weight_entity``'s records.

    Raises:
        ValueError: If the target's entity is not nested under the calibrated
            entity through the frame schema's declared person memberships.
        KeyError: If a measure/filter column is missing.
    """
    if target.entity == weight_entity:
        return target.constraint_row(frame)

    row = target.constraint_row(frame)
    positions = _nested_positions(target.entity, frame, weight_entity)
    collapsed = np.zeros(frame.n(weight_entity), dtype=np.float64)
    np.add.at(collapsed, positions, row)
    return collapsed


def _nested_positions(
    entity: str,
    frame: Frame,
    weight_entity: str,
) -> np.ndarray:
    """Position of each ``entity`` row within ``weight_entity``'s table.

    The frame schema declares group relationships as person membership columns.
    A person-level target collapses directly through ``person_<group>_id``.
    A group-level target collapses to another group only when every source
    group id appears with exactly one destination group id in the person table.
    """

    schema = frame.schema
    person_entity = schema.person_entity
    if entity == person_entity:
        if weight_entity not in schema.group_entities:
            raise ValueError(
                f"Cannot collapse target entity {entity!r} to non-group "
                f"weight entity {weight_entity!r}."
            )
        return frame._group_positions(weight_entity)
    if (
        entity not in schema.group_entities
        or weight_entity not in schema.group_entities
    ):
        raise ValueError(
            f"Cannot collapse target entity {entity!r} to weight entity "
            f"{weight_entity!r}; schema declares person entity "
            f"{person_entity!r} and groups {list(schema.group_entities)}."
        )

    person = frame.person
    source_column = schema.membership_column(entity)
    weight_column = schema.membership_column(weight_entity)
    source_to_weight: dict[object, object] = {}
    split_source_ids: list[object] = []
    for source_id, destination_id in (
        person[[source_column, weight_column]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        existing = source_to_weight.get(source_id)
        if existing is not None and existing != destination_id:
            split_source_ids.append(source_id)
            continue
        source_to_weight[source_id] = destination_id
    if split_source_ids:
        raise ValueError(
            f"Cannot collapse target entity {entity!r} to weight entity "
            f"{weight_entity!r}: {entity} id(s) span multiple {weight_entity} "
            f"ids, including {split_source_ids[:5]}."
        )

    source_ids = frame.table(entity)[schema.id_column(entity)].to_numpy()
    missing = [
        source_id for source_id in source_ids if source_id not in source_to_weight
    ]
    if missing:
        raise ValueError(
            f"Cannot collapse target entity {entity!r} to weight entity "
            f"{weight_entity!r}: {entity} id(s) have no person membership, "
            f"including {missing[:5]}."
        )
    destination_ids = np.asarray(
        [source_to_weight[source_id] for source_id in source_ids]
    )
    weight_ids = frame.table(weight_entity)[schema.id_column(weight_entity)].to_numpy()
    order = np.argsort(weight_ids)
    sorted_weight_ids = weight_ids[order]
    sorted_positions = np.searchsorted(sorted_weight_ids, destination_ids)
    valid = sorted_positions < len(sorted_weight_ids)
    valid[valid] = sorted_weight_ids[sorted_positions[valid]] == destination_ids[valid]
    if not bool(valid.all()):
        bad = destination_ids[~valid][:5].tolist()
        raise ValueError(
            f"Cannot collapse target entity {entity!r} to weight entity "
            f"{weight_entity!r}: mapped {weight_entity} id(s) are absent from "
            f"the weight table, including {bad}."
        )
    return order[sorted_positions]


def build_constraint_matrix(
    frame: Frame,
    targets: TargetSet,
    weight_entity: str = "household",
) -> CalibrationProblem:
    """Compile ``targets`` against ``frame`` into a sparse constraint system.

    Each compilable ``(target, period)`` becomes one CSR row of ``A`` over the
    weight vector of ``weight_entity``; multi-period targets stack as extra rows
    over the *same* weight column set (the "one weight per trajectory" rule). A
    target the frame cannot compile is skipped and recorded as a
    :class:`SkippedTarget`, never dropped silently.

    Args:
        frame: The frame to compile against. Its ``weight_entity`` weights are
            the calibration's starting point and mass reference.
        targets: The facts to compile.
        weight_entity: The entity whose weights are calibrated (default
            ``"household"``). Must be declared by the frame's schema.

    Returns:
        A :class:`CalibrationProblem` carrying the matrix, target vector, row
        names, initial weights, and any skipped targets.

    Raises:
        TypeError: If ``targets`` is not a :class:`TargetSet`.
        ValueError: If ``weight_entity`` is not a declared entity, or every
            target was skipped (an empty system has nothing to calibrate — the
            message lists each skip reason so the cause is visible).
    """
    if not isinstance(targets, TargetSet):
        raise TypeError(f"targets must be a TargetSet, got {type(targets).__name__}.")
    frame.table(weight_entity)  # validates the entity name
    initial = frame.resolve_weights(weight_entity)
    w0 = initial.values
    n_weights = len(w0)

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr: list[int] = [0]
    values: list[float] = []
    names: list[str] = []
    compiled: list[Target] = []
    skipped: list[SkippedTarget] = []

    for target in targets:
        try:
            row = _entity_row(target, frame, weight_entity)
        except (KeyError, ValueError) as exc:
            skipped.append(SkippedTarget(target=target, reason=str(exc)))
            continue
        if row.shape != (n_weights,):  # pragma: no cover - defensive
            skipped.append(
                SkippedTarget(
                    target=target,
                    reason=(
                        f"compiled row has shape {row.shape}, expected "
                        f"({n_weights},) for entity {weight_entity!r}."
                    ),
                )
            )
            continue
        if not np.isfinite(row).all():
            n_bad = int((~np.isfinite(row)).sum())
            skipped.append(
                SkippedTarget(
                    target=target,
                    reason=(
                        f"compiled row has {n_bad} non-finite value(s) (a "
                        "measure produced NaN/inf on the entity)."
                    ),
                )
            )
            continue
        indices = np.flatnonzero(row)
        if len(indices):
            index_parts.append(indices.astype(np.int64, copy=False))
            data_parts.append(row[indices].astype(np.float64, copy=False))
        indptr.append(indptr[-1] + int(len(indices)))
        values.append(target.value)
        names.append(target.row_name)
        compiled.append(target)

    if not compiled:
        detail = (
            "; ".join(f"{s.target.row_name}: {s.reason}" for s in skipped)
            or "no targets supplied"
        )
        raise ValueError(
            "No targets compiled into the constraint system "
            f"({len(skipped)} skipped): {detail}."
        )

    if data_parts:
        data = np.concatenate(data_parts)
        indices = np.concatenate(index_parts)
    else:
        data = np.empty(0, dtype=np.float64)
        indices = np.empty(0, dtype=np.int64)
    matrix = sparse.csr_array(
        (
            data,
            indices,
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(compiled), n_weights),
    )
    return CalibrationProblem(
        matrix=matrix,
        target_vector=np.asarray(values, dtype=np.float64),
        names=tuple(names),
        initial_weights=initial,
        weight_entity=weight_entity,
        targets=tuple(compiled),
        skipped=tuple(skipped),
    )
