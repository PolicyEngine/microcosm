"""Compile targets + a Frame into a sparse linear constraint system.

:func:`build_constraint_matrix` turns a :class:`~populace.calibrate.target.TargetSet`
and a :class:`~populace.frame.Frame` into a :class:`CalibrationProblem`: a sparse
CSR matrix ``A`` (one row per compilable ``(target, period)``, one column per
record of the calibrated entity), the target vector ``b``, the row names, and the
initial weights resolved from the frame.

The matrix is the linear form of the charter's calibration objective: an
estimate of every target's aggregate is ``A @ w`` for a weight vector ``w``, and
calibration searches for the ``w`` that drives ``A @ w`` to ``b``. Multi-period
targets stack as extra rows over the *same* ``w`` — the "one weight per
trajectory" rule — because cross-sections are not calibrated independently.

A target the frame cannot compile (a measure column missing on the entity, a
``mean`` with zero denominator mass, a length mismatch) is **skipped and
reported**, never silently dropped: the returned problem carries a
:class:`SkippedTarget` for each, naming the target and the reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from populace.calibrate.target import Target, TargetSet
from populace.frame import Frame, Weights

__all__ = ["CalibrationProblem", "SkippedTarget", "build_constraint_matrix"]


@dataclass(frozen=True)
class SkippedTarget:
    """A target that could not be compiled against the frame.

    Attributes:
        target: The target that was skipped.
        reason: Human-readable explanation naming the culprit (e.g. a missing
            column, a zero denominator).
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
        target_vector: The right-hand side ``b`` of length ``n_targets``. For
            ``sum``/``count`` rows it is the target value; for a linearized
            ``mean`` row it is the value shifted by the row's offset so
            ``row @ w`` reproduces the target mean at the linearization point.
        names: Row labels (``"name@period"``), aligned to ``matrix`` rows.
        initial_weights: The :class:`~populace.frame.Weights` resolved from the
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


def _linearization_weights(
    target: Target,
    frame: Frame,
    weights: np.ndarray,
    weight_entity: str,
) -> np.ndarray:
    """Weights aligned to ``target.entity`` for the ``mean`` linearization.

    The constraint row and its ``offset`` are both built at a linearization
    point on the *target's* entity. When the target is measured on the calibrated
    entity that point is ``weights`` itself; when it is a person-level target
    collapsed onto a group, the point is the group weights broadcast onto persons
    (members share the group weight). Resolving this once keeps the row and the
    offset on the *same* point — passing the group vector to a person-length
    ``offset`` is the broadcast bug this guards against.

    Args:
        target: The target to compile.
        frame: The frame to read from.
        weights: The calibrated entity's current weights.
        weight_entity: The entity being calibrated.

    Returns:
        The weight vector aligned to ``target.entity``'s records.

    Raises:
        ValueError: If the target's entity is neither the calibrated entity nor a
            person entity nested under the calibrated group entity.
    """
    if target.entity == weight_entity:
        return np.asarray(weights, dtype=np.float64)
    person_entity = frame.schema.person_entity
    group_entities = set(frame.schema.group_entities)
    if target.entity == person_entity and weight_entity in group_entities:
        return frame._group_values_to_person(weight_entity, weights)
    raise ValueError(
        f"Target {target.name!r}: measured on entity {target.entity!r} but "
        f"calibrating {weight_entity!r}. Compilation supports a target on the "
        "calibrated entity, or a person-level target collapsed onto a group "
        f"entity persons belong to; {target.entity!r} is neither here."
    )


def _entity_row(
    target: Target,
    frame: Frame,
    entity_weights: np.ndarray,
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
        entity_weights: The linearization point aligned to ``target.entity``
            (from :func:`_linearization_weights`) — the calibrated entity's own
            weights, or the group weights broadcast onto persons for the
            cross-entity case.
        weight_entity: The entity being calibrated.

    Returns:
        The constraint row aligned to ``weight_entity``'s records.

    Raises:
        ValueError: If the target's entity is neither the calibrated entity nor
            a person entity nested under it (the only collapse supported here),
            or the ``mean`` denominator is zero.
        KeyError: If a measure/filter column is missing.
    """
    if target.entity == weight_entity:
        # The row is built against the calibrated entity's own weights.
        return target.constraint_row(frame, entity_weights)

    # Supported cross-entity case: target measured on persons, weights on a
    # group the persons belong to. ``entity_weights`` is already the group
    # weights broadcast onto persons; build the per-person row at that point,
    # then collapse to one value per group by summation (members share the
    # group weight).
    person_row = target.constraint_row(frame, entity_weights)
    positions = frame._group_positions(weight_entity)
    collapsed = np.zeros(frame.n(weight_entity), dtype=np.float64)
    np.add.at(collapsed, positions, person_row)
    return collapsed


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

    rows: list[np.ndarray] = []
    values: list[float] = []
    names: list[str] = []
    compiled: list[Target] = []
    skipped: list[SkippedTarget] = []

    for target in targets:
        try:
            # The row and its offset must share one linearization point on the
            # target's *own* entity; resolving it once is the fix for person-
            # entity mean/offset targets on multi-person frames (offset was
            # otherwise handed the group vector and broadcast against persons).
            entity_weights = _linearization_weights(target, frame, w0, weight_entity)
            row = _entity_row(target, frame, entity_weights, weight_entity)
            offset = target.offset(frame, entity_weights)
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
        rows.append(row)
        values.append(target.value + offset)
        names.append(target.row_name)
        compiled.append(target)

    if not rows:
        detail = (
            "; ".join(f"{s.target.row_name}: {s.reason}" for s in skipped)
            or "no targets supplied"
        )
        raise ValueError(
            "No targets compiled into the constraint system "
            f"({len(skipped)} skipped): {detail}."
        )

    matrix = sparse.csr_array(np.vstack(rows))
    return CalibrationProblem(
        matrix=matrix,
        target_vector=np.asarray(values, dtype=np.float64),
        names=tuple(names),
        initial_weights=initial,
        weight_entity=weight_entity,
        targets=tuple(compiled),
        skipped=tuple(skipped),
    )
