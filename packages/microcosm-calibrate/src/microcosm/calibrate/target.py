"""Targets: the declarative facts calibration combines into weights.

A :class:`Target` is one calibration *fact* — a known population aggregate
that the calibrated weights should reproduce. Each target names which entity's
records it measures, how to read a per-record value off them (a column or a
callable), and the value to hit. Optionally it carries a period, a tolerance,
and a provenance string.

The defining property — the one that makes calibration a *linear* problem — is
that every target is a **sum constraint on the weight vector** of its entity:
``sum`` of column ``c`` on entity ``e`` is the constraint row of per-record
values of ``c`` (aligned to ``e``'s weight vector): ``row @ w`` estimates the
total, so ``row = values(c)``. Count-like facts are represented by summing an
indicator/count column prepared before calibration.

:class:`TargetSet` is an ordered, immutable container of targets. The matrix
compiler (:mod:`microcosm.calibrate.matrix`) turns a target set plus a
:class:`~microcosm.frame.Frame` into the sparse system the solver consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from microcosm.frame import Frame

__all__ = ["Target", "TargetSet"]

#: Type of a measure callable: given a frame, return one float per record of the
#: target's entity (aligned to that entity's weight vector).
MeasureFn = Callable[[Frame], np.ndarray]


@dataclass(frozen=True)
class Target:
    """One calibration fact: a known aggregate the weights should reproduce.

    A target is a sum constraint on the calibrated weights of its ``entity``.
    Count-like facts use a prepared indicator/count measure column and the same
    sum machinery.

    Attributes:
        name: Unique label for the target (used in diagnostics and the
            compiled matrix row index).
        entity: Entity whose weight vector this target constrains (e.g.
            ``"household"`` or ``"person"``). The measure and filter are read
            on this entity's table.
        measure: How to read the per-record quantity: a column name (resolved
            on ``entity``'s table) or a callable ``measure(frame) -> ndarray``
            returning one value per ``entity`` record.
        value: The target value the aggregate should reproduce.
        period: The period this target applies to. A scalar tag (an int year,
            a string label, or the default ``0`` for a single-period frame);
            ``(target, period)`` pairs become distinct constraint rows over the
            *same* weight vector — the charter's "one weight per trajectory".
            The frame is responsible for carrying period-specific columns;
            ``period`` is metadata the compiler stacks on, not a column lookup.
        tolerance: Optional absolute tolerance for "is this target hit". Used
            by diagnostics, not by the capped weighted-MAPE loss.
        filter: Optional column name or callable producing a boolean (or
            0/1) per-record mask. It gates the summed value. ``None`` means
            "all records".
        source: Free-text provenance (e.g. ``"ACS 2024 table B19001"``).
        metadata: Declarative target semantics used by release gates. For
            example, a JCT tax-expenditure target records that the row is an
            ``income_tax`` delta from a simple neutralization reform.

    Raises:
        ValueError: If ``name`` or ``entity`` is empty, ``value`` is not finite,
            ``tolerance`` is given and is negative, or ``measure`` is omitted.
    """

    name: str
    entity: str
    measure: str | MeasureFn | None = None
    value: float = 0.0
    period: int | str = 0
    tolerance: float | None = None
    filter: str | MeasureFn | None = None
    source: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target.name must be a non-empty label.")
        if not self.entity:
            raise ValueError(f"Target {self.name!r}: entity must be non-empty.")
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError(
                f"Target {self.name!r}: value must be finite, got {self.value!r}."
            )
        object.__setattr__(self, "value", value)
        if self.tolerance is not None:
            tolerance = float(self.tolerance)
            if tolerance < 0 or not np.isfinite(tolerance):
                raise ValueError(
                    f"Target {self.name!r}: tolerance must be a finite "
                    f"non-negative number, got {self.tolerance!r}."
                )
            object.__setattr__(self, "tolerance", tolerance)
        if self.measure is None or self.measure == "":
            raise ValueError(
                f"Target {self.name!r}: measure is required; count-like facts "
                "must be represented as sums of prepared indicator columns."
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"Target {self.name!r}: metadata must be a mapping from "
                f"str to str, got {type(self.metadata).__name__}."
            )
        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        bad_metadata = sorted(
            key for key, value in metadata.items() if not key or not value
        )
        if bad_metadata:
            raise ValueError(
                f"Target {self.name!r}: metadata keys and values must be "
                f"non-empty strings; bad keys {bad_metadata}."
            )
        object.__setattr__(self, "metadata", metadata)

    @property
    def key(self) -> tuple[str, int | str]:
        """The ``(name, period)`` pair: this target's identity in the matrix.

        A target is one constraint *per period*, so the row index of the
        compiled matrix is keyed by this pair, not by name alone.
        """
        return (self.name, self.period)

    @property
    def row_name(self) -> str:
        """Human-readable row label ``"name@period"`` for diagnostics."""
        return f"{self.name}@{self.period}"

    def _record_values(
        self, frame: Frame, measure: str | MeasureFn, n: int
    ) -> np.ndarray:
        """Resolve a measure to a length-``n`` float vector on the entity.

        Args:
            frame: The frame to read from.
            measure: Column name (read on ``self.entity``) or callable.
            n: Expected length (the entity's record count).

        Returns:
            A float64 vector of length ``n``.

        Raises:
            KeyError: If a column measure names a column absent from the
                entity's table.
            ValueError: If a callable returns the wrong length or a column
                lives on a different entity.
        """
        if callable(measure):
            values = np.asarray(measure(frame), dtype=np.float64)
        else:
            table = frame.table(self.entity)
            if measure not in table.columns:
                raise KeyError(
                    f"Target {self.name!r}: measure column {measure!r} not on "
                    f"the {self.entity!r} table (columns: "
                    f"{list(table.columns)[:8]}...)."
                )
            values = table[measure].to_numpy(dtype=np.float64)
        if values.shape != (n,):
            raise ValueError(
                f"Target {self.name!r}: measure produced shape {values.shape}, "
                f"expected ({n},) to align with the {self.entity!r} weights."
            )
        return values

    def _filter_mask(self, frame: Frame, n: int) -> np.ndarray:
        """Resolve the filter to a length-``n`` float mask (1.0 / 0.0).

        ``None`` filter means every record passes.
        """
        if self.filter is None:
            return np.ones(n, dtype=np.float64)
        mask = self._record_values(frame, self.filter, n)
        return (mask != 0).astype(np.float64)

    def constraint_row(self, frame: Frame) -> np.ndarray:
        """Build this target's constraint row over ``entity``'s weights.

        The row ``r`` is such that ``r @ w`` estimates the target's weighted sum
        under weights ``w``: ``r = measure * filter``. With no filter it is the
        bare measure.

        Args:
            frame: The frame to read measure/filter values from.
        Returns:
            The constraint row as a float64 vector aligned to ``entity``'s
            weight vector.

        Raises:
            KeyError: From :meth:`_record_values` if a column is missing.
        """
        n = frame.n(self.entity)
        filter_mask = self._filter_mask(frame, n)
        values = self._record_values(frame, self.measure, n)
        return values * filter_mask

    def achieved_value(self, frame: Frame, weights: np.ndarray) -> float:
        """The *true* aggregate under ``weights`` (no linearization).

        This evaluates ``sum(measure * filter * weights)`` exactly at the given
        weights.

        Args:
            frame: The frame to read measure/filter values from.
            weights: The weights of ``entity`` to evaluate at (aligned to
                ``entity``'s records).

        Returns:
            The true weighted aggregate.

        Raises:
            KeyError: From :meth:`_record_values` if a column is missing.
        """
        n = frame.n(self.entity)
        filter_mask = self._filter_mask(frame, n)
        weights = np.asarray(weights, dtype=np.float64)
        values = self._record_values(frame, self.measure, n)
        return float((values * filter_mask * weights).sum())


@dataclass(frozen=True)
class TargetSet:
    """An ordered, immutable collection of :class:`Target` facts.

    Attributes:
        targets: The targets, in order. Duplicate ``(name, period)`` keys are
            rejected: a single fact cannot be declared twice.

    Raises:
        ValueError: If two targets share a ``(name, period)`` key.
    """

    targets: tuple[Target, ...] = field(default_factory=tuple)

    def __init__(self, targets: Iterable[Target] = ()) -> None:
        materialized = tuple(targets)
        seen: dict[tuple[str, int | str], Target] = {}
        for target in materialized:
            if not isinstance(target, Target):
                raise TypeError(
                    f"TargetSet entries must be Target instances, got "
                    f"{type(target).__name__}."
                )
            if target.key in seen:
                raise ValueError(
                    f"Duplicate target key {target.key!r}: a (name, period) "
                    "pair may be declared only once."
                )
            seen[target.key] = target
        object.__setattr__(self, "targets", materialized)

    def __iter__(self) -> Iterator[Target]:
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Target:
        return self.targets[index]

    @property
    def entities(self) -> tuple[str, ...]:
        """Distinct entities the targets constrain, in first-seen order."""
        seen: dict[str, None] = {}
        for target in self.targets:
            seen.setdefault(target.entity, None)
        return tuple(seen)

    @property
    def periods(self) -> tuple[int | str, ...]:
        """Distinct periods present, in first-seen order."""
        seen: dict[int | str, None] = {}
        for target in self.targets:
            seen.setdefault(target.period, None)
        return tuple(seen)
