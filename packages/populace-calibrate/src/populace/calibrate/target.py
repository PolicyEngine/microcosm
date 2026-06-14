"""Targets: the declarative facts calibration combines into weights.

A :class:`Target` is one calibration *fact* — a known population aggregate
(a control total, a count, an average) that the calibrated weights should
reproduce. Each target names which entity's records it measures, how to read a
per-record value off them (a column or a callable), how to aggregate that value
(``sum`` / ``count`` / ``mean``), and the value to hit. Optionally it carries a
period, a tolerance, and a provenance string.

The defining property — the one that makes calibration a *linear* problem — is
that every target is a **linear constraint on the weight vector** of its entity:

- ``sum`` of column ``c`` on entity ``e`` is the constraint row of per-record
  values of ``c`` (aligned to ``e``'s weight vector): ``row @ w`` estimates the
  total, so ``row = values(c)``.
- ``count`` is the indicator of the records that pass the target's filter (1 per
  passing record, 0 otherwise): ``row @ w`` estimates the weighted count.
- ``mean`` is a *ratio* of two sums — ``sum(value) / sum(filter)`` — which is not
  linear in ``w``. It is linearized about the current weights into a single row
  so the whole problem stays one sparse matrix; see
  :meth:`Target.constraint_row` for the exact linearization and its caveat.

:class:`TargetSet` is an ordered, immutable container of targets. The matrix
compiler (:mod:`populace.calibrate.matrix`) turns a target set plus a
:class:`~populace.frame.Frame` into the sparse system the solver consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from populace.frame import Frame

__all__ = ["Target", "TargetSet", "AGGREGATIONS"]

#: The aggregations a target may declare. ``sum`` and ``count`` are exactly
#: linear in the weights; ``mean`` is a ratio linearized about the current
#: weights (see :meth:`Target.constraint_row`).
AGGREGATIONS: tuple[str, ...] = ("sum", "count", "mean")

#: Type of a measure callable: given a frame, return one float per record of the
#: target's entity (aligned to that entity's weight vector).
MeasureFn = Callable[[Frame], np.ndarray]


@dataclass(frozen=True)
class Target:
    """One calibration fact: a known aggregate the weights should reproduce.

    A target is a linear constraint on the calibrated weights of its
    ``entity``. The constraint row depends on the ``aggregation``:

    - ``sum``: the per-record values of ``measure`` (a column or callable).
    - ``count``: the indicator of records passing ``filter`` (``measure`` is
      ignored for ``count`` and may be omitted; a missing ``measure`` means
      "count every record").
    - ``mean``: ratio ``sum(measure) / sum(filter)``, linearized about the
      current weights (see :meth:`constraint_row`).

    Attributes:
        name: Unique label for the target (used in diagnostics and the
            compiled matrix row index).
        entity: Entity whose weight vector this target constrains (e.g.
            ``"household"`` or ``"person"``). The measure and filter are read
            on this entity's table.
        measure: How to read the per-record quantity: a column name (resolved
            on ``entity``'s table) or a callable ``measure(frame) -> ndarray``
            returning one value per ``entity`` record. Optional for ``count``.
        aggregation: One of :data:`AGGREGATIONS`.
        value: The target value the aggregate should reproduce.
        period: The period this target applies to. A scalar tag (an int year,
            a string label, or the default ``0`` for a single-period frame);
            ``(target, period)`` pairs become distinct constraint rows over the
            *same* weight vector — the charter's "one weight per trajectory".
            The frame is responsible for carrying period-specific columns;
            ``period`` is metadata the compiler stacks on, not a column lookup.
        tolerance: Optional absolute tolerance for "is this target hit". Used
            by diagnostics, not by the loss (the loss is the bounded relative
            error over all targets jointly).
        filter: Optional column name or callable producing a boolean (or
            0/1) per-record mask. For ``count`` it selects which records to
            count; for ``mean`` it selects the denominator population; for
            ``sum`` it gates the summed value. ``None`` means "all records".
        source: Free-text provenance (e.g. ``"ACS 2024 table B19001"``).

    Raises:
        ValueError: If ``name`` or ``entity`` is empty, ``aggregation`` is not
            one of :data:`AGGREGATIONS`, ``value`` is not finite, ``tolerance``
            is given and is negative, or ``measure`` is omitted for a
            non-``count`` aggregation.
    """

    name: str
    entity: str
    measure: str | MeasureFn | None = None
    aggregation: str = "sum"
    value: float = 0.0
    period: int | str = 0
    tolerance: float | None = None
    filter: str | MeasureFn | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target.name must be a non-empty label.")
        if not self.entity:
            raise ValueError(f"Target {self.name!r}: entity must be non-empty.")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Target {self.name!r}: aggregation must be one of "
                f"{AGGREGATIONS}, got {self.aggregation!r}."
            )
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
        if self.aggregation != "count" and self.measure is None:
            raise ValueError(
                f"Target {self.name!r}: aggregation {self.aggregation!r} needs a "
                "measure (a column name or callable); only 'count' may omit it."
            )

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

    def constraint_row(self, frame: Frame, weights: np.ndarray) -> np.ndarray:
        """Build this target's constraint row over ``entity``'s weights.

        The row ``r`` is such that ``r @ w`` estimates the target's aggregate
        under weights ``w``. By aggregation:

        - ``sum``: ``r = measure * filter`` (the filter gates the summed value;
          with no filter it is the bare measure).
        - ``count``: ``r = filter`` (1 per passing record).
        - ``mean``: the aggregate is the ratio ``S / D`` with
          ``S = sum(measure * filter * w)`` and ``D = sum(filter * w)``. A ratio
          is not linear in ``w``, so it is linearized about the supplied
          ``weights`` ``w0`` by a first-order expansion:

          ``S/D ≈ S0/D0 + (1/D0)·(measure*filter - S0/D0)·filter · (w - w0)``,

          which gives the constant row ``r = (filter / D0)·(measure - S0/D0)``
          whose ``r @ (w - w0)`` is the change in the mean, plus a constant
          offset folded into the compiled target value (the matrix compiler
          adjusts the right-hand side so ``r @ w`` hits ``value`` at the
          linearization point). The approximation is exact to first order and
          good while calibration does not move the denominator mass sharply; a
          target family that needs the mean held tightly under large mass moves
          should be expressed as the two underlying ``sum`` rows (numerator and
          denominator) instead.

        Args:
            frame: The frame to read measure/filter values from.
            weights: The current weights of ``entity`` (the linearization
                point for ``mean``; unused for ``sum``/``count``).

        Returns:
            The constraint row as a float64 vector aligned to ``entity``'s
            weight vector.

        Raises:
            ValueError: If a ``mean`` target's denominator mass is zero at the
                linearization point (the ratio is undefined).
            KeyError: From :meth:`_record_values` if a column is missing.
        """
        n = frame.n(self.entity)
        filter_mask = self._filter_mask(frame, n)
        if self.aggregation == "count":
            return filter_mask
        values = self._record_values(frame, self.measure, n)
        if self.aggregation == "sum":
            return values * filter_mask
        # mean: linearize the ratio S/D about the supplied weights.
        weights = np.asarray(weights, dtype=np.float64)
        denominator = float((filter_mask * weights).sum())
        if denominator <= 0:
            raise ValueError(
                f"Target {self.name!r}: mean is undefined — its filtered "
                "denominator mass is zero at the current weights. Either widen "
                "the filter or express the mean as two sum targets."
            )
        numerator = float((values * filter_mask * weights).sum())
        current_mean = numerator / denominator
        return (filter_mask / denominator) * (values - current_mean)

    def achieved_value(self, frame: Frame, weights: np.ndarray) -> float:
        """The *true* aggregate under ``weights`` (no linearization).

        Unlike ``constraint_row(frame, w0) @ w`` — which for a ``mean`` is only
        the first-order linearized value about ``w0`` — this evaluates the
        aggregate exactly at ``weights``:

        - ``sum``: ``sum(measure * filter * weights)``.
        - ``count``: ``sum(filter * weights)``.
        - ``mean``: the true ratio ``sum(measure*filter*weights) /
          sum(filter*weights)``.

        Diagnostics use this so a ``mean`` target's ``relative_error`` and
        ``within_tolerance`` describe the achieved ratio, not the linearization
        (which can read as a perfect hit after a large mass move).

        Args:
            frame: The frame to read measure/filter values from.
            weights: The weights of ``entity`` to evaluate at (aligned to
                ``entity``'s records).

        Returns:
            The true weighted aggregate.

        Raises:
            ValueError: If a ``mean`` target's filtered denominator mass is zero
                under ``weights`` (the ratio is undefined).
            KeyError: From :meth:`_record_values` if a column is missing.
        """
        n = frame.n(self.entity)
        filter_mask = self._filter_mask(frame, n)
        weights = np.asarray(weights, dtype=np.float64)
        if self.aggregation == "count":
            return float((filter_mask * weights).sum())
        values = self._record_values(frame, self.measure, n)
        if self.aggregation == "sum":
            return float((values * filter_mask * weights).sum())
        # mean: the true ratio under these weights.
        denominator = float((filter_mask * weights).sum())
        if denominator <= 0:
            raise ValueError(
                f"Target {self.name!r}: mean is undefined — its filtered "
                "denominator mass is zero under the given weights."
            )
        return float((values * filter_mask * weights).sum()) / denominator

    def offset(self, frame: Frame, weights: np.ndarray) -> float:
        """Constant the compiler adds to ``value`` to anchor a ``mean`` row.

        ``sum`` and ``count`` rows satisfy ``r @ w == aggregate`` directly, so
        the offset is zero and the right-hand side is ``value``. The linearized
        ``mean`` row instead satisfies ``r @ (w - w0) == mean(w) - mean(w0)``, so
        ``r @ w == value`` must target ``value`` shifted by ``r @ w0 - mean(w0)``:
        this returns that shift, and the compiler sets the row's right-hand side
        to ``value + offset``.

        Args:
            frame: The frame the row was built against.
            weights: The same linearization point passed to
                :meth:`constraint_row`.

        Returns:
            ``0.0`` for ``sum``/``count``; for ``mean`` the constant
            ``r @ w0 - mean(w0)`` that makes ``r @ w == value`` reproduce the
            target mean at the linearization point.
        """
        if self.aggregation != "mean":
            return 0.0
        n = frame.n(self.entity)
        filter_mask = self._filter_mask(frame, n)
        values = self._record_values(frame, self.measure, n)
        weights = np.asarray(weights, dtype=np.float64)
        denominator = float((filter_mask * weights).sum())
        numerator = float((values * filter_mask * weights).sum())
        current_mean = numerator / denominator
        row = (filter_mask / denominator) * (values - current_mean)
        return float(row @ weights) - current_mean


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
