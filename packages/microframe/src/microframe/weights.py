"""Typed weight vectors with kernel-enforced conservation invariants.

A :class:`Weights` is an immutable float64 vector tagged with a
:class:`WeightKind` declaring its provenance. Construction validates the
invariants every weight vector in the stack must satisfy (finite,
non-negative, not all zero), so a corrupted vector can never enter a bundle.
Weight kinds only move forward — ``design -> importance -> calibrated`` —
and :func:`assert_kind_transition` is the single authority for that rule.
"""

import enum
from dataclasses import dataclass

import numpy as np

__all__ = ["WeightKind", "Weights", "assert_kind_transition"]


class WeightKind(enum.Enum):
    """Provenance of a weight vector.

    Attributes:
        DESIGN: Survey design weights carried from the source data.
        IMPORTANCE: Sampling-importance weights produced by pool assembly
            (e.g. oversampling strata where support is scarce).
        CALIBRATED: Weights produced by a calibration stage to hit declared
            targets. The terminal kind: once calibrated, weights never revert.
    """

    DESIGN = "design"
    IMPORTANCE = "importance"
    CALIBRATED = "calibrated"


_KIND_RANK: dict[WeightKind, int] = {
    WeightKind.DESIGN: 0,
    WeightKind.IMPORTANCE: 1,
    WeightKind.CALIBRATED: 2,
}


def assert_kind_transition(old: WeightKind, new: WeightKind) -> None:
    """Validate that a weight-kind transition moves forward.

    Allowed transitions follow the pipeline order
    ``design -> importance -> calibrated``; replacing weights with the same
    kind is allowed (e.g. recalibration). Moving backward is a kernel error.

    Args:
        old: Kind of the weights being replaced.
        new: Kind of the replacement weights.

    Raises:
        ValueError: If ``new`` ranks below ``old`` in the pipeline order.
    """
    if _KIND_RANK[new] < _KIND_RANK[old]:
        raise ValueError(
            f"Weight kind cannot move backward: {old.value!r} -> {new.value!r}. "
            "Allowed direction is design -> importance -> calibrated."
        )


@dataclass(frozen=True)
class Weights:
    """An immutable, validated weight vector for one entity.

    Attributes:
        values: One-dimensional float64 array, one weight per entity row. The
            stored array is a read-only copy of the input.
        kind: Provenance of the vector (:class:`WeightKind`).

    Raises:
        TypeError: If ``kind`` is not a :class:`WeightKind`.
        ValueError: If ``values`` is not one-dimensional, is empty, contains
            NaN or infinity, contains negative values, or is entirely zero.
    """

    values: np.ndarray
    kind: WeightKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WeightKind):
            raise TypeError(
                f"Weights.kind must be a WeightKind, got {type(self.kind).__name__}."
            )
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError(
                f"Weights.values must be one-dimensional, got shape {values.shape}."
            )
        if values.size == 0:
            raise ValueError("Weights.values cannot be empty.")
        if not np.isfinite(values).all():
            bad = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"Weights must be finite; found {bad} NaN/inf value(s)."
            )
        if (values < 0).any():
            bad = int((values < 0).sum())
            raise ValueError(
                f"Weights must be non-negative; found {bad} negative value(s)."
            )
        if not (values > 0).any():
            raise ValueError("Weights cannot be all zero.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return int(self.values.size)

    @property
    def total(self) -> float:
        """Total mass of the vector (the sum of all weights)."""
        return float(self.values.sum())

    def with_values(self, values: np.ndarray, kind: WeightKind) -> "Weights":
        """Return a new validated :class:`Weights` with the given values and kind.

        Args:
            values: Replacement weight values (validated on construction).
            kind: Kind of the new vector. Kind-transition rules are enforced
                by the bundle, not here.

        Returns:
            A new :class:`Weights` instance.
        """
        return Weights(values=np.asarray(values, dtype=np.float64), kind=kind)

    def assert_mass_conserved(self, other: "Weights", rtol: float = 1e-9) -> None:
        """Raise unless ``other`` carries the same total mass as this vector.

        Args:
            other: Weight vector to compare against.
            rtol: Relative tolerance for the totals comparison.

        Raises:
            ValueError: If the totals differ beyond ``rtol``. The message
                carries both totals.
        """
        a = self.total
        b = other.total
        if not np.isclose(a, b, rtol=rtol, atol=0.0):
            raise ValueError(
                f"Weight mass not conserved: total {a!r} ({self.kind.value}) "
                f"!= total {b!r} ({other.kind.value}) at rtol={rtol}."
            )
