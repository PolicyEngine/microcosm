"""Positive scaling of numeric columns, optionally by the sign of each value."""

from dataclasses import dataclass

import numpy as np

__all__ = ["ScaleFactor", "SignedScale", "apply_scale"]


def _positive_scale(value: float) -> float:
    factor = float(value)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Scale factors must be finite and positive.")
    return factor


@dataclass(frozen=True)
class SignedScale:
    """Separate positive factors for positive values and negative values.

    Both factors must be finite and strictly positive, so scaling never
    reverses a value's sign. Zero values remain zero.
    """

    positive: float
    negative: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive", _positive_scale(self.positive))
        object.__setattr__(self, "negative", _positive_scale(self.negative))


ScaleFactor = float | SignedScale


def apply_scale(values: np.ndarray, factor: ScaleFactor) -> np.ndarray:
    """Return a new float64 array with a scalar or sign-specific scale applied.

    The input array is not mutated. Factors must be finite and positive;
    nonfinite inputs or outputs raise ``ValueError``.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Values to scale must be finite.")
    if isinstance(factor, SignedScale):
        scales = np.where(values < 0, factor.negative, factor.positive)
    else:
        scales = _positive_scale(factor)
    with np.errstate(over="ignore", invalid="ignore"):
        result = values * scales
    if not np.isfinite(result).all():
        raise ValueError("Scaled values must be finite.")
    return np.asarray(result, dtype=np.float64)
