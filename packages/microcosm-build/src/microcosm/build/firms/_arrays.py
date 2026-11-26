"""Array argument checks shared by the firm mechanics."""

from __future__ import annotations

import numpy as np


def as_1d(name: str, values: object, length: int | None = None) -> np.ndarray:
    """``values`` as a one-dimensional array, optionally of ``length`` entries."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has {len(array)} entries; expected {length}")
    return array
