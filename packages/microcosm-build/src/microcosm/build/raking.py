"""Generic iterative proportional fitting helpers for source stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MarginSpec", "iterative_proportional_fit"]


@dataclass(frozen=True)
class MarginSpec:
    """Mean targets for one categorical margin.

    ``targets`` maps category value -> output column -> target mean. The helper
    deliberately works in means because the UK LCFS/NEED application receives
    published mean kWh/spend cells, but the utility is country-neutral.
    """

    column: str
    targets: Mapping[object, Mapping[str, float]]


def iterative_proportional_fit(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    margins: Sequence[MarginSpec],
    iterations: int,
    weight_column: str | None = None,
    fail_on_unattainable: bool = False,
) -> pd.DataFrame:
    """Scale columns in-place-by-copy to match declared cell means.

    Empty cells and categories absent from the declared targets are skipped.
    Populated zero-current-mean cells with positive targets are recorded in the
    returned frame's ``raking_zero_current_cells`` evidence attribute. Callers
    that require fail-closed behavior can set ``fail_on_unattainable``.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not columns:
        raise ValueError("at least one column is required")

    result = frame.copy()
    for column in columns:
        if column not in result:
            raise KeyError(f"frame is missing raked column {column!r}")
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    if weight_column is not None and weight_column not in result:
        raise KeyError(f"frame is missing weight column {weight_column!r}")

    weights = None
    if weight_column is not None:
        weights = pd.to_numeric(result[weight_column], errors="coerce").fillna(0.0)
        if (weights < 0).any():
            raise ValueError("raking weights must be nonnegative")

    zero_current_cells: list[dict[str, object]] = []
    for _ in range(iterations):
        for margin in margins:
            if margin.column not in result:
                raise KeyError(f"frame is missing margin column {margin.column!r}")
            for category, target_by_column in margin.targets.items():
                mask = result[margin.column] == category
                if not bool(mask.any()):
                    continue
                for column in columns:
                    if column not in target_by_column:
                        continue
                    current = _cell_mean(
                        result.loc[mask, column],
                        None if weights is None else weights.loc[mask],
                    )
                    target = float(target_by_column[column])
                    if not np.isfinite(target) or target < 0:
                        raise ValueError(
                            f"target for {margin.column!r}={category!r}, "
                            f"{column!r} must be finite and nonnegative"
                        )
                    if current <= 0 or not np.isfinite(current):
                        if target > 0:
                            evidence = {
                                "margin": margin.column,
                                "category": category,
                                "column": column,
                                "target": target,
                            }
                            zero_current_cells.append(evidence)
                            if fail_on_unattainable:
                                raise ValueError(
                                    f"cannot rake {margin.column!r}={category!r} "
                                    f"for {column!r}: current mean is "
                                    f"zero/non-finite but target is {target}."
                                )
                        continue
                    result.loc[mask, column] *= target / current
    result.attrs["raking_zero_current_cells"] = tuple(zero_current_cells)
    return result


def _cell_mean(values: pd.Series, weights: pd.Series | None) -> float:
    data = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if weights is None:
        return float(data.mean()) if len(data) else 0.0
    w = weights.to_numpy(dtype=float)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    return float(np.dot(data, w) / total)
