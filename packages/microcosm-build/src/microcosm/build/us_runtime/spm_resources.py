"""US SPM release diagnostics.

These helpers summarize modeled SPM resources against thresholds. They are
validation diagnostics, not calibration targets: official CPS SPM references can
travel in the output metadata, but callers must not use this module to turn CPS
SPM rates into hard target rows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["spm_resource_diagnostics", "write_spm_resource_diagnostics"]


def _weighted_sum(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights))


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> dict[str, float | None]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {str(q): None for q in quantiles}
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    return {
        str(q): float(values[np.searchsorted(cumulative, q * total, side="left")])
        for q in quantiles
    }


def spm_resource_diagnostics(
    frame: pd.DataFrame,
    *,
    resource_column: str,
    threshold_column: str,
    weight_column: str,
    group_columns: tuple[str, ...] = (),
    validation_references: Mapping[str, object] | None = None,
) -> dict:
    """Return SPM below-threshold and negative-resource diagnostics for a frame.

    Args:
        frame: One row per SPM unit or person, already aligned to resources,
            thresholds, and weights.
        resource_column: Modeled SPM resources.
        threshold_column: SPM threshold.
        weight_column: Person or SPM-unit weight.
        group_columns: Optional subgroup columns for below-threshold-rate breakdowns.
        validation_references: Metadata about official/public validation
            references. These are copied under ``validation_references`` and
            marked validation-only.
    """
    required = (resource_column, threshold_column, weight_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"SPM diagnostic frame missing required columns {missing}.")

    resources = frame[resource_column].to_numpy(dtype=float)
    thresholds = frame[threshold_column].to_numpy(dtype=float)
    weights = frame[weight_column].to_numpy(dtype=float)
    valid = np.isfinite(resources) & np.isfinite(thresholds) & np.isfinite(weights)
    valid &= weights > 0
    if not np.any(valid):
        raise ValueError("SPM diagnostic frame has no positive-weight finite rows.")

    below_threshold = (resources < thresholds) & valid
    negative = (resources < 0) & valid
    population = float(weights[valid].sum())
    below_spm_threshold_count = _weighted_sum(
        below_threshold[valid].astype(float), weights[valid]
    )
    negative_count = _weighted_sum(negative[valid].astype(float), weights[valid])

    groups: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for column in group_columns:
        if column not in frame.columns:
            raise ValueError(f"SPM diagnostic group column {column!r} is missing.")
        groups[column] = {}
        for value, subframe in frame.loc[valid].groupby(column, dropna=False):
            sub = spm_resource_diagnostics(
                subframe,
                resource_column=resource_column,
                threshold_column=threshold_column,
                weight_column=weight_column,
            )
            groups[column][str(value)] = {
                "population": sub["population"],
                "below_spm_threshold_count": sub["below_spm_threshold_count"],
                "below_spm_threshold_rate": sub["below_spm_threshold_rate"],
            }

    payload = {
        "schema_version": 1,
        "classification": "validation_only",
        "population": population,
        "below_spm_threshold_count": below_spm_threshold_count,
        "below_spm_threshold_rate": below_spm_threshold_count / population,
        "resource_quantiles": _weighted_quantile(
            resources[valid], weights[valid], (0.1, 0.25, 0.5, 0.75, 0.9)
        ),
        "negative_resources": {
            "count": negative_count,
            "share": negative_count / population,
            "minimum": float(np.min(resources[valid])),
            "total_negative_mass": float(
                np.sum(resources[negative] * weights[negative])
            ),
        },
        "groups": groups,
        "validation_references": dict(validation_references or {}),
    }
    return payload


def write_spm_resource_diagnostics(
    payload: Mapping[str, object], path: Path | str
) -> Path:
    """Write an SPM diagnostics payload as strict JSON."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
