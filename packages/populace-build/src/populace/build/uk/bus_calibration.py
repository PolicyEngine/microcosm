"""Executable DfT level-calibration for the UK bus-spending variables.

The imputation stages (``bus_source_stages``) place ``bus_fare_spending`` and
``bus_subsidy_spending`` on the household table, but a survey imputation does
not reproduce the Department for Transport (DfT) national totals on its own —
left unanchored, fare spending lands roughly twice the DfT fare total and
subsidy well below the DfT net-support total.

This module applies the same correction the incumbent enhanced-FRS build uses:
a per-variable multiplicative **value scaling** so each variable's weighted
total equals its DfT target (``bus_calibration_targets.UK_BUS_TARGET_REGISTRY``).
Scaling the values changes the level only; it leaves which households spend and
the relative shape of the distribution untouched (the spender share is set by
the imputation, not by this step, exactly as in the incumbent build).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from populace.build.uk.bus_calibration_targets import UK_BUS_TARGET_REGISTRY


def uk_bus_targets() -> dict[str, float]:
    """The DfT weighted-total target for each bus variable, by column name."""
    return {
        spec.measure: float(spec.value)
        for spec in UK_BUS_TARGET_REGISTRY.specs
        if spec.measure is not None
    }


def calibrate_bus_spending_levels(
    household: pd.DataFrame,
    *,
    weight_column: str = "household_weight",
    targets: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scale each bus-spending column so its weighted total equals its target.

    Mirrors the incumbent ``calibrate_bus_fare_spending`` /
    ``calibrate_bus_subsidy_spending`` step: ``scale = target / actual`` then
    ``column *= scale``. Pure value scaling — the set of spending households
    and the distribution's shape are unchanged.

    Args:
        household: Household table carrying ``weight_column`` and every target
            column.
        weight_column: Survey weight column used for the weighted totals.
        targets: ``column -> target weighted total``. Defaults to the DfT
            registry totals (:func:`uk_bus_targets`).

    Returns:
        ``(calibrated_household, scales)`` — a new table (the input is not
        mutated) and the multiplicative scale applied to each column.

    Raises:
        KeyError: If the weight column or a target column is missing.
        ValueError: If a target column's current weighted total is not
            positive (cannot scale a zero/negative aggregate to a target).
    """
    if targets is None:
        targets = uk_bus_targets()
    if weight_column not in household.columns:
        raise KeyError(f"household table has no weight column {weight_column!r}.")

    calibrated = household.copy()
    weights = calibrated[weight_column].to_numpy(dtype=float)
    scales: dict[str, float] = {}
    for column, target in targets.items():
        if column not in calibrated.columns:
            raise KeyError(f"household table has no target column {column!r}.")
        values = calibrated[column].to_numpy(dtype=float)
        actual = float(np.sum(values * weights))
        if not actual > 0:
            raise ValueError(
                f"cannot calibrate {column!r}: weighted aggregate is {actual} "
                "(must be positive)."
            )
        scale = float(target) / actual
        calibrated[column] = values * scale
        scales[column] = scale
    return calibrated, scales


__all__ = ["calibrate_bus_spending_levels", "uk_bus_targets"]
