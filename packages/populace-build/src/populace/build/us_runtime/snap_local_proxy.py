"""US SNAP local proxy release diagnostics.

ACS S2201 congressional-district SNAP household estimates are validation-only
references. These helpers compare district-level Populace SNAP household
receipt/support against those references without promoting them into calibration
targets.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "SNAP_LOCAL_PROXY_SCHEMA_VERSION",
    "snap_local_proxy_diagnostics",
    "write_snap_local_proxy_diagnostics",
]

SNAP_LOCAL_PROXY_SCHEMA_VERSION = 1
SNAP_LOCAL_PROXY_FAMILY = "snap_local_proxy"


def _finite(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _effective_sample_size(weights: np.ndarray) -> float | None:
    if weights.size == 0:
        return None
    total = float(weights.sum())
    squared = float(np.square(weights).sum())
    if total <= 0 or squared <= 0:
        return None
    return total * total / squared


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str | None, ...],
    *,
    label: str,
) -> None:
    missing = [column for column in columns if column and column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns {missing}.")


def _reference_by_district(
    acs_reference: pd.DataFrame | None,
    *,
    district_column: str,
    snap_households_column: str,
    snap_households_moe_column: str | None,
) -> dict[str, dict[str, float | None]] | None:
    if acs_reference is None:
        return None
    _require_columns(
        acs_reference,
        (district_column, snap_households_column, snap_households_moe_column),
        label="ACS SNAP local proxy reference",
    )
    duplicated = acs_reference[district_column].duplicated(keep=False)
    if duplicated.any():
        examples = acs_reference.loc[duplicated, district_column].head(5).tolist()
        raise ValueError(f"ACS reference has duplicate district rows: {examples}.")

    references: dict[str, dict[str, float | None]] = {}
    for _, row in acs_reference.iterrows():
        district = str(row[district_column])
        estimate = pd.to_numeric(row[snap_households_column], errors="coerce")
        moe = (
            pd.to_numeric(row[snap_households_moe_column], errors="coerce")
            if snap_households_moe_column
            else None
        )
        references[district] = {
            "acs_snap_households": _finite(estimate),
            "acs_snap_households_moe": _finite(moe),
        }
    return references


def _group_state(
    group: pd.DataFrame,
    *,
    state_column: str | None,
    flags: list[str],
) -> str | None:
    if state_column is None:
        return None
    values = tuple(sorted(str(value) for value in group[state_column].dropna().unique()))
    if not values:
        return None
    if len(values) > 1:
        flags.append("mixed_state")
        return None
    return values[0]


def _group_state_error(
    group: pd.DataFrame,
    *,
    state_snap_relative_error_column: str | None,
) -> float | None:
    if state_snap_relative_error_column is None:
        return None
    values = pd.to_numeric(
        group[state_snap_relative_error_column], errors="coerce"
    ).dropna()
    if values.empty:
        return None
    return _finite(values.iloc[0])


def snap_local_proxy_diagnostics(
    household_frame: pd.DataFrame,
    *,
    district_column: str,
    weight_column: str,
    snap_receipt_column: str,
    snap_amount_column: str | None = None,
    state_column: str | None = None,
    state_snap_relative_error_column: str | None = None,
    acs_reference: pd.DataFrame | None = None,
    acs_district_column: str | None = None,
    acs_snap_households_column: str = "acs_snap_households",
    acs_snap_households_moe_column: str | None = None,
    low_positive_sample_threshold: int = 30,
    low_positive_ess_threshold: float = 10.0,
    state_outlier_abs_relative_error: float = 0.10,
    validation_source: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Return validation-only SNAP local proxy diagnostics.

    Args:
        household_frame: One row per household or household-like record.
        district_column: Congressional district GEOID/display column.
        weight_column: Household analysis weight.
        snap_receipt_column: Boolean/numeric SNAP receipt indicator.
        snap_amount_column: Optional household SNAP dollar amount.
        state_column: Optional state label repeated on household rows.
        state_snap_relative_error_column: Optional state SNAP spending fit error.
        acs_reference: Optional ACS S2201 reference frame, one row per district.
        acs_district_column: District column in ``acs_reference``. Defaults to
            ``district_column``.
        acs_snap_households_column: ACS SNAP household estimate column.
        acs_snap_households_moe_column: Optional ACS margin-of-error column.
        low_positive_sample_threshold: Raw positive household count flag.
        low_positive_ess_threshold: Positive-recipient ESS flag.
        state_outlier_abs_relative_error: State SNAP fit flag threshold.
        validation_source: Metadata about the ACS/reference source.
    """
    if low_positive_sample_threshold < 0:
        raise ValueError("low_positive_sample_threshold must be nonnegative.")
    if low_positive_ess_threshold < 0:
        raise ValueError("low_positive_ess_threshold must be nonnegative.")
    if state_outlier_abs_relative_error < 0:
        raise ValueError("state_outlier_abs_relative_error must be nonnegative.")

    _require_columns(
        household_frame,
        (
            district_column,
            weight_column,
            snap_receipt_column,
            snap_amount_column,
            state_column,
            state_snap_relative_error_column,
        ),
        label="SNAP local proxy household frame",
    )

    acs_district_column = acs_district_column or district_column
    references = _reference_by_district(
        acs_reference,
        district_column=acs_district_column,
        snap_households_column=acs_snap_households_column,
        snap_households_moe_column=acs_snap_households_moe_column,
    )

    frame = household_frame.copy()
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    valid = weights.notna() & np.isfinite(weights) & (weights > 0)
    valid &= frame[district_column].notna()
    if not valid.any():
        raise ValueError("SNAP local proxy frame has no positive-weight district rows.")

    frame = frame.loc[valid].copy()
    frame["_snap_local_proxy_weight"] = weights.loc[valid].astype(float)
    frame["_snap_local_proxy_receipt"] = (
        pd.to_numeric(frame[snap_receipt_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if snap_amount_column is not None:
        frame["_snap_local_proxy_amount"] = (
            pd.to_numeric(frame[snap_amount_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    rows: list[dict[str, Any]] = []
    for district, group in frame.groupby(district_column, dropna=False, sort=True):
        flags: list[str] = []
        district_id = str(district)
        group_weights = group["_snap_local_proxy_weight"].to_numpy(dtype=float)
        receipt = group["_snap_local_proxy_receipt"].to_numpy(dtype=float) > 0
        positive_weights = group_weights[receipt]
        weighted_households = float(group_weights.sum())
        weighted_snap_households = float(positive_weights.sum())
        raw_positive_sample_households = int(receipt.sum())
        positive_ess = _effective_sample_size(positive_weights)
        if raw_positive_sample_households < low_positive_sample_threshold:
            flags.append("low_positive_sample")
        if positive_ess is not None and positive_ess < low_positive_ess_threshold:
            flags.append("low_positive_ess")

        state_snap_relative_error = _group_state_error(
            group,
            state_snap_relative_error_column=state_snap_relative_error_column,
        )
        if (
            state_snap_relative_error is not None
            and abs(state_snap_relative_error) >= state_outlier_abs_relative_error
        ):
            flags.append("state_snap_outlier")

        reference = references.get(district_id) if references is not None else None
        acs_snap_households = None
        acs_snap_households_moe = None
        snap_household_difference = None
        snap_household_relative_error = None
        outside_acs_moe = None
        if references is not None and reference is None:
            flags.append("missing_acs_reference")
        elif reference is not None:
            acs_snap_households = reference["acs_snap_households"]
            acs_snap_households_moe = reference["acs_snap_households_moe"]
            if acs_snap_households is not None:
                snap_household_difference = (
                    weighted_snap_households - acs_snap_households
                )
                if acs_snap_households != 0:
                    snap_household_relative_error = (
                        snap_household_difference / acs_snap_households
                    )
                if acs_snap_households_moe is not None:
                    outside_acs_moe = (
                        abs(snap_household_difference) > acs_snap_households_moe
                    )
                    if outside_acs_moe:
                        flags.append("outside_acs_moe")

        snap_dollars = None
        if snap_amount_column is not None:
            amounts = group["_snap_local_proxy_amount"].to_numpy(dtype=float)
            snap_dollars = float(np.sum(amounts * group_weights))

        mean_positive_weight = (
            float(positive_weights.mean()) if positive_weights.size else None
        )
        max_to_mean = (
            float(positive_weights.max() / mean_positive_weight)
            if mean_positive_weight and mean_positive_weight > 0
            else None
        )
        rows.append(
            {
                "congressional_district": district_id,
                "state": _group_state(group, state_column=state_column, flags=flags),
                "weighted_households": _finite(weighted_households),
                "weighted_snap_households": _finite(weighted_snap_households),
                "snap_household_share": _finite(
                    weighted_snap_households / weighted_households
                    if weighted_households
                    else None
                ),
                "snap_dollars": _finite(snap_dollars),
                "raw_positive_snap_households": raw_positive_sample_households,
                "positive_snap_ess": _finite(positive_ess),
                "positive_snap_max_to_mean_weight_ratio": _finite(max_to_mean),
                "state_snap_relative_error": _finite(state_snap_relative_error),
                "acs_snap_households": _finite(acs_snap_households),
                "acs_snap_households_moe": _finite(acs_snap_households_moe),
                "snap_household_difference": _finite(snap_household_difference),
                "snap_household_relative_error": _finite(
                    snap_household_relative_error
                ),
                "outside_acs_moe": outside_acs_moe,
                "flags": tuple(sorted(flags)),
            }
        )

    total_weighted_households = sum(row["weighted_households"] or 0.0 for row in rows)
    total_weighted_snap_households = sum(
        row["weighted_snap_households"] or 0.0 for row in rows
    )
    total_snap_dollars = (
        sum(row["snap_dollars"] or 0.0 for row in rows)
        if snap_amount_column is not None
        else None
    )
    total_acs_snap_households = (
        sum(row["acs_snap_households"] or 0.0 for row in rows)
        if references is not None
        else None
    )
    outside_moe_values = [
        row["outside_acs_moe"] for row in rows if row["outside_acs_moe"] is not None
    ]
    payload = {
        "schema_version": SNAP_LOCAL_PROXY_SCHEMA_VERSION,
        "classification": "validation_only",
        "source_family": SNAP_LOCAL_PROXY_FAMILY,
        "validation_source": dict(validation_source or {}),
        "thresholds": {
            "low_positive_sample": int(low_positive_sample_threshold),
            "low_positive_ess": float(low_positive_ess_threshold),
            "state_outlier_abs_relative_error": float(
                state_outlier_abs_relative_error
            ),
        },
        "summary": {
            "districts": len(rows),
            "districts_with_acs_reference": sum(
                row["acs_snap_households"] is not None for row in rows
            ),
            "low_positive_sample_districts": sum(
                "low_positive_sample" in row["flags"] for row in rows
            ),
            "low_positive_ess_districts": sum(
                "low_positive_ess" in row["flags"] for row in rows
            ),
            "state_snap_outlier_districts": sum(
                "state_snap_outlier" in row["flags"] for row in rows
            ),
            "outside_acs_moe_districts": (
                sum(bool(value) for value in outside_moe_values)
                if outside_moe_values
                else None
            ),
            "weighted_households": _finite(total_weighted_households),
            "weighted_snap_households": _finite(total_weighted_snap_households),
            "snap_household_share": _finite(
                total_weighted_snap_households / total_weighted_households
                if total_weighted_households
                else None
            ),
            "snap_dollars": _finite(total_snap_dollars),
            "acs_snap_households": _finite(total_acs_snap_households),
        },
        "districts": rows,
    }
    return json.loads(json.dumps(payload, allow_nan=False))


def write_snap_local_proxy_diagnostics(
    payload: Mapping[str, object], path: Path | str
) -> Path:
    """Write a SNAP local proxy diagnostics payload as strict JSON."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
