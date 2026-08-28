"""Standard calibration diagnostics for UK release candidates.

The shared :mod:`microcosm.calibrate.diagnostics` payload is the release
contract: it carries the target surface, every target row, solver options, and
the concentration scalars used by US releases.  UK needs a little more release
evidence without changing that shared schema (and therefore without changing
US output): the effective-sample-size fraction, shipped-weight concentration,
zero-weight rows split by their declared support strata, and target fit by UK
geography level.

This module wraps the shared payload and places those additions under a
separately versioned ``uk_diagnostics`` block.  The common top-level
``schema_version`` remains owned by ``microcosm-calibrate``.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.calibrate import (
    TargetRegistry,
    diagnostics_payload,
    effective_sample_size,
)
from microcosm.calibrate.solve import CalibrationResult
from microcosm.frame import Frame

__all__ = [
    "UK_DIAGNOSTICS_SCHEMA_VERSION",
    "UK_TARGET_GEOGRAPHY_LEVELS",
    "uk_calibration_diagnostics_payload",
    "uk_target_geography_levels",
    "uk_weight_summary",
    "uk_zero_weight_strata",
    "write_uk_calibration_diagnostics",
]

#: UK-only extension version nested inside the shared calibration diagnostics.
UK_DIAGNOSTICS_SCHEMA_VERSION = 1

#: Stable vocabulary used by the UK target registry and future OA-ladder rows.
#: ``"la"`` is accepted only as an input adapter and is serialized as
#: ``"local_authority"``.
UK_TARGET_GEOGRAPHY_LEVELS: tuple[str, ...] = (
    "national",
    "region",
    "country",
    "local_authority",
    "constituency",
)

_UK_DEFAULT_ZERO_WEIGHT_STRATUM_COLUMNS: tuple[str, ...] = (
    "household_is_spi_synthetic",
    "household_is_capital_gains_clone",
)

_TARGET_PASS_RELATIVE_ERROR = 0.10


def _as_weights(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return one finite, non-negative, non-empty weight vector."""

    try:
        weights = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("UK diagnostic weights must be numeric.") from exc
    if weights.ndim != 1:
        raise ValueError(
            f"UK diagnostic weights must be one-dimensional, got shape {weights.shape}."
        )
    if weights.size == 0:
        raise ValueError("UK diagnostic weights must not be empty.")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("UK diagnostic weights must be finite and non-negative.")
    return weights


def uk_weight_summary(
    weights: Sequence[float] | np.ndarray,
) -> dict[str, int | float | None]:
    """Summarize one shipped UK household-weight vector.

    The ESS fraction uses *all* shipped records as its denominator, including
    zero-weight support rows.  Median weight uses positive records only: zeros
    are reported separately and must not turn the max-to-median concentration
    diagnostic into an accidental function of how many dead rows ship.

    An all-zero vector remains reportable so a batched terminal gate can name
    every failure.  Its ESS and top-one-percent share are zero; the positive
    median and max-to-median ratio are undefined and serialize as ``null``.
    """

    values = _as_weights(weights)
    positive = values > 0.0
    positive_values = values[positive]
    total = float(values.sum())
    ess = float(effective_sample_size(values))
    median_positive = (
        float(np.median(positive_values)) if positive_values.size else None
    )
    maximum = float(values.max())
    max_to_median = (
        maximum / median_positive
        if median_positive is not None and median_positive > 0.0
        else None
    )
    top_count = max(1, math.ceil(0.01 * values.size))
    top_share = (
        float(np.sort(values)[-top_count:].sum() / total) if total > 0.0 else 0.0
    )
    return {
        "n_records": int(values.size),
        "positive_weight_records": int(positive.sum()),
        "zero_weight_records": int((values == 0.0).sum()),
        "total_weight": total,
        "effective_sample_size": ess,
        "ess_fraction": ess / values.size,
        "median_positive_weight": median_positive,
        "max_weight": maximum,
        "max_to_median_positive_weight": max_to_median,
        "top_1pct_weight_share": top_share,
    }


def _json_scalar(value: object, *, column: str) -> object:
    """Normalize one stratum value to strict, stable JSON."""

    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if not isinstance(missing, (bool, np.bool_)):
        raise TypeError(
            f"UK diagnostic stratum column {column!r} must contain scalar values."
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"UK diagnostic stratum column {column!r} contains a non-finite value."
        )
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    raise TypeError(
        f"UK diagnostic stratum column {column!r} contains unsupported "
        f"value type {type(value).__name__}."
    )


def _stratum_columns(columns: Sequence[str]) -> tuple[str, ...]:
    materialized = tuple(str(column) for column in columns)
    if not materialized or any(not column for column in materialized):
        raise ValueError("UK diagnostic stratum_columns must be non-empty names.")
    if len(set(materialized)) != len(materialized):
        raise ValueError("UK diagnostic stratum_columns must be unique.")
    return materialized


def uk_zero_weight_strata(
    household: pd.DataFrame,
    weights: Sequence[float] | np.ndarray,
    *,
    stratum_columns: Sequence[str] = _UK_DEFAULT_ZERO_WEIGHT_STRATUM_COLUMNS,
) -> list[dict[str, object]]:
    """Count positive and zero shipped weights in every observed stratum.

    Rows are sorted by canonical JSON of the stratum values, making the output
    deterministic across pandas versions and input row order.  Every observed
    combination is retained, including combinations with no zero-weight rows,
    so the counts reconcile to the complete release surface.
    """

    if not isinstance(household, pd.DataFrame):
        raise TypeError("UK diagnostic household data must be a pandas DataFrame.")
    values = _as_weights(weights)
    if len(household) != values.size:
        raise ValueError(
            "UK diagnostic household rows must align with weights, got "
            f"{len(household)} rows for {values.size} weights."
        )
    columns = _stratum_columns(stratum_columns)
    missing = sorted(set(columns) - set(household.columns))
    if missing:
        raise ValueError(
            f"UK diagnostic household data is missing stratum column(s): {missing}."
        )

    groups: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}
    stratum_values = household.loc[:, list(columns)].itertuples(
        index=False,
        name=None,
    )
    for raw_stratum, weight in zip(stratum_values, values, strict=True):
        stratum = {
            column: _json_scalar(raw_value, column=column)
            for column, raw_value in zip(columns, raw_stratum, strict=True)
        }
        # Python considers ``True == 1`` and ``False == 0``.  Keep a JSON-type
        # tag in the grouping key so distinct serialized strata cannot merge.
        key = tuple(
            (
                "null" if value is None else type(value).__name__,
                value,
            )
            for value in stratum.values()
        )
        group = groups.setdefault(
            key,
            {
                "stratum": stratum,
                "rows": 0,
                "positive_weight_rows": 0,
                "zero_weight_rows": 0,
                "weight_sum": 0.0,
            },
        )
        group["rows"] = int(group["rows"]) + 1
        if weight > 0.0:
            group["positive_weight_rows"] = int(group["positive_weight_rows"]) + 1
        else:
            group["zero_weight_rows"] = int(group["zero_weight_rows"]) + 1
        group["weight_sum"] = float(group["weight_sum"]) + float(weight)

    rows = sorted(
        groups.values(),
        key=lambda row: json.dumps(
            row["stratum"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    if sum(int(row["rows"]) for row in rows) != values.size:
        raise RuntimeError("UK diagnostic stratum row counts do not reconcile.")
    if sum(int(row["zero_weight_rows"]) for row in rows) != int((values == 0.0).sum()):
        raise RuntimeError("UK diagnostic zero-weight stratum counts do not reconcile.")
    return rows


def _normalize_geography_level(value: object) -> str:
    raw = getattr(value, "value", value)
    level = str(raw).strip().lower()
    if level == "la":
        level = "local_authority"
    if level not in UK_TARGET_GEOGRAPHY_LEVELS:
        raise ValueError(
            f"Unknown UK target geography level {raw!r}; expected one of "
            f"{UK_TARGET_GEOGRAPHY_LEVELS} (or adapter alias 'la')."
        )
    return level


def uk_target_geography_levels(registry: TargetRegistry) -> dict[str, str]:
    """Map declared UK target row names to their contract geography level."""

    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    targets = {row["target_id"]: row for row in payload["targets"]}
    levels: dict[str, str] = {}
    for spec in registry.specs:
        target_id = spec.metadata.get("contract_target_id")
        target = targets.get(str(target_id))
        if target is None:
            raise ValueError(
                f"UK calibration target {spec.name!r} references unknown "
                f"contract target {target_id!r}."
            )
        geography_levels = tuple(target.get("geography_levels") or ())
        if not geography_levels:
            raise ValueError(
                f"UK calibration target {spec.name!r} has no geography level."
            )
        levels[spec.to_target().row_name] = str(geography_levels[0])
    return levels


def _geography_mapping(
    target_names: Sequence[str],
    values: Mapping[str, object],
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(
            "target_geography_levels must map declared target names to levels."
        )
    normalized: dict[str, str] = {}
    for raw_name, raw_level in values.items():
        name = str(raw_name)
        if name in normalized:
            raise ValueError(
                f"target_geography_levels has duplicate normalized name {name!r}."
            )
        normalized[name] = _normalize_geography_level(raw_level)
    expected = set(target_names)
    missing = sorted(expected - set(normalized))
    extra = sorted(set(normalized) - expected)
    if missing or extra:
        raise ValueError(
            "target_geography_levels must exactly cover the declared UK target "
            "registry; "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return normalized


def _require_uk_target_registry(value: object) -> TargetRegistry:
    """Require the non-empty UK registry a release contract can validate."""

    if not isinstance(value, TargetRegistry):
        raise TypeError(
            "UK calibration diagnostics require a TargetRegistry, got "
            f"{type(value).__name__}."
        )
    if value.country != "uk":
        raise ValueError(
            "UK calibration diagnostics require target_registry.country == 'uk', "
            f"got {value.country!r}."
        )
    if not value.specs:
        raise ValueError("UK calibration diagnostics require a non-empty registry.")
    if not isinstance(value.version, str) or not value.version:
        raise ValueError(
            "UK calibration diagnostics require a non-empty registry version."
        )
    return value


def _declared_target_partition(
    result: CalibrationResult,
    target_rows: list[dict[str, object]],
    registry: TargetRegistry,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Reconcile registry declarations with compiled and skipped target rows."""

    declared = tuple(spec.to_target().row_name for spec in registry.specs)
    compiled = tuple(str(row["name"]) for row in target_rows)
    skipped: list[str] = []
    for item in result.skipped:
        target = getattr(item, "target", None)
        row_name = getattr(target, "row_name", None)
        if not isinstance(row_name, str) or not row_name:
            raise ValueError(
                "UK calibration diagnostics contain a malformed skipped target."
            )
        skipped.append(row_name)
    skipped_names = tuple(skipped)

    if len(set(compiled)) != len(compiled):
        raise ValueError("UK calibration diagnostics contain duplicate target names.")
    if len(set(skipped_names)) != len(skipped_names):
        raise ValueError(
            "UK calibration diagnostics contain duplicate skipped target names."
        )
    overlap = sorted(set(compiled) & set(skipped_names))
    observed = set(compiled) | set(skipped_names)
    missing = sorted(set(declared) - observed)
    extra = sorted(observed - set(declared))
    if overlap or missing or extra or len(observed) != len(declared):
        raise ValueError(
            "UK calibration result must exactly partition the declared target "
            "registry into compiled and skipped rows; "
            f"overlap={overlap[:10]}, missing={missing[:10]}, extra={extra[:10]}."
        )
    return declared, compiled, skipped_names


def _target_pass_rates(
    target_rows: list[dict[str, object]],
    skipped_names: Sequence[str],
    geography_levels: Mapping[str, str],
) -> list[dict[str, object]]:
    counts = {
        level: {
            "n_targets": 0,
            "n_scored": 0,
            "n_skipped": 0,
            "n_within_10pct": 0,
        }
        for level in UK_TARGET_GEOGRAPHY_LEVELS
    }
    for row in target_rows:
        name = str(row["name"])
        level = geography_levels[name]
        raw_error = row.get("relative_error")
        if not isinstance(raw_error, (int, float)) or isinstance(raw_error, bool):
            raise ValueError(
                f"UK target diagnostic {name!r} has no finite relative_error."
            )
        error = float(raw_error)
        if not math.isfinite(error):
            raise ValueError(
                f"UK target diagnostic {name!r} has no finite relative_error."
            )
        counts[level]["n_targets"] += 1
        counts[level]["n_scored"] += 1
        if abs(error) <= _TARGET_PASS_RELATIVE_ERROR:
            counts[level]["n_within_10pct"] += 1
    for name in skipped_names:
        level = geography_levels[name]
        counts[level]["n_targets"] += 1
        counts[level]["n_skipped"] += 1

    return [
        {
            "geography_level": level,
            "n_targets": counts[level]["n_targets"],
            "n_scored": counts[level]["n_scored"],
            "n_skipped": counts[level]["n_skipped"],
            "n_within_10pct": counts[level]["n_within_10pct"],
            "pass_rate": (
                counts[level]["n_within_10pct"] / counts[level]["n_targets"]
                if counts[level]["n_targets"]
                else None
            ),
        }
        for level in UK_TARGET_GEOGRAPHY_LEVELS
    ]


def uk_calibration_diagnostics_payload(
    result: CalibrationResult,
    frame: Frame,
    *,
    target_geography_levels: Mapping[str, object],
    target_registry: TargetRegistry,
    stratum_columns: Sequence[str] = _UK_DEFAULT_ZERO_WEIGHT_STRATUM_COLUMNS,
    build: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Render shared diagnostics plus the versioned UK release evidence.

    ``target_geography_levels`` is deliberately explicit and exact.  Target
    names are never parsed to guess their geography, and a missing, stale, or
    unknown mapping aborts instead of silently shrinking a scoreboard level.
    """

    registry = _require_uk_target_registry(target_registry)
    if not isinstance(frame, Frame):
        raise TypeError("UK diagnostic data must be a Frame.")
    household = frame.table("household")
    result_weights = _as_weights(result.weights)
    shipped_weights = _as_weights(frame.weights_for("household").values)
    if shipped_weights.shape != result_weights.shape or not np.array_equal(
        shipped_weights,
        result_weights,
    ):
        raise ValueError(
            "UK diagnostic household_weight must exactly match the result weights."
        )

    payload: dict[str, object] = diagnostics_payload(
        result,
        target_registry=registry,
        build=build,
    )
    target_rows = payload.get("targets")
    if not isinstance(target_rows, list) or any(
        not isinstance(row, dict) for row in target_rows
    ):
        raise RuntimeError("Shared calibration diagnostics returned malformed targets.")
    declared, _, skipped = _declared_target_partition(result, target_rows, registry)
    geography = _geography_mapping(declared, target_geography_levels)
    weights = uk_weight_summary(result_weights)
    pass_rates = _target_pass_rates(target_rows, skipped, geography)
    strata = uk_zero_weight_strata(
        household,
        result_weights,
        stratum_columns=stratum_columns,
    )

    payload["uk_diagnostics"] = {
        "schema_version": UK_DIAGNOSTICS_SCHEMA_VERSION,
        "weights": weights,
        "zero_weight_rows_by_stratum": strata,
        "target_pass_rates_by_geography_level": pass_rates,
    }
    return payload


def write_uk_calibration_diagnostics(
    result: CalibrationResult,
    path: Path | str,
    frame: Frame,
    *,
    target_geography_levels: Mapping[str, object],
    target_registry: TargetRegistry,
    stratum_columns: Sequence[str] = _UK_DEFAULT_ZERO_WEIGHT_STRATUM_COLUMNS,
    build: dict[str, Any] | None = None,
) -> Path:
    """Atomically write strict shared-plus-UK diagnostics."""

    output = Path(path)
    encoded = json.dumps(
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=target_geography_levels,
            target_registry=target_registry,
            stratum_columns=stratum_columns,
            build=build,
        ),
        indent=1,
        allow_nan=False,
    )
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
