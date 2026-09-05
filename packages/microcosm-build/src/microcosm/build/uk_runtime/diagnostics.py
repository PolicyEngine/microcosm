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
    "uk_support_limited_misses",
    "uk_target_geography_levels",
    "uk_weakest_areas_by_fit",
    "uk_weakest_families",
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
_AREA_FIT_LIMIT = 15

_COUNTRY_BY_AREA_PREFIX = {
    "E": "England",
    "N": "Northern Ireland",
    "S": "Scotland",
    "W": "Wales",
}


def _finite_target_error(row: Mapping[str, object]) -> tuple[str, float]:
    name = str(row.get("name") or "")
    raw_error = row.get("relative_error")
    if (
        not name
        or not isinstance(raw_error, (int, float))
        or isinstance(raw_error, bool)
    ):
        raise ValueError("UK target rollups require named finite relative errors.")
    error = float(raw_error)
    if not math.isfinite(error):
        raise ValueError("UK target rollups require named finite relative errors.")
    return name, error


def _finite_loss_contribution(row: Mapping[str, object]) -> float:
    raw = row.get("final_loss_contribution")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(
            "UK target rollups require schema-v6 final_loss_contribution values."
        )
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "UK target rollups require finite non-negative loss contributions."
        )
    return value


def uk_weakest_families(
    target_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rank every scored family by its schema-v6 loss contribution."""

    buckets: dict[str, list[tuple[str, float, float]]] = {}
    for row in target_rows:
        registry = row.get("registry")
        if not isinstance(registry, Mapping) or not str(registry.get("family") or ""):
            raise ValueError(
                "UK target rollups require a registry family on every row."
            )
        name, error = _finite_target_error(row)
        buckets.setdefault(str(registry["family"]), []).append(
            (name, abs(error), _finite_loss_contribution(row))
        )
    total_loss = math.fsum(
        contribution for rows in buckets.values() for _, _, contribution in rows
    )
    result: list[dict[str, object]] = []
    for family, rows in buckets.items():
        worst_name, worst_error, _ = max(rows, key=lambda item: (item[1], item[0]))
        contribution = math.fsum(item[2] for item in rows)
        within = sum(item[1] <= _TARGET_PASS_RELATIVE_ERROR for item in rows)
        result.append(
            {
                "family": family,
                "n_targets": len(rows),
                "n_within_10pct": within,
                "pass_rate": within / len(rows),
                "worst_target": worst_name,
                "worst_abs_relative_error": worst_error,
                "loss_contribution": contribution,
                "loss_share": contribution / total_loss if total_loss else 0.0,
            }
        )
    result.sort(
        key=lambda row: (
            -float(row["loss_contribution"]),
            str(row["family"]),
        )
    )
    return result


def uk_weakest_areas_by_fit(
    target_rows: Sequence[Mapping[str, object]],
    area_support: pd.DataFrame,
    *,
    limit: int = _AREA_FIT_LIMIT,
) -> dict[str, object]:
    """Return bottom-by-fit local areas plus country-level fit rollups."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("UK weakest-area limit must be a positive integer.")
    required = {
        "geography_level",
        "area_code",
        "nonzero_households",
        "nonzero_source_households",
        "effective_sample_size",
    }
    missing = sorted(required - set(area_support.columns))
    if missing:
        raise ValueError(f"UK area support is missing rollup column(s): {missing}.")
    support_rows = {
        (str(row.geography_level), str(row.area_code)): row
        for row in area_support.itertuples(index=False)
    }
    if len(support_rows) != len(area_support):
        raise ValueError("UK area support contains duplicate geography/area rows.")

    grouped: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
    for row in target_rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        area_type = metadata.get("area_type")
        if area_type is None:
            continue
        level = _normalize_geography_level(area_type)
        area_code = str(metadata.get("area_code") or "")
        if level not in {"constituency", "local_authority"} or not area_code:
            continue
        name, error = _finite_target_error(row)
        grouped.setdefault((level, area_code), []).append(
            (name, abs(error), _finite_loss_contribution(row))
        )

    area_rows: list[dict[str, object]] = []
    for (level, area_code), rows in grouped.items():
        support = support_rows.get((level, area_code))
        if support is None:
            raise ValueError(
                "UK area rollups require exact support for every scored area; "
                f"missing {(level, area_code)!r}."
            )
        worst_name, worst_error, _ = max(rows, key=lambda item: (item[1], item[0]))
        within = sum(item[1] <= _TARGET_PASS_RELATIVE_ERROR for item in rows)
        area_rows.append(
            {
                "geography_level": level,
                "area_code": area_code,
                "country": _country_for_area(area_code),
                "n_targets": len(rows),
                "n_within_10pct": within,
                "pass_rate": within / len(rows),
                "worst_target": worst_name,
                "worst_abs_relative_error": worst_error,
                "loss_contribution": math.fsum(item[2] for item in rows),
                "nonzero_households": int(support.nonzero_households),
                "nonzero_source_households": int(support.nonzero_source_households),
                "effective_sample_size": float(support.effective_sample_size),
            }
        )
    area_rows.sort(
        key=lambda row: (
            -float(row["worst_abs_relative_error"]),
            str(row["geography_level"]),
            str(row["area_code"]),
        )
    )

    countries: list[dict[str, object]] = []
    for country in _COUNTRY_BY_AREA_PREFIX.values():
        for level in ("constituency", "local_authority"):
            members = [
                row
                for row in area_rows
                if row["country"] == country and row["geography_level"] == level
            ]
            if not members:
                continue
            n_targets = sum(int(row["n_targets"]) for row in members)
            n_within = sum(int(row["n_within_10pct"]) for row in members)
            worst = max(
                members,
                key=lambda row: (
                    float(row["worst_abs_relative_error"]),
                    str(row["worst_target"]),
                ),
            )
            countries.append(
                {
                    "country": country,
                    "geography_level": level,
                    "n_areas": len(members),
                    "n_targets": n_targets,
                    "n_within_10pct": n_within,
                    "pass_rate": n_within / n_targets,
                    "worst_target": worst["worst_target"],
                    "worst_abs_relative_error": worst["worst_abs_relative_error"],
                    "loss_contribution": math.fsum(
                        float(row["loss_contribution"]) for row in members
                    ),
                }
            )
    countries.sort(key=lambda row: (str(row["country"]), str(row["geography_level"])))
    # Keyed by role, not by a count: the limit is a parameter, so a fixed
    # "bottom_15" name would disagree with the list whenever a caller passes
    # anything else, or whenever fewer areas were scored than the limit.
    return {
        "limit": limit,
        "n_areas_scored": len(area_rows),
        "bottom_by_fit": area_rows[:limit],
        "countries": countries,
    }


def _country_for_area(area_code: str) -> str:
    try:
        return _COUNTRY_BY_AREA_PREFIX[area_code[0]]
    except (IndexError, KeyError) as error:
        raise ValueError(
            f"Cannot derive UK country from area code {area_code!r}."
        ) from error


def _local_grain_column(frame: pd.DataFrame) -> str:
    for column in ("grain", "area_type", "geography_level"):
        if column in frame.columns:
            return column
    raise ValueError(
        "UK local diagnostics require a grain, area_type, or geography_level column."
    )


def uk_support_limited_misses(
    local_diagnostics: pd.DataFrame,
    area_support: Mapping[str, pd.DataFrame],
    *,
    max_abs_relative_error: float,
) -> dict[str, dict[str, object]]:
    """Relate failing local cells to each area's measured support."""

    if not math.isfinite(max_abs_relative_error) or max_abs_relative_error < 0:
        raise ValueError("max_abs_relative_error must be finite and non-negative.")
    required = {"area_code", "abs_relative_error"}
    missing = sorted(required - set(local_diagnostics.columns))
    if missing:
        raise ValueError(f"UK local diagnostics are missing columns {missing}.")
    grain_column = _local_grain_column(local_diagnostics)
    result: dict[str, dict[str, object]] = {}
    for raw_grain, support in sorted(area_support.items()):
        grain = str(raw_grain)
        required_support = {
            "area_code",
            "assigned_households",
            "effective_sample_size",
            "nonzero_source_households",
        }
        support_missing = sorted(required_support - set(support.columns))
        if support_missing:
            raise ValueError(
                f"UK area support for {grain!r} is missing columns {support_missing}."
            )
        grain_aliases = {grain}
        if grain == "la":
            grain_aliases.add("local_authority")
        elif grain == "local_authority":
            grain_aliases.add("la")
        cells = local_diagnostics.loc[
            local_diagnostics[grain_column].astype(str).isin(grain_aliases)
            & local_diagnostics["area_code"].notna()
        ].copy()
        cells["abs_relative_error"] = pd.to_numeric(
            cells["abs_relative_error"], errors="raise"
        )
        failing = cells.loc[cells["abs_relative_error"] > max_abs_relative_error]
        support_rows = support.copy()
        support_rows["area_code"] = support_rows["area_code"].astype(str)
        support_rows["effective_sample_size"] = pd.to_numeric(
            support_rows["effective_sample_size"], errors="raise"
        )
        bottom_cutoff = (
            float(support_rows["effective_sample_size"].quantile(0.1))
            if len(support_rows)
            else None
        )
        failing_with_support = failing.merge(
            support_rows, on="area_code", how="left", validate="many_to_one"
        )
        if (
            len(failing_with_support)
            and failing_with_support["effective_sample_size"].isna().any()
        ):
            missing_areas = sorted(
                failing_with_support.loc[
                    failing_with_support["effective_sample_size"].isna(), "area_code"
                ]
                .astype(str)
                .unique()
            )
            raise ValueError(
                f"UK area support for {grain!r} is missing failing areas {missing_areas}."
            )
        share_bottom = (
            float(
                (failing_with_support["effective_sample_size"] <= bottom_cutoff).mean()
            )
            if len(failing_with_support)
            else None
        )
        worst = (
            cells.groupby("area_code", sort=True)["abs_relative_error"]
            .max()
            .reset_index(name="worst_abs_relative_error")
            .merge(support_rows, on="area_code", how="left", validate="one_to_one")
        )
        correlation = (
            float(
                worst["worst_abs_relative_error"].corr(
                    worst["effective_sample_size"], method="spearman"
                )
            )
            if len(worst) > 1
            else None
        )
        if correlation is not None and not math.isfinite(correlation):
            correlation = None
        elif correlation is not None and math.isclose(abs(correlation), 1.0):
            correlation = math.copysign(1.0, correlation)
        worst = worst.sort_values(
            ["worst_abs_relative_error", "area_code"],
            ascending=[False, True],
            kind="stable",
        ).head(10)
        result[grain] = {
            "n_failing_cells": int(len(failing)),
            "share_failing_cells_in_bottom_ess_decile": share_bottom,
            "spearman_worst_abs_relative_error_vs_ess": correlation,
            "worst_areas": [
                {
                    "area_code": str(row["area_code"]),
                    "worst_abs_relative_error": float(row["worst_abs_relative_error"]),
                    "rows": int(row["assigned_households"]),
                    "ess": float(row["effective_sample_size"]),
                    "sources": int(row["nonzero_source_households"]),
                }
                for _, row in worst.iterrows()
            ],
        }
    return result


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
    local_area_support: pd.DataFrame | None = None,
    rotated_holdout: Mapping[str, object] | None = None,
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

    uk_diagnostics: dict[str, object] = {
        "schema_version": UK_DIAGNOSTICS_SCHEMA_VERSION,
        "weights": weights,
        "zero_weight_rows_by_stratum": strata,
        "target_pass_rates_by_geography_level": pass_rates,
        "target_observation_basis": {
            spec.to_target().row_name: str(spec.metadata["observation_basis"])
            for spec in registry.specs
            if spec.metadata.get("observation_basis")
        },
    }
    if local_area_support is not None:
        uk_diagnostics["weakest_families"] = uk_weakest_families(target_rows)
        uk_diagnostics["weakest_areas_by_fit"] = uk_weakest_areas_by_fit(
            target_rows,
            local_area_support,
        )
    if rotated_holdout is not None:
        uk_diagnostics["rotated_holdout"] = dict(rotated_holdout)
    payload["uk_diagnostics"] = uk_diagnostics
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
    local_area_support: pd.DataFrame | None = None,
    rotated_holdout: Mapping[str, object] | None = None,
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
            local_area_support=local_area_support,
            rotated_holdout=rotated_holdout,
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
