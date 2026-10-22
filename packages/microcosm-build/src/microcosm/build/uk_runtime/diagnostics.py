"""Standard calibration diagnostics for UK release candidates.

The shared :mod:`microcosm.calibrate.diagnostics` payload is the release
contract: it carries the target surface, every target row, solver options,
the schema-8 provider/category/geography/dimension/target hierarchy, and the
concentration scalars used by US releases. UK adds the effective-sample-size
fraction, shipped-weight concentration, zero-weight rows split by their
declared support strata, and target fit by UK geography level.

This module wraps the shared payload and places those additions under a
separately versioned ``uk_diagnostics`` block.  The common top-level
``schema_version`` remains owned by ``microcosm-calibrate``.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Collection, Mapping, Sequence
from importlib import resources as importlib_resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy import sparse

from microcosm.calibrate import (
    TargetRegistry,
    diagnostics_payload,
    effective_sample_size,
)
from microcosm.calibrate.diagnostics import past_cap_census
from microcosm.calibrate.solve import CalibrationResult
from microcosm.diagnostics import (
    UK_DIAGNOSTICS_SCHEMA_VERSION,
    UK_MODEL_FIGURE_FIELDS,
    UK_TARGET_GEOGRAPHY_LEVELS,
    UK_WITHHELD_PUBLISHED_ITERATE_FIELDS,
    UK_WITHHELD_PUBLISHED_OPTIONS,
    CalibrationDiagnosticsV8,
    DiagnosticsWriteOutcome,
    failed_diagnostics_outcome,
)
from microcosm.diagnostics import (
    write_calibration_diagnostics as write_typed_calibration_diagnostics,
)
from microcosm.frame import Frame

__all__ = [
    "UK_DIAGNOSTICS_SCHEMA_VERSION",
    "UK_TARGET_GEOGRAPHY_LEVELS",
    "uk_calibration_diagnostics_payload",
    "uk_consumption_drift",
    "uk_fit_by_family",
    "uk_support_limited_misses",
    "uk_target_geography_levels",
    "uk_weakest_areas_by_fit",
    "uk_weakest_families",
    "uk_weight_summary",
    "uk_zero_weight_strata",
    "write_uk_calibration_diagnostics",
]

_UK_DEFAULT_ZERO_WEIGHT_STRATUM_COLUMNS: tuple[str, ...] = (
    "household_is_spi_synthetic",
    "household_is_capital_gains_clone",
)

_TARGET_PASS_RELATIVE_ERROR = 0.10
_AREA_FIT_LIMIT = 15
#: A target row's model figures leave the build only when at least this many
#: distinct source households carry them (UKDS CD137 clause 8): María's ruling
#: of 2026-10-17 on the #1305 review for BRMA rows, extended to national and
#: regional rows by her rule of 2026-10-21, which binds a thin row rather
#: than leave it out (local rows keep the area-support rule).
UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS = 10
#: The BRMA rows' floor, the same ten under its own ruling of 2026-10-17 (the
#: support gate covers constituencies and authorities, not the partial BRMA
#: grain).
UK_BRMA_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS = 10
#: The fields of a target row that the model's weights produce (the shared
#: tuple the release contract checks too).
_MODEL_FIGURE_FIELDS = UK_MODEL_FIGURE_FIELDS
_WITHHELD_NOTE = (
    "model figures withheld: fewer than 10 distinct source households carry this row"
)
_COMPLEMENT_NOTE = (
    "model figures withheld: with the published rows they would give back a row "
    "that fewer than 10 distinct source households carry"
)

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
    """Return bottom-by-fit local areas plus country-level fit rollups.

    ``area_support`` must cover every scored area. A row carries the area's
    ``support_below_floor`` flag when the table has one, never its rows, ESS
    or distinct sources: these diagnostics ship in a release, and per-area
    support figures stay in the build's area-support table (microcosm#1301).
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("UK weakest-area limit must be a positive integer.")
    required = {"geography_level", "area_code"}
    missing = sorted(required - set(area_support.columns))
    if missing:
        raise ValueError(f"UK area support is missing rollup column(s): {missing}.")
    support_rows = {
        (str(row.geography_level), str(row.area_code)): row
        for row in area_support.itertuples(index=False)
    }
    if len(support_rows) != len(area_support):
        raise ValueError("UK area support contains duplicate geography/area rows.")
    flagged = "support_below_floor" in area_support.columns

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
                **_support_flag(support.support_below_floor if flagged else None),
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


def _support_flag(value: object) -> dict[str, bool]:
    """``{"support_below_floor": flag}`` when the area's flag is known, else empty."""

    if value is None or pd.isna(value):
        return {}
    return {"support_below_floor": bool(value)}


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
    """Relate failing local cells to each area's measured support.

    The per-grain rollups (failing cells, their share in the bottom ESS
    decile, the rank correlation of fit with ESS) are over many areas. The
    worst areas are listed by code and fit, with ``support_below_floor`` when
    the support table carries it; their rows, ESS and distinct sources stay
    in the build's area-support table (microcosm#1301).
    """

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
        required_support = {"area_code", "effective_sample_size"}
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
                    **_support_flag(row.get("support_below_floor")),
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


#: Lineage columns that identify a support-split family: every copy the split
#: created shares its root's source household, support channel, support clone
#: index and clone flag. These are the geography identity kernel's inputs
#: (microcosm#932) less the copy flag and index, so on the national spine only
#: the split creates rows that agree on all four; on a K-expanded frame the
#: geographic clone index (:func:`uk_support_family_geographic_clone_column`)
#: joins the key, so each geographic clone folds only its own copies
#: (microcosm#1045).
UK_SUPPORT_FAMILY_KEY_COLUMNS = (
    "source_household_id",
    "household_support_channel",
    "household_support_clone_index",
    "household_is_capital_gains_clone",
)
UK_SUPPORT_COPY_FLAG_COLUMN = "household_is_cgt_support_copy"
#: The residential split's arms (microcosm#1063) carry their household's
#: lineage key, so they fold onto it like the support copies.
UK_RESIDENTIAL_ARM_FLAG_COLUMN = "household_is_cgt_residential_clone"


def uk_support_family_geographic_clone_column() -> str:
    """The household geographic clone-index column ``uk.full.expand`` writes."""

    from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column

    return ladder_clone_index_column("household")


def uk_support_family_weights(
    household: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fold shipped household weights over support-split families.

    Each root and the copies ``cgt_support_split`` made of it (and, on the
    clone side, its clone and the copies' clones) sum to one weight, so the
    max-to-median ratio and the ESS fraction read the quantity the June
    certification measured on a frame without copies, not the number of
    copies. The key is the explicit lineage the geography identity keys on,
    less the copy flag and index: on a geographically expanded frame the
    clone-index column is part of it, so the K clones of a household stay
    apart and only their own copies fold onto each. A table without the copy
    flag, or with no copy, folds to itself. A table with copies but without
    the family key columns refuses.
    """

    weights = pd.to_numeric(household["household_weight"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    copies = np.zeros(len(household), dtype=bool)
    for flag in (UK_SUPPORT_COPY_FLAG_COLUMN, UK_RESIDENTIAL_ARM_FLAG_COLUMN):
        if flag in household.columns:
            copies |= household[flag].fillna(False).to_numpy(dtype=bool)
    if not copies.any():
        return weights.copy(), {
            "basis": "rows",
            "families": int(len(household)),
            "support_copy_rows": 0,
            "rows_folded": 0,
        }
    missing = [c for c in UK_SUPPORT_FAMILY_KEY_COLUMNS if c not in household.columns]
    if missing:
        raise ValueError(
            "Family-folded weights need the spine lineage column(s) "
            f"{missing} once the household table carries support copies."
        )
    geographic_clone_column = uk_support_family_geographic_clone_column()
    key_columns = list(UK_SUPPORT_FAMILY_KEY_COLUMNS)
    if geographic_clone_column in household.columns:
        key_columns.append(geographic_clone_column)
    key = household.loc[:, key_columns]
    if key.isna().any().any():
        raise ValueError("Family-folded weights refuse null lineage values.")
    family = key.groupby(key_columns, sort=True).ngroup()
    folded = pd.Series(weights, index=household.index).groupby(family.to_numpy()).sum()
    return folded.to_numpy(dtype=np.float64), {
        "basis": "support_family_fold",
        "key_columns": key_columns,
        "geographic_clone_index_in_key": geographic_clone_column in key_columns,
        "families": int(folded.size),
        "support_copy_rows": int(copies.sum()),
        "rows_folded": int(len(household) - folded.size),
    }


def uk_fit_by_family(
    diagnostics: pd.DataFrame,
    *,
    name_column: str = "target_name",
) -> list[dict[str, object]]:
    """Summarize target-fit diagnostics by family."""

    if diagnostics.empty:
        return []
    rows = []
    for family, group in diagnostics.groupby("family", sort=True):
        errors = group["abs_relative_error"].to_numpy(dtype=np.float64)
        worst_index = int(np.argmax(errors))
        worst = group.iloc[worst_index]
        rows.append(
            {
                "family": str(family),
                "n_targets": len(group),
                "share_within_10pct": float((errors <= 0.10).mean()),
                "share_within_25pct": float((errors <= 0.25).mean()),
                "worst_abs_relative_error": float(errors[worst_index]),
                "worst_cell": str(worst.get(name_column, worst.get("name", "unknown"))),
            }
        )
    return rows


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


#: Columns whose level the lcfs_consumption stage sets (energy at DESNZ volume
#: and QEP prices, road fuel at ONS 07.2.2) and calibration should leave alone.
_UK_DRIFT_TOTAL_COLUMNS: tuple[str, ...] = (
    "electricity_consumption",
    "gas_consumption",
    "domestic_energy_consumption",
    "petrol_spending",
    "diesel_spending",
)


def _drift_value(design: float, final: float) -> dict[str, float | None]:
    return {
        "design": float(design),
        "final": float(final),
        "change": float(final - design),
        "relative_change": None if design == 0 else float((final - design) / design),
    }


def uk_consumption_drift(
    household: pd.DataFrame,
    design_weights: Sequence[float] | np.ndarray,
    final_weights: Sequence[float] | np.ndarray,
) -> dict[str, object] | None:
    """Stage-levelled spend and its composition at design and final weights.

    Report only (microcosm#1113): the stage sets the energy and road-fuel
    levels, the gas connection and the fuel incidence, so calibration should
    barely move them. Returns None for a frame without the consumption columns.
    Where the frame carries the ETB VAT stage's full-rate expenditure rate,
    the totals also hold the full-rate VAT base, consumption times that rate
    (microcosm#1207: no bound row holds the VAT base, so this is where its
    drift shows). Where it carries the household's region, they also hold
    the electricity and gas kWh the stage raked, read back from the spend at
    the stage's declared prices (microcosm#1207).
    """

    from microcosm.build.uk_runtime.domestic_energy_targets import (
        household_energy_kwh,
        household_energy_prices,
    )
    from microcosm.build.uk_runtime.etb_vat import UK_ETB_VAT_OUTPUT_COLUMNS
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
    )

    needed = {
        *_UK_DRIFT_TOTAL_COLUMNS,
        *UK_LCFS_COICOP_DIVISION_COLUMNS,
        "has_fuel_consumption",
    }
    if not needed <= set(household.columns):
        return None
    design = _as_weights(design_weights)
    final = _as_weights(final_weights)
    if design.shape != (len(household),) or final.shape != (len(household),):
        raise ValueError("UK drift weights must align with the household rows.")

    def column(name: str) -> np.ndarray:
        return household[name].to_numpy(dtype=np.float64)

    totals = {
        name: _drift_value(design @ column(name), final @ column(name))
        for name in _UK_DRIFT_TOTAL_COLUMNS
    }
    consumption = sum(column(name) for name in UK_LCFS_COICOP_DIVISION_COLUMNS)
    totals["consumption"] = _drift_value(design @ consumption, final @ consumption)
    (vat_rate_column,) = UK_ETB_VAT_OUTPUT_COLUMNS
    if vat_rate_column in household.columns:
        vat_base = consumption * column(vat_rate_column)
        totals["full_rate_vat_base"] = _drift_value(design @ vat_base, final @ vat_base)
    if "region" in household.columns:
        kwh = household_energy_kwh(
            column("electricity_consumption"),
            column("gas_consumption"),
            region=household["region"].astype(str).to_numpy(),
            prices=household_energy_prices(),
        )
        for name, values in kwh.items():
            totals[name] = _drift_value(design @ values, final @ values)
    road_fuel = column("petrol_spending") + column("diesel_spending")
    flagged = household["has_fuel_consumption"].to_numpy(dtype=bool)

    def share(mask: np.ndarray, base: np.ndarray, weights: np.ndarray) -> float:
        mass = float(weights[base].sum())
        return float(weights[mask & base].sum()) / mass if mass > 0 else 0.0

    everyone = np.ones(len(household), dtype=bool)
    shares = {
        name: _drift_value(share(mask, base, design), share(mask, base, final))
        for name, mask, base in (
            ("gas_connected_households", column("gas_consumption") > 0, everyone),
            ("fuel_buying_households", road_fuel > 0, everyone),
            ("fuel_car_households_without_fuel", ~(road_fuel > 0), flagged),
        )
    }
    return {"report_only": True, "totals": totals, "shares": shares}


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


#: Adapter and Ledger spellings of a diagnostics level: the rowwise area
#: type ``la``, and ``statistical_scope``, the Ledger level of PIPR's Scottish
#: BRMA rows, the only targets the contract binds at it (microcosm#1262).
_GEOGRAPHY_LEVEL_ALIASES = {"la": "local_authority", "statistical_scope": "brma"}


def _normalize_geography_level(value: object) -> str:
    raw = getattr(value, "value", value)
    level = str(raw).strip().lower()
    level = _GEOGRAPHY_LEVEL_ALIASES.get(level, level)
    if level not in UK_TARGET_GEOGRAPHY_LEVELS:
        raise ValueError(
            f"Unknown UK target geography level {raw!r}; expected one of "
            f"{UK_TARGET_GEOGRAPHY_LEVELS} (or alias 'la' or 'statistical_scope')."
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


#: The local area levels whose rows the area-support rule governs
#: (``uk_local_area_support``) rather than the per-row floor below: María's
#: rule of 2026-10-21 has not been applied to the local line, and the report
#: guard matches a withheld local row by its bare area code, which suits a
#: BRMA's one family but not an area that binds many.
UK_AREA_SUPPORT_LEVELS = frozenset({"constituency", "local_authority"})


#: How near a combination of withheld rows may come to the published rows'
#: span and still count as given back: the squared distance of its unit-normed
#: row, so within 0.1% in norm, which the published figures reproduce to the
#: same precision. Exact recoveries sit near 1e-9 and the rest above 1e-3.
_RECOVERY_TOLERANCE = 1e-6
#: Below this share of the largest entry, an entry of a row-reduced
#: combination, of its values or of its published coefficients is rounding
#: (the eigenvectors carry errors near 1e-8). Treating a small entry as zero
#: can only shrink a combination's support, so it errs toward withholding.
_RECOVERY_ROUNDING = 1e-6


def _row_level(geography_levels: Mapping[str, object] | None):
    def level(name: str) -> str:
        raw = (geography_levels or {}).get(name, "")
        text = str(getattr(raw, "value", raw)).strip().lower()
        return _GEOGRAPHY_LEVEL_ALIASES.get(text, text)

    return level


def _scoped_rows(
    problem: Any,
    weights: Sequence[float] | np.ndarray,
    frame: Frame,
    target_rows: Sequence[Mapping[str, object]],
    geography_levels: Mapping[str, object] | None,
) -> tuple[list[str], sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """The rows the per-row floor governs, sorted, over the positive-weight households.

    Returns their names, their problem rows (stored zeros dropped: a household
    with a zero entry does not carry the row), those households' source ids,
    each row's distinct source households and each row's floor.
    """

    level = _row_level(geography_levels)
    names = sorted(
        {
            str(row["name"])
            for row in target_rows
            if level(str(row["name"])) not in UK_AREA_SUPPORT_LEVELS
        }
    )
    if not names:
        empty = np.zeros(0)
        return [], sparse.csr_matrix((0, 0)), empty, empty, empty
    household = frame.table("household")
    if "source_household_id" not in household.columns:
        raise ValueError(
            "UK diagnostics need household source_household_id to count each "
            "row's carriers."
        )
    sources = household["source_household_id"].to_numpy()
    positive = np.asarray(weights, dtype=float) > 0
    if positive.shape != sources.shape:
        raise ValueError("UK row carriers need one weight per household.")
    index = {str(name): position for position, name in enumerate(problem.names)}
    for name in names:
        if name not in index:
            raise ValueError(f"UK target diagnostic {name!r} has no problem row.")
    matrix = sparse.csr_matrix(problem.matrix)[[index[name] for name in names]]
    matrix = sparse.csr_matrix(matrix[:, np.flatnonzero(positive)])
    matrix.eliminate_zeros()
    sources = sources[positive]
    carriers = np.array(
        [
            len(pd.unique(sources[matrix.indices[start:stop]]))
            for start, stop in zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True)
        ]
    )
    floors = np.array(
        [
            UK_BRMA_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS
            if level(name) == "brma"
            else UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS
            for name in names
        ]
    )
    return names, matrix, sources, carriers, floors


def uk_thin_rows(
    problem: Any,
    weights: Sequence[float] | np.ndarray,
    frame: Frame,
    target_rows: Sequence[Mapping[str, object]],
    geography_levels: Mapping[str, object] | None = None,
) -> set[str]:
    """The rows whose model figures rest on fewer than ten source households.

    National, regional and BRMA rows: María's rule of 2026-10-21 binds a row
    however few households carry it, and keeps its figures in the build, as
    BRMA rows' have been kept since her ruling of 2026-10-17. A row whose
    level in ``geography_levels`` is a constituency or a local authority is
    left to the area-support rule (``UK_AREA_SUPPORT_LEVELS``). ``problem`` is
    the numerical problem the rows were scored on (its ``matrix`` and
    ``names``) and ``weights`` the weights they were scored at. A row's
    carriers are the distinct ``source_household_id`` values of the
    positive-weight households with a nonzero entry in its problem row, as the
    area support counts them.
    """

    names, _, _, carriers, floors = _scoped_rows(
        problem, weights, frame, target_rows, geography_levels
    )
    return {
        name
        for name, count, floor in zip(names, carriers, floors, strict=True)
        if count < floor
    }


def uk_withheld_row_sets(
    problem: Any,
    weights: Sequence[float] | np.ndarray,
    frame: Frame,
    target_rows: Sequence[Mapping[str, object]],
    geography_levels: Mapping[str, object] | None = None,
) -> tuple[set[str], set[str]]:
    """The thin rows (:func:`uk_thin_rows`) and the published rows withheld beside them.

    A published row's model figure is its problem row times the weights, so a
    combination of withheld rows is given back exactly when its row lies in
    the span of the published rows over the positive-weight households (a
    family total less its published bands gives back the bands withheld).
    María's hard rule (UKDS CD171 section 5.2.1): no difference of published
    figures may isolate fewer than ten cases, and where a total would give a
    withheld cell back, one more cell is withheld (Vahid's second review of
    #1333, item 10). While the published rows give back a combination of
    withheld rows that rests on fewer source households than its rows' floor,
    the published row with the fewest carriers among those that give it back
    is withheld too (the next-smallest band beside a thin one, not its total),
    and the span is measured again. Constituency and authority rows stay
    outside, as they stay outside the per-row floor: whether a local row can
    give back a national one is part of the local-line ruling still owed.
    """

    names, matrix, sources, carriers, floors = _scoped_rows(
        problem, weights, frame, target_rows, geography_levels
    )
    thin = {
        name
        for name, count, floor in zip(names, carriers, floors, strict=True)
        if count < floor
    }
    if not thin:
        return thin, set()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    unit = sparse.csr_matrix(sparse.diags(scale) @ matrix)
    gram = (unit @ unit.T).toarray()
    withheld = set(thin)
    for _ in range(len(names)):
        held = [i for i, name in enumerate(names) if name in withheld and norms[i]]
        published = [
            i for i, name in enumerate(names) if name not in withheld and norms[i]
        ]
        if not held or not published:
            break
        recovery = _small_recovery(unit, gram, sources, held, published, floors)
        if recovery is None:
            break
        coefficients, leverage = recovery
        size = np.abs(coefficients)
        givers = np.flatnonzero(size > _RECOVERY_ROUNDING * size.max())
        if not len(givers):
            raise ValueError(
                "UK diagnostics found a thin combination the published rows give "
                "back but no published row that gives it."
            )
        # A giver that no other published row can stand in for (it sits in no
        # dependency among them) carries the same coefficient in every way of
        # giving the combination back, so withholding it breaks them all. When
        # every giver has a stand-in, two relations give the combination back,
        # and withholding a giver breaks one of them.
        essential = [j for j in givers if leverage[j] > 1 - _RECOVERY_TOLERANCE]
        pick = min(
            (published[j] for j in (essential or givers)),
            key=lambda i: (carriers[i], names[i]),
        )
        withheld.add(names[pick])
    return thin, withheld - thin


def _small_recovery(
    unit: sparse.csr_matrix,
    gram: np.ndarray,
    sources: np.ndarray,
    held: list[int],
    published: list[int],
    floors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """How the published rows give back a too-small combination of held rows.

    ``unit`` holds the unit-normed rows and ``gram`` their inner products. A
    combination of the held rows is given back when what is left of it off
    the published rows' span is a negligible share of it (its squared norm
    under ``_RECOVERY_TOLERANCE`` of the combination's own). Each held row is
    tried alone first (a total less its published bands gives one back); then
    the combinations, from the generalized eigenvectors of that remainder
    against the combination's own size, so that two proportional held rows,
    whose difference is nothing, give nothing back. The first combination
    that rests on fewer distinct source households than the highest floor
    among its rows returns the minimum-norm published coefficients that give
    it back, and each published row's leverage (1 when no other published row
    can stand in for it); ``None`` when there is none.
    """

    epsilon = np.finfo(float).eps
    values, vectors = np.linalg.eigh(gram[np.ix_(published, published)])
    keep = values > values.max() * len(values) * epsilon
    values, vectors = values[keep], vectors[:, keep]
    cross = gram[np.ix_(published, held)]
    reach = (vectors.T @ cross) / np.sqrt(values)[:, None]
    own = gram[np.ix_(held, held)]
    remainder = own - reach.T @ reach

    def rests_below_floor(alpha: np.ndarray) -> bool:
        rows = np.flatnonzero(alpha)
        combination = np.asarray(unit[[held[i] for i in rows]].T @ alpha[rows]).ravel()
        size = np.abs(combination)
        if not size.any():
            return False
        support = size > _RECOVERY_ROUNDING * size.max()
        floor = max(int(floors[held[i]]) for i in rows)
        return len(pd.unique(sources[support])) < floor

    def found(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coefficients = vectors @ ((vectors.T @ (cross @ alpha)) / values)
        return coefficients, np.sum(vectors**2, axis=1)

    for row in np.flatnonzero(np.diag(remainder) < _RECOVERY_TOLERANCE):
        alpha = np.zeros(len(held))
        alpha[row] = 1.0
        if rests_below_floor(alpha):
            return found(alpha)
    own_values, own_vectors = np.linalg.eigh(own)
    nonzero = own_values > own_values.max() * len(own_values) * epsilon
    lift = own_vectors[:, nonzero] / np.sqrt(own_values[nonzero])
    share = lift.T @ remainder @ lift
    share_values, share_vectors = np.linalg.eigh((share + share.T) / 2)
    given_back = lift @ share_vectors[:, share_values < _RECOVERY_TOLERANCE]
    for alpha in _reduced_basis(given_back):
        if rests_below_floor(alpha):
            return found(alpha)
    return None


def _reduced_basis(basis: np.ndarray) -> list[np.ndarray]:
    """The reduced row echelon basis of a span given by its columns.

    A vector in the span is fixed by its entries on the pivots, so separate
    relations come out as separate vectors. Each column is scaled to a largest
    entry of one first, so the rounding share means the same for all.
    """

    reduced = basis.T.copy()
    largest = np.abs(reduced).max(axis=1, initial=0.0)
    reduced = reduced[largest > 0] / largest[largest > 0, None]
    pivot = 0
    for column in range(reduced.shape[1]):
        if pivot == reduced.shape[0]:
            break
        best = pivot + int(np.argmax(np.abs(reduced[pivot:, column])))
        if abs(reduced[best, column]) < _RECOVERY_ROUNDING:
            continue
        reduced[[pivot, best]] = reduced[[best, pivot]]
        reduced[pivot] /= reduced[pivot, column]
        for other in range(reduced.shape[0]):
            if other != pivot:
                reduced[other] -= reduced[other, column] * reduced[pivot]
        pivot += 1
    reduced[np.abs(reduced) < _RECOVERY_ROUNDING] = 0.0
    return [vector for vector in reduced[:pivot] if vector.any()]


def uk_withheld_rows(
    problem: Any,
    weights: Sequence[float] | np.ndarray,
    frame: Frame,
    target_rows: Sequence[Mapping[str, object]],
    geography_levels: Mapping[str, object] | None = None,
) -> set[str]:
    """Every row whose model figures stay in the build: :func:`uk_withheld_row_sets`."""

    thin, complementary = uk_withheld_row_sets(
        problem, weights, frame, target_rows, geography_levels
    )
    return thin | complementary


def uk_withheld_brma_rows(
    problem: Any,
    weights: Sequence[float] | np.ndarray,
    frame: Frame,
    target_rows: Sequence[Mapping[str, object]],
    geography_levels: Mapping[str, str],
) -> set[str]:
    """The BRMA rows among :func:`uk_thin_rows` (their ``carrier_floor_met`` flag)."""

    brma = [
        row for row in target_rows if geography_levels.get(str(row["name"])) == "brma"
    ]
    return (
        uk_thin_rows(problem, weights, frame, brma, geography_levels) if brma else set()
    )


def uk_withheld_flag(value: object) -> bool:
    """Whether a row's ``model_figures_withheld`` flag is set (a true bool)."""

    return isinstance(value, (bool, np.bool_)) and bool(value)


def uk_withheld_gate_keys(rows: object) -> frozenset[str]:
    """The local gate keys (``family/area_code/metric``) of withheld rows."""

    records = rows.to_dict(orient="records") if hasattr(rows, "to_dict") else rows
    return frozenset(
        f"{str(row.get('family', '')).strip()}/"
        f"{str(row.get('area_code', '')).strip()}/"
        f"{str(row.get('metric', '')).strip()}"
        for row in records or ()
        if isinstance(row, Mapping)
        and uk_withheld_flag(row.get("model_figures_withheld"))
    )


def uk_withhold_brma_figures(
    rows: Sequence[Mapping[str, object]],
    withheld: frozenset[str],
    *,
    fields: Sequence[str] = ("estimate", "relative_error", "abs_relative_error"),
) -> list[dict[str, object]]:
    """Copies of written target rows with a withheld row's model figures blank.

    While any row is withheld every row carries ``model_figures_withheld``, a
    bool, so a reader of the written table tells a withheld row from a
    missing figure.
    """

    out = []
    for row in rows:
        copy = dict(row)
        hidden = str(copy.get("name")) in withheld
        if hidden:
            for field in fields:
                if field in copy:
                    copy[field] = None
        if withheld:
            copy["model_figures_withheld"] = hidden
        else:
            copy.pop("model_figures_withheld", None)
        out.append(copy)
    return out


#: The guard's one message (the fourth review of #1305). A failed receipt
#: keeps an exception's message and the emitter sends it on, so it names no
#: place: a report path can carry the withheld row's key, and beside the
#: public register, its status.
UK_WITHHELD_ROW_NAMED_MESSAGE = (
    "The gate report names a withheld row; the build's local stderr says where."
)


class UKWithheldRowNamedError(ValueError):
    """A written gate report names a withheld row (microcosm#1262)."""

    def __init__(self) -> None:
        super().__init__(UK_WITHHELD_ROW_NAMED_MESSAGE)


def uk_refuse_named_withheld_rows(
    payload: object, *, names: frozenset[str], keys: frozenset[str]
) -> None:
    """Fail closed if a written gate report names a withheld row at all.

    María's ruling of 2026-10-19 on the third review of #1305: the fit gates
    decide on the published rows and count the withheld ones, so no gate has
    a reason to name one, and a name beside a figure or a verdict would give
    the row back. A withheld row's name, gate key or area code, in any case,
    in any key or string of the report, stops the build before the report
    leaves it. The error's message is constant; the place goes to this
    machine's stderr only.

    It matches literal text, and a bare area code: one family is bound per
    BRMA today, and a second would stop the build on its published rows
    until the guard matched each withheld row's own keys instead.
    """

    areas = {key.split("/")[1] for key in keys if key.count("/") >= 2}
    labels = {label.casefold() for label in (*names, *keys, *areas) if label}
    if not labels:
        return

    def check(text: str, path: str) -> None:
        folded = text.casefold()
        if any(label in folded for label in labels):
            print(f"error: {path} names a withheld row.", file=sys.stderr)
            raise UKWithheldRowNamedError

    def walk(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                check(str(key), f"{path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for position, item in enumerate(value):
                walk(item, f"{path}[{position}]")
        elif isinstance(value, str):
            check(value, path)

    walk(payload, "report")


_WITHHELD_TOTALS_NOTE = (
    "A target row resting on fewer than ten distinct source households keeps "
    "its place without its model figures, and so does a row that, with the "
    "published rows, would give such a row back. initial_loss, final_loss, "
    "fraction_within_10pct and past_cap_census are restated over the published "
    "rows, on the solve's own loss weights, scales and cap (initial_loss at the "
    "input weights, the rows' initial estimates); loss_trajectory, the solve's "
    "path over every row, is empty, and options keep their settings only."
)


def _restate_totals_over_published_rows(
    payload: dict[str, object],
    result: CalibrationResult,
    withheld: frozenset[str],
) -> dict[str, object]:
    """Restate the payload's totals over the published rows (microcosm#1262).

    María's ruling of 2026-10-18 on the #1305 re-review: a total over every
    row would give a withheld row back by difference against the published
    rows' contributions. The final loss becomes the published rows' summed
    contributions (each a share of the solve's whole loss weight, as the rows
    record them), the initial loss the same at the input weights, and the
    fraction within 10% and the past-cap census count the published rows
    only. The loss trajectory cannot be restated and is emptied. Returns the
    ``uk_diagnostics`` block that labels the basis.
    """

    rows = list(result.diagnostics)
    kept = [index for index, row in enumerate(rows) if str(row.name) not in withheld]

    def published_loss(estimates: Sequence[float]) -> float | None:
        try:
            weights = np.asarray(result.target_loss_weights, dtype=float)
            scales = np.asarray(result.target_loss_scales, dtype=float)
            cap = float(result.target_loss_cap)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            weights.shape != (len(rows),)
            or scales.shape != (len(rows),)
            or not np.isfinite(weights).all()
            or not np.isfinite(scales).all()
            or (scales <= 0).any()
            or not weights.sum() > 0
            or not math.isfinite(cap)
        ):
            return None
        shares = weights / weights.sum()
        terms = []
        for index in kept:
            target, estimate = float(rows[index].target), float(estimates[index])
            if not (math.isfinite(target) and math.isfinite(estimate)):
                return None
            terms.append(
                float(shares[index])
                * min(abs(estimate - target) / float(scales[index]), cap)
            )
        return math.fsum(terms)

    payload["initial_loss"] = published_loss([row.initial_estimate for row in rows])
    payload["final_loss"] = published_loss([row.final_estimate for row in rows])
    payload["fraction_within_10pct"] = (
        sum(abs(float(rows[index].relative_error)) <= 0.10 for index in kept)
        / len(kept)
        if kept
        else None
    )
    payload["loss_trajectory"] = []
    payload["options"] = uk_published_options(dict(payload.get("options") or {}))
    payload["past_cap_census"] = past_cap_census(
        SimpleNamespace(
            options=getattr(result, "options", None),
            diagnostics=[rows[index] for index in kept],
        )
    )
    return {
        "targets": len(withheld),
        "min_source_households": UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS,
        "totals_basis": "published_rows",
        "note": _WITHHELD_TOTALS_NOTE,
    }


def uk_published_row_losses(
    result: CalibrationResult, withheld: frozenset[str]
) -> tuple[float | None, float | None]:
    """The initial and final loss over the published rows (microcosm#1262).

    For a manifest that would otherwise carry the solve's loss over every row,
    which against the published rows' contributions gives a withheld row back
    (María's ruling of 2026-10-18); ``None`` where the result cannot restate.
    """

    totals: dict[str, object] = {}
    _restate_totals_over_published_rows(totals, result, frozenset(withheld))
    return totals["initial_loss"], totals["final_loss"]


def uk_published_options(options: Mapping[str, object]) -> dict[str, object]:
    """A payload's solve options while any row is withheld (microcosm#1262).

    The third review of #1305: the solve keeps its best iterate's loss over
    every row in ``iterate_selection_receipt``, and that loss less the
    restated final loss gives the withheld rows' contribution back. Settings
    on the allow-list stay; every other option, and the receipt's two
    iterate losses, read null.
    """

    published: dict[str, object] = {}
    for key, value in options.items():
        if key in UK_WITHHELD_PUBLISHED_OPTIONS:
            published[key] = value
        elif key == "iterate_selection_receipt" and isinstance(value, Mapping):
            published[key] = {
                field: (item if field in UK_WITHHELD_PUBLISHED_ITERATE_FIELDS else None)
                for field, item in value.items()
            }
        else:
            published[key] = None
    return published


def _target_pass_rates(
    target_rows: list[dict[str, object]],
    skipped_names: Sequence[str],
    geography_levels: Mapping[str, str],
    withheld: frozenset[str] = frozenset(),
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
        if name in withheld:
            # Scored, never counted as passing: its outcome stays unknowable.
            counts[level]["n_targets"] += 1
            counts[level]["n_scored"] += 1
            continue
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
    withheld: Collection[str] | None = None,
) -> dict[str, object]:
    """Render shared diagnostics plus the versioned UK release evidence.

    ``target_geography_levels`` is deliberately explicit and exact.  Target
    names are never parsed to guess their geography, and a missing, stale, or
    unknown mapping aborts instead of silently shrinking a scoreboard level.

    ``withheld`` names the rows whose model figures stay in the build
    (:func:`uk_withheld_rows`); the gate kernel passes the set its gate
    evidence carries, and by default it is counted here on the same rule.
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
    if withheld is None:
        thin, complementary = uk_withheld_row_sets(
            result.problem, result.weights, frame, target_rows, geography
        )
        withheld = frozenset(thin | complementary)
    else:
        withheld = frozenset(withheld)
        # The floor's own verdict, so a row the caller withholds beside a
        # thin one is told apart from the thin row itself.
        thin = (
            uk_thin_rows(result.problem, result.weights, frame, target_rows, geography)
            if withheld
            else set()
        )
    for row in target_rows:
        name = str(row["name"])
        if geography.get(name) == "brma":
            # Every BRMA row states its ten-household verdict, so the release
            # contract can see one that skipped the floor (María's ruling of
            # 2026-10-19 on the fourth review of #1305: it warns).
            row["metadata"] = {
                **dict(row.get("metadata") or {}),
                "carrier_floor_met": name not in thin,
            }
        if name in withheld:
            for field in _MODEL_FIGURE_FIELDS:
                if field in row:
                    row[field] = None
            row["metadata"] = {
                **dict(row.get("metadata") or {}),
                "model_figures_withheld": (
                    _WITHHELD_NOTE if name in thin else _COMPLEMENT_NOTE
                ),
                **({} if name in thin else {"complementary_withholding": True}),
            }
    pass_rates = _target_pass_rates(target_rows, skipped, geography, withheld)
    withheld_block = (
        _restate_totals_over_published_rows(payload, result, withheld)
        if withheld
        else None
    )
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
        # A withheld row stays out of every rollup, so that no family or area
        # total can be differenced against the published rows to recover it,
        # and no rollup reads a figure the row no longer carries.
        published = [row for row in target_rows if str(row["name"]) not in withheld]
        uk_diagnostics["weakest_families"] = uk_weakest_families(published)
        uk_diagnostics["weakest_areas_by_fit"] = uk_weakest_areas_by_fit(
            published,
            local_area_support,
        )
    if rotated_holdout is not None:
        uk_diagnostics["rotated_holdout"] = dict(rotated_holdout)
    drift = uk_consumption_drift(household, result.initial_weights, result_weights)
    if drift is not None:
        uk_diagnostics["consumption_drift"] = drift
    if withheld_block is not None:
        uk_diagnostics["model_figures_withheld"] = withheld_block
    payload["uk_diagnostics"] = uk_diagnostics
    return CalibrationDiagnosticsV8.model_validate(payload).model_dump(
        mode="python",
        exclude_defaults=True,
    )


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
    withheld: Collection[str] | None = None,
) -> DiagnosticsWriteOutcome:
    """Validate and write shared-plus-UK diagnostics through the canonical writer.

    ``withheld`` is the payload's: the rows whose model figures stay in the
    build, counted on the payload's rule when not given.
    """

    output = Path(path)
    try:
        payload = uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=target_geography_levels,
            target_registry=target_registry,
            stratum_columns=stratum_columns,
            build=build,
            local_area_support=local_area_support,
            rotated_holdout=rotated_holdout,
            withheld=withheld,
        )
        diagnostics = CalibrationDiagnosticsV8.model_validate(payload)
    except ValidationError as error:
        return failed_diagnostics_outcome(
            error,
            error_code="validation_error",
            path=output,
        )
    except Exception as error:  # diagnostics must not escape without validation
        return failed_diagnostics_outcome(error, path=output)
    return write_typed_calibration_diagnostics(diagnostics, output)
