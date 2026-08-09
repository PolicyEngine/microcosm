"""Longwise UK local-geography build primitives.

This module owns the representation that lets Microcosm replace the legacy
UK incumbent ``areas x households`` matrix artifacts:

* a stacked sparse matrix whose columns are
  ``area_index * n_households + household_index``; and
* a longweight sidecar with one row per non-zero
  ``(area, household, weight)`` assignment.

The code is deliberately independent of the incumbent UK data package. Target
providers may read HMRC, ONS, DWP, or other public target files, and engine
runners may compute household metrics with ``policyengine-uk``; those are
inputs to these functions rather than imports hidden inside Microcosm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

LONG_GEOGRAPHY_COLUMNS = (
    "area_type",
    "area_code",
    "area_index",
    "household_index",
    "household_id",
    "source_year",
    "source_household_id",
    "source_household_key",
    "clone_index",
    "weight",
    "weight_source",
)

_AREA_METADATA_COLUMNS = frozenset(
    {
        "area_code",
        "area_index",
        "area_name",
        "area_type",
        "code",
        "country",
        "name",
    }
)


@dataclass(frozen=True)
class StackedLocalMatrix:
    """Sparse stacked local-area calibration matrix and aligned targets."""

    matrix: sp.csr_matrix
    targets: np.ndarray
    target_frame: pd.DataFrame
    area_codes: tuple[str, ...]
    metric_names: tuple[str, ...]
    n_households: int

    @property
    def n_areas(self) -> int:
        """Number of local areas represented in the stack."""
        return len(self.area_codes)


def sort_households_by_id(
    household: pd.DataFrame,
    *,
    id_column: str = "household_id",
) -> pd.DataFrame:
    """Return a stable household-ID-sorted frame, rejecting ambiguous IDs.

    This is the Microcosm-side guard for the 2024-25 FRS bug class where
    household IDs were sorted but household attributes were assigned by raw
    row position. Call this before positional household arrays are attached to
    an ID frame.
    """

    if id_column not in household.columns:
        raise ValueError(f"household frame is missing {id_column!r}.")
    if household[id_column].isna().any():
        raise ValueError(f"{id_column!r} contains missing values.")
    duplicated = household[id_column][household[id_column].duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            f"{id_column!r} must be unique; duplicate value(s): "
            f"{list(map(str, duplicated[:5]))}."
        )
    return household.sort_values(id_column, kind="mergesort").reset_index(drop=True)


def align_area_targets(
    targets: pd.DataFrame,
    area_codes: Sequence[str],
    *,
    metric_names: Sequence[str] | None = None,
    code_column: str = "code",
) -> pd.DataFrame:
    """Align target rows to ``area_codes`` and return metric columns only."""

    codes = _area_code_tuple(area_codes)
    frame = targets.copy()
    if code_column in frame.columns:
        frame[code_column] = frame[code_column].astype(str)
        frame = frame.set_index(code_column, drop=True)
    else:
        frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()
        raise ValueError(
            f"target area code index must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )

    missing = [code for code in codes if code not in frame.index]
    if missing:
        raise ValueError(f"target frame is missing area code(s): {missing[:5]}.")

    if metric_names is None:
        metric_names = [
            col
            for col in frame.columns
            if col not in _AREA_METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(frame[col])
        ]
    metrics = tuple(str(name) for name in metric_names)
    absent = [name for name in metrics if name not in frame.columns]
    if absent:
        raise ValueError(f"target frame is missing metric column(s): {absent}.")

    aligned = frame.loc[list(codes), list(metrics)].astype(float)
    values = aligned.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("target values must all be finite.")
    return aligned


def build_stacked_local_matrix(
    metrics: pd.DataFrame | Mapping[str, pd.DataFrame],
    targets: pd.DataFrame,
    *,
    area_codes: Sequence[str] | None = None,
    area_groups: Mapping[str, str] | None = None,
    household_ids: Sequence[Any] | None = None,
    area_type: str = "constituency",
    code_column: str = "code",
) -> StackedLocalMatrix:
    """Build a block-local sparse matrix for local-area calibration.

    Args:
        metrics: Household-level metric columns. Pass a single DataFrame when
            every area uses the same policy context, or a mapping such as
            ``{"England": df, "Scotland": df}`` with ``area_groups`` when
            metrics differ by country/devolution block.
        targets: Area target frame. Rows are aligned by ``code_column`` and
            columns must match the metric columns.
        area_codes: Canonical area order. Defaults to the target frame order.
        area_groups: Area-code to metric-table key mapping, required when
            ``metrics`` is a mapping.
        household_ids: Optional explicit household order. When supplied, every
            metric table must be indexed by these IDs in exactly this order.
            When omitted, grouped metric tables must still share an identical
            index so rows cannot drift across devolution blocks silently.
        area_type: Stored on ``target_frame`` for diagnostics.
        code_column: Target frame column holding area codes.

    Returns:
        A :class:`StackedLocalMatrix` whose column order is
        ``area_index * n_households + household_index``.
    """

    if area_codes is None:
        if code_column not in targets.columns:
            raise ValueError(
                "area_codes must be supplied when targets has no "
                f"{code_column!r} column."
            )
        area_codes = targets[code_column].astype(str).tolist()
    codes = _area_code_tuple(area_codes)
    metric_tables, groups = _normalise_metric_tables(
        metrics,
        area_codes=codes,
        area_groups=area_groups,
        household_ids=household_ids,
    )
    first = next(iter(metric_tables.values()))
    metric_names = tuple(str(col) for col in first.columns)
    target_values = align_area_targets(
        targets,
        codes,
        metric_names=metric_names,
        code_column=code_column,
    )

    n_households = len(first)
    n_areas = len(codes)
    n_metrics = len(metric_names)
    n_targets = n_areas * n_metrics
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    target_rows: list[dict[str, Any]] = []
    nonzero_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for area_index, area_code in enumerate(codes):
        group = groups[area_code]
        for metric_index, metric_name in enumerate(metric_names):
            target_index = area_index * n_metrics + metric_index
            target_rows.append(
                {
                    "target_index": target_index,
                    "area_type": area_type,
                    "area_code": area_code,
                    "area_index": area_index,
                    "area_group": group,
                    "metric": metric_name,
                    "metric_index": metric_index,
                    "value": float(target_values.loc[area_code, metric_name]),
                }
            )
            cache_key = (group, metric_index)
            if cache_key not in nonzero_cache:
                column = metric_tables[group].iloc[:, metric_index].to_numpy(
                    dtype=np.float64
                )
                if not np.isfinite(column).all():
                    raise ValueError(
                        f"metric {metric_name!r} for group {group!r} "
                        "contains non-finite values."
                    )
                nz = np.flatnonzero(column)
                nonzero_cache[cache_key] = (nz, column[nz])
            nz, values = nonzero_cache[cache_key]
            if len(nz) == 0:
                continue
            rows.append(np.full(len(nz), target_index, dtype=np.int64))
            cols.append((area_index * n_households + nz).astype(np.int64))
            data.append(values.astype(np.float64, copy=False))

    if rows:
        row_array = np.concatenate(rows)
        col_array = np.concatenate(cols)
        data_array = np.concatenate(data)
    else:
        row_array = np.array([], dtype=np.int64)
        col_array = np.array([], dtype=np.int64)
        data_array = np.array([], dtype=np.float64)
    matrix = sp.csr_matrix(
        (data_array, (row_array, col_array)),
        shape=(n_targets, n_areas * n_households),
        dtype=np.float64,
    )
    target_frame = pd.DataFrame(target_rows)
    return StackedLocalMatrix(
        matrix=matrix,
        targets=target_frame["value"].to_numpy(dtype=np.float64),
        target_frame=target_frame,
        area_codes=codes,
        metric_names=metric_names,
        n_households=n_households,
    )


def stacked_design_weights(
    base_weights: Sequence[float],
    n_areas: int,
    *,
    min_weight: float = 0.0,
) -> np.ndarray:
    """Tile base design weights across areas, splitting mass evenly."""

    weights = np.asarray(base_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("base_weights must be one-dimensional.")
    if n_areas <= 0:
        raise ValueError("n_areas must be positive.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("base_weights must be finite and non-negative.")
    if not np.isfinite(min_weight) or min_weight < 0:
        raise ValueError("min_weight must be finite and non-negative.")
    return np.maximum(np.tile(weights / n_areas, n_areas), min_weight)


def stacked_weights_to_long(
    weights: Sequence[float],
    area_codes: Sequence[str],
    household_ids: Sequence[Any],
    *,
    area_type: str,
    household_frame: pd.DataFrame | None = None,
    source_year: int | None = None,
    weight_source: str = "populace_local_stacked",
    drop_zero: bool = True,
) -> pd.DataFrame:
    """Convert stacked weights to the long local-geography sidecar format."""

    codes = _area_code_tuple(area_codes)
    hh_ids = np.asarray(household_ids)
    n_areas = len(codes)
    n_households = len(hh_ids)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    expected = n_areas * n_households
    if len(w) != expected:
        raise ValueError(
            f"weights length must equal n_areas * n_households "
            f"({expected}), got {len(w)}."
        )
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative.")

    household_frame = _align_household_frame(household_frame, hh_ids)
    source_year_values = _metadata_values(
        household_frame,
        "source_year",
        default=source_year,
        length=n_households,
    )
    source_household_ids = _metadata_values(
        household_frame,
        "source_household_id",
        default=hh_ids,
        length=n_households,
    )
    source_keys = _metadata_values(
        household_frame,
        "source_household_key",
        default=_source_keys(source_year_values, source_household_ids),
        length=n_households,
    )
    clone_index = _metadata_values(
        household_frame,
        "clone_index",
        default=0,
        length=n_households,
    )

    area_index = np.repeat(np.arange(n_areas, dtype=np.int64), n_households)
    household_index = np.tile(np.arange(n_households, dtype=np.int64), n_areas)
    out = pd.DataFrame(
        {
            "area_type": area_type,
            "area_code": np.repeat(np.asarray(codes, dtype=object), n_households),
            "area_index": area_index,
            "household_index": household_index,
            "household_id": np.tile(hh_ids, n_areas),
            "source_year": np.tile(source_year_values, n_areas),
            "source_household_id": np.tile(source_household_ids, n_areas),
            "source_household_key": np.tile(source_keys, n_areas),
            "clone_index": np.tile(clone_index, n_areas),
            "weight": w,
            "weight_source": weight_source,
        }
    )
    if drop_zero:
        out = out[out["weight"] != 0].reset_index(drop=True)
    return out.loc[:, LONG_GEOGRAPHY_COLUMNS]


def area_support_summary(
    long_weights: pd.DataFrame,
    *,
    area_codes: Sequence[str] | None = None,
    area_type: str | None = None,
) -> pd.DataFrame:
    """Summarize non-zero household support by local area.

    Passing ``area_codes`` includes requested areas with no positive assigned
    households, which is important for sparse/L0 local solves.
    """

    missing = sorted(set(LONG_GEOGRAPHY_COLUMNS) - set(long_weights.columns))
    if missing:
        raise ValueError(f"long weight frame is missing column(s): {missing}.")
    positive = long_weights[long_weights["weight"] > 0]
    summary = (
        positive.groupby(["area_type", "area_code"], sort=True)
        .agg(
            nonzero_households=("household_id", "nunique"),
            nonzero_source_households=("source_household_key", "nunique"),
            weight_sum=("weight", "sum"),
            max_weight=("weight", "max"),
            effective_sample_size=("weight", _effective_sample_size),
        )
        .reset_index()
    )
    if area_codes is None:
        return summary

    codes = _area_code_tuple(area_codes)
    if area_type is None:
        area_types = long_weights["area_type"].dropna().unique()
        if len(area_types) != 1:
            raise ValueError(
                "area_type must be supplied when area_codes are supplied and "
                "long_weights does not contain exactly one area_type."
            )
        area_type = str(area_types[0])
    full = pd.DataFrame(
        {
            "area_type": area_type,
            "area_code": list(codes),
        }
    )
    completed = full.merge(summary, on=["area_type", "area_code"], how="left")
    completed["nonzero_households"] = (
        completed["nonzero_households"].fillna(0).astype(int)
    )
    completed["nonzero_source_households"] = (
        completed["nonzero_source_households"].fillna(0).astype(int)
    )
    completed["weight_sum"] = completed["weight_sum"].fillna(0.0).astype(float)
    completed["max_weight"] = completed["max_weight"].fillna(0.0).astype(float)
    completed["effective_sample_size"] = (
        completed["effective_sample_size"].fillna(0.0).astype(float)
    )
    return completed


def write_long_geography_weights(
    long_weights: pd.DataFrame,
    path: str | Path,
) -> None:
    """Write a long local-geography sidecar as ``csv`` or ``csv.gz``."""

    missing = sorted(set(LONG_GEOGRAPHY_COLUMNS) - set(long_weights.columns))
    if missing:
        raise ValueError(f"long weight frame is missing column(s): {missing}.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    long_weights.loc[:, LONG_GEOGRAPHY_COLUMNS].to_csv(path, index=False)


def _normalise_metric_tables(
    metrics: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    area_codes: tuple[str, ...],
    area_groups: Mapping[str, str] | None,
    household_ids: Sequence[Any] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    expected_index = None if household_ids is None else pd.Index(household_ids)
    if isinstance(metrics, pd.DataFrame):
        tables = {
            "__all__": _validated_metric_frame(
                metrics,
                group="__all__",
                expected_index=expected_index,
            )
        }
        return tables, {code: "__all__" for code in area_codes}

    if area_groups is None:
        raise ValueError("area_groups is required when metrics is a mapping.")
    tables = {
        str(group): _validated_metric_frame(
            frame,
            group=str(group),
            expected_index=expected_index,
        )
        for group, frame in metrics.items()
    }
    if not tables:
        raise ValueError("metrics mapping must not be empty.")
    groups = {str(code): str(group) for code, group in area_groups.items()}
    missing_group_codes = [code for code in area_codes if code not in groups]
    if missing_group_codes:
        raise ValueError(f"area_groups is missing area code(s): {missing_group_codes}.")
    unknown_groups = sorted({groups[code] for code in area_codes} - set(tables))
    if unknown_groups:
        raise ValueError(f"area_groups references unknown group(s): {unknown_groups}.")

    first = next(iter(tables.values()))
    for group, frame in tables.items():
        if len(frame) != len(first):
            raise ValueError(
                f"metric table {group!r} has {len(frame)} rows; expected "
                f"{len(first)}."
            )
        if not frame.index.equals(first.index):
            raise ValueError(
                f"metric table {group!r} household index does not match the "
                "first metric table."
            )
        if tuple(frame.columns) != tuple(first.columns):
            raise ValueError(
                f"metric table {group!r} columns do not match the first table."
            )
    return tables, groups


def _validated_metric_frame(
    frame: pd.DataFrame,
    *,
    group: str,
    expected_index: pd.Index | None,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"metric table {group!r} must not be empty.")
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()
        raise ValueError(
            f"metric table {group!r} household index must be unique; "
            f"duplicate value(s): {list(map(str, duplicates[:5]))}."
        )
    if expected_index is not None and not frame.index.equals(expected_index):
        raise ValueError(
            f"metric table {group!r} household index must match household_ids."
        )
    non_numeric = [
        col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    if non_numeric:
        raise ValueError(
            f"metric table {group!r} has non-numeric column(s): {non_numeric}."
        )
    return frame.copy()


def _area_code_tuple(area_codes: Sequence[str]) -> tuple[str, ...]:
    codes = tuple(str(code) for code in area_codes)
    if not codes:
        raise ValueError("area_codes must not be empty.")
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        raise ValueError(f"area_codes must be unique; duplicates: {duplicates}.")
    return codes


def _metadata_values(
    household_frame: pd.DataFrame | None,
    column: str,
    *,
    default: Any,
    length: int,
) -> np.ndarray:
    if household_frame is not None and column in household_frame.columns:
        values = household_frame[column].to_numpy()
    elif np.ndim(default) == 1:
        values = np.asarray(default)
    else:
        values = np.repeat(default, length)
    if len(values) != length:
        raise ValueError(
            f"{column!r} metadata length must equal household count "
            f"({length}), got {len(values)}."
        )
    return values


def _align_household_frame(
    household_frame: pd.DataFrame | None,
    household_ids: np.ndarray,
) -> pd.DataFrame | None:
    if household_frame is None:
        return None
    if "household_id" not in household_frame.columns:
        raise ValueError("household_frame must include 'household_id'.")
    if household_frame["household_id"].isna().any():
        raise ValueError("household_frame household_id contains missing values.")
    if household_frame["household_id"].duplicated().any():
        duplicates = household_frame.loc[
            household_frame["household_id"].duplicated(), "household_id"
        ].unique()
        raise ValueError(
            "household_frame household_id must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    aligned = household_frame.set_index("household_id", drop=False).reindex(
        pd.Index(household_ids)
    )
    if aligned["household_id"].isna().any():
        missing = pd.Index(household_ids)[aligned["household_id"].isna()].tolist()
        raise ValueError(
            f"household_frame is missing household_id value(s): {missing[:5]}."
        )
    return aligned.reset_index(drop=True)


def _source_keys(
    source_year: Sequence[Any],
    source_household_id: Sequence[Any],
) -> np.ndarray:
    year_values = np.asarray(source_year, dtype=object)
    id_values = np.asarray(source_household_id, dtype=object)
    keys = []
    for year, household_id in zip(year_values, id_values, strict=True):
        if year is None or pd.isna(year):
            keys.append(str(household_id))
        else:
            keys.append(f"{year}:{household_id}")
    return np.asarray(keys, dtype=object)


def _effective_sample_size(weights: pd.Series) -> float:
    values = weights.to_numpy(dtype=np.float64)
    square_sum = float(np.square(values).sum())
    if square_sum == 0:
        return 0.0
    return float(values.sum() ** 2 / square_sum)
