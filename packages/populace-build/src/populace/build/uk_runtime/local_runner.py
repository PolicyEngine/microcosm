"""Reusable runner pieces for Populace-owned UK local builds.

The core local-geography modules deliberately avoid data-package imports:
target providers hand Populace explicit area tables, and engine runners hand it
household metric tables. This module is the thin orchestration layer that ties
those pieces together for pilot and full UK local candidate builds.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.uk_runtime.local_geography import (
    StackedLocalMatrix,
    area_support_summary,
    build_stacked_local_matrix,
    sort_households_by_id,
    stacked_weights_to_long,
    write_long_geography_weights,
)
from populace.build.uk_runtime.local_solver import (
    StackedLocalSolveResult,
    solve_stacked_local_weights,
)
from populace.build.uk_runtime.local_targets import (
    COUNTRY_TO_REGION,
    area_groups_from_codes,
    compute_household_metrics,
)


@dataclass(frozen=True)
class UKLocalCandidateResult:
    """Solved UK local candidate outputs and diagnostics."""

    problem: StackedLocalMatrix
    solve_result: StackedLocalSolveResult
    long_weights: pd.DataFrame
    support_summary: pd.DataFrame


def read_local_table(path: str | Path) -> pd.DataFrame:
    """Read a local-build input table from CSV, CSV.GZ, or Parquet."""

    table_path = Path(path)
    suffixes = [suffix.lower() for suffix in table_path.suffixes]
    if not suffixes:
        raise ValueError(f"Cannot infer table format for {table_path}.")
    if suffixes[-1] == ".csv" or suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(table_path)
    if suffixes[-1] in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)
    raise ValueError(
        f"Unsupported local table format for {table_path}; expected CSV, "
        "CSV.GZ, or Parquet."
    )


def prepare_area_frame(
    area_frame: pd.DataFrame | str | Path,
    *,
    code_column: str = "code",
    group_column: str = "country",
    sort_by_code: bool = True,
    max_areas: int | None = None,
) -> pd.DataFrame:
    """Return a validated, canonical area frame.

    ``code_column`` supplies the output area order. If ``group_column`` is
    present it is authoritative for country/devolution grouping; otherwise the
    ONS area-code prefix is used later by :func:`area_groups_from_codes`.
    """

    frame = _as_frame(area_frame).copy()
    if code_column not in frame.columns:
        raise ValueError(f"area frame is missing {code_column!r}.")
    frame[code_column] = _normalise_nonblank_strings(
        frame[code_column],
        column=code_column,
    )
    if frame[code_column].duplicated().any():
        duplicates = frame.loc[frame[code_column].duplicated(), code_column].unique()
        raise ValueError(
            f"area codes must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )

    if group_column in frame.columns:
        frame[group_column] = _normalise_nonblank_strings(
            frame[group_column],
            column=group_column,
        )

    if sort_by_code:
        frame = frame.sort_values(code_column, kind="mergesort")
    if max_areas is not None:
        if not isinstance(max_areas, int) or max_areas <= 0:
            raise ValueError("max_areas must be a positive integer when supplied.")
        frame = frame.head(max_areas)
    return frame.reset_index(drop=True)


def prepare_household_frame(
    household_frame: pd.DataFrame | str | Path,
    *,
    id_column: str = "household_id",
    weight_column: str = "household_weight",
    source_year: int | None = None,
) -> pd.DataFrame:
    """Sort households by ID, validate weights, and attach lineage columns."""

    frame = sort_households_by_id(_as_frame(household_frame), id_column=id_column)
    if id_column != "household_id":
        frame = frame.rename(columns={id_column: "household_id"})
    if weight_column not in frame.columns:
        raise ValueError(f"household frame is missing {weight_column!r}.")
    weights = frame[weight_column].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"{weight_column!r} must be finite and non-negative.")

    if weight_column != "household_weight":
        frame = frame.rename(columns={weight_column: "household_weight"})
    if "source_household_id" not in frame.columns:
        frame["source_household_id"] = frame["household_id"]
    if "source_year" not in frame.columns and source_year is not None:
        frame["source_year"] = source_year
    if "clone_index" not in frame.columns:
        frame["clone_index"] = 0
    if "source_household_key" not in frame.columns:
        frame["source_household_key"] = _source_household_keys(
            frame,
            source_year=source_year,
        )
    return frame.reset_index(drop=True)


def load_metric_tables(
    paths: Mapping[str, str | Path],
    *,
    household_id_column: str = "household_id",
) -> dict[str, pd.DataFrame]:
    """Load explicit household metric tables keyed by area group/country."""

    if not paths:
        raise ValueError("metric table paths must not be empty.")
    return {
        str(group): _metric_table_from_frame(
            read_local_table(path),
            household_id_column=household_id_column,
            group=str(group),
        )
        for group, path in paths.items()
    }


def load_uk_dataset(path: str | Path) -> Any:
    """Load a PolicyEngine-UK single-year H5 dataset lazily."""

    try:
        from policyengine_uk.data import UKSingleYearDataset
    except ImportError as exc:  # pragma: no cover - exercised only with engine absent
        raise ImportError(
            "Loading a UK H5 dataset requires policyengine-uk. Install the "
            "UK engine before calling load_uk_dataset()."
        ) from exc
    return UKSingleYearDataset(file_path=str(path))


def set_simulation_area_group(
    sim: Any,
    group: str,
    *,
    period: int | str,
    n_households: int | None = None,
    region_variable: str = "region",
) -> Any:
    """Set a PolicyEngine-UK-like simulation to the area's devolution group."""

    if group not in COUNTRY_TO_REGION:
        raise ValueError(
            f"Unknown UK area group {group!r}; expected one of "
            f"{tuple(COUNTRY_TO_REGION)}."
        )
    if n_households is None:
        household_ids = _values(
            sim.calculate("household_id", period=period, map_to="household")
        )
        n_households = len(household_ids)
    sim.set_input(
        region_variable,
        period,
        np.asarray([COUNTRY_TO_REGION[group]] * n_households, dtype=object),
    )
    return sim


def build_metric_tables_from_dataset(
    dataset: Any,
    area_groups: Mapping[str, str],
    area_type: str,
    *,
    period: int | str | None = None,
    household_ids: Sequence[Any] | None = None,
    simulation_factory: Callable[[Any], Any] | None = None,
    target_profile: Mapping[str, Any] | Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute household metric tables once per country/devolution group."""

    if not area_groups:
        raise ValueError("area_groups must not be empty.")
    run_period = _infer_period(dataset, period)
    if simulation_factory is None:
        simulation_factory = _default_uk_simulation_factory
    tables: dict[str, pd.DataFrame] = {}
    for group in sorted(set(area_groups.values())):
        sim = simulation_factory(dataset)
        simulation_household_ids = _simulation_household_ids(sim, run_period)
        set_simulation_area_group(
            sim,
            group,
            period=run_period,
            n_households=len(simulation_household_ids),
        )
        metric_kwargs: dict[str, Any] = {"period": run_period}
        if target_profile is not None:
            metric_kwargs["target_profile"] = target_profile
        table = compute_household_metrics(sim, area_type, **metric_kwargs)
        if not table.index.equals(pd.Index(simulation_household_ids)):
            raise ValueError(
                f"metric table {group!r} index must match simulation "
                "household_id order."
            )
        if household_ids is not None:
            table = _align_metric_table_to_households(
                table,
                household_ids,
                group=group,
            )
        tables[group] = table
    return tables


def build_local_candidate(
    *,
    area_type: str,
    area_frame: pd.DataFrame | str | Path,
    targets: pd.DataFrame | str | Path,
    metrics: pd.DataFrame | Mapping[str, pd.DataFrame],
    household_frame: pd.DataFrame | str | Path,
    code_column: str = "code",
    group_column: str = "country",
    sort_areas_by_code: bool = True,
    max_areas: int | None = None,
    source_year: int | None = None,
    weight_source: str = "populace_uk_local",
    solver_options: Mapping[str, Any] | None = None,
) -> UKLocalCandidateResult:
    """Build, solve, and export a UK local candidate in longwise form."""

    areas = prepare_area_frame(
        area_frame,
        code_column=code_column,
        group_column=group_column,
        sort_by_code=sort_areas_by_code,
        max_areas=max_areas,
    )
    households = prepare_household_frame(
        household_frame,
        source_year=source_year,
    )
    area_codes = tuple(areas[code_column].astype(str))
    area_groups = area_groups_from_codes(
        areas,
        code_column=code_column,
        group_column=group_column,
    )
    household_ids = households["household_id"].to_numpy()
    base_weights = households["household_weight"].to_numpy(dtype=np.float64)
    target_frame = _as_frame(targets)
    problem = build_stacked_local_matrix(
        metrics,
        target_frame,
        area_codes=area_codes,
        area_groups=area_groups,
        household_ids=household_ids,
        area_type=area_type,
        code_column=code_column,
    )
    solve_result = solve_stacked_local_weights(
        problem,
        base_weights,
        **dict(solver_options or {}),
    )
    long_weights = stacked_weights_to_long(
        solve_result.weights,
        area_codes,
        household_ids,
        area_type=area_type,
        household_frame=households,
        source_year=source_year,
        weight_source=weight_source,
    )
    return UKLocalCandidateResult(
        problem=problem,
        solve_result=solve_result,
        long_weights=long_weights,
        support_summary=area_support_summary(
            long_weights,
            area_codes=area_codes,
            area_type=area_type,
        ),
    )


def build_local_candidate_from_dataset(
    dataset: Any | str | Path,
    *,
    area_type: str,
    area_frame: pd.DataFrame | str | Path,
    targets: pd.DataFrame | str | Path,
    household_frame: pd.DataFrame | str | Path,
    period: int | str | None = None,
    code_column: str = "code",
    group_column: str = "country",
    sort_areas_by_code: bool = True,
    max_areas: int | None = None,
    source_year: int | None = None,
    weight_source: str = "populace_uk_local",
    simulation_factory: Callable[[Any], Any] | None = None,
    target_profile: Mapping[str, Any] | Any | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> UKLocalCandidateResult:
    """Build a UK local candidate from a Populace UK H5 or dataset object."""

    dataset_obj = (
        load_uk_dataset(dataset) if isinstance(dataset, str | Path) else dataset
    )
    areas = prepare_area_frame(
        area_frame,
        code_column=code_column,
        group_column=group_column,
        sort_by_code=sort_areas_by_code,
        max_areas=max_areas,
    )
    households = prepare_household_frame(
        household_frame,
        source_year=source_year,
    )
    area_groups = area_groups_from_codes(
        areas,
        code_column=code_column,
        group_column=group_column,
    )
    household_ids = households["household_id"].to_numpy()
    metrics = build_metric_tables_from_dataset(
        dataset_obj,
        area_groups,
        area_type,
        period=period,
        household_ids=household_ids,
        simulation_factory=simulation_factory,
        target_profile=target_profile,
    )
    return build_local_candidate(
        area_type=area_type,
        area_frame=areas,
        targets=targets,
        metrics=metrics,
        household_frame=households,
        code_column=code_column,
        group_column=group_column,
        sort_areas_by_code=False,
        source_year=source_year,
        weight_source=weight_source,
        solver_options=solver_options,
    )


def summarize_local_candidate(result: UKLocalCandidateResult) -> dict[str, Any]:
    """Return a compact JSON-serializable summary for candidate run logs."""

    support = result.support_summary
    return {
        "area_type": (
            None
            if result.solve_result.diagnostics.empty
            else str(result.solve_result.diagnostics["area_type"].iloc[0])
        ),
        "n_areas": int(result.problem.n_areas),
        "n_households": int(result.problem.n_households),
        "n_targets": int(len(result.problem.targets)),
        "n_long_rows": int(len(result.long_weights)),
        "n_nonzero": int(result.solve_result.n_nonzero),
        "initial_loss": float(result.solve_result.initial_loss),
        "final_loss": float(result.solve_result.final_loss),
        "weight_sum": float(result.long_weights["weight"].sum()),
        "min_area_support": (
            0 if support.empty else int(support["nonzero_households"].min())
        ),
        "median_area_support": (
            0.0 if support.empty else float(support["nonzero_households"].median())
        ),
        "max_area_support": (
            0 if support.empty else int(support["nonzero_households"].max())
        ),
        "min_area_source_support": (
            0 if support.empty else int(support["nonzero_source_households"].min())
        ),
        "median_area_source_support": (
            0.0
            if support.empty
            else float(support["nonzero_source_households"].median())
        ),
        "max_area_source_support": (
            0 if support.empty else int(support["nonzero_source_households"].max())
        ),
        "min_area_effective_sample_size": (
            0.0 if support.empty else float(support["effective_sample_size"].min())
        ),
        "median_area_effective_sample_size": (
            0.0 if support.empty else float(support["effective_sample_size"].median())
        ),
    }


def write_local_candidate_outputs(
    result: UKLocalCandidateResult,
    output_dir: str | Path,
    *,
    weights_filename: str = "local_geography_weights.csv.gz",
) -> dict[str, Any]:
    """Write long weights, diagnostics, support, and a JSON summary."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_long_geography_weights(result.long_weights, out / weights_filename)
    result.solve_result.diagnostics.to_csv(out / "solve_diagnostics.csv", index=False)
    result.support_summary.to_csv(out / "area_support_summary.csv", index=False)
    summary = summarize_local_candidate(result)
    (out / "solve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def _as_frame(frame_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path.copy()
    return read_local_table(frame_or_path)


def _normalise_nonblank_strings(values: pd.Series, *, column: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{column!r} must not contain missing values.")
    strings = values.astype(str).str.strip()
    if (strings == "").any():
        raise ValueError(f"{column!r} must not contain blank values.")
    return strings


def _source_household_keys(
    household_frame: pd.DataFrame,
    *,
    source_year: int | None,
) -> list[str]:
    if "source_year" in household_frame.columns:
        years = household_frame["source_year"].tolist()
    else:
        years = [source_year] * len(household_frame)
    keys = []
    for year, source_id in zip(
        years,
        household_frame["source_household_id"],
        strict=True,
    ):
        if year is None or pd.isna(year):
            keys.append(str(source_id))
        else:
            keys.append(f"{year}:{source_id}")
    return keys


def _metric_table_from_frame(
    frame: pd.DataFrame,
    *,
    household_id_column: str,
    group: str,
) -> pd.DataFrame:
    if household_id_column not in frame.columns:
        raise ValueError(f"metric table {group!r} is missing {household_id_column!r}.")
    table = frame.copy()
    if table[household_id_column].isna().any():
        raise ValueError(
            f"metric table {group!r} {household_id_column!r} must not contain "
            "missing values."
        )
    if pd.api.types.is_string_dtype(table[household_id_column]):
        strings = table[household_id_column].astype(str).str.strip()
        if (strings == "").any():
            raise ValueError(
                f"metric table {group!r} {household_id_column!r} must not "
                "contain blank values."
            )
        table[household_id_column] = strings
    if table[household_id_column].duplicated().any():
        duplicates = table.loc[
            table[household_id_column].duplicated(),
            household_id_column,
        ].unique()
        raise ValueError(
            f"metric table {group!r} household IDs must be unique; "
            f"duplicate value(s): {list(map(str, duplicates[:5]))}."
        )
    return table.set_index(household_id_column, drop=True)


def _simulation_household_ids(sim: Any, period: int | str) -> np.ndarray:
    return _values(sim.calculate("household_id", period=period, map_to="household"))


def _align_metric_table_to_households(
    table: pd.DataFrame,
    household_ids: Sequence[Any],
    *,
    group: str,
) -> pd.DataFrame:
    expected = pd.Index(household_ids)
    if expected.has_duplicates:
        duplicates = expected[expected.duplicated()].unique()
        raise ValueError(
            "household_ids must be unique before metric alignment; duplicate "
            f"value(s): {list(map(str, duplicates[:5]))}."
        )
    if table.index.has_duplicates:
        duplicates = table.index[table.index.duplicated()].unique()
        raise ValueError(
            f"metric table {group!r} household index must be unique; "
            f"duplicate value(s): {list(map(str, duplicates[:5]))}."
        )
    missing = expected.difference(table.index)
    if len(missing):
        raise ValueError(
            f"metric table {group!r} is missing household_id value(s): "
            f"{list(map(str, missing[:5]))}."
        )
    extra = table.index.difference(expected)
    if len(extra):
        raise ValueError(
            f"metric table {group!r} has unexpected household_id value(s): "
            f"{list(map(str, extra[:5]))}."
        )
    return table.reindex(expected)


def _infer_period(dataset: Any, period: int | str | None) -> int | str:
    if period is not None:
        return period
    for attr in ("time_period", "fiscal_year", "default_calculation_period"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return value
    raise ValueError("period is required when it cannot be inferred from the dataset.")


def _default_uk_simulation_factory(dataset: Any) -> Any:
    try:
        from policyengine_uk import Microsimulation
    except ImportError as exc:  # pragma: no cover - exercised only with engine absent
        raise ImportError(
            "Computing UK local metrics requires policyengine-uk. Install the "
            "UK engine or pass explicit metric tables."
        ) from exc
    return Microsimulation(dataset=dataset)


def _values(result: Any) -> np.ndarray:
    if hasattr(result, "values"):
        return np.asarray(result.values)
    return np.asarray(result)
