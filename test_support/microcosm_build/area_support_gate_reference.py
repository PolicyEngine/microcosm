"""The area support gate exactly as it stood before #1729.

Verbatim copies of ``area_support_shortfalls``, ``area_support_below_floor``
and ``area_support_gate`` of ``microcosm.build.gates`` at fccd9a68b
(origin/main when the gate gained optional floors and a floor relative to the
starting ESS), under their original names, so ``git diff`` against that
commit's module shows them unchanged. The differential tests hold the new
gate, called with the three absolute floors as the UK local build calls it,
equal to this copy. Do not edit it; it is the "before" of the change.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult


def area_support_shortfalls(
    rows: float,
    effective_sample_size: float,
    sources: float,
    *,
    minimum_rows: int,
    minimum_effective_sample_size: float,
    minimum_distinct_sources: int,
) -> list[str]:
    """The support floors one area misses, worded as :func:`area_support_gate` reports them.

    Each shortfall states the area's measured value beside the floor: the
    release ships per-area support figures since María's ruling of
    2026-10-26, which dropped their withholding (microcosm#1301).
    """

    shortfalls = []
    if rows < minimum_rows:
        shortfalls.append(f"rows {int(rows)} < {minimum_rows}")
    if effective_sample_size < minimum_effective_sample_size:
        shortfalls.append(
            "ESS "
            f"{float(effective_sample_size):.6g} < "
            f"{float(minimum_effective_sample_size):.6g}"
        )
    if sources < minimum_distinct_sources:
        shortfalls.append(
            f"distinct sources {int(sources)} < {minimum_distinct_sources}"
        )
    return shortfalls


def area_support_below_floor(
    area_support: pd.DataFrame,
    *,
    minimum_rows: int,
    minimum_effective_sample_size: float,
    minimum_distinct_sources: int,
) -> pd.Series:
    """Whether each area of a support table misses any floor, by the gate's rule.

    A reviewed exclusion does not lift the flag: an excluded area is still
    below the floor, which is what a consumer of the area's estimates needs.
    """

    return pd.Series(
        [
            bool(
                area_support_shortfalls(
                    row.nonzero_households,
                    row.effective_sample_size,
                    row.nonzero_source_households,
                    minimum_rows=minimum_rows,
                    minimum_effective_sample_size=minimum_effective_sample_size,
                    minimum_distinct_sources=minimum_distinct_sources,
                )
            )
            for row in area_support.itertuples(index=False)
        ],
        index=area_support.index,
        dtype=bool,
    )


def area_support_gate(
    area_support: pd.DataFrame,
    *,
    area_roster: Mapping[str, Iterable[str]],
    geography_levels: Iterable[str],
    minimum_rows: int,
    minimum_effective_sample_size: float,
    minimum_distinct_sources: int,
    reviewed_exclusions: Mapping[str, Mapping[str, object]] | None = None,
) -> GateResult:
    """Require adequate positive-weight support in every declared local area.

    ``area_support`` is the long-form candidate receipt.  The independently
    declared ``area_roster`` is load-bearing: a missing weak area cannot make
    the gate look better by disappearing from the evidence table.

    A failure and a reviewed exclusion state the area's measured support
    beside the floors it misses, and each geography level reports how many
    areas were checked and sit below a floor, and its minima (María's ruling
    of 2026-10-26 dropped the withholding of microcosm#1301).
    """

    levels = tuple(str(level) for level in geography_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("geography_levels must be non-empty and unique.")
    if (
        isinstance(minimum_rows, bool)
        or not isinstance(minimum_rows, int)
        or minimum_rows < 1
    ):
        raise ValueError("minimum_rows must be a positive integer.")
    if (
        isinstance(minimum_distinct_sources, bool)
        or not isinstance(minimum_distinct_sources, int)
        or minimum_distinct_sources < 1
    ):
        raise ValueError("minimum_distinct_sources must be a positive integer.")
    if (
        not math.isfinite(float(minimum_effective_sample_size))
        or minimum_effective_sample_size <= 0
    ):
        raise ValueError("minimum_effective_sample_size must be positive and finite.")
    if not isinstance(area_support, pd.DataFrame):
        raise TypeError("area_support must be a pandas DataFrame.")
    required_columns = {
        "geography_level",
        "area_code",
        "nonzero_households",
        "nonzero_source_households",
        "effective_sample_size",
    }
    missing_columns = sorted(required_columns - set(area_support.columns))
    if missing_columns:
        raise ValueError(
            f"area_support is missing required column(s): {missing_columns}."
        )
    unexpected_levels = sorted(
        set(area_support["geography_level"].astype(str)) - set(levels)
    )
    if unexpected_levels:
        raise ValueError(
            "area_support contains geography level(s) outside the declared "
            f"scope: {unexpected_levels}."
        )

    rows = area_support.copy()
    rows["geography_level"] = rows["geography_level"].astype(str)
    rows["area_code"] = rows["area_code"].astype(str)
    duplicate_mask = rows.duplicated(["geography_level", "area_code"], keep=False)
    if duplicate_mask.any():
        duplicates = sorted(
            {
                f"{row.geography_level}/{row.area_code}"
                for row in rows.loc[
                    duplicate_mask, ["geography_level", "area_code"]
                ].itertuples(index=False)
            }
        )
        raise ValueError(f"area_support contains duplicate area rows: {duplicates}.")

    expected_pairs: set[tuple[str, str]] = set()
    for level in levels:
        if level not in area_roster:
            raise ValueError(f"area_roster is missing geography level {level!r}.")
        codes = tuple(str(code) for code in area_roster[level])
        if not codes or len(set(codes)) != len(codes):
            raise ValueError(f"area_roster[{level!r}] must be non-empty and unique.")
        expected_pairs.update((level, code) for code in codes)
    actual_pairs = set(zip(rows["geography_level"], rows["area_code"], strict=True))
    missing_areas = sorted(expected_pairs - actual_pairs)
    extra_areas = sorted(actual_pairs - expected_pairs)
    if missing_areas or extra_areas:
        raise ValueError(
            "area_support must exactly cover the declared roster; "
            f"missing={missing_areas[:10]}, extra={extra_areas[:10]}."
        )

    numeric_columns = (
        "nonzero_households",
        "nonzero_source_households",
        "effective_sample_size",
    )
    values = rows.loc[:, list(numeric_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("area_support contains invalid support values.")

    exclusions = {} if reviewed_exclusions is None else dict(reviewed_exclusions)
    failures: list[str] = []
    used_exclusions: dict[str, dict[str, object]] = {}
    stale_exclusions: list[str] = []
    below_floor = dict.fromkeys(levels, 0)
    for row in rows.sort_values(["geography_level", "area_code"]).itertuples(
        index=False
    ):
        key = f"{row.geography_level}/{row.area_code}"
        shortfalls = area_support_shortfalls(
            row.nonzero_households,
            row.effective_sample_size,
            row.nonzero_source_households,
            minimum_rows=minimum_rows,
            minimum_effective_sample_size=minimum_effective_sample_size,
            minimum_distinct_sources=minimum_distinct_sources,
        )
        if shortfalls:
            below_floor[row.geography_level] += 1
        record = exclusions.get(key)
        if record is not None and shortfalls:
            used_exclusions[key] = {
                "shortfalls": shortfalls,
                "rows": int(row.nonzero_households),
                "ess": float(row.effective_sample_size),
                "sources": int(row.nonzero_source_households),
                "record": dict(record),
            }
        elif record is not None:
            stale_exclusions.append(key)
            failures.append(
                f"{key}: reviewed exclusion is stale; every support floor is met, "
                "so retire the exclusion."
            )
        elif shortfalls:
            failures.append(f"{key}: " + ", ".join(shortfalls))

    known_keys = {f"{level}/{code}" for level, code in expected_pairs}
    unknown_exclusions = sorted(set(exclusions) - known_keys)
    failures.extend(
        f"{key}: reviewed exclusion is unknown to the area roster."
        for key in unknown_exclusions
    )

    by_level: dict[str, dict[str, object]] = {}
    for level in levels:
        level_rows = rows.loc[rows["geography_level"] == level]
        by_level[level] = {
            "areas_checked": int(len(level_rows)),
            "areas_below_floor": below_floor[level],
            "minimum_rows": int(level_rows["nonzero_households"].min()),
            "minimum_effective_sample_size": float(
                level_rows["effective_sample_size"].min()
            ),
            "minimum_distinct_sources": int(
                level_rows["nonzero_source_households"].min()
            ),
        }
    return GateResult(
        name="area_support",
        passed=not failures,
        failures=tuple(failures),
        details={
            "floors": {
                "minimum_rows": minimum_rows,
                "minimum_effective_sample_size": float(minimum_effective_sample_size),
                "minimum_distinct_sources": minimum_distinct_sources,
            },
            "by_geography_level": by_level,
            "areas_checked": len(expected_pairs),
            "areas_failed": len(failures),
            "reviewed_exclusions": used_exclusions,
            "excluded_area_count": len(used_exclusions),
            "stale_exclusions": stale_exclusions,
            "unknown_exclusions": unknown_exclusions,
        },
    )
