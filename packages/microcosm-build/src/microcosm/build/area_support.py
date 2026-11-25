"""Support of each local area: its table, its floors and one gate (microcosm#1729).

A local-area file is reliable in an area only when enough records carry that
area's weight. This module measures that for any country and any pooled
build:

- :func:`effective_sample_size_by_group` is the Kish effective sample size
  (ESS) of the records in each area.
- :func:`area_support_table` builds the long-form support table the gate
  reads, from a weight vector and each record's area: positive-weight rows,
  distinct positive-weight source households, ESS, and (with the starting
  weights) the ESS the area started with.
- :func:`area_support_gate` checks every declared area against the floors a
  country declares. Each floor is optional:

  - ``minimum_rows``: positive-weight records,
  - ``minimum_distinct_sources``: distinct positive-weight source households,
  - ``minimum_effective_sample_size``: an absolute ESS floor,
  - ``minimum_effective_sample_size_ratio``: an ESS floor relative to the
    area's starting ESS (the rule microcosm#1078 adopted for US districts:
    an area collapses when calibration leaves it below a share of the ESS its
    starting weights had).

  The gate reports; whether a failure blocks a release is the caller's
  choice, declared with the country's gates.

- :func:`area_support_floor_table` gives the same verdict area by area, for
  a report that lists the areas.

With the three absolute floors and no relative floor, :func:`area_support_gate`
is :func:`microcosm.build.gates.area_support_gate`, which it calls, so the UK
local gate and this one cannot disagree. Every other declaration runs the
general rule here, which the tests hold equal to that function wherever both
apply.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult, area_support_shortfalls
from microcosm.build.gates import area_support_gate as _absolute_floor_gate

__all__ = [
    "AREA_SUPPORT_COLUMNS",
    "area_support_floor_table",
    "area_support_gate",
    "area_support_table",
    "effective_sample_size_by_group",
]

ROWS = "nonzero_households"
SOURCES = "nonzero_source_households"
ESS = "effective_sample_size"
STARTING_ESS = "starting_effective_sample_size"
ESS_RATIO = "effective_sample_size_ratio"
#: The support columns of the long-form table, beside ``geography_level`` and
#: ``area_code``. A gate needs only the columns of the floors it declares.
AREA_SUPPORT_COLUMNS = (ROWS, SOURCES, ESS, STARTING_ESS)


# ---------------------------------------------------------------------------
# Measuring support
# ---------------------------------------------------------------------------


def _checked_weights(weights: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError(f"{label} must be finite and non-negative.")
    return weights


def effective_sample_size_by_group(
    weights: Sequence[float] | np.ndarray, codes: Sequence | np.ndarray
) -> dict[str, float]:
    """Kish ESS, ``(sum w)^2 / sum(w^2)``, of the records in each group.

    Groups are keyed by ``str(code)``, in sorted order. A group whose weights
    are all zero has ESS ``0.0``. Refuses negative, NaN or infinite weights,
    as :func:`microcosm.calibrate.solve.effective_sample_size` does.
    """

    weights = _checked_weights(weights, "weights")
    codes = np.asarray(codes)
    if codes.shape != weights.shape:
        raise ValueError(
            f"{codes.shape} group codes do not align with {weights.shape} weights."
        )
    frame = pd.DataFrame({"code": codes.astype(str), "w": weights})
    frame["w2"] = frame["w"] ** 2
    sums = frame.groupby("code", sort=True)[["w", "w2"]].sum()
    return {
        str(code): float(row.w**2 / row.w2) if row.w2 > 0 else 0.0
        for code, row in sums.iterrows()
    }


def area_support_table(
    weights: Sequence[float] | np.ndarray,
    area_codes: Sequence | np.ndarray,
    *,
    geography_level: str,
    source_ids: Sequence | np.ndarray | None = None,
    starting_weights: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """The support of each area under ``weights``, one row per area.

    Columns: ``geography_level``, ``area_code``, ``nonzero_households`` (the
    records with a positive weight), ``effective_sample_size`` (Kish ESS over
    records), and

    - with ``source_ids``: ``nonzero_source_households``, the distinct source
      households among the positive-weight records. A source id is whatever
      identifies one source household across its copies and clones (for a
      pooled build, a pair such as spine and source id joined into one key);
    - with ``starting_weights``: ``starting_effective_sample_size``, the ESS
      of the same records under the weights the solve started from.

    Areas come in sorted order of ``str(area_code)``. An area appears when at
    least one record names it, whatever its weight.
    """

    weights = _checked_weights(weights, "weights")
    codes = np.asarray(area_codes)
    if codes.shape != weights.shape:
        raise ValueError(
            f"{codes.shape} area codes do not align with {weights.shape} weights."
        )
    frame = pd.DataFrame({"area_code": codes.astype(str), "weight": weights})
    positive = frame["weight"] > 0.0
    table = pd.DataFrame(
        {
            ROWS: positive.groupby(frame["area_code"], sort=True).sum().astype(int),
        }
    )
    if source_ids is not None:
        sources = np.asarray(source_ids)
        if sources.shape != weights.shape:
            raise ValueError(
                f"{sources.shape} source ids do not align with {weights.shape} weights."
            )
        distinct = (
            pd.DataFrame({"area_code": frame["area_code"], "source": sources})[
                positive.to_numpy()
            ]
            .groupby("area_code", sort=True)["source"]
            .nunique()
        )
        table[SOURCES] = distinct.reindex(table.index, fill_value=0).astype(int)
    ess = effective_sample_size_by_group(weights, codes)
    table[ESS] = [ess[code] for code in table.index]
    if starting_weights is not None:
        starting = _checked_weights(starting_weights, "starting_weights")
        if starting.shape != weights.shape:
            raise ValueError(
                f"{starting.shape} starting weights do not align with "
                f"{weights.shape} weights."
            )
        starting_ess = effective_sample_size_by_group(starting, codes)
        table[STARTING_ESS] = [starting_ess[code] for code in table.index]
    table = table.reset_index().rename(columns={"index": "area_code"})
    table.insert(0, "geography_level", str(geography_level))
    return table


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------


def _checked_floors(
    minimum_rows: int | None,
    minimum_effective_sample_size: float | None,
    minimum_distinct_sources: int | None,
    minimum_effective_sample_size_ratio: float | None,
) -> None:
    for label, floor in (
        ("minimum_rows", minimum_rows),
        ("minimum_distinct_sources", minimum_distinct_sources),
    ):
        if floor is not None and (
            isinstance(floor, bool) or not isinstance(floor, int) or floor < 1
        ):
            raise ValueError(f"{label} must be a positive integer.")
    if minimum_effective_sample_size is not None and (
        isinstance(minimum_effective_sample_size, bool)
        or not math.isfinite(float(minimum_effective_sample_size))
        or minimum_effective_sample_size <= 0
    ):
        raise ValueError("minimum_effective_sample_size must be positive and finite.")
    if minimum_effective_sample_size_ratio is not None and (
        isinstance(minimum_effective_sample_size_ratio, bool)
        or not math.isfinite(float(minimum_effective_sample_size_ratio))
        or not 0.0 < minimum_effective_sample_size_ratio <= 1.0
    ):
        raise ValueError("minimum_effective_sample_size_ratio must be in (0, 1].")


def _required_columns(
    minimum_rows: int | None,
    minimum_effective_sample_size: float | None,
    minimum_distinct_sources: int | None,
    minimum_effective_sample_size_ratio: float | None,
) -> list[str]:
    columns = ["geography_level", "area_code"]
    if minimum_rows is not None:
        columns.append(ROWS)
    if minimum_distinct_sources is not None:
        columns.append(SOURCES)
    if (
        minimum_effective_sample_size is not None
        or minimum_effective_sample_size_ratio is not None
    ):
        columns.append(ESS)
    if minimum_effective_sample_size_ratio is not None:
        columns.append(STARTING_ESS)
    return columns


def _checked_rows(area_support: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    if not isinstance(area_support, pd.DataFrame):
        raise TypeError("area_support must be a pandas DataFrame.")
    missing_columns = sorted(set(required) - set(area_support.columns))
    if missing_columns:
        raise ValueError(
            f"area_support is missing required column(s): {missing_columns}."
        )
    rows = area_support.copy()
    rows["geography_level"] = rows["geography_level"].astype(str)
    rows["area_code"] = rows["area_code"].astype(str)
    present = [column for column in AREA_SUPPORT_COLUMNS if column in rows.columns]
    values = rows.loc[:, present].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("area_support contains invalid support values.")
    return rows


def area_support_floor_table(
    area_support: pd.DataFrame,
    *,
    minimum_rows: int | None = None,
    minimum_effective_sample_size: float | None = None,
    minimum_distinct_sources: int | None = None,
    minimum_effective_sample_size_ratio: float | None = None,
) -> pd.DataFrame:
    """Each area against each declared floor, one row per area, in input order.

    Beside the input's columns: ``below_<floor>`` (bool) for each declared
    floor, ``below_any_floor``, and, when the table carries the starting ESS,
    ``effective_sample_size_ratio`` (ESS over starting ESS; NaN where the
    starting ESS is zero).

    An area is below the relative floor when its ESS is less than
    ``minimum_effective_sample_size_ratio`` times its starting ESS. The test
    multiplies, so an area whose starting ESS is zero is never below it.
    """

    _checked_floors(
        minimum_rows,
        minimum_effective_sample_size,
        minimum_distinct_sources,
        minimum_effective_sample_size_ratio,
    )
    rows = _checked_rows(
        area_support,
        _required_columns(
            minimum_rows,
            minimum_effective_sample_size,
            minimum_distinct_sources,
            minimum_effective_sample_size_ratio,
        ),
    )
    flags: list[str] = []
    if STARTING_ESS in rows.columns and ESS in rows.columns:
        ess = rows[ESS].to_numpy(dtype=np.float64)
        starting = rows[STARTING_ESS].to_numpy(dtype=np.float64)
        ratio = np.full(len(rows), np.nan, dtype=np.float64)
        # A ratio too large for a float is infinite, as plain division gives.
        with np.errstate(over="ignore"):
            np.divide(ess, starting, out=ratio, where=starting > 0)
        rows[ESS_RATIO] = ratio
    if minimum_rows is not None:
        rows["below_minimum_rows"] = rows[ROWS] < minimum_rows
        flags.append("below_minimum_rows")
    if minimum_effective_sample_size is not None:
        rows["below_minimum_effective_sample_size"] = (
            rows[ESS] < minimum_effective_sample_size
        )
        flags.append("below_minimum_effective_sample_size")
    if minimum_distinct_sources is not None:
        rows["below_minimum_distinct_sources"] = (
            rows[SOURCES] < minimum_distinct_sources
        )
        flags.append("below_minimum_distinct_sources")
    if minimum_effective_sample_size_ratio is not None:
        rows["below_minimum_effective_sample_size_ratio"] = rows[ESS] < (
            minimum_effective_sample_size_ratio * rows[STARTING_ESS]
        )
        flags.append("below_minimum_effective_sample_size_ratio")
    rows["below_any_floor"] = (
        rows[flags].any(axis=1) if flags else np.zeros(len(rows), dtype=bool)
    )
    return rows


def _shortfalls(row, floors: Mapping[str, object]) -> list[str]:
    """One area's missed floors, worded as the absolute-floor gate words them."""

    # The absolute floors, in the gate's order and wording. A floor that is
    # not declared is given a value no area can miss.
    shortfalls = area_support_shortfalls(
        getattr(row, ROWS, 0) if floors["minimum_rows"] is not None else 0,
        (
            getattr(row, ESS)
            if floors["minimum_effective_sample_size"] is not None
            else math.inf
        ),
        (
            getattr(row, SOURCES, 0)
            if floors["minimum_distinct_sources"] is not None
            else 0
        ),
        minimum_rows=floors["minimum_rows"] or 0,
        minimum_effective_sample_size=floors["minimum_effective_sample_size"] or 0.0,
        minimum_distinct_sources=floors["minimum_distinct_sources"] or 0,
    )
    ratio = floors["minimum_effective_sample_size_ratio"]
    if ratio is not None:
        ess, starting = float(getattr(row, ESS)), float(getattr(row, STARTING_ESS))
        if ess < ratio * starting:
            shortfalls.append(
                f"ESS {ess:.6g} < {float(ratio):.6g} x starting ESS {starting:.6g}"
            )
    return shortfalls


def _general_area_support_gate(
    area_support: pd.DataFrame,
    *,
    area_roster: Mapping[str, Iterable[str]],
    geography_levels: Iterable[str],
    minimum_rows: int | None,
    minimum_effective_sample_size: float | None,
    minimum_distinct_sources: int | None,
    minimum_effective_sample_size_ratio: float | None,
    reviewed_exclusions: Mapping[str, Mapping[str, object]] | None,
) -> GateResult:
    """The gate under any declaration of floors (see :func:`area_support_gate`)."""

    levels = tuple(str(level) for level in geography_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("geography_levels must be non-empty and unique.")
    _checked_floors(
        minimum_rows,
        minimum_effective_sample_size,
        minimum_distinct_sources,
        minimum_effective_sample_size_ratio,
    )
    floors = {
        "minimum_rows": minimum_rows,
        "minimum_effective_sample_size": (
            None
            if minimum_effective_sample_size is None
            else float(minimum_effective_sample_size)
        ),
        "minimum_distinct_sources": minimum_distinct_sources,
        "minimum_effective_sample_size_ratio": (
            None
            if minimum_effective_sample_size_ratio is None
            else float(minimum_effective_sample_size_ratio)
        ),
    }
    if not isinstance(area_support, pd.DataFrame):
        raise TypeError("area_support must be a pandas DataFrame.")
    required = _required_columns(
        minimum_rows,
        minimum_effective_sample_size,
        minimum_distinct_sources,
        minimum_effective_sample_size_ratio,
    )
    missing_columns = sorted(set(required) - set(area_support.columns))
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

    support_columns = [column for column in required if column in AREA_SUPPORT_COLUMNS]
    values = rows.loc[:, support_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("area_support contains invalid support values.")

    exclusions = {} if reviewed_exclusions is None else dict(reviewed_exclusions)
    failures: list[str] = []
    used_exclusions: dict[str, dict[str, object]] = {}
    stale_exclusions: list[str] = []
    below_floor = dict.fromkeys(levels, 0)
    below_ratio = dict.fromkeys(levels, 0)
    ratio_floor = floors["minimum_effective_sample_size_ratio"]
    for row in rows.sort_values(["geography_level", "area_code"]).itertuples(
        index=False
    ):
        key = f"{row.geography_level}/{row.area_code}"
        shortfalls = _shortfalls(row, floors)
        if shortfalls:
            below_floor[row.geography_level] += 1
        if ratio_floor is not None and float(getattr(row, ESS)) < ratio_floor * float(
            getattr(row, STARTING_ESS)
        ):
            below_ratio[row.geography_level] += 1
        record = exclusions.get(key)
        if record is not None and shortfalls:
            used: dict[str, object] = {"shortfalls": shortfalls}
            if ROWS in rows.columns:
                used["rows"] = int(getattr(row, ROWS))
            if ESS in rows.columns:
                used["ess"] = float(getattr(row, ESS))
            if SOURCES in rows.columns:
                used["sources"] = int(getattr(row, SOURCES))
            used["record"] = dict(record)
            used_exclusions[key] = used
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
        summary: dict[str, object] = {
            "areas_checked": int(len(level_rows)),
            "areas_below_floor": below_floor[level],
        }
        if ROWS in level_rows.columns:
            summary["minimum_rows"] = int(level_rows[ROWS].min())
        if ESS in level_rows.columns:
            summary["minimum_effective_sample_size"] = float(level_rows[ESS].min())
        if SOURCES in level_rows.columns:
            summary["minimum_distinct_sources"] = int(level_rows[SOURCES].min())
        if ratio_floor is not None:
            starting = level_rows[STARTING_ESS].to_numpy(dtype=np.float64)
            ess = level_rows[ESS].to_numpy(dtype=np.float64)
            rated = starting > 0
            summary["areas_below_effective_sample_size_ratio"] = below_ratio[level]
            with np.errstate(over="ignore"):
                ratios = ess[rated] / starting[rated]
            summary["minimum_effective_sample_size_ratio"] = (
                float(ratios.min()) if rated.any() else None
            )
        by_level[level] = summary
    declared = {
        key: value
        for key, value in floors.items()
        if key != "minimum_effective_sample_size_ratio" or value is not None
    }
    return GateResult(
        name="area_support",
        passed=not failures,
        failures=tuple(failures),
        details={
            "floors": declared,
            "by_geography_level": by_level,
            "areas_checked": len(expected_pairs),
            "areas_failed": len(failures),
            "reviewed_exclusions": used_exclusions,
            "excluded_area_count": len(used_exclusions),
            "stale_exclusions": stale_exclusions,
            "unknown_exclusions": unknown_exclusions,
        },
    )


def area_support_gate(
    area_support: pd.DataFrame,
    *,
    area_roster: Mapping[str, Iterable[str]],
    geography_levels: Iterable[str],
    minimum_rows: int | None = None,
    minimum_effective_sample_size: float | None = None,
    minimum_distinct_sources: int | None = None,
    minimum_effective_sample_size_ratio: float | None = None,
    reviewed_exclusions: Mapping[str, Mapping[str, object]] | None = None,
) -> GateResult:
    """Check every declared area against the floors a country declares.

    ``area_support`` is the long-form table (:func:`area_support_table`); it
    needs only the columns of the declared floors. The independently declared
    ``area_roster`` is load-bearing: a weak area cannot make the gate look
    better by being absent from the table.

    Each floor is optional (``None`` leaves it unchecked):

    - ``minimum_rows``, ``minimum_effective_sample_size`` and
      ``minimum_distinct_sources`` are the absolute floors of
      :func:`microcosm.build.gates.area_support_gate`. With all three and no
      relative floor this function returns that function's result.
    - ``minimum_effective_sample_size_ratio`` (in ``(0, 1]``) fails an area
      whose ESS is below that share of its ``starting_effective_sample_size``.

    The result reports; it blocks nothing by itself. A caller that publishes
    a report-only entry decides what a failure means.
    """

    if minimum_effective_sample_size_ratio is None and None not in (
        minimum_rows,
        minimum_effective_sample_size,
        minimum_distinct_sources,
    ):
        return _absolute_floor_gate(
            area_support,
            area_roster=area_roster,
            geography_levels=geography_levels,
            minimum_rows=minimum_rows,
            minimum_effective_sample_size=minimum_effective_sample_size,
            minimum_distinct_sources=minimum_distinct_sources,
            reviewed_exclusions=reviewed_exclusions,
        )
    return _general_area_support_gate(
        area_support,
        area_roster=area_roster,
        geography_levels=geography_levels,
        minimum_rows=minimum_rows,
        minimum_effective_sample_size=minimum_effective_sample_size,
        minimum_distinct_sources=minimum_distinct_sources,
        minimum_effective_sample_size_ratio=minimum_effective_sample_size_ratio,
        reviewed_exclusions=reviewed_exclusions,
    )
