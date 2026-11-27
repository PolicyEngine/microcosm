"""Support of each local area: its table and its floors (microcosm#1729).

A local-area file is reliable in an area only when enough records carry that
area's weight. This module measures that for any country and any pooled
build:

- :func:`effective_sample_size_by_group` is the Kish effective sample size
  (ESS) of the records in each area.
- :func:`area_support_table` builds the long-form support table the gate
  reads, from a weight vector and each record's area: positive-weight rows,
  distinct positive-weight source households, ESS, and (with the starting
  weights) the ESS the area started with.
- :func:`area_support_floor_table` checks each area against each floor a
  country declares, one flag column for each floor, for a report that lists
  the areas.

The gate itself is :func:`microcosm.build.gates.area_support_gate`, which
this module re-exports. Each of its floors is optional: ``minimum_rows``,
``minimum_distinct_sources``, ``minimum_effective_sample_size``, and
``minimum_effective_sample_size_ratio``, an ESS floor relative to the area's
starting ESS (the rule microcosm#1078 adopted for US districts). The gate
reports; whether a failure blocks a release is the caller's choice, declared
with the country's gates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.build.gates import (
    AREA_SUPPORT_COLUMNS,
    area_support_gate,
    area_support_required_columns,
    checked_area_support_floors,
)
from microcosm.build.gates import AREA_SUPPORT_ESS as ESS
from microcosm.build.gates import AREA_SUPPORT_ROWS as ROWS
from microcosm.build.gates import AREA_SUPPORT_SOURCES as SOURCES
from microcosm.build.gates import AREA_SUPPORT_STARTING_ESS as STARTING_ESS

__all__ = [
    "AREA_SUPPORT_COLUMNS",
    "area_support_floor_table",
    "area_support_gate",
    "area_support_table",
    "effective_sample_size_by_group",
]

ESS_RATIO = "effective_sample_size_ratio"


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

    checked_area_support_floors(
        minimum_rows,
        minimum_effective_sample_size,
        minimum_distinct_sources,
        minimum_effective_sample_size_ratio,
    )
    rows = _checked_rows(
        area_support,
        area_support_required_columns(
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
