"""Static aging: project a base-year frame to later years.

Two things change between a base year and a projection year, and this operator
keeps them apart:

* **Who exists** goes into the weights. For each projection year the weight
  entity's weights are recalibrated to that year's projected population by
  demographic cell (age and sex, and any other cell columns the projection
  declares). Nothing else is targeted: a program count or an income total is
  an output of rules, take-up and demographics, and a weight vector forced to
  match one can no longer be wrong about it, so it can no longer be checked.
* **How much** goes into per-column factors. A column that follows a national
  total gets the factor that makes its weighted total under the year's
  weights grow, from the base year, by exactly the total's projected growth,
  so the factor absorbs only what the reweighting did not explain. The
  series' level never enters: a column may follow a series for a broader
  concept than its own. A column that follows a per-person rate or a price
  index gets the index ratio. Applying the factors to the base-year values
  gives the year's values.

Program counts and totals are therefore predictions the projected frame makes,
and :func:`score_predictions` compares them with an external projection
without ever feeding it back.

The base-year cross-section is reweighted once per year with no person
identity across years. That is the cross-sectional special case of the
charter's longitudinal rule, not a replacement for the Dynamics operator: an
employment or marriage transition never happens here, and a projected
downturn shows up only as slower per-capita income growth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.calibrate.solve import FREE_MASS, CalibrationResult, calibrate
from microcosm.calibrate.target import Target, TargetSet
from microcosm.frame import Frame, MassChange, WeightKind, Weights

__all__ = [
    "DemographicProjection",
    "SeriesProjection",
    "StaticAgingResult",
    "YearProjection",
    "score_predictions",
    "ssa_population_projection",
    "static_aging",
]

COUNT = "count"
YEAR = "year"


@dataclass(frozen=True)
class DemographicProjection:
    """Projected person counts by year and demographic cell.

    Attributes:
        counts: One row per (year, cell) with the cell columns, a ``year``
            column and a ``count`` column. Every year must carry the same set
            of cells.
        cells: The cell column names, each of which must also be a person
            column on the frame (for example ``("age", "is_female")``).
        source: Free-text provenance.
    """

    counts: pd.DataFrame
    cells: tuple[str, ...]
    source: str = ""

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError(
                "DemographicProjection.cells must name at least one column."
            )
        required = {YEAR, COUNT, *self.cells}
        missing = sorted(required - set(self.counts.columns))
        if missing:
            raise ValueError(
                f"DemographicProjection.counts is missing columns {missing}."
            )
        counts = self.counts[[YEAR, *self.cells, COUNT]].copy()
        counts[YEAR] = counts[YEAR].astype(int)
        counts[COUNT] = counts[COUNT].astype(float)
        if (counts[COUNT] < 0).any() or not np.isfinite(counts[COUNT]).all():
            raise ValueError(
                "DemographicProjection counts must be finite and non-negative."
            )
        keys = counts[[YEAR, *self.cells]]
        if keys.duplicated().any():
            raise ValueError("DemographicProjection has duplicate (year, cell) rows.")
        cell_sets = counts.groupby(YEAR)[list(self.cells)].apply(
            lambda frame: frozenset(map(tuple, frame.to_numpy().tolist()))
        )
        if cell_sets.nunique() != 1:
            raise ValueError(
                "DemographicProjection must carry the same cells in every year."
            )
        object.__setattr__(self, "counts", counts.reset_index(drop=True))
        object.__setattr__(self, "cells", tuple(self.cells))

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted(self.counts[YEAR].unique().tolist()))

    def for_year(self, year: int) -> pd.DataFrame:
        """The cells and counts of one year, in a stable cell order."""
        if year not in self.years:
            raise ValueError(
                f"DemographicProjection has no year {year}; years are {self.years}."
            )
        rows = self.counts[self.counts[YEAR] == year]
        return rows.sort_values(list(self.cells)).reset_index(drop=True)

    def total(self, year: int) -> float:
        return float(self.for_year(year)[COUNT].sum())


@dataclass(frozen=True)
class SeriesProjection:
    """Projected national totals and per-person indices by series and year.

    Attributes:
        totals: National totals, ``series -> {year: value}``. A column mapped
            to one of these gets the factor that makes its weighted total
            under the year's weights grow from the base year by the series'
            growth; the series' level is never imposed.
        indices: Per-person rates and price indices, ``series -> {year:
            value}``. A column mapped to one of these gets the ratio of the
            year's value to the base year's.
        source: Free-text provenance.
    """

    totals: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    indices: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        overlap = sorted(set(self.totals) & set(self.indices))
        if overlap:
            raise ValueError(f"Series {overlap} appear as both totals and indices.")
        object.__setattr__(self, "totals", {k: dict(v) for k, v in self.totals.items()})
        object.__setattr__(
            self, "indices", {k: dict(v) for k, v in self.indices.items()}
        )

    def value(self, series: str, year: int) -> float:
        table = self.totals.get(series) or self.indices.get(series)
        if table is None:
            raise KeyError(f"SeriesProjection has no series {series!r}.")
        if year not in table:
            raise KeyError(f"Series {series!r} has no value for {year}.")
        value = float(table[year])
        if not np.isfinite(value):
            raise ValueError(f"Series {series!r} at {year} is not finite.")
        return value


@dataclass(frozen=True)
class YearProjection:
    """One projected year.

    Attributes:
        year: The projection year.
        weights: The weight entity's weights for the year.
        factors: ``column -> factor`` to multiply base-year values by. Columns
            without a mapped series are absent and carry over unchanged.
        demographic_fit: One row per cell with ``target`` and ``achieved``
            weighted person counts.
        calibration: The full calibration record for the year.
    """

    year: int
    weights: Weights
    factors: Mapping[str, float]
    demographic_fit: pd.DataFrame
    calibration: CalibrationResult

    @property
    def fraction_within_10pct(self) -> float:
        return self.calibration.fraction_within_10pct


@dataclass(frozen=True)
class StaticAgingResult:
    """The projected years of one static-aging run."""

    base_year: int
    weight_entity: str
    years: tuple[YearProjection, ...]

    def year(self, year: int) -> YearProjection:
        for projection in self.years:
            if projection.year == year:
                return projection
        raise KeyError(f"No projection for {year}; years are {self.year_list}.")

    @property
    def year_list(self) -> tuple[int, ...]:
        return tuple(projection.year for projection in self.years)

    def frame_for(self, frame: Frame, year: int) -> Frame:
        """``frame`` with the year's weights on the weight entity and the
        year's factors applied to its columns."""
        projection = self.year(year)
        aged = frame.with_weights(
            self.weight_entity,
            projection.weights,
            mass=MassChange(
                factor=None,
                reason=f"static aging: {year} population by demographic cell",
            ),
        )
        tables = {entity: aged.table(entity).copy() for entity in aged.entities}
        for column, factor in projection.factors.items():
            entity = aged.column_entity(column)
            tables[entity][column] = tables[entity][column].to_numpy() * factor
        weights = {
            entity: aged.weights_for(entity) for entity in aged.weighted_entities
        }
        return Frame(
            tables,
            aged.schema,
            weights,
            aged.strata,
            mass_log=aged.mass_log,
            metadata={**aged.metadata, "static_aging_year": year},
        )


def _group_positions(frame: Frame, group: str) -> np.ndarray:
    """Position of each person's ``group`` row within the group table."""
    schema = frame.schema
    membership = frame.person[schema.membership_column(group)].to_numpy()
    ids = frame.table(group)[schema.id_column(group)].to_numpy()
    order = np.argsort(ids, kind="stable")
    positions = order[np.searchsorted(ids[order], membership)]
    if not np.array_equal(ids[positions], membership):
        raise ValueError(f"Every person must belong to a {group} row.")
    return positions


def _cell_masks(
    frame: Frame, projection: DemographicProjection, cells: pd.DataFrame
) -> list[np.ndarray]:
    person = frame.person
    for column in projection.cells:
        if column not in person.columns:
            raise ValueError(
                f"Demographic cell column {column!r} is not a person column."
            )
    values = [person[column].to_numpy() for column in projection.cells]
    masks = []
    for row in cells[list(projection.cells)].itertuples(index=False):
        mask = np.ones(len(person), dtype=bool)
        for column_values, wanted in zip(values, row, strict=True):
            mask &= column_values == wanted
        masks.append(mask)
    return masks


def _cell_label(projection: DemographicProjection, row: tuple) -> str:
    return ",".join(f"{c}={v}" for c, v in zip(projection.cells, row, strict=True))


def static_aging(
    frame: Frame,
    *,
    base_year: int,
    years: Sequence[int],
    demographics: DemographicProjection,
    series: SeriesProjection | None = None,
    column_series: Mapping[str, str] | None = None,
    weight_entity: str = "household",
    anchor: str = "frame",
    epochs: int = 256,
    learning_rate: float = 0.02,
    max_weight_ratio: float | None = 5.0,
    l2_lambda: float = 0.0,
    seed: int = 0,
) -> StaticAgingResult:
    """Project ``frame`` from ``base_year`` to each of ``years``.

    Args:
        frame: The base-year frame. Its weight-entity weights are the starting
            point of every year's calibration.
        base_year: The year the frame's values and weights describe.
        years: Projection years; each must be in ``demographics`` and, for
            every mapped series, in ``series``.
        demographics: Projected person counts by cell and year.
        series: Projected national totals and indices.
        column_series: ``column -> series`` for the columns that follow a
            series. Every series must be in ``series``. Unmapped columns carry
            over unchanged.
        weight_entity: The entity whose weights carry the demographics.
        anchor: ``"frame"`` (default) targets each cell at the frame's own
            base-year weighted count times the projection's growth for that
            cell, so the base-year calibration is kept and only projected
            change is applied. ``"projection"`` targets the projection's
            absolute counts.
        epochs, learning_rate, max_weight_ratio, l2_lambda, seed: Passed to
            :func:`~microcosm.calibrate.calibrate`. ``max_weight_ratio`` bounds
            how far any record's weight may move from its base-year value.

    Returns:
        A :class:`StaticAgingResult` with one :class:`YearProjection` per year.
    """
    if anchor not in ("frame", "projection"):
        raise ValueError(f"anchor must be 'frame' or 'projection', got {anchor!r}.")
    years = tuple(int(year) for year in years)
    if not years:
        raise ValueError("years must name at least one projection year.")
    if any(year <= base_year for year in years):
        raise ValueError(f"Projection years must follow base year {base_year}.")
    column_series = dict(column_series or {})
    series = series if series is not None else SeriesProjection()
    for column, name in column_series.items():
        if name not in series.totals and name not in series.indices:
            raise ValueError(
                f"Column {column!r} maps to series {name!r}, which the "
                "SeriesProjection does not carry."
            )
        frame.column_entity(column)  # raises if the column is absent
    if anchor == "frame" and base_year not in demographics.years:
        raise ValueError(
            f"anchor='frame' needs the base year {base_year} in the projection."
        )

    base_weights = frame.resolve_weights("person").values
    base_cells = demographics.for_year(demographics.years[0])
    masks = _cell_masks(frame, demographics, base_cells)
    positions = _group_positions(frame, weight_entity)
    n_groups = frame.n(weight_entity)
    cell_counts = [
        np.bincount(positions, weights=mask.astype(float), minlength=n_groups)
        for mask in masks
    ]
    base_counts = np.array([float((mask * base_weights).sum()) for mask in masks])
    labels = [
        _cell_label(demographics, row)
        for row in base_cells[list(demographics.cells)].itertuples(index=False)
    ]
    if anchor == "frame":
        base_projection = demographics.for_year(base_year)[COUNT].to_numpy()

    projections = []
    for year in years:
        cells = demographics.for_year(year)
        if not cells[list(demographics.cells)].equals(
            base_cells[list(demographics.cells)]
        ):
            raise ValueError("Demographic cells differ between years.")
        projected = cells[COUNT].to_numpy()
        if anchor == "frame":
            growth = np.divide(
                projected,
                base_projection,
                out=np.ones_like(projected),
                where=base_projection > 0,
            )
            target_counts = base_counts * growth
        else:
            target_counts = projected
        targets = []
        for label, counts, value in zip(
            labels, cell_counts, target_counts, strict=True
        ):
            if counts.sum() == 0:
                continue  # no record supports this cell; it cannot be targeted
            targets.append(
                Target(
                    name=f"population[{label}]",
                    entity=weight_entity,
                    measure=_constant_measure(counts),
                    value=float(value),
                    period=year,
                    source=demographics.source,
                )
            )
        result = calibrate(
            frame,
            TargetSet(targets),
            weight_entity=weight_entity,
            epochs=epochs,
            learning_rate=learning_rate,
            mass=FREE_MASS,
            mass_reason=f"static aging: {year} population by {', '.join(demographics.cells)}",
            max_weight_ratio=max_weight_ratio,
            l2_lambda=l2_lambda,
            seed=seed,
        )
        year_person_weights = result.frame.resolve_weights("person").values
        achieved = np.array(
            [float((mask * year_person_weights).sum()) for mask in masks]
        )
        fit = base_cells[list(demographics.cells)].copy()
        fit["base"] = base_counts
        fit["target"] = target_counts
        fit["achieved"] = achieved
        factors = _factors(result.frame, frame, base_year, year, series, column_series)
        projections.append(
            YearProjection(
                year=year,
                weights=Weights(values=result.weights, kind=WeightKind.CALIBRATED),
                factors=factors,
                demographic_fit=fit,
                calibration=result,
            )
        )
    return StaticAgingResult(
        base_year=base_year, weight_entity=weight_entity, years=tuple(projections)
    )


def _constant_measure(values: np.ndarray) -> Callable[[Frame], np.ndarray]:
    def measure(frame: Frame) -> np.ndarray:  # noqa: ARG001 - Target protocol
        return values

    return measure


def _factors(
    aged: Frame,
    base: Frame,
    base_year: int,
    year: int,
    series: SeriesProjection,
    column_series: Mapping[str, str],
) -> dict[str, float]:
    factors: dict[str, float] = {}
    for column, name in column_series.items():
        entity = base.column_entity(column)
        values = base.table(entity)[column].to_numpy(dtype=float)
        if name in series.totals:
            base_total = float((base.resolve_weights(entity).values * values).sum())
            aged_total = float((aged.resolve_weights(entity).values * values).sum())
            if base_total == 0.0 or aged_total == 0.0:
                factors[column] = 1.0
                continue
            growth = series.value(name, year) / series.value(name, base_year)
            factors[column] = growth * base_total / aged_total
        else:
            factors[column] = series.value(name, year) / series.value(name, base_year)
    return factors


def score_predictions(
    frame: Frame,
    result: StaticAgingResult,
    measures: Mapping[str, Callable[[Frame], float]],
    projections: Mapping[str, Mapping[int, float]],
) -> pd.DataFrame:
    """Compare what the projected frames imply with external projections.

    Nothing here feeds back into weights or factors: this is the scorecard for
    the counts and totals static aging predicts rather than targets.

    Args:
        frame: The base-year frame.
        result: The static-aging result.
        measures: ``name -> measure(frame) -> float`` computed on each
            projected frame (for example a weighted count of recipients).
        projections: ``name -> {year: external value}``.

    Returns:
        One row per (name, year) with ``predicted``, ``external`` and
        ``ratio`` columns, for the years both sides carry.
    """
    rows = []
    for name, measure in measures.items():
        external = projections.get(name, {})
        for projection in result.years:
            if projection.year not in external:
                continue
            aged = result.frame_for(frame, projection.year)
            predicted = float(measure(aged))
            value = float(external[projection.year])
            rows.append(
                {
                    "name": name,
                    YEAR: projection.year,
                    "predicted": predicted,
                    "external": value,
                    "ratio": predicted / value if value else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=["name", YEAR, "predicted", "external", "ratio"])


def ssa_population_projection(
    path: str | Path,
    *,
    age_top: int = 85,
    sex_column: str = "is_female",
    age_column: str = "age",
) -> DemographicProjection:
    """Read SSA's single-year-of-age population projection into cells.

    The file is the Trustees Report population table (``SSPopJul_TR<year>.csv``)
    with ``Year``, ``Age``, ``M Tot`` and ``F Tot`` columns. Ages at or above
    ``age_top`` are pooled into one cell, matching survey top-coding.
    """
    table = pd.read_csv(path)
    required = {"Year", "Age", "M Tot", "F Tot"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"SSA population file is missing columns {missing}.")
    age = table["Age"].astype(int).clip(upper=age_top)
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    YEAR: table["Year"].astype(int),
                    age_column: age,
                    sex_column: female,
                    COUNT: table[column].astype(float),
                }
            )
            for column, female in (("M Tot", False), ("F Tot", True))
        ]
    )
    counts = long.groupby([YEAR, age_column, sex_column], as_index=False)[COUNT].sum()
    return DemographicProjection(
        counts=counts,
        cells=(age_column, sex_column),
        source=f"SSA Trustees Report population projection ({Path(path).name})",
    )
