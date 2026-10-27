"""HMRC's Income Tax liabilities projections as a check on a calibrated file.

HMRC's Income Tax liabilities statistics (July 2026) project the 2023-24
Survey of Personal Incomes to 2026-27. This module reads three of their
tables at the calibration year's calendar window (3/12 of the tax year
opening the year before, 9/12 of the one opening in it, as
``CALENDAR_YEAR_WINDOW_WEIGHTS`` defines for every bound row) and sets them
beside the same counts on a calibrated file:

- Table 2.5: United Kingdom Income Tax payers in each total-income band, and
  in all bands;
- Table 2.1: United Kingdom Income Tax payers of State Pension age and aged
  65 and over, and below State Pension age as Table 2.5's all-band count
  less Table 2.1's State Pension age count;
- Table 2.2: the same two age groups in each of the twelve region-tier
  areas, and their sum over the areas, the residents within Table 2.1's
  United Kingdom count.

The same Table 2.5 cells at the 2023-24 outturn column (``SINGLE_TAX_YEAR``)
check a file's dated SPI measures (``dated_people``): the file's own
2023-24 taxpayers by band, from the engine's simulation of that year on the
carried-back inputs, against the year HMRC observed rather than projected.

Each cell carries a tolerance fixed before any file is read
(``NATIONAL_TOLERANCE``, ``REGIONAL_TOLERANCE``) and says whether a bound
target reads it (``in_sample``): a bound cell checks the solve, an unbound
one checks the file against a source it was not fitted to. A model count is
the exactly rounded sum of its people's weights (``math.fsum``), and cells
sort by table, measure and key, so a report depends on the rows and the
file, not on their order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.ledger_targets import CALENDAR_YEAR_WINDOW_WEIGHTS
from microcosm.build.uk_runtime.hmrc_uprating import ITL_BANDS
from microcosm.calibrate.geography_constants import UK_REGION_TIER

TAX = "income_tax_before_winter_fuel_payment_charge_annual"
TOTAL_INCOME = "total_income_after_loss_relief_annual"

#: Largest relative miss a cell may show: national cells are sums over
#: millions of weighted taxpayers, a region's over a twelfth of them.
NATIONAL_TOLERANCE = 0.05
REGIONAL_TOLERANCE = 0.10

UNITED_KINGDOM = "K02000001"
#: The ``key`` of the sum over the twelve region-tier areas.
RESIDENTS = "residents"
AGE_MEASURES = ("state_pension_age", "age_65_and_over")
BAND_COUNT = "taxpayers_by_total_income_band"
ALL_TAXPAYERS = "all_taxpayers"
BELOW_STATE_PENSION_AGE = "below_state_pension_age"
_TABLE_2_5_COUNT = "total_taxpayer_count"
_TABLE = re.compile(r"\.table_(\d+)_(\d+)\.")

#: The target that binds each cell family, when the contract carries it.
BOUND_BY: Mapping[tuple[str, str], str] = {
    ("2.5", BAND_COUNT): "hmrc.itl.taxpayers_by_total_income_band",
    ("2.2", "state_pension_age"): "hmrc.itl.state_pension_age_taxpayers",
    ("2.2", "age_65_and_over"): "hmrc.itl.taxpayers_aged_65_and_over",
}

Band = tuple[int, int | None]

#: Window weights that read one tax year: the 2023-24 outturn column the dated
#: ``spi_2023_*`` measures sit beside.
SINGLE_TAX_YEAR: Mapping[int, float] = {0: 1.0}


@dataclass(frozen=True)
class ITLCell:
    """One published count at the calendar window, with what selects it."""

    table: str
    measure: str
    expected: float
    tolerance: float
    in_sample: bool
    sources: tuple[str, ...]
    band: Band | None = None
    area: str | None = None

    @property
    def key(self) -> str:
        if self.band is not None:
            lower, upper = self.band
            return f"{lower}+" if upper is None else f"{lower}-{upper}"
        return self.area or UNITED_KINGDOM

    @property
    def sort_key(self) -> tuple[Any, ...]:
        band = self.band or (-1, None)
        return (self.table, self.measure, band[0], self.key)


@dataclass(frozen=True)
class ITLPeople:
    """Per-person arrays a cell reads, all of one length."""

    weight: np.ndarray
    taxpayer: np.ndarray
    total_income: np.ndarray
    state_pension_age: np.ndarray
    age: np.ndarray
    #: Region-tier GSS code of each person's household, "" outside the tier.
    area: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            np.asarray(getattr(self, name)).shape
            for name in (
                "weight",
                "taxpayer",
                "total_income",
                "state_pension_age",
                "age",
                "area",
            )
        }
        if len(lengths) != 1 or len(next(iter(lengths))) != 1:
            raise ValueError("ITLPeople arrays must be one-dimensional and align.")
        weight = np.asarray(self.weight, dtype=float)
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("ITLPeople weights must be finite and non-negative.")


def _table(row: Mapping[str, Any]) -> str:
    record_set = str((row.get("layout") or {}).get("record_set_id") or "")
    match = _TABLE.search(f"{record_set}.")
    if match is None:
        raise ValueError(f"no Income Tax liabilities table in {record_set!r}.")
    return f"{match.group(1)}.{match.group(2)}"


def _band(row: Mapping[str, Any]) -> Band | None:
    dimensions = row.get("dimensions") or {}
    lower = dimensions.get("total_income_lower_bound")
    if lower is None:
        return None
    upper = dimensions.get("total_income_upper_bound")
    return int(lower), None if upper is None else int(upper)


def calendar_window(
    rows: Iterable[Mapping[str, Any]],
    *,
    year: int,
    weights: Mapping[int, float] = CALENDAR_YEAR_WINDOW_WEIGHTS,
) -> dict[tuple[str, str, str, Band | None], tuple[float, tuple[str, ...]]]:
    """Every published count at the calendar window of ``year``.

    Keyed by ``(table, measure_id, geography id, band)``: the window weights
    times the tax years opening in ``year - 1`` and ``year``. A count with
    two rows for one year, or a row in only one of the window's years, is
    refused; one with neither year is left out. ``weights`` of ``{0: 1.0}``
    read the tax year opening in ``year`` alone (``SINGLE_TAX_YEAR``).
    """

    years = {year + offset: weight for offset, weight in weights.items()}
    by_cell: dict[tuple[str, str, str, Band | None], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if (row.get("period") or {}).get("type") != "tax_year":
            continue
        opening = int(row["period"]["value"])
        if opening not in years:
            continue
        cell = (
            _table(row),
            str(row["measure_id"]),
            str(row["geography"]["id"]),
            _band(row),
        )
        found = by_cell.setdefault(cell, {})
        if opening in found:
            raise ValueError(f"two rows for {cell} in the tax year opening {opening}.")
        found[opening] = row
    window = {}
    for cell, found in by_cell.items():
        if set(found) != set(years):
            raise ValueError(
                f"{cell} has rows for tax years opening {sorted(found)}; the "
                f"window at {year} needs {sorted(years)}."
            )
        value = math.fsum(
            years[opening] * float(found[opening]["value"]) for opening in sorted(years)
        )
        sources = tuple(
            str(found[opening].get("source_record_id") or "")
            for opening in sorted(years)
        )
        window[cell] = (value, sources)
    return window


def expected_cells(
    rows: Iterable[Mapping[str, Any]],
    *,
    year: int,
    bound_targets: Collection[str] = (),
    weights: Mapping[int, float] = CALENDAR_YEAR_WINDOW_WEIGHTS,
) -> tuple[ITLCell, ...]:
    """The cells HMRC's rows give at ``year``'s window, in report order.

    Table 2.5 must carry every band of ``ITL_BANDS`` and its all-band count.
    Tables 2.1 and 2.2 enter only when their rows are present, and a Table
    2.2 measure must then carry all twelve region-tier areas.
    """

    window = calendar_window(rows, year=year, weights=weights)

    def cell(table, measure, value_sources, *, band=None, area=None, regional=False):
        value, sources = value_sources
        target = BOUND_BY.get((table, measure))
        return ITLCell(
            table=table,
            measure=measure,
            expected=value,
            tolerance=REGIONAL_TOLERANCE if regional else NATIONAL_TOLERANCE,
            in_sample=target is not None and target in bound_targets,
            sources=sources,
            band=band,
            area=area,
        )

    def need(key):
        if key not in window:
            raise ValueError(f"no Income Tax liabilities rows for {key} at {year}.")
        return window[key]

    cells = [
        cell(
            "2.5",
            BAND_COUNT,
            need(("2.5", _TABLE_2_5_COUNT, UNITED_KINGDOM, band)),
            band=band,
        )
        for band in ITL_BANDS
    ]
    all_taxpayers = need(("2.5", _TABLE_2_5_COUNT, UNITED_KINGDOM, None))
    cells.append(cell("2.5", ALL_TAXPAYERS, all_taxpayers))
    for measure in AGE_MEASURES:
        national = window.get(("2.1", measure, UNITED_KINGDOM, None))
        if national is not None:
            cells.append(cell("2.1", measure, national))
        if measure == "state_pension_age" and national is not None:
            below = (
                all_taxpayers[0] - national[0],
                all_taxpayers[1] + national[1],
            )
            cells.append(cell("2.1", BELOW_STATE_PENSION_AGE, below))
        areas = [
            window.get(("2.2", measure, code, None)) for _level, code in UK_REGION_TIER
        ]
        if all(area is None for area in areas):
            continue
        missing = [
            code
            for (_level, code), area in zip(UK_REGION_TIER, areas, strict=True)
            if area is None
        ]
        if missing:
            raise ValueError(
                f"Table 2.2 {measure} has no rows for {missing} at {year}."
            )
        for (_level, code), area in zip(UK_REGION_TIER, areas, strict=True):
            cells.append(cell("2.2", measure, area, area=code, regional=True))
        residents = (
            math.fsum(area[0] for area in areas),
            tuple(source for area in areas for source in area[1]),
        )
        cells.append(cell("2.2", measure, residents, area=RESIDENTS))
    return tuple(sorted(cells, key=lambda item: item.sort_key))


def cell_mask(cell: ITLCell, people: ITLPeople) -> np.ndarray:
    """The people a cell counts: taxpayers, then its band, age and area."""

    mask = np.asarray(people.taxpayer, dtype=bool).copy()
    if cell.band is not None:
        lower, upper = cell.band
        income = np.asarray(people.total_income, dtype=float)
        mask &= income >= lower
        if upper is not None:
            mask &= income < upper
    if cell.measure == "state_pension_age":
        mask &= np.asarray(people.state_pension_age, dtype=bool)
    elif cell.measure == BELOW_STATE_PENSION_AGE:
        mask &= ~np.asarray(people.state_pension_age, dtype=bool)
    elif cell.measure == "age_65_and_over":
        mask &= np.asarray(people.age, dtype=float) >= 65
    elif cell.measure not in (BAND_COUNT, ALL_TAXPAYERS):
        raise ValueError(f"no model reading for the measure {cell.measure!r}.")
    area = np.asarray(people.area).astype(str)
    if cell.area == RESIDENTS:
        mask &= np.isin(area, [code for _level, code in UK_REGION_TIER])
    elif cell.area is not None:
        mask &= area == cell.area
    return mask


def model_count(cell: ITLCell, people: ITLPeople) -> float:
    """The weighted people a cell counts, summed exactly rounded."""

    weight = np.asarray(people.weight, dtype=float)
    return math.fsum(weight[cell_mask(cell, people)].tolist())


def residual(model: float, expected: float) -> float:
    """The model's relative miss, ``model / expected - 1``."""

    if not expected > 0:
        raise ValueError(f"an expected count must be positive; found {expected!r}.")
    return model / expected - 1.0


def differential(cells: Iterable[ITLCell], people: ITLPeople) -> dict[str, Any]:
    """Each cell's model count, miss and verdict, and the misses by family."""

    rows = []
    for cell in sorted(cells, key=lambda item: item.sort_key):
        model = model_count(cell, people)
        miss = residual(model, cell.expected)
        rows.append(
            {
                "table": cell.table,
                "measure": cell.measure,
                "key": cell.key,
                "expected": cell.expected,
                "model": model,
                "residual": miss,
                "tolerance": cell.tolerance,
                "within_tolerance": abs(miss) <= cell.tolerance,
                "in_sample": cell.in_sample,
                "sources": list(cell.sources),
            }
        )
    families: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = families.setdefault(
            f"{row['table']}:{row['measure']}",
            {
                "cells": 0,
                "misses": 0,
                "largest_abs_residual": 0.0,
                "in_sample": row["in_sample"],
            },
        )
        family["cells"] += 1
        family["misses"] += 0 if row["within_tolerance"] else 1
        family["largest_abs_residual"] = max(
            family["largest_abs_residual"], abs(row["residual"])
        )
    return {
        "tolerances": {"national": NATIONAL_TOLERANCE, "regional": REGIONAL_TOLERANCE},
        "cells": rows,
        "families": dict(sorted(families.items())),
        "misses": [
            f"{row['table']} {row['measure']} {row['key']}: {row['residual']:+.2%}"
            for row in rows
            if not row["within_tolerance"]
        ],
    }


def measure_resolver(
    h5: Path, *, year: int, scratch: Path, reason: str | None = None
) -> Any:
    """A measure resolver on a calibrated file at ``year``, read once for both
    checks. ``reason`` reads current income for a file without annual income
    descriptors (the incumbent)."""

    from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver

    resolver = UKMeasureResolver(
        simulation_source=h5, scratch_dir=scratch, year=year, frame=None
    )
    if reason:
        resolver.use_current_income_for_annual_measures(reason=reason)
    return resolver


def _person(resolver: Any, variable: str) -> np.ndarray:
    values, _route = resolver.compute("person", variable)
    return np.asarray(values)


def resolved_people(resolver: Any) -> ITLPeople:
    """The people a calibrated file holds, as the bound rows read them.

    Weights are each person's household weight, as ``person_count`` mapped to
    the household reads them; taxpayers have
    ``income_tax_before_winter_fuel_payment_charge_annual > 0`` and bands read
    ``total_income_after_loss_relief_annual``, the measures the Table 2.5 rows
    bind; State Pension age is policyengine-uk's ``is_SP_age``.
    """

    from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM

    by_name = {name: code for code, name in UK_REGION_TIER_ENUM.items()}
    region = [str(name).rsplit(".", 1)[-1] for name in _person(resolver, "region")]
    return ITLPeople(
        weight=_person(resolver, "household_weight").astype(float),
        taxpayer=_person(resolver, TAX).astype(float) > 0,
        total_income=_person(resolver, TOTAL_INCOME).astype(float),
        state_pension_age=_person(resolver, "is_SP_age").astype(bool),
        age=_person(resolver, "age").astype(float),
        area=np.array([by_name.get(name, "") for name in region]),
    )


def dated_people(resolver: Any, dated_year: int) -> ITLPeople:
    """The people a calibrated file holds, as they were in ``dated_year``.

    Reads the dated SPI measures (``uk_runtime.spi_dated_measures``): each
    person's household weight times their population factor, a taxpayer at
    ``spi_<year>_income_tax > 0`` and the band at ``spi_<year>_total_income``,
    both from the engine's own simulation of that tax year on the carried-back
    inputs. Ages and areas are not dated, so only Table 2.5 cells read them.
    """

    def dated(measure: str) -> np.ndarray:
        return _person(resolver, f"spi_{dated_year}_{measure}").astype(float)

    count = dated("person_count")
    weight = _person(resolver, "household_weight").astype(float)
    return ITLPeople(
        weight=weight * count,
        taxpayer=dated("income_tax") > 0,
        total_income=dated("total_income"),
        state_pension_age=np.zeros(count.size, dtype=bool),
        age=np.full(count.size, np.nan),
        area=np.full(count.size, ""),
    )
