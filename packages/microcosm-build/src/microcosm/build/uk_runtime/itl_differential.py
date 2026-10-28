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
from dataclasses import dataclass, replace
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
    #: The cells whose model counts fix this one's (their ``sort_key``s): a
    #: sum's parts, or the two counts the one below State Pension age is the
    #: difference of. Such a cell's ``in_sample`` says every part is bound;
    #: ``differential`` keeps it only when the file holds nobody the cell
    #: counts outside its parts.
    parts: tuple[tuple[Any, ...], ...] = ()

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


def table_of(row: Mapping[str, Any]) -> str | None:
    """The Income Tax liabilities table a row comes from (``"2.5"``), or None."""

    record_set = str((row.get("layout") or {}).get("record_set_id") or "")
    match = _TABLE.search(f"{record_set}.")
    return None if match is None else f"{match.group(1)}.{match.group(2)}"


def _table(row: Mapping[str, Any]) -> str:
    table = table_of(row)
    if table is None:
        record_set = (row.get("layout") or {}).get("record_set_id")
        raise ValueError(f"no Income Tax liabilities table in {record_set!r}.")
    return table


_REGION_TIER_CODES = frozenset(code for _level, code in UK_REGION_TIER)


def _read_by_cells(row: Mapping[str, Any]) -> bool:
    """Whether a cell reads the row: Table 2.5's counts and Table 2.1's ages for
    the UK, Table 2.2's ages for the twelve region-tier areas."""

    table = table_of(row)
    measure = str(row.get("measure_id"))
    geography = str((row.get("geography") or {}).get("id"))
    if table == "2.5":
        return measure == _TABLE_2_5_COUNT and geography == UNITED_KINGDOM
    if table == "2.1":
        return measure in AGE_MEASURES and geography == UNITED_KINGDOM
    return (
        table == "2.2" and measure in AGE_MEASURES and geography in _REGION_TIER_CODES
    )


def itl_band_key(band: Band) -> str:
    """A band as the liabilities targets name it: ``band_12570_15000``, ``band_2000000_plus``."""

    lower, upper = band
    return f"band_{lower}_{'plus' if upper is None else upper}"


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

    Table 2.5 must carry every band of ``ITL_BANDS``, and no other, and its
    all-band count. Tables 2.1 and 2.2 enter only when their rows are
    present, and a Table 2.2 measure must then carry all twelve region-tier
    areas. Rows no cell reads (another measure, another geography) are left
    out before the window, so their gaps do not refuse the report.
    ``bound_targets`` names what the run bound at ``year``: a contract
    target, which binds every cell of its family, or one target by its name
    without the period (``hmrc.itl.taxpayers_by_total_income_band.band_12570_15000``,
    ``hmrc.itl.state_pension_age_taxpayers@E12000001``).

    A cell no target names is in sample through its ``parts`` only: the
    all-band count through its bands, the residents' count and Table 2.1's
    UK count through Table 2.2's areas of their measure, and the count below
    State Pension age through the all-band count and Table 2.1's State
    Pension age count, whose difference it is. Here such a cell is marked in
    sample when every part is; ``differential`` keeps the mark only when the
    file holds nobody the cell counts outside its parts (a taxpayer below the
    lowest band edge, or one in none of the twelve areas).
    """

    window = calendar_window(
        (row for row in rows if _read_by_cells(row)), year=year, weights=weights
    )
    unknown = sorted(
        str(key[3])
        for key in window
        if key[0] == "2.5" and key[3] is not None and key[3] not in ITL_BANDS
    )
    if unknown:
        raise ValueError(f"Table 2.5 bands {unknown} are not ITL_BANDS at {year}.")

    def bound(target, band, area):
        if target is None:
            return False
        if target in bound_targets:
            return True
        if band is not None:
            return f"{target}.{itl_band_key(band)}" in bound_targets
        return area is not None and f"{target}@{area}" in bound_targets

    def cell(table, measure, value_sources, *, band=None, area=None, regional=False):
        value, sources = value_sources
        return ITLCell(
            table=table,
            measure=measure,
            expected=value,
            tolerance=REGIONAL_TOLERANCE if regional else NATIONAL_TOLERANCE,
            in_sample=bound(BOUND_BY.get((table, measure)), band, area),
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
    all_cell = _summed(cell("2.5", ALL_TAXPAYERS, all_taxpayers), cells)
    cells.append(all_cell)
    for measure in AGE_MEASURES:
        national = window.get(("2.1", measure, UNITED_KINGDOM, None))
        areas = [
            window.get(("2.2", measure, code, None)) for _level, code in UK_REGION_TIER
        ]
        area_cells = []
        if any(area is not None for area in areas):
            missing = [
                code
                for (_level, code), area in zip(UK_REGION_TIER, areas, strict=True)
                if area is None
            ]
            if missing:
                raise ValueError(
                    f"Table 2.2 {measure} has no rows for {missing} at {year}."
                )
            area_cells = [
                cell("2.2", measure, area, area=code, regional=True)
                for (_level, code), area in zip(UK_REGION_TIER, areas, strict=True)
            ]
            residents = (
                math.fsum(area[0] for area in areas),
                tuple(source for area in areas for source in area[1]),
            )
            cells.extend(area_cells)
            cells.append(
                _summed(cell("2.2", measure, residents, area=RESIDENTS), area_cells)
            )
        if national is not None:
            # The UK count sums the areas' for everyone who lives in one of
            # the twelve; without Table 2.2 rows it has no parts.
            national_cell = _summed(cell("2.1", measure, national), area_cells)
            cells.append(national_cell)
            if measure == "state_pension_age":
                below = (
                    all_taxpayers[0] - national[0],
                    all_taxpayers[1] + national[1],
                )
                cells.append(
                    _summed(
                        cell("2.1", BELOW_STATE_PENSION_AGE, below),
                        (all_cell, national_cell),
                    )
                )
    return tuple(sorted(cells, key=lambda item: item.sort_key))


def _summed(cell: ITLCell, parts: Collection[ITLCell]) -> ITLCell:
    """A cell whose model count its ``parts`` fix, in sample when they all are.

    Its family's target binds its parts, not the sum: binding a band or an
    area does not bind the people outside them.
    """

    return replace(
        cell,
        in_sample=bool(parts) and all(part.in_sample for part in parts),
        parts=tuple(part.sort_key for part in parts),
    )


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
    """Each cell's model count, miss and verdict, and the misses by family.

    A cell with ``parts`` stays in sample only when every part does and the
    file's weight outside its parts (``outside_parts``, the weighted people
    it counts that no part does) is zero: otherwise the solve did not fix its
    model count. A family is in sample when every cell of it is.
    """

    cells = sorted(cells, key=lambda item: item.sort_key)
    by_key = {cell.sort_key: cell for cell in cells}
    weight = np.asarray(people.weight, dtype=float)
    in_sample: dict[tuple[Any, ...], bool] = {}
    outside: dict[tuple[Any, ...], float] = {}

    def resolved(cell: ITLCell) -> bool:
        if cell.sort_key not in in_sample:
            if not cell.parts:
                in_sample[cell.sort_key] = cell.in_sample
            else:
                # A part not among the cells covers nobody and is not bound.
                parts = [by_key[key] for key in cell.parts if key in by_key]
                covered = np.zeros(len(weight), dtype=bool)
                for part in parts:
                    covered |= cell_mask(part, people)
                outside[cell.sort_key] = math.fsum(
                    weight[cell_mask(cell, people) & ~covered].tolist()
                )
                in_sample[cell.sort_key] = (
                    cell.in_sample
                    and len(parts) == len(cell.parts)
                    and all(resolved(part) for part in parts)
                    and outside[cell.sort_key] == 0.0
                )
        return in_sample[cell.sort_key]

    rows = []
    for cell in cells:
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
                "in_sample": resolved(cell),
                "outside_parts": outside.get(cell.sort_key),
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
                "in_sample_cells": 0,
            },
        )
        family["cells"] += 1
        family["in_sample_cells"] += 1 if row["in_sample"] else 0
        family["misses"] += 0 if row["within_tolerance"] else 1
        family["largest_abs_residual"] = max(
            family["largest_abs_residual"], abs(row["residual"])
        )
    for family in families.values():
        family["in_sample"] = family["in_sample_cells"] == family["cells"]
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
