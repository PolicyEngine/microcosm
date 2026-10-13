"""Reference statement of the proposed PUF donor rebase.

Not production code. It states the semantics of the methodology proposal so
that the invariants can be tested before anything is built. Implementation
tests belong in the package suites.

The model
---------
Each dollar column of the processed donor is a function of TY2015 PUF fields.
A *leg* is one field (optionally restricted to one sign) together with

- ``carried``: the factor the donor value carries over the TY2015 field,
  verified record by record against the source file (for most columns the
  archived 2015-to-2021 factor; 1.0 where the archived generator applied
  none; a product where it applied two);
- ``series``: the growth series the field should follow from TY2015 to the
  build year.

Per-return growth of a series from year ``a`` to year ``b`` is
``(L[series][b] / L[series][a]) / (N[b] / N[a])`` for an aggregate level table
``L`` and a return-count table ``N``. A level table is spliced from observed
actuals up to a pivot year and a projection index from the pivot on, the way
``us_runtime/target_aging.py`` chains SOI actuals to a CBO projection.

The rebase factor of a leg is ``growth(series, 2015, Y) / carried``. Applying
it turns a verified donor value into ``field × growth(series, 2015, Y)``: the
TY2015 amount aged once, by its own series, to the build year ``Y``. That is
what "start from the donor's verified price year" means when columns carry
different price years.

Three kinds of column:

- **Single-expression columns** (one leg, or a positive and a negative leg of
  one field): verified by the gate on every regular record, zeros included,
  then rebased leg by leg.
- **Dividends on regular records.** The donor stores qualified and
  non-qualified dividends, the second as ordinary less qualified after each was
  aged by its own factor. The TY2015 ordinary and qualified amounts are
  recovered exactly from the two stored columns, then qualified follows the
  qualified line and non-qualified follows the ordinary-less-qualified line.
  Non-qualified is zero exactly where the source's ordinary equals its
  qualified.
- **Dividends on synthetic records** (the aggregate-record replacements, which
  have no source record and are not whole TY2015 dollars): each stored column
  is scaled by its own column's factor, so signs are kept, including the
  negative non-qualified amounts some of them already hold.
- **Derived columns** (simulated from, or a structural transform of, a base
  amount): each record is scaled by the ratio of its rebased to its carried
  base, which keeps the record's ratio to its base. A record whose carried
  base is zero is left unchanged. They are not verified by the gate.

A donor stamped as rebased cannot be rebased again, and a donor whose values
do not verify against the source refuses to rebase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

ARCHIVED_LINEAGE = "policyengine-us-data 42ed5d45 uprate_puf 2015-to-2021"
SOURCE_YEAR = 2015
FLOAT64_RTOL = 1e-9
FLOAT32_RTOL = 1e-6


class RebaseError(ValueError):
    """The donor's price lineage is not the one the rebase expects."""


@dataclass(frozen=True)
class Leg:
    field: str
    series: str
    carried: float
    sign: int = 0  # 0: all values; +1: positive values only; -1: negative only


@dataclass(frozen=True)
class Column:
    name: str
    legs: tuple[Leg, ...]
    rtol: float = FLOAT64_RTOL


@dataclass
class Donor:
    """Dollar columns, untouched columns and the stamped price lineage."""

    columns: dict[str, np.ndarray]
    untouched: dict[str, np.ndarray] = field(default_factory=dict)
    lineage: str = ARCHIVED_LINEAGE
    price_year: int | None = None


@dataclass(frozen=True)
class LevelTable:
    """Aggregate levels by series and year, and return counts by year."""

    levels: Mapping[str, Mapping[int, float]]
    returns: Mapping[int, float]

    def growth(self, series: str, start: int, end: int) -> float:
        level = self.levels[series]
        return (level[end] / level[start]) / (self.returns[end] / self.returns[start])


def splice(
    actuals: Mapping[int, float], projection: Mapping[int, float]
) -> dict[int, float]:
    """Observed levels up to the pivot, then the projection's growth from it.

    The pivot is the earliest projection year, as in
    ``target_aging._projection_factor``: observed growth for observed years,
    projected growth only where nothing is observed. The pivot year's observed
    level is kept, so the pivot is counted once.
    """

    pivot = min(projection)
    if pivot not in actuals:
        raise RebaseError(f"no observed level for the pivot year {pivot}")
    spliced = {year: level for year, level in actuals.items() if year <= pivot}
    for year, index in projection.items():
        if year > pivot:
            spliced[year] = actuals[pivot] * index / projection[pivot]
    return spliced


def leg_mask(values: np.ndarray, leg: Leg) -> np.ndarray:
    if leg.sign > 0:
        return values > 0
    if leg.sign < 0:
        return values < 0
    return np.ones(values.shape, dtype=bool)


def verify_carried(
    donor_values: np.ndarray,
    source_values: np.ndarray,
    carried: float,
    *,
    rtol: float = FLOAT64_RTOL,
) -> bool:
    """Every record equals its TY2015 value times the carried factor."""

    expected = source_values * carried
    return bool(
        np.all(
            np.abs(donor_values - expected) <= rtol * np.maximum(1.0, np.abs(expected))
        )
    )


def expected_carried(source_values: np.ndarray, column: Column) -> np.ndarray:
    """The whole column the donor should hold: each leg's records times its
    carried factor, and zero wherever the source is zero."""

    expected = np.zeros_like(source_values, dtype=np.float64)
    for leg in column.legs:
        mask = leg_mask(source_values, leg)
        expected[mask] = source_values[mask] * leg.carried
    return expected


def rebase_column(
    values: np.ndarray, column: Column, table: LevelTable, build_year: int
) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for leg in column.legs:
        factor = table.growth(leg.series, SOURCE_YEAR, build_year) / leg.carried
        mask = leg_mask(values, leg)
        out[mask] = values[mask] * factor
    return out


def recover_source_dividends(
    qualified: np.ndarray,
    non_qualified: np.ndarray,
    *,
    carried_ordinary: float,
    carried_qualified: float,
) -> tuple[np.ndarray, np.ndarray]:
    """TY2015 ordinary and qualified dividends from the two stored columns.

    The donor stores ``qualified = Q × a`` and ``non_qualified = O × b − Q × a``,
    so ``Q = qualified / a`` and ``O = (non_qualified + qualified) / b``. PUF
    amounts are whole dollars; the recovered values are snapped to that grid
    and refused if they are not on it.
    """

    recovered = (
        (non_qualified + qualified) / carried_ordinary,
        qualified / carried_qualified,
    )
    # Rounding enters through the stored products, at their own magnitude, and
    # is divided back by the carried factor.
    stored_magnitudes = (
        np.maximum.reduce(
            [
                np.abs(non_qualified),
                np.abs(qualified),
                np.abs(non_qualified + qualified),
            ]
        )
        / carried_ordinary,
        np.abs(qualified) / carried_qualified,
    )
    snapped = tuple(np.rint(values) for values in recovered)
    for values, grid, stored in zip(recovered, snapped, stored_magnitudes, strict=True):
        if not np.all(on_dollar_grid(values, grid, stored)):
            raise RebaseError("recovered dividends are not whole TY2015 dollars")
        if np.any(grid < 0):
            raise RebaseError("recovered dividends are negative")
    if np.any(snapped[1] > snapped[0]):
        raise RebaseError("recovered qualified dividends exceed ordinary dividends")
    return snapped


GRID_ULPS = 8
GRID_LIMIT = 2.0**52


def on_dollar_grid(
    values: np.ndarray, grid: np.ndarray, stored_magnitude: np.ndarray
) -> np.ndarray:
    """Within the rounding a recovery by division can add, of a whole dollar.

    The allowance is GRID_ULPS units in the last place, of the whole-dollar
    value or of the stored magnitude it was divided out of, whichever is
    larger. When the two carried factors are close, as on the donor
    (1.3914 and 1.3628), that is about 8 units of the value; the donor's
    regular records deviate by at most 2 (the audit receipt's
    ``dividends.recovery_max_ulps``). A quarter dollar at 1e13 is refused.
    Values at or above 2**52, a chosen domain limit far above any PUF amount
    (the donor's largest is $125.5M), are refused; near that limit the
    allowance itself approaches half a dollar.
    """

    magnitude = np.maximum(np.abs(grid), 1.0)
    allowance = GRID_ULPS * np.maximum(
        np.spacing(magnitude), np.spacing(np.maximum(stored_magnitude, 1.0))
    )
    return (magnitude < GRID_LIMIT) & (np.abs(values - grid) <= allowance)


def rebase_dividends(
    qualified: np.ndarray,
    non_qualified: np.ndarray,
    *,
    carried_ordinary: float,
    carried_qualified: float,
    table: LevelTable,
    build_year: int,
    qualified_series: str = "qualified_dividends",
    non_qualified_series: str = "non_qualified_dividends",
) -> tuple[np.ndarray, np.ndarray]:
    """Qualified by its own line; non-qualified by ordinary less qualified."""

    ordinary_2015, qualified_2015 = recover_source_dividends(
        qualified,
        non_qualified,
        carried_ordinary=carried_ordinary,
        carried_qualified=carried_qualified,
    )
    qualified_out = qualified_2015 * table.growth(
        qualified_series, SOURCE_YEAR, build_year
    )
    non_qualified_out = (ordinary_2015 - qualified_2015) * table.growth(
        non_qualified_series, SOURCE_YEAR, build_year
    )
    return qualified_out, non_qualified_out


def rebase_synthetic_dividends(
    qualified: np.ndarray,
    non_qualified: np.ndarray,
    *,
    carried_ordinary: float,
    carried_qualified: float,
    table: LevelTable,
    build_year: int,
    qualified_series: str = "qualified_dividends",
    non_qualified_series: str = "non_qualified_dividends",
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic records: each stored column scaled by its column's factor.

    There is no source record to recover, and the stored pair need not be whole
    dollars or have qualified below ordinary. Non-qualified takes the carried
    ordinary factor as its carried factor.
    """

    qualified_out = qualified * (
        table.growth(qualified_series, SOURCE_YEAR, build_year) / carried_qualified
    )
    non_qualified_out = non_qualified * (
        table.growth(non_qualified_series, SOURCE_YEAR, build_year) / carried_ordinary
    )
    return qualified_out, non_qualified_out


def scale_derived(
    values: np.ndarray, base_carried: np.ndarray, base_rebased: np.ndarray
) -> np.ndarray:
    """Scale a derived column by its base amount's rebase, record by record."""

    ratio = np.divide(
        base_rebased,
        base_carried,
        out=np.ones_like(base_carried, dtype=np.float64),
        where=base_carried != 0,
    )
    return values * ratio


def rebase_donor(
    donor: Donor,
    columns: tuple[Column, ...],
    table: LevelTable,
    build_year: int,
    *,
    source: Mapping[str, np.ndarray],
) -> Donor:
    """The proposed stage: verify the carried factors, then rebase once.

    ``source`` is required: there is no path that rebases an unverified donor.
    """

    if donor.lineage != ARCHIVED_LINEAGE or donor.price_year is not None:
        raise RebaseError(
            f"donor lineage {donor.lineage!r} (price year {donor.price_year}) "
            "is not the archived lineage this rebase starts from"
        )
    for column in columns:
        fields = {leg.field for leg in column.legs}
        if len(fields) != 1:
            raise RebaseError(f"{column.name}: a column's legs must share one field")
        field = fields.pop()
        expected = expected_carried(source[field], column)
        values = donor.columns[column.name]
        if not verify_carried(values, expected, 1.0, rtol=column.rtol):
            raise RebaseError(
                f"{column.name}: donor values do not equal {field} times its "
                "carried factors on every record"
            )
    rebased = {name: values.copy() for name, values in donor.columns.items()}
    for column in columns:
        rebased[column.name] = rebase_column(
            donor.columns[column.name], column, table, build_year
        )
    return Donor(
        columns=rebased,
        untouched={name: values.copy() for name, values in donor.untouched.items()},
        lineage=f"rebased from TY{SOURCE_YEAR} by series",
        price_year=build_year,
    )


def naive_declared_uprate(
    donor_values: np.ndarray, series: str, table: LevelTable, build_year: int
) -> np.ndarray:
    """The ``uprate`` sources.yaml declares (2015 to the build year), applied
    to the processed donor as it stands. Stated to record why it is wrong."""

    return donor_values * table.growth(series, SOURCE_YEAR, build_year)
