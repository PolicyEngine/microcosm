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
``L`` and a return-count table ``N``. Deriving every factor from one level
table makes chained factors consistent by construction.

The rebase factor of a leg is ``growth(series, 2015, Y) / carried``. Applying
it turns a verified donor value into ``field × growth(series, 2015, Y)``: the
TY2015 amount aged once, by its own series, to the build year ``Y``. That is
what "start from the donor's verified price year" means when columns carry
different price years.

A donor stamped as rebased cannot be rebased again, and a donor whose values
do not verify against the source refuses to rebase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

ARCHIVED_LINEAGE = "policyengine-us-data 42ed5d45 uprate_puf 2015-to-2021"
SOURCE_YEAR = 2015


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
    rtol: float = 1e-9,
) -> bool:
    """Every record equals its TY2015 value times the carried factor."""

    expected = source_values * carried
    return bool(
        np.all(
            np.abs(donor_values - expected) <= rtol * np.maximum(1.0, np.abs(expected))
        )
    )


def rebase_column(
    values: np.ndarray, column: Column, table: LevelTable, build_year: int
) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for leg in column.legs:
        factor = table.growth(leg.series, SOURCE_YEAR, build_year) / leg.carried
        mask = leg_mask(values, leg)
        out[mask] = values[mask] * factor
    return out


def rebase_dividends(
    qualified: np.ndarray,
    non_qualified: np.ndarray,
    *,
    ordinary_leg: Leg,
    qualified_leg: Leg,
    table: LevelTable,
    build_year: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rebase ordinary and qualified dividends, then derive non-qualified.

    The donor stores qualified and non-qualified dividends; ordinary dividends
    are their sum. Each of ordinary and qualified follows its own series, and
    non-qualified is the difference, clipped at zero. Returns the clipped mass.
    """

    ordinary = qualified + non_qualified
    ordinary_out = ordinary * (
        table.growth(ordinary_leg.series, SOURCE_YEAR, build_year)
        / ordinary_leg.carried
    )
    qualified_out = qualified * (
        table.growth(qualified_leg.series, SOURCE_YEAR, build_year)
        / qualified_leg.carried
    )
    difference = ordinary_out - qualified_out
    non_qualified_out = np.maximum(difference, 0.0)
    clipped = float(-difference[difference < 0].sum())
    return qualified_out, non_qualified_out, clipped


def rebase_donor(
    donor: Donor,
    columns: tuple[Column, ...],
    table: LevelTable,
    build_year: int,
    *,
    source: Mapping[str, np.ndarray] | None = None,
) -> Donor:
    """The proposed stage: verify the carried factors, then rebase once."""

    if donor.lineage != ARCHIVED_LINEAGE or donor.price_year is not None:
        raise RebaseError(
            f"donor lineage {donor.lineage!r} (price year {donor.price_year}) "
            "is not the archived lineage this rebase starts from"
        )
    if source is not None:
        for column in columns:
            values = donor.columns[column.name]
            for leg in column.legs:
                mask = leg_mask(source[leg.field], leg)
                if not verify_carried(
                    values[mask], source[leg.field][mask], leg.carried
                ):
                    raise RebaseError(
                        f"{column.name}: donor values do not equal "
                        f"{leg.field} x {leg.carried}"
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
