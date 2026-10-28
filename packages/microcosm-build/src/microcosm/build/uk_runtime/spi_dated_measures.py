"""HMRC SPI measures dated by their tax year, as the CGT measures are.

The SPI band tables (Table 3.6: State Pension and private pension, count and
amount by total income band) describe a published tax year, 2023-24, while
the national weights are solved at calendar 2025. Projecting each banded fact
to 2025 by HMRC's all-taxpayer growth per nominal band understates pensioner
taxpayers (microcosm#1095 pensioner review, option 1 of BAND-PROJECTION.md):
the bands move under people whose incomes grow at different rates.

A dated SPI measure instead computes each person's year at the fact's own
tax year, so the published value binds as published, with no uprating:

* ``spi_<year>_total_income`` and ``spi_<year>_income_tax`` classify: the
  person's total income after loss relief and income tax before the Winter
  Fuel Payment charge over tax year ``<year>``, from the annual income
  simulation run at ``<year>``.
* ``spi_<year>_person_count``, ``spi_<year>_state_pension`` and
  ``spi_<year>_private_pension_income`` add up: the person's count, State
  Pension and private pension in ``<year>``, each times the person's
  population factor (below). A factor is positive, so a filter ``> 0`` on an
  additive measure selects the same people as on the unweighted amount.

The year before the dataset's is not one the engine projects to: it holds a
single-year dataset's inputs from that year onwards only, and reads every
input at an earlier year as its default. The resolver sets each stored input
at the earlier year as the engine's own loader would have to carry it
forward: savings interest gross of ISA interest, as the engine projects it; a
column the engine uprates by an index (``uprating_indices.yaml``) divided by
one plus the index's growth for each year in between
(:func:`back_projection_factors`, :func:`back_projected_inputs`);
residential stocks by the cumulative house price ratio the engine applies
instead; and every other column held, since the engine holds it too, as it
holds Bereavement Support Payment despite its index once the file stores the
separated bereavement benefits. The
engine's own ``extend_single_year_dataset`` must return the carried-back file
to the stored one, every input column of it (:func:`round_trip_mismatches`),
except the few non-income inputs it moves by regional rules, which are held
by declaration. The State Pension's formulas split the reported amount at the
dataset year's rates (``min(dataset.years)``), so at an earlier year they pay
that share of the earlier year's rates, and the additional pension moves by
September CPI: the uprating runs backwards.

Population. The weights are solved at 2025 against ONS's mid-2025 population
by sex and age band, and a 2023-24 fact counts 2023-24's population, about
3.5% fewer people over State Pension age. Each person's additive measures are
scaled by the ratio of the population of their sex and age band at the fact
year's midpoint (1 October, three months after mid-year, interpolated
linearly between the mid-year estimates) to that at the calibration year's
midpoint (mid-year). The bands are the ONS sex-by-age bands the calibration
binds (``ons.population.<sex>_<band>``), so where the weights meet those
targets, a band's weighted factor sums to ONS's population at the fact year.
Ages are held flat across years, as the engine holds them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

__all__ = [
    "UK_SPI_ADDITIVE_MEASURES",
    "UK_SPI_CLASSIFIER_MEASURES",
    "UK_SPI_DATED_MEASURE",
    "UK_SPI_POPULATION_AGE_BANDS",
    "UK_SPI_POPULATION_RESOURCES",
    "back_projected_inputs",
    "back_projection_factors",
    "population_factor_for_people",
    "population_factors",
    "round_trip_mismatches",
    "uk_spi_dated_measure_year",
    "uk_spi_dated_measures_named",
]

#: An SPI measure dated by its tax year, such as ``spi_2023_state_pension``.
UK_SPI_DATED_MEASURE = re.compile(
    r"^spi_(20[0-9]{2})_"
    r"(person_count|state_pension|private_pension_income|total_income|income_tax)$"
)
#: Measures summed over people: each carries the person's population factor.
#: The value is the engine variable on the dated annual simulation, or None
#: for the count, which is the factor alone.
UK_SPI_ADDITIVE_MEASURES: Mapping[str, str | None] = {
    "person_count": None,
    "state_pension": "state_pension",
    "private_pension_income": "private_pension_income",
}
#: Measures that classify a person (a band, a taxpayer filter): the engine
#: variable on the dated annual simulation, without a population factor.
UK_SPI_CLASSIFIER_MEASURES: Mapping[str, str] = {
    "total_income": "total_income_after_loss_relief",
    "income_tax": "income_tax_before_winter_fuel_payment_charge",
}

#: The ONS sex-by-age bands the national calibration binds
#: (``ons.population.<sex>_<lower>_<upper>``), inclusive; 90 is 90 and over.
UK_SPI_POPULATION_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 9),
    (10, 14),
    (15, 15),
    (16, 17),
    (18, 19),
    *((lower, lower + 4) for lower in range(20, 90, 5)),
    (90, 90),
)
_OPEN_AGE = 90
_SEXES = ("FEMALE", "MALE")
#: The vendored ONS mid-year estimates by single year of age and sex (MYE2),
#: by the year of their 30 June.
UK_SPI_POPULATION_RESOURCES: Mapping[int, str] = {
    2023: "ons_single_year_age_populations.json",
    2025: "ons_mid2025_single_year_age_by_sex.json",
}
#: A UK tax year's midpoint, 1 October, is three months after the mid-year
#: estimate of its first calendar year.
_TAX_YEAR_MIDPOINT_MONTHS_AFTER_MID_YEAR = 3


def uk_spi_dated_measure_year(variable: str) -> int | None:
    """The tax year a dated SPI measure is computed at, else None."""

    match = UK_SPI_DATED_MEASURE.fullmatch(variable)
    return int(match[1]) if match else None


def uk_spi_dated_measure_kind(variable: str) -> str | None:
    """The measure a dated SPI name computes (``state_pension``), else None."""

    match = UK_SPI_DATED_MEASURE.fullmatch(variable)
    return match[2] if match else None


#: A dated SPI measure's name inside any text: a field, a filter, an expression.
_DATED_MEASURE_IN_TEXT = re.compile(
    r"\b" + UK_SPI_DATED_MEASURE.pattern.removeprefix("^").removesuffix("$") + r"\b"
)


def uk_spi_dated_measures_named(value: Any) -> set[str]:
    """Every dated SPI measure a binding names, in any field at any depth.

    A provider reads fields of its own (``affected_flag_variable``,
    ``count_of``, ``value_reduction``), so a check that lists the plain-column
    fields would miss a dated measure named there; this scans every string.
    """

    if isinstance(value, str):
        return {match.group(0) for match in _DATED_MEASURE_IN_TEXT.finditer(value)}
    if isinstance(value, Mapping):
        items: Any = list(value.keys()) + list(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        return set()
    return set().union(*(uk_spi_dated_measures_named(item) for item in items))


def _band_lower(age: int) -> int:
    for lower, upper in UK_SPI_POPULATION_AGE_BANDS:
        if lower <= age <= upper:
            return lower
    raise ValueError(f"age {age} is in no ONS calibration band.")


def ons_single_year_populations(year: int) -> dict[tuple[str, int], float]:
    """ONS's mid-``year`` UK persons by sex and single year (90 is 90+)."""

    try:
        resource = UK_SPI_POPULATION_RESOURCES[int(year)]
    except KeyError:
        raise ValueError(
            f"No vendored ONS mid-{year} single-year population; vendored years: "
            f"{sorted(UK_SPI_POPULATION_RESOURCES)}."
        ) from None
    populations: dict[tuple[str, int], float] = {}
    for row in vendored_rows(resource, period_value=str(year)):
        dimensions = row.get("dimensions") or {}
        raw_age = str(dimensions.get("age", ""))
        key = (
            str(dimensions.get("sex", "")).upper(),
            _OPEN_AGE if raw_age == "90_plus" else int(raw_age),
        )
        if key in populations:
            raise ValueError(f"{resource} repeats {key!r}.")
        populations[key] = float(row["value"])
    expected = {(sex, age) for sex in _SEXES for age in range(_OPEN_AGE + 1)}
    if set(populations) != expected:
        missing = sorted(expected - set(populations))[:5]
        raise ValueError(
            f"{resource} must carry ages 0 to {_OPEN_AGE} for both sexes; "
            f"missing {missing}."
        )
    return populations


def population_factors(
    fact_year: int,
    calibration_year: int,
    *,
    populations: Mapping[int, Mapping[tuple[str, int], float]] | None = None,
) -> dict[tuple[str, int], float]:
    """Each sex and band's population at the fact year over the calibration's.

    Keyed by sex and the band's lower age. The fact year's population is at
    its tax year's midpoint, interpolated linearly between the two mid-year
    estimates; the calibration year's is its mid-year estimate.
    """

    fact_year, calibration_year = int(fact_year), int(calibration_year)
    if fact_year >= calibration_year:
        raise ValueError(
            f"A dated SPI fact year ({fact_year}) must precede the calibration "
            f"year ({calibration_year})."
        )
    if populations is None:
        populations = {
            year: ons_single_year_populations(year)
            for year in (fact_year, calibration_year)
        }
    then, now = populations[fact_year], populations[calibration_year]
    share = _TAX_YEAR_MIDPOINT_MONTHS_AFTER_MID_YEAR / (
        12.0 * (calibration_year - fact_year)
    )
    factors = {}
    for sex in _SEXES:
        for lower, upper in UK_SPI_POPULATION_AGE_BANDS:
            ages = range(lower, upper + 1)
            before = sum(float(then[(sex, age)]) for age in ages)
            after = sum(float(now[(sex, age)]) for age in ages)
            if not (before > 0.0 and after > 0.0):
                raise ValueError(
                    f"ONS population of {sex} {lower}-{upper} must be positive."
                )
            factors[(sex, lower)] = (before + share * (after - before)) / after
    return factors


def population_factor_for_people(
    age: Any, sex: Any, factors: Mapping[tuple[str, int], float]
) -> np.ndarray:
    """Each person's factor by the ONS band of their whole-year age and sex."""

    ages = np.asarray(age, dtype=float)
    sexes = np.asarray(pd.Series(np.asarray(sex)).astype(str).str.upper())
    if ages.shape != sexes.shape:
        raise ValueError("age and sex must align person for person.")
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError("Every person needs a finite, non-negative age.")
    unknown = sorted(set(sexes) - set(_SEXES))
    if unknown:
        raise ValueError(f"Unknown sex values for the population factor: {unknown}.")
    whole = np.minimum(np.floor(ages).astype(int), _OPEN_AGE)
    lower = np.array([_band_lower(int(value)) for value in whole], dtype=int)
    return np.array(
        [factors[(s, int(b))] for s, b in zip(sexes, lower, strict=True)],
        dtype=float,
    )


def back_projection_factors(
    index_variables: Mapping[str, Sequence[str]],
    growth: Mapping[str, Mapping[int, float]],
    *,
    from_year: int,
    to_year: int,
) -> dict[str, float]:
    """The factor the engine carries each indexed input by, ``to`` to ``from``.

    ``index_variables`` is the engine's index-to-inputs map and ``growth``
    each index's year-on-year growth by the year it enters. An input carried
    by an index from ``to_year`` (earlier) to ``from_year`` (the dataset's
    year) is multiplied by one plus the growth of every year after
    ``to_year`` up to ``from_year``; an input listed under two indices
    refuses, since the engine's last write would then decide.
    """

    if to_year >= from_year:
        raise ValueError(
            f"Back-projection runs to an earlier year: {to_year} >= {from_year}."
        )
    factors: dict[str, float] = {}
    owner: dict[str, str] = {}
    for index, variables in index_variables.items():
        product = 1.0
        for year in range(to_year + 1, from_year + 1):
            rate = float(growth[index][year])
            if not np.isfinite(rate) or rate <= -1.0:
                raise ValueError(f"{index} growth for {year} is not a growth rate.")
            product *= 1.0 + rate
        for variable in dict.fromkeys(variables):
            if variable in owner and owner[variable] != index:
                raise ValueError(
                    f"{variable} is uprated by both {owner[variable]} and {index}."
                )
            owner[variable] = index
            factors[variable] = product
    return factors


def back_projected_inputs(
    tables: Mapping[str, pd.DataFrame],
    factors: Mapping[str, float],
    *,
    dropped: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    """The stored inputs at the earlier year: indexed ones divided, rest held.

    ``tables`` are the dataset year's entity tables. A numeric column with a
    factor is divided by it; every other column is copied; ``dropped`` names
    year-specific inputs the engine never carries into another year.
    """

    projected = {}
    for entity, table in tables.items():
        out = table.drop(columns=[c for c in dropped if c in table.columns]).copy()
        for column in out.columns:
            factor = factors.get(column)
            if factor is None:
                continue
            if not pd.api.types.is_numeric_dtype(out[column]):
                raise ValueError(
                    f"{entity}.{column} is uprated by an index but is not numeric."
                )
            out[column] = out[column].to_numpy(dtype=float) / float(factor)
        projected[entity] = out
    return projected


def round_trip_mismatches(
    stored: Mapping[str, pd.DataFrame],
    carried_forward: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    *,
    rtol: float = 1e-6,
    atol: float = 0.01,
    compared_where_unchanged: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Columns the engine's forward step does not carry back to their values.

    ``carried_forward`` is the engine's own uprating of the back-projected
    inputs to the dataset's year. Returns each listed column whose values
    differ from ``stored`` beyond tolerance, with its largest absolute miss;
    a column of labels misses by the share of its rows that differ, and a
    column the forward step lost or reshaped by infinity. A missing value on
    one side only misses by infinity; missing on both sides matches.

    ``compared_where_unchanged`` maps a column to a label column of the same
    entity, for a forward rule that resets the amount to zero exactly where
    it reassigns the label: the column is compared on the rows whose label
    the forward step left as stored, and must be zero on the others. The
    stored amount on a reassigned row is overwritten, so it cannot be checked.
    """

    conditions = dict(compared_where_unchanged or {})
    misses: dict[str, float] = {}
    for entity, table in stored.items():
        forward = carried_forward[entity]
        for column in columns:
            if column not in table.columns:
                continue
            if column not in forward.columns:
                misses[column] = float("inf")
                continue
            if len(table) != len(forward):
                misses[column] = float("inf")
                continue
            numeric = pd.api.types.is_numeric_dtype(table[column])
            if not numeric or not pd.api.types.is_numeric_dtype(forward[column]):
                labels = table[column].astype(str).to_numpy()
                differ = labels != forward[column].astype(str).to_numpy()
                if differ.any():
                    misses[column] = float(differ.mean())
                continue
            expected = table[column].to_numpy(dtype=float)
            actual = forward[column].to_numpy(dtype=float)
            label = conditions.get(column)
            if label is not None:
                if label not in table.columns or label not in forward.columns:
                    misses[column] = float("inf")
                    continue
                kept = (
                    table[label].astype(str).to_numpy()
                    == forward[label].astype(str).to_numpy()
                )
                reset = actual[~kept]
                if (~np.isfinite(reset)).any() or (reset != 0.0).any():
                    misses[column] = float("inf")
                    continue
                expected, actual = expected[kept], actual[kept]
            missing = np.isnan(expected)
            if (missing != np.isnan(actual)).any():
                misses[column] = float("inf")
                continue
            gap = np.abs(actual[~missing] - expected[~missing])
            if (gap > atol + rtol * np.abs(expected[~missing])).any():
                misses[column] = float(gap.max())
    return misses
