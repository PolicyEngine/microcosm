"""One-year back-cast of a UK dataset with policyengine-uk's own indices (microcosm#1326 A2, A3).

policyengine-uk projects a dataset forward only. Asked for a year before the
dataset's own, it returns zero incomes, zero tax and almost no weight rather
than refusing. So the only way to 2023-24 from a 2024-25 file is to invert
the engine's own forward step: divide each input by the growth
``apply_single_year_uprating`` would multiply it by. Household weights move
with ONS population growth, as they do going forward.

What the back-cast leaves at the file's own year:

- the forward step's custom pieces. Council tax and rent move by region, and
  student-loan plans are re-labelled. None of them enters income tax.
- every stored input that ``uprating_indices.yaml`` doesn't map, which the
  forward step leaves flat too. Some feed income tax, such as ``gift_aid``,
  ``loss_relief``, ``capital_allowances``, ``employment_expenses`` and
  salary-sacrificed pension contributions. They are small, but not zero.
  ``held_at_file_year`` lists them for the run's manifest.

The indices are OBR national aggregates per person (per-capita GDP for
property income, mortgage interest for finance costs, ONS population for
weights). Differences between income bands are left to the reweight.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.data.economic_assumptions import (
    LEGACY_PROJECTION_END_YEAR,
    _uprating_index_map,
)

#: Inputs the forward step moves by its own rules, which this back-cast leaves
#: at the file's year.
NOT_INVERTED = ("council_tax", "rent", "student_loan_plan")


def refuse_earlier_period(dataset: UKSingleYearDataset, year: int) -> None:
    """Stop before asking the engine for a year it would silently zero."""

    if int(year) < int(dataset.time_period):
        raise ValueError(
            f"The dataset is for {dataset.time_period}; policyengine-uk returns zero "
            f"incomes and weights for {year} instead of refusing. Back-cast it first."
        )


def backcast_one_year(
    dataset: UKSingleYearDataset, parameters: Any
) -> tuple[UKSingleYearDataset, dict[str, dict[str, Any]]]:
    """The dataset one year earlier, and the growth divided out of each input."""

    year = int(dataset.time_period)
    if year > LEGACY_PROJECTION_END_YEAR:
        raise ValueError(
            f"Inputs outside uprating_indices.yaml also move after {LEGACY_PROJECTION_END_YEAR}; "
            "this back-cast inverts only the mapped ones."
        )
    tables = {
        "person": dataset.person.copy(),
        "benunit": dataset.benunit.copy(),
        "household": dataset.household.copy(),
    }
    divided: dict[str, dict[str, Any]] = {}
    for index_name, variables in _uprating_index_map().items():
        # The forward step reads the growth at the later year's start.
        growth = float(parameters.get_child(index_name)(str(year)))
        for variable in variables:
            for table in tables.values():
                if variable in table.columns:
                    table[variable] = table[variable] / (1.0 + growth)
                    divided[variable] = {"index": index_name, "growth": growth}
    earlier = UKSingleYearDataset(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        fiscal_year=year - 1,
    )
    return earlier, divided


def with_household_weights(
    dataset: UKSingleYearDataset, weights: Any
) -> UKSingleYearDataset:
    """A copy of the dataset with new household weights."""

    household = dataset.household.copy()
    household["household_weight"] = weights
    return UKSingleYearDataset(
        person=dataset.person.copy(),
        benunit=dataset.benunit.copy(),
        household=household,
        fiscal_year=int(dataset.time_period),
    )


#: Identifier and weight columns that carry no money amount.
_NOT_AMOUNTS = ("_id", "_weight", "_index", "_build_id", "_source_id", "_key")


def held_at_file_year(
    dataset: UKSingleYearDataset, divided: dict[str, Any]
) -> dict[str, list[str]]:
    """The stored numeric columns, by table, that the back-cast leaves at the
    file's own year: neither divided nor among the custom pieces."""

    held = {}
    for name in ("person", "benunit", "household"):
        table = getattr(dataset, name)
        held[name] = sorted(
            column
            for column in table.columns
            if pd.api.types.is_float_dtype(table[column].dtype)
            and column not in divided
            and column not in NOT_INVERTED
            and not column.endswith(_NOT_AMOUNTS)
        )
    return held
