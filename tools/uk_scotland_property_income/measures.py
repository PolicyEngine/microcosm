"""Person contributions to the SPI facts, on the build's own bindings (microcosm#1326 A4).

``uk_population_targets.json`` binds each fact as a weighted sum over persons:

- property amount: ``property_income`` of taxpayers (``income_tax > 0``);
- property count: taxpayers with ``property_income > 0``;
- regional taxpayers: persons with ``income_tax > 0``;
- regional total income: ``total_income_after_loss_relief`` of everyone in the
  band (the binding has no taxpayer filter);
- regional income tax: ``income_tax`` of everyone in the band.

Every band reads ``total_income_after_loss_relief``: the engine's
``total_income`` with self-employment profit after losses brought forward
(``uk_runtime/measure_simulation.py``, microcosm#1095).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from tools.uk_scotland_property_income.registry import SpiFact

#: Person columns the contributions read. ``household_row`` is the position of
#: the person's household in the household table.
PERSON_COLUMNS = (
    "household_row",
    "region",
    "income_tax",
    "property_income",
    "total_income_after_loss_relief",
)


def total_income_after_loss_relief(
    total_income: object, self_employment_income: object, loss_relief: object
) -> np.ndarray:
    """Total income with the trading profit after losses brought forward."""

    profit = np.asarray(self_employment_income, dtype=float)
    after = np.maximum(profit - np.asarray(loss_relief, dtype=float), 0.0)
    return np.asarray(total_income, dtype=float) - profit + after


def contribution(persons: pd.DataFrame, fact: SpiFact) -> np.ndarray:
    """Each person's unweighted contribution to the fact."""

    income = persons["total_income_after_loss_relief"].to_numpy(dtype=float)
    selected = (income >= fact.lower) & (income < fact.upper)
    if fact.region is not None:
        selected &= persons["region"].to_numpy() == fact.region
    tax = persons["income_tax"].to_numpy(dtype=float)
    profit = persons["property_income"].to_numpy(dtype=float)
    if fact.measure == "property_amount":
        value = np.where(tax > 0, profit, 0.0)
    elif fact.measure == "property_count":
        value = ((tax > 0) & (profit > 0)).astype(float)
    elif fact.measure == "taxpayers":
        value = (tax > 0).astype(float)
    elif fact.measure == "total_income":
        value = income
    elif fact.measure == "income_tax":
        value = tax
    else:
        raise ValueError(f"Unknown measure {fact.measure!r}.")
    return np.where(selected, value, 0.0)


def household_matrix(
    persons: pd.DataFrame, facts: Sequence[SpiFact], n_households: int
) -> sparse.csr_matrix:
    """Households by facts: each household's summed contribution.

    A household's weight multiplies every member, so the fact's estimate is
    ``matrix.T @ household_weights``.
    """

    households = persons["household_row"].to_numpy(dtype=np.int64)
    rows, cols, data = [], [], []
    for column, fact in enumerate(facts):
        values = contribution(persons, fact)
        nonzero = values != 0
        rows.append(households[nonzero])
        cols.append(np.full(int(nonzero.sum()), column, dtype=np.int64))
        data.append(values[nonzero])
    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_households, len(facts)),
    )
    return matrix.tocsr()


def region_matrix(
    regions: Sequence[str], household_regions: object
) -> sparse.csr_matrix:
    """Households by regions: one where the household sits in the region."""

    household_regions = np.asarray(household_regions)
    columns = [
        sparse.csr_matrix((household_regions == region).astype(float)[:, None])
        for region in regions
    ]
    return sparse.hstack(columns, format="csr")
