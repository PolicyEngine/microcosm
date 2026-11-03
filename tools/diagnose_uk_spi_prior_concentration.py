"""Size the weight concentration of the over-£100,000 young-child cell on a UK dataset.

The cell Vahid's review of microcosm#1338 (round 1, item 4) asked to see
before and after the SPI channel's prior is shaped by its copies' design
weights: households with a benefit unit that has a child aged 0 to 4 and an
adult whose adjusted net income is above £100,000 at the year, the families
the childcare cliff-edge reaches. On a spine (prior weights) or a calibrated
dataset (final weights), the tool reports the cell's weighted households, its
Kish effective sample size over households and over their FRS source
households (a source household and its copies as one unit of their joint
weight), the share of its weight its ten largest source households hold,
the share on SPI-channel copies, and its support in source households,
rounded. Every figure is an aggregate over the cell, and a share is never
given for a cell of one or two source households. Development diagnostic: it
never recalibrates and writes only an aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    UK_WITHHELD,
    rounded_count,
)

#: The cell: an adult above this adjusted net income in a unit with a child
#: below this age (the childcare cliff-edge, microcosm#1339).
INCOME_ABOVE_GBP = 100_000.0
CHILD_AGE_BELOW = 5
#: The concentration share's units: the cell's largest source households.
LARGEST_SOURCES = 10


def kish_effective_size(weights: np.ndarray) -> float:
    """Kish's effective sample size of positive weights: (sum w)^2 / sum w^2."""

    weights = np.asarray(weights, dtype=float)
    weights = weights[weights > 0]
    if not len(weights):
        return 0.0
    return float(weights.sum() ** 2 / (weights**2).sum())


def concentration(
    weights: np.ndarray, sources: np.ndarray, spi_copy: np.ndarray
) -> dict[str, object]:
    """The cell's concentration figures from its households' weights and sources.

    ``weights``, ``sources`` and ``spi_copy`` are one entry per household of
    the cell. Shares and effective sizes are withheld ("<3") for a cell on
    one or two source households.
    """

    weights = np.asarray(weights, dtype=float)
    by_source = pd.Series(weights).groupby(np.asarray(sources)).sum()
    support = int((by_source > 0).sum())
    total = float(weights.sum())
    if support < UK_DISCLOSURE_FLOOR:
        return {"source_households": UK_WITHHELD if support else 0, "withheld": True}
    largest = by_source.sort_values(ascending=False).head(LARGEST_SOURCES)
    return {
        "weighted_households": total,
        "kish_effective_households": kish_effective_size(weights),
        "kish_effective_source_households": kish_effective_size(by_source.to_numpy()),
        f"largest_{LARGEST_SOURCES}_source_households_weight_share": float(
            largest.sum() / total
        ),
        "spi_copy_weight_share": float(
            weights[np.asarray(spi_copy, bool)].sum() / total
        ),
        "source_households": rounded_count(support),
    }


def cell_households(dataset: str, *, year: int) -> pd.DataFrame:
    """The cell's households with their weights, source households and SPI flag."""

    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=dataset)

    def values(variable: str, **kwargs) -> np.ndarray:
        return np.asarray(simulation.calculate(variable, year, **kwargs).values)

    age = values("age").astype(float)
    income = values("adjusted_net_income").astype(float)
    benunit = values("benunit_id", map_to="person")
    household = values("household_id", map_to="person")
    person = pd.DataFrame(
        {
            "benunit": benunit,
            "household": household,
            "child": age < CHILD_AGE_BELOW,
            "high": (age >= 18) & (income > INCOME_ABOVE_GBP),
        }
    )
    units = person.groupby("benunit")[["child", "high"]].any()
    in_cell = units.index[units["child"] & units["high"]]
    households = person.loc[person["benunit"].isin(in_cell), "household"].unique()
    table = pd.read_hdf(dataset, "household").set_index("household_id")
    ids = values("household_id")
    weight = pd.Series(values("household_weight").astype(float), index=ids)
    cell = table.loc[households]
    return pd.DataFrame(
        {
            "weight": weight.loc[households].to_numpy(),
            "source": cell["source_household_id"].to_numpy(),
            "spi_copy": cell["household_is_spi_synthetic"].astype(bool).to_numpy(),
        },
        index=households,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="+", type=Path, help="UK spine or calibrated H5s."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for dataset in args.datasets:
        cell = cell_households(str(dataset), year=args.year)
        report[str(dataset)] = {
            "year": args.year,
            "cell": (
                f"households with a benefit unit holding a child under "
                f"{CHILD_AGE_BELOW} and an adult whose adjusted net income is "
                f"above {INCOME_ABOVE_GBP:,.0f}"
            ),
            **concentration(
                cell["weight"].to_numpy(),
                cell["source"].to_numpy(),
                cell["spi_copy"].to_numpy(),
            ),
        }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
