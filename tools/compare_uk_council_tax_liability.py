#!/usr/bin/env python
"""Compare the engine's council tax with the stored FRS bill, by authority and band.

CT-C (microcosm#1095, #953, #736). policyengine-uk computes a household's
council tax from its local authority's Band D amount and the statutory band
ratios (``simulated_council_tax``), and from the statutory discount too
(``simulated_council_tax_less_discounts``; PolicyEngine/policyengine-uk#1855
and #1864, in the lock since 2.151.0). The UK build stores the FRS bill
before Council Tax Reduction as ``council_tax`` (microcosm#1100), which the
engine reads as given. This tool reads all three at one year on a dense H5
the engine loads, whose households carry ``local_authority``, ``la_code_oa``
and ``household_source_id``, and writes:

- each one's weighted total, and the ratio of each simulated total to the
  stored one, for Great Britain, each nation and each region; and
- for each local authority and band, each one's weighted mean per household
  and the same two ratios, where at least ``UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS``
  distinct source households carry the cell.

A cell gives means, never totals or weighted counts, so a published total less
the published cells cannot give a withheld cell back. A thinner cell is listed
only as withheld, and a total that rests on fewer source households is
withheld too. Northern Ireland levies domestic rates, not council tax, and is
left out. Private development diagnostic: it never recalibrates, and its
output stays local until a disclosure check clears it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.diagnostics import UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS

#: The engine's two council tax liabilities, each compared with the stored bill.
SIMULATED = ("simulated_council_tax", "simulated_council_tax_less_discounts")
STORED = "council_tax"
_MEASURES = (STORED, *SIMULATED)
#: The stored household columns the comparison is cut by.
_STORED_COLUMNS = ("household_id", "la_code_oa", "household_source_id")


def household_table(dataset: str, year: int) -> pd.DataFrame:
    """One row per household: the three liabilities at ``year`` and the cuts."""

    from policyengine_uk import Microsimulation

    stored = pd.read_hdf(dataset, "household")
    missing = [column for column in _STORED_COLUMNS if column not in stored]
    if missing:
        raise ValueError(
            f"{dataset} carries no {missing}: CT-C needs a dense release whose "
            "households have an authority and a source household."
        )
    simulation = Microsimulation(dataset=dataset)

    def values(name: str) -> np.ndarray:
        return np.asarray(simulation.calculate(name, year).values)

    table = pd.DataFrame(
        {
            "household_id": values("household_id").astype("int64"),
            "household_weight": values("household_weight").astype(float),
            "local_authority": values("local_authority").astype(str),
            "council_tax_band": values("council_tax_band").astype(str),
            "country": values("country").astype(str),
            "region": values("region").astype(str),
            **{name: values(name).astype(float) for name in _MEASURES},
        }
    )
    cuts = stored.loc[:, list(_STORED_COLUMNS)].astype({"household_id": "int64"})
    joined = table.merge(cuts, on="household_id", how="left", validate="one_to_one")
    if joined["la_code_oa"].isna().any():
        raise ValueError(f"{dataset}: a simulated household has no stored row.")
    return joined


def _support(rows: pd.DataFrame) -> int:
    """Distinct source households with a positive weight among ``rows``."""

    return int(rows.loc[rows["household_weight"] > 0, "household_source_id"].nunique())


def _ratios(numerators: Mapping[str, float], stored: float) -> dict[str, float | None]:
    return {
        f"{name}_over_{STORED}": (None if stored == 0 else value / stored)
        for name, value in numerators.items()
    }


def _totals(rows: pd.DataFrame, floor: int) -> dict[str, Any]:
    support = _support(rows)
    if support < floor:
        return {"withheld": True}
    weight = rows["household_weight"].to_numpy(dtype=float)
    totals = {
        name: float(np.dot(weight, rows[name].to_numpy(dtype=float)))
        for name in _MEASURES
    }
    return {
        "source_households": support,
        "totals": totals,
        "ratios": _ratios({name: totals[name] for name in SIMULATED}, totals[STORED]),
    }


def _cell(rows: pd.DataFrame, floor: int) -> dict[str, Any]:
    support = _support(rows)
    weight = rows["household_weight"].to_numpy(dtype=float)
    if support < floor or weight.sum() <= 0:
        return {"withheld": True}
    means = {
        name: float(np.average(rows[name].to_numpy(dtype=float), weights=weight))
        for name in _MEASURES
    }
    return {
        "source_households": support,
        "means": means,
        "ratios": _ratios({name: means[name] for name in SIMULATED}, means[STORED]),
    }


def summarise(
    table: pd.DataFrame, *, floor: int = UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS
) -> dict[str, Any]:
    """The totals and the authority-by-band cells, under the support floor."""

    great_britain = table.loc[table["country"] != "NORTHERN_IRELAND"]
    cells = []
    for (code, band), rows in great_britain.groupby(
        ["la_code_oa", "council_tax_band"], sort=True
    ):
        authority = sorted(set(rows["local_authority"]))
        cells.append(
            {
                "la_code_oa": str(code),
                "local_authority": authority[0] if len(authority) == 1 else authority,
                "council_tax_band": str(band),
                **_cell(rows, floor),
            }
        )
    no_band_d = great_britain.loc[
        (great_britain["simulated_council_tax"] == 0)
        & (great_britain["household_weight"] > 0)
    ]
    return {
        "floor_source_households": floor,
        "great_britain": _totals(great_britain, floor),
        "by_country": {
            str(country): _totals(rows, floor)
            for country, rows in great_britain.groupby("country", sort=True)
        },
        "by_region": {
            str(region): _totals(rows, floor)
            for region, rows in great_britain.groupby("region", sort=True)
        },
        "cells": cells,
        # Authorities the engine charges nothing, such as districts abolished
        # in April 2023, which have no Band D amount: a name, not a figure.
        "authorities_without_band_d": sorted(set(no_band_d["la_code_oa"].astype(str))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="a dense H5 the engine loads")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from importlib.metadata import version

    report = {
        "dataset": args.dataset,
        "year": args.year,
        "policyengine_uk": version("policyengine-uk"),
        **summarise(household_table(args.dataset, args.year)),
    }
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["great_britain"], indent=1))


if __name__ == "__main__":
    main()
