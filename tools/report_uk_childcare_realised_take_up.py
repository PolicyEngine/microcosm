#!/usr/bin/env python
"""Report the childcare take-up a built UK file realises at its final weights.

The take-up contract holds the rates the spine draws at (``frs_take_up``); a
costing reads the calibrated file, whose weights move toward the units the DfE
and HMRC rows need, so the share a reform actually uses can differ from the
contract rate (Vahid's round-2 review of microcosm#1338, item 3). This tool
reads a built H5's frame tables and final household weights (no engine) and
writes, for each childcare flag, the contract rate beside the realised share,
unweighted and weighted, over:

- benefit units with a child under 5;
- the same split into native units and copies, where the file keeps lineage;
- benefit units with a child under 5 and an adult whose employment plus
  self-employment income exceeds £100,000 (a pay proxy, not adjusted net
  income);

and each DfE age's under-2 usage share over the children of that age in units that
claim the working-parent hours. Every cell names its carriers as distinct source
households and reads "<3" in place of its shares when fewer than 3 carry it
(UKDS disclosure control). The receipt belongs in the run tree; it is never
written into the take-up contract, a hashed build input.

    uv run python tools/report_uk_childcare_realised_take_up.py \\
        --h5 <run>/microcosm_uk_2024_25.h5 --output <run>/childcare_realised_take_up.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_take_up import (
    UK_CHILDCARE_TAKE_UP_FLAGS,
    UK_EXTENDED_CHILDCARE_OUTPUT,
    UK_UNDER_TWO_USAGE_KEYS,
    UK_UNDER_TWO_USAGE_OUTPUT,
    person_dfe_age,
    person_unit_flag,
    under_two_children,
    universal_childcare_marginal_rate,
)
from microcosm.build.uk_runtime.receipt_counts import UK_DISCLOSURE_FLOOR, UK_WITHHELD
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)

MINIMUM_CARRIERS = UK_DISCLOSURE_FLOOR
HIGH_PAY_THRESHOLD = 100_000.0
YOUNG_CHILD_AGE = 5
_PAY_COLUMNS = ("employment_income", "self_employment_income")


def _contract_rate(key: str, contract: UKTakeUpContract) -> float:
    if key == "universal_childcare":
        return universal_childcare_marginal_rate(contract)
    return contract.rate(key)


def _cell(
    values: np.ndarray,
    weights: np.ndarray,
    sources: np.ndarray,
    mask: np.ndarray,
) -> Mapping[str, object]:
    carriers = int(np.unique(sources[mask]).size)
    if carriers < MINIMUM_CARRIERS:
        return {"source_households": UK_WITHHELD}
    chosen = np.asarray(values, dtype=float)[mask]
    return {
        "source_households": carriers,
        "unweighted_share": float(chosen.mean()),
        "weighted_share": float(np.average(chosen, weights=weights[mask])),
    }


def realised_childcare_take_up(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    *,
    contract: UKTakeUpContract,
    copies: np.ndarray | None = None,
) -> dict[str, object]:
    """The receipt's body for one file (frame tables and final weights)."""

    weight_of = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    source_of = household.set_index("household_id")["source_household_id"]
    unit_household = (
        person.drop_duplicates("person_benunit_id")
        .set_index("person_benunit_id")["person_household_id"]
        .reindex(benunit["benunit_id"].to_numpy())
    )
    unit_weights = weight_of.loc[unit_household.to_numpy()].to_numpy()
    unit_sources = source_of.loc[unit_household.to_numpy()].to_numpy()
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    young = (
        pd.Series(age < YOUNG_CHILD_AGE)
        .groupby(person["person_benunit_id"].to_numpy())
        .any()
        .reindex(benunit["benunit_id"].to_numpy(), fill_value=False)
        .to_numpy(dtype=bool)
    )
    pay = sum(
        pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        for column in _PAY_COLUMNS
        if column in person.columns
    )
    high_pay = (
        pd.Series(np.asarray(pay) > HIGH_PAY_THRESHOLD)
        .groupby(person["person_benunit_id"].to_numpy())
        .any()
        .reindex(benunit["benunit_id"].to_numpy(), fill_value=False)
        .to_numpy(dtype=bool)
    )
    cells = {
        "units_with_a_child_under_5": young,
        "units_with_a_child_under_5_and_pay_over_100k": young & high_pay,
    }
    if copies is not None:
        cells["native_units_with_a_child_under_5"] = young & ~copies
        cells["copy_units_with_a_child_under_5"] = young & copies
    flags = {}
    for output, key in UK_CHILDCARE_TAKE_UP_FLAGS:
        values = benunit[output].to_numpy(dtype=bool)
        flags[output] = {
            "contract_key": key,
            "contract_rate": _contract_rate(key, contract),
            "cells": {
                name: _cell(values, unit_weights, unit_sources, mask)
                for name, mask in cells.items()
            },
        }
    person_weights = weight_of.loc[person["person_household_id"].to_numpy()].to_numpy()
    person_sources = source_of.loc[person["person_household_id"].to_numpy()].to_numpy()
    claiming = person_unit_flag(
        person, benunit, benunit[UK_EXTENDED_CHILDCARE_OUTPUT].to_numpy()
    )
    uses = person[UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(dtype=float) > 0
    dfe_age = person_dfe_age(person)
    for child_age, key in UK_UNDER_TWO_USAGE_KEYS.items():
        flags[f"{UK_UNDER_TWO_USAGE_OUTPUT}.age_{child_age}"] = {
            "contract_key": key,
            "contract_rate": contract.rate(key),
            "cells": {
                f"children_aged_{child_age}_in_claiming_units": _cell(
                    uses,
                    person_weights,
                    person_sources,
                    under_two_children(person) & claiming & (dfe_age == child_age),
                )
            },
        }
    return {
        "minimum_carriers": MINIMUM_CARRIERS,
        "weights": "final household weights of the built file",
        "pay_proxy": "employment plus self-employment income over £100,000 for any "
        "adult of the unit (not adjusted net income)",
        "flags": flags,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from microcosm.build.uk_runtime.childcare_copy_take_up import copy_benunits
    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame

    frame, _ = load_uk_national_frame(args.h5)
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    try:
        copies = copy_benunits(person, benunit, household)
    except (KeyError, ValueError):
        copies = None  # a release file without lineage columns
    receipt = realised_childcare_take_up(
        person,
        benunit,
        household,
        frame.weights_for("household").values,
        contract=load_uk_take_up_contract(),
        copies=copies,
    )
    receipt["h5"] = str(args.h5)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
