"""microcosm#1278 C1 receipt: compare two spines' council-tax bands (counts as bands only).

Usage: compare_twin_bands.py <control.h5> <branch.h5> <out.json>

The two spines are built from the licensed inputs with the same recipe and seed,
one on main and one on the branch, and must hold the same base households (those
that are not CGT copies). Reports, by nation:
  - base households whose band lies outside the nation's bands, in each spine;
  - base households whose band changed between the spines, by support channel;
  - the base-household columns that differ, with the rows they differ on at all and
    by more than 0.1% and 1% of the control value (numeric columns).
Every count of rows or households is written only as a band (0, <10, 10-29, 30+), and
the script writes no weighted figure. The output stays with the licensed spines,
outside the repository; experiments/1278-uk-local-surface-receipts.md reports it.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS_BY_NATION,
    UK_NATION_BY_REGION,
)


def band(n: int) -> str:
    return "0" if n == 0 else "<10" if n < 10 else "10-29" if n < 30 else "30+"


def load(path: str) -> dict[str, pd.DataFrame]:
    with pd.HDFStore(path, mode="r") as store:
        return {name: store[name] for name in ("person", "benunit", "household")}


def outside(household: pd.DataFrame) -> pd.Series:
    nation = household["region"].astype(str).map(UK_NATION_BY_REGION)
    allowed = nation.map(UK_COUNCIL_TAX_BANDS_BY_NATION)
    return pd.Series(
        [
            b not in a
            for b, a in zip(
                household["council_tax_band"].astype(str), allowed, strict=True
            )
        ],
        index=household.index,
    )


_COPY_FLAGS = (
    "household_is_capital_gains_clone",
    "household_is_cgt_support_copy",
    "household_is_cgt_residential_clone",
)
_LINEAGE_KEY = [
    "household_source_id",
    "household_support_channel",
    "household_support_clone_index",
    "household_is_spi_income_band_donor",
    "spi_income_band_donor_lower_bound",
]


def base_households(household: pd.DataFrame) -> pd.DataFrame:
    """Households that are not CGT copies, keyed by lineage.

    Copy stages after the housing shell can mint a different number of copies
    when a band moves (they condition on housing and wealth), so household
    ids shift; the lineage key names the same household in both spines.
    """

    copy = household[list(_COPY_FLAGS)].astype(bool).any(axis=1)
    base = household.loc[~copy].copy()
    base.index = pd.MultiIndex.from_frame(base[_LINEAGE_KEY].astype(str))
    if not base.index.is_unique:
        raise SystemExit("The lineage key does not name each base household once.")
    return base.sort_index()


def main(control_path: str, branch_path: str, out_path: str) -> None:
    control, branch = load(control_path), load(branch_path)
    ch, bh = base_households(control["household"]), base_households(branch["household"])
    if not ch.index.equals(bh.index):
        raise SystemExit("The two spines do not hold the same base households.")
    nation = ch["region"].astype(str).map(UK_NATION_BY_REGION)
    channel = ch["household_support_channel"].astype(str)
    changed = (
        ch["council_tax_band"].astype(str).to_numpy()
        != bh["council_tax_band"].astype(str).to_numpy()
    )
    out_c = outside(ch.reset_index(drop=True)).to_numpy()
    out_b = outside(bh.reset_index(drop=True)).to_numpy()
    out: dict[str, object] = {
        "rows": {
            entity: {
                "control": band(len(control[entity])),
                "branch": band(len(branch[entity])),
                "difference": band(abs(len(control[entity]) - len(branch[entity]))),
            }
            for entity in ("person", "benunit", "household")
        },
        "base_households": band(len(ch)),
        "rows_outside_their_nation": {
            "control_all_rows": band(int(outside(control["household"]).sum())),
            "branch_all_rows": band(int(outside(branch["household"]).sum())),
        },
        "changed_bands_were_all_outside_in_control": bool((~changed | out_c).all()),
        "valid_control_bands_changed": band(int((changed & ~out_c).sum())),
        "nations": {},
        "base_household_columns_that_differ": {},
    }
    for name in UK_COUNCIL_TAX_BANDS_BY_NATION:
        rows = (nation == name).to_numpy()
        record = {
            "outside_nation_bands_control": band(int((rows & out_c).sum())),
            "outside_nation_bands_branch": band(int((rows & out_b).sum())),
            "band_changed": {
                ch_name: band(
                    int((rows & changed & (channel == ch_name).to_numpy()).sum())
                )
                for ch_name in ("frs", "spi")
            },
        }
        out["nations"][name] = record
    for column in sorted(set(ch.columns) & set(bh.columns)):
        a, b = ch[column].reset_index(drop=True), bh[column].reset_index(drop=True)
        same = (a == b) | (a.isna() & b.isna())
        if bool(same.all()):
            continue
        record = {"any": band(int((~same).sum()))}
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            gap = np.abs(y - x) / np.maximum(np.abs(x), 1e-9)
            for threshold, label in ((0.001, "over_0.1%"), (0.01, "over_1%")):
                moved = ~same.to_numpy() & ~(gap <= threshold)
                record[label] = band(int(moved.sum()))
        out["base_household_columns_that_differ"][column] = record
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    print(json.dumps(out, indent=1)[:8000])


if __name__ == "__main__":
    main(*sys.argv[1:4])
