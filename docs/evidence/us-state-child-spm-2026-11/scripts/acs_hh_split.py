"""Household and SPM-unit keys for clustering, from the per-state split of the ACS-local file. Also checks the split's size and the rent totals."""

import glob
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
D = f"{SP}/run/acs_split/"
right = (
    pd.read_parquet(f"{SP}/run/rent_right.parquet")
    .set_index("household_source_id")
    .rent_right
)
hs, us = [], []
was = now = 0.0
n_p = cols = 0
for f in sorted(glob.glob(D + "*-household.pkl.gz")):
    h = pd.read_pickle(f)
    p = pd.read_pickle(f.replace("-household", "-person"))
    hs.append(
        h[
            [
                "household_id",
                "state_fips",
                "household_support_channel",
                "household_source_id",
                "household_weight",
                "TYPEHUGQ",
            ]
        ]
    )
    us.append(p[["person_household_id", "person_spm_unit_id"]].drop_duplicates())
    acs = h.household_support_channel.astype(str).str.startswith("acs").values
    hr = pd.Series(
        np.where(
            acs,
            12
            * pd.to_numeric(h.acs_monthly_contract_rent, errors="coerce")
            .fillna(0)
            .values,
            h.household_source_id.map(right).values,
        ),
        index=h.household_id.values,
    )
    assert hr.notna().all()
    head = p.is_household_head.astype(bool).values
    was += p.pre_subsidy_rent.sum()
    now += p.person_household_id[head].map(hr).sum()
    n_p += len(p)
    cols = p.shape[1]
h = pd.concat(hs, ignore_index=True)
u = pd.concat(us, ignore_index=True)
h.to_parquet(f"{SP}/run/acs_hh.parquet", compression="zstd")
u.to_parquet(f"{SP}/run/acs_unit_hh.parquet", compression="zstd")
print(
    "households", h.shape, "unit keys", u.shape, "people", n_p, "person columns", cols
)
print("rent was", round(was / 1e9, 2), "now", round(now / 1e9, 2))
print("weight total", round(h.household_weight.sum()))
