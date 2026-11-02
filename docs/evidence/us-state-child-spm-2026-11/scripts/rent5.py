"""Rent, real estate tax and housing receipt by tenure across files, including the ACS-local release."""

import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = S + "725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5"
p = pd.read_hdf(f, "person")
h = pd.read_hdf(f, "household")
s = pd.read_hdf(f, "spm_unit")
print(
    "rent cols",
    [
        c
        for c in p.columns
        if "rent" in c.lower() and "parent" not in c and "current" not in c
    ],
    [
        c
        for c in h.columns
        if "rent" in c.lower() or "RNTP" in c or "tenure" in c.lower()
    ],
)
print(
    "hh tenure",
    h.tenure_type.astype(str).value_counts(normalize=True).round(3).to_dict(),
)
a = p.groupby("person_household_id")[
    ["pre_subsidy_rent"] + (["real_estate_taxes"] if "real_estate_taxes" in p else [])
].sum()
h = h.merge(a, left_on="household_id", right_index=True)
w = "household_weight"
t = (
    h.groupby([h.household_support_channel.astype(str), h.tenure_type.astype(str)])
    .apply(
        lambda g: pd.Series(
            {
                "w": g[w].sum() / 1e6,
                "any_rent": np.average(g.pre_subsidy_rent > 0, weights=g[w] + 1e-12),
                "mean_pos": g.pre_subsidy_rent[g.pre_subsidy_rent > 0].mean(),
                "any_ret": np.average(
                    g.get("real_estate_taxes", pd.Series(0, index=g.index)) > 0,
                    weights=g[w] + 1e-12,
                ),
                "n": len(g),
            }
        )
    )
    .round(3)
)
print(t.to_string())
heads = p[p.is_household_head.astype(bool)]
print(
    "same position household/head",
    round((heads.person_household_id.values == h.household_id.values).mean(), 4)
    if len(heads) == len(h)
    else (len(heads), len(h)),
)
m = [c for c in s.columns if "hous" in c.lower()]
print(
    {
        c: round(
            float(pd.to_numeric(s[c], errors="coerce").fillna(0).astype(float).mean()),
            4,
        )
        for c in m
    }
)
