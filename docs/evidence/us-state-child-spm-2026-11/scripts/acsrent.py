"""ACS-local file: corrected rent per household. Donors take the base record's right rent; ACS rows take twelve times native monthly contract rent."""

import glob
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = S + "725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5"
h = pd.read_hdf(f, "household")
p = (
    pd.read_hdf(f, "person")[
        [
            "person_household_id",
            "is_household_head",
            "pre_subsidy_rent",
            "person_support_channel" if False else "person_household_id",
        ]
    ]
    if False
    else pd.read_hdf(f, "person")
)
p = p[["person_household_id", "is_household_head", "pre_subsidy_rent"]]
ch = h.household_support_channel.astype(str)
print(ch.value_counts().to_dict())
de = pd.read_hdf(
    [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0], "household"
)[["household_id", "household_source_id"]]
fx = (
    pd.read_parquet(f"{SP}/run/rentfix_dense.parquet")
    .merge(de, on="household_id")
    .drop_duplicates("household_source_id")
    .set_index("household_source_id")
)
print(
    "ACS rows",
    h[ch.str.startswith("acs")][
        ["TEN", "RNTP", "GRNTP", "acs_monthly_contract_rent", "acs_monthly_gross_rent"]
    ]
    .describe()
    .loc[["count", "mean", "50%", "max"]]
    .round(1)
    .to_string(),
)
print(
    h[ch.str.startswith("acs")]
    .groupby(h.TEN.astype(str))
    .agg(
        n=("RNTP", "size"),
        rntp_pos=(
            "RNTP",
            lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean(),
        ),
        cr_pos=(
            "acs_monthly_contract_rent",
            lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean(),
        ),
    )
    .round(3)
    .to_string()
)
print(
    pd.crosstab(
        h.TEN[ch.str.startswith("acs")].astype(str),
        h.tenure_type[ch.str.startswith("acs")].astype(str),
    ).to_string()
)
hd = (
    p[p.is_household_head.astype(bool)]
    .set_index("person_household_id")
    .pre_subsidy_rent
)
h["was"] = h.household_id.map(hd)
print(
    "heads per household: households with no head",
    int(h.was.isna().sum()),
    "by channel",
    h.was.isna().groupby(ch).sum().to_dict(),
)
don = ~ch.str.startswith("acs")
m = h.household_source_id[don].map(fx.rent_was)
print(
    "donor source found",
    round(m.notna().mean(), 4),
    "donor rent equal",
    round(
        float(np.isclose(m.dropna(), h.was[don][m.notna()], rtol=1e-3, atol=1).mean()),
        4,
    ),
)
