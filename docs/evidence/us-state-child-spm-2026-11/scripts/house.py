"""Where do Census housing subsidy recipients drop out of the PE chain? Default file, asec rows, child units."""

import glob
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = glob.glob(S + "9a814a3b*/populace_us_2024.h5")[0]
s = pd.read_hdf(f, "spm_unit")
print([c for c in s.columns if "hous" in c.lower() or "take" in c.lower()][:20])
sid = [c for c in s.columns if c.startswith("spm_unit_id")][0]
cols = [c for c in s.columns if "housing" in c.lower()]
print(
    {
        c: (
            round(
                float(
                    pd.to_numeric(s[c], errors="coerce").fillna(0).astype(float).mean()
                ),
                4,
            )
        )
        for c in cols
    }
)
u = pd.read_parquet(
    f"{SP}/run/default_units.parquet",
    columns=[
        "spm_unit_id",
        "kids",
        "w1",
        "state",
        "household_support_channel",
        "SPM_CAPHOUSESUB",
        "SPM_RESOURCES",
        "SPM_POVTHRESHOLD",
        "spm_unit_capped_housing_subsidy__2024",
    ],
)
u = u[(u.kids > 0) & (u.household_support_channel == "asec")].copy()
u["W"] = u.w1 * u.kids
u = u.merge(s[[sid] + cols], left_on="spm_unit_id", right_on=sid, how="left")
rec = u.SPM_CAPHOUSESUB.fillna(0) > 0
print(
    "census recipient share of child weight",
    round(u.W[rec].sum() / u.W.sum(), 4),
    "PE subsidy>0 share",
    round(u.W[u.spm_unit_capped_housing_subsidy__2024 > 0].sum() / u.W.sum(), 4),
)
for c in cols:
    v = pd.to_numeric(u[c], errors="coerce").fillna(0)
    print(
        c,
        "among census recipients",
        round(np.average(v[rec] > 0, weights=u.W[rec]), 3),
        "among others",
        round(np.average(v[~rec] > 0, weights=u.W[~rec]), 3),
    )
print(
    "census recipients with PE subsidy>0",
    round(
        np.average(u.spm_unit_capped_housing_subsidy__2024[rec] > 0, weights=u.W[rec]),
        3,
    ),
)
print(
    "mean subsidy among census recipients: census",
    round(np.average(u.SPM_CAPHOUSESUB[rec], weights=u.W[rec])),
    "PE",
    round(np.average(u.spm_unit_capped_housing_subsidy__2024[rec], weights=u.W[rec])),
)
both = rec & (u.spm_unit_capped_housing_subsidy__2024 > 0)
print(
    "mean among both: census",
    round(np.average(u.SPM_CAPHOUSESUB[both], weights=u.W[both])),
    "PE",
    round(np.average(u.spm_unit_capped_housing_subsidy__2024[both], weights=u.W[both])),
)
