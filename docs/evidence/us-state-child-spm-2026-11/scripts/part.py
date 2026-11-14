"""Arithmetic on real simulation output, not a re-simulation: what if partnership and S-corp income entered SPM resources before tax?"""

import glob
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
for tag, pat, fn in (
    ("dense", "15cefac4*", "*.h5"),
    ("default", "9a814a3b*", "populace_us_2024.h5"),
):
    f = [x for x in glob.glob(S + pat + "/" + fn) if "calib" not in x][0]
    p = pd.read_hdf(f, "person")
    cols = [
        c
        for c in (
            "partnership_income",
            "s_corp_income",
            "partnership_s_corp_income",
            "partnership_se_income",
        )
        if c in p
    ]
    print(tag, cols, {c: round(float(p[c].abs().sum() / 1e9), 1) for c in cols})
    use = [c for c in ("partnership_income", "s_corp_income") if c in p] or [
        "partnership_s_corp_income"
    ]
    a = p.groupby("person_spm_unit_id")[use].sum().sum(axis=1).rename("ps")
    u = pd.read_parquet(
        f"{SP}/run/{tag}_units.parquet",
        columns=[
            "spm_unit_id",
            "kids",
            "w1",
            "spm_unit_net_income__2024",
            "spm_unit_spm_threshold__2024",
            "spm_unit_is_in_spm_poverty__2024",
            "household_support_channel",
            "state",
        ],
    )
    u = u[u.kids > 0].merge(a, left_on="spm_unit_id", right_index=True, how="left")
    u["ps"] = u.ps.fillna(0)
    W = u.w1 * u.kids
    was = u.spm_unit_net_income__2024 < u.spm_unit_spm_threshold__2024
    now = (u.spm_unit_net_income__2024 + u.ps) < u.spm_unit_spm_threshold__2024
    print(
        "  check identity",
        round(
            float((was == u.spm_unit_is_in_spm_poverty__2024.astype(bool)).mean()), 5
        ),
        "share child units with nonzero",
        round(float(np.average(u.ps != 0, weights=W)), 4),
    )
    print(
        "  child rate was",
        round(100 * np.average(was, weights=W), 2),
        "with income added",
        round(100 * np.average(now, weights=W), 2),
    )
    print(
        "  by channel",
        {
            k: (
                round(100 * np.average(was[g.index], weights=W[g.index]), 2),
                round(100 * np.average(now[g.index], weights=W[g.index]), 2),
            )
            for k, g in u.groupby("household_support_channel")
        },
    )
