"""Paired record-level comparison of PE and Census resource parts by state. Pairing cancels most sampling noise."""

import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
u = pd.read_parquet(f"{SP}/run/{TAG}_units.parquet")
u = u[u.kids > 0].copy()
u = u[
    u.household_support_channel == "asec"
].copy()  # Census flags and PE inputs share a source only on asec rows
W = (u.w1 * u.kids).values
c = lambda n: u[n].fillna(0).values.astype(float)
p = lambda n: u[f"{n}__2024"].values.astype(float)
low = c("SPM_RESOURCES") < 2 * c("SPM_POVTHRESHOLD")
d = pd.DataFrame(
    {
        "state": u.state.values,
        "W": W,
        "low": low,
        "snap_c": c("SPM_SNAPSUB"),
        "snap_p": p("snap"),
        "house_c": c("SPM_CAPHOUSESUB"),
        "house_p": p("spm_unit_capped_housing_subsidy"),
        "lunch_c": c("SPM_SCHLUNCH"),
        "lunch_p": p("free_school_meals") + p("reduced_price_school_meals"),
        "wic_c": c("SPM_WICVAL"),
        "wic_p": p("wic"),
        "eng_c": c("SPM_ENGVAL"),
        "eng_p": p("spm_unit_energy_subsidy"),
        "T_c": c("SPM_POVTHRESHOLD"),
        "T_p": p("spm_unit_spm_threshold"),
        "geo_c": c("SPM_GEOADJ"),
        "geo_p": p("spm_unit_geographic_adjustment"),
        "exp_c": c("SPM_CAPWKCCXPNS") + c("SPM_MEDXPNS") + c("SPM_CHILDSUPPD"),
        "exp_p": p("spm_unit_spm_expenses"),
        "eitc_c": c("SPM_EITC"),
        "eitc_p": p("eitc"),
        "actc_c": c("SPM_ACTC"),
        "actc_p": p("refundable_ctc"),
        "sttax_c": c("SPM_STTAX"),
        "sttax_p": p("spm_unit_state_tax"),
        "cash_c": c("SPM_TOTVAL"),
        "cash_p": p("spm_unit_market_income"),
    }
)


def wm(g, n):
    return np.average(g[n], weights=g.W)


out = []
for s, g in list(d.groupby("state")) + [("US", d)]:
    l = g[g.low]  # noqa: E741
    r = dict(state=s, n=len(g))
    for k in ("snap", "house", "lunch", "wic", "eng", "eitc", "actc", "sttax", "exp"):
        r[k + "_c"] = wm(l, k + "_c")
        r[k + "_p"] = wm(l, k + "_p")
    r["geo_ratio"] = np.exp(
        np.average(np.log(g.geo_p / g.geo_c.replace(0, np.nan)).fillna(0), weights=g.W)
    )
    out.append(r)
o = pd.DataFrame(out).set_index("state")
pd.set_option("display.width", 230)
show = o[
    [
        "n",
        "snap_c",
        "snap_p",
        "house_c",
        "house_p",
        "lunch_c",
        "lunch_p",
        "eitc_c",
        "eitc_p",
        "actc_c",
        "actc_p",
        "sttax_c",
        "sttax_p",
        "exp_c",
        "exp_p",
        "geo_ratio",
    ]
]
print(
    show.loc[
        [
            "US",
            "LA",
            "MS",
            "AR",
            "HI",
            "MT",
            "DE",
            "MD",
            "MA",
            "MO",
            "TN",
            "IL",
            "DC",
            "FL",
        ]
    ]
    .round(0)
    .assign(geo_ratio=show.geo_ratio.round(3))
    .to_string()
)
o.to_csv(f"{SP}/run/{TAG}_paired.csv")
