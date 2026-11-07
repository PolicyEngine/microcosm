"""Exact Shapley of Census-flag to PE-2024 by resource component, by state and by support channel, on final weights."""

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
CH = sys.argv[2] if len(sys.argv) > 2 else "all"
u = pd.read_parquet(f"{SP}/run/{TAG}_units.parquet")
u = u[u.kids > 0].copy()
if CH != "all":
    u = u[u.household_support_channel == CH].copy()
c = lambda n: u[n].fillna(0).values.astype(float) if n in u else np.zeros(len(u))
p = lambda n, y=2024: u[f"{n}__{y}"].values.astype(float)
Cn = dict(
    cash=c("SPM_TOTVAL"),
    snap=c("SPM_SNAPSUB"),
    housing=c("SPM_CAPHOUSESUB"),
    otherkind=c("SPM_SCHLUNCH") + c("SPM_ENGVAL") + c("SPM_WICVAL") + c("SPM_BBSUBVAL"),
    fed=-(c("SPM_FEDTAXBC") - c("SPM_EITC") - c("SPM_ACTC")),
    payst=-(c("SPM_FICA") + c("SPM_STTAX")),
    exp=-(c("SPM_CAPWKCCXPNS") + c("SPM_MEDXPNS") + c("SPM_CHILDSUPPD")),
)
snap = p("snap")
house = p("spm_unit_capped_housing_subsidy")
ok = (
    p("wic")
    + p("free_school_meals")
    + p("reduced_price_school_meals")
    + p("spm_unit_energy_subsidy")
    + p("acp")
    + p("ebb")
)
Pe = dict(
    cash=p("spm_unit_market_income")
    + p("spm_unit_benefits")
    + p("acp")
    + p("ebb")
    - snap
    - house
    - ok,
    snap=snap,
    housing=house,
    otherkind=ok,
    fed=-p("spm_unit_federal_tax"),
    payst=-(
        p("spm_unit_payroll_tax")
        + p("spm_unit_self_employment_tax")
        + p("spm_unit_state_tax")
        + p("spm_unit_local_tax")
    ),
    exp=-p("spm_unit_spm_expenses"),
)
print(
    "identity",
    np.abs(sum(Cn.values()) - c("SPM_RESOURCES")).max(),
    np.abs(sum(Pe.values()) - p("spm_unit_net_income")).max(),
)
Tc = c("SPM_POVTHRESHOLD")
Tp = p("spm_unit_spm_threshold")
pf = pd.Series(Tp / Tc).groupby(u.source_year.values).transform("median").values
FACT = ["priceyear", "threshold_rest"] + list(Cn)
W = (u.w1 * u.kids).values
states = sorted(set(u.state))
si = pd.Categorical(u.state, categories=states).codes
den = np.bincount(si, W, minlength=len(states))


def val(bits):
    sw = dict(zip(FACT, bits))
    T = (
        (Tp if sw["threshold_rest"] else Tp / pf)
        if sw["priceyear"]
        else (Tp / pf if sw["threshold_rest"] else Tc)
    )
    T = (
        Tp
        if (sw["priceyear"] and sw["threshold_rest"])
        else (Tc * pf if sw["priceyear"] else (Tp / pf if sw["threshold_rest"] else Tc))
    )
    R = sum((Pe[k] if sw[k] else Cn[k]) for k in Cn)
    f = (R < T).astype(float) * W
    return np.append(
        np.bincount(si, f, minlength=len(states)) / den * 100, f.sum() / W.sum() * 100
    )


n = len(FACT)
V = {b: val(b) for b in itertools.product([0, 1], repeat=n)}
out = {}
for i, f in enumerate(FACT):
    tot = 0
    for bits in itertools.product([0, 1], repeat=n - 1):
        s = sum(bits)
        w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        tot = tot + w * (V[bits[:i] + (1,) + bits[i:]] - V[bits[:i] + (0,) + bits[i:]])
    out[f] = tot
o = pd.DataFrame(out, index=states + ["US"])
o.insert(0, "E", V[(0,) * n])
o["pe24"] = V[(1,) * n]
o["n"] = list(pd.Series(si).value_counts().sort_index().values) + [len(u)]
o.to_csv(f"{SP}/run/{TAG}_shapley9_{CH}.csv")
pd.set_option("display.width", 200)
print(
    CH,
    o.loc[
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
            "CA",
        ]
    ]
    .round(1)
    .to_string(),
)
print("mean", o.drop(index="US").mean().round(2).to_dict())
