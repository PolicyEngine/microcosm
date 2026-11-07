"""Waterfall and exact Shapley decomposition of the state child SPM gap."""

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
u = pd.read_parquet(f"{SP}/run/{TAG}_units.parquet")
u = u[u.kids > 0].copy()
Y0, Y1 = 2024, 2025
c = lambda n: u[n].fillna(0).values.astype(float)
p = lambda n, y=Y0: u[f"{n}__{y}"].values.astype(float)
# Census's identity, verified to the dollar on all four ASEC files.
census_cash = c("SPM_TOTVAL")
census_inkind = (
    c("SPM_SNAPSUB")
    + c("SPM_CAPHOUSESUB")
    + c("SPM_SCHLUNCH")
    + c("SPM_ENGVAL")
    + c("SPM_WICVAL")
    + c("SPM_BBSUBVAL")
)
census_fed = c("SPM_FEDTAXBC") - c("SPM_EITC") - c("SPM_ACTC")
census_payst = c("SPM_FICA") + c("SPM_STTAX")
census_exp = c("SPM_CAPWKCCXPNS") + c("SPM_MEDXPNS") + c("SPM_CHILDSUPPD")
check = census_cash + census_inkind - census_fed - census_payst - census_exp
print("identity max abs", np.abs(check - c("SPM_RESOURCES")).max())
INKIND = [
    "snap",
    "wic",
    "free_school_meals",
    "reduced_price_school_meals",
    "spm_unit_capped_housing_subsidy",
    "spm_unit_energy_subsidy",
    "acp",
    "ebb",
]


def pe_parts(y):
    inkind = sum(p(n, y) for n in INKIND)
    # spm_unit_benefits holds cash and in-kind together.
    cash = (
        p("spm_unit_market_income", y)
        + p("spm_unit_benefits", y)
        + p("acp", y)
        + p("ebb", y)
        - inkind
    )
    fed = p("spm_unit_federal_tax", y)
    payst = (
        p("spm_unit_payroll_tax", y)
        + p("spm_unit_self_employment_tax", y)
        + p("spm_unit_state_tax", y)
        + p("spm_unit_local_tax", y)
    )
    exp = p("spm_unit_spm_expenses", y)
    return dict(cash=cash, inkind=inkind, fed=fed, payst=payst, exp=exp)


pe24, pe25 = pe_parts(Y0), pe_parts(Y1)
net = pe24["cash"] + pe24["inkind"] - pe24["fed"] - pe24["payst"] - pe24["exp"]
print(
    "PE identity: max abs diff to spm_unit_net_income 2024",
    np.abs(net - p("spm_unit_net_income")).max(),
    "taxes parts vs spm_unit_taxes",
    np.abs(pe24["fed"] + pe24["payst"] - p("spm_unit_taxes")).max(),
)
census = dict(
    cash=census_cash,
    inkind=census_inkind,
    fed=census_fed,
    payst=census_payst,
    exp=census_exp,
)
T_c, T_p = c("SPM_POVTHRESHOLD"), p("spm_unit_spm_threshold")
# National source-year price factor from Census thresholds to PE's 2024 thresholds.
ratio = T_p / T_c
priceyear = pd.Series(ratio).groupby(u.source_year.values).transform("median").values
print(
    "median T_PE/T_Census by source year",
    pd.Series(ratio).groupby(u.source_year.values).median().round(4).to_dict(),
)
T_c_y = T_c * priceyear
FACT = ["priceyear", "threshold_rest", "cash", "inkind", "fed", "payst", "exp"]


def poor(sw):
    T = T_c if not sw["priceyear"] else T_c_y
    if sw["threshold_rest"]:
        T = T_p if sw["priceyear"] else T_p / priceyear
    R = 0.0
    for k in ("cash", "inkind"):
        R = R + (pe24[k] if sw[k] else census[k])
    for k in ("fed", "payst", "exp"):
        R = R - (pe24[k] if sw[k] else census[k])
    return R < T


st = u.state.values
states = sorted(set(st))
W = (u.w1 * u.kids).values


def bystate(flag):
    num = pd.Series(flag * W).groupby(st).sum()
    den = pd.Series(W).groupby(st).sum()
    return 100 * num / den


n = len(FACT)
vals = {}
for bits in itertools.product([0, 1], repeat=n):
    vals[bits] = bystate(poor(dict(zip(FACT, bits))))
out = pd.DataFrame(index=states)
for i, f in enumerate(FACT):
    tot = 0
    for bits in itertools.product([0, 1], repeat=n - 1):
        s = sum(bits)
        w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        a = bits[:i] + (1,) + bits[i:]
        b = bits[:i] + (0,) + bits[i:]
        tot = tot + w * (vals[a] - vals[b])
    out[f] = tot
out["E_census_flag"] = vals[(0,) * n]
out["pe_2024"] = bystate(p("spm_unit_is_in_spm_poverty"))
out["pe_all_swapped"] = vals[(1,) * n]
out["pe_2025"] = bystate(p("spm_unit_is_in_spm_poverty", Y1))
out["year"] = out.pe_2025 - out.pe_2024
print(
    "swap reproduces PE 2024: max diff", (out.pe_2024 - out.pe_all_swapped).abs().max()
)
print(
    "census flag check",
    (bystate(c("SPM_RESOURCES") < c("SPM_POVTHRESHOLD")) - out.E_census_flag)
    .abs()
    .max(),
)
out.to_csv(f"{SP}/run/{TAG}_shapley.csv")
print(out.round(1).to_string())
print(out.drop(columns=["pe_all_swapped"]).mean().round(2).to_string())
nat = {}
Wt = W.sum()
for k, v in {"E": vals[(0,) * n], "pe24": out.pe_2024, "pe25": out.pe_2025}.items():
    wst = pd.Series(W).groupby(st).sum()
    nat[k] = float((v * wst).sum() / wst.sum())
print("national", nat)
