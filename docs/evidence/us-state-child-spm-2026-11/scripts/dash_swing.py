"""How far the stored dashboard audits move each state's 2021 CTC effect between builds, against the sampling error of one build."""

import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 200)
d = {}
for tag, f in (
    ("p24", "state_audit_buildp_v2_2024_raw.json"),
    ("p25", "state_audit_buildp_v2_y25_2025_raw.json"),
    ("o24", "state_audit_o1017_2301_2024_raw.json"),
    ("o25", "state_audit_o1017_2301_y25_2025_raw.json"),
    ("o25_18252", "state_audit_o1017_18252_2025_raw.json"),
    ("o24_221", "state_audit_o1017_221_2024_raw.json"),
):
    x = json.load(open(f"{SP}/dash/{f}"))
    if tag == "p24":
        print(list(x[0].keys())[:20])
    key = [k for k in x[0] if k in ("state", "state_code", "state_abbr")][0]
    d[tag] = pd.Series(
        {r[key]: r.get("ref_ctc2021_child_pov_change_pp_measured") for r in x},
        dtype=float,
    )
t = pd.DataFrame(d)
print(t.isna().sum().to_dict(), len(t))
for a, b in (("p24", "o24"), ("p25", "o25")):
    e = (t[a] - t[b]).dropna()
    print(
        a,
        b,
        "states",
        len(e),
        "mean abs swing",
        round(e.abs().mean(), 2),
        "sd",
        round(e.std(), 2),
        "max",
        round(e.abs().max(), 2),
        e.abs().idxmax(),
        "corr",
        round(np.corrcoef(t.loc[e.index, a], t.loc[e.index, b])[0, 1], 3),
    )
a = pd.read_csv(f"{SP}/run/acs_eval.csv", index_col=0)
print(
    "this run vs stored o25, mean abs",
    round((a.delta - t.o25.reindex(a.index)).abs().mean(), 3),
)
e = (t.p25 - t.o25).reindex(a.index)
print(
    "implied sd of between-build noise per build",
    round(e.std() / np.sqrt(2), 2),
    "median delta se",
    round(a.delta_se.median(), 2),
    "rms delta se",
    round(np.sqrt((a.delta_se**2).mean()), 2),
)
same = (t.p25 - t.o25_18252).dropna()
print(
    "same model 1.825.2, Build P against Build O, 2025: mean abs",
    round(same.abs().mean(), 2),
    "sd",
    round(same.std(), 2),
    "per build",
    round(same.std() / np.sqrt(2), 2),
    "max",
    round(same.abs().max(), 2),
    same.abs().idxmax(),
)
ver = (t.o25 - t.o25_18252).dropna()
print(
    "same file across 1.825.2 and 2.30.1, 2025: mean abs",
    round(ver.abs().mean(), 2),
    "sd",
    round(ver.std(), 2),
    "max",
    round(ver.abs().max(), 2),
    ver.abs().idxmax(),
)
print(t.loc[["AZ", "DC", "NC"]].round(2).to_string())
t.to_csv(f"{SP}/run/dash_swing.csv")
