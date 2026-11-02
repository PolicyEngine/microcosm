"""Sum of steps from Census to the model, by state, on the dense file and the sparse file: means and shares of cross-state variance."""

import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
sr = pd.read_csv(f"{SP}/asec/state_raw.csv", index_col=0)
out = {}
for tag, fit, lab in (  # noqa: B007
    ("dense", "dense_fit", "dense b5db384"),
    ("default", "default_noise", "sparse Build P"),
):
    s = pd.read_csv(f"{SP}/run/{tag}_shapley9_all.csv", index_col=0).drop(index="US")
    if tag == "dense":
        f = pd.read_csv(f"{SP}/run/dense_fit.csv", index_col=0)
        D = f.D
        model = f.model
    else:
        w = pd.read_csv(f"{SP}/run/waterfall.csv", index_col=0)
        D = w.D
        model = w.model
    T = pd.DataFrame(
        {
            "period_and_sample": sr.avg_22_24 - c.rate,
            "selection": D - sr.avg_22_24,
            "calibration": s.E - D,
        }
    )
    for k in (
        "priceyear",
        "threshold_rest",
        "cash",
        "snap",
        "housing",
        "otherkind",
        "fed",
        "payst",
        "exp",
    ):
        T[k] = s[k]
    T["year"] = model - s.pe24
    T["gap"] = model - c.rate
    assert (T.drop(columns="gap").sum(1) - T.gap).abs().max() < 1e-6
    var = T.gap.var()
    R = pd.DataFrame(
        {
            "mean": T.drop(columns="gap").mean(),
            "var_share": [
                np.cov(T[k], T.gap)[0, 1] / var for k in T.columns if k != "gap"
            ],
            "sd": T.drop(columns="gap").std(),
            "corr_with_gap": [
                np.corrcoef(T[k], T.gap)[0, 1] for k in T.columns if k != "gap"
            ],
        }
    )
    print(lab, "gap mean", round(T.gap.mean(), 2), "sd", round(T.gap.std(), 2))
    print(R.round(2).to_string())
    out[tag] = T
    T.to_csv(f"{SP}/run/{tag}_steps.csv")
pd.set_option("display.width", 230)
top = ["DC", "DE", "TN", "MD", "FL", "LA", "MT"]
print(out["dense"].loc[top].round(1).T.to_string())
print(out["default"].loc[top].round(1).T.to_string())
