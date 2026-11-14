"""State fit and noise under the chi-square re-solves of the Build P support."""

import glob
import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
h = pd.read_hdf(
    S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5", "household"
)[["household_id"]]
pos = pd.Series(np.arange(len(h)), index=h.household_id.values)
u = pd.read_parquet(f"{SP}/run/default_units.parquet")
u = u[u.kids > 0].copy()
ix = pos.reindex(u.household_id.values).values.astype(int)
P = u["spm_unit_is_in_spm_poverty__2025"].values.astype(float)
E = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).values.astype(float)
k = u.kids.values
st = u.state.values
clu = pd.factorize(u.source_year.astype(str) + ":" + u.source_household_id.astype(str))[
    0
]
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
cse = c.moe / 1.645
z = np.load(
    S + "8ab57ffc2ca41d8af631ff62ebbce95e966299f3/populace_us_2024_calibration.npz",
    allow_pickle=True,
)
states = np.unique(st)


def stats(w):
    W = w[ix] * k
    r = []
    se = []
    for s in states:
        m = st == s
        Wm = W[m]
        pm = P[m]
        T = Wm.sum()
        rr = (Wm * pm).sum() / T
        zz = Wm * (pm - rr) / T
        cc = np.bincount(pd.factorize(clu[m])[0], weights=zz)
        n = len(cc)
        r.append(100 * rr)
        se.append(100 * np.sqrt(n / (n - 1) * (cc**2).sum()))
    r = pd.Series(r, index=states)
    se = pd.Series(se, index=states)
    g = r - c.rate
    v = se**2 + cse**2
    wt = 1 / v
    mu = (wt * g).sum() / wt.sum()
    Q = (wt * (g - mu) ** 2).sum()
    t2 = max(0, (Q - 50) / (wt.sum() - (wt**2).sum() / wt.sum()))
    return dict(
        kish=w.sum() ** 2 / (w**2).sum(),
        natE=100 * (W * E).sum() / W.sum(),
        natP25=100 * (W * P).sum() / W.sum(),
        kids_m=W.sum() / 1e6,
        med_se=se.median(),
        mean_gap=g.mean(),
        sd_gap=g.std(),
        mad=g.abs().mean(),
        corr=np.corrcoef(r, c.rate)[0, 1],
        tau=np.sqrt(t2),
    )


runs = {"published": z["household_weight"], "initial": z["initial_household_weight"]}
for f in sorted(glob.glob(f"{SP}/run/chi/*.npz")):
    runs[os.path.basename(f)[:-4]] = np.load(f)["weights"]
T = pd.DataFrame({n: stats(np.asarray(w, float)) for n, w in runs.items()}).T
print(T.round(2).to_string())
T.to_csv(f"{SP}/run/chi_eval.csv")
