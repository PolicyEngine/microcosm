"""Weight concentration, clustered SE and gap structure under stored re-solves; initial target misses by family."""

import json
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser("~/reviews/pe30-population-2026-09-27/")
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
h = pd.read_hdf(
    S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5", "household"
)[["household_id", "state_fips"]]
pos = pd.Series(np.arange(len(h)), index=h.household_id.values)
u = pd.read_parquet(f"{SP}/run/default_units.parquet")
u = u[u.kids > 0].copy()
ix = pos.reindex(u.household_id.values).values.astype(int)
P = u["spm_unit_is_in_spm_poverty__2025"].values.astype(float)
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


def dl(y, v):
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum()
    kk = len(y)
    t2 = max(0.0, (Q - (kk - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    return t2, Q


def stats(w):
    W = w[ix] * k
    rows = []  # noqa: F841
    states = np.unique(st)
    r = np.empty(51)
    se = np.empty(51)
    for i, s in enumerate(states):
        m = st == s
        Wm = W[m]
        pm = P[m]
        T = Wm.sum()
        rr = (Wm * pm).sum() / T
        zz = Wm * (pm - rr) / T
        cc = np.bincount(pd.factorize(clu[m])[0], weights=zz)
        n = len(cc)
        r[i] = 100 * rr
        se[i] = 100 * np.sqrt(n / (n - 1) * (cc**2).sum())
    return (
        pd.Series(r, index=states),
        pd.Series(se, index=states),
        w.sum() ** 2 / (w**2).sum(),
    )


runs = {"initial": z["initial_household_weight"], "published": z["household_weight"]}
for n in [
    "R0_full",
    "keep_pep",
    "keep_pep_ctc_only",
    "keep_pep_ctc_actc_count",
    "keep_pep_all_soi",
    "keep_pep_snap_ssi_aca",
    "keep_pep_medicaid",
    "drop_snap",
    "drop_medicaid_chip",
    "drop_soi_return_count",
    "drop_ctc_actc",
    "drop_soi_other",
]:
    runs[n] = np.load(f"{D}out/calib_resolve/{n}.npz")["weights"]
out = []
for n, w in runs.items():
    r, se, kish = stats(np.asarray(w, float))
    g = r - c.rate
    v = se**2 + cse**2
    t2, Q = dl(g, v)
    out.append(
        dict(
            run=n,
            kish=kish,
            med_se=se.median(),
            mean_gap=g.mean(),
            sd_gap=g.std(),
            mad=g.abs().mean(),
            corr=np.corrcoef(r, c.rate)[0, 1],
            tau=np.sqrt(t2),
            Q=Q,
        )
    )
print(pd.DataFrame(out).set_index("run").round(2).to_string())
# Initial misses by family on Build P and dense
sys.path.insert(0, D + "scripts")
diag = json.load(open(D + "rel/calibration_diagnostics.json"))
T = diag["targets"]


def fam(t):
    m = t["metadata"]
    rg = t["registry"]["family"]
    src = m.get("source_variable")
    role = m.get("target_role")
    return f"{rg}:{role or src}"


d = pd.DataFrame(
    {
        "fam": [fam(t) for t in T],
        "t": [t["target"] for t in T],
        "i": [t["initial_estimate"] for t in T],
        "f": [t["final_estimate"] for t in T],
        "geo": [t["metadata"].get("geography_scope") for t in T],
    }
)
d["ri"] = d.i / d.t - 1
d["rf"] = d.f / d.t - 1
g = (
    d[d.t != 0]
    .groupby(["fam", "geo"])
    .agg(
        n=("t", "size"),
        t=("t", "sum"),
        i=("i", "sum"),
        f=("f", "sum"),
        med_ri=("ri", "median"),
        med_rf=("rf", "median"),
    )
)
g["agg_ri"] = g.i / g.t - 1
g["agg_rf"] = g.f / g.t - 1
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
print(
    g[g.n >= 1]
    .sort_values("agg_ri")
    .round(3)[["n", "agg_ri", "agg_rf", "med_ri", "med_rf"]]
    .to_string()
)
