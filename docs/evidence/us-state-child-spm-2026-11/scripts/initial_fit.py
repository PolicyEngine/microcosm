"""State fit at initial and published weights on the dense and default files, with clustered noise and a random-effects fit."""

import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
cse = c.moe / 1.645


def lin(g, W, f):  # noqa: N803
    T = g[W].sum()
    r = (g[W] * g[f]).sum() / T
    z = (g[W] * (g[f] - r) / T).values
    cc = np.bincount(pd.factorize(g.clu)[0], weights=z)
    n = len(cc)
    return 100 * r, 100 * np.sqrt(n / (n - 1) * (cc**2).sum())


def dl(g, se):
    v = se**2 + cse.reindex(g.index) ** 2
    w = 1 / v
    mu = (w * g).sum() / w.sum()
    Q = (w * (g - mu) ** 2).sum()
    k = len(g)
    t2 = max(0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    return Q, np.sqrt(t2)


rows = []
out = {}
for tag in ("dense", "default"):
    u = pd.read_parquet(f"{SP}/run/{tag}_units.parquet")
    u = u[u.kids > 0].copy()
    u["clu"] = pd.factorize(
        u.source_year.astype(str) + ":" + u.source_household_id.astype(str)
    )[0]
    u["P25"] = u["spm_unit_is_in_spm_poverty__2025"].astype(float)
    u["P24"] = u["spm_unit_is_in_spm_poverty__2024"].astype(float)
    u["E"] = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).astype(float)
    u["W1"] = u.w1 * u.kids
    u["W0"] = u.w0 * u.kids
    u["ch"] = np.where(
        u.household_support_channel.astype(str).str.lower().str.contains("puf"),
        "puf",
        "asec",
    )
    for W in ("W0", "W1"):
        for ch in ("all", "asec", "puf"):
            x = u if ch == "all" else u[u.ch == ch]
            r = {}
            for s, g in x.groupby("state"):
                m, se = lin(g, W, "P25")
                r[s] = (m, se)
            d = pd.DataFrame(r, index=["model", "se"]).T
            g = d.model - c.rate.reindex(d.index)
            Q, t = dl(g, d.se)
            k = x[W].groupby(x.clu).sum()
            nat = lambda f: 100 * (x[W] * x[f]).sum() / x[W].sum()
            rows.append(
                dict(
                    file=tag,
                    weights={"W0": "initial", "W1": "published"}[W],
                    rows=ch,
                    natP25=nat("P25"),
                    natP24=nat("P24"),
                    natE=nat("E"),
                    mean_gap=g.mean(),
                    sd=g.std(),
                    mad=g.abs().mean(),
                    corr=np.corrcoef(d.model, c.rate.reindex(d.index))[0, 1],
                    beyond=int((g.abs() > c.moe.reindex(d.index)).sum()),
                    med_se=d.se.median(),
                    Q=Q,
                    tau=t,
                )
            )
            if ch == "all":
                out[f"{tag}_{W}"] = g
                out[f"{tag}_{W}_se"] = d.se
r = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(r.round(2).to_string())
s = pd.DataFrame(out)
s.to_csv(f"{SP}/run/initial_fit_states.csv")
r.to_csv(f"{SP}/run/initial_fit.csv", index=False)
print(
    s.loc[
        ["DC", "DE", "TN", "MD", "FL", "MA", "LA", "MT", "MS", "HI", "RI", "VT", "MO"],
        ["dense_W0", "dense_W1", "default_W0", "default_W1"],
    ]
    .round(1)
    .to_string()
)
print(
    "corr dense W0 gap vs W1 gap", round(np.corrcoef(s.dense_W0, s.dense_W1)[0, 1], 3)
)
# Independent check: ACS rows share no households with ASEC or PUF rows, so gaps they share point to the model or the benchmark.
e = pd.read_csv(f"{SP}/run/acs_eval.csv", index_col=0)
ag = (e.acs_only - e.census).reindex(s.index)
for tag in ("dense", "default"):
    for W in ("W0", "W1"):
        print(
            "corr with ACS-row gaps",
            tag,
            W,
            round(np.corrcoef(s[f"{tag}_{W}"], ag)[0, 1], 3),
        )
# Calibration step by flag, state means
rows = []
for tag in ("dense", "default"):
    u = pd.read_parquet(f"{SP}/run/{tag}_units.parquet")
    u = u[u.kids > 0].copy()
    u["P25"] = u["spm_unit_is_in_spm_poverty__2025"].astype(float)
    u["P24"] = u["spm_unit_is_in_spm_poverty__2024"].astype(float)
    u["E"] = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).astype(float)
    u["W1"] = u.w1 * u.kids
    u["W0"] = u.w0 * u.kids
    r = lambda W, f: (  # noqa: N803
        100 * (u[W] * u[f]).groupby(u.state).sum() / u[W].groupby(u.state).sum()
    )
    for f in ("E", "P24", "P25"):
        d = r("W1", f) - r("W0", f)
        rows.append(
            dict(
                file=tag,
                flag=f,
                state_mean=d.mean(),
                sd=d.std(),
                mad=d.abs().mean(),
                lo=d.min(),
                hi=d.max(),
                nat0=100 * (u.W0 * u[f]).sum() / u.W0.sum(),
                nat1=100 * (u.W1 * u[f]).sum() / u.W1.sum(),
            )
        )
    k = lambda W: pd.Series(  # noqa: N803
        {
            st: g[W].sum() ** 2 / (g[W] ** 2).sum()
            for st, g in u.assign(
                src=u.source_year.astype(str) + ":" + u.source_household_id.astype(str)
            )
            .groupby(["state", "src"])[["W0", "W1"]]
            .sum()
            .reset_index()
            .groupby("state")
        }
    )
    print(
        tag,
        "median child Kish initial",
        round(k("W0").median()),
        "published",
        round(k("W1").median()),
        "initial min",
        k("W0").idxmin(),
        round(k("W0").min()),
        "max",
        k("W0").idxmax(),
        round(k("W0").max()),
    )
c2 = pd.DataFrame(rows)
print(c2.round(2).to_string())
c2.to_csv(f"{SP}/run/calibration_step.csv", index=False)
