"""Between-state variance, noise-only null, and independent source-year and channel splits."""

import json
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
n = pd.read_csv(f"{SP}/run/{TAG}_noise_raw.csv", index_col=0)
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
d = n.join(c)
d["cse"] = d.moe / 1.645
d["gap"] = d.model - d.rate
d["v"] = d.se**2 + d.cse**2


def dl(y, v):
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum()
    k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mus = (ws * y).sum() / ws.sum()
    return dict(
        mu_fixed=mu,
        Q=Q,
        df=k - 1,
        tau2=tau2,
        tau=np.sqrt(tau2),
        mu_re=mus,
        se_mu=np.sqrt(1 / ws.sum()),
        I2=max(0, (Q - (k - 1)) / Q),
    )


print(
    "gap sd",
    d.gap.std(),
    "mean",
    d.gap.mean(),
    "MAD",
    d.gap.abs().mean(),
    "corr",
    np.corrcoef(d.model, d.rate)[0, 1],
    "beyond moe",
    (d.gap.abs() > d.moe).sum(),
)
print(
    "mean se_model",
    d.se.mean(),
    "rms se_model",
    np.sqrt((d.se**2).mean()),
    "rms se census",
    np.sqrt((d.cse**2).mean()),
)
r = dl(d.gap, d.v)
print("DL model-census", {k: round(v, 3) for k, v in r.items()})
d["se_E"] = d.E_se
g2 = d.E - d.rate
print("DL E-census", {k: round(v, 3) for k, v in dl(g2, d.E_se**2 + d.cse**2).items()})
# Empirical-Bayes posterior of the systematic state gap.
tau2, mu = r["tau2"], r["mu_re"]
B = tau2 / (tau2 + d.v)
d["eb"] = mu + B * (d.gap - mu)
d["eb_sd"] = np.sqrt(B * d.v)
# Noise-only null: true state rates from shrunk Census, one common shift, no state-specific error.
rng = np.random.default_rng(20261101)
cy = d.rate
cv = d.cse**2
cdl = dl(cy, cv)
ct = cdl["tau2"]
Bc = ct / (ct + cv)
truth = cdl["mu_re"] + Bc * (cy - cdl["mu_re"])
post_sd = np.sqrt(Bc * cv)
S = 20000
for label, shift, tau_extra in [
    ("no shift, noise only", 0, 0),
    ("common shift, noise only", d.gap.mean(), 0),
    ("common shift + DL tau", d.gap.mean(), np.sqrt(tau2)),
]:
    T = truth.values + post_sd.values * rng.standard_normal((S, 51))
    C = T + d.cse.values * rng.standard_normal((S, 51))
    M = (
        T
        + shift
        + tau_extra * rng.standard_normal((S, 51))
        + d.se.values * rng.standard_normal((S, 51))
    )
    gap = M - C
    mad = np.abs(gap).mean(1)
    mean = gap.mean(1)
    sd = gap.std(1, ddof=1)
    beyond = (np.abs(gap) > 1.645 * d.cse.values).sum(1)
    cor = np.array([np.corrcoef(M[i], C[i])[0, 1] for i in range(4000)])
    q = lambda a: np.round(np.percentile(a, [5, 50, 95]), 2)
    print(
        label,
        "MAD",
        q(mad),
        "mean",
        q(mean),
        "sd",
        q(sd),
        "beyond",
        q(beyond),
        "corr",
        q(cor),
    )
print(
    d.loc[
        [
            "DC",
            "DE",
            "TN",
            "MD",
            "FL",
            "RI",
            "VT",
            "MO",
            "LA",
            "MT",
            "MS",
            "HI",
            "AR",
            "CA",
            "TX",
            "NY",
        ],
        ["model", "rate", "gap", "se", "cse", "eb", "eb_sd"],
    ]
    .round(1)
    .to_string()
)
d.to_csv(f"{SP}/run/{TAG}_noise.csv")
# Independent subsamples
u = pd.read_parquet(f"{SP}/run/{TAG}_units.parquet")
u = u[u.kids > 0].copy()
u["W"] = u.w1 * u.kids
u["poor"] = u["spm_unit_is_in_spm_poverty__2025"].astype(float)
u["p24"] = u["spm_unit_is_in_spm_poverty__2024"].astype(float)
u["clu"] = u.source_year.astype(str) + ":" + u.source_household_id.astype(str)


def rate(x, col="poor"):
    return 100 * (x.W * x[col]).groupby(x.state).sum() / x.W.groupby(x.state).sum()


def se(x, col="poor"):
    out = {}
    for st, g in x.groupby("state"):
        r = (g.W * g[col]).sum() / g.W.sum()
        z = g.W * (g[col] - r) / g.W.sum()
        c_ = z.groupby(g.clu).sum().values
        k = len(c_)
        out[st] = 100 * np.sqrt(k / (k - 1) * (c_**2).sum()) if k > 1 else np.nan
    return pd.Series(out)


cen = d.rate
by = {}
for y, x in u.groupby("source_year"):
    by[y] = rate(x)
    by[f"se{y}"] = se(x)
    by[f"W{y}"] = x.W.groupby(x.state).sum() / u.W.groupby(u.state).sum()
Y = pd.DataFrame(by)
print(
    "weight share by source year mean",
    Y[[f"W{y}" for y in (2022, 2023, 2024)]].mean().round(3).to_dict(),
)
a = Y[[2022, 2023, 2024]]
print(
    "corr of state rates across independent source years\n",
    a.corr().round(2).to_string(),
)
print(
    "corr of state gaps across source years\n",
    a.sub(cen, axis=0).corr().round(2).to_string(),
)
for y in (2022, 2023, 2024):
    print(
        y,
        "corr with census",
        round(np.corrcoef(a[y], cen)[0, 1], 2),
        "mean gap",
        round((a[y] - cen).mean(), 2),
        "MAD",
        round((a[y] - cen).abs().mean(), 2),
        "median se",
        round(Y[f"se{y}"].median(), 2),
    )
m = a.mean(axis=1)
print(
    "simple mean of 3 yrs corr",
    round(np.corrcoef(m, cen)[0, 1], 2),
    "MAD",
    round((m - cen).abs().mean(), 2),
)
print(
    pd.concat(
        [
            a.round(1),
            cen.rename("census"),
            Y[[f"se{y}" for y in (2022, 2023, 2024)]].round(1),
        ],
        axis=1,
    )
    .loc[["DC", "DE", "TN", "MD", "FL", "RI", "VT", "MO", "LA", "MT", "MS", "HI", "AR"]]
    .to_string()
)
ch = {}
for k, x in u.groupby("household_support_channel"):
    ch[k] = rate(x)
    ch[f"se_{k}"] = se(x)
    ch[f"W_{k}"] = x.W.groupby(x.state).sum() / u.W.groupby(u.state).sum()
C = pd.DataFrame(ch)
print("channel corr", C[["asec", "puf_tax_detail"]].corr().round(2).to_string())
for k in ("asec", "puf_tax_detail"):
    print(
        k,
        "corr with census",
        round(np.corrcoef(C[k], cen)[0, 1], 2),
        "mean gap",
        round((C[k] - cen).mean(), 2),
        "MAD",
        round((C[k] - cen).abs().mean(), 2),
        "median se",
        round(C[f"se_{k}"].median(), 2),
        "share",
        round(C[f"W_{k}"].mean(), 3),
    )
print(
    pd.concat([C.round(1), cen], axis=1)
    .loc[["DC", "DE", "TN", "MD", "FL", "RI", "VT", "MO", "LA", "MT", "MS", "HI", "AR"]]
    .to_string()
)
