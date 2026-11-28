"""Dense b5db384 state fit, clustered noise, between-state variance, and the join to the sparse gaps."""

import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
u = pd.read_parquet(f"{SP}/run/dense_units.parquet")
u = u[u.kids > 0].copy()
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
cse = c.moe / 1.645
u["W"] = u.w1 * u.kids
u["W0"] = u.w0 * u.kids
u["clu"] = pd.factorize(
    u.source_year.astype(str) + ":" + u.source_household_id.astype(str)
)[0]
u["P25"] = u["spm_unit_is_in_spm_poverty__2025"].astype(float)
u["P24"] = u["spm_unit_is_in_spm_poverty__2024"].astype(float)
u["E"] = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).astype(float)


def lin(g, W, f):  # noqa: N803
    T = g[W].sum()
    r = (g[W] * g[f]).sum() / T
    z = (g[W] * (g[f] - r) / T).values
    cc = np.bincount(pd.factorize(g.clu)[0], weights=z)
    n = len(cc)
    return 100 * r, 100 * np.sqrt(n / (n - 1) * (cc**2).sum()), n


rows = []
for s, g in u.groupby("state"):
    r, se, n = lin(g, "W", "P25")
    r24, _, _ = lin(g, "W", "P24")
    e, ese, _ = lin(g, "W", "E")
    d, dse, _ = lin(g, "W0", "E")
    k = g.W.sum() ** 2 / (g.W**2).sum()
    kc = g.groupby("clu").W.sum()
    kc = kc.sum() ** 2 / (kc**2).sum()
    rows.append(
        dict(
            state=s,
            model=r,
            se=se,
            pe24=r24,
            E=e,
            E_se=ese,
            D=d,
            D_se=dse,
            clusters=n,
            kish_units=k,
            kish_src=kc,
        )
    )
d = pd.DataFrame(rows).set_index("state").join(c)
d["cse"] = cse
d["gap"] = d.model - d.rate
sp = pd.read_csv(f"{SP}/run/waterfall.csv", index_col=0)
acs = pd.read_csv(f"{SP}/run/acs_stored.csv", index_col=0)
d["sp57"] = sp.model
d["sp57_se"] = sp.se_model_rep
d["acs_stored"] = acs["o1017@2.30.1 2025"]
d.to_csv(f"{SP}/run/dense_fit.csv")
g = d.gap
print(
    "gap mean",
    g.mean().round(2),
    "sd",
    g.std().round(2),
    "MAD",
    g.abs().mean().round(2),
    "corr",
    np.corrcoef(d.model, d.rate)[0, 1].round(3),
    "beyond",
    (g.abs() > d.moe).sum(),
)
print(
    "median se",
    d.se.median().round(2),
    "rms",
    np.sqrt((d.se**2).mean()).round(2),
    "median Kish source",
    d.kish_src.median().round(0),
    "units",
    d.kish_units.median().round(0),
    "min",
    d.kish_src.min().round(0),
    d.kish_src.idxmin(),
)
v = d.se**2 + d.cse**2
w = 1 / v
mu = (w * g).sum() / w.sum()
Q = (w * (g - mu) ** 2).sum()
t2 = max(0, (Q - 50) / (w.sum() - (w**2).sum() / w.sum()))
ws = 1 / (v + t2)
print(
    "DL Q",
    Q.round(1),
    "tau",
    np.sqrt(t2).round(2),
    "mu_re",
    ((ws * g).sum() / ws.sum()).round(2),
)
mc = g - g.mean()
sg = d.sp57 - d.rate
ag = d.acs_stored - d.rate
print(
    "corr of gaps: dense-sparse",
    np.corrcoef(g, sg)[0, 1].round(2),
    "dense-acs",
    np.corrcoef(g, ag)[0, 1].round(2),
    "sparse-acs",
    np.corrcoef(sg, ag)[0, 1].round(2),
)
print(
    "corr of rates: dense-sparse",
    np.corrcoef(d.model, d.sp57)[0, 1].round(2),
    "dense-acs",
    np.corrcoef(d.model, d.acs_stored)[0, 1].round(2),
)
print(
    "E vs pool",
    np.corrcoef(d.E, pd.read_csv(f"{SP}/asec/state_raw.csv", index_col=0).avg_22_24)[
        0, 1
    ].round(3),
)
nat = lambda f, W="W": 100 * (u[W] * u[f]).sum() / u[W].sum()  # noqa: N803
print(
    "national D",
    nat("E", "W0").round(2),
    "E",
    nat("E").round(2),
    "P24",
    nat("P24").round(2),
    "P25",
    nat("P25").round(2),
    "kids m",
    (u.W.sum() / 1e6).round(2),
    "kids0",
    (u.W0.sum() / 1e6).round(2),
)
pd.set_option("display.width", 200)
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
            "MA",
            "LA",
            "MT",
            "MS",
            "HI",
            "AR",
            "NM",
            "MI",
            "CA",
            "TX",
        ],
        [
            "rate",
            "moe",
            "sp57",
            "sp57_se",
            "acs_stored",
            "D",
            "E",
            "pe24",
            "model",
            "se",
            "gap",
        ],
    ]
    .round(1)
    .to_string()
)
