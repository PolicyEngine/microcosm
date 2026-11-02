"""Aged pooled incomes and the 2021 CTC reform on the default file, from national simulations, against the state cuts."""

import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
u = pd.read_parquet(
    f"{SP}/run/default_units.parquet",
    columns=[
        "spm_unit_id",
        "kids",
        "w1",
        "state",
        "household_support_channel",
        "source_household_id",
        "spm_unit_is_in_spm_poverty__2025",
        "spm_unit_is_in_spm_poverty__2024",
    ],
)
Y = 2025
b = pd.read_parquet(f"{SP}/run/cf_base.parquet")
a = pd.read_parquet(f"{SP}/run/cf_aged.parquet")
r = pd.read_parquet(f"{SP}/run/cf_reform.parquet")
print(list(b.columns)[:8], len(b), len(a), len(r))
d = u[u.kids > 0].copy()
for tag, t in (("base", b), ("aged", a), ("reform", r)):
    d = d.merge(
        t[["spm_unit_id", f"spm_unit_is_in_spm_poverty__{Y}"]].rename(
            columns={f"spm_unit_is_in_spm_poverty__{Y}": f"p_{tag}"}
        ),
        on="spm_unit_id",
    )
d["W"] = d.w1 * d.kids
d["cluster"] = (
    d.household_support_channel.astype(str).str[:4]
    + ":"
    + d.source_household_id.astype(str)
)
print(
    "state cuts vs national base: units differing",
    int((d.p_base != d[f"spm_unit_is_in_spm_poverty__{Y}"]).sum()),
    "of",
    len(d),
)
nat = lambda v: 100 * np.average(d[v], weights=d.W)
print(
    "national base",
    round(nat("p_base"), 2),
    "aged",
    round(nat("p_aged"), 2),
    "reform",
    round(nat("p_reform"), 2),
)
rows = []
for st, g in d.groupby("state"):
    W = g.W
    T = W.sum()
    rate = lambda v: 100 * (W * g[v]).sum() / T

    def se(f):
        rr = (W * f).sum() / T
        z = W * (f - rr) / T
        cc = z.groupby(g.cluster).sum().values
        n = len(cc)
        return 100 * np.sqrt(n / (n - 1) * (cc**2).sum())

    rows.append(
        dict(
            state=st,
            census=c.rate[st],
            base=rate("p_base"),
            aged=rate("p_aged"),
            reform=rate("p_reform"),
            aged_d=rate("p_aged") - rate("p_base"),
            aged_se=se(g.p_aged - g.p_base),
            ref_d=rate("p_reform") - rate("p_base"),
            ref_se=se(g.p_reform - g.p_base),
        )
    )
t = pd.DataFrame(rows).set_index("state")
t.to_csv(f"{SP}/run/cf_state.csv")
for v in ("base", "aged"):
    e = t[v] - t.census
    print(
        v,
        "mean gap",
        round(e.mean(), 2),
        "MAD",
        round(e.abs().mean(), 2),
        "sd",
        round(e.std(), 2),
        "corr",
        round(np.corrcoef(t[v], t.census)[0, 1], 3),
        "beyond",
        int((e.abs() > c.moe.reindex(t.index)).sum()),
    )
print(
    "aged step mean",
    round(t.aged_d.mean(), 2),
    "min",
    round(t.aged_d.min(), 2),
    "max",
    round(t.aged_d.max(), 2),
    "median se",
    round(t.aged_se.median(), 2),
)
print(
    t.loc[
        ["DC", "DE", "TN", "MD", "FL", "LA", "MT", "MA", "HI"],
        ["base", "aged", "aged_d", "aged_se"],
    ]
    .round(2)
    .to_string()
)
print(
    "reform delta: national",
    round(nat("p_reform") - nat("p_base"), 2),
    "state median",
    round(t.ref_d.median(), 2),
    "min",
    round(t.ref_d.min(), 2),
    "max",
    round(t.ref_d.max(), 2),
    "median se",
    round(t.ref_se.median(), 2),
    "p90 se",
    round(t.ref_se.quantile(0.9), 2),
)
print(
    t.loc[
        ["DC", "DE", "TN", "MD", "FL", "LA", "MT", "MA", "HI", "CA", "TX", "NY"],
        ["ref_d", "ref_se"],
    ]
    .round(2)
    .T.to_string()
)
