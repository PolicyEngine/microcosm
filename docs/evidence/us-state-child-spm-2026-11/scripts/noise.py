"""Sampling noise in state child SPM rates: clustered linearization, cluster bootstrap, DL between-state variance."""

import json
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
YEAR = 2025
u = pd.read_parquet(f"{SP}/run/{TAG}_units.parquet")
u = u[u.kids > 0].copy()
u["W"] = u.w1 * u.kids
u["poor"] = u[f"spm_unit_is_in_spm_poverty__{YEAR}"].astype(float)
u["epoor"] = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).astype(float)
u["clu"] = u.source_year.astype(str) + ":" + u.source_household_id.astype(str)
census = json.load(open(f"{SP}/census_child_spm.json"))
census = census.get("states", census) if isinstance(census, dict) else census
rows = []
rng = np.random.default_rng(20261101)


def lin(g, col):
    r = (g.W * g[col]).sum() / g.W.sum()
    z = g.W * (g[col] - r) / g.W.sum()
    c = z.groupby(g.clu).sum().values
    n = len(c)
    return r, np.sqrt(n / (n - 1) * (c**2).sum()), n


def boot(g, col, B=400):  # noqa: N803
    a = g.groupby("clu").agg(
        wp=("W", lambda s: (s * g.loc[s.index, col]).sum()), w=("W", "sum")
    )
    wp, w = a.wp.values, a.w.values
    n = len(w)
    idx = rng.integers(0, n, size=(B, n))
    return np.std(wp[idx].sum(1) / w[idx].sum(1), ddof=1)


for st, g in u.groupby("state"):
    r, se, n = lin(g, "poor")
    re, see, _ = lin(g, "epoor")
    w = g.w1.values
    hh = g.groupby("household_id" if "household_id" in g else "clu").w1.first().values
    kish = hh.sum() ** 2 / (hh**2).sum()
    wc = g.W.values
    kc = wc.sum() ** 2 / (wc**2).sum()
    rows.append(
        dict(
            state=st,
            model=100 * r,
            se=100 * se,
            clusters=n,
            boot=100 * boot(g, "poor"),
            E=100 * re,
            E_se=100 * see,
            kish_units=kish,
            kish_child=kc,
            se_kish=100 * np.sqrt(r * (1 - r) / kc),
            top5_share=np.sort(wc)[-5:].sum() / wc.sum(),
            max_share=wc.max() / wc.sum(),
        )
    )
d = pd.DataFrame(rows).set_index("state")
d.to_csv(f"{SP}/run/{TAG}_noise_raw.csv")
print(d.round(2).to_string())
print(d.median(numeric_only=True).round(2).to_string())
