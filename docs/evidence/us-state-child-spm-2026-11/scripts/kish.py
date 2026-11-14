"""Child-weighted Kish effective sample by state, clustered on source household, on each file."""

import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
out = {}
for tag in ("default", "dense"):
    u = pd.read_parquet(
        f"{SP}/run/{tag}_units.parquet",
        columns=["kids", "w0", "w1", "state", "source_year", "source_household_id"],
    )
    u = u[u.kids > 0].copy()
    # An ASEC record and its PUF clone share one source household, so they form one cluster.
    u["src"] = u.source_year.astype(str) + ":" + u.source_household_id.astype(str)
    for w, suffix in (("w1", ""), ("w0", "_initial")):
        u["W"] = u[w] * u.kids
        k = {}
        for s, g in u.groupby("state"):
            v = g.W.groupby(g.src).sum()
            k[s] = v.sum() ** 2 / (v**2).sum()
        out[tag + suffix] = pd.Series(k)
out["acs_local"] = pd.read_csv(f"{SP}/run/acs_eval.csv", index_col=0).kish_src
t = pd.DataFrame(out)
t.to_csv(f"{SP}/run/kish.csv")
print(t.median().round(0).to_dict())
print(t.agg(["min", "max"]).round(0).to_dict())
print(t.idxmin().to_dict(), t.idxmax().to_dict())
