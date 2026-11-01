"""Child-weighted Kish effective sample by state, clustered on source household, on each file."""

import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
for tag in ("default", "dense"):
    u = pd.read_parquet(
        f"{SP}/run/{tag}_units.parquet",
        columns=[
            "kids",
            "w1",
            "state",
            "household_support_channel",
            "source_household_id",
        ],
    )
    u = u[u.kids > 0].copy()
    u["W"] = u.w1 * u.kids
    u["src"] = (
        u.household_support_channel.astype(str).str[:4]
        + ":"
        + u.source_household_id.astype(str)
    )
    k = {}
    for s, g in u.groupby("state"):
        v = g.W.groupby(g.src).sum()
        k[s] = v.sum() ** 2 / (v**2).sum()
    k = pd.Series(k)
    print(
        tag,
        "median",
        round(k.median()),
        "min",
        k.idxmin(),
        round(k.min()),
        "max",
        k.idxmax(),
        round(k.max()),
    )
