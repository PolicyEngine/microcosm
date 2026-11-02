"""Why Florida's 2021 CTC effect moves with the rent fix: which source households carry the change."""

import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))


def load(tag, fips):
    u = pd.read_parquet(f"{SP}/run/{tag}/{fips:02d}.parquet")
    f = f"{SP}/run/{tag}_r/{fips:02d}.parquet"
    if not os.path.exists(f):
        return u
    return u.drop(columns=[c for c in ("poor_r",) if c in u]).merge(
        pd.read_parquet(f), on="spm_unit_id"
    )


h = pd.read_parquet(f"{SP}/run/acs_hh.parquet")
uh = pd.read_parquet(f"{SP}/run/acs_unit_hh.parquet").drop_duplicates(
    "person_spm_unit_id"
)
for st, fips in (("FL", 12), ("DE", 10), ("MO", 29)):
    a = load("acs", fips)
    b = load("acsrent", fips)
    m = a[["spm_unit_id", "kids", "poor", "poor_r"]].merge(
        b[["spm_unit_id", "poor", "poor_r"]],
        on="spm_unit_id",
        suffixes=("_pub", "_rent"),
    )
    m = (
        m[m.kids > 0]
        .merge(uh, left_on="spm_unit_id", right_on="person_spm_unit_id")
        .merge(h, left_on="person_household_id", right_on="household_id")
    )
    m["W"] = m.household_weight * m.kids
    T = m.W.sum()
    m["channel"] = np.where(
        m.household_support_channel.astype(str) == "acs_2024_1yr", "acs", "donor"
    )
    m["src"] = m.channel + ":" + m.household_source_id.astype(str)
    m["d_base"] = 100 * m.W * (m.poor_rent - m.poor_pub) / T
    m["d_ref"] = 100 * m.W * (m.poor_r_rent - m.poor_r_pub) / T
    m["d_eff"] = m.d_ref - m.d_base
    print(
        st,
        "baseline",
        round(m.d_base.sum(), 2),
        "reform world",
        round(m.d_ref.sum(), 2),
        "effect change",
        round(m.d_eff.sum(), 2),
    )
    print(
        "  by channel",
        m.groupby("channel")[["d_base", "d_ref", "d_eff"]]
        .sum()
        .round(2)
        .to_dict("index"),
    )
    print(
        "  units whose status changes: baseline",
        int((m.poor_rent != m.poor_pub).sum()),
        "reform world",
        int((m.poor_r_rent != m.poor_r_pub).sum()),
    )
    g = m.groupby("src").agg(
        d_ref=("d_ref", "sum"),
        d_base=("d_base", "sum"),
        d_eff=("d_eff", "sum"),
        W=("W", "sum"),
    )
    g["share_child_weight"] = 100 * g.W / T
    top = g.reindex(g.d_eff.abs().sort_values(ascending=False).index).head(6)
    print(top.round(2).to_string())
    c = g.d_eff.abs().sort_values(ascending=False)
    print(
        "  top 1, 5, 10, 25 clusters of gross change",
        [round(100 * c.head(k).sum() / c.sum()) for k in (1, 5, 10, 25)],
        "gross",
        round(c.sum(), 2),
        "n",
        int((c > 0).sum()),
    )
