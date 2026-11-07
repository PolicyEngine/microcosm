"""Why the rent fix moves a state's 2021 CTC effect: which source households carry the change.

It traces every state whose effect moves more than a point in run/acsrent_reform.csv (acs_reform_cmp.py writes it), largest
move first, and ends with one summary row per state. It prints source ids, channels and shares of child weight, and no incomes.
"""

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
FIPS = {
    "AL": 1,
    "AK": 2,
    "AZ": 4,
    "AR": 5,
    "CA": 6,
    "CO": 8,
    "CT": 9,
    "DE": 10,
    "DC": 11,
    "FL": 12,
    "GA": 13,
    "HI": 15,
    "ID": 16,
    "IL": 17,
    "IN": 18,
    "IA": 19,
    "KS": 20,
    "KY": 21,
    "LA": 22,
    "ME": 23,
    "MD": 24,
    "MA": 25,
    "MI": 26,
    "MN": 27,
    "MS": 28,
    "MO": 29,
    "MT": 30,
    "NE": 31,
    "NV": 32,
    "NH": 33,
    "NJ": 34,
    "NM": 35,
    "NY": 36,
    "NC": 37,
    "ND": 38,
    "OH": 39,
    "OK": 40,
    "OR": 41,
    "PA": 42,
    "RI": 44,
    "SC": 45,
    "SD": 46,
    "TN": 47,
    "TX": 48,
    "UT": 49,
    "VT": 50,
    "VA": 51,
    "WA": 53,
    "WV": 54,
    "WI": 55,
    "WY": 56,
}
rc = pd.read_csv(f"{SP}/run/acsrent_reform.csv")
rc = rc[rc.corrected_minus_published.abs() > 1].sort_values(
    "corrected_minus_published", key=abs, ascending=False
)
rows = []
for st in rc.state:
    fips = FIPS[st]
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
    rows.append(
        {
            "state": st,
            "effect_change": round(m.d_eff.sum(), 2),
            "donor_part": round(m.loc[m.channel == "donor", "d_eff"].sum(), 2),
            "acs_part": round(m.loc[m.channel == "acs", "d_eff"].sum(), 2),
            "gross": round(c.sum(), 2),
            "clusters": int((c > 0).sum()),
            "top1_pct": round(100 * c.head(1).sum() / c.sum()),
            "top10_pct": round(100 * c.head(10).sum() / c.sum()),
            "donor_gross_pct": round(
                100
                * g.loc[g.index.str.startswith("donor"), "d_eff"].abs().sum()
                / c.sum()
            ),
        }
    )
print()
print(pd.DataFrame(rows).to_string(index=False))
