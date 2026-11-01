"""ACS-local state rates at current policyengine-us: clustered SE, channel split, reform delta, and alternative #1078 solves."""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "acs"
B = os.path.expanduser(
    "~/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/runs/"
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
INV = {v: k for k, v in FIPS.items()}
h = pd.read_parquet(f"{SP}/run/acs_hh.parquet")
uh = pd.read_parquet(f"{SP}/run/acs_unit_hh.parquet").drop_duplicates(
    "person_spm_unit_id"
)
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
stored = pd.read_csv(f"{SP}/run/acs_stored.csv", index_col=0)
fs = sorted(glob.glob(f"{SP}/run/{TAG}/*.parquet"))
u = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
u = (
    u[u.kids > 0]
    .merge(uh, left_on="spm_unit_id", right_on="person_spm_unit_id")
    .merge(h, left_on="person_household_id", right_on="household_id")
)
u["state"] = u.state_fips.map(INV)
u["src"] = (
    u.household_support_channel.astype(str).str[:4]
    + ":"
    + u.household_source_id.astype(str)
)


def lin(g, w, f):
    W = g[w] * g.kids
    T = W.sum()
    r = (W * f).sum() / T
    z = W * (f - r) / T
    cc = z.groupby(g.src).sum().values
    n = len(cc)
    return 100 * r, 100 * np.sqrt(n / (n - 1) * (cc**2).sum())


rows = []
for st, g in u.groupby("state"):
    r, se = lin(g, "household_weight", g.poor)
    d = g.poor_r - g.poor
    dr, dse = lin(g, "household_weight", d)
    W = g.household_weight * g.kids
    kish = W.groupby(g.src).sum()
    kish = kish.sum() ** 2 / (kish**2).sum()
    ac = g[g.household_support_channel == "acs_2024_1yr"]
    dn = g[g.household_support_channel != "acs_2024_1yr"]
    ra, sa = lin(ac, "household_weight", ac.poor)
    rd, sd = lin(dn, "household_weight", dn.poor)
    rows.append(
        dict(
            state=st,
            census=c.rate[st],
            moe=c.moe[st],
            acs_local=r,
            se=se,
            kish_src=kish,
            acs_share=W[g.household_support_channel == "acs_2024_1yr"].sum() / W.sum(),
            acs_only=ra,
            acs_only_se=sa,
            donor_only=rd,
            donor_only_se=sd,
            delta=dr,
            delta_se=dse,
            stored_2301=stored["o1017@2.30.1 2025"][st],
        )
    )
d = pd.DataFrame(rows).set_index("state")
d["gap"] = d.acs_local - d.census
print(d.round(2).to_string())
print(
    "median se",
    d.se.median().round(2),
    "median delta se",
    d.delta_se.median().round(2),
    "current vs stored 2.30.1 mean abs",
    (d.acs_local - d.stored_2301).abs().mean().round(2),
)
d.to_csv(f"{SP}/run/{TAG}_eval.csv")
for lab, col in (
    ("all rows", "acs_local"),
    ("ACS rows", "acs_only"),
    ("donor rows", "donor_only"),
):
    e = d[col] - d.census
    print(
        lab,
        "states",
        len(d),
        "mean gap",
        round(e.mean(), 2),
        "sd",
        round(e.std(), 2),
        "MAD",
        round(e.abs().mean(), 2),
        "corr",
        round(np.corrcoef(d[col], d.census)[0, 1], 3),
        "beyond",
        int((e.abs() > d.moe).sum()),
        "median se",
        round(
            d[
                {
                    "acs_local": "se",
                    "acs_only": "acs_only_se",
                    "donor_only": "donor_only_se",
                }[col]
            ].median(),
            2,
        ),
    )
print(
    "reform delta median",
    round(d.delta.median(), 2),
    "min",
    round(d.delta.min(), 2),
    "max",
    round(d.delta.max(), 2),
    "median se",
    round(d.delta_se.median(), 2),
)
# Alternative solves. Stored solves used an earlier frame with the same household ids.
key = pd.read_hdf(
    os.path.dirname(B[:-1]) + "/checkpoint/target_frame_lean.h5", "household"
)[["household_id"]]
pos = pd.Series(np.arange(len(key)), index=key.household_id.values)
ix = pos.reindex(u.household_id.values).values.astype(int)
out = []
names = [
    "dup_release_repro",
    "proj_chi_s050_0.03",
    "soft_chi_s050_0.1",
    "soft_chi_s700_0",
    "soft_chi_s700_0.03",
    "soft_chi_s700_0.1",
    "soft_chi_s900_0",
    "soft_chi_s900_0.03",
    "soft_chi_s900_0.1",
    "soft_chi_s964_0",
    "soft_chi_s964_0.03",
]
st = u.state.values
for n in names:
    p = B + n + "/weights.npz"
    if not os.path.exists(p):
        continue
    w = np.load(p)["weights"]
    u["w"] = w[ix]
    R = []
    for s, g in u.groupby("state"):
        r, se = lin(g, "w", g.poor)
        W = g.w * g.kids
        a = W[g.household_support_channel == "acs_2024_1yr"].sum() / W.sum()
        R.append(dict(state=s, r=r, se=se, a=a))
    R = pd.DataFrame(R).set_index("state")
    g_ = R.r - c.rate.reindex(R.index)
    out.append(
        dict(
            run=n,
            mean_gap=g_.mean(),
            mad=g_.abs().mean(),
            beyond=int((g_.abs() > c.moe.reindex(R.index)).sum()),
            corr=np.corrcoef(R.r, c.rate.reindex(R.index))[0, 1],
            med_se=R.se.median(),
            acs_share=R.a.mean(),
            DE=R.r["DE"],
            DC=R.r["DC"],
            TN=R.r["TN"],
            MD=R.r["MD"],
            LA=R.r["LA"],
            MT=R.r["MT"],
            HI=R.r["HI"],
            MA=R.r["MA"],
        )
    )
o = pd.DataFrame(out).set_index("run")
o.to_csv(f"{SP}/run/{TAG}_solves.csv")
print(o.round(2).to_string())
