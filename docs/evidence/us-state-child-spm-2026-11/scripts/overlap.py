"""Sample overlap between the ASEC pool and Table 18, and state misses that persist.

The pool holds income years 2022 to 2024 and Table 18 averages 2023 to 2025, so the two share
households. Three checks follow: raw ASEC shows the overlap, a refit subtracts the covariance
from shared clusters, and a benchmark built from later years shares no household with 2022 records.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
)
cse = c.moe / 1.645
FIPS = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}
a = pd.read_parquet(
    f"{SP}/asec/raw_asec_2022_2025.parquet",
    columns=[
        "A_AGE",
        "MARSUPWT",
        "SPM_RESOURCES",
        "SPM_POVTHRESHOLD",
        "GESTFIPS",
        "income_year",
    ],
)
k = a[a.A_AGE < 18].copy()
k["pov"] = (k.SPM_RESOURCES < k.SPM_POVTHRESHOLD).astype(float)
k["state"] = k.GESTFIPS.astype(int).map(FIPS)
raw = (
    (k.pov * k.MARSUPWT).groupby([k.state, k.income_year]).sum().unstack()
    / k.MARSUPWT.groupby([k.state, k.income_year]).sum().unstack()
    * 100
)
raw = raw.reindex(c.index)
print("raw ASEC against Table 18")
for lab, ys in (
    ("2023-25, Table 18's window", [2023, 2024, 2025]),
    ("2022-24, the pool's window", [2022, 2023, 2024]),
):
    m = raw[ys].mean(axis=1)
    g = m - c.rate
    print(
        f"  {lab}: corr {m.corr(c.rate):.3f}, gap sd {g.std():.2f}, mean abs gap {g.abs().mean():.2f}"
    )
print(
    f"  2022 against 2025, no shared income year: corr {raw[2022].corr(raw[2025]):.2f}, difference sd {(raw[2022] - raw[2025]).std():.2f}"
)


def load(tag):
    u = pd.read_parquet(
        f"{SP}/run/{tag}_units.parquet",
        columns=[
            "kids",
            "w0",
            "w1",
            "state",
            "source_year",
            "source_household_id",
            "household_support_channel",
            "SPM_RESOURCES",
            "SPM_POVTHRESHOLD",
            "spm_unit_is_in_spm_poverty__2025",
        ],
    )
    u = u[u.kids > 0].copy()
    u["y"] = u["spm_unit_is_in_spm_poverty__2025"].astype(float)
    u["E"] = (u.SPM_RESOURCES.fillna(0) < u.SPM_POVTHRESHOLD.fillna(0)).astype(float)
    u["clu"] = u.source_year.astype(str) + ":" + u.source_household_id.astype(str)
    u["ch"] = np.where(
        u.household_support_channel.astype(str).str.contains("puf"), "puf", "asec"
    )
    u["W0"] = u.w0 * u.kids
    u["W1"] = u.w1 * u.kids
    return u


def influence(x, W, v):  # noqa: N803
    T = x.groupby("state")[W].transform("sum")
    R = (x[W] * x[v]).groupby(x.state).transform("sum") / T
    return x[W] * (x[v] - R) / T


def fit(gap, v):
    wt = 1 / v
    m = (wt * gap).sum() / wt.sum()
    Q = (wt * (gap - m) ** 2).sum()
    t2 = max(0, (Q - 50) / (wt.sum() - (wt**2).sum() / wt.sum()))

    def Qf(t):  # noqa: N802
        w = 1 / (v + t)
        m_ = (w * gap).sum() / w.sum()
        return (w * (gap - m_) ** 2).sum()

    bound = lambda p: (
        0.0
        if Qf(0) < stats.chi2.ppf(p, 50)
        else optimize.brentq(lambda t: Qf(t) - stats.chi2.ppf(p, 50), 0, 400)
    )
    return Q, np.sqrt(t2), np.sqrt(bound(0.975)), np.sqrt(bound(0.025))


dense = load("dense")
# Dense ASEC rows at initial weights stand in for Census's estimator on the two shared income years.
cen = dense[(dense.ch == "asec") & dense.source_year.isin([2023, 2024])]
zc = (
    pd.concat(
        [
            pd.DataFrame(
                {"clu": g.clu, "state": g.state, "zc": influence(g, "W0", "E") / 3}
            )
            for _, g in cen.groupby("source_year")
        ]
    )
    .groupby(["state", "clu"])
    .zc.sum()
)
rows = []
covs = {}
for tag, u in (("dense", dense), ("default", load("default"))):
    for W, wlab in (("W0", "initial"), ("W1", "published")):
        for ch in ("all", "asec", "puf"):
            x = u if ch == "all" else u[u.ch == ch]
            zm = influence(x, W, "y").groupby([x.state, x.clu]).sum().rename("zm")
            j = pd.concat([zm, zc], axis=1).dropna()
            cov = (j.zm * j.zc).groupby(level=0).sum() * 1e4
            covs[(tag, wlab, ch)] = cov
            sem = np.sqrt((zm**2).groupby(level=0).sum()) * 100
            gap = (x[W] * x.y).groupby(x.state).sum() / x[W].groupby(
                x.state
            ).sum() * 100 - c.rate
            v0 = sem**2 + cse**2
            v1 = (v0 - 2 * cov).clip(lower=0.05)
            q0 = fit(gap, v0)
            q1 = fit(gap, v1)
            rows.append(
                dict(
                    file=tag,
                    weights=wlab,
                    rows=ch,
                    mean_cov=cov.mean(),
                    median_rho=(cov / (sem * cse)).median(),
                    Q=q0[0],
                    tau=q0[1],
                    tau_lo=q0[2],
                    tau_hi=q0[3],
                    Q_adj=q1[0],
                    tau_adj=q1[1],
                    tau_adj_lo=q1[2],
                    tau_adj_hi=q1[3],
                )
            )
f = pd.DataFrame(rows)
f.to_csv(f"{SP}/run/overlap_fits.csv", index=False)
print(f.round(2).to_string(index=False))

# Income years 2024 and 2025 share no household with income year 2022: the CPS keeps a household for
# one month in consecutive Marches at most, so a two-year separation leaves no rotation overlap.
bench = raw[[2024, 2025]].mean(axis=1)
bse = cse * np.sqrt(3 / 2)
for y in (2022, 2023, 2024):
    x = dense[(dense.source_year == y) & (dense.ch == "asec")]
    m = (x.W0 * x.E).groupby(x.state).sum() / x.W0.groupby(x.state).sum() * 100
    print(
        f"Census's flag on ASEC rows at initial weights against raw ASEC, {y}: corr {m.corr(raw[y]):.3f}"
    )
print(f"benchmark 2024-25 against Table 18: corr {bench.corr(c.rate):.3f}")
rows = []
for lab, x in (
    ("all", dense[dense.source_year == 2022]),
    ("asec", dense[(dense.source_year == 2022) & (dense.ch == "asec")]),
    ("puf", dense[(dense.source_year == 2022) & (dense.ch == "puf")]),
):
    for W, wlab in (("W0", "initial"), ("W1", "published")):
        m = (x[W] * x.y).groupby(x.state).sum() / x[W].groupby(x.state).sum() * 100
        se = (
            np.sqrt(
                (influence(x, W, "y").groupby([x.state, x.clu]).sum() ** 2)
                .groupby(level=0)
                .sum()
            )
            * 100
        )
        g = m - bench
        rows.append(
            dict(
                rows=lab,
                weights=wlab,
                records=len(x),
                mean_gap=g.mean(),
                sd=g.std(),
                mad=g.abs().mean(),
                mad_demeaned=(g - g.mean()).abs().mean(),
                corr=m.corr(bench),
                median_se=se.median(),
                beyond_2se=int((g.abs() > 1.96 * np.sqrt(se**2 + bse**2)).sum()),
            )
        )
b = pd.DataFrame(rows)
b.to_csv(f"{SP}/run/overlap_benchmark.csv", index=False)
print(b.round(2).to_string(index=False))

s = pd.read_csv(f"{SP}/run/initial_fit_states.csv", index_col=0)
e = pd.read_csv(f"{SP}/run/acs_eval.csv", index_col=0)
t = pd.DataFrame(
    {
        "dense_initial": s.dense_W0,
        "dense_initial_se": s.dense_W0_se,
        "acs_rows": e.acs_only - e.census,
        "acs_rows_se": e.acs_only_se,
        "census_se": cse,
    }
)
t["z_dense"] = t.dense_initial / np.sqrt(t.dense_initial_se**2 + t.census_se**2)
t["z_acs"] = t.acs_rows / np.sqrt(t.acs_rows_se**2 + t.census_se**2)
t.to_csv(f"{SP}/run/state_persistence.csv")
# ACS rows share no household with Census, so the gaps' sampling covariance equals Var(Census) less Cov(dense, Census).
print(
    f"gap corr dense initial, ACS rows {t.dense_initial.corr(t.acs_rows):.2f}; p {stats.pearsonr(t.dense_initial, t.acs_rows)[1]:.3f}"
)
print(
    f"  Census noise alone would give about {(t.census_se**2 - covs[('dense', 'initial', 'all')]).mean() / (t.dense_initial.std() * t.acs_rows.std()):.2f}"
)
print(f"gap corr dense published, ACS rows {s.dense_W1.corr(t.acs_rows):.2f}")
print(
    f"  Census noise alone would give about {(t.census_se**2 - covs[('dense', 'published', 'all')]).mean() / (s.dense_W1.std() * t.acs_rows.std()):.2f}"
)
rng = np.random.default_rng(0)
same = lambda z1, z2, th: (
    (np.sign(z1) == np.sign(z2)) & (np.abs(z1) > th) & (np.abs(z2) > th)
).sum()
for th in (1.28, 1.645, 1.96):
    obs = same(t.z_dense.values, t.z_acs.values, th)
    sim = np.array(
        [
            same(t.z_dense.values, t.z_acs.values[rng.permutation(51)], th)
            for _ in range(20000)
        ]
    )
    pick = t.index[
        (np.sign(t.z_dense) == np.sign(t.z_acs))
        & (t.z_dense.abs() > th)
        & (t.z_acs.abs() > th)
    ].tolist()
    print(
        f"|z|>{th}: {obs} states {pick}, permutation mean {sim.mean():.2f}, p {(sim >= obs).mean():.3f}"
    )
t["joint_z"] = np.where(
    np.sign(t.z_dense) == np.sign(t.z_acs),
    np.minimum(t.z_dense.abs(), t.z_acs.abs()),
    0,
)
t.to_csv(f"{SP}/run/state_persistence.csv")
print(
    t.sort_values("joint_z", ascending=False)
    .head(8)[["dense_initial", "acs_rows", "z_dense", "z_acs", "joint_z"]]
    .round(2)
    .to_string()
)
