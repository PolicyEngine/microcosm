"""What rent misplacement does: same weights, same everything, rent on the household it was drawn for."""

import glob
import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
c = pd.DataFrame(json.load(open(f"{SP}/census_child_spm.json"))["rows"]).set_index(
    "state"
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
VARS = [
    "spm_unit_is_in_spm_poverty__2024",
    "spm_unit_is_in_spm_poverty__2025",
    "snap__2024",
    "snap__2025",
    "spm_unit_capped_housing_subsidy__2024",
    "spm_unit_net_income__2025",
    "spm_unit_spm_threshold__2025",
]
out = {}
for base, fix in (("dense", "denserent"), ("default", "sparserent")):
    fs = sorted(glob.glob(f"{SP}/run/{fix}/*.parquet"))
    if not fs:
        continue
    f = pd.concat(
        [pd.read_parquet(x, columns=["spm_unit_id"] + VARS) for x in fs],
        ignore_index=True,
    ).rename(columns={v: v + "_fx" for v in VARS})
    u = pd.read_parquet(
        f"{SP}/run/{base}_units.parquet",
        columns=[
            "spm_unit_id",
            "kids",
            "w0",
            "w1",
            "state",
            "household_support_channel",
        ]
        + VARS,
    )
    u = u.merge(f, on="spm_unit_id", validate="one_to_one")
    n = len(fs)
    print(f"\n{base}: {n} states, {len(u)} units")
    for w in ("w0", "w1"):
        print(
            " ",
            w,
            "SNAP 2024 $bn was",
            round((u[w] * u.snap__2024).sum() / 1e9, 2),
            "fix",
            round((u[w] * u.snap__2024_fx).sum() / 1e9, 2),
            "; housing subsidy $bn was",
            round((u[w] * u.spm_unit_capped_housing_subsidy__2024).sum() / 1e9, 2),
            "fix",
            round((u[w] * u.spm_unit_capped_housing_subsidy__2024_fx).sum() / 1e9, 2),
            "; units with subsidy M was",
            round(
                (u[w] * (u.spm_unit_capped_housing_subsidy__2024 > 0)).sum() / 1e6, 2
            ),
            "fix",
            round(
                (u[w] * (u.spm_unit_capped_housing_subsidy__2024_fx > 0)).sum() / 1e6, 2
            ),
        )
    k = u[u.kids > 0].copy()
    for w in ("w0", "w1"):
        W = k[w] * k.kids
        print(
            " ",
            w,
            "child SPM 2024 was",
            round(100 * np.average(k.spm_unit_is_in_spm_poverty__2024, weights=W), 2),
            "fix",
            round(
                100 * np.average(k.spm_unit_is_in_spm_poverty__2024_fx, weights=W), 2
            ),
            "2025 was",
            round(100 * np.average(k.spm_unit_is_in_spm_poverty__2025, weights=W), 2),
            "fix",
            round(
                100 * np.average(k.spm_unit_is_in_spm_poverty__2025_fx, weights=W), 2
            ),
        )
    W = k.w1 * k.kids
    r = lambda v: (W * k[v]).groupby(k.state).sum() / W.groupby(k.state).sum() * 100
    t = pd.DataFrame(
        {
            "census": c.rate,
            "was": r("spm_unit_is_in_spm_poverty__2025"),
            "fix": r("spm_unit_is_in_spm_poverty__2025_fx"),
        }
    ).dropna()
    t["moe"] = c.moe.reindex(t.index)
    t["step"] = t.fix - t.was
    for v in ("was", "fix"):
        e = t[v] - t.census
        print(
            " ",
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
            int((e.abs() > t.moe).sum()),
        )
    print(
        "  step mean",
        round(t.step.mean(), 2),
        "min",
        round(t.step.min(), 2),
        "max",
        round(t.step.max(), 2),
    )
    t.to_csv(f"{SP}/run/{base}_rentfix_states.csv")
    print(
        t.reindex(
            [
                s
                for s in [
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
                    "CA",
                    "TX",
                    "NY",
                ]
                if s in t.index
            ]
        )
        .round(1)
        .T.to_string()
    )
