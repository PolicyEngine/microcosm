"""Growth in child SPM resources at fixed percentiles, 2024 to 2025: Census raw ASEC against PolicyEngine at published weights. Thresholds move too, so the table also compares resources over thresholds."""

import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
PS = [10, 15, 20, 25, 30, 40, 50]


def wq(x, w, ps):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(np.array(ps) / 100 * c[-1], c, x[o])


a = pd.read_parquet(f"{SP}/asec/raw_asec_2022_2025.parquet")
Y = "income_year" if "income_year" in a else ("year" if "year" in a else "survey_year")
k = a[a.A_AGE < 18]
W = "MARSUPWT" if "MARSUPWT" in k else "A_FNLWGT"
rows = {}
yy = sorted(k[Y].unique())
print(Y, yy)


def census(y):
    g = k[k[Y] == y]
    return wq(g.SPM_RESOURCES.values.astype(float), g[W].values.astype(float), PS), wq(
        (g.SPM_RESOURCES / g.SPM_POVTHRESHOLD).values.astype(float),
        g[W].values.astype(float),
        PS,
    )


cy = [y for y in yy if y in (2024, 2025)]
sh = 2 if len(cy) == 2 else None
if len(cy) == 2:
    r0, x0 = census(2024)
    r1, x1 = census(2025)
else:
    last = yy[-2:]
    print("using", last)
    r0, x0 = census(last[0])
    r1, x1 = census(last[1])
rows["census_resources"] = 100 * (r1 / r0 - 1)
rows["census_ratio"] = 100 * (x1 / x0 - 1)
for tag in ("default", "dense"):
    u = pd.read_parquet(
        f"{SP}/run/{tag}_units.parquet",
        columns=[
            "kids",
            "w1",
            "spm_unit_net_income__2024",
            "spm_unit_net_income__2025",
            "spm_unit_spm_threshold__2024",
            "spm_unit_spm_threshold__2025",
        ],
    )
    u = u[u.kids > 0]
    w = (u.w1 * u.kids).values
    a0, a1 = [
        wq(u[f"spm_unit_net_income__{y}"].values.astype(float), w, PS)
        for y in (2024, 2025)
    ]
    b0, b1 = [
        wq(
            (
                u[f"spm_unit_net_income__{y}"] / u[f"spm_unit_spm_threshold__{y}"]
            ).values.astype(float),
            w,
            PS,
        )
        for y in (2024, 2025)
    ]
    rows[f"{tag}_resources"] = 100 * (a1 / a0 - 1)
    rows[f"{tag}_ratio"] = 100 * (b1 / b0 - 1)
t = pd.DataFrame(rows, index=[f"p{p}" for p in PS])
pd.set_option("display.width", 200)
print(t.round(2).to_string())
t.to_csv(f"{SP}/run/resource_growth.csv")
