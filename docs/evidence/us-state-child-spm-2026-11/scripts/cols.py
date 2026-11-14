import os

SP = os.path.dirname(os.path.abspath(__file__))
import os

import numpy as np
import pandas as pd

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5"
for t in ("person", "tax_unit", "spm_unit", "household", "family"):
    d = pd.read_hdf(f, t)
    x = []
    for c in d.columns:
        s = d[c]
        if (
            s.dtype.kind == "f"
            and s.abs().max() > 1000
            and not c.endswith("_id")
            and "weight" not in c
            and not c.startswith("SPM_")
            and "id" not in c
        ):
            x.append(f"{c}:{s.abs().sum() / 1e9:.1f}")
    print(t, d.shape, len(x))
    print(" ".join(x))
u = pd.read_parquet(f"{SP}/run/default_units.parquet")
u = u[u.kids > 0]
print(
    u.groupby("SPM_TENMORTSTATUS")[
        [
            "spm_unit_reference_spm_threshold__2025",
            "spm_unit_reference_spm_threshold__2024",
            "spm_unit_unadjusted_spm_threshold__2025",
        ]
    ]
    .median()
    .round(0)
    .to_string()
)
print(
    (u.spm_unit_spm_threshold__2025 / (u.spm_unit_reference_spm_threshold__2025))
    .describe()
    .round(3)
    .to_dict()
)
a = pd.read_parquet(f"{SP}/asec/raw_asec_2022_2025.parquet")
k = a[a.A_AGE < 18]


def wq(d, q, col="SPM_TOTVAL"):
    d = d.sort_values(col)
    c = d.MARSUPWT.cumsum() / d.MARSUPWT.sum()
    return d[col].values[np.searchsorted(c.values, q)]


print(
    {
        y: [round(wq(d, q)) for q in (0.1, 0.2, 0.3, 0.5)]
        for y, d in k.groupby("income_year")
    }
)
print(
    "mean TOTVAL",
    k.groupby("income_year")
    .apply(lambda d: np.average(d.SPM_TOTVAL, weights=d.MARSUPWT), include_groups=False)
    .round(0)
    .to_dict(),
)
print(
    "mean RESOURCES q",
    {
        y: [round(wq(d, q, "SPM_RESOURCES")) for q in (0.1, 0.2, 0.3, 0.5)]
        for y, d in k.groupby("income_year")
    },
)
