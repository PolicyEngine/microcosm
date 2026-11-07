"""Swap Census's 2024 reference thresholds into PolicyEngine's 2024 poverty test on the default file.

This is arithmetic on simulation output: resources stay as simulated and only the comparison
threshold changes, by the ratio of Census's reference threshold to PolicyEngine's for each tenure.
"""

import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
a = pd.read_parquet(
    f"{SP}/asec/raw_asec_2022_2025.parquet",
    columns=[
        "A_AGE",
        "SPM_POVTHRESHOLD",
        "SPM_GEOADJ",
        "SPM_EQUIVSCALE",
        "SPM_TENMORTSTATUS",
        "income_year",
    ],
)
a = a[(a.A_AGE < 18) & (a.income_year == 2024)]
cen = (
    (a.SPM_POVTHRESHOLD / (a.SPM_GEOADJ * a.SPM_EQUIVSCALE))
    .groupby(a.SPM_TENMORTSTATUS)
    .median()
)
u = pd.read_parquet(
    f"{SP}/run/default_units.parquet",
    columns=[
        "kids",
        "w1",
        "spm_unit_net_income__2024",
        "spm_unit_spm_threshold__2024",
        "spm_unit_reference_spm_threshold__2024",
        "spm_unit_is_in_spm_poverty__2024",
        "SPM_TENMORTSTATUS",
    ],
)
u = u[u.kids > 0].copy()
ref = u["spm_unit_reference_spm_threshold__2024"].round(0)
vals = sorted(ref.unique())
print("Census reference thresholds by tenure", cen.round(0).to_dict())
print("PolicyEngine reference thresholds", vals)
# Read the tenure behind each PolicyEngine value from the records, since near-equal values sort unreliably.
t = pd.crosstab(ref, u.SPM_TENMORTSTATUS)
print(t.to_string())
pairs = {v: cen[t.loc[v].idxmax()] for v in vals}
print("pairs", {float(k): round(float(v)) for k, v in pairs.items()})
ratio = ref.map(pairs) / ref
W = u.w1 * u.kids
base = (
    100
    * (W * (u["spm_unit_net_income__2024"] < u["spm_unit_spm_threshold__2024"])).sum()
    / W.sum()
)
swap = (
    100
    * (
        W * (u["spm_unit_net_income__2024"] < u["spm_unit_spm_threshold__2024"] * ratio)
    ).sum()
    / W.sum()
)
flag = 100 * (W * u["spm_unit_is_in_spm_poverty__2024"].astype(float)).sum() / W.sum()
print(
    f"child rate 2024: simulated {flag:.2f}, recomputed {base:.2f}, with Census thresholds {swap:.2f}"
)
