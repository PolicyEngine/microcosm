import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
u = pd.read_parquet(f"{SP}/run/dense_units.parquet")
u = u[u.kids > 0]
p = lambda n, y=2024: u[f"{n}__{y}"].values.astype(float)
snap = p("snap")
net = p("spm_unit_net_income")
R = (
    p("spm_unit_market_income")
    + p("spm_unit_benefits")
    + p("acp")
    + p("ebb")
    - p("spm_unit_federal_tax")
    - p("spm_unit_payroll_tax")
    - p("spm_unit_self_employment_tax")
    - p("spm_unit_state_tax")
    - p("spm_unit_local_tax")
    - p("spm_unit_spm_expenses")
)
d = net - R
print(
    "n off by >1",
    (np.abs(d) > 1).sum(),
    "of",
    len(d),
    "weight share",
    (u.w1 * u.kids)[np.abs(d) > 1].sum() / (u.w1 * u.kids).sum(),
    "max",
    np.abs(d).max(),
)
x = u[np.abs(d) > 1]
print(x[["state", "kids", "household_support_channel", "w1"]].head(8).to_string())
print(pd.Series(d[np.abs(d) > 1]).describe().round(1).to_dict())
print(
    x[
        [
            "spm_unit_taxes__2024",
            "spm_unit_federal_tax__2024",
            "spm_unit_state_tax__2024",
            "spm_unit_local_tax__2024",
            "spm_unit_payroll_tax__2024",
            "spm_unit_self_employment_tax__2024",
            "acp__2024",
            "ebb__2024",
        ]
    ]
    .head(4)
    .T.round(0)
    .to_string()
)
