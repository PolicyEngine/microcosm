"""Which SPM units flip poverty between PE 2024 and PE 2025, and how much weight they carry."""

import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
u = pd.read_parquet(f"{SP}/run/default_units.parquet")
u = u[u.kids > 0].copy()
u["W"] = u.w1 * u.kids
p24 = u["spm_unit_is_in_spm_poverty__2024"].astype(bool)
p25 = u["spm_unit_is_in_spm_poverty__2025"].astype(bool)
u["in"] = (~p24) & p25
u["out"] = p24 & (~p25)
g = u.groupby("state")
tot = g.W.sum()
s = pd.DataFrame(
    {
        "pe24": 100 * (u.W * p24).groupby(u.state).sum() / tot,
        "pe25": 100 * (u.W * p25).groupby(u.state).sum() / tot,
        "in_pts": 100 * (u.W * u["in"]).groupby(u.state).sum() / tot,
        "out_pts": 100 * (u.W * u["out"]).groupby(u.state).sum() / tot,
        "n_in": u["in"].groupby(u.state).sum(),
        "n_out": u["out"].groupby(u.state).sum(),
    }
)
s["top1_in_share"] = (
    u[u["in"]].groupby("state").W.max() / u[u["in"]].groupby("state").W.sum()
)
s["top1_in_pts"] = 100 * u[u["in"]].groupby("state").W.max() / tot
print(s.sort_values("in_pts", ascending=False).round(2).head(16).to_string())
print(
    "all states: in",
    (u.W * u["in"]).sum() / u.W.sum() * 100,
    "out",
    (u.W * u["out"]).sum() / u.W.sum() * 100,
    "n_in",
    u["in"].sum(),
    "n_out",
    u["out"].sum(),
)
cols = [
    "state",
    "kids",
    "n_people__x" if "n_people__x" in u else "people",
    "w1",
    "w0",
    "source_year",
    "household_support_channel",
    "SPM_TENMORTSTATUS" if "SPM_TENMORTSTATUS" in u else "SPM_POOR",
]
for st in ["MD", "DC"]:
    x = u[(u.state == st) & u["in"]].sort_values("W", ascending=False).head(6)
    show = x[
        ["kids", "w0", "w1", "W", "source_year", "household_support_channel"]
    ].copy()
    for n in [
        "spm_unit_net_income",
        "spm_unit_spm_threshold",
        "spm_unit_market_income",
        "spm_unit_benefits",
        "spm_unit_taxes",
        "spm_unit_spm_expenses",
        "snap",
        "spm_unit_state_tax",
        "spm_unit_federal_tax",
        "refundable_ctc",
        "eitc",
        "employment_income",
    ]:
        for y in (2024, 2025):
            show[f"{n[:18]}_{y % 100}"] = x[f"{n}__{y}"]
    print(st)
    print(show.round(0).T.to_string())
