"""Stored-column checks on a US release H5, read without the engine.

Usage: uv run python raw_h5_checks.py <populace_us_2024.h5>

Cross-checks the engine-side receipt from the stored person table: the gross
wage total, who holds 401(k) deferrals, the split by support channel, and
whether a PUF-channel person's wage tracks the same source person's ASEC wage.
"""

import sys

import numpy as np
import pandas as pd

WAGE = "employment_income_before_lsr"
DEFERRAL = "traditional_401k_contributions_desired"

with pd.HDFStore(sys.argv[1], mode="r") as store:
    person = store["person"]
    household = store["household"]

weight = (
    household.set_index("household_id")["household_weight"]
    .reindex(person["person_household_id"].to_numpy())
    .to_numpy()
)
wage = person[WAGE].to_numpy(dtype=np.float64)
deferral = person[DEFERRAL].to_numpy(dtype=np.float64)
channel = person["person_support_channel"].astype(str).to_numpy()

print(f"weighted persons            {weight.sum():,.0f}")
print(f"gross wages                 {np.sum(weight * wage):,.0f}")
print(f"rows with negative wages    {int((wage < 0).sum())}")
print(f"wage earners                {weight[wage > 0].sum():,.0f}")
print(f"stored desired deferrals    {np.sum(weight * deferral):,.0f}")
print(f"deferrals capped at wages   {np.sum(weight * np.minimum(deferral, wage)):,.0f}")
print(f"contributors                {weight[deferral > 0].sum():,.0f}")
print(f"contributors with wages     {weight[(deferral > 0) & (wage > 0)].sum():,.0f}")
for name in (
    "traditional_403b_contributions_desired",
    "pre_tax_health_insurance_premiums",
    "health_savings_account_payroll_contributions",
):
    print(f"stored column {name}: {name in person.columns}")

print("\nby support channel")
for name in np.unique(channel):
    mask = channel == name
    print(
        f"  {name:<16} persons {int(mask.sum()):>7,}"
        f"  wages {np.sum((weight * wage)[mask]) / 1e9:>9,.1f}B"
        f"  desired deferrals {np.sum((weight * deferral)[mask]) / 1e9:>7,.1f}B"
    )

keys = ["source_year", "source_person_id"]
frame = person[keys].copy()
frame["wage"] = wage
frame["channel"] = channel
asec = frame[frame.channel == "asec"].drop_duplicates(keys).set_index(keys)
puf = frame[frame.channel == "puf_tax_detail"].drop_duplicates(keys).set_index(keys)
both = asec.join(puf, lsuffix="_asec", rsuffix="_puf", how="inner")
a = both["wage_asec"].to_numpy()
p = both["wage_puf"].to_numpy()
positive = (a > 0) & (p > 0)
print("\nsource persons on both channels")
print(
    f"  asec persons {len(asec):,}; puf_tax_detail persons {len(puf):,}; on both {len(both):,}"
)
print(f"  correlation of the two wages          {np.corrcoef(a, p)[0, 1]:.4f}")
print(f"  both positive                         {int(positive.sum()):,}")
print(
    f"  median puf/asec ratio, both positive  {np.median(p[positive] / a[positive]):.4f}"
)
print(
    "  5th and 95th percentile of that ratio "
    f"{np.percentile(p[positive] / a[positive], 5):.4f}"
    f" {np.percentile(p[positive] / a[positive], 95):.4f}"
)
print(f"  positive on one channel only          {int(((a > 0) != (p > 0)).sum()):,}")
print(f"  unweighted mean wage, asec and puf    {a.mean():,.0f} {p.mean():,.0f}")
