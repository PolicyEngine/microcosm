"""Which units carry a state's move between the -1 fill and fill 0, for the baseline and for the 2021 CTC reform.

Usage: fill_trace.py <state> <folder under -1> [<folder under 0>]. The folders sit under run/; the second defaults to acs.
A unit counts as moved when its baseline or reform poverty status differs between the fills. The rate and effect moves come
from the units, weighted by household weight times children, as acs_eval.py weights them.
It prints each moved source household's channel, status changes and share of the state's child weight, as fl_reform.py
does, and no incomes. UNIT_DETAIL=1 adds each unit's net income and threshold, for a local check only.
"""

import os
import re
import sys

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
FIPS = eval(
    re.search(r"FIPS\s*=\s*(\{.*?\})", open(f"{SP}/acs_eval.py").read(), re.S).group(1)
)
st, m1 = sys.argv[1], sys.argv[2]
z = sys.argv[3] if len(sys.argv) > 3 else "acs"
fips = FIPS[st]
h = pd.read_parquet(f"{SP}/run/acs_hh.parquet")
uh = pd.read_parquet(f"{SP}/run/acs_unit_hh.parquet").drop_duplicates(
    "person_spm_unit_id"
)
a = pd.read_parquet(f"{SP}/run/{z}/{fips:02d}.parquet")
b = pd.read_parquet(f"{SP}/run/{m1}/{fips:02d}.parquet")
assert a.spm_unit_id.is_unique and set(a.spm_unit_id) == set(b.spm_unit_id), (
    "the fills hold different units"
)
m = a.merge(b, on="spm_unit_id", suffixes=("_0", "_m1"))
assert (m.kids_0 == m.kids_m1).all(), "the fills count children differently"
m = (
    m[m.kids_0 > 0]
    .merge(uh, left_on="spm_unit_id", right_on="person_spm_unit_id")
    .merge(h, left_on="person_household_id", right_on="household_id")
)
assert (m.state_fips == fips).all()
m["W"] = m.household_weight * m.kids_0
T = m.W.sum()


def rate(col):
    return 100 * (m.W * m[col]).sum() / T


out = {"state": st, "child_units": len(m)}
for f in ("0", "m1"):
    out[f"rate_{f}"] = rate(f"poor_{f}")
    if f"poor_r_{f}" in m:
        out[f"reform_rate_{f}"] = rate(f"poor_r_{f}")
        out[f"effect_{f}"] = out[f"reform_rate_{f}"] - out[f"rate_{f}"]
moved = m.poor_0 != m.poor_m1
if "poor_r_0" in m and "poor_r_m1" in m:
    moved |= m.poor_r_0 != m.poor_r_m1
mv = m[moved].copy()
out["units_moved"] = len(mv)
out["source_households"] = mv.household_source_id.nunique()
out["by_channel"] = mv.household_support_channel.astype(str).value_counts().to_dict()
out["moved_child_weight_share_pct"] = 100 * mv.W.sum() / T
out["largest_source_household_share_pct"] = (
    100 * mv.groupby("household_source_id").W.sum().max() / T if len(mv) else 0.0
)
for k, v in out.items():
    print(k, round(v, 6) if isinstance(v, float) else v)
status = ["poor_0", "poor_m1"] + [c for c in ("poor_r_0", "poor_r_m1") if c in mv]
mv["share_pct"] = 100 * mv.W / T
mv["src"] = (
    mv.household_support_channel.astype(str) + ":" + mv.household_source_id.astype(str)
)
by_src = mv.groupby("src").agg(
    units=("spm_unit_id", "size"),
    share_pct=("share_pct", "sum"),
    **{c: (c, "max") for c in status},
)
print(by_src.sort_values("share_pct", ascending=False).round(4).to_string())
if os.environ.get("UNIT_DETAIL") == "1":
    mv["net_move"] = mv.net_m1 - mv.net_0
    print(
        mv[["src", "kids_0"] + status + ["share_pct", "net_0", "net_move", "thr_0"]]
        .round(4)
        .to_string(index=False)
    )
