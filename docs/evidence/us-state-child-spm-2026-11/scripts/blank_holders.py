"""Who holds a blank parent link in the ACS-local file, by channel, household and state. Reads the split's tables, with no simulation.
Usage: blank_holders.py <split dir> <state csv>"""

import glob
import os
import sys

import pandas as pd

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(d + "/*-person.pkl.gz")):
    fips = int(os.path.basename(f)[:2])
    p = pd.read_pickle(f)
    q = p[
        [
            "parent_1_id",
            "parent_2_id",
            "person_support_channel",
            "person_household_id",
            "age",
        ]
    ].copy()
    del p
    h = pd.read_pickle(f.replace("-person", "-household"))[
        ["household_id", "household_weight"]
    ]
    q = q.merge(h, left_on="person_household_id", right_on="household_id", how="left")
    q["fips"] = fips
    rows.append(q)
a = pd.concat(rows, ignore_index=True)
print("people", len(a), "households", a.person_household_id.nunique())
b1, b2 = a.parent_1_id.isna(), a.parent_2_id.isna()
a["blank"] = b1 | b2
w = a.household_weight
kids = a.age < 18
print(
    "blank parent_1",
    int(b1.sum()),
    "parent_2",
    int(b2.sum()),
    "both",
    int((b1 & b2).sum()),
    "either",
    int(a.blank.sum()),
)
print(
    "weighted share of people",
    round(float((w * a.blank).sum() / w.sum()), 4),
    "of children",
    round(float((w * (a.blank & kids)).sum() / (w * kids).sum()), 4),
)
print(
    "parent_1 zero",
    int((a.parent_1_id == 0).sum()),
    "positive",
    int((a.parent_1_id > 0).sum()),
    "negative",
    int((a.parent_1_id < 0).sum()),
)
print(
    a.groupby("person_support_channel", observed=True)
    .blank.agg(["size", "sum"])
    .to_string()
)
hh = a.groupby("person_household_id").agg(blank=("blank", "sum"), n=("blank", "size"))
print(
    "households with a blank",
    int((hh.blank > 0).sum()),
    "of them all blank",
    int(((hh.blank > 0) & (hh.blank == hh.n)).sum()),
)
st = a.groupby("fips").apply(
    lambda g: pd.Series(
        {
            "people": len(g),
            "blank": int(g.blank.sum()),
            "weighted_share": round(
                float((g.household_weight * g.blank).sum() / g.household_weight.sum()),
                4,
            ),
        }
    ),
    include_groups=False,
)
st.to_csv(sys.argv[2])
print(st.loc[[6, 10, 12, 25, 29]].to_string())
