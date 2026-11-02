"""Undo the rent misalignment on the dense file: the n-th head in person order holds the rent drawn for the n-th household in table order.
Writes person-level corrected rent; clones take their base record's corrected rent through household_source_id."""

import glob
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0]
p = pd.read_hdf(f, "person")
h = pd.read_hdf(f, "household")
heads = p[p.is_household_head.astype(bool)]
ch = h.household_support_channel.astype(str)
hs = h[ch == "asec"]
hp = heads[
    (heads.person_support_channel.astype(str) == "asec")
    & (heads.person_support_clone_index == 0)
]
assert len(hs) == len(hp)
print(
    "source ids unique in base",
    hs.household_source_id.is_unique,
    type(hs.household_source_id.iloc[0]),
)
right = pd.Series(
    hp.pre_subsidy_rent.values, index=hs.household_source_id.values, name="rent_right"
)
print(
    "base households",
    len(right),
    "positive",
    int((right > 0).sum()),
    "sum",
    round(right.sum() / 1e9, 2),
)
print(
    "other channel sources found",
    h.household_source_id[ch != "asec"].isin(right.index).mean(),
)
print(
    "clone index by channel",
    h.groupby([ch, h.household_support_clone_index]).size().to_dict(),
)
h["rent_right"] = h.household_source_id.map(right)
assert h.rent_right.notna().all()
own = heads.groupby("person_household_id").pre_subsidy_rent.sum().rename("rent_was")
h = h.merge(own, left_on="household_id", right_index=True)
w = h.household_weight
print(
    h.groupby(h.tenure_type.astype(str))
    .apply(
        lambda g: pd.Series(
            {
                "weight_M": g.household_weight.sum() / 1e6,
                "was_any": np.average(g.rent_was > 0, weights=g.household_weight),
                "right_any": np.average(g.rent_right > 0, weights=g.household_weight),
                "was_mean": np.average(g.rent_was, weights=g.household_weight),
                "right_mean": np.average(g.rent_right, weights=g.household_weight),
            }
        )
    )
    .round(3)
    .to_string()
)
print(
    "weighted total rent, was",
    round((w * h.rent_was).sum() / 1e9, 1),
    "right",
    round((w * h.rent_right).sum() / 1e9, 1),
)
print(
    "records moved",
    round(float((h.rent_was != h.rent_right).mean()), 4),
    "weight moved",
    round(float(w[h.rent_was != h.rent_right].sum() / w.sum()), 4),
)
h[["household_id", "rent_right", "rent_was"]].to_parquet(
    f"{SP}/run/rentfix_dense.parquet"
)
