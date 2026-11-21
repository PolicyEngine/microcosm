"""Position test: person_rent[head_mask] = predicted writes household-order predictions in person order.
If the two orders differ, pairing the n-th head's rent with the n-th household's tenure should restore the rent/tenure link."""

import glob
import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0]
p = pd.read_hdf(f, "person")
h = pd.read_hdf(f, "household")
print(
    [
        c
        for c in h.columns
        if "source" in c or "clone" in c or "support" in c or c.endswith("_id")
    ]
)
print([c for c in p.columns if "source" in c or "clone" in c or "support" in c][:10])
print(
    "first household ids",
    h.household_id.head(4).tolist(),
    "monotone hh",
    h.household_id.is_monotonic_increasing,
    "monotone person hh",
    p.person_household_id.is_monotonic_increasing,
)
heads = p[p.is_household_head.astype(bool)]
same = heads.person_household_id.values == h.household_id.values
print(
    "heads",
    len(heads),
    "households",
    len(h),
    "same position share",
    round(same.mean(), 4),
)
hh = h.set_index("household_id")
# the base frame before cloning: unit of test is one channel at clone zero
for ch in sorted(h.household_support_channel.astype(str).unique()):
    hs = h[h.household_support_channel.astype(str) == ch]
    for cl in sorted(p.person_support_clone_index.unique())[:3]:
        hp = heads[
            (heads.person_support_channel.astype(str) == ch)
            & (heads.person_support_clone_index == cl)
        ]
        ids = set(hp.person_household_id)
        hs2 = hs[hs.household_id.isin(ids)]
        print(ch, cl, "heads", len(hp), "hh", len(hs2))
        if len(hp) != len(hs2) or len(hp) == 0:
            continue
        same = (hp.person_household_id.values == hs2.household_id.values).mean()
        ten_own = (
            hs2.set_index("household_id")
            .tenure_type.astype(str)
            .reindex(hp.person_household_id)
            .values
        )
        ten_pos = hs2.tenure_type.astype(str).values
        rent = hp.pre_subsidy_rent.values
        print("   same position", round(same, 4))
        for lab, t in (("own row", ten_own), ("nth household", ten_pos)):
            print(
                "   ",
                lab,
                {
                    k: round(float((rent[t == k] > 0).mean()), 3)
                    for k in ("RENTED", "OWNED_WITH_MORTGAGE", "NONE")
                },
            )
