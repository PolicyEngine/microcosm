import glob
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
for tag, pat, fn in (
    ("default", "9a814a3b*", "populace_us_2024.h5"),
    ("dense", "15cefac4*", "*.h5"),
):
    f = [x for x in glob.glob(S + pat + "/" + fn) if "calib" not in x][0]
    p = pd.read_hdf(f, "person")[
        [
            "person_household_id",
            "person_spm_unit_id",
            "is_household_head",
            "pre_subsidy_rent",
            "age",
        ]
    ]
    h = pd.read_hdf(f, "household")
    s = pd.read_hdf(f, "spm_unit")
    wc = [c for c in h.columns if c == "household_weight"][0]
    print(
        tag,
        "hh cols",
        [c for c in h.columns if "tenure" in c],
        "spm",
        [c for c in s.columns if "tenure" in c],
    )
    a = p.groupby("person_household_id").pre_subsidy_rent.sum().rename("rent")
    h = h.merge(a, left_on="household_id", right_index=True)
    t = (
        h.groupby(["household_support_channel", "tenure_type"])
        .apply(
            lambda g: pd.Series(
                {
                    "w_share": g[wc].sum() / h[wc].sum(),
                    "any_rent": np.average(g.rent > 0, weights=g[wc]),
                    "mean_rent_pos": g.rent[g.rent > 0].mean(),
                    "n": len(g),
                }
            )
        )
        .round(3)
    )
    print(t.to_string())
