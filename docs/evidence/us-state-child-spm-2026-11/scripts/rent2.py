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
r = p.pre_subsidy_rent
print(
    "persons",
    len(p),
    "nan",
    round(r.isna().mean(), 4),
    "zero",
    round((r == 0).mean(), 4),
    "pos",
    round((r > 0).mean(), 4),
)
print(
    p.assign(pos=r > 0, nan=r.isna())
    .groupby("person_support_channel")[["pos", "nan"]]
    .mean()
    .round(4)
    .to_dict()
)
for c in ("is_household_head", "is_tax_unit_head"):
    if c in p:
        print(c, p.assign(pos=r > 0).groupby(c).pos.mean().round(4).to_dict())
u = p.groupby("person_spm_unit_id").agg(
    n=("pre_subsidy_rent", "size"),
    pos=("pre_subsidy_rent", lambda s: (s > 0).sum()),
    tot=("pre_subsidy_rent", "sum"),
)
print(
    "units share any rent",
    round((u.pos > 0).mean(), 4),
    "mean members with rent among those",
    round(u.pos[u.pos > 0].mean(), 3),
    "mean rent",
    round(u.tot[u.pos > 0].mean()),
)
d = pd.read_parquet(
    f"{SP}/run/dense_units.parquet",
    columns=[
        "spm_unit_id",
        "kids",
        "w1",
        "household_support_channel",
        "SPM_TENMORTSTATUS",
        "SPM_RESOURCES",
        "SPM_POVTHRESHOLD",
        "SPM_CAPHOUSESUB",
    ],
)
d = d.merge(u, left_on="spm_unit_id", right_index=True)
d["W"] = d.w1 * d.kids
d = d[d.kids > 0]
print(
    d.groupby(["household_support_channel", "SPM_TENMORTSTATUS"])
    .apply(
        lambda g: pd.Series(
            {
                "child_w_share": g.W.sum() / d.W.sum(),
                "any_rent": np.average(g.pos > 0, weights=g.W),
                "n": len(g),
            }
        )
    )
    .round(3)
)
