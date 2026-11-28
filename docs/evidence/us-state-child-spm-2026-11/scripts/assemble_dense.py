import os

SP = os.path.dirname(os.path.abspath(__file__))
import sys

sys.argv = ["x"]
import glob

import assemble as a
import pandas as pd

h, pp = a.people_table(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/15cefac484174ae0ee11a82b164f58023ad05c84/populace_us_2024.h5"
    ),
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/15cefac484174ae0ee11a82b164f58023ad05c84/populace_us_2024_calibration.npz"
    ),
)
u = a.units(h, pp)
print(len(u))
pe = pd.concat(
    [pd.read_parquet(f) for f in sorted(glob.glob(f"{SP}/run/dense/*.parquet"))],
    ignore_index=True,
)
m = u.merge(pe, on="spm_unit_id", how="inner", validate="one_to_one")
assert (m.kids == m.n_children).all()
print(len(m), len(pe))
m.to_parquet(f"{SP}/run/dense_units.parquet")
