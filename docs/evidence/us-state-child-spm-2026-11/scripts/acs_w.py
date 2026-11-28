import os

SP = os.path.dirname(os.path.abspath(__file__))
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd

B = os.path.expanduser(
    os.path.expanduser("~/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928")
)
ns = sorted(os.listdir(B + "/runs"))
print(len(ns), [n for n in ns if not n.startswith("hold")][:60])
z = np.load(B + "/runs/dup_release_repro/weights.npz")
print({k: z[k].shape for k in z.files})
m = json.load(open(B + "/runs/dup_release_repro/metrics.json"))
print(list(m)[:30])
f = glob.glob(B + "/checkpoint/*.h5")
print(f)
with h5py.File(f[0], "r") as h:
    print(list(h.keys())[:20])
    print(list(h["household"].keys()) if "household" in h else "")
c = pd.read_hdf(f[0], "household")
print(c.shape, list(c.columns)[:20])
print(c.head(3).to_string()[:600])
a = pd.read_parquet(f"{SP}/run/acs_hh.parquet")
print(a.head(3).to_string()[:500])
s = set(c.household_id)
print("overlap ids", len(s & set(a.household_id)), len(a))
