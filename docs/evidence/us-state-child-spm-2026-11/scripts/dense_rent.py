"""All 51 states of the dense file with rent back on the household it was drawn for. Same variables and years as the dense run. Weights unchanged.
Usage: dense_rent.py <workers>"""

import glob
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from cutrun import Path, Pool, ex, run, version

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
h5 = [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0]
workers = int(sys.argv[1])
tag = "denserent"
years = (2024, 2025)
d = Path(SP) / "run" / f"{tag}_cuts"
d.mkdir(parents=True, exist_ok=True)
out = Path(SP) / "run" / tag
out.mkdir(parents=True, exist_ok=True)
ex.load(h5)
right = (
    pd.read_parquet(f"{SP}/run/rentfix_dense.parquet")
    .set_index("household_id")
    .rent_right
)
p = ex.tables["person"]
head = p.is_household_head.astype(bool).values
rent = np.zeros(len(p))
rent[head] = p.person_household_id[head].map(right).values
assert np.isfinite(rent).all()
p["pre_subsidy_rent"] = rent
ex.tables["person"] = p
fips = sorted(int(v) for v in ex.tables["household"]["state_fips"].unique())
for f in fips:
    q = d / f"{f:02d}.pkl"
    if not q.exists():
        ex.state_file(f).to_pickle(q)
ex.tables = None
sizes = {f: (d / f"{f:02d}.pkl").stat().st_size for f in fips}
order = sorted(fips, key=sizes.get)
print(
    "policyengine-us",
    version("policyengine-us"),
    "states",
    len(order),
    "cut GB",
    round(sum(sizes.values()) / 1e9, 2),
    flush=True,
)
with Pool(workers, maxtasksperchild=1) as pool:
    for done, sec in pool.imap_unordered(
        run, [(f, str(d), str(out), years) for f in order]
    ):
        print(done, sec, flush=True)
