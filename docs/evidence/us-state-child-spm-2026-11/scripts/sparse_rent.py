"""All 51 states of the default file with rent back on the household it was drawn for; weights unchanged. Source ids share the dense file's draws.
Usage: sparse_rent.py <workers>"""

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
h5 = S + "9a814a3b3b53c0ecd6e1737b6ec862c31300ef6f/populace_us_2024.h5"
workers = int(sys.argv[1])
tag = "sparserent"
years = (2024, 2025)
d = Path(SP) / "run" / f"{tag}_cuts"
d.mkdir(parents=True, exist_ok=True)
out = Path(SP) / "run" / tag
out.mkdir(parents=True, exist_ok=True)
de = pd.read_hdf(
    [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0], "household"
)[["household_id", "household_source_id"]]
right = (
    pd.read_parquet(f"{SP}/run/rentfix_dense.parquet")
    .merge(de, on="household_id")
    .drop_duplicates("household_source_id")
    .set_index("household_source_id")
    .rent_right
)
ex.load(h5)
h = ex.tables["household"]
hr = pd.Series(h.household_source_id.map(right).values, index=h.household_id.values)
assert hr.notna().all()
p = ex.tables["person"]
head = p.is_household_head.astype(bool).values
rent = np.zeros(len(p))
rent[head] = p.person_household_id[head].map(hr).values
was = p.pre_subsidy_rent.values
print(
    "records moved",
    round(float((was != rent).mean()), 4),
    "rent total was",
    round(was.sum() / 1e6, 1),
    "right",
    round(rent.sum() / 1e6, 1),
    flush=True,
)
p["pre_subsidy_rent"] = rent
ex.tables["person"] = p
fips = sorted(int(v) for v in h["state_fips"].unique())
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
