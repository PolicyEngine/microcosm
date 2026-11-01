"""ACS-local file with rent on the right households: donors take the base record's right rent, ACS renter heads take twelve times native monthly contract rent. Weights unchanged."""

import glob
import os
import sys

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from pathlib import Path

import extract_state as ex

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
ex.load(S + "725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5")
d = Path(SP) / "run/acsrent_cuts"
d.mkdir(parents=True, exist_ok=True)
(Path(SP) / "run/acsrent").mkdir(exist_ok=True)
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
h = ex.tables["household"]
p = ex.tables["person"]
acs = h.household_support_channel.astype(str).str.startswith("acs").values
hr = np.where(
    acs,
    12 * pd.to_numeric(h.acs_monthly_contract_rent, errors="coerce").fillna(0).values,
    h.household_source_id.map(right).values,
)
hr = pd.Series(hr, index=h.household_id.values)
assert hr.notna().all()
head = p.is_household_head.astype(bool).values
rent = np.zeros(len(p))
rent[head] = p.person_household_id[head].map(hr).values
print(
    "acs renters with no head",
    int(
        (
            (hr.values > 0)
            & ~pd.Series(hr.index).isin(p.person_household_id[head]).values
        ).sum()
    ),
    "rent was",
    round(p.pre_subsidy_rent.sum() / 1e9, 2),
    "now",
    round(rent.sum() / 1e9, 2),
    flush=True,
)
p["pre_subsidy_rent"] = rent
ex.tables["person"] = p
for f in sorted(int(v) for v in h["state_fips"].unique()):
    q = d / f"{f:02d}.pkl"
    if not q.exists():
        ex.state_file(f).to_pickle(q)
print("cut", flush=True)
