"""Merge one state's parts from acs_parts.sh into the state's output, once the parts hold each of the state's SPM units exactly
once. Rows go in SPM unit id order, the order a whole-state run writes.
Usage: parts_merge.py <fips> <tag> <parts> <parts folder> <output folder>
"""

import os
import sys

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
fips, tag, K, parts, dest = (
    int(sys.argv[1]),
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
u = pd.concat(
    [pd.read_parquet(f"{parts}/p{k}/{fips:02d}.parquet") for k in range(K)],
    ignore_index=True,
)
assert u.spm_unit_id.is_unique, "an SPM unit sits in two parts"
hh = pd.read_parquet(f"{SP}/run/acs_hh.parquet", columns=["household_id", "state_fips"])
uh = pd.read_parquet(f"{SP}/run/acs_unit_hh.parquet")
want = set(
    uh.person_spm_unit_id[
        uh.person_household_id.isin(hh.household_id[hh.state_fips == fips])
    ]
)
got = set(u.spm_unit_id)
assert got == want, (
    f"parts hold {len(got)} SPM units and the state has {len(want)}: {len(got - want)} extra, {len(want - got)} missing"
)
u = u.sort_values("spm_unit_id", kind="stable").reset_index(drop=True)
os.makedirs(dest, exist_ok=True)
target = f"{dest}/{fips:02d}.parquet"
u.to_parquet(target + ".tmp", compression="zstd")
os.replace(target + ".tmp", target)
print(fips, tag, "parts", K, "units", len(u), "merged into", dest, flush=True)
