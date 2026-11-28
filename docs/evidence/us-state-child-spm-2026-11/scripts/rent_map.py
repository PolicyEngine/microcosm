"""The base record's right rent by source household, for the corrected-rent runs on the ACS-local file."""

import glob
import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
de = pd.read_hdf(
    [x for x in glob.glob(S + "15cefac4*/*.h5") if "calib" not in x][0], "household"
)[["household_id", "household_source_id"]]
right = (
    pd.read_parquet(f"{SP}/run/rentfix_dense.parquet")
    .merge(de, on="household_id")
    .drop_duplicates("household_source_id")[["household_source_id", "rent_right"]]
)
right.to_parquet(f"{SP}/run/rent_right.parquet", compression="zstd")
print(
    "source households", len(right), "rent $bn", round(right.rent_right.sum() / 1e9, 2)
)
