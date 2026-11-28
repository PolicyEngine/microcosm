"""Blank, zero, negative and positive parent links on the default and dense files, which take no fill."""

import glob
import os

import pandas as pd

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
for tag, rev in (("default", "9a814a3b"), ("dense", "15cefac4")):
    p = pd.read_hdf(
        [x for x in glob.glob(S + rev + "*/*.h5") if "calib" not in x][0], "person"
    )
    print(tag, "people", len(p))
    cols = [c for c in ("parent_1_id", "parent_2_id") if c in p]
    if not cols:
        print("  stores no parent columns")
    for c in cols:
        v = p[c]
        print(
            " ",
            c,
            v.dtype,
            "blank",
            int(v.isna().sum()),
            "zero",
            int((v == 0).sum()),
            "negative",
            int((v < 0).sum()),
            "positive",
            int((v > 0).sum()),
        )
    del p
