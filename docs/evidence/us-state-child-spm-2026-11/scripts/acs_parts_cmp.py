"""Check that a state run in household parts (acs_parts.sh) writes exactly what a whole-state run writes: every column, value,
dtype and row order. The whole-state files come from the rerun itself.
"""

import sys

import pandas as pd

PAIRS = [
    ("run/acs/56.parquet", "run/parts_check/acs/56.parquet"),
    ("run/acs/10.parquet", "run/parts_check/acs/10.parquet"),
    ("run/acs/05.parquet", "run/parts_check/acs/05.parquet"),
    ("run/acsrent/56.parquet", "run/parts_check/acsrent/56.parquet"),
    ("run/acsrent/50.parquet", "run/parts_check/acsrent/50.parquet"),
]
differ = 0
for whole, parts in PAIRS:
    a, b = pd.read_parquet(whole), pd.read_parquet(parts)
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        print(
            "same",
            whole,
            parts,
            a.shape,
            " ".join(f"{c}:{t}" for c, t in a.dtypes.items()),
            flush=True,
        )
    except AssertionError as error:
        differ += 1
        print("differ", whole, parts, str(error)[:600], flush=True)
sys.exit(1 if differ else 0)
