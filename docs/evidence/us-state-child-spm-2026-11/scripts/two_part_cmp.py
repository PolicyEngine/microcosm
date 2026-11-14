"""Check that acs_rerun.py in two steps writes exactly what one process writes: every column, value, dtype and row order.

The rerun itself wrote the one-process files in run/acs and run/acsrent before the two-step mode existed. The acsrent_one file
comes from the new script forced to one process, so it also checks that the new script's one-process path changed nothing.
"""

import sys

import pandas as pd

PAIRS = [
    ("run/acs/56.parquet", "run/two_part_check/acs/56.parquet"),
    ("run/acs/10.parquet", "run/two_part_check/acs/10.parquet"),
    ("run/acs/05.parquet", "run/two_part_check/acs/05.parquet"),
    ("run/acsrent/56.parquet", "run/two_part_check/acsrent_one/56.parquet"),
    ("run/acsrent/56.parquet", "run/two_part_check/acsrent/56.parquet"),
]
differ = 0
for one, two in PAIRS:
    a, b = pd.read_parquet(one), pd.read_parquet(two)
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        print(
            "same",
            one,
            two,
            a.shape,
            " ".join(f"{c}:{t}" for c, t in a.dtypes.items()),
            flush=True,
        )
    except AssertionError as error:
        differ += 1
        print("differ", one, two, str(error)[:600], flush=True)
sys.exit(1 if differ else 0)
