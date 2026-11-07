"""Run every stage of measure.py in its own process, most important scenarios first.

Usage: run_all.py LABEL PYTHON OUT_PREFIX DATASET [--widen]
A failed stage is logged and the next one still runs; combine runs last.
"""

import subprocess
import sys
import time

label, python, out_prefix, dataset, *extra = sys.argv[1:]
here = "/Users/maxghenis/reviews/microcosm-sehi-medicare-2026-10-11"
# (scenario, light, neutralize): heavy runs add AGI, federal and state income tax, the
# medical expense deduction and SNAP; neutralized runs report income_tax with
# the deduction neutralized (JCT's tax-expenditure row).
ORDER = [
    ("shipped", False, False),
    ("shipped", False, True),
    ("reported_plus_modeled_part_b", False, False),
    ("reported_plus_modeled_part_b", False, True),
    ("reported_only", False, False),
    ("reported_only", False, True),
    ("engine_side_standard_part_b", False, False),
    ("reported_where_not_enrolled", False, False),
    ("reported_where_not_enrolled", False, True),
    ("reported_plus_reported_part_b", True, False),
    ("reported_plus_modeled_part_b_where_reported", True, False),
    ("reported_plus_modeled_part_b_sole_proprietors", True, False),
    ("reported_plus_all_modeled_medicare", True, False),
    ("engine_side", True, False),
]
base = [python, "-I", f"{here}/measure.py", label, out_prefix, dataset, *extra]
stages = [["--stage", "prepare"]]
stages += [
    [
        "--stage",
        "scenario",
        "--name",
        n,
        *(["--light"] if light else []),
        *(["--neutralize"] if neutralize else []),
    ]
    for n, light, neutralize in ORDER
]
stages += [["--stage", "combine"]]
for stage in stages:
    t0 = time.time()
    rc = subprocess.call(base + stage, stdin=subprocess.DEVNULL)
    print(
        f"{label} stage {' '.join(stage)} exit {rc} after {time.time() - t0:.0f} s",
        flush=True,
    )
