"""Run measure.py stages one process at a time, each under a memory guard.

The shared heavy lock stops the whole process tree above 20 GB. pe-us 2.2.1's
full tax-and-benefit cascade passed 47 GB in one process, so heavy outputs run
in separate processes by group, and this driver stops any single stage whose
footprint passes GUARD_MB itself (logging it) and moves on to the next stage.

Usage: run_guarded.py LABEL PYTHON OUT_PREFIX DATASET PLAN [--widen]
PLAN names a list below.
"""

import os
import subprocess
import sys
import time

label, python, out_prefix, dataset, plan, *extra = sys.argv[1:]
here = "/Users/maxghenis/reviews/microcosm-sehi-medicare-2026-10-11"
GUARD_MB = int(os.environ.get("GUARD_MB", "18500"))


def sc(name, *flags):
    return ["--stage", "scenario", "--name", name, *flags]


HEAVY = (
    "shipped",
    "reported_plus_modeled_part_b",
    "reported_only",
    "reported_where_not_enrolled",
)
PLANS = {
    # pe-us 2.2.1: light core first, then the federal group and the neutralized
    # deduction, then state and SNAP, each in its own process.
    "peus221": (
        # Federal group (which also writes the core measures) and the
        # neutralized deduction for the decision-relevant scenarios.
        [
            x
            for n in HEAVY
            for x in (
                sc(n, "--outputs", "federal", "--tag", "federal"),
                sc(n, "--neutralize"),
            )
        ]
        + [
            sc(
                "engine_side_standard_part_b",
                "--outputs",
                "federal",
                "--tag",
                "federal",
            ),
            sc("engine_side_standard_part_b", "--neutralize"),
        ]
        + [
            sc(n, "--light")
            for n in (
                "reported_plus_reported_part_b",
                "reported_plus_modeled_part_b_where_reported",
                "reported_plus_all_modeled_medicare",
                "engine_side",
            )
        ]
        + [
            ["--stage", "outyear", "--name", n]
            for n in ("shipped", "reported_plus_modeled_part_b")
        ]
        + [sc(n, "--outputs", "state", "--tag", "state") for n in HEAVY[:3]]
        + [["--stage", "combine"]]
    ),
    # pe-us 2.2.1 needs more than 28 GB for income tax on the full file, so
    # only deduction-level measures and input uprating run at the pinned engine.
    "peus221_light": [
        sc(n, "--light")
        for n in (
            "shipped",
            "reported_where_not_enrolled",
            "reported_only",
            "reported_plus_modeled_part_b",
            "engine_side_standard_part_b",
            "reported_plus_reported_part_b",
            "reported_plus_modeled_part_b_where_reported",
            "reported_plus_all_modeled_medicare",
            "engine_side",
        )
    ]
    + [
        ["--stage", "outyear", "--name", n, "--light"]
        for n in ("shipped", "reported_plus_modeled_part_b")
    ]
    + [["--stage", "combine"]],
    # The #10120 run's stages that failed run_all.py's float64-tolerance input
    # check (non-integer premiums), then the candidate out-year files.
    "peus10120_retry": [
        sc("reported_plus_modeled_part_b"),
        sc("reported_plus_modeled_part_b", "--neutralize"),
        sc("engine_side_standard_part_b", "--neutralize"),
    ]
    + [
        ["--stage", "outyear", "--name", n]
        for n in ("shipped", "reported_plus_modeled_part_b")
    ]
    + [["--stage", "combine"]],
    # The #10120 run's remaining work after run_all.py: candidate out-year files.
    "peus10120_outyear": [
        ["--stage", "outyear", "--name", n]
        for n in ("shipped", "reported_plus_modeled_part_b")
    ]
    + [["--stage", "combine"]],
}


def footprint_mb(pid):
    try:
        text = subprocess.run(
            ["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        text = ""
    for line in text.splitlines():
        if "phys_footprint:" in line:
            parts = line.split()
            value, unit = float(parts[1]), parts[2]
            return value * {"GB": 1024, "MB": 1, "KB": 1 / 1024, "B": 1 / 1048576}.get(
                unit, 1
            )
    try:
        rss_kb = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        ).stdout
        return float(rss_kb.strip() or 0) / 1024
    except Exception:
        return 0.0


base = [python, "-I", f"{here}/measure.py", label, out_prefix, dataset, *extra]
header = out_prefix + ".header.json"
stages = PLANS[plan]
if not os.path.exists(header) or not os.path.exists(out_prefix + ".inputs.npz"):
    stages = [["--stage", "prepare"]] + stages
for stage in stages:
    t0 = time.time()
    process = subprocess.Popen(base + stage, stdin=subprocess.DEVNULL)
    peak, guarded = 0.0, False
    while process.poll() is None:
        used = footprint_mb(process.pid)
        peak = max(peak, used)
        if used > GUARD_MB:
            guarded = True
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(3)
    rc = process.wait()
    note = f" STOPPED by the driver memory guard at {peak:.0f} MB" if guarded else ""
    print(
        f"{label} stage {' '.join(stage)} exit {rc} after {time.time() - t0:.0f} s, "
        f"peak footprint {peak:.0f} MB{note}",
        flush=True,
    )
