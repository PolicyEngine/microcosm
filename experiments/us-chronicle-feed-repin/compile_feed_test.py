"""Compile a Chronicle consumer feed through microcosm's US target registry (read-only)."""

import collections
import json
import sys
import time
from pathlib import Path

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.fiscal_targets import compile_us_fiscal_target_registry
from microcosm.build.us_runtime.medicaid_take_up import (
    apply_us_medicaid_enrollment_substitutions,
)
from microcosm.build.us_runtime.release_target_parity import us_target_family_id

feed = Path(sys.argv[1])
pinned = Path(sys.argv[2])
inventory = json.loads(Path(sys.argv[3]).read_text())


def scan(p):
    rows = [json.loads(line) for line in p.open() if line.strip()]
    return rows


t0 = time.perf_counter()
new = scan(feed)
old = scan(pinned)


def summary(rows, name):
    periods = collections.Counter(str(r.get("period")) for r in rows)
    rsets = set(r.get("layout", {}).get("record_set_id") for r in rows)
    labeled = sum(1 for r in rows if r.get("dimension_labels"))
    print(
        f"{name}: rows={len(rows)} labeled_rows={labeled} record_sets={len(rsets)} periods={dict(sorted(periods.items()))}"
    )
    return rsets


rs_new = summary(new, "NEW")
rs_old = summary(old, "PINNED v9_4")
print("record sets in pinned but not in new:", len(rs_old - rs_new))
print(sorted(rs_old - rs_new)[:40])
print("record sets in new but not in pinned:", len(rs_new - rs_old))
art = load_ledger_consumer_artifact(feed)
xw = load_congressional_district_vintage_crosswalk(
    default_congressional_district_vintage_crosswalk_path()
)
t1 = time.perf_counter()
try:
    reg = compile_us_fiscal_target_registry(
        art.facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=xw,
        age_targets=True,
    )
    reg, subs = apply_us_medicaid_enrollment_substitutions(reg)
    fams = collections.Counter(us_target_family_id(s) for s in reg)
    print(
        f"COMPILED OK: targets={len(list(reg))} families={len(fams)} in {time.perf_counter() - t1:.1f}s"
    )
    pinned_compiled = {k for k, v in inventory.get("families", {}).items()}
    print(
        "pinned inventory families:", len(pinned_compiled), "| compiled now:", len(fams)
    )
    print(
        "families compiled now but absent from pinned inventory:",
        sorted(set(fams) - pinned_compiled)[:30],
    )
    print(
        "pinned inventory families not compiled now:",
        sorted(pinned_compiled - set(fams))[:60],
    )
except Exception as e:
    print("COMPILE FAILED:", type(e).__name__, str(e)[:600])
print(f"total {time.perf_counter() - t0:.1f}s")
