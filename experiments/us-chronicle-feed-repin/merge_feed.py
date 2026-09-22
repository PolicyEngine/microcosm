"""Merge per-year Chronicle bundles into one US consumer feed (scratch; nothing is pinned).

Rule: keep rows whose record-set namespace is one the pinned US feed uses; one row per
aggregate_fact_key; refuse if two rows with the same key disagree on value; sort by key.
"""

import collections
import hashlib
import json
import os
import sys

EXPORT_DIR = os.path.expanduser(
    "~/PolicyEngine/_recovered/scratch-backup/893/chronicle-export"
)
PINNED = os.path.expanduser(
    "~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl"
)
years = sys.argv[1:]


def ns(row):
    return ((row.get("layout") or {}).get("record_set_id") or "").split(".")[0]


keep_ns, pinned_rs = set(), set()
with open(PINNED) as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            keep_ns.add(ns(r))
            pinned_rs.add((r.get("layout") or {}).get("record_set_id"))
rows, conflicts, seen_in = {}, [], collections.defaultdict(set)
for y in years:
    n = 0
    with open(f"{EXPORT_DIR}/ledger-us-{y}/consumer_facts.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if ns(r) not in keep_ns:
                continue
            k = r["aggregate_fact_key"]
            n += 1
            seen_in[k].add(y)
            if k in rows:
                if rows[k][0].get("value") != r.get("value"):
                    conflicts.append((k, rows[k][0].get("value"), r.get("value")))
                continue
            rows[k] = (r, line if line.endswith("\n") else line + "\n")
    print(f"year {y}: {n} US-namespace rows")
print("unique facts:", len(rows), "| value conflicts across years:", len(conflicts))
for c in conflicts[:5]:
    print("  CONFLICT", c)
multi = sum(1 for v in seen_in.values() if len(v) > 1)
print("facts present in more than one year's bundle:", multi)
out = f"{EXPORT_DIR}/us-merged-{'-'.join(years)}.jsonl"
h = hashlib.sha256()
with open(out, "w") as f:
    for k in sorted(rows):
        f.write(rows[k][1])
        h.update(rows[k][1].encode())
new_rs = {(r.get("layout") or {}).get("record_set_id") for r, _ in rows.values()}
print(
    "wrote",
    out,
    "sha256",
    h.hexdigest()[:16],
    "| record sets:",
    len(new_rs),
    "| pinned sets missing:",
    len(pinned_rs - new_rs),
)
print("missing:", sorted(x for x in pinned_rs - new_rs if x)[:40])
