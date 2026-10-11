"""Add the TY2023 state AGI-band pairs to the US Chronicle feed scope (#940).

Usage: python extend_scope.py <scope.json> <chronicle commit>
Idempotent: re-running with the same commit rewrites the same bytes.
"""

from __future__ import annotations

import json
import sys

POSTALS = (
    "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms "
    "mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy"
).split()
PACKAGE = "packages/irs_soi/historic_table_2_state_agi_2023"
PACKAGE_ID = "soi-historic-table-2-state-agi-2023"
RULE = (
    "The (record_set_id, period) pairs of the pinned feed "
    "consumer_facts_buildn_v9_4.jsonl, plus the 51 TY2023 Historic Table 2 "
    "state AGI-band record sets (irs_soi.ty2023.historic_table_2.state_agi.<st>, "
    "microcosm#940); each built from the named package with "
    "`chronicle build-bundle --year <build_year> --source <package>`."
)

path, commit = sys.argv[1], sys.argv[2]
assert len(commit) == 40, commit
scope = json.load(open(path))
assert len(POSTALS) == 51 == len(set(POSTALS))
pairs = [
    pair
    for pair in scope["pairs"]
    if not pair["record_set_id"].startswith(
        "irs_soi.ty2023.historic_table_2.state_agi."
    )
]
for postal in POSTALS:
    pairs.append(
        {
            "record_set_id": f"irs_soi.ty2023.historic_table_2.state_agi.{postal}",
            "period_type": "tax_year",
            "period_value": "2023",
            "package_id": PACKAGE_ID,
            "package": PACKAGE,
            "build_year": 2023,
        }
    )
pairs.sort(key=lambda e: (e["record_set_id"], e["period_type"], e["period_value"]))
runs: dict[str, set[str]] = {}
for pair in pairs:
    runs.setdefault(str(pair["build_year"]), set()).add(pair["package"])
scope["source_commit"] = commit
scope["rule"] = RULE
scope["pair_count"] = len(pairs)
scope["runs"] = {year: sorted(runs[year]) for year in sorted(runs)}
scope["pairs"] = pairs
with open(path, "w") as handle:
    handle.write(json.dumps(scope, indent=1) + "\n")
print(len(pairs), sum(len(v) for v in scope["runs"].values()))
