#!/usr/bin/env bash
# microcosm#1003 determinism receipt: a second candidate build from the same tree and inputs,
# payload-compared with spine-lisa (every store key, the root mass log, the was_lisa stage evidence).
set -uo pipefail
A=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1003-lisa
U=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds
CAND=/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1003-shard
TREE=/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1003
OUT=$A/diff-spine-lisa-vs-spine-lisa-2; mkdir -p "$OUT"
echo "$(date +%H:%M:%S) shard head $(git -C $CAND rev-parse HEAD); dirty: $(git -C $CAND status --short | tr '\n' ' ')"
bash $A/scripts/build_1003.sh $CAND $A/spine-lisa-2 spine-lisa-2 --was-person-tab $U/was_2006_22/was_round_8_person_eul_may_2025_230525.tab > $A/logs/spine-lisa-2.build.log 2>&1; echo "rebuild exit $?"
cd "$TREE"
.venv/bin/python tools/compare_uk_h5_payload.py "$A/spine-lisa/spine-lisa.h5" "$A/spine-lisa-2/spine-lisa-2.h5" --json-out "$OUT/payload_diff.json" > "$OUT/compare.log" 2>&1; echo "compare exit $?"
.venv/bin/python -W ignore - "$A" "$OUT" <<'PY'
import sys, json, pandas as pd
A, OUT = sys.argv[1], sys.argv[2]
one = pd.HDFStore(f"{A}/spine-lisa/spine-lisa.h5", "r"); two = pd.HDFStore(f"{A}/spine-lisa-2/spine-lisa-2.h5", "r")
report = {"store_keys_equal": list(one.keys()) == list(two.keys()), "keys": {}}
for key in one.keys():
    report["keys"][key] = bool(one[key].equals(two[key]))
report["mass_log_equal"] = one._handle.root._v_attrs.populace_mass_log_json == two._handle.root._v_attrs.populace_mass_log_json
one.close(); two.close()
e1 = json.load(open(f"{A}/spine-lisa/spine-lisa.build.json"))["stage_evidence"]["was_lisa"]
e2 = json.load(open(f"{A}/spine-lisa-2/spine-lisa-2.build.json"))["stage_evidence"]["was_lisa"]
report["was_lisa_evidence_equal"] = e1 == e2
report["verdict"] = bool(report["store_keys_equal"] and all(report["keys"].values()) and report["mass_log_equal"] and report["was_lisa_evidence_equal"])
json.dump(report, open(f"{OUT}/determinism.json", "w"), indent=2); print(json.dumps(report, indent=1))
PY
echo "determinism exit $?"
