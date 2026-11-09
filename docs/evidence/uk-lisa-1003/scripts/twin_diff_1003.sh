#!/usr/bin/env bash
# Twin diff spine-ctl (main) vs spine-lisa (the was_lisa branch): payload compare, classification,
# and the adjudication of the two undeclarable consequences (appended columns move the column
# order; the stage's mass receipt joins the root mass log).
set -uo pipefail
TREE=/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1003
A=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1003-lisa
OUT=$A/diff-spine-lisa-vs-spine-ctl; mkdir -p "$OUT"; cd "$TREE"
.venv/bin/python tools/compare_uk_h5_payload.py "$A/spine-ctl/spine-ctl.h5" "$A/spine-lisa/spine-lisa.h5" --json-out "$OUT/payload_diff.json" > "$OUT/compare.log" 2>&1
echo "compare exit $?"
.venv/bin/python tools/classify_uk_payload_diff.py "$OUT/payload_diff.json" "$A/spine-lisa-payload-expectation.json" --json-out "$OUT/classified.json" > "$OUT/classify.log" 2>&1
echo "classify exit $?"
.venv/bin/python -W ignore - "$A" "$OUT" <<'PY'
import sys, json, pandas as pd
from microcosm.build.uk_runtime.was_lisa import UK_WAS_LISA_MASS_CONSERVATION_REASON
A, OUT = sys.argv[1], sys.argv[2]
new = {"person": ["has_lifetime_isa", "lifetime_isa_balance"], "household": ["household_lifetime_isa_balance"]}
c = json.load(open(f"{OUT}/classified.json"))
report = {"classify_summary": c["summary"], "unexpected": [(u["entity"], u["column"], u["surface"]) for u in c["unexpected"]]}
ctl = pd.HDFStore(f"{A}/spine-ctl/spine-ctl.h5", "r"); cand = pd.HDFStore(f"{A}/spine-lisa/spine-lisa.h5", "r")
report["store_keys_equal"] = list(ctl.keys()) == list(cand.keys())
for key in ctl.keys():
    a, b = ctl[key], cand[key]
    entity = key.strip("/").split("/")[-1]
    if isinstance(b, pd.DataFrame):
        extra = [col for col in b.columns if col not in a.columns]
        report[key] = {
            "new_columns": extra,
            "new_columns_as_expected": sorted(extra) == sorted(new.get(entity, [])),
            "new_columns_appended_last": list(b.columns[len(a.columns):]) == extra,
            "rest_equals_control": bool(b.drop(columns=extra).equals(a)),
            "dtypes": {col: str(b[col].dtype) for col in extra},
        }
    else:
        report[key] = {"equals_control": bool(b.equals(a))}
left = json.loads(ctl._handle.root._v_attrs.populace_mass_log_json)
right = json.loads(cand._handle.root._v_attrs.populace_mass_log_json)
ctl.close(); cand.close()
added = [r for r in right if r not in left]
report["mass_log"] = {
    "control_records": len(left), "candidate_records": len(right),
    "control_is_prefix_or_subsequence": all(r in right for r in left),
    "added_records": len(added),
    "added_is_the_was_lisa_receipt": len(added) == 1 and added[0].get("reason") == UK_WAS_LISA_MASS_CONSERVATION_REASON,
    "added_conserved": len(added) == 1 and added[0].get("old_total") == added[0].get("new_total"),
}
only_expected_unexpected = all(u[1] == "__column_order__" and u[0] in new for u in report["unexpected"])
report["verdict"] = bool(only_expected_unexpected and report["store_keys_equal"] and all(v.get("rest_equals_control", v.get("equals_control", False)) and v.get("new_columns_as_expected", True) for k, v in report.items() if k.startswith("/")) and report["mass_log"]["added_is_the_was_lisa_receipt"] and report["mass_log"]["added_conserved"] and report["mass_log"]["control_is_prefix_or_subsequence"])
json.dump(report, open(f"{OUT}/adjudication.json", "w"), indent=2); print(json.dumps(report, indent=1))
PY
