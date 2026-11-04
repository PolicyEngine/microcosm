"""Receipt and compiled FSA plan from the v4 fixture vs the v3-restamped fixture.
Usage: python receipt_v3_v4.py <v4.jsonl.gz> <v3.jsonl.gz>"""
import gzip, json, sys
from microcosm.build.us_runtime.fsa_student_loan_targets import fsa_student_loan_target_receipt, plan_fsa_student_loan_targets
r = {}
for tag, p in zip(("v4", "v3"), sys.argv[1:3]):
    rows = [json.loads(l) for l in gzip.open(p, "rt")]
    r[tag] = fsa_student_loan_target_receipt(rows, target_period=2024)
    plan = plan_fsa_student_loan_targets(rows, target_period=2024)
    r[tag + "_specs"] = (plan.reference_period, plan.values, plan.metadata, plan.labels, plan.controls, [repr(x) for x in plan.references])
print("receipt v3 == v4:", r["v3"] == r["v4"], "| FSA plan (references, values, metadata, labels, controls) v3 == v4:", r["v3_specs"] == r["v4_specs"], "| planned targets:", len(r["v4_specs"][1]))
