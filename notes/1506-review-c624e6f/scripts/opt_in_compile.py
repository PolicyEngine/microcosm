"""Step 7: on the #1506+#1357 scratch merge, compile the US registry on a feed
as #1357's release tool does (tools/build_us_fiscal_refresh_release.py at
0ff293b, lines 13008-13062), with student_loan_targets True and False.
Usage: python opt_in_compile.py <merge root> <feed jsonl> <out receipt json> [<fixture.jsonl.gz>]"""
import collections, gzip, importlib.util, json, sys
from pathlib import Path

root, feed, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("rel", root / "tools" / "build_us_fiscal_refresh_release.py")
rel = importlib.util.module_from_spec(spec); spec.loader.exec_module(rel)
from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
from microcosm.build.us_runtime.fsa_student_loan_targets import FSA_TARGET_FAMILY, FSA_MEASURE_KEY, fsa_student_loan_target_receipt
pin = load_us_chronicle_feed()
art = rel.load_ledger_consumer_artifact(feed, expected_facts_sha256=pin.facts_sha256)
rel._check_committed_us_ledger_feed_pin(art.facts_sha256, manifest_sha256=art.manifest_sha256, allow_unpinned_feed=False)
print("pin check passed; feed rows", art.fact_row_count)
xwalk = rel.load_congressional_district_vintage_crosswalk(rel.default_congressional_district_vintage_crosswalk_path())

def compile_(flag):
    reg = rel.compile_us_fiscal_target_registry(art.facts, target_period=rel.PERIOD, congressional_district_vintage_crosswalk=xwalk, age_targets=True, allow_unaged_dollar_targets=False, student_loan_targets=flag)
    reg, _ = rel.apply_us_medicaid_enrollment_substitutions(reg)
    return reg

for flag in (True, False):
    reg = compile_(flag)
    fsa = [s for s in reg.specs if s.family == FSA_TARGET_FAMILY]
    by = collections.Counter(s.metadata.get(FSA_MEASURE_KEY) for s in fsa)
    print(f"student_loan_targets={flag}: targets {len(reg.specs)}, FSA {len(fsa)}, by measure {dict(sorted(by.items()))}")
    parity = rel._without_opt_in_student_loan_targets(reg)
    print(f"  parity registry: {len(parity.specs)} targets")
    rel.assert_target_parity_manifest_current(registry=parity)
    g = rel.us_release_target_parity_gate(parity)
    print("  assert_target_parity_manifest_current passed; gate", g.passed, g.details)
    if flag:
        try:
            rel.assert_target_parity_manifest_current(registry=reg)
            print("  unfiltered registry: parity assert PASSED")
        except ValueError as exc:
            print("  unfiltered registry: parity assert refuses:", str(exc).splitlines()[1][:200])
        fsa_specs = {s.name: (s.value, s.period, dict(s.metadata)) for s in fsa}

receipt = fsa_student_loan_target_receipt(art.facts, target_period=rel.PERIOD)
print("receipt status", receipt["status"], "reference_period", receipt["reference_period"], "facts", len(receipt["dispositions"]))
print("disposition_counts", receipt["disposition_counts"])
out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
if len(sys.argv) > 4:
    rows = [json.loads(l) for l in gzip.open(sys.argv[4], "rt")]
    r2 = fsa_student_loan_target_receipt(rows, target_period=rel.PERIOD)
    print("receipt from fixture", sys.argv[4], "equals receipt from feed:", r2 == receipt)
    if r2 != receipt:
        print("  differing keys:", [k for k in receipt if receipt[k] != r2.get(k)])
        print("  fixture disposition_counts", r2["disposition_counts"])
        same_targets = r2["targets"] == receipt["targets"] and r2["controls"] == receipt["controls"]
        print("  fixture targets and controls equal the feed's:", same_targets)
