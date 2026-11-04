"""Step 4: compile the default US registry on a feed the way
tools/build_us_fiscal_refresh_release.py:12793-12858 does, then run the
release's parity checks. Usage: python default_compile.py <repo root> <feed>
[--student-loan-targets]"""
import collections, importlib.util, json, sys, resource, time
from pathlib import Path

root, feed = Path(sys.argv[1]), sys.argv[2]
student = "--student-loan-targets" in sys.argv
sys.path.insert(0, str(root / "tools"))
spec = importlib.util.spec_from_file_location("rel", root / "tools" / "build_us_fiscal_refresh_release.py")
rel = importlib.util.module_from_spec(spec); spec.loader.exec_module(rel)
from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
from microcosm.build.us_runtime.release_target_parity import us_target_family_id
t0 = time.time()
pin = load_us_chronicle_feed()
art = rel.load_ledger_consumer_artifact(feed, expected_facts_sha256=pin.facts_sha256)
rel._check_committed_us_ledger_feed_pin(art.facts_sha256, manifest_sha256=art.manifest_sha256, allow_unpinned_feed=False)
esi = rel._check_us_esi_anchor_against_feed(art.facts, allow_unpinned_feed=False, allow_esi_premium_gaps=False)
print("esi anchor mismatches:", esi)
xwalk = rel.load_congressional_district_vintage_crosswalk(rel.default_congressional_district_vintage_crosswalk_path())
kw = dict(target_period=rel.PERIOD, congressional_district_vintage_crosswalk=xwalk, age_targets=True, allow_unaged_dollar_targets=False)
if student:
    kw["student_loan_targets"] = True
reg = rel.compile_us_fiscal_target_registry(art.facts, **kw)
reg, subs = rel.apply_us_medicaid_enrollment_substitutions(reg)
names = [s.name for s in reg.specs]
fams = collections.Counter(us_target_family_id(n) for n in names if us_target_family_id(n))
print("PERIOD", rel.PERIOD, "targets", len(names), "families", len(fams))
print("registry version", getattr(reg, "version", None))
fsa = [n for n in names if "ed_fsa" in n or "student_loan" in n]
print("ed_fsa/student_loan target names:", len(fsa), fsa[:5])
print("ed_fsa families:", {f: c for f, c in fams.items() if f.startswith("ed_fsa")})
# any spec metadata referencing ed_fsa
hits = 0
for s in reg.specs:
    blob = " ".join(str(x) for x in (s.name, s.source, s.family, s.notes, dict(s.metadata), s.filter, s.measure))
    if "ed_fsa" in blob or "Federal Student Aid" in blob:
        hits += 1
print("specs whose name/source/family/notes/metadata/filter/measure mention ed_fsa or Federal Student Aid:", hits)
print("distinct spec.source values (first 40):", sorted({s.source for s in reg.specs})[:40])
out = {"targets": len(names), "families": dict(sorted(fams.items())), "names_sha": __import__("hashlib").sha256("\n".join(names).encode()).hexdigest()}
if not student:
    rel.assert_target_parity_manifest_current(registry=reg)
    print("assert_target_parity_manifest_current(registry=...): passed")
    g = rel.us_release_target_parity_gate(reg)
    print("us_release_target_parity_gate:", g.passed, g.details, g.failures)
print(json.dumps(out)[:400])
print("seconds", round(time.time() - t0), "maxrss_GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
