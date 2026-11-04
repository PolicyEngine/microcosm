"""Step 6: run test_us_esi_premiums.py's report-chain checks against mutated
scratch copies of the committed reports, and name the assert that fires.

Usage: python chain_mutations.py <repo root> <scratch dir>
The repository is never written; mutated copies live under <scratch dir>.
"""
import copy, importlib.util, json, shutil, sys, traceback
from pathlib import Path

repo, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
test_path = repo / "packages/microcosm-build/tests/engine_free/us/test_us_esi_premiums.py"
spec = importlib.util.spec_from_file_location("t_esi", test_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
src_lines = test_path.read_text().splitlines()

def where(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    frame = [f for f in tb if f.filename == str(test_path)][-1]
    return f"{type(exc).__name__} at test_us_esi_premiums.py:{frame.lineno}: {src_lines[frame.lineno-1].strip()}  | msg={str(exc)[:160]!r}"

def run_chain(label, mutate_compare=None, mutate_added=None, reports=None):
    root = scratch / "chain_root" / label
    if root.exists(): shutil.rmtree(root)
    for rel in ("experiments/us-esi-454/feed-repin/value_diff.json",
                "experiments/us-feed-agi-age-labels/compare.json",
                "experiments/us-feed-ed-fsa/compare.json",
                "experiments/us-feed-ed-fsa/added_rows.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / rel, root / rel)
    for rel, fn in (("experiments/us-feed-ed-fsa/compare.json", mutate_compare),
                    ("experiments/us-feed-ed-fsa/added_rows.json", mutate_added)):
        if fn:
            d = json.loads((root / rel).read_text()); fn(d)
            (root / rel).write_text(json.dumps(d, indent=1) + "\n")
    saved = (mod._REPOSITORY_ROOT, mod._LATER_ROW_KEEPING_REPIN_REPORTS)
    mod._REPOSITORY_ROOT = root
    if reports is not None:
        mod._LATER_ROW_KEEPING_REPIN_REPORTS = reports
    try:
        mod.TestAnchorFeedCheck().test_the_committed_feed_pin_banks_the_anchor()
        print(f"[{label}] PASSED (not refused)")
    except Exception as exc:
        print(f"[{label}] REFUSED: {where(exc)}")
    finally:
        mod._REPOSITORY_ROOT, mod._LATER_ROW_KEEPING_REPIN_REPORTS = saved

print("== committed chain, unmutated")
run_chain("baseline")

print("== mutations of the committed ed_fsa report and listing")
NEWKEY = "aggregate_fact:" + "f" * 64
def add_non_fsa(c):  # a re-pin that also added an irs_soi row, honestly listed
    c["rows"]["keys_only_in_new"] = sorted(c["rows"]["keys_only_in_new"] + [NEWKEY]); c["rows"]["new_rows"] += 1
def add_non_fsa_listing(a):
    a["rows"].append({"aggregate_fact_key": NEWKEY, "source_name": "irs_soi", "record_set_id": "irs_soi.ty2023.table_1_4", "period": {"type": "tax_year", "value": 2023}})
run_chain("add-non-ed_fsa-row (listed honestly)", add_non_fsa, add_non_fsa_listing)
run_chain("add-non-ed_fsa-row (not listed)", add_non_fsa, None)
def mislabel(a):  # the same extra row, listed but mislabelled as ed_fsa
    a["rows"].append({"aggregate_fact_key": NEWKEY, "source_name": "ed_fsa", "record_set_id": "irs_soi.ty2023.table_1_4", "period": {"type": "tax_year", "value": 2023}})
run_chain("add-non-ed_fsa-row (listing claims ed_fsa)", add_non_fsa, mislabel)

def drop_listed(a): a["rows"].pop(0)
run_chain("drop-a-listed-row (from added_rows.json)", None, drop_listed)
def drop_from_report(c):  # the report no longer adds one of the listed keys
    c["rows"]["keys_only_in_new"].pop(0); c["rows"]["new_rows"] -= 1
run_chain("drop-a-listed-row (from compare.json keys_only_in_new)", drop_from_report, None)
def drop_kept(c):  # a kept row disappears from the new feed
    c["rows"]["keys_only_in_old"] = ["aggregate_fact:" + "0" * 64]; c["rows"]["new_rows"] -= 1
run_chain("drop-a-kept-row", drop_kept, None)

def change_value(c):
    c["rows"]["changed_rows"] = 1; c["rows"]["changed_fields"] = {"value": {"rows": 1, "by_record_set": {"irs_soi.ty2023.table_1_4": 1}}}
run_chain("change-a-kept-row (value)", change_value, None)
def change_label_anchor(c):
    c["rows"]["changed_rows"] = 1; c["rows"]["changed_fields"] = {"layout/groupby_value_label": {"rows": 1, "by_record_set": {"bea_nipa.cy2024.private_group_health_insurance": 1}}}
run_chain("change-a-kept-row (label, in an anchor record set)", change_label_anchor, None)
def change_label_other(c):
    c["rows"]["changed_rows"] = 1; c["rows"]["changed_fields"] = {"layout/groupby_value_label": {"rows": 1, "by_record_set": {"irs_soi.ty2023.table_1_4": 1}}}
run_chain("change-a-kept-row (label, elsewhere; allowed by design)", change_label_other, None)
def change_source(c):
    c["rows"]["changed_rows"] = 1; c["rows"]["changed_fields"] = {"source/source_name": {"rows": 1, "by_record_set": {"irs_soi.ty2023.table_1_4": 1}}}
run_chain("change-a-kept-row (source field)", change_source, None)

base = mod._LATER_ROW_KEEPING_REPIN_REPORTS
print("== declarations")
run_chain("declare-second-source (tuple of two sources)", reports=(base[0], ("experiments/us-feed-ed-fsa/compare.json", ("experiments/us-feed-ed-fsa/added_rows.json", ("ed_fsa", "irs_soi")))))
run_chain("declare-second-source (three-element declaration)", reports=(base[0], ("experiments/us-feed-ed-fsa/compare.json", ("experiments/us-feed-ed-fsa/added_rows.json", "ed_fsa", "irs_soi"))))
run_chain("declare-wrong-source (irs_soi)", reports=(base[0], ("experiments/us-feed-ed-fsa/compare.json", ("experiments/us-feed-ed-fsa/added_rows.json", "irs_soi"))))
run_chain("declare-nothing (None)", reports=(base[0], ("experiments/us-feed-ed-fsa/compare.json", None)))
run_chain("drop-the-ed_fsa-report", reports=(base[0],))
def dup_listing(a): a["rows"].append(dict(a["rows"][0]))
run_chain("listing-repeats-a-key", None, dup_listing)
def other_newfeed(a): a["new_feed"]["sha256"] = "0" * 64
run_chain("listing-for-another-feed", None, other_newfeed)

print("== the PR's parametrized unit cases: which assert fires")
T = mod.TestAnchorFeedCheck
cases = {
 "undeclared-additions": (["k1"], None, None),
 "other-source": (["k1","k2"], [("k1","ed_fsa","ed_fsa.a"),("k2","irs_soi","x")], "ed_fsa"),
 "listing-short": (["k1","k2"], [("k1","ed_fsa","ed_fsa.a")], "ed_fsa"),
 "listing-extra": (["k1"], [("k1","ed_fsa","a"),("k9","ed_fsa","b")], "ed_fsa"),
 "anchor-record-set": (["k1"], [("k1","ed_fsa","bea_nipa.cy2024.private_group_health_insurance")], "ed_fsa"),
 "no-source": (["k1"], [("k1","ed_fsa","ed_fsa.a")], None),
}
for cid, (keys, rows, src) in cases.items():
    later = T._later_report(added_keys=keys)
    try:
        mod._follow_later_repin_report(later, feed_sha256="old", feed_rows=10, anchor_record_sets=T._ANCHORS, added=None if rows is None else T._added(rows), declared_source=src)
        print(f"[{cid}] NOT REFUSED")
    except AssertionError as exc:
        print(f"[{cid}] {where(exc)}")
muts = {
 "dropped-row": lambda r: r["rows"].update(keys_only_in_old=["k0"]),
 "row-count": lambda r: r["rows"].update(new_rows=r["rows"]["new_rows"] + 1),
 "value-change": lambda r: r["rows"]["changed_fields"].update(value={"rows": 1, "by_record_set": {"x": 1}}),
 "wrong-base": lambda r: r["feeds"]["old"].update(sha256="other"),
}
for cid, m in muts.items():
    later = T._later_report(added_keys=["k1"]); m(later)
    try:
        mod._follow_later_repin_report(later, feed_sha256="old", feed_rows=10, anchor_record_sets=T._ANCHORS, added=T._added([("k1","ed_fsa","ed_fsa.a")]), declared_source="ed_fsa")
        print(f"[{cid}] NOT REFUSED")
    except AssertionError as exc:
        print(f"[{cid}] {where(exc)}")

print("== the mislabelled listing again, with a pin that agrees with the report (a re-pin that did add the row)")
pin = json.loads((repo / "packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json").read_text())
pin["fact_row_count"] += 1
fake = scratch / "fake_pkg"; fake.mkdir(parents=True, exist_ok=True)
(fake / "chronicle_feed.json").write_text(json.dumps(pin))
real_files = mod.files
mod.files = lambda name: fake if name == "microcosm.build.us" else real_files(name)
try:
    run_chain("add-non-ed_fsa-row (listing claims ed_fsa, pin row count +1)", add_non_fsa, mislabel)
finally:
    mod.files = real_files
