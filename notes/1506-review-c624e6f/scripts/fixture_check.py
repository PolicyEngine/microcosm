"""Step 7 fixture claims. Usage: python fixture_check.py <fixture.jsonl.gz> <ed_fsa_rows.jsonl> <out v3 fixture .jsonl.gz>"""
import gzip, hashlib, json, sys
fx_path, notes_path, out = sys.argv[1:4]
V3, V4 = b'"chronicle.consumer_fact.v3"', b'"chronicle.consumer_fact.v4"'
raw = open(fx_path, "rb").read()
print("fixture gz sha256", hashlib.sha256(raw).hexdigest())
fx = gzip.decompress(raw).splitlines(keepends=True)
notes = open(notes_path, "rb").read().splitlines(keepends=True)
print("fixture rows", len(fx), "notes rows", len(notes))
print("fixture stamps", {json.loads(l)["schema_version"] for l in fx})
for l in notes: assert l.count(V3) == 1, "notes row stamp not once"
for l in fx: assert l.count(V4) == 1, "fixture row stamp not once"
notes_set = set(notes)
print("fixture rows byte-identical to a notes row:", sum(l in notes_set for l in fx))
notes_v4 = {l.replace(V3, V4) for l in notes}
print("fixture rows byte-identical to a notes row restamped v4:", sum(l in notes_v4 for l in fx))
by_key = {json.loads(l)["aggregate_fact_key"]: l for l in notes}
diff_only_stamp = 0
for l in fx:
    a = json.loads(l); b = json.loads(by_key[a["aggregate_fact_key"]])
    a.pop("schema_version"); b.pop("schema_version")
    diff_only_stamp += a == b
print("fixture rows sharing a key with a notes row and differing only in schema_version:", diff_only_stamp)
print("notes rows the fixture lacks:", len(notes) - len({json.loads(l)["aggregate_fact_key"] for l in fx} & set(by_key)))
v3 = [l.replace(V4, V3) for l in fx]
print("restamped-v3 fixture rows byte-identical to a notes row:", sum(l in notes_set for l in v3))
with gzip.open(out, "wb") as g: g.writelines(v3)
print("wrote", out)
