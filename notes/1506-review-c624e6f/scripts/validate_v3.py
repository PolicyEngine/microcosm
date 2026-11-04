"""Validate every feed row against Chronicle's consumer_fact.v3 schema file
(Draft 2020-12, as policyengine_chronicle/schema.py's _validator does).
This checks only the schema half of build-consumer-artifact, not the command.
Usage: python validate_v3.py <schema.json> <feed.jsonl>"""
import collections, json, sys
from jsonschema import Draft202012Validator
v = Draft202012Validator(json.load(open(sys.argv[1])))
bad = collections.Counter(); first = None; n = 0
for i, line in enumerate(open(sys.argv[2]), 1):
    row = json.loads(line); n += 1
    errs = list(v.iter_errors(row))
    if errs:
        e = errs[0]
        bad[(row["source"]["source_name"], "/".join(map(str, e.absolute_path)), e.message[:80])] += 1
        first = first or (i, e.message[:120])
print("rows", n, "invalid", sum(bad.values()), "first invalid", first)
for k, c in bad.most_common(): print(c, k)
