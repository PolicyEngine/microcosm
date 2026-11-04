"""Rebuild #1345's feed: drop rows in the scope pairs the new scope adds."""
import json, sys
new_scope, old_scope, feed, out = sys.argv[1:5]
def pairs(p):
    s = json.load(open(p))
    return {(e["record_set_id"], e["period_type"], str(e["period_value"])) for e in s["pairs"]}
added = pairs(new_scope) - pairs(old_scope)
print("added pairs", len(added))
n = k = 0
with open(feed, "rb") as f, open(out, "wb") as o:
    for line in f:
        r = json.loads(line)
        key = (r["layout"]["record_set_id"], r["period"]["type"], str(r["period"]["value"]))
        if key in added:
            n += 1
        else:
            o.write(line); k += 1
print("dropped", n, "kept", k)
