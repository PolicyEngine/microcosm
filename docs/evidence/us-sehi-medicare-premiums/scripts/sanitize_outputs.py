"""Copy a combined output into this folder without single-record values.

The combined outputs are weighted aggregates and counts, except for a few
maxima of person-level differences (``max_abs_change``), which are one
record's value. This drops every nonzero maximum; a zero maximum (an exact
equality check) is kept.
"""

import json
import sys


def scrub(node):
    if isinstance(node, dict):
        return {
            key: scrub(value)
            for key, value in node.items()
            if not ("max_abs" in key and isinstance(value, (int, float)) and value != 0)
        }
    if isinstance(node, list):
        return [scrub(value) for value in node]
    return node


source, target = sys.argv[1], sys.argv[2]
with open(source) as handle:
    data = json.load(handle)
with open(target, "w") as handle:
    json.dump(scrub(data), handle, indent=1)
