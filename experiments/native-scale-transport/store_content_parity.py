"""Compare two graph stores by content rather than by address.

Node keys are the store's addresses, and every US stage's implementation hash is
over its whole module roster, so any edit to an inventoried module moves every
address. What must not move is the content: the column, mask, index and payload
bytes each node wrote. This walks both stores, groups every payload file by its
own sha256, and reports what each store has that the other does not.

``meta.json`` and cache-record JSON are reported separately, because they embed
node keys by construction and are expected to differ.

Reads only; writes only the named output file. Not a build, not a certification,
not release eligible.

    <venv>/bin/python store_content_parity.py <store-a> <store-b> <out.json>
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

_RECORD_NAMES = frozenset({"meta.json", "record.json"})


def _inventory(root: pathlib.Path):
    payloads = collections.Counter()
    records = collections.Counter()
    names = collections.Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        names[path.name] += 1
        if path.name in _RECORD_NAMES or path.suffix == ".json":
            records[digest] += 1
        else:
            payloads[digest] += 1
    return payloads, records, names


def main():
    left_root, right_root, out = (pathlib.Path(value) for value in sys.argv[1:4])
    left, left_records, left_names = _inventory(left_root)
    right, right_records, right_names = _inventory(right_root)
    only_left = left - right
    only_right = right - left
    result = {
        "left": str(left_root),
        "right": str(right_root),
        "payload_files": {"left": sum(left.values()), "right": sum(right.values())},
        "distinct_payload_digests": {"left": len(left), "right": len(right)},
        "payload_digests_shared": len(set(left) & set(right)),
        "payload_digests_only_left": len(only_left),
        "payload_digests_only_right": len(only_right),
        "payload_content_identical": not only_left and not only_right,
        "json_files": {
            "left": sum(left_records.values()),
            "right": sum(right_records.values()),
        },
        "json_digests_shared": len(set(left_records) & set(right_records)),
        "json_digests_only_left": len(set(left_records) - set(right_records)),
        "json_digests_only_right": len(set(right_records) - set(left_records)),
        "file_names": {
            name: {"left": left_names.get(name, 0), "right": right_names.get(name, 0)}
            for name in sorted(set(left_names) | set(right_names))
        },
        "release_eligible": False,
        "scope": (
            "Content parity of two graph stores, by payload digest rather than "
            "by address. Not a build, not a certification, not release eligible."
        ),
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_names"}, indent=2))
    print("file names:", json.dumps(result["file_names"]))


if __name__ == "__main__":
    main()
