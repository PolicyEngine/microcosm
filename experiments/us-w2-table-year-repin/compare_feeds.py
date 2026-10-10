"""What moving the US Chronicle feed pin changed, row by row.

Usage: python compare_feeds.py <old feed jsonl> <new feed jsonl> <out.json>

Rows are compared as bytes, then by ``lineage.source_record_id``: ids only in
the old feed, ids only in the new feed, and ids in both whose bytes differ.
For each new-only id the report names the old-only id with the same record
set family, source row and source column (its twin under another year stamp)
and lists the fields that differ between the two.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

old_path, new_path, out_path = sys.argv[1:4]
PERIOD_TOKEN = re.compile(r"\.(?:ty|cy|fy)\d{4}(?=\.)")


def load(path: str) -> tuple[dict[str, bytes], str, int]:
    payload = Path(path).read_bytes()
    rows: dict[str, bytes] = {}
    for line in payload.splitlines(keepends=True):
        if not line.strip():
            continue
        source_record_id = json.loads(line)["lineage"]["source_record_id"]
        assert source_record_id not in rows, source_record_id
        rows[source_record_id] = line
    return rows, hashlib.sha256(payload).hexdigest(), len(payload)


def flat(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(flat(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}


def cell(row: dict) -> tuple[str, str, str]:
    layout = row["layout"]
    return (
        PERIOD_TOKEN.sub("", "." + layout["record_set_id"]).lstrip("."),
        str(layout.get("source_row_id")),
        str(layout.get("source_column_id")),
    )


old, old_sha256, old_bytes = load(old_path)
new, new_sha256, new_bytes = load(new_path)
only_old = sorted(set(old) - set(new))
only_new = sorted(set(new) - set(old))
changed = sorted(i for i in set(old) & set(new) if old[i] != new[i])
old_cells = {cell(json.loads(old[i])): i for i in only_old}
twins = {}
for source_record_id in only_new:
    row = json.loads(new[source_record_id])
    twin_id = old_cells.get(cell(row))
    entry: dict[str, object] = {"old_source_record_id": twin_id}
    if twin_id is not None:
        before, after = flat(json.loads(old[twin_id])), flat(row)
        entry["value_equal"] = before["value"] == after["value"]
        entry["source_cell_keys_equal"] = (
            json.loads(old[twin_id])["lineage"]["source_cell_keys"]
            == row["lineage"]["source_cell_keys"]
        )
        entry["fields_changed"] = {
            key: [before.get(key), after.get(key)]
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
    twins[source_record_id] = entry
report = {
    "old": {"rows": len(old), "facts_sha256": old_sha256, "bytes": old_bytes},
    "new": {"rows": len(new), "facts_sha256": new_sha256, "bytes": new_bytes},
    "rows_byte_identical": len(set(old) & set(new)) - len(changed),
    "ids_in_both_with_different_bytes": changed,
    "ids_only_in_old": only_old,
    "ids_only_in_new": only_new,
    "old_only_ids_with_no_new_twin": sorted(
        set(only_old) - {entry["old_source_record_id"] for entry in twins.values()}
    ),
    "new_only_rows": twins,
    "new_feed_sorted_by_aggregate_fact_key": [
        json.loads(line)["aggregate_fact_key"] for line in new.values()
    ]
    == sorted(json.loads(line)["aggregate_fact_key"] for line in new.values()),
}
Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
print(
    json.dumps(
        {
            key: (len(value) if isinstance(value, list | dict) else value)
            for key, value in report.items()
            if key not in {"old", "new"}
        },
        indent=1,
    )
)
