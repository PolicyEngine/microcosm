"""List the rows a re-pin adds, with the source that emits each one.

Usage:
    uv run python experiments/us-feed-ed-fsa/list_added_rows.py \\
        <compare.json> <new feed jsonl> <out.json>

``compare_feeds.py`` (experiments/us-feed-agi-age-labels) records the keys a
new feed adds (``rows.keys_only_in_new``) but not where they come from. This
reads those keys back out of the new feed and writes, for each, its
``source.source_name``, record set and period, plus counts by source and by
record set. ``test_the_committed_feed_pin_banks_the_anchor`` reads the result
to check that a re-pin adds rows from its declared source and nothing else.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    compare_path, new_path, out_path = sys.argv[1:4]
    report = json.loads(Path(compare_path).read_text())
    if file_sha256(new_path) != report["feeds"]["new"]["sha256"]:
        raise SystemExit(f"{new_path} is not the report's new feed")
    added_keys = set(report["rows"]["keys_only_in_new"])
    rows = []
    for line in open(new_path, "rb"):
        row = json.loads(line)
        if row["aggregate_fact_key"] in added_keys:
            rows.append(
                {
                    "aggregate_fact_key": row["aggregate_fact_key"],
                    "source_name": row["source"]["source_name"],
                    "record_set_id": row["layout"]["record_set_id"],
                    "period": row["period"],
                }
            )
    if {row["aggregate_fact_key"] for row in rows} != added_keys:
        raise SystemExit("the new feed lacks some of the report's added keys")
    by_record_set = collections.Counter(row["record_set_id"] for row in rows)
    payload = {
        "compare_report": Path(compare_path).name,
        "new_feed": report["feeds"]["new"],
        "row_count": len(rows),
        "by_source": dict(
            sorted(collections.Counter(r["source_name"] for r in rows).items())
        ),
        "by_record_set": dict(sorted(by_record_set.items())),
        "rows": rows,
    }
    Path(out_path).write_text(json.dumps(payload, indent=1) + "\n")
    print(
        json.dumps(
            {k: payload[k] for k in ("row_count", "by_source")}
            | {"record_sets": len(by_record_set)},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
