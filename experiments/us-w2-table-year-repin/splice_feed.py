"""Predict the re-pinned feed from the old pin and one package run.

Usage: python splice_feed.py <old feed jsonl> <package consumer_facts.jsonl> <out jsonl>

``docs/us-fact-to-target.md`` (step 1) describes a new feed as the previous
feed plus the new rows, deduplicated on ``lineage.source_record_id``. This
applies that rule to the W-2 item package: drop every old row the
``soi-w2-statistics-2020`` package emitted, add the rows the package emits at
the new commit, and sort by ``aggregate_fact_key`` as the feed builder does.
The result is an independent prediction of what
``tools/build_us_chronicle_feed.py`` writes from the new scope; the two must
be byte-identical (see README.md).
"""

from __future__ import annotations

import hashlib
import json
import sys

W2_SOURCE_FILE = "20in04w2all.xlsx"

old_path, package_path, out_path = sys.argv[1:4]
kept: dict[str, bytes] = {}
dropped: list[str] = []
with open(old_path, "rb") as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        if row["source"].get("source_file") == W2_SOURCE_FILE:
            dropped.append(row["lineage"]["source_record_id"])
            continue
        kept[row["aggregate_fact_key"]] = line
added: list[str] = []
seen_ids: set[str] = set()
with open(package_path, "rb") as stream:
    for raw in stream:
        if not raw.strip():
            continue
        line = raw if raw.endswith(b"\n") else raw + b"\n"
        row = json.loads(line)
        source_record_id = row["lineage"]["source_record_id"]
        assert source_record_id not in seen_ids, source_record_id
        seen_ids.add(source_record_id)
        assert row["aggregate_fact_key"] not in kept, source_record_id
        kept[row["aggregate_fact_key"]] = line
        added.append(source_record_id)
payload = b"".join(kept[key] for key in sorted(kept))
with open(out_path, "wb") as stream:
    stream.write(payload)
print(
    json.dumps(
        {
            "rows": len(kept),
            "facts_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "dropped": sorted(dropped),
            "added": sorted(added),
            "superseded_without_replacement": sorted(set(dropped) - set(added)),
        },
        indent=1,
    )
)
