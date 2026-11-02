"""Compare two US Chronicle feeds row by row (microcosm#454 re-pin).

    uv run python experiments/us-esi-454/feed-repin/compare_feeds.py \
        <previous consumer_facts.jsonl> <new consumer_facts.jsonl> <report.json>

Joins on ``aggregate_fact_key`` and compares each row's exact bytes. The
re-pin is accepted only if no previous row is missing or changed; the report
lists every added row's record set, period, value and source digest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _rows(path: Path) -> dict[str, bytes]:
    rows: dict[str, bytes] = {}
    with path.open("rb") as stream:
        for line in stream:
            key = json.loads(line)["aggregate_fact_key"]
            if key in rows:
                raise SystemExit(f"{path}: repeated aggregate_fact_key {key}")
            rows[key] = line
    return rows


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    previous_path, new_path, report_path = (Path(arg) for arg in sys.argv[1:4])
    previous, new = _rows(previous_path), _rows(new_path)
    missing = sorted(key for key in previous if key not in new)
    changed = sorted(
        key for key in previous if key in new and previous[key] != new[key]
    )
    added = []
    for key in sorted(set(new) - set(previous)):
        row = json.loads(new[key])
        added.append(
            {
                "aggregate_fact_key": key,
                "record_set_id": row["layout"]["record_set_id"],
                "period": row["period"],
                "value": row["value"],
                "source_sha256": row["source"].get("source_sha256"),
            }
        )
    report = {
        "previous": {
            "file": previous_path.name,
            "sha256": _sha256(previous_path),
            "rows": len(previous),
        },
        "new": {"file": new_path.name, "sha256": _sha256(new_path), "rows": len(new)},
        "previous_rows_missing": len(missing),
        "previous_rows_changed": len(changed),
        "rows_added": added,
    }
    Path(report_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows_added"}, indent=1))
    if missing or changed:
        raise SystemExit("previous rows missing or changed")


if __name__ == "__main__":
    main()
