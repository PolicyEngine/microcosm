"""Record the exact eight-pair addition to the 41b2fb2 US Chronicle feed.

Usage: python compare_feeds.py <old.jsonl> <new.jsonl> <compare.json>

Keep publisher bytes intact: remove only the eight reviewed pairs from the
new feed, then compare every remaining line with the old feed in order.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

NEW_PAIRS = {
    **{
        (f"census_acs.acs1_2024.{table}.{geography}", "calendar_year", "2024"): (
            "census_acs"
        )
        for table in ("b11001", "b11003")
        for geography in ("national", "state")
    },
    **{
        (f"irs_soi.ty2023.table_1_3.filing_status_{measure}", "tax_year", "2023"): (
            "irs_soi"
        )
        for measure in ("returns", "agi", "ctc_returns", "ctc_amount")
    },
}
ANCHOR_RECORD_SETS = {
    f"bea_nipa.cy{year}.private_group_health_insurance" for year in (2023, 2024, 2025)
}


def anchor(fact: dict) -> dict:
    return {
        "value": fact["value"],
        "source_sha256": fact["source"]["source_sha256"],
    }


def main() -> None:
    old_path, new_path, report_path = map(Path, sys.argv[1:4])
    old_digest, new_digest, retained_digest = (
        hashlib.sha256(),
        hashlib.sha256(),
        hashlib.sha256(),
    )
    added_keys, new_keys = [], set()
    old_anchors, new_anchors = {}, {}
    old_count, new_count = 0, 0

    def retained_lines():
        nonlocal new_count
        with new_path.open("rb") as stream:
            for line in stream:
                fact = json.loads(line)
                new_digest.update(line)
                new_count += 1
                key = fact["aggregate_fact_key"]
                if key in new_keys:
                    raise ValueError(f"Repeated aggregate key: {key}")
                new_keys.add(key)
                record_set = fact["layout"]["record_set_id"]
                if record_set in ANCHOR_RECORD_SETS:
                    new_anchors[record_set] = anchor(fact)
                pair = (
                    record_set,
                    fact["period"]["type"],
                    str(fact["period"]["value"]),
                )
                if pair in NEW_PAIRS:
                    if fact["source"]["source_name"] != NEW_PAIRS[pair]:
                        raise ValueError(f"Unreviewed source for added pair: {pair}")
                    added_keys.append(key)
                else:
                    retained_digest.update(line)
                    yield line

    with old_path.open("rb") as stream:
        for index, (old_line, new_line) in enumerate(
            itertools.zip_longest(stream, retained_lines()), 1
        ):
            if old_line != new_line:
                raise ValueError(f"Retained row {index} differs byte for byte")
            old_count += 1
            old_digest.update(old_line)
            if b"private_group_health_insurance" in old_line:
                fact = json.loads(old_line)
                record_set = fact["layout"]["record_set_id"]
                if record_set in ANCHOR_RECORD_SETS:
                    old_anchors[record_set] = anchor(fact)
    if set(old_anchors) != ANCHOR_RECORD_SETS or old_anchors != new_anchors:
        raise ValueError("The three B4923C anchors differ or are missing")
    report = {
        "feeds": {
            "old": {"name": old_path.name, "sha256": old_digest.hexdigest()},
            "new": {"name": new_path.name, "sha256": new_digest.hexdigest()},
        },
        "rows": {
            "old_rows": old_count,
            "new_rows": new_count,
            "keys_only_in_old": [],
            "keys_only_in_new": sorted(added_keys),
            "changed_rows": 0,
            "changed_fields": {},
            "retained_bytes_equal_old_feed": True,
            "retained_sha256": retained_digest.hexdigest(),
        },
        "anchors": {
            record_set: {"old": old_anchors[record_set], "new": new_anchors[record_set]}
            for record_set in sorted(ANCHOR_RECORD_SETS)
        },
    }
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{old_count} retained rows byte-identical; {len(added_keys)} added rows")


if __name__ == "__main__":
    main()
