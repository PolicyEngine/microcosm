"""Compare the rebuilt US Chronicle feed with the pinned v9_4 feed on a stable join.

Fact keys were re-derived between Chronicle generations, so rows are joined on
what identifies a cell in the source table: (record_set_id, period type, period
value, layout.source_row_id, layout.source_column_id). Every joined pair is
compared on ``value``; rows present on one side only are listed by record set.

    uv run python experiments/us-chronicle-feed-repin/value_diff.py \
        --pinned ~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
        --new ~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl \
        --out experiments/us-chronicle-feed-repin/value_diff.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _cell_key(row: dict) -> tuple[str, str, str, str, str]:
    layout = row["layout"]
    period = row["period"]
    return (
        layout["record_set_id"],
        period["type"],
        str(period["value"]),
        str(layout.get("source_row_id")),
        str(layout.get("source_column_id")),
    )


def _load(path: Path) -> dict[tuple, list[dict]]:
    rows: dict[tuple, list[dict]] = collections.defaultdict(list)
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[_cell_key(row)].append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = _load(args.pinned)
    new = _load(args.new)
    old_rows = sum(len(v) for v in old.values())
    new_rows = sum(len(v) for v in new.values())

    def by_record_set(keys) -> dict[str, int]:
        counter = collections.Counter(key[0] for key in keys)
        return dict(sorted(counter.items()))

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    shared = sorted(set(old) & set(new))
    dup_old = {k: len(v) for k, v in old.items() if len(v) > 1}
    dup_new = {k: len(v) for k, v in new.items() if len(v) > 1}

    differences = []
    equal = 0
    for key in shared:
        old_values = sorted({json.dumps(r.get("value")) for r in old[key]})
        new_values = sorted({json.dumps(r.get("value")) for r in new[key]})
        if old_values == new_values:
            equal += 1
        else:
            differences.append(
                {
                    "cell": list(key),
                    "pinned_values": old_values,
                    "new_values": new_values,
                    "pinned_semantic_fact_keys": sorted(
                        r.get("semantic_fact_key") for r in old[key]
                    ),
                    "new_semantic_fact_keys": sorted(
                        r.get("semantic_fact_key") for r in new[key]
                    ),
                }
            )

    report = {
        "pinned": {"path": str(args.pinned), "rows": old_rows, "cells": len(old)},
        "new": {"path": str(args.new), "rows": new_rows, "cells": len(new)},
        "join": [
            "record_set_id",
            "period.type",
            "period.value",
            "layout.source_row_id",
            "layout.source_column_id",
        ],
        "shared_cells": len(shared),
        "shared_cells_equal_value": equal,
        "shared_cells_different_value": len(differences),
        "differences": differences,
        "cells_only_in_pinned": len(only_old),
        "cells_only_in_pinned_by_record_set": by_record_set(only_old),
        "cells_only_in_new": len(only_new),
        "cells_only_in_new_by_record_set": by_record_set(only_new),
        "duplicate_cells_in_pinned": len(dup_old),
        "duplicate_cells_in_pinned_by_record_set": by_record_set(dup_old),
        "duplicate_cells_in_new": len(dup_new),
        "duplicate_cells_in_new_by_record_set": by_record_set(dup_new),
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"pinned {old_rows} rows / {len(old)} cells; new {new_rows} rows / "
        f"{len(new)} cells; shared {len(shared)} (equal {equal}, different "
        f"{len(differences)}); only pinned {len(only_old)}; only new {len(only_new)}; "
        f"duplicate cells pinned {len(dup_old)} new {len(dup_new)}"
    )


if __name__ == "__main__":
    main()
