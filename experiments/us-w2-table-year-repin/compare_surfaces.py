"""Diff two measure_surface.py reports target by target.

Usage: python compare_surfaces.py <before.json> <after.json> <out.json>

Lists targets only in one report, targets whose value changed and targets
whose reported metadata changed, and restates the W-2 rows of both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

before_path, after_path, out_path = sys.argv[1:4]
before = json.loads(Path(before_path).read_text())
after = json.loads(Path(after_path).read_text())
old, new = before["targets"], after["targets"]
shared = sorted(set(old) & set(new))
summary_keys = (
    "feed",
    "feed_sha256",
    "fact_count",
    "compiled_targets",
    "compiled_families",
    "national_state_targets",
    "national_state_version",
)
report = {
    "before": {key: before[key] for key in summary_keys},
    "after": {key: after[key] for key in summary_keys},
    "targets_only_before": {name: old[name] for name in sorted(set(old) - set(new))},
    "targets_only_after": {name: new[name] for name in sorted(set(new) - set(old))},
    "shared_targets": len(shared),
    "shared_value_changes": [
        {"name": name, "before": old[name]["value"], "after": new[name]["value"]}
        for name in shared
        if old[name]["value"] != new[name]["value"]
    ],
    "shared_metadata_changes": [
        {"name": name, "before": old[name], "after": new[name]}
        for name in shared
        if {k: v for k, v in old[name].items() if k != "value"}
        != {k: v for k, v in new[name].items() if k != "value"}
    ],
    "w2_item_facts_before": before["w2_item_facts"],
    "w2_item_facts_after": after["w2_item_facts"],
    "exclusion_receipt_w2_ids_before": before["exclusion_receipt_w2_ids"],
    "exclusion_receipt_w2_ids_after": after["exclusion_receipt_w2_ids"],
}
Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
print(
    json.dumps(
        {
            key: (len(value) if isinstance(value, list | dict) else value)
            for key, value in report.items()
            if key.startswith(("targets_only", "shared"))
        },
        indent=1,
    )
)
