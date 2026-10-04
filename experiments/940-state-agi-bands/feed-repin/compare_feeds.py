"""What moving the US Chronicle feed pin changes, besides the #940 facts.

Usage: python compare_feeds.py <old feed jsonl> <new feed jsonl> <out.json>

1. Cell join (record set, period, source row, source column): every old cell
   must be in the new feed with an equal value; list added and removed cells
   by record set.
2. Field diff on joined cells: which fact fields changed (e.g. concept ids,
   schema_version), counted per record-set family.
3. Compile both feeds as the release does (target 2024, packaged CD
   crosswalk, age_targets=True, Medicaid substitutions) and compare targets:
   names added/removed, value changes, metadata changes, excluding the state
   AGI band rows #940 adds.
"""

from __future__ import annotations

import collections
import json
import sys

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    apply_us_medicaid_enrollment_substitutions,
    compile_us_fiscal_target_registry,
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)

old_path, new_path, out_path = sys.argv[1:4]


def cell_key(row):
    layout = row["layout"]
    return (
        layout["record_set_id"],
        row["period"]["type"],
        str(row["period"]["value"]),
        str(layout.get("source_row_id")),
        str(layout.get("source_column_id")),
    )


def family(record_set_id: str) -> str:
    parts = [p for p in record_set_id.split(".") if p[:2] not in {"ty", "cy", "fy"}]
    return ".".join(parts[:3])


def load_rows(path):
    rows = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            rows[cell_key(row)].append(row)
    return rows


old_rows, new_rows = load_rows(old_path), load_rows(new_path)
shared = set(old_rows) & set(new_rows)
value_changes = []
field_changes = collections.Counter()
field_examples = {}
for key in sorted(shared):
    old, new = old_rows[key][0], new_rows[key][0]
    if old["value"] != new["value"]:
        value_changes.append({"cell": key, "old": old["value"], "new": new["value"]})
    for field in sorted(set(old) | set(new)):
        if old.get(field) != new.get(field):
            if field in {"aggregate_fact_key", "semantic_fact_key", "legacy_fact_key"}:
                continue
            tag = (family(key[0]), field)
            field_changes[tag] += 1
            field_examples.setdefault(
                f"{tag[0]}|{field}",
                {"cell": key, "old": old.get(field), "new": new.get(field)},
            )
only_old = collections.Counter(family(k[0]) for k in set(old_rows) - set(new_rows))
only_new = collections.Counter(
    k[0].rsplit(".", 1)[0] for k in set(new_rows) - set(old_rows)
)

crosswalk = load_congressional_district_vintage_crosswalk(
    default_congressional_district_vintage_crosswalk_path()
)


def compile_(path):
    facts = load_ledger_consumer_artifact(path).facts
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    return registry


def is_band(spec):
    return spec.metadata.get("requires_state_agi_band_rebase") == "true"


old_reg, new_reg = compile_(old_path), compile_(new_path)
old_specs = {s.name: s for s in old_reg.specs}
new_specs = {s.name: s for s in new_reg.specs}
old_core = {n for n, s in old_specs.items() if not is_band(s)}
new_core = {n for n, s in new_specs.items() if not is_band(s)}
target_value_changes = []
target_metadata_changes = collections.Counter()
target_metadata_examples = {}
for name in sorted(old_core & new_core):
    a, b = old_specs[name], new_specs[name]
    if abs(a.value - b.value) > 1e-9 * max(1.0, abs(a.value)):
        target_value_changes.append({"name": name, "old": a.value, "new": b.value})
    for key in sorted(set(a.metadata) | set(b.metadata)):
        if a.metadata.get(key) != b.metadata.get(key):
            target_metadata_changes[key] += 1
            target_metadata_examples.setdefault(
                key,
                {"name": name, "old": a.metadata.get(key), "new": b.metadata.get(key)},
            )

report = {
    "cells": {
        "old": len(old_rows),
        "new": len(new_rows),
        "shared": len(shared),
        "value_changes": value_changes[:50],
        "value_change_count": len(value_changes),
        "only_old_by_family": dict(only_old),
        "only_new_by_record_set_prefix": dict(collections.Counter(only_new)),
        "field_changes": {f"{f}|{k}": n for (f, k), n in sorted(field_changes.items())},
        "field_change_examples": field_examples,
    },
    "targets": {
        "old_count": len(old_specs),
        "new_count": len(new_specs),
        "old_state_bands": len(old_specs) - len(old_core),
        "new_state_bands": len(new_specs) - len(new_core),
        "core_removed": sorted(old_core - new_core)[:100],
        "core_removed_count": len(old_core - new_core),
        "core_added": sorted(new_core - old_core)[:100],
        "core_added_count": len(new_core - old_core),
        "core_value_change_count": len(target_value_changes),
        "core_value_changes": target_value_changes[:50],
        "core_metadata_changes": dict(target_metadata_changes),
        "core_metadata_change_examples": target_metadata_examples,
        "old_version": old_reg.version,
        "new_version": new_reg.version,
    },
}
json.dump(report, open(out_path, "w"), indent=1, default=str)
print(
    json.dumps(
        {
            "cells": {
                k: v
                for k, v in report["cells"].items()
                if k not in {"value_changes", "field_change_examples"}
            },
            "targets": {
                k: v
                for k, v in report["targets"].items()
                if k
                not in {
                    "core_removed",
                    "core_added",
                    "core_value_changes",
                    "core_metadata_change_examples",
                }
            },
        },
        indent=1,
        default=str,
    )
)
