"""What the label-only US Chronicle feed re-pin changes, row by row and target by target.

Usage:
    uv run python experiments/us-feed-agi-age-labels/compare_feeds.py \\
        <old feed jsonl> <new feed jsonl> <out.json>

1. Rows, joined on ``aggregate_fact_key``: both feeds must hold the same keys.
   For every shared row, every changed leaf field is counted per record set;
   every changed label is listed in full, and every ``record_set_spec_hash``
   change once per record set.
2. Feed labels: every (dimension, value) that carries more than one label in
   either feed, with its labels.
3. Compile both feeds as the release does (target period 2024, packaged
   congressional-district crosswalk, age targets, Medicaid substitutions) and
   compare every target's full ``TargetSpec.to_dict()`` by name: names added or
   removed, and every changed field of every shared target, listed in full.
   Also the registry version and the ``national_state`` surface the release
   calibrates to.
4. Compiled labels: every (dimension, value) whose targets carry more than one
   value label, in either registry.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    apply_us_medicaid_enrollment_substitutions,
    compile_us_fiscal_target_registry,
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.calibrate import TargetRegistry

REPOSITORY = Path(__file__).resolve().parents[2]
LABEL_FIELDS = ("dimension_value_labels/", "layout/groupby_value_label")


def load_release_tool():
    path = REPOSITORY / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value, prefix=""):
    """Leaf paths of a JSON value; hierarchy dimensions are keyed by their id."""

    if isinstance(value, dict):
        if not value:
            yield prefix, {}
        for key, child in value.items():
            yield from flatten(child, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)) and prefix.endswith("hierarchy/dimensions"):
        for dimension in value:
            yield from flatten(dimension, f"{prefix}/{dimension['id']}")
    else:
        yield prefix, value


def changed_leaves(old, new):
    old_leaves, new_leaves = dict(flatten(old)), dict(flatten(new))
    return {
        path: (old_leaves.get(path, "<absent>"), new_leaves.get(path, "<absent>"))
        for path in sorted(set(old_leaves) | set(new_leaves))
        if old_leaves.get(path, "<absent>") != new_leaves.get(path, "<absent>")
    }


def load_rows(path):
    rows = {}
    for line in open(path, "rb"):
        if line.strip():
            row = json.loads(line)
            rows[row["aggregate_fact_key"]] = row
    return rows


def feed_label_variants(rows):
    labels = collections.defaultdict(collections.Counter)
    for row in rows.values():
        for dimension, values in (row.get("dimension_value_labels") or {}).items():
            for value_id, label in values.items():
                labels[f"{dimension}={value_id}"][label] += 1
    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(labels.items())
        if len(counter) > 1
    }


def compare_rows(old_rows, new_rows):
    path_counts = collections.defaultdict(collections.Counter)
    label_changes = []
    spec_hash_changes = {}
    for key in sorted(set(old_rows) & set(new_rows)):
        old_row, new_row = old_rows[key], new_rows[key]
        record_set = old_row["layout"]["record_set_id"]
        for path, (old_value, new_value) in changed_leaves(old_row, new_row).items():
            path_counts[path][record_set] += 1
            if path.startswith(LABEL_FIELDS):
                label_changes.append(
                    {
                        "record_set_id": record_set,
                        "source_record_id": old_row["lineage"]["source_record_id"],
                        "field": path,
                        "old": old_value,
                        "new": new_value,
                    }
                )
            elif path == "layout/record_set_spec_hash":
                spec_hash_changes[record_set] = {"old": old_value, "new": new_value}
            else:
                label_changes.append(
                    {
                        "record_set_id": record_set,
                        "source_record_id": old_row["lineage"]["source_record_id"],
                        "field": path,
                        "old": old_value,
                        "new": new_value,
                        "unexpected": True,
                    }
                )
    return {
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "keys_only_in_old": sorted(set(old_rows) - set(new_rows)),
        "keys_only_in_new": sorted(set(new_rows) - set(old_rows)),
        "changed_rows": len(
            {
                key
                for key in set(old_rows) & set(new_rows)
                if old_rows[key] != new_rows[key]
            }
        ),
        "changed_fields": {
            path: {
                "rows": sum(counts.values()),
                "by_record_set": dict(sorted(counts.items())),
            }
            for path, counts in sorted(path_counts.items())
        },
        "label_and_other_changes": label_changes,
        "record_set_spec_hash_changes": dict(sorted(spec_hash_changes.items())),
    }


def compile_registry(path, crosswalk, release_tool):
    facts = load_ledger_consumer_artifact(path).facts
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    surface_specs, _ = release_tool._select_target_surface(
        registry.specs, "national_state"
    )
    return registry, TargetRegistry(surface_specs, country="us")


def target_label_variants(registry):
    labels = collections.defaultdict(collections.Counter)
    for spec in registry.specs:
        if spec.hierarchy is None:
            continue
        for dimension in spec.hierarchy.dimensions:
            labels[f"{dimension.id}={dimension.value_id}"][dimension.value_label] += 1
    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(labels.items())
        if len(counter) > 1
    }


def compare_targets(old_registry, new_registry):
    old_specs = {spec.name: spec.to_dict() for spec in old_registry.specs}
    new_specs = {spec.name: spec.to_dict() for spec in new_registry.specs}
    differences = []
    path_counts = collections.Counter()
    for name in sorted(set(old_specs) & set(new_specs)):
        for path, (old_value, new_value) in changed_leaves(
            old_specs[name], new_specs[name]
        ).items():
            path_counts[path] += 1
            differences.append(
                {"name": name, "field": path, "old": old_value, "new": new_value}
            )
    return {
        "old_count": len(old_specs),
        "new_count": len(new_specs),
        "same_order": [spec.name for spec in old_registry.specs]
        == [spec.name for spec in new_registry.specs],
        "names_only_in_old": sorted(set(old_specs) - set(new_specs)),
        "names_only_in_new": sorted(set(new_specs) - set(old_specs)),
        "changed_targets": len({difference["name"] for difference in differences}),
        "changed_field_counts": dict(sorted(path_counts.items())),
        "differences": differences,
    }


def main() -> int:
    old_path, new_path, out_path = sys.argv[1:4]
    old_rows, new_rows = load_rows(old_path), load_rows(new_path)
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    release_tool = load_release_tool()
    old_registry, old_surface = compile_registry(old_path, crosswalk, release_tool)
    new_registry, new_surface = compile_registry(new_path, crosswalk, release_tool)
    report = {
        "feeds": {
            "old": {"name": Path(old_path).name, "sha256": file_sha256(old_path)},
            "new": {"name": Path(new_path).name, "sha256": file_sha256(new_path)},
        },
        "rows": compare_rows(old_rows, new_rows),
        "feed_label_variants": {
            "old": feed_label_variants(old_rows),
            "new": feed_label_variants(new_rows),
        },
        "targets": compare_targets(old_registry, new_registry),
        "registry_version": {"old": old_registry.version, "new": new_registry.version},
        "national_state_surface": {
            "old": {"targets": len(old_surface.specs), "version": old_surface.version},
            "new": {"targets": len(new_surface.specs), "version": new_surface.version},
        },
        "target_label_variants": {
            "old": target_label_variants(old_registry),
            "new": target_label_variants(new_registry),
        },
    }
    Path(out_path).write_text(json.dumps(report, indent=1, default=str) + "\n")
    summary = {
        "rows": {
            key: value
            for key, value in report["rows"].items()
            if key not in {"label_and_other_changes", "record_set_spec_hash_changes"}
        },
        "unexpected_row_changes": [
            change
            for change in report["rows"]["label_and_other_changes"]
            if change.get("unexpected")
        ],
        "feed_label_variants": report["feed_label_variants"],
        "targets": {
            key: value
            for key, value in report["targets"].items()
            if key != "differences"
        },
        "registry_version": report["registry_version"],
        "national_state_surface": report["national_state_surface"],
        "target_label_variants": report["target_label_variants"],
    }
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
