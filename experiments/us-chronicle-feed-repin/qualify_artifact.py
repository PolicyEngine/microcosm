"""Qualify Chronicle #278 against #955's preceding labelled US feed.

This source-authority repair must preserve every source cell and every target
field except the explicitly enumerated provenance fields. The expected counts
come from Chronicle #277's 994 missing-authority rows. No population is loaded.

Run from a Microcosm checkout with its locked environment::

    uv run python experiments/us-chronicle-feed-repin/qualify_artifact.py \
        --old /path/to/consumer_facts_us_c5e5bf8.jsonl \
        --new /path/to/validated-artifact --out /path/to/comparison.json

``--require-identical-targets`` additionally requires full TargetSpec equality.
It exits 1 on this qualified repair because provenance metadata changes; the
report retains both full equality and the narrower acceptance result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.fiscal_targets import compile_us_fiscal_target_registry
from microcosm.build.us_runtime.medicaid_take_up import (
    apply_us_medicaid_enrollment_substitutions,
)
from microcosm.build.us_runtime.release_target_parity import us_target_family_id

# Exact #955 input and #278 output qualified by this one-off experiment.
_OLD_FACTS_SHA256 = "b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801"
_NEW_FACTS_SHA256 = "4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f"
_NEW_MANIFEST_SHA256 = (
    "38ec5bf1efe5a0bd017ec5279065e2ea7645b37da237197f03ae2fbca28cadac"
)

# Chronicle #277's missing publisher authorities, repaired by #278.
_AUTHORITY_ADDITIONS = {"census": 468, "cms": 515, "jct": 11}
_SOURCE_DIFFS = {
    "concept_alignment.authority": 994,
    "concept_alignment.concept_alignment_key": 994,
    "layout.record_set_spec_hash": 994,
    "legacy_fact_key": 994,
}
# The Rhode Island substitution drops per-fact keys, hence 166 versus 165.
_TARGET_DIFFS = {
    "metadata.ledger_concept_authority": 166,
    "metadata.ledger_legacy_fact_key": 165,
}
_TARGET_FAMILY_CHANGES = {
    "cms_medicaid.state_enrollment": 155,
    "jct.tax_expenditures": 11,
}
_TARGET_IDENTITIES = {
    "old": {
        "registry_version": "b74d86d94a76",
        "canonical_specs_sha256": (
            "bb72cf6028ffb9943dab25a0751477f703620d6dd694c76174c355a67fccd151"
        ),
    },
    "new": {
        "registry_version": "749a7b0627ce",
        "canonical_specs_sha256": (
            "f57fa5a526d2afdcf1daa5d7a0649b15c68b0b26761e8e2102d110b8bfa8b189"
        ),
    },
}


def _diff_paths(before: Any, after: Any, path: str = "") -> list[str]:
    if type(before) is not type(after):
        return [path]
    if isinstance(before, dict):
        paths = []
        for key in before.keys() | after.keys():
            child = f"{path}.{key}" if path else key
            paths.extend(
                [child]
                if key not in before or key not in after
                else _diff_paths(before[key], after[key], child)
            )
        return paths
    if isinstance(before, list):
        if len(before) != len(after):
            return [path]
        return [
            child
            for left, right in zip(before, after, strict=True)
            for child in _diff_paths(left, right, f"{path}[]")
        ]
    return [] if before == after else [path]


def _cell_key(row: dict) -> tuple:
    layout, period = row["layout"], row["period"]
    return (
        layout["record_set_id"],
        period["type"],
        str(period["value"]),
        str(layout.get("source_row_id")),
        str(layout.get("source_column_id")),
    )


def _source_comparison(before: tuple[dict, ...], after: tuple[dict, ...]) -> dict:
    old = {_cell_key(row): row for row in before}
    new = {_cell_key(row): row for row in after}
    if len(old) != len(before) or len(new) != len(after):
        raise ValueError("A feed has duplicate source-cell identities.")
    changes: Counter[str] = Counter()
    changed_rows = different_values = 0
    authorities: Counter[str] = Counter()
    authority_rewrites = 0
    record_sets: Counter[str] = Counter()
    for key in old.keys() & new.keys():
        paths = _diff_paths(old[key], new[key])
        changes.update(set(paths))
        if paths:
            changed_rows += 1
            record_sets[key[0]] += 1
        different_values += old[key]["value"] != new[key]["value"]
        if "concept_alignment.authority" in paths:
            previous = old[key].get("concept_alignment", {}).get("authority")
            current = new[key].get("concept_alignment", {}).get("authority")
            if previous is not None or not isinstance(current, str) or not current:
                authority_rewrites += 1
            else:
                authorities[current] += 1
    return {
        "shared_cells": len(old.keys() & new.keys()),
        "cells_only_old": len(old.keys() - new.keys()),
        "cells_only_new": len(new.keys() - old.keys()),
        "different_values": different_values,
        "changed_rows": changed_rows,
        "changed_field_paths": dict(sorted(changes.items())),
        "changed_rows_by_record_set": dict(sorted(record_sets.items())),
        "authority_additions": dict(sorted(authorities.items())),
        "authority_rewrites_or_invalid_additions": authority_rewrites,
        "old_scope_pairs": len({key[:3] for key in old}),
        "new_scope_pairs": len({key[:3] for key in new}),
        "same_scope_pairs": {key[:3] for key in old} == {key[:3] for key in new},
    }


def _compile(facts: tuple[dict, ...]) -> tuple[dict, dict]:
    started = time.monotonic()
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    specs = {spec.key: asdict(spec) for spec in registry}
    canonical = json.dumps(
        sorted(specs.values(), key=lambda spec: (spec["name"], str(spec["period"]))),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    summary = {
        "count": len(registry),
        "families": dict(
            sorted(Counter(us_target_family_id(s.name) for s in registry).items())
        ),
        "all_have_hierarchy": all(spec.hierarchy is not None for spec in registry),
        "registry_version": registry.version,
        "canonical_specs_sha256": hashlib.sha256(canonical).hexdigest(),
        "compile_seconds": time.monotonic() - started,
    }
    return specs, summary


def _target_comparison(before: dict, after: dict, old_summary: dict, new_summary: dict):
    changed_specs = changed_values = 0
    changes: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for key in before.keys() & after.keys():
        paths = _diff_paths(before[key], after[key])
        changes.update(set(paths))
        changed_specs += bool(paths)
        if paths:
            families[us_target_family_id(before[key]["name"])] += 1
        changed_values += before[key]["value"] != after[key]["value"]
    return {
        "old": old_summary,
        "new": new_summary,
        "same_keys": before.keys() == after.keys(),
        "only_old": len(before.keys() - after.keys()),
        "only_new": len(after.keys() - before.keys()),
        "changed_specs": changed_specs,
        "changed_values": changed_values,
        "changed_field_paths": dict(sorted(changes.items())),
        "changed_specs_by_family": dict(sorted(families.items())),
        "full_structure_equal": before == after,
    }


def qualify(old_path: Path, new_path: Path) -> dict:
    """Compare public facts and their compiled targets without changing either."""
    old = load_ledger_consumer_artifact(
        old_path, expected_facts_sha256=_OLD_FACTS_SHA256
    )
    new = load_ledger_consumer_artifact(
        new_path,
        expected_facts_sha256=_NEW_FACTS_SHA256,
        expected_manifest_sha256=_NEW_MANIFEST_SHA256,
    )
    if new.manifest is None:
        raise ValueError("--new must be a consumer-artifact directory with a manifest.")
    old_identity, new_identity = old.provenance(), new.provenance()
    source = _source_comparison(old.facts, new.facts)
    before, old_summary = _compile(old.facts)
    del old
    gc.collect()
    after, new_summary = _compile(new.facts)
    del new
    gc.collect()
    targets = _target_comparison(before, after, old_summary, new_summary)
    checks = {
        "same_39158_source_cells": source["shared_cells"] == 39158
        and source["cells_only_old"] == source["cells_only_new"] == 0,
        "same_586_scope_pairs": source["same_scope_pairs"]
        and source["new_scope_pairs"] == 586,
        "source_values_equal": source["different_values"] == 0,
        "only_expected_source_provenance_changes": source["changed_field_paths"]
        == _SOURCE_DIFFS
        and source["changed_rows"] == 994,
        "expected_publisher_authorities_added": source["authority_additions"]
        == _AUTHORITY_ADDITIONS
        and source["authority_rewrites_or_invalid_additions"] == 0,
        "same_32867_target_keys": targets["same_keys"]
        and old_summary["count"] == new_summary["count"] == 32867,
        "same_32_target_families": old_summary["families"] == new_summary["families"]
        and len(new_summary["families"]) == 32,
        "all_target_values_equal": targets["changed_values"] == 0,
        "only_expected_target_provenance_changes": targets["changed_field_paths"]
        == _TARGET_DIFFS
        and targets["changed_specs"] == 166,
        "expected_changed_target_families": targets["changed_specs_by_family"]
        == _TARGET_FAMILY_CHANGES,
        "expected_registry_versions": all(
            targets[side]["registry_version"] == identity["registry_version"]
            for side, identity in _TARGET_IDENTITIES.items()
        ),
        "expected_canonical_spec_hashes": all(
            targets[side]["canonical_specs_sha256"]
            == identity["canonical_specs_sha256"]
            for side, identity in _TARGET_IDENTITIES.items()
        ),
        "all_targets_have_hierarchy": old_summary["all_have_hierarchy"]
        and new_summary["all_have_hierarchy"],
    }
    return {
        "old": old_identity,
        "new": new_identity,
        "source_comparison": source,
        "target_comparison": targets,
        "checks": checks,
        "provenance_only_qualification_passed": all(checks.values()),
        "strict_full_target_equality_passed": targets["full_structure_equal"],
        "population_or_release_validation_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-identical-targets", action="store_true")
    args = parser.parse_args()
    report = qualify(args.old, args.new)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    passed = report["provenance_only_qualification_passed"] and (
        not args.require_identical_targets
        or report["strict_full_target_equality_passed"]
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "checks": report["checks"],
                "strict_full_target_equality_passed": report[
                    "strict_full_target_equality_passed"
                ],
                "exit_code": 0 if passed else 1,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
