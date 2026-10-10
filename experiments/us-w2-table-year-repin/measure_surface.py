"""Compile a US Chronicle feed as the release does and report the W-2 rows.

Usage: python measure_surface.py <feed jsonl> <feed sha256> <out.json>

The compile is the release's: ``compile_us_fiscal_target_registry`` at target
period 2024 with the packaged congressional-district crosswalk and aging on,
then the Medicaid enrollment substitutions, then
``--target-surface national_state``. The report lists every compiled target
(name, value, aging metadata and a digest of all its metadata) so two runs can
be compared target by target.
"""

from __future__ import annotations

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
    us_fiscal_target_exclusion_receipt,
)
from microcosm.calibrate import TargetRegistry

feed_path, feed_sha256, out_path = sys.argv[1:4]
REPO = Path(__file__).resolve().parents[2]
AGING_KEYS = (
    "source_period",
    "aged_to",
    "aging_factor",
    "aging_factor_source",
    "basis",
    "target_role",
    "ledger_source_record_id",
)


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


facts = load_ledger_consumer_artifact(
    feed_path, expected_facts_sha256=feed_sha256, expected_manifest_sha256=None
).facts
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
builder = _load_tool("build_us_fiscal_refresh_release")
surface_specs, _ = builder._select_target_surface(registry.specs, "national_state")
surface = TargetRegistry(surface_specs, country="us")
receipt = us_fiscal_target_exclusion_receipt(
    facts, target_period=2024, congressional_district_vintage_crosswalk=crosswalk
)
surface_names = {spec.name for spec in surface.specs}


def _record(spec) -> dict[str, object]:
    return {
        "value": spec.value,
        "family": spec.family,
        "in_national_state": spec.name in surface_names,
        "metadata_sha256": hashlib.sha256(
            json.dumps(dict(spec.metadata), sort_keys=True, default=str).encode()
        ).hexdigest(),
        "metadata": {
            key: spec.metadata[key]
            for key in sorted(spec.metadata)
            if key in AGING_KEYS
        },
    }


report = {
    "feed": Path(feed_path).name,
    "feed_sha256": feed_sha256,
    "compiler": str(Path(compile_us_fiscal_target_registry.__code__.co_filename)),
    "fact_count": len(facts),
    "compiled_targets": len(registry.specs),
    "compiled_families": len({spec.family for spec in registry.specs}),
    "national_state_targets": len(surface.specs),
    "national_state_version": surface.version,
    "w2_item_facts": sorted(
        (
            {
                "source_record_id": fact["lineage"]["source_record_id"],
                "period": fact["period"]["value"],
                "vintage": fact["source"]["vintage"],
                "source_table": fact["source"]["source_table"],
                "value": fact["value"],
            }
            for fact in facts
            if fact["layout"].get("groupby_dimension") == "irs_soi.form_w2_item"
        ),
        key=lambda row: row["source_record_id"],
    ),
    "w2_item_targets": {
        spec.name: _record(spec) for spec in registry.specs if "form_w2" in spec.name
    },
    "exclusion_receipt_w2_ids": {
        rule: [i for i in body["source_record_ids"] if "form_w2" in i]
        for rule, body in receipt["rules"].items()
    },
    "targets": {spec.name: _record(spec) for spec in registry.specs},
}
Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "targets"}, indent=1))
