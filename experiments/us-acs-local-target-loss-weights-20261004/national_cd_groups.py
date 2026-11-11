"""Do the national mapping's district rows share concept budgets?

Compiles the national fiscal registry from the feed ``chronicle_feed.json``
pins (the compile the national release and the ACS local release both start
from), takes its congressional-district rows (which ``--target-surface full``,
the national parser's default, calibrates), and counts their concept groups
under the national mapping as is, and with the two per-row labels the ledger
compiler stamps (``ledger_fact_label``, ``ledger_layout_groupby_value_label``)
also excluded. It reports each family's share of the loss both ways.

This is a finding for a follow-up, not a change: the national mapping is moved
unchanged in this branch. Route A calibrated ``--target-surface
national_state``, which has no district rows, so it is unaffected.

Run from the repository root::

    uv run python experiments/us-acs-local-target-loss-weights-20261004/national_cd_groups.py
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FEED = Path(
    "/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl"
)
LABELS = frozenset({"ledger_fact_label", "ledger_layout_groupby_value_label"})


def main() -> int:
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime import target_loss_weights as lw
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )

    pin = json.loads(
        (
            REPO / "packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json"
        ).read_text()
    )
    feed_sha = hashlib.sha256(FEED.read_bytes()).hexdigest()
    if feed_sha != pin["facts_sha256"]:
        raise SystemExit(f"{FEED} is {feed_sha}, not the pinned feed.")
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    registry = compile_us_fiscal_target_registry(
        load_ledger_consumer_artifact(str(FEED)).facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
    )
    specs = registry.specs

    def without_labels(spec):
        return lw._ledger_concept_budget_key(
            spec,
            lw.fiscal_target_value_basis(spec),
            lw.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS | LABELS,
        )

    probe = lw.TargetLossRowMapping(
        "national_without_district_labels_probe",
        lw.fiscal_target_value_basis,
        without_labels,
    )
    national = lw.target_loss_weights(specs)
    relabelled = lw.target_loss_weights(specs, row_mapping=probe)
    district = np.asarray(
        [
            spec.metadata.get("ledger_geography_level") == "congressional_district"
            for spec in specs
        ]
    )
    families = np.asarray([spec.family for spec in specs])

    def groups(key) -> dict[str, int]:
        sizes = collections.Counter(key(spec) for spec in np.asarray(specs)[district])
        return {
            "n_groups": len(sizes),
            "n_singletons": sum(1 for size in sizes.values() if size == 1),
            "max_size": max(sizes.values(), default=0),
        }

    result = {
        "feed_sha256": feed_sha,
        "n_targets": len(specs),
        "n_district_rows": int(district.sum()),
        "district_families": dict(collections.Counter(families[district].tolist())),
        "district_groups": {
            "national_as_is": groups(lw.fiscal_target_concept_budget_key),
            "labels_excluded": groups(without_labels),
        },
        "district_loss_share": {
            "equal": float(district.mean()),
            "national_as_is": float(national[district].sum() / national.sum()),
            "labels_excluded": float(relabelled[district].sum() / relabelled.sum()),
        },
        "family_loss_share": {
            family: {
                "n_targets": int((families == family).sum()),
                "national_as_is": float(
                    national[families == family].sum() / national.sum()
                ),
                "labels_excluded": float(
                    relabelled[families == family].sum() / relabelled.sum()
                ),
            }
            for family in sorted(set(families.tolist()))
        },
    }
    (HERE / "national_cd_groups.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
