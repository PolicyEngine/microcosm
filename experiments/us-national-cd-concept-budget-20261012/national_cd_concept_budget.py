"""The national concept budget for congressional-district rows, before and after.

Compiles the national fiscal registry from the pinned feed (the
``chronicle_feed.json`` pin, the compile ``--target-surface full`` calibrates),
and computes its loss weights twice:

- **before**: the pre-change code, the verbatim copy in
  ``test_support/microcosm_build/us_target_loss_weights_reference.py``, whose
  district key reads the compiler's per-row labels;
- **after**: ``us_runtime.target_loss_weights`` as it is now, whose national
  mapping ignores them and refuses a split district concept.

It writes ``results.json`` (group counts and loss shares by geography level,
basis and family) and the CI fixture
``packages/microcosm-build/tests/fixtures/us_national_district_concept_rows.json``:
every compiled district row of seven concepts in West Virginia (two
districts), Montana (two districts, plus Historic Table 2 proxies) and Wyoming
(at large, plus proxies), with the metadata the compiler stamped.

Run from the repository root (needs the pinned feed on disk; ~2 minutes)::

    uv run python experiments/us-national-cd-concept-budget-20261012/national_cd_concept_budget.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FEED = (
    Path.home() / "PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_51aa40f.jsonl"
)
FIXTURE = (
    REPO
    / "packages/microcosm-build/tests/fixtures/us_national_district_concept_rows.json"
)
DISTRICT = "congressional_district"
#: West Virginia, Montana, Wyoming.
FIXTURE_STATES = ("54", "30", "56")
#: The TargetSpec fields the loss weights read, which the fixture keeps.
FIXTURE_FIELDS = (
    "name",
    "entity",
    "value",
    "measure",
    "filter",
    "period",
    "family",
    "source",
    "signed",
    "metadata",
)
#: (source_measure_id, record set spec) of the fixture's concepts.
FIXTURE_CONCEPTS = (
    ("adjusted_gross_income", "irs_soi.congressional_district_2022.all_returns.v1"),
    ("return_count", "irs_soi.congressional_district_2022.all_returns.v1"),
    ("eitc_two_children_claims", "irs_soi.congressional_district_2022.all_returns.v1"),
    (
        "itemized_deductions_amount",
        "irs_soi.congressional_district_2022.all_returns.v1",
    ),
    ("net_capital_gains_amount", "irs_soi.congressional_district_2022.all_returns.v1"),
    ("net_capital_gains_amount", "irs_soi.historic_table_2.state_broad_totals.v1"),
    ("return_count", "irs_soi.historic_table_2.state_broad_totals.v1"),
)


def _git_head() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": head, "tracked_tree_clean": not dirty}


def _shares(weights, mask_by_label):
    total = float(weights.sum())
    return {
        label: round(float(weights[mask].sum()) / total, 6)
        for label, mask in mask_by_label.items()
    }


def main() -> int:
    sys.path.insert(0, str(REPO))
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime import target_loss_weights as lw
    from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )
    from test_support.microcosm_build import (
        us_target_loss_weights_reference as before,
    )

    pin = load_us_chronicle_feed()
    feed_sha = hashlib.sha256(FEED.read_bytes()).hexdigest()
    if feed_sha != pin.facts_sha256:
        raise SystemExit(f"{FEED} is {feed_sha}, not the pinned feed.")
    registry = compile_us_fiscal_target_registry(
        load_ledger_consumer_artifact(
            FEED, expected_facts_sha256=pin.facts_sha256, expected_manifest_sha256=None
        ).facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(
            default_congressional_district_vintage_crosswalk_path()
        ),
        age_targets=True,
    )
    specs = registry.specs
    old = before._fiscal_target_loss_weights(registry)
    new = lw.fiscal_target_loss_weights(registry)

    levels = np.asarray([lw.target_geography_level(spec) for spec in specs])
    bases = np.asarray([lw.fiscal_target_value_basis(spec) for spec in specs])
    families = np.asarray([spec.family for spec in specs])
    district = levels == DISTRICT
    district_specs = [spec for spec, flag in zip(specs, district, strict=True) if flag]

    def groups(key) -> dict[str, object]:
        keys = [key(spec) for spec in district_specs]
        sizes = collections.Counter(keys)
        singleton_states = collections.Counter(
            spec.metadata["state_fips"]
            for spec, k in zip(district_specs, keys, strict=True)
            if sizes[k] == 1
        )
        return {
            "n_groups": len(sizes),
            "n_singletons": sum(1 for size in sizes.values() if size == 1),
            "max_size": max(sizes.values()),
            "singletons_by_state": dict(sorted(singleton_states.items())),
        }

    identities = {lw.district_concept_identity(spec) for spec in district_specs}
    districts_per_state = collections.defaultdict(set)
    for spec in district_specs:
        districts_per_state[spec.metadata["state_fips"]].add(
            spec.metadata["congressional_district_geoid"]
        )
    level_masks = {level: levels == level for level in sorted(set(levels))}
    basis_level_masks = {
        f"{basis}/{level}": (bases == basis) & (levels == level)
        for basis in sorted(set(bases))
        for level in sorted(set(levels))
    }
    family_masks = {family: families == family for family in sorted(set(families))}
    non_district_ratio = {
        basis: sorted(
            {
                round(float(ratio), 9)
                for ratio in new[(bases == basis) & ~district]
                / old[(bases == basis) & ~district]
            }
        )
        for basis in sorted(set(bases))
    }
    district_ratio = new[district] / old[district]
    results = {
        "provenance": {
            **_git_head(),
            "feed_facts_sha256": feed_sha,
            "chronicle_source_commit": pin.source_commit,
            "target_period": 2024,
            "age_targets": True,
        },
        "n_targets": len(specs),
        "n_district_rows": int(district.sum()),
        "n_district_concept_identities": len(identities),
        "single_district_states": sorted(
            state for state, geoids in districts_per_state.items() if len(geoids) == 1
        ),
        "district_groups": {
            "before": groups(before._fiscal_target_concept_budget_key),
            "after": groups(lw.fiscal_target_concept_budget_key),
        },
        "equal_share_by_level": _shares(np.ones(len(specs)), level_masks),
        "loss_share_by_level": {
            "before": _shares(old, level_masks),
            "after": _shares(new, level_masks),
        },
        "loss_share_by_basis_and_level": {
            "before": _shares(old, basis_level_masks),
            "after": _shares(new, basis_level_masks),
        },
        "loss_share_by_family": {
            "before": _shares(old, family_masks),
            "after": _shares(new, family_masks),
        },
        "weight_ratio_after_over_before": {
            "non_district_rows_by_basis": non_district_ratio,
            "district_rows": {
                "min": round(float(district_ratio.min()), 6),
                "median": round(float(np.median(district_ratio)), 6),
                "max": round(float(district_ratio.max()), 6),
            },
        },
        "weights": {
            "before": {"min": float(old.min()), "max": float(old.max())},
            "after": {"min": float(new.min()), "max": float(new.max())},
        },
    }
    (HERE / "results.json").write_text(
        json.dumps(results, indent=1, sort_keys=True) + "\n"
    )

    wanted = set(FIXTURE_CONCEPTS)
    rows = [
        {field: getattr(spec, field) for field in FIXTURE_FIELDS}
        for spec in district_specs
        if spec.metadata["state_fips"] in FIXTURE_STATES
        and (
            spec.metadata.get("source_measure_id"),
            spec.metadata.get("ledger_layout_record_set_spec_id"),
        )
        in wanted
    ]
    fixture = {
        "description": (
            "Every compiled congressional-district row of seven concepts in WV, "
            "MT and WY, with the metadata the ledger compiler stamped, from the "
            "pinned feed. Written by experiments/us-national-cd-concept-budget-"
            "20261012/national_cd_concept_budget.py; do not edit by hand."
        ),
        "feed_facts_sha256": feed_sha,
        "chronicle_source_commit": pin.source_commit,
        "compiled_target_count": len(specs),
        "states": list(FIXTURE_STATES),
        "concepts": [list(concept) for concept in FIXTURE_CONCEPTS],
        "rows": sorted(rows, key=lambda row: (row["name"], str(row["period"]))),
    }
    FIXTURE.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    print(json.dumps(results, indent=1, sort_keys=True))
    print(f"wrote {len(rows)} fixture rows to {FIXTURE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
