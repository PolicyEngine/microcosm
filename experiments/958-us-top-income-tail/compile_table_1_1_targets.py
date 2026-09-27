"""Compile the US fiscal target registry from a Chronicle feed and list Table 1.1 rows.

Usage:
    python compile_table_1_1_targets.py FACTS.jsonl CD_VINTAGE_CROSSWALK.csv OUT.json

Prints how many Publication 1304 Table 1.1 rows bind, and every national SOI row whose
AGI lower bound is at or above $500,000. OUT.json receives the Table 1.1 specs, which
``rake_candidate.py`` consumes as its targets.
"""

import json
import sys
import warnings

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime.congressional_district_vintage import (
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.fiscal_targets import (
    compile_us_fiscal_target_registry,
)

warnings.filterwarnings("ignore")

facts_path, crosswalk_path, out_path = sys.argv[1:4]
artifact = load_ledger_consumer_artifact(
    facts_path, expected_facts_sha256=None, expected_manifest_sha256=None
)
registry = compile_us_fiscal_target_registry(
    artifact.facts,
    target_period=2024,
    age_targets=True,
    congressional_district_vintage_crosswalk=(
        load_congressional_district_vintage_crosswalk(crosswalk_path)
    ),
)
specs = registry.specs
print("facts", len(artifact.facts), "| compiled specs", len(specs))


def lower(spec) -> float:
    return float(spec.metadata.get("agi_lower_bound", "-inf"))


table_1_1 = sorted(
    (s for s in specs if ".table_1_1." in s.name),
    key=lambda s: (s.metadata.get("source_measure_id", ""), lower(s)),
)
print("Table 1.1 specs bound:", len(table_1_1))
for spec in table_1_1:
    meta = spec.metadata
    print(
        f"  {spec.name:<62} {spec.value:.6g} | agi [{meta.get('agi_lower_bound')},"
        f"{meta.get('agi_upper_bound')}) | {meta.get('measure_mode')} | rebase "
        f"{meta.get('uprating_from_period')}->{meta.get('uprating_to_period')} "
        f"x{meta.get('uprating_factor')} | aging x{meta.get('aging_factor')}"
    )
print("national SOI specs with an AGI lower bound at or above $500k:")
for spec in specs:
    if spec.family != "irs_soi" or spec.metadata.get("state_fips"):
        continue
    if lower(spec) >= 500_000:
        print(f"  {spec.name} | {spec.value:.6g}")
with open(out_path, "w") as handle:
    json.dump(
        [
            {"name": s.name, "value": s.value, "metadata": dict(s.metadata)}
            for s in table_1_1
        ],
        handle,
        indent=1,
    )
