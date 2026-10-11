"""Compile the US fiscal target registry on two feeds and compare (microcosm#454).

    uv run python experiments/us-esi-454/feed-repin/compare_registries.py \
        <previous consumer_facts.jsonl> <new consumer_facts.jsonl> <report.json>

Uses the same compile as tools/build_us_target_parity_manifest.py (target
period 2024, age targets, canonical congressional-district crosswalk, the
reviewed Medicaid substitutions) and compares every target's name, value and
metadata.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _generator():
    path = ROOT / "tools" / "build_us_target_parity_manifest.py"
    spec = importlib.util.spec_from_file_location("parity_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry(generator, path: Path) -> dict[str, dict]:
    facts, _ = generator._load_feed(path)
    registry = generator.compile_us_fiscal_target_registry(
        facts,
        target_period=generator.TARGET_PERIOD,
        congressional_district_vintage_crosswalk=(
            generator.load_congressional_district_vintage_crosswalk(
                generator.default_congressional_district_vintage_crosswalk_path()
            )
        ),
        age_targets=True,
    )
    registry, _ = generator.apply_us_medicaid_enrollment_substitutions(registry)
    out = {}
    for spec in registry.specs:
        fields = (
            dataclasses.asdict(spec) if dataclasses.is_dataclass(spec) else vars(spec)
        )
        out[spec.name] = json.loads(json.dumps(fields, default=str, sort_keys=True))
    return out


def main() -> None:
    previous_path, new_path, report_path = (Path(arg) for arg in sys.argv[1:4])
    generator = _generator()
    previous = _registry(generator, previous_path)
    new = _registry(generator, new_path)
    report = {
        "previous_targets": len(previous),
        "new_targets": len(new),
        "only_previous": sorted(set(previous) - set(new)),
        "only_new": sorted(set(new) - set(previous)),
        "changed": sorted(n for n in previous if n in new and previous[n] != new[n]),
    }
    Path(report_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1))
    if report["only_previous"] or report["only_new"] or report["changed"]:
        raise SystemExit("the compiled registry differs between the feeds")


if __name__ == "__main__":
    main()
