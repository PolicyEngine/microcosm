"""Compile the US fiscal target registry on two feeds and compare (microcosm#454).

    uv run python experiments/us-esi-454/feed-repin/compare_registries.py \
        <previous consumer_facts.jsonl> <new consumer_facts.jsonl> <report.json>

Compiles at target period 2024 with the canonical congressional-district
crosswalk and the reviewed Medicaid substitutions, once the way
tools/build_us_target_parity_manifest.py does (targets aged) and once with
aging off (dollar targets allowed unaged). Each registry is compared as an
ordered list, so a repeated, reordered, added, dropped or changed target shows
up; every field of every target is serialized. The report records the commit
the compiler ran at and whether its sources were clean there, since the
registry digests move whenever main changes the compiler.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# The compiler's sources, whose cleanliness the report records. This script
# (its target period and serialization) and uv.lock also shape the registry
# and its digests but are not checked; compiler.commit pins them only when
# they are unmodified there.
COMPILER_PATHS = ("packages", "tools")

CONFIGURATIONS = {
    "aged": {"age_targets": True},
    "unaged": {"age_targets": False, "allow_unaged_dollar_targets": True},
}


def _generator():
    path = ROOT / "tools" / "build_us_target_parity_manifest.py"
    spec = importlib.util.spec_from_file_location("parity_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry(generator, facts: list[dict], **options) -> list[str]:
    registry = generator.compile_us_fiscal_target_registry(
        facts,
        target_period=generator.TARGET_PERIOD,
        congressional_district_vintage_crosswalk=(
            generator.load_congressional_district_vintage_crosswalk(
                generator.default_congressional_district_vintage_crosswalk_path()
            )
        ),
        **options,
    )
    registry, _ = generator.apply_us_medicaid_enrollment_substitutions(registry)
    rows = []
    for spec in registry.specs:
        fields = (
            dataclasses.asdict(spec) if dataclasses.is_dataclass(spec) else vars(spec)
        )
        rows.append(json.dumps(fields, default=str, sort_keys=True))
    return rows


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _compiler() -> dict:
    dirty = _git("status", "--porcelain", "--", *COMPILER_PATHS).splitlines()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "paths": list(COMPILER_PATHS),
        "sources_clean": not dirty,
    }


def main() -> None:
    previous_path, new_path, report_path = (Path(arg) for arg in sys.argv[1:4])
    generator = _generator()
    previous_facts, _ = generator._load_feed(previous_path)
    new_facts, _ = generator._load_feed(new_path)
    report: dict[str, dict] = {"compiler": _compiler()}
    differs = False
    for label, options in CONFIGURATIONS.items():
        previous = _registry(generator, previous_facts, **options)
        new = _registry(generator, new_facts, **options)
        names = [json.loads(row)["name"] for row in new]
        first_difference = next(
            (i for i, (a, b) in enumerate(zip(previous, new, strict=False)) if a != b),
            None,
        )
        report[label] = {
            "options": options,
            "previous_targets": len(previous),
            "new_targets": len(new),
            "distinct_target_names": len(set(names)),
            "ordered_lists_identical": previous == new,
            "first_differing_index": first_difference,
            "registry_sha256": hashlib.sha256("\n".join(new).encode()).hexdigest(),
        }
        differs = differs or previous != new
    Path(report_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    if differs:
        raise SystemExit("the compiled registry differs between the feeds")


if __name__ == "__main__":
    main()
