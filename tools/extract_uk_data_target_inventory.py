#!/usr/bin/env python
"""Statically extract the pinned uk-data target and credibility inventory.

The inventory is a sha-stamped snapshot of the incumbent's target-bearing
modules at one commit. It is re-extracted manually against a local checkout
and committed; CI checks the parity register against the committed snapshot,
and ``verify_uk_data_target_inventory_against_tree`` (run by the env-gated
test, or by hand) re-hashes it against a tree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import date
from pathlib import Path

DEFAULT_OUTPUT = Path(
    "packages/microcosm-build/src/microcosm/build/uk/uk_data_target_inventory.json"
)
PACKAGE_ROOT = "policyengine_" + "uk_data"

# Modules that carry targets but sit outside a globbable directory.
LOCAL_TARGET_MODULES = {
    "datasets/local_areas/constituencies/devolved_housing.py": (
        "local_target_producer"
    ),
    "datasets/local_areas/constituencies/loss.py": "local_matrix",
    "datasets/local_areas/local_authorities/loss.py": "local_matrix",
}

# Every per-grain target-producer script is globbed, so a new grain or a new
# script under any ``local_areas/*/targets/`` directory lands in the
# inventory and must be covered by a parity concern.
LOCAL_TARGET_PRODUCER_GLOB = "datasets/local_areas/*/targets/*.py"

DATASET_ANCHOR_MODULES = {
    "datasets/imputations/consumption.py",
    "datasets/imputations/services/services.py",
    "datasets/imputations/vat.py",
}

# These tests are the incumbent checks accounted for by the parity register's
# credibility concerns. Unit tests for unrelated ingestion and imputation
# mechanics are deliberately outside this target/anchor inventory.
CREDIBILITY_GATE_MODULES = {
    "tests/microsimulation/test_reform_impacts.py",
    "tests/test_aggregates.py",
    "tests/test_bus_fare_distribution.py",
    "tests/test_la_council_tax_targets.py",
    "tests/test_la_loss_council_tax.py",
    "tests/test_la_loss_missing_sources.py",
    "tests/test_obr_nic_signal.py",
    "tests/test_population.py",
    "tests/test_population_fidelity.py",
    "tests/test_public_sector_employment_target.py",
    "tests/test_publish_local_h5s.py",
    "tests/test_regional_land_value_targets.py",
    "tests/test_target_db.py",
    "tests/test_target_registry.py",
    "tests/test_vehicle_ownership.py",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uk-data-tree", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--extracted-on",
        type=date.fromisoformat,
        required=True,
        help=(
            "ISO date recorded in the artifact. Passed explicitly so a "
            "re-extraction of an unchanged tree is byte-stable."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _inventory_id(path: Path) -> str:
    return str(path.with_suffix("")).replace("/", ".")


def _entry(tree: Path, relative: Path, *, kind: str) -> dict[str, str]:
    path = tree / relative
    if not path.is_file():
        raise ValueError(f"uk-data inventory path does not exist: {relative}.")
    source = path.read_bytes()
    try:
        ast.parse(source, filename=str(relative))
    except SyntaxError as error:
        raise ValueError(
            f"cannot statically parse uk-data module {relative}."
        ) from error
    return {
        "inventory_id": _inventory_id(relative),
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(source).hexdigest(),
        "kind": kind,
    }


def build_inventory(
    tree: Path, *, commit: str, extracted_on: date
) -> dict[str, object]:
    """Build the canonical inventory without importing the archived package."""

    package = tree / PACKAGE_ROOT
    if not package.is_dir():
        raise ValueError(
            f"--uk-data-tree must contain the archived {PACKAGE_ROOT} package."
        )
    relative_kinds: dict[Path, str] = {}
    source_root = Path(PACKAGE_ROOT, "targets/sources")
    for source_path in sorted((tree / source_root).glob("*.py")):
        relative = source_path.relative_to(tree)
        relative_kinds[relative] = (
            "helper" if source_path.name.startswith("_") else "target_source"
        )
    for producer_path in sorted(package.glob(LOCAL_TARGET_PRODUCER_GLOB)):
        relative = producer_path.relative_to(tree)
        relative_kinds[relative] = (
            "helper" if producer_path.name.startswith("_") else "local_target_producer"
        )

    for relative, kind in LOCAL_TARGET_MODULES.items():
        relative_kinds[Path(PACKAGE_ROOT, relative)] = kind
    for relative in DATASET_ANCHOR_MODULES:
        relative_kinds[Path(PACKAGE_ROOT, relative)] = "dataset_anchor"
    for relative in CREDIBILITY_GATE_MODULES:
        relative_kinds[Path(PACKAGE_ROOT, relative)] = "credibility_gate"

    entries = [
        _entry(tree, relative, kind=kind) for relative, kind in relative_kinds.items()
    ]
    entries.sort(key=lambda row: row["inventory_id"])
    return {
        "schema_version": 1,
        "incumbent_commit": str(commit),
        "extracted_on": extracted_on.isoformat(),
        "entries": entries,
    }


def main() -> None:
    args = _parser().parse_args()
    inventory = build_inventory(
        args.uk_data_tree,
        commit=args.commit,
        extracted_on=args.extracted_on,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
