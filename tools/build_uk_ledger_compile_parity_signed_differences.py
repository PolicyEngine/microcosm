"""Regenerate UK Ledger compile-parity signed-difference resources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path

from microcosm.build.gates import ledger_compile_parity_signed_differences
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry

UK_PACKAGE = "microcosm.build.uk"
UK_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "uk"
)


@dataclass(frozen=True)
class ParityReceiptSpec:
    fixture_resource: str
    output_resource: str
    target_period: int


RECEIPTS = (
    ParityReceiptSpec(
        fixture_resource="parity_fixture_production_2023.json",
        output_resource="ledger_compile_parity_production_2023_signed_differences.json",
        target_period=2023,
    ),
    ParityReceiptSpec(
        fixture_resource="registry_parity_fixture_2025.json",
        output_resource="ledger_compile_parity_incumbent_2025_signed_differences.json",
        target_period=2025,
    ),
)

_CGT_GAINS_TOTAL_RATIONALE = (
    "Ledger carries the HMRC 2023-24 outturn value GBP 65,937,000,000 and "
    "holds it by identity under the current doctrine; the incumbent Fixture B "
    "row is GBP 67,727,478,991.60 at 2025 because it carries a forecast/uprated "
    "value. Signed as a doctrine consequence, not a binding error."
)

_ONS_TERMINAL_BAND_RATIONALES = {
    "ons.population.female_85_89": (
        "María ruling 2026-08-21: incumbent ons/female_85_90 maps to the "
        "uniform female 85_89 declaration. The value gap is the single-age-90 "
        "share; ages 90+ are constrained separately."
    ),
    "ons.population.male_85_89": (
        "María ruling 2026-08-21: incumbent ons/male_85_90 maps to the "
        "uniform male 85_89 declaration. The value gap is the single-age-90 "
        "share; ages 90+ are constrained separately."
    ),
    "ons.population.female_90_plus": (
        "María ruling 2026-08-21: new female 90_plus tail constrains ages 90+; "
        "the incumbent six-year terminal band left ages 91+ unconstrained."
    ),
    "ons.population.male_90_plus": (
        "María ruling 2026-08-21: new male 90_plus tail constrains ages 90+; "
        "the incumbent six-year terminal band left ages 91+ unconstrained."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Chronicle consumer JSONL artifact used to compile UK references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UK_PACKAGE_DIR,
        help="Directory receiving the signed-difference JSON resources.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_fixture(resource: str) -> dict:
    return json.loads(importlib_resources.files(UK_PACKAGE).joinpath(resource).read_text())


def _aligned_fixture(fixture: dict) -> dict:
    """Apply current contract-row names to retired-fixture rows before comparison."""

    rows = []
    for row in fixture.get("rows", ()):
        if not isinstance(row, dict):
            rows.append(row)
            continue
        updated = dict(row)
        if updated.get("name") == "ons/female_85_90":
            updated["contract_target_id"] = "ons.population.female_85_89"
            updated["measure"] = "ons.population.female_85_89"
        elif updated.get("name") == "ons/male_85_90":
            updated["contract_target_id"] = "ons.population.male_85_89"
            updated["measure"] = "ons.population.male_85_89"
        rows.append(updated)
    aligned = dict(fixture)
    aligned["rows"] = rows
    return aligned


def _add_signed_rationale_notes(
    report: dict[str, object],
    *,
    fixture_resource: str,
) -> None:
    if fixture_resource != "registry_parity_fixture_2025.json":
        return
    for row in report.get("differences", ()):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        if name == "hmrc.cgt.gains_total":
            row["reason"] = _CGT_GAINS_TOTAL_RATIONALE
        elif name in _ONS_TERMINAL_BAND_RATIONALES:
            row["reason"] = _ONS_TERMINAL_BAND_RATIONALES[name]


def main() -> None:
    args = _parse_args()
    facts = _load_jsonl(args.ledger_facts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in RECEIPTS:
        compilation = compile_uk_target_registry(
            facts,
            target_period=spec.target_period,
        )
        report = ledger_compile_parity_signed_differences(
            compilation.registry,
            _aligned_fixture(_load_fixture(spec.fixture_resource)),
            unsupported=compilation.unsupported,
        )
        _add_signed_rationale_notes(report, fixture_resource=spec.fixture_resource)
        output_path = args.output_dir / spec.output_resource
        output_path.write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "compiled_count": report["compiled_count"],
                    "counts_by_kind": report["counts_by_kind"],
                    "difference_count": report["difference_count"],
                    "fixture": spec.fixture_resource,
                    "fixture_count": report["fixture_count"],
                    "output": spec.output_resource,
                    "target_period": spec.target_period,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
