#!/usr/bin/env python3
"""Emit the fail-closed F0 compiler coverage attestation.

Coverage has two independent parts.  The field-usage ledger assigns every
terminal authored or resolver-generated configuration field to one exact,
pinned compiler route.  The inventory report then checks the generation-0
authority structures named in the D1 journal.  Lossless resource copies and
generic spec hashing are never semantic evidence; hashing is accepted only by
the small, explicitly reviewed ``identity_only`` routes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    COMPILER_IR_ABI_VERSION,
    CompiledSpecIR,
    compile_spec,
)
from microcosm.build.spec_engine.field_usage import (
    EXPECTED_AUTHORED_FIELD_COUNT,
    EXPECTED_CONFIGURATION_FIELD_COUNT,
    EXPECTED_RESOLVED_BINDING_FIELD_COUNT,
    FieldUsageError,
    build_field_usage_ledger,
    default_usage_claims,
)
from microcosm.build.spec_engine.inventory_coverage import (
    InventoryCoverageError,
    assert_inventory_coverage_complete,
    build_inventory_coverage,
)
from microcosm.build.spec_engine.legacy_adapter import compile_to_legacy_payload
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import ResolvedSpec

REPORT_SCHEMA_VERSION = 3
EXPECTED_POINTER_INVENTORY_SHA256 = (
    "ffeeb52f5cdff5e4d516f270a4e45011b2cca255cab4c17c59d50a673a88e522"
)
DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "evidence"
    / "spec-engine"
    / "us-f0-coverage.json"
)


class CoverageError(AssertionError):
    """The compiler coverage proof is incomplete or its report is stale."""


def build_coverage_report(
    spec: ResolvedSpec,
    *,
    compiled: CompiledSpecIR | None = None,
    legacy_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the compact evidence report, refusing incomplete evidence."""

    if not isinstance(spec, ResolvedSpec):
        raise TypeError("build_coverage_report requires a ResolvedSpec")
    compiled = compile_spec(spec) if compiled is None else compiled
    legacy = (
        compile_to_legacy_payload(spec)
        if legacy_payload is None
        else dict(legacy_payload)
    )
    try:
        # Pass the caller's payload to the ledger so a copied compiler surface
        # cannot rescue a missing adapter sink.
        ledger = build_field_usage_ledger(
            spec,
            compiled=compiled,
            legacy_payload=legacy,
        )
        claims_by_id = {claim.id: claim for claim in default_usage_claims()}
        claim_rows = []
        for receipt in ledger.claims:
            claim = claims_by_id[receipt.id]
            claim_rows.append(
                {
                    **receipt.to_wire(),
                    "source_prefix": claim.source_prefix,
                    "mode": claim.mode.value,
                    "generation0_effect": claim.generation0_effect.value,
                    "consumer": claim.consumer,
                    "verifier": claim.verifier,
                    "pointer_class": claim.pointer_class,
                    "legacy_sinks": list(claim.legacy_sinks),
                    "relative_sink_prefix": claim.relative_sink_prefix,
                    "rationale": claim.rationale,
                }
            )
        authored_count = sum(
            field.pointer.startswith("/authored/") for field in ledger.fields
        )
        resolved_count = sum(
            field.pointer.startswith("/resolved/") for field in ledger.fields
        )
        field_usage = {
            "configuration_field_count": len(ledger.fields),
            "authored_normative_field_count": authored_count,
            "resolved_binding_field_count": resolved_count,
            "consumed_field_count": len(ledger.fields),
            "unused_field_count": 0,
            "multiple_primary_use_field_count": 0,
            "mode_counts": ledger.mode_counts,
            "generation0_effect_counts": ledger.generation0_effect_counts,
            "pointer_inventory_sha256": sha256_json(
                [field.pointer for field in ledger.fields]
            ),
            "claim_count": len(ledger.claims),
            "claims": claim_rows,
        }
        inventory = build_inventory_coverage(
            spec,
            compiled=compiled,
            legacy_payload=legacy,
        )
        assert_inventory_coverage_complete(inventory)
    except (FieldUsageError, InventoryCoverageError) as error:
        raise CoverageError(str(error)) from error

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass",
        "country": spec.country,
        "spec_binding": spec.spec_binding.to_wire(),
        "documentation_sha256": spec.documentation_sha256,
        "compiler_ir_abi": compiled.compiler_ir_abi.to_wire(),
        "field_usage": field_usage,
        "inventory_coverage": inventory,
    }
    assert_coverage_complete(report)
    return report


def assert_coverage_complete(report: Mapping[str, object]) -> None:
    """Recompute the compact report's claims and validate both proof axes."""

    failures: list[str] = []
    expected_top_level = {
        "report_schema_version",
        "status",
        "country",
        "spec_binding",
        "documentation_sha256",
        "compiler_ir_abi",
        "field_usage",
        "inventory_coverage",
    }
    if set(report) != expected_top_level:
        failures.append("report top-level fields differ")
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        failures.append(
            f"report schema version differs: {report.get('report_schema_version')!r}"
        )
    if report.get("status") != "pass":
        failures.append("report status is not pass")
    if report.get("country") != "us":
        failures.append(f"report country differs: {report.get('country')!r}")

    def valid_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    binding = report.get("spec_binding")
    if not isinstance(binding, Mapping):
        failures.append("spec_binding is missing")
    else:
        if set(binding) != {
            "attestation",
            "canonicalizer_version",
            "country",
            "schema_id",
            "schema_version",
            "spec_sha256",
        }:
            failures.append("spec_binding fields differ")
        if (
            binding.get("attestation") != "mirror-attested"
            or binding.get("canonicalizer_version") != 1
            or binding.get("country") != "us"
            or binding.get("schema_id") != "country_spec"
            or binding.get("schema_version") != 1
        ):
            failures.append("spec_binding contract differs")
        if not valid_sha256(binding.get("spec_sha256")):
            failures.append("spec_binding/spec_sha256 is invalid")

    compiler_abi = report.get("compiler_ir_abi")
    if not isinstance(compiler_abi, Mapping):
        failures.append("compiler_ir_abi is missing")
    else:
        if set(compiler_abi) != {"version", "sha256"} or compiler_abi.get(
            "version"
        ) != COMPILER_IR_ABI_VERSION:
            failures.append("compiler_ir_abi contract differs")
        if not valid_sha256(compiler_abi.get("sha256")):
            failures.append("compiler_ir_abi/sha256 is invalid")
    if not valid_sha256(report.get("documentation_sha256")):
        failures.append("documentation_sha256 is invalid")

    field_usage = report.get("field_usage")
    if not isinstance(field_usage, Mapping):
        failures.append("field_usage is missing")
    else:
        expected_field_usage_fields = {
            "configuration_field_count",
            "authored_normative_field_count",
            "resolved_binding_field_count",
            "consumed_field_count",
            "unused_field_count",
            "multiple_primary_use_field_count",
            "mode_counts",
            "generation0_effect_counts",
            "pointer_inventory_sha256",
            "claim_count",
            "claims",
        }
        if set(field_usage) != expected_field_usage_fields:
            failures.append("field_usage fields differ")
        expected = {
            "configuration_field_count": EXPECTED_CONFIGURATION_FIELD_COUNT,
            "authored_normative_field_count": EXPECTED_AUTHORED_FIELD_COUNT,
            "resolved_binding_field_count": EXPECTED_RESOLVED_BINDING_FIELD_COUNT,
            "consumed_field_count": EXPECTED_CONFIGURATION_FIELD_COUNT,
            "unused_field_count": 0,
            "multiple_primary_use_field_count": 0,
        }
        for name, expected_value in expected.items():
            if field_usage.get(name) != expected_value:
                failures.append(
                    f"field_usage/{name}: expected {expected_value}, "
                    f"got {field_usage.get(name)!r}"
                )
        claims = field_usage.get("claims")
        definitions = default_usage_claims()
        if not isinstance(claims, list):
            failures.append("field_usage claim receipts are incomplete")
            claims = []
        if field_usage.get("claim_count") != len(definitions) or len(
            claims
        ) != len(definitions):
            failures.append(
                "field_usage claim count differs from the closed claim registry"
            )
        expected_claim_fields = {
            "id",
            "pointer_count",
            "pointer_sha256",
            "source_prefix",
            "mode",
            "generation0_effect",
            "consumer",
            "verifier",
            "pointer_class",
            "legacy_sinks",
            "relative_sink_prefix",
            "rationale",
        }
        claim_ids = [
            row.get("id") if isinstance(row, Mapping) else None for row in claims
        ]
        expected_ids = [claim.id for claim in definitions]
        claim_ids_are_strings = all(isinstance(value, str) for value in claim_ids)
        if (
            claim_ids != expected_ids
            or not claim_ids_are_strings
            or len(set(claim_ids)) != len(claim_ids)
        ):
            failures.append("field_usage claim ids or order differ")

        derived_modes: dict[str, int] = {}
        derived_effects: dict[str, int] = {}
        derived_total = 0
        for index, definition in enumerate(definitions):
            if index >= len(claims) or not isinstance(claims[index], Mapping):
                continue
            row = claims[index]
            if set(row) != expected_claim_fields:
                failures.append(f"field_usage claim {definition.id} fields differ")
                continue
            expected_row = {
                "id": definition.id,
                "pointer_count": definition.expected_pointer_count,
                "pointer_sha256": definition.expected_pointer_sha256,
                "source_prefix": definition.source_prefix,
                "mode": definition.mode.value,
                "generation0_effect": definition.generation0_effect.value,
                "consumer": definition.consumer,
                "verifier": definition.verifier,
                "pointer_class": definition.pointer_class,
                "legacy_sinks": list(definition.legacy_sinks),
                "relative_sink_prefix": definition.relative_sink_prefix,
                "rationale": definition.rationale,
            }
            if dict(row) != expected_row:
                failures.append(f"field_usage claim {definition.id} differs")
                continue
            count = definition.expected_pointer_count
            derived_total += count
            derived_modes[definition.mode.value] = (
                derived_modes.get(definition.mode.value, 0) + count
            )
            derived_effects[definition.generation0_effect.value] = (
                derived_effects.get(definition.generation0_effect.value, 0) + count
            )
        if derived_total != EXPECTED_CONFIGURATION_FIELD_COUNT:
            failures.append(
                "closed claim registry does not sum to the configuration universe"
            )
        if field_usage.get("mode_counts") != derived_modes:
            failures.append("field_usage mode_counts do not match claim expansion")
        if field_usage.get("generation0_effect_counts") != derived_effects:
            failures.append(
                "field_usage generation0_effect_counts do not match claim expansion"
            )
        if (
            field_usage.get("pointer_inventory_sha256")
            != EXPECTED_POINTER_INVENTORY_SHA256
        ):
            failures.append("field_usage pointer_inventory_sha256 differs")

    inventory = report.get("inventory_coverage")
    if not isinstance(inventory, Mapping):
        failures.append("inventory_coverage is missing")
    else:
        try:
            assert_inventory_coverage_complete(inventory)
        except InventoryCoverageError as error:
            failures.append(str(error))
        inventory_binding = inventory.get("spec_binding")
        if isinstance(binding, Mapping) and inventory_binding != binding:
            failures.append("inventory and coverage spec bindings differ")
        inventory_abi = inventory.get("compiler_ir_abi")
        if isinstance(compiler_abi, Mapping) and inventory_abi != compiler_abi:
            failures.append("inventory and coverage compiler IR ABIs differ")

    if failures:
        raise CoverageError("spec-engine coverage failed:\n- " + "\n- ".join(failures))


def coverage_report_bytes(report: Mapping[str, object]) -> bytes:
    """Return stable, reviewable report bytes."""

    return (
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="coverage report path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="refuse when the committed report differs instead of writing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_coverage_report(load_bundle("us"))
    payload = coverage_report_bytes(report)
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != payload:
            raise CoverageError(f"coverage report is stale: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    fields = report["field_usage"]
    inventory = report["inventory_coverage"]
    assert isinstance(fields, Mapping)
    assert isinstance(inventory, Mapping)
    print(
        "spec-engine coverage: "
        f"{fields['consumed_field_count']}/{fields['configuration_field_count']} "
        "configuration fields; "
        f"{inventory['covered_item_count']}/{inventory['required_item_count']} "
        "inventory checks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
