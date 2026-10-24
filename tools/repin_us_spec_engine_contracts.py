#!/usr/bin/env python3
"""Recompute the reviewed US F0 migration pins after an intentional change.

Run ``--legacy-only`` before regenerating the US bundle, then run without it
before regenerating ``tools/spec_engine_coverage.py``. Only observed hashes,
field counts, and their exact receipt and semantic test vectors are rewritten. Compiler
routes and inventory clauses remain authoritative: the final coverage command
validates the full inventory after pins are written. No microdata, build,
publication, or remote access is involved.

    uv run --no-sync python tools/repin_us_spec_engine_contracts.py --legacy-only
    uv run --no-sync python tools/generate_us_bundle_from_constants.py
    uv run --no-sync python tools/repin_us_spec_engine_contracts.py
    uv run --no-sync python tools/spec_engine_coverage.py

Use ``--check`` to refuse stale pins without modifying files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tokenize
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/microcosm-build"
BUILD = PACKAGE / "src/microcosm/build"
FIELD_USAGE = BUILD / "spec_engine/field_usage.py"
INVENTORY = BUILD / "spec_engine/inventory_coverage.py"
FIELD_TEST = PACKAGE / "tests/engine_free/shared/test_spec_engine_field_usage.py"
COVERAGE_TEST = PACKAGE / "tests/engine_contract/us/test_spec_engine_coverage_tool.py"


def _assignment(source: str, name: str) -> ast.expr:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return node.value
    raise ValueError(f"missing assignment: {name}")


def _replace_node(source: str, node: ast.AST, value: object) -> str:
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[: node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset
    return source[:start] + repr(value) + source[end:]


def _replace_assignment(source: str, name: str, value: object) -> str:
    node = _assignment(source, name)
    if ast.literal_eval(node) == value:
        return source
    return _replace_node(source, node, value)


def _replace_dictionary(source: str, name: str, values: dict) -> str:
    for key, value in values.items():
        node = _assignment(source, name)
        if not isinstance(node, ast.Dict):
            raise TypeError(f"{name} is not a literal dictionary")
        matching = [
            child
            for candidate, child in zip(node.keys, node.values, strict=True)
            if candidate is not None and ast.literal_eval(candidate) == key
        ]
        if len(matching) != 1:
            raise ValueError(f"{name}: expected one key {key!r}")
        if ast.literal_eval(matching[0]) == value:
            continue
        source = _replace_node(source, matching[0], value)
    return source


def _replace_literals(source: str, replacements: dict) -> str:
    """Replace exact Python literals, leaving comments and substrings alone."""

    edits = []
    lines = source.splitlines(keepends=True)
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type not in {tokenize.NUMBER, tokenize.STRING}:
            continue
        try:
            old = ast.literal_eval(token.string)
        except (ValueError, SyntaxError):
            continue
        if old in replacements and replacements[old] != old:
            start = sum(map(len, lines[: token.start[0] - 1])) + token.start[1]
            end = sum(map(len, lines[: token.end[0] - 1])) + token.end[1]
            edits.append((start, end, repr(replacements[old])))
    for start, end, value in reversed(edits):
        source = source[:start] + value + source[end:]
    return source


def _save(path: Path, source: str, *, check: bool) -> None:
    if source == path.read_text():
        return
    relative = path.relative_to(ROOT)
    if check:
        raise SystemExit(f"stale US F0 pins: {relative}")
    path.write_text(source)
    print(f"updated {relative}")


def _add_count_replacement(replacements: dict, old: int, new: int) -> None:
    if old in replacements and replacements[old] != new:
        raise ValueError(f"ambiguous count replacement for {old}")
    replacements[old] = new


def _replace_length_assertions(source: str, expected: dict[str, int]) -> str:
    matches = {}
    for assertion in ast.walk(ast.parse(source)):
        if not isinstance(assertion, ast.Assert):
            continue
        node = assertion.test
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
            and len(node.left.args) == 1
            and len(node.comparators) == 1
        ):
            continue
        target = ast.unparse(node.left.args[0])
        if target in expected:
            if target in matches:
                raise ValueError(f"duplicate inventory assertion: {target}")
            matches[target] = node.comparators[0]
    if set(matches) != set(expected):
        raise ValueError("typed inventory assertion targets changed")
    for target, node in sorted(
        matches.items(), key=lambda row: row[1].lineno, reverse=True
    ):
        if ast.literal_eval(node) != expected[target]:
            source = _replace_node(source, node, expected[target])
    return source


def _replace_literal_assertions(source: str, expected: dict[str, str]) -> str:
    """Refresh each exact receipt expression once, retaining literal assertions."""

    matches = {}
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
        ):
            continue
        target = ast.unparse(node.test.left)
        if target in expected:
            if target in matches:
                raise ValueError(f"duplicate receipt assertion: {target}")
            matches[target] = node.test.comparators[0]
    if set(matches) != set(expected):
        raise ValueError("receipt assertion targets changed")
    for target, node in sorted(
        matches.items(), key=lambda row: row[1].lineno, reverse=True
    ):
        old = ast.literal_eval(node)
        if not isinstance(old, str):
            raise ValueError(f"receipt assertion is not a literal string: {target}")
        if old != expected[target]:
            source = _replace_node(source, node, expected[target])
    return source


def _replace_asserted_dictionary_value(
    source: str, target: str, keys: tuple[str, ...], value: str
) -> str:
    """Refresh a nested literal pin inside one complete dictionary assertion."""

    matching = [
        node.test.comparators[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and ast.unparse(node.test.left) == target
    ]
    if len(matching) != 1:
        raise ValueError(f"expected one asserted dictionary: {target}")
    node = matching[0]
    for key in keys:
        if not isinstance(node, ast.Dict):
            raise ValueError(f"receipt key is not in a literal dictionary: {key}")
        matching = [
            child
            for candidate, child in zip(node.keys, node.values, strict=True)
            if candidate is not None and ast.literal_eval(candidate) == key
        ]
        if len(matching) != 1:
            raise ValueError(f"expected one receipt dictionary key: {key}")
        node = matching[0]
    old = ast.literal_eval(node)
    if not isinstance(old, str):
        raise ValueError("receipt dictionary pin is not a literal string")
    return source if old == value else _replace_node(source, node, value)


def _semantic_test_pins(spec, observed_hashes: dict, rendered: dict) -> None:
    """Derive the existing test vectors through their production projections."""

    from microcosm.build.spec_engine import (
        KernelRegistry,
        ResourceKind,
        project_legacy_calibration_contract,
    )
    from microcosm.build.spec_engine.canonical import sha256_json
    from microcosm.build.spec_engine.legacy_adapter import compile_to_legacy_payload
    from microcosm.build.spec_engine.loader import load_bundle
    from test_support.microcosm_build.spec_engine_loader import (
        SELECTION_KERNEL_IDS,
        _rich_minimal,
    )

    legacy = compile_to_legacy_payload(spec)
    imputation = legacy["imputation"]
    schedule = imputation["late_producer_schedule_receipt"]
    receipts = {
        "authority": legacy["stacked_authority_receipt"]["sha256"],
        "late_schedule": schedule["schedule_sha256"],
        "ownership": imputation["overlap_ownership"]["sha256"],
        "take_up": legacy["take_up_contract_identity"]["resource_sha256"],
    }
    for key, value in receipts.items():
        if value != observed_hashes[key]:
            raise ValueError(f"{key}: compiled and inventory receipt digests differ")

    # Use the identical fixture builder, country, note, store and selected
    # registry as the golden test; no test assertion or constant is consulted.
    with TemporaryDirectory(prefix="microcosm-spec-pin-") as directory:
        root = _rich_minimal(Path(directory) / "xx", note="first", store="local:a")
        minimal = load_bundle(
            root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
        )
    path = PACKAGE / "tests/engine_free/shared/test_spec_engine_loader.py"
    rendered[path] = _replace_literal_assertions(
        path.read_text(), {"first.spec_sha256": minimal.spec_sha256}
    )

    calibration = spec.domain(ResourceKind.CALIBRATION).to_wire()
    path = (
        PACKAGE / "tests/engine_free/shared/test_spec_engine_calibration_semantics.py"
    )
    rendered[path] = _replace_assignment(
        path.read_text(),
        "CANONICAL_LEGACY_PROJECTION_SHA256",
        sha256_json(project_legacy_calibration_contract(calibration)),
    )

    path = PACKAGE / "tests/engine_free/shared/test_spec_engine_compiler_ir.py"
    rendered[path] = _replace_assignment(
        rendered.get(path, path.read_text()),
        "US_SCHEDULE_SHA256",
        receipts["late_schedule"],
    )
    assertion_files = {
        "engine_free/shared/test_spec_engine_legacy_adapter.py": {
            "legacy_payload['stacked_authority_receipt']['sha256']": receipts[
                "authority"
            ],
            "imputation['late_producer_schedule_receipt']['schedule_sha256']": receipts[
                "late_schedule"
            ],
            "imputation['overlap_ownership']['sha256']": receipts["ownership"],
            "legacy_payload['take_up_contract_identity']['resource_sha256']": receipts[
                "take_up"
            ],
        },
        "engine_free/shared/test_spec_engine_imputation_semantics.py": {
            "projected['late_producer_schedule_receipt']['schedule_sha256']": receipts[
                "late_schedule"
            ],
            "projected['overlap_ownership']['sha256']": receipts["ownership"],
        },
        "engine_free/us/test_us_spec_bundle.py": {
            "compiled_schedule['schedule_sha256']": receipts["late_schedule"],
            "compiled_schedule['payload_sha256']": schedule["payload_sha256"],
        },
    }
    for relative, expected in assertion_files.items():
        path = PACKAGE / "tests" / relative
        rendered[path] = _replace_literal_assertions(
            rendered.get(path, path.read_text()), expected
        )
    path = PACKAGE / "tests/engine_workflow/us/test_us_multispine_pool_tool.py"
    rendered[path] = _replace_asserted_dictionary_value(
        path.read_text(),
        "run_config",
        ("spec_binding", "spec_sha256"),
        spec.spec_sha256,
    )


def _legacy_pins(*, check: bool) -> None:
    generator = ROOT / "tools/generate_us_bundle_from_constants.py"
    observed = hashlib.sha256(
        (BUILD / "us/source_stages.json").read_bytes()
    ).hexdigest()
    for path, name in (
        (generator, "FROZEN_LEGACY_RESOURCE_SHA256"),
        (
            ROOT / "test_support/microcosm_build/us_spec_bundle.py",
            "LEGACY_COMPATIBILITY_SHA256",
        ),
    ):
        source = _replace_dictionary(
            path.read_text(), name, {"source_stages.json": observed}
        )
        _save(path, source, check=check)
    path = PACKAGE / "tests/engine_free/us/test_us_bundle_core_contracts.py"
    source = path.read_text()
    pins = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        identifier = pairs.get("id")
        if isinstance(identifier, ast.Constant) and identifier.value == "source_stages":
            pins.append(pairs["sha256"])
    if len(pins) != 1:
        raise ValueError("expected one source-stage asset test pin")
    if ast.literal_eval(pins[0]) != observed:
        source = _replace_node(source, pins[0], observed)
    _save(path, source, check=check)


def _compiler_pins(*, check: bool) -> None:
    # Imports follow the legacy-only bootstrap so stale generation-0 pins cannot
    # prevent the bootstrap itself from reading and hashing its local resource.
    sys.path.insert(0, str(ROOT))
    from microcosm.build.spec_engine import field_usage, inventory_coverage
    from microcosm.build.spec_engine.canonical import sha256_json
    from microcosm.build.spec_engine.loader import load_bundle

    spec = load_bundle("us")
    expansions = field_usage.claim_expansion_inventory(spec)
    sources = field_usage.configuration_sources(spec)
    pointers = [path for path, _ in field_usage._terminal_rows(sources, path="")]
    counts = {
        "EXPECTED_AUTHORED_FIELD_COUNT": sum(
            p.startswith("/authored/") for p in pointers
        ),
        "EXPECTED_RESOLVED_BINDING_FIELD_COUNT": sum(
            p.startswith("/resolved/") for p in pointers
        ),
        "EXPECTED_CONFIGURATION_FIELD_COUNT": len(pointers),
    }
    old_counts = {name: getattr(field_usage, name) for name in counts}
    source = FIELD_USAGE.read_text()
    source = _replace_dictionary(source, "_PINS", expansions)
    for name, value in counts.items():
        source = _replace_assignment(source, name, value)
    rendered = {FIELD_USAGE: source}

    # Derive an independently validated ledger using the observed expansions.
    # Every verifier and exact single-owner check still runs; only stale pin
    # values and totals are supplied from the source just inspected above.
    from dataclasses import replace

    claims = tuple(
        replace(
            claim,
            expected_pointer_count=expansions[claim.id][0],
            expected_pointer_sha256=expansions[claim.id][1],
        )
        for claim in field_usage.default_usage_claims()
    )
    ledger = field_usage.build_field_usage_ledger(
        spec, claims=claims, enforce_expected_total=False
    )
    coverage_tool = ROOT / "tools/spec_engine_coverage.py"
    source = _replace_assignment(
        coverage_tool.read_text(),
        "EXPECTED_POINTER_INVENTORY_SHA256",
        sha256_json(sorted(pointers)),
    )
    rendered[coverage_tool] = source

    report = inventory_coverage.build_inventory_coverage(spec)
    observed_hashes = {}
    # Match each named pin through its exact expected receipt field. Pins may
    # appear in multiple receipts; all independently observed values must agree.
    for key, old in inventory_coverage.EXPECTED_HASHES.items():
        observed = set()
        for item in report["items"].values():
            expected = item["expected"]
            if not isinstance(expected, dict):
                continue
            for field, value in expected.items():
                if value == old:
                    actual_field = (
                        "families_sha256"
                        if field == "sha256" and "families_sha256" in item["observed"]
                        else field
                    )
                    actual = item["observed"].get(actual_field)
                    if isinstance(actual, str):
                        observed.add(actual)
        if len(observed) != 1:
            raise ValueError(f"{key}: no unambiguous observed digest: {observed}")
        observed_hashes[key] = observed.pop()
    source = _replace_dictionary(
        INVENTORY.read_text(), "EXPECTED_HASHES", observed_hashes
    )
    source = _replace_dictionary(source, "EXPECTED_INVENTORY_COUNTS", report["counts"])
    old_tail = inventory_coverage.EXPECTED_INVENTORY_COUNTS["tail_control_fields"]
    new_tail = report["counts"]["tail_control_fields"]
    source = _replace_literals(source, {old_tail: new_tail})
    rendered[INVENTORY] = source

    # Tests deliberately pin literal counts, so derive their corresponding
    # expectations from the same validated ledger rather than weakening them.
    old_report = json.loads(
        (ROOT / "docs/evidence/spec-engine/us-f0-coverage.json").read_text()
    )
    replacements = {}
    for name, value in counts.items():
        _add_count_replacement(replacements, old_counts[name], value)
    for group, observed in (
        ("mode_counts", ledger.mode_counts),
        ("generation0_effect_counts", ledger.generation0_effect_counts),
    ):
        for key, value in observed.items():
            _add_count_replacement(
                replacements, old_report["field_usage"][group][key], value
            )
    _add_count_replacement(replacements, old_tail, new_tail)
    for path in (FIELD_TEST, COVERAGE_TEST):
        rendered[path] = _replace_literals(path.read_text(), replacements)
    observed_counts = report["counts"]
    column_count = observed_counts["typed_columns"]
    artifact_count = observed_counts["typed_artifacts"]
    scope_count = observed_counts["typed_scopes"]
    assertion_files = {
        "engine_free/us/test_us_spec_bundle.py": {
            "catalogs['columns']": column_count,
            "resolved_us_spec.columns": column_count,
            "resolved_us_spec.scopes": scope_count,
        },
        "engine_free/shared/test_spec_engine_compiler_ir.py": {
            "inventory['entities']": observed_counts["typed_entities"],
            "inventory['columns']": column_count,
            "inventory['artifacts']": artifact_count,
            "inventory['scopes']": scope_count,
            "inventory['references']": observed_counts["resolved_references"],
        },
        "engine_contract/us/test_spec_engine_typed_closure.py": {
            "typed_result.columns": column_count,
            "typed_result.artifacts": artifact_count,
            "typed_result.scopes": scope_count,
        },
        "engine_contract/us/test_us_bundle_core_contracts.py": {
            "columns": column_count,
            "{row['key'] for row in columns}": column_count,
            "resolved.columns": column_count,
        },
    }
    for relative, expected in assertion_files.items():
        path = PACKAGE / "tests" / relative
        rendered[path] = _replace_length_assertions(path.read_text(), expected)
    path = ROOT / "test_support/microcosm_build/spec_engine_inventory_coverage.py"
    rendered[path] = _replace_dictionary(
        path.read_text(), "EXPECTED_COUNTS", observed_counts
    )
    _semantic_test_pins(spec, observed_hashes, rendered)
    # Finish all derivations before writing so a failed compiler route or an
    # ambiguous inventory observation cannot leave partially refreshed pins.
    for path, source in rendered.items():
        _save(path, source, check=check)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _legacy_pins(check=args.check)
    if not args.legacy_only:
        _compiler_pins(check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
