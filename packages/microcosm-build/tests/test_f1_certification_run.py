from __future__ import annotations

import copy
import importlib.util
import json
import uuid
from pathlib import Path
from types import ModuleType

import pytest

from microcosm.build.spec_engine.artifact_selector_contract import (
    ARTIFACT_LOCATOR_GRAMMAR,
    ARTIFACT_SELECTOR_CONTRACT_SHA256,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import current_compiler_ir_abi
from microcosm.build.spec_engine.executor import (
    RunProvenanceIdentity,
    build_run_provenance_identity,
)
from microcosm.build.spec_engine.f1_certification import (
    CERTIFICATION_JSON_FILENAME,
    CERTIFICATION_MARKDOWN_FILENAME,
    COLD_BUILD_RECEIPT_FILENAME,
    F1ColdBuildReceipt,
    F1CoverageEvidence,
    F1ProductionEvidence,
    F1RunRequest,
    assert_f1_selector_coverage_contract_current,
    atomic_write_json,
    compare_f1_cold_build_receipts,
    complete_coverage_evidence,
)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "f1_certification_run.py"
    spec = importlib.util.spec_from_file_location("f1_certification_run_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_grant() -> dict[str, object]:
    sources: list[dict[str, object]] = []
    unsigned = {
        "domain": "microcosm.spec-engine.source-broker-grant.v1",
        "owner": {"kind": "source_stage", "id": "declared_source_preflight"},
        "effects": ["declared_source_read"],
        "sources": sources,
        "source_set_sha256": sha256_json(
            {
                "domain": "microcosm.spec-engine.source-broker-grant.v1",
                "sources": sources,
            }
        ),
    }
    return {**unsigned, "sha256": sha256_json(unsigned)}


def _plan_lock() -> dict[str, object]:
    artifact = {
        "id": "payload",
        "kind": "fixture_payload",
        "producer_ref": "node:z",
        "stage_ref": "published",
        "protocol_ref": "fixture:bytes-v1",
        "locator_ref": "fixture:payload",
        "content_selector_ref": "selector:file_bytes_v1",
        "surface": "normative",
        "comparison": "raw_byte_exact",
        "required": True,
    }
    code_unsigned = {
        "domain": "fixture-f1-certification-v1",
        "content_selectors": ["selector:file_bytes_v1"],
        "locator_grammar": ARTIFACT_LOCATOR_GRAMMAR,
        "artifact_selector_contract_sha256": ARTIFACT_SELECTOR_CONTRACT_SHA256,
        "compiler_ir_abi_sha256": current_compiler_ir_abi().sha256,
        "receipt_difference_match": "exactly_one_sealed_rule",
    }
    code_abi = {
        **code_unsigned,
        "implementation_sha256": sha256_json(code_unsigned),
    }
    unsigned = {
        "schema_version": 1,
        "present": True,
        "pipeline": {
            "id": "fixture",
            "artifact_protocol": {"fixture": "bytes-v1"},
            "operator_order": ["publish"],
            "producer_order": ["node:z", "node:a"],
            "seed_stream_map_sha256": "5" * 64,
        },
        "operations": [],
        "logical_stages": [],
        "durable_checkpoints": [],
        "code_abi": code_abi,
        "normative_artifact_vector": [artifact],
        "artifact_bindings": [],
        "source_broker_grant": _source_grant(),
        "receipt_comparison_vector": [
            {
                "artifact_role": "publication_manifest",
                "json_pointer_pattern": "/operational",
                "rule": "operational_excluded",
                "category": "fixture",
            }
        ],
        "resume_predicate": None,
    }
    execution_abi = {**unsigned, "sha256": sha256_json(unsigned)}
    return {
        "compiler_ir_abi": current_compiler_ir_abi().to_wire(),
        "grammar_receipt": {
            "schema_version": 3,
            "canonicalizer_version": 1,
            "migration_chain": [],
        },
        "spec_binding": {
            "country": "zz",
            "schema_id": "country_spec",
            "schema_version": 3,
            "canonicalizer_version": 1,
            "spec_sha256": "b" * 64,
            "attestation": "mirror-attested",
        },
        "surfaces": {
            name: {}
            for name in (
                "normative",
                "run_request",
                "execution_profile",
                "operational",
                "chain_state",
                "documentation",
            )
        },
        "typed_inventory": {
            name: []
            for name in ("entities", "artifacts", "scopes", "columns", "references")
        },
        "authorities": {"generated": {}, "vintages": {"records": {}}},
        "runtime_authorities": {
            "schema_version": 1,
            "surfaces": {
                name: {}
                for name in (
                    "normative",
                    "execution_profile",
                    "run_request",
                    "behavior",
                )
            },
            "domain_sha256": {
                name: {}
                for name in (
                    "normative",
                    "execution_profile",
                    "run_request",
                    "behavior",
                )
            },
            "sha256": "c" * 64,
        },
        "execution_abi": execution_abi,
        "stage_dag": {
            "external_stages": [],
            "nodes": [],
            "edges": [],
            "waves": [],
            "order": [],
        },
        "producer_graph": {
            "present": False,
            "nodes": [],
            "edges": [],
            "waves": [],
            "order": [],
            "incomparable_node_policy": {
                "requirement": "commute_or_disjoint_writes",
                "proof_method": (
                    "transitive_closure_and_closed_cell_segment_intersection"
                ),
                "overlap_rule": "explicit_commutativity_proof_required",
                "commutativity_proofs": [],
                "incomparable_pair_count": 0,
                "disjoint_write_pair_count": 0,
            },
            "schedule_sha256": "d" * 64,
        },
        "seed_stream_map": {
            "protocol_id": "legacy-v1",
            "implementation_id": "fixture",
            "implementation_sha256": "e" * 64,
            "sites": [],
            "owners": [],
        },
        "nodes": [],
    }


def _identities(
    plan: dict[str, object],
) -> tuple[RunProvenanceIdentity, RunProvenanceIdentity]:
    execution_abi = plan["execution_abi"]
    assert isinstance(execution_abi, dict)
    bundle = build_run_provenance_identity(
        identity_generation=1,
        source_grammar_receipt={
            "schema_version": 3,
            "canonicalizer_version": 1,
            "migration_chain": [],
        },
        spec_binding={
            "country": "fixture",
            "schema_id": "country-spec",
            "schema_version": 3,
            "canonicalizer_version": 1,
            "spec_sha256": "b" * 64,
            "attestation": "bundle-authoritative",
        },
        authority_versions={
            "stacked_authority": "1" * 64,
            "checkpoint_materializer": "2" * 64,
            "runtime_authority": "c" * 64,
            "execution_abi": execution_abi["sha256"],
        },
        code_inventory_digest="d" * 64,
        artifact_protocol_inventory={"fixture": "bytes-v1"},
        run_request={
            "pipeline": "fixture",
            "sample_fraction": 0.01,
            "fraction_token": "f001",
            "sample_seed": 578,
            "clone_attachment_fraction": 1.0,
            "clone_attachment_seed": 578,
        },
        execution_receipt={
            "authority_mode": "bundle",
            "pipeline": "fixture",
            "code_pin": "fixture-code-pin",
        },
    )
    wire = bundle.to_wire()
    bundle_versions = wire["authority_versions"]
    bundle_execution = wire["execution_receipt"]
    assert isinstance(bundle_versions, dict) and isinstance(bundle_execution, dict)
    constants = build_run_provenance_identity(
        identity_generation=0,
        source_grammar_receipt=None,
        spec_binding=None,
        authority_versions={
            "stacked_authority": bundle_versions["stacked_authority"],
            "checkpoint_materializer": bundle_versions["checkpoint_materializer"],
            "runtime_authority": None,
            "execution_abi": None,
        },
        code_inventory_digest=wire["code_inventory_digest"],
        artifact_protocol_inventory=wire["artifact_protocol_inventory"],
        run_request=wire["run_request"],
        execution_receipt={
            "authority_mode": "constants",
            "pipeline": bundle_execution["pipeline"],
            "code_pin": bundle_execution["code_pin"],
        },
    )
    return constants, bundle


def _publication(
    *, operational: str, primary_status: str = "initialized"
) -> dict[str, object]:
    return {
        "operational": operational,
        "sampling": {"sample_fraction": 0.01, "sample_seed": 578},
        "clone_attachment": {"fraction": 1.0, "seed": 578},
        "stage_checkpoints": {"deepest_resumed_stage": None, "stages": {}},
        "stage_receipts": {
            "impute": {
                "primary_puf_qrf": {"resume_status": primary_status},
                "acs_qrf_transfer": {
                    "target_bank": {
                        "directions": {
                            "forward": {
                                "targets": {
                                    "0": {
                                        "load_status": "missing",
                                        "source": "rebuilt",
                                    }
                                }
                            }
                        },
                        "late_producer_groups": {},
                    }
                },
            }
        },
    }


def _coverage(
    *,
    complete: bool = True,
    node_reuse_complete: bool = True,
) -> F1CoverageEvidence:
    plan = _plan_lock()
    execution_abi = plan["execution_abi"]
    spec_binding = plan["spec_binding"]
    assert isinstance(execution_abi, dict) and isinstance(spec_binding, dict)
    selector_body = {
        "domain": "microcosm.f1-test.selector-coverage.v1",
        "schema_version": 1,
        "pipeline_id": "fixture",
        "spec_sha256": spec_binding["spec_sha256"],
        "execution_abi_sha256": execution_abi["sha256"],
        "container_member_coverage_complete": complete,
        "reason": "synthetic_comparator_fixture",
    }
    calibration_body = {
        "domain": "microcosm.f1-test.calibration-scope-coverage.v1",
        "schema_version": 1,
        "pipeline_id": "fixture",
        "spec_sha256": spec_binding["spec_sha256"],
        "execution_abi_sha256": execution_abi["sha256"],
        "calibration_scope_complete": complete,
        "reason": "synthetic_comparator_fixture",
    }
    return F1CoverageEvidence.create(
        plan_artifact_ids=("payload",),
        bound_locator_refs=("fixture:payload",),
        node_reuse_ids=("node:z", "node:a") if node_reuse_complete else (),
        node_reuse_inventory_complete=node_reuse_complete,
        selector_coverage_receipt={
            **selector_body,
            "receipt_sha256": sha256_json(selector_body),
        },
        selector_inventory_complete=complete,
        calibration_scope_receipt={
            **calibration_body,
            "receipt_sha256": sha256_json(calibration_body),
        },
        calibration_scope_complete=complete,
    )


def _production_plan_and_selector_receipt() -> tuple[
    dict[str, object], dict[str, object]
]:
    plan = copy.deepcopy(_plan_lock())
    execution_abi = plan["execution_abi"]
    assert isinstance(execution_abi, dict)
    artifacts = [
        {
            "id": "bank",
            "kind": "fixture_bank",
            "producer_ref": "node:z",
            "stage_ref": "published",
            "protocol_ref": "fixture:bank-v1",
            "locator_ref": "fixture:bank",
            "content_selector_ref": "selector:directory_tree_bytes_v1",
            "surface": "normative",
            "comparison": "raw_byte_exact",
            "required": True,
        },
        {
            "id": "entities",
            "kind": "logical_h5_content",
            "producer_ref": "node:z",
            "stage_ref": "published",
            "protocol_ref": "fixture:h5-v1",
            "locator_ref": "fixture:pool_h5",
            "content_selector_ref": ("selector:h5_all_entity_tables_and_columns_v1"),
            "surface": "normative",
            "comparison": "raw_byte_exact",
            "required": True,
        },
        {
            "id": "weights",
            "kind": "logical_h5_content",
            "producer_ref": "node:z",
            "stage_ref": "published",
            "protocol_ref": "fixture:h5-v1",
            "locator_ref": "fixture:pool_h5",
            "content_selector_ref": "selector:h5_all_weight_vectors_v1",
            "surface": "normative",
            "comparison": "raw_byte_exact",
            "required": True,
        },
    ]
    execution_abi["normative_artifact_vector"] = artifacts
    code_abi = execution_abi["code_abi"]
    assert isinstance(code_abi, dict)
    code_abi["content_selectors"] = [
        "selector:directory_tree_bytes_v1",
        "selector:h5_all_entity_tables_and_columns_v1",
        "selector:h5_all_weight_vectors_v1",
    ]
    code_body = {
        key: value for key, value in code_abi.items() if key != "implementation_sha256"
    }
    code_abi["implementation_sha256"] = sha256_json(code_body)
    execution_body = {
        key: value for key, value in execution_abi.items() if key != "sha256"
    }
    execution_abi["sha256"] = sha256_json(execution_body)

    member = {"relative_path": "targets", "kind": "directory"}
    bank = {
        "artifact_id": "bank",
        "locator_ref": "fixture:bank",
        "authority_ref": "producer_node:node:z",
        "bank_kind": "primary_qrf",
        "expected_member_count": 1,
        "expected_members_sha256": sha256_json([member]),
        "expected_members": [member],
    }
    final_h5 = {
        "artifact_ids": ["entities", "weights"],
        "locator_ref": "fixture:pool_h5",
        "selector_refs": [
            "selector:h5_all_entity_tables_and_columns_v1",
            "selector:h5_all_weight_vectors_v1",
        ],
        "status": "unsupported",
        "unsupported_reason": (
            "compiler_authority_lacks_final_h5_entity_column_weight_inventory"
        ),
    }
    spec_binding = plan["spec_binding"]
    assert isinstance(spec_binding, dict)
    contract_body = {
        "domain": "microcosm.us-pool-artifact-member-coverage.v1",
        "schema_version": 1,
        "authority_sha256": "a" * 64,
        "spec_sha256": spec_binding["spec_sha256"],
        "execution_abi_sha256": execution_abi["sha256"],
        "target_banks": [bank],
        "final_pool_h5": final_h5,
    }
    contract = {**contract_body, "sha256": sha256_json(contract_body)}
    result = {
        "artifact_id": "bank",
        "locator_ref": "fixture:bank",
        "bank_kind": "primary_qrf",
        "root_status": "directory",
        "expected_member_count": 1,
        "expected_members_sha256": sha256_json([member]),
        "observed_member_count": 1,
        "observed_members_sha256": sha256_json([member]),
        "missing_members": [],
        "extra_members": [],
        "status": "complete",
        "complete": True,
    }
    receipt_body = {
        "domain": "microcosm.us-pool-artifact-member-coverage.v1",
        "schema_version": 1,
        "contract": contract,
        "target_banks": [result],
        "bank_member_coverage_complete": True,
        "final_pool_h5": final_h5,
        "container_member_coverage_complete": False,
        "status": "unsupported",
    }
    return plan, {**receipt_body, "receipt_sha256": sha256_json(receipt_body)}


def _production_evidence_fixture(
    *,
    selector_receipt: dict[str, object] | None = None,
) -> F1ProductionEvidence:
    plan, default_selector = _production_plan_and_selector_receipt()
    selector = default_selector if selector_receipt is None else selector_receipt
    execution_abi = plan["execution_abi"]
    assert isinstance(execution_abi, dict)
    coverage = complete_coverage_evidence(
        plan,
        bound_locator_refs=("fixture:bank", "fixture:pool_h5"),
        node_reuse_ids=("node:z", "node:a"),
        node_reuse_inventory_complete=True,
        selector_inventory_complete=False,
        calibration_scope_complete=False,
        selector_coverage_receipt=selector,
        calibration_scope_receipt={
            "domain": "microcosm.us-f1-calibration-scope-coverage.v1",
            "schema_version": 1,
            "calibration_scope_complete": False,
            "reason": "normative_artifact_vector_omits_calibration_weights",
        },
    )
    constants, _bundle = _identities(plan)
    return F1ProductionEvidence.create(
        mode="constants",
        plan_lock=plan,
        artifacts={"bank": b"bank", "entities": b"entities", "weights": b"weights"},
        receipt_surfaces={
            "publication_manifest": _publication(operational="production-fixture")
        },
        run_provenance_identity=constants,
        node_reuse_keys={"node:z": "1" * 64, "node:a": "2" * 64},
        coverage=coverage,
    )


def _cold_receipt(
    mode: str,
    *,
    run_number: int,
    payload: bytes = b"same",
    coverage_complete: bool = True,
    node_reuse_complete: bool = True,
    primary_status: str = "initialized",
) -> F1ColdBuildReceipt:
    plan = _plan_lock()
    constants, bundle = _identities(plan)
    evidence = F1ProductionEvidence.create(
        mode=mode,
        plan_lock=plan,
        artifacts={"payload": payload},
        receipt_surfaces={
            "publication_manifest": _publication(
                operational=f"run-{run_number}",
                primary_status=primary_status,
            )
        },
        run_provenance_identity=constants if mode == "constants" else bundle,
        node_reuse_keys=(
            {"node:z": "1" * 64, "node:a": "2" * 64} if node_reuse_complete else {}
        ),
        coverage=_coverage(
            complete=coverage_complete,
            node_reuse_complete=node_reuse_complete,
        ),
    )
    return F1ColdBuildReceipt.create(
        request=F1RunRequest(
            sample_fraction=0.01,
            seed=578,
            clone_attachment_seed=578,
        ),
        production_evidence=evidence,
        certification_run_id=str(uuid.UUID(int=run_number)),
    )


def _four_receipts() -> tuple[F1ColdBuildReceipt, ...]:
    return (
        _cold_receipt("constants", run_number=1),
        _cold_receipt("constants", run_number=2),
        _cold_receipt("bundle", run_number=3),
        _cold_receipt("bundle", run_number=4),
    )


def _write_four(
    tmp_path: Path, receipts: tuple[F1ColdBuildReceipt, ...]
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, receipt in enumerate(receipts):
        path = tmp_path / f"receipt-{index}.json"
        atomic_write_json(path, receipt.to_wire())
        paths.append(path)
    return tuple(paths)


def _source_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--asec-raw-stage-h5",
        str(tmp_path / "asec.h5"),
        "--asec-raw-stage-h5-sha256",
        "a" * 64,
        "--acs-household-zip",
        str(tmp_path / "household.zip"),
        "--acs-household-zip-sha256",
        "b" * 64,
        "--acs-person-zip",
        str(tmp_path / "person.zip"),
        "--acs-person-zip-sha256",
        "c" * 64,
        "--acs-rent-h5",
        str(tmp_path / "rent.h5"),
        "--acs-rent-h5-sha256",
        "d" * 64,
        "--puf-h5",
        str(tmp_path / "puf.h5"),
        "--puf-h5-sha256",
        "e" * 64,
        "--puf-source-year-csv",
        str(tmp_path / "puf.csv"),
        "--puf-source-year-csv-sha256",
        "f" * 64,
    ]


def test_synthetic_four_receipt_comparator_passes() -> None:
    constants_a, constants_b, bundle_a, bundle_b = _four_receipts()
    verdict = compare_f1_cold_build_receipts(
        constants_a=constants_a,
        constants_b=constants_b,
        bundle_a=bundle_a,
        bundle_b=bundle_b,
    ).to_wire()
    assert verdict["passed"] is True
    assert verdict["vector_coverage"]["passed"] is True
    assert verdict["within_mode_determinism"]["passed"] is True
    assert verdict["cross_mode_equality"]["passed"] is True


def test_synthetic_within_mode_drift_fails() -> None:
    constants_a, _, bundle_a, bundle_b = _four_receipts()
    constants_b = _cold_receipt("constants", run_number=2, payload=b"changed")
    verdict = compare_f1_cold_build_receipts(
        constants_a=constants_a,
        constants_b=constants_b,
        bundle_a=bundle_a,
        bundle_b=bundle_b,
    ).to_wire()
    assert verdict["passed"] is False
    assert verdict["within_mode_determinism"]["constants"]["passed"] is False


def test_synthetic_cross_mode_drift_fails_with_deterministic_modes() -> None:
    constants_a, constants_b, _, _ = _four_receipts()
    bundle_a = _cold_receipt("bundle", run_number=3, payload=b"bundle")
    bundle_b = _cold_receipt("bundle", run_number=4, payload=b"bundle")
    verdict = compare_f1_cold_build_receipts(
        constants_a=constants_a,
        constants_b=constants_b,
        bundle_a=bundle_a,
        bundle_b=bundle_b,
    ).to_wire()
    assert verdict["within_mode_determinism"]["passed"] is True
    assert verdict["cross_mode_equality"]["passed"] is False
    assert verdict["passed"] is False


def test_synthetic_coverage_and_cold_audits_fail_closed() -> None:
    constants_a, constants_b, bundle_a, _ = _four_receipts()
    bundle_b = _cold_receipt(
        "bundle",
        run_number=4,
        coverage_complete=False,
        primary_status="resumed",
    )
    verdict = compare_f1_cold_build_receipts(
        constants_a=constants_a,
        constants_b=constants_b,
        bundle_a=bundle_a,
        bundle_b=bundle_b,
    ).to_wire()
    assert verdict["vector_coverage"]["passed"] is False
    assert verdict["cold_builds"]["passed"] is False
    assert verdict["passed"] is False


def test_incomplete_node_reuse_inventory_round_trips_and_comparator_fails() -> None:
    receipts = tuple(
        _cold_receipt(
            mode,
            run_number=run_number,
            node_reuse_complete=False,
        )
        for mode, run_number in (
            ("constants", 1),
            ("constants", 2),
            ("bundle", 3),
            ("bundle", 4),
        )
    )
    restored = tuple(
        F1ColdBuildReceipt.from_mapping(receipt.to_wire()) for receipt in receipts
    )
    verdict = compare_f1_cold_build_receipts(
        constants_a=restored[0],
        constants_b=restored[1],
        bundle_a=restored[2],
        bundle_b=restored[3],
    ).to_wire()

    assert verdict["vector_coverage"]["passed"] is False
    assert verdict["within_mode_determinism"]["passed"] is False
    assert verdict["cross_mode_equality"]["passed"] is False
    assert verdict["passed"] is False


def test_cold_receipt_rejects_digest_tampering() -> None:
    receipt = _cold_receipt("constants", run_number=1)
    wire = receipt.to_wire()
    wire["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seal mismatch"):
        F1ColdBuildReceipt.from_mapping(wire)


def test_cold_receipt_rederives_resume_audit_from_publication() -> None:
    receipt = _cold_receipt("constants", run_number=1)
    wire = receipt.to_wire()
    wire["resume_audit"]["primary_qrf"] = {
        "resume_status": "resumed",
        "resumed_count": 1,
    }
    wire["resume_audit"]["total_resume_count"] = 1
    wire["resume_audit"]["passed"] = False
    body = {key: value for key, value in wire.items() if key != "receipt_sha256"}
    wire["receipt_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="differs from the publication"):
        F1ColdBuildReceipt.from_mapping(wire)


def test_compare_cli_writes_typed_json_and_markdown(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_assert_current_plan", lambda _plan: None)
    monkeypatch.setattr(
        runner,
        "_assert_current_selector_coverage_contracts",
        lambda _receipts: None,
    )
    constants_a, constants_b, bundle_a, bundle_b = _write_four(
        tmp_path, _four_receipts()
    )
    output = tmp_path / "verdict"
    assert (
        runner.main(
            [
                "compare",
                "--constants-a",
                str(constants_a),
                "--constants-b",
                str(constants_b),
                "--bundle-a",
                str(bundle_a),
                "--bundle-b",
                str(bundle_b),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / CERTIFICATION_JSON_FILENAME).read_bytes())
    assert payload["passed"] is True
    assert "Overall verdict: **PASS**" in (
        output / CERTIFICATION_MARKDOWN_FILENAME
    ).read_text(encoding="utf-8")


def test_compare_cli_returns_one_for_valid_fail(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_assert_current_plan", lambda _plan: None)
    monkeypatch.setattr(
        runner,
        "_assert_current_selector_coverage_contracts",
        lambda _receipts: None,
    )
    receipts = list(_four_receipts())
    receipts[1] = _cold_receipt("constants", run_number=2, payload=b"changed")
    constants_a, constants_b, bundle_a, bundle_b = _write_four(
        tmp_path, tuple(receipts)
    )
    assert (
        runner.main(
            [
                "compare",
                "--constants-a",
                str(constants_a),
                "--constants-b",
                str(constants_b),
                "--bundle-a",
                str(bundle_a),
                "--bundle-b",
                str(bundle_b),
                "--output-root",
                str(tmp_path / "verdict"),
            ]
        )
        == 1
    )


def test_compare_cli_writes_valid_fail_for_incomplete_node_reuse(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_assert_current_plan", lambda _plan: None)
    monkeypatch.setattr(
        runner,
        "_assert_current_selector_coverage_contracts",
        lambda _receipts: None,
    )
    receipts = tuple(
        _cold_receipt(mode, run_number=number, node_reuse_complete=False)
        for mode, number in (
            ("constants", 1),
            ("constants", 2),
            ("bundle", 3),
            ("bundle", 4),
        )
    )
    constants_a, constants_b, bundle_a, bundle_b = _write_four(tmp_path, receipts)
    output = tmp_path / "incomplete-verdict"
    assert (
        runner.main(
            [
                "compare",
                "--constants-a",
                str(constants_a),
                "--constants-b",
                str(constants_b),
                "--bundle-a",
                str(bundle_a),
                "--bundle-b",
                str(bundle_b),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    payload = json.loads((output / CERTIFICATION_JSON_FILENAME).read_bytes())
    assert payload["vector_coverage"]["passed"] is False
    assert payload["within_mode_determinism"]["passed"] is False
    assert payload["cross_mode_equality"]["passed"] is False
    assert "Overall verdict: **FAIL**" in (
        output / CERTIFICATION_MARKDOWN_FILENAME
    ).read_text(encoding="utf-8")


def test_production_evidence_rejects_node_reuse_coverage_mismatch() -> None:
    receipt = _cold_receipt(
        "constants",
        run_number=1,
        node_reuse_complete=False,
    )
    evidence = receipt.production_evidence
    with pytest.raises(ValueError, match="differ from coverage ids"):
        F1ProductionEvidence.create(
            mode=evidence.mode,
            plan_lock=evidence.plan_lock,
            artifacts={row.artifact_id: row.digest for row in evidence.artifacts},
            receipt_surfaces=evidence.receipt_surfaces,
            run_provenance_identity=evidence.run_provenance_identity,
            node_reuse_keys={"node:z": "1" * 64},
            coverage=evidence.coverage,
        )


def test_production_evidence_create_and_load_require_full_plan_schema() -> None:
    receipt = _cold_receipt("constants", run_number=1)
    evidence = receipt.production_evidence
    malformed = copy.deepcopy(evidence.plan_lock)
    malformed.pop("nodes")
    with pytest.raises(ValueError, match="plan_lock is invalid"):
        F1ProductionEvidence.create(
            mode=evidence.mode,
            plan_lock=malformed,
            artifacts={row.artifact_id: row.digest for row in evidence.artifacts},
            receipt_surfaces=evidence.receipt_surfaces,
            run_provenance_identity=evidence.run_provenance_identity,
            node_reuse_keys=evidence.node_reuse_keys,
            coverage=evidence.coverage,
        )

    wire = evidence.to_wire()
    wire["plan_lock"].pop("nodes")
    with pytest.raises(ValueError, match="plan_lock is invalid"):
        F1ProductionEvidence.from_mapping(wire)


def test_production_selector_receipt_validates_seals_and_current_contract() -> None:
    evidence = _production_evidence_fixture()
    selector = evidence.coverage.selector_coverage_receipt
    assert isinstance(selector, dict)
    contract = selector["contract"]
    assert_f1_selector_coverage_contract_current(evidence.coverage, contract)

    changed = copy.deepcopy(contract)
    changed["authority_sha256"] = "f" * 64
    changed_body = {key: value for key, value in changed.items() if key != "sha256"}
    changed["sha256"] = sha256_json(changed_body)
    with pytest.raises(ValueError, match="freshly compiled authority"):
        assert_f1_selector_coverage_contract_current(evidence.coverage, changed)


def test_production_selector_receipt_rejects_resealed_false_summary() -> None:
    plan, selector = _production_plan_and_selector_receipt()
    selector["container_member_coverage_complete"] = True
    body = {key: value for key, value in selector.items() if key != "receipt_sha256"}
    selector["receipt_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="summary mismatch"):
        complete_coverage_evidence(
            plan,
            bound_locator_refs=("fixture:bank", "fixture:pool_h5"),
            node_reuse_ids=("node:z", "node:a"),
            node_reuse_inventory_complete=True,
            selector_inventory_complete=True,
            calibration_scope_complete=False,
            selector_coverage_receipt=selector,
        )


def test_production_selector_and_calibration_plan_links_fail_closed() -> None:
    plan, selector = _production_plan_and_selector_receipt()
    contract = selector["contract"]
    assert isinstance(contract, dict)
    contract["execution_abi_sha256"] = "f" * 64
    contract_body = {key: value for key, value in contract.items() if key != "sha256"}
    contract["sha256"] = sha256_json(contract_body)
    selector_body = {
        key: value for key, value in selector.items() if key != "receipt_sha256"
    }
    selector["receipt_sha256"] = sha256_json(selector_body)
    coverage = complete_coverage_evidence(
        plan,
        bound_locator_refs=("fixture:bank", "fixture:pool_h5"),
        node_reuse_ids=("node:z", "node:a"),
        node_reuse_inventory_complete=True,
        selector_inventory_complete=False,
        calibration_scope_complete=False,
        selector_coverage_receipt=selector,
    )
    constants, _bundle = _identities(plan)
    with pytest.raises(ValueError, match="execution link differs from plan"):
        F1ProductionEvidence.create(
            mode="constants",
            plan_lock=plan,
            artifacts={
                "bank": b"bank",
                "entities": b"entities",
                "weights": b"weights",
            },
            receipt_surfaces={
                "publication_manifest": _publication(operational="link-test")
            },
            run_provenance_identity=constants,
            node_reuse_keys={"node:z": "1" * 64, "node:a": "2" * 64},
            coverage=coverage,
        )

    valid = _production_evidence_fixture()
    calibration = dict(valid.coverage.calibration_scope_receipt)
    calibration["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt_sha256: mismatch"):
        F1CoverageEvidence.create(
            plan_artifact_ids=valid.coverage.plan_artifact_ids,
            bound_locator_refs=valid.coverage.bound_locator_refs,
            node_reuse_ids=valid.coverage.node_reuse_ids,
            node_reuse_inventory_complete=True,
            selector_coverage_receipt=valid.coverage.selector_coverage_receipt,
            selector_inventory_complete=False,
            calibration_scope_receipt=calibration,
            calibration_scope_complete=False,
        )


def test_run_refuses_existing_root_before_launch(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(
        runner,
        "_launch_pool_child",
        lambda *_args: pytest.fail("child must not launch"),
    )
    assert (
        runner.main(
            [
                "run",
                "--mode",
                "constants",
                "--sample-fraction",
                "0.01",
                "--seed",
                "578",
                "--output-root",
                str(output),
                *_source_cli_args(tmp_path),
            ]
        )
        == 2
    )


def test_run_launches_exactly_one_sanitized_forbid_child(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "cold"
    calls: list[tuple[list[str], dict[str, str]]] = []

    def launch(command: list[str], environment: dict[str, str]) -> int:
        calls.append((command, environment))
        evidence_path = Path(command[command.index("--f1-evidence-out") + 1])
        evidence = _cold_receipt("constants", run_number=9).production_evidence
        atomic_write_json(evidence_path, evidence.to_wire())
        return 0

    monkeypatch.setattr(runner, "_launch_pool_child", launch)
    monkeypatch.setattr(runner, "_assert_current_plan", lambda _plan: None)
    for key in runner._SANITIZED_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "must-not-leak")
    assert (
        runner.main(
            [
                "run",
                "--mode",
                "constants",
                "--sample-fraction",
                "0.01",
                "--seed",
                "578",
                "--output-root",
                str(output),
                *_source_cli_args(tmp_path),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    command, environment = calls[0]
    assert command.count("--resume-policy") == 1
    assert command[command.index("--resume-policy") + 1] == "forbid"
    assert command[command.index("--config-authority") + 1] == "constants"
    assert "--logbook-prev-row-digest" not in command
    assert all(key not in environment for key in runner._SANITIZED_ENVIRONMENT_KEYS)
    assert (output / COLD_BUILD_RECEIPT_FILENAME).is_file()


def test_failed_child_emits_no_cold_receipt(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(runner, "_launch_pool_child", lambda *_args: 7)
    assert (
        runner.main(
            [
                "run",
                "--mode",
                "bundle",
                "--sample-fraction",
                "0.01",
                "--seed",
                "578",
                "--output-root",
                str(output),
                *_source_cli_args(tmp_path),
            ]
        )
        == 7
    )
    assert not (output / COLD_BUILD_RECEIPT_FILENAME).exists()


def test_run_seals_incomplete_coverage_for_comparator_adjudication(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "incomplete"

    def launch(command: list[str], _environment: dict[str, str]) -> int:
        evidence_path = Path(command[command.index("--f1-evidence-out") + 1])
        evidence = _cold_receipt(
            "bundle",
            run_number=10,
            coverage_complete=False,
            primary_status="resumed",
        ).production_evidence
        atomic_write_json(evidence_path, evidence.to_wire())
        return 0

    monkeypatch.setattr(runner, "_launch_pool_child", launch)
    monkeypatch.setattr(runner, "_assert_current_plan", lambda _plan: None)
    assert (
        runner.main(
            [
                "run",
                "--mode",
                "bundle",
                "--sample-fraction",
                "0.01",
                "--seed",
                "578",
                "--output-root",
                str(output),
                *_source_cli_args(tmp_path),
            ]
        )
        == 0
    )
    receipt = F1ColdBuildReceipt.from_mapping(
        json.loads((output / COLD_BUILD_RECEIPT_FILENAME).read_bytes())
    )
    assert receipt.production_evidence.coverage.complete is False
    assert receipt.resume_audit.passed is False


def test_resume_gate_is_documentation_only(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_lock()
    execution_abi = plan["execution_abi"]
    assert isinstance(execution_abi, dict)
    execution_abi["resume_predicate"] = {
        "candidate_order": ["simulated", "transferred", "assembled"],
        "required_artifact_roles_by_stage": {
            "simulated": ["checkpoint:simulated:payload"],
            "transferred": ["checkpoint:transferred:payload"],
            "assembled": ["checkpoint:assembled:payload"],
        },
        "identity_fields": ["input_pins", "sampling_request"],
        "integrity_validators": ["content_digest_mismatch"],
        "last_durable_stage": "simulated",
    }
    monkeypatch.setattr(runner, "_compile_current_plan_lock", lambda: plan)
    monkeypatch.setattr(
        runner,
        "_launch_pool_child",
        lambda *_args: pytest.fail("resume-gate must not launch"),
    )
    output = tmp_path / "resume.md"
    assert (
        runner.main(
            [
                "resume-gate",
                "--mode",
                "bundle",
                "--sample-fraction",
                "0.01",
                "--seed",
                "578",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = output.read_text(encoding="utf-8")
    assert "DOCUMENTATION ONLY" in document
    assert "content_digest_mismatch" in document
    assert "executes no build" in document


def test_pool_resume_policy_refuses_preexisting_state(tmp_path: Path) -> None:
    from tools import build_us_multispine_pool as pool_tool

    outputs = pool_tool._stacked_output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=tmp_path / "checkpoints",
    )
    outputs.manifest.write_text("existing", encoding="utf-8")
    args = type("Args", (), {"resume_policy": "forbid"})()
    with pytest.raises(ValueError, match="resume-policy forbid"):
        pool_tool._refuse_preexisting_resume_state(args, outputs)


def test_pool_resume_policy_default_allow_preserves_existing_state(
    tmp_path: Path,
) -> None:
    from tools import build_us_multispine_pool as pool_tool

    outputs = pool_tool._stacked_output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=tmp_path / "checkpoints",
    )
    outputs.checkpoint_root.mkdir()
    args = type("Args", (), {})()
    pool_tool._refuse_preexisting_resume_state(args, outputs)


def test_pool_resume_policy_refuses_preexisting_f1_evidence(tmp_path: Path) -> None:
    from tools import build_us_multispine_pool as pool_tool

    outputs = pool_tool._stacked_output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=tmp_path / "checkpoints",
    )
    evidence = tmp_path / "us-f1-production-evidence.json"
    evidence.write_text("existing", encoding="utf-8")
    args = type(
        "Args",
        (),
        {"resume_policy": "forbid", "f1_evidence_out": evidence},
    )()
    with pytest.raises(ValueError, match="resume-policy forbid"):
        pool_tool._refuse_preexisting_resume_state(args, outputs)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"resume_policy": "allow"}, "resume-policy forbid"),
        ({"config_authority": "constants_adapter"}, "constants or bundle"),
        ({"legacy_two_spine": True}, "legacy-two-spine"),
    ),
)
def test_pool_f1_evidence_request_is_cold_stacked_constants_or_bundle_only(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    from tools import build_us_multispine_pool as pool_tool

    outputs = pool_tool._stacked_output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=tmp_path / "checkpoints",
    )
    values: dict[str, object] = {
        "f1_evidence_out": tmp_path / "evidence.json",
        "resume_policy": "forbid",
        "config_authority": "constants",
        "legacy_two_spine": False,
        **{
            name: tmp_path / name
            for name in (
                "asec_raw_stage_h5",
                "acs_household_zip",
                "acs_person_zip",
                "acs_rent_h5",
                "puf_h5",
                "puf_source_year_csv",
            )
        },
        **override,
    }
    args = type("Args", (), values)()
    with pytest.raises(ValueError, match=message):
        pool_tool._validate_f1_evidence_request(args, outputs)
