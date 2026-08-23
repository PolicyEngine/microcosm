"""Closed-plan collection of physical artifacts and receipt sidecars."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.spec_engine import (
    artifact_collection as artifact_collection_module,
)
from microcosm.build.spec_engine import compile_spec, load_bundle
from microcosm.build.spec_engine.artifact_collection import (
    ArtifactCollectionError,
    ArtifactLocatorRegistry,
    capture_artifact_file_identity,
    collect_artifact_digests,
    collect_artifact_surfaces,
)
from microcosm.build.spec_engine.artifact_comparison import ArtifactDigest
from microcosm.build.spec_engine.artifact_selector_contract import (
    ARTIFACT_LOCATOR_GRAMMAR,
    ARTIFACT_SELECTOR_CONTRACT_SHA256,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    EXECUTION_ABI,
    current_compiler_ir_abi,
)
from microcosm.build.spec_engine.model import thaw_json


@pytest.fixture(scope="module")
def compiled_us():
    return compile_spec(load_bundle("us"))


@pytest.fixture(scope="module")
def execution_abi(compiled_us) -> dict[str, object]:
    return thaw_json(compiled_us.execution_abi)


@pytest.fixture(scope="module")
def seed_stream_map(compiled_us) -> dict[str, object]:
    return compiled_us.seed_stream_map.to_wire()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _write_pool_h5(
    path: Path,
    *,
    income: int = 10,
    weight: float = 1.5,
    period: int = 2024,
    artifact_kind: str = "fixture",
    publication_run_id: str = "operational",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(path, mode="w") as store:
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "income": [income, 20],
                    "household_weight": [weight, 2.5],
                }
            ),
            format="fixed",
        )
        store.put("_time_period", pd.Series([period]), format="table")
        store.put(
            "_populace_staging_metadata",
            pd.Series(
                [
                    json.dumps(
                        {
                            "artifact_kind": artifact_kind,
                            "publication_run_id": publication_run_id,
                        }
                    )
                ]
            ),
            format="table",
        )


def _generation_value(field: str, mode: str) -> object:
    if field == "config_authority":
        return mode
    if field == "spec_binding_status":
        return "absent" if mode == "constants" else "resolved"
    if field == "identity_generation":
        return 0 if mode == "constants" else 1
    return f"{mode}:{field}"


def _run_provenance_wire(
    mode: str, execution_abi: dict[str, object]
) -> dict[str, object]:
    bundle = mode == "bundle"
    return {
        "identity_generation": 1 if bundle else 0,
        "source_grammar_receipt": (
            {
                "schema_version": 3,
                "canonicalizer_version": 1,
                "migration_chain": [],
            }
            if bundle
            else None
        ),
        "spec_binding": (
            {
                "country": "fixture",
                "schema_id": "country-spec",
                "schema_version": 3,
                "canonicalizer_version": 1,
                "spec_sha256": "a" * 64,
                "attestation": "bundle-authoritative",
            }
            if bundle
            else None
        ),
        "authority_versions": {
            "stacked_authority": "1" * 64,
            "checkpoint_materializer": "2" * 64,
            "runtime_authority": "3" * 64 if bundle else None,
            "execution_abi": execution_abi["sha256"] if bundle else None,
        },
        "code_inventory_digest": "4" * 64,
        "artifact_protocol_inventory": {"fixture": "v1"},
        "run_request": {"pipeline": "fixture", "sample_seed": 578},
        "execution_receipt": {
            "authority_mode": mode,
            "pipeline": "fixture",
            "code_pin": "fixture-code-pin",
        },
    }


def _run_config(mode: str, execution_abi: dict[str, object]) -> dict[str, object]:
    return {
        "config_authority": mode,
        "spec_binding_status": "absent" if mode == "constants" else "resolved",
        "identity_generation": 0 if mode == "constants" else 1,
        "run_provenance_identity": _run_provenance_wire(mode, execution_abi),
    }


def _source_broker_receipt(
    execution_abi: dict[str, object],
    *,
    provenance: dict[str, object],
) -> dict[str, object]:
    grant = execution_abi["source_broker_grant"]
    assert isinstance(grant, dict)
    sources = grant["sources"]
    assert isinstance(sources, list)
    events = [
        {
            "sequence": index,
            "broker": "file",
            "operation": "open_snapshot",
            "resource": source["id"],
            "disposition": "allowed",
            "reason_code": "declared_source_read",
            "details": {
                "resolved_path": f"/fixture/sources/{source['id']}",
                "content_sha256": source["sha256"],
                "byte_size": source["byte_size"],
            },
        }
        for index, source in enumerate(sources)
    ]
    body = {
        "domain": "microcosm.spec-engine.broker-access-receipt.v1",
        "schema_version": 1,
        "surface": "operational",
        "owner": {"kind": "source_stage", "id": "declared_source_preflight"},
        "node_key": None,
        "attempt": 0,
        "attempt_scope": None,
        "status": "complete",
        "run_provenance_identity": provenance,
        "events": events,
    }
    return {**body, "receipt_sha256": sha256_json(body)}


def _set_pointer(document: dict[str, object], pointer: str, value: object) -> None:
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")
    ]
    current = document
    for token in tokens[:-1]:
        child = current.setdefault(token, {})
        assert isinstance(child, dict)
        current = child
    current[tokens[-1]] = value


def _has_pointer(document: dict[str, object], pointer: str) -> bool:
    current: object = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            return False
        current = current[token]
    return True


def _publication_document(
    mode: str,
    root: Path,
    *,
    execution_abi: dict[str, object],
    publication_run_id: str,
) -> dict[str, object]:
    prefix = "populace" if mode == "constants" else "microcosm"
    pool_path = root / "pool.h5"
    diagnostics_path = root / "pool.gates.json"
    run_config = _run_config(mode, execution_abi)
    return {
        "release_id": (
            f"{prefix}-us-fixture-f001-s578-"
            f"{'20260819T010203Z-a1b2c3d4' if mode == 'constants' else '20260820T111213Z-deadbeef'}"
        ),
        "publication_run_id": publication_run_id,
        "run_config": run_config,
        "source_broker_receipt": (
            None
            if mode == "constants"
            else _source_broker_receipt(
                execution_abi,
                provenance=run_config["run_provenance_identity"],  # type: ignore[arg-type]
            )
        ),
        "provenance_pins": {"source": {"path": str(root / "input")}},
        "pool_h5": {
            "path": str(pool_path.resolve()),
            "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
            "size_bytes": pool_path.stat().st_size,
            "publication_run_id": publication_run_id,
        },
        "agreement_diagnostics": {
            "path": str(diagnostics_path.resolve()),
            "sha256": hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
            "size_bytes": diagnostics_path.stat().st_size,
            "publication_run_id": publication_run_id,
        },
        "primary_qrf_checkpoint_dir": str(root / "primary"),
        "acs_transfer_checkpoint_dir": str(root / "transfer"),
        "period": 2024,
        "status": "simulation_ready",
        "stage_checkpoints": {"fixture": {"path": str(root / "checkpoint")}},
        "stage_receipts": {"fixture": {"path": str(root / "receipt")}},
    }


def _diagnostics_document(mode: str, *, publication_run_id: str) -> dict[str, object]:
    prefix = "populace" if mode == "constants" else "microcosm"
    return {
        "release_id": (
            f"{prefix}-us-fixture-f001-s578-"
            f"{'20260819T010203Z-a1b2c3d4' if mode == 'constants' else '20260820T111213Z-deadbeef'}"
        ),
        "publication_run_id": publication_run_id,
        "simulation_ready": True,
        "terminal_gates": {"gates": {"fixture": {"passed": True}}},
    }


def _live_registry(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    root: Path,
    *,
    mode: str = "constants",
    skip_locator: str | None = None,
    omit_required_sidecar: str | None = None,
    create_forbidden_sidecar: str | None = None,
    prewrapped_sidecar: str | None = None,
    mismatched_sidecar: str | None = None,
) -> ArtifactLocatorRegistry:
    root.mkdir(parents=True, exist_ok=True)
    publication_run_id = "a" * 32 if mode == "constants" else "b" * 32
    registry = ArtifactLocatorRegistry(allowed_roots=[root])
    artifacts = execution_abi["normative_artifact_vector"]
    assert isinstance(artifacts, list)
    selectors_by_locator: dict[str, set[str]] = {}
    for row in artifacts:
        assert isinstance(row, dict)
        selectors_by_locator.setdefault(row["locator_ref"], set()).add(
            row["content_selector_ref"]
        )

    rules = execution_abi["receipt_comparison_vector"]
    assert isinstance(rules, list)
    rules_by_role: dict[str, list[dict[str, object]]] = {}
    for raw_rule in rules:
        assert isinstance(raw_rule, dict)
        rules_by_role.setdefault(str(raw_rule["artifact_role"]), []).append(raw_rule)

    def complete_document(role: str, document: dict[str, object]) -> dict[str, object]:
        for rule in rules_by_role.get(role, []):
            pointer = str(rule["json_pointer_pattern"])
            if "*" in pointer or _has_pointer(document, pointer):
                continue
            field = pointer.rsplit("/", 1)[-1]
            _set_pointer(document, pointer, _generation_value(field, mode))
        return document

    paths: dict[str, Path] = {}
    for index, (locator, selectors) in enumerate(selectors_by_locator.items()):
        assert isinstance(locator, str)
        bind_locator = locator != skip_locator
        if selectors == {"selector:canonical_json_bytes_v1"}:
            if bind_locator:
                registry.bind_json(locator, seed_stream_map)
        elif "selector:h5_all_entity_tables_and_columns_v1" in selectors:
            path = root / "pool.h5"
            _write_pool_h5(path, publication_run_id=publication_run_id)
            if bind_locator:
                registry.bind_file(locator, path)
            paths[locator] = path
        elif selectors == {"selector:publication_normative_vector_v1"}:
            path = root / "pool.manifest.json"
            _write_json(
                path,
                complete_document(
                    "publication_manifest",
                    _publication_document(
                        mode,
                        root,
                        execution_abi=execution_abi,
                        publication_run_id=publication_run_id,
                    ),
                ),
            )
            if bind_locator:
                registry.bind_file(locator, path)
            paths[locator] = path
        elif selectors == {"selector:terminal_gate_normative_rows_v1"}:
            path = root / "pool.gates.json"
            _write_json(
                path,
                complete_document(
                    "terminal_gates",
                    _diagnostics_document(
                        mode,
                        publication_run_id=publication_run_id,
                    ),
                ),
            )
            if bind_locator:
                registry.bind_file(locator, path)
            paths[locator] = path
        elif selectors == {"selector:directory_tree_bytes_v1"}:
            path = root / "banks" / f"bank-{index:03d}"
            (path / "targets").mkdir(parents=True)
            (path / "targets" / "000__fixture.h5").write_bytes(b"bank-target")
            if bind_locator:
                registry.bind_directory(locator, path)
            paths[locator] = path
        else:
            assert selectors == {"selector:file_bytes_v1"}
            path = root / "checkpoints" / f"checkpoint-{index:03d}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"checkpoint:{locator}".encode())
            if bind_locator:
                registry.bind_file(locator, path)
            paths[locator] = path

    checkpoints = execution_abi["durable_checkpoints"]
    assert isinstance(checkpoints, list)
    for checkpoint in checkpoints:
        assert isinstance(checkpoint, dict)
        checkpoint_id = str(checkpoint["id"])
        payload_role = f"checkpoint:{checkpoint_id}:payload"
        manifest_role = f"checkpoint:{checkpoint_id}:manifest"
        receipts_role = f"checkpoint:{checkpoint_id}:receipts"
        payload_path = paths.get(payload_role)
        if payload_path is None:
            continue
        payload = payload_path.read_bytes()
        payload_binding = {
            "filename": payload_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        manifest: dict[str, object] = {
            "artifact_kind": "fixture_checkpoint_manifest",
            "schema_version": 1,
            "materializer_version": 1,
            "stage": checkpoint_id,
            "identity_sha256": "c" * 64,
            "checkpoint": payload_binding,
            "run_config": _run_config(mode, execution_abi),
        }
        for rule in rules_by_role.get(manifest_role, []):
            pointer = str(rule["json_pointer_pattern"])
            assert "*" not in pointer
            _set_pointer(
                manifest,
                pointer,
                _generation_value(pointer.rsplit("/", 1)[-1], mode),
            )
        manifest_path = root / "checkpoints" / f"{checkpoint_id}.manifest.json"
        _write_json(manifest_path, manifest)
        registry.bind_file(manifest_role, manifest_path)

        sidecar_path = root / "checkpoints" / f"{checkpoint_id}.receipts.json"
        policy = checkpoint["operational_receipts_sidecar"]
        should_write = policy == "required" and checkpoint_id != omit_required_sidecar
        should_write = should_write or checkpoint_id == create_forbidden_sidecar
        if should_write:
            sidecar: object = {
                "artifact_kind": "fixture_checkpoint_operational_receipts",
                "schema_version": 1,
                "materializer_version": 1,
                "stage": checkpoint_id,
                "identity_sha256": "c" * 64,
                "checkpoint": {
                    **payload_binding,
                    **(
                        {"sha256": "d" * 64}
                        if checkpoint_id == mismatched_sidecar
                        else {}
                    ),
                },
                "operational_stage_receipts": {"path": str(root)},
            }
            if checkpoint_id == prewrapped_sidecar:
                sidecar = {
                    "present": True,
                    "canonical": sidecar,
                    "operational": {"hidden": True},
                }
            _write_json(sidecar_path, sidecar)
        registry.bind_optional_file(receipts_role, sidecar_path)
    return registry


def _sealed_abi(
    artifacts: list[dict[str, object]],
    *,
    rules: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    selectors = sorted({str(row["content_selector_ref"]) for row in artifacts})
    code_unsigned = {
        "domain": EXECUTION_ABI,
        "content_selectors": selectors,
        "locator_grammar": ARTIFACT_LOCATOR_GRAMMAR,
        "artifact_selector_contract_sha256": ARTIFACT_SELECTOR_CONTRACT_SHA256,
        "compiler_ir_abi_sha256": current_compiler_ir_abi().sha256,
        "receipt_difference_match": "exactly_one_sealed_rule",
    }
    code_abi = {
        **code_unsigned,
        "implementation_sha256": sha256_json(code_unsigned),
    }
    source_rows: list[dict[str, object]] = []
    source_set_sha256 = sha256_json(
        {
            "domain": "microcosm.spec-engine.source-broker-grant.v1",
            "sources": source_rows,
        }
    )
    grant_unsigned = {
        "domain": "microcosm.spec-engine.source-broker-grant.v1",
        "owner": {"kind": "source_stage", "id": "declared_source_preflight"},
        "effects": ["declared_source_read"],
        "sources": source_rows,
        "source_set_sha256": source_set_sha256,
    }
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "present": True,
        "pipeline": {},
        "operations": [],
        "logical_stages": [],
        "durable_checkpoints": [],
        "code_abi": code_abi,
        "normative_artifact_vector": artifacts,
        "artifact_bindings": [],
        "source_broker_grant": {
            **grant_unsigned,
            "sha256": sha256_json(grant_unsigned),
        },
        "receipt_comparison_vector": [] if rules is None else rules,
        "resume_predicate": None,
    }
    return {**unsigned, "sha256": sha256_json(unsigned)}


def _artifact(
    artifact_id: str,
    locator: str,
    selector: str,
) -> dict[str, object]:
    return {
        "id": artifact_id,
        "kind": "fixture",
        "producer_ref": "fixture",
        "stage_ref": "fixture",
        "protocol_ref": "fixture",
        "locator_ref": locator,
        "content_selector_ref": selector,
        "surface": "normative",
        "comparison": "raw_byte_exact",
        "required": True,
    }


def test_live_thirty_artifact_vector_collects_all_selectors_and_raw_sidecars(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
) -> None:
    constants_registry = _live_registry(
        execution_abi,
        seed_stream_map,
        tmp_path / "constants",
    )
    constants = collect_artifact_surfaces(
        execution_abi,
        registry=constants_registry,
        authority_mode="constants",
    )
    constant_digests = collect_artifact_digests(
        execution_abi,
        registry=constants_registry,
        authority_mode="constants",
    )
    bundle = collect_artifact_surfaces(
        execution_abi,
        registry=_live_registry(
            execution_abi,
            seed_stream_map,
            tmp_path / "bundle",
            mode="bundle",
        ),
        authority_mode="bundle",
    )

    expected_artifacts = execution_abi["normative_artifact_vector"]
    assert isinstance(expected_artifacts, list)
    assert len(expected_artifacts) == 30
    assert set(constants.artifacts) == {str(row["id"]) for row in expected_artifacts}
    assert constant_digests.artifacts == {
        artifact_id: ArtifactDigest.from_bytes(surface)
        for artifact_id, surface in constants.artifacts.items()
    }
    assert constant_digests.receipts == constants.receipts
    assert constants.artifacts == bundle.artifacts
    assert set(constants.receipts) == {
        "publication_manifest",
        "terminal_gates",
        "checkpoint:assembled:manifest",
        "checkpoint:assembled:receipts",
        "checkpoint:transferred:manifest",
        "checkpoint:transferred:receipts",
        "checkpoint:simulated:manifest",
        "checkpoint:simulated:receipts",
    }
    assert constants.receipts["checkpoint:assembled:receipts"] == {
        "present": False,
        "canonical": {},
        "operational": {},
    }
    assert constants.receipts["checkpoint:transferred:receipts"]["present"] is True
    assert (
        constants.receipts["publication_manifest"]
        != bundle.receipts["publication_manifest"]
    )


@pytest.mark.parametrize(
    ("option", "message"),
    (
        ({"omit_required_sidecar": "transferred"}, "must be present"),
        ({"create_forbidden_sidecar": "assembled"}, "must be absent"),
        ({"prewrapped_sidecar": "transferred"}, "raw checkpoint"),
        ({"mismatched_sidecar": "transferred"}, "payload binding differs"),
    ),
)
def test_checkpoint_sidecars_are_read_raw_and_enforce_sealed_presence_and_binding(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
    option: dict[str, str],
    message: str,
) -> None:
    registry = _live_registry(execution_abi, seed_stream_map, tmp_path, **option)
    with pytest.raises((ArtifactCollectionError, ValueError), match=message):
        collect_artifact_surfaces(
            execution_abi,
            registry=registry,
            authority_mode="constants",
        )


def _bound_path(registry: ArtifactLocatorRegistry, locator_ref: str) -> Path:
    path = registry.snapshot()[locator_ref].path
    assert path is not None
    return path


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "f" * 64, "published digest differs"),
        ("size_bytes", 1, "published size differs"),
        ("publication_run_id", "c" * 32, "envelope publication_run_id differs"),
    ],
)
def test_publication_envelope_is_authenticated_before_exclusion(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    registry = _live_registry(
        execution_abi,
        seed_stream_map,
        tmp_path,
        mode="bundle",
    )
    manifest_path = _bound_path(registry, "runtime_output:manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pool_h5"][field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactCollectionError, match=message):
        collect_artifact_surfaces(
            execution_abi,
            registry=registry,
            authority_mode="bundle",
        )


def test_h5_and_diagnostics_embedded_identities_cross_authenticate_manifest(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
) -> None:
    h5_registry = _live_registry(
        execution_abi,
        seed_stream_map,
        tmp_path / "h5",
        mode="bundle",
    )
    h5_path = _bound_path(h5_registry, "runtime_output:pool_h5")
    _write_pool_h5(h5_path, publication_run_id="c" * 32)
    manifest_path = _bound_path(h5_registry, "runtime_output:manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pool_h5"]["sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    manifest["pool_h5"]["size_bytes"] = h5_path.stat().st_size
    _write_json(manifest_path, manifest)
    with pytest.raises(ArtifactCollectionError, match="embedded publication_run_id"):
        collect_artifact_surfaces(
            execution_abi,
            registry=h5_registry,
            authority_mode="bundle",
        )

    diagnostics_registry = _live_registry(
        execution_abi,
        seed_stream_map,
        tmp_path / "diagnostics",
        mode="bundle",
    )
    diagnostics_path = _bound_path(
        diagnostics_registry,
        "runtime_output:agreement_diagnostics",
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["release_id"] = "microcosm-us-other-f001-s578-20260820T111213Z-deadbeef"
    _write_json(diagnostics_path, diagnostics)
    manifest_path = _bound_path(diagnostics_registry, "runtime_output:manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agreement_diagnostics"]["sha256"] = hashlib.sha256(
        diagnostics_path.read_bytes()
    ).hexdigest()
    manifest["agreement_diagnostics"]["size_bytes"] = diagnostics_path.stat().st_size
    _write_json(manifest_path, manifest)
    with pytest.raises(ArtifactCollectionError, match="diagnostics release_id"):
        collect_artifact_surfaces(
            execution_abi,
            registry=diagnostics_registry,
            authority_mode="bundle",
        )


def test_bundle_source_broker_receipt_must_match_compiler_grant_exactly(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
) -> None:
    registry = _live_registry(
        execution_abi,
        seed_stream_map,
        tmp_path,
        mode="bundle",
    )
    manifest_path = _bound_path(registry, "runtime_output:manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = manifest["source_broker_receipt"]
    receipt["events"][0]["details"]["content_sha256"] = "f" * 64
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = sha256_json(body)
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactCollectionError, match="source identity differs"):
        collect_artifact_surfaces(
            execution_abi,
            registry=registry,
            authority_mode="bundle",
        )


def test_registry_refuses_duplicate_missing_and_extra_locators(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
) -> None:
    registry = ArtifactLocatorRegistry(allowed_roots=[tmp_path])
    registry.bind_json("fixture", {})
    with pytest.raises(ArtifactCollectionError, match="duplicate locator"):
        registry.bind_json("fixture", {})

    artifacts = execution_abi["normative_artifact_vector"]
    assert isinstance(artifacts, list)
    skipped = str(artifacts[0]["locator_ref"])
    missing = _live_registry(
        execution_abi,
        seed_stream_map,
        tmp_path / "missing",
        skip_locator=skipped,
    )
    with pytest.raises(ArtifactCollectionError, match="locator inventory mismatch"):
        collect_artifact_surfaces(
            execution_abi,
            registry=missing,
            authority_mode="constants",
        )

    extra = _live_registry(execution_abi, seed_stream_map, tmp_path / "extra")
    extra.bind_json("unexpected:locator", {})
    with pytest.raises(ArtifactCollectionError, match="locator inventory mismatch"):
        collect_artifact_surfaces(
            execution_abi,
            registry=extra,
            authority_mode="constants",
        )

    wrong_plan_value = _live_registry(
        execution_abi,
        {"wrong": "seed-stream-map"},
        tmp_path / "wrong-plan-value",
    )
    with pytest.raises(ArtifactCollectionError, match="plan component digest"):
        collect_artifact_surfaces(
            execution_abi,
            registry=wrong_plan_value,
            authority_mode="constants",
        )


def test_collector_accepts_arbitrary_additional_sealed_checkpoint_receipt_rows(
    execution_abi: dict[str, object],
    seed_stream_map: dict[str, object],
    tmp_path: Path,
) -> None:
    extended = copy.deepcopy(execution_abi)
    rules = extended["receipt_comparison_vector"]
    assert isinstance(rules, list)
    rules.append(
        {
            "artifact_role": "checkpoint:assembled:manifest",
            "json_pointer_pattern": "/collector_fixture_operational",
            "rule": "operational_excluded",
            "category": "fixture",
        }
    )
    unsigned = {key: value for key, value in extended.items() if key != "sha256"}
    extended["sha256"] = sha256_json(unsigned)

    collected = collect_artifact_surfaces(
        extended,
        registry=_live_registry(extended, seed_stream_map, tmp_path),
        authority_mode="constants",
    )

    assembled = collected.receipts["checkpoint:assembled:manifest"]
    assert (
        assembled["collector_fixture_operational"]
        == "constants:collector_fixture_operational"
    )


def test_directory_selector_refuses_escape_symlink_and_special_file(
    tmp_path: Path,
) -> None:
    abi = _sealed_abi(
        [
            _artifact(
                "tree",
                "runtime_output:tree",
                "selector:directory_tree_bytes_v1",
            )
        ]
    )
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    registry = ArtifactLocatorRegistry(allowed_roots=[allowed])
    registry.bind_directory("runtime_output:tree", outside)
    with pytest.raises(ArtifactCollectionError, match="escapes allowed roots"):
        collect_artifact_surfaces(abi, registry=registry, authority_mode="constants")

    target = allowed / "tree"
    target.mkdir()
    (target / "real").write_bytes(b"value")
    (target / "link").symlink_to(target / "real")
    registry = ArtifactLocatorRegistry(allowed_roots=[allowed])
    registry.bind_directory("runtime_output:tree", target)
    with pytest.raises(ArtifactCollectionError, match="symlink"):
        collect_artifact_surfaces(abi, registry=registry, authority_mode="constants")

    (target / "link").unlink()
    fifo = target / "fifo"
    os.mkfifo(fifo)
    registry = ArtifactLocatorRegistry(allowed_roots=[allowed])
    registry.bind_directory("runtime_output:tree", target)
    with pytest.raises(ArtifactCollectionError, match="special file"):
        collect_artifact_surfaces(abi, registry=registry, authority_mode="constants")


def test_h5_entity_and_weight_selectors_are_disjoint_logical_surfaces(
    tmp_path: Path,
) -> None:
    locator = "runtime_output:pool_h5"
    abi = _sealed_abi(
        [
            _artifact(
                "entities",
                locator,
                "selector:h5_all_entity_tables_and_columns_v1",
            ),
            _artifact(
                "weights",
                locator,
                "selector:h5_all_weight_vectors_v1",
            ),
        ]
    )

    def collect(
        path: Path,
        *,
        income: int,
        weight: float,
        period: int = 2024,
        artifact_kind: str = "fixture",
        publication_run_id: str = "operational",
    ):
        _write_pool_h5(
            path,
            income=income,
            weight=weight,
            period=period,
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
        )
        registry = ArtifactLocatorRegistry(allowed_roots=[tmp_path])
        registry.bind_file(locator, path)
        return collect_artifact_surfaces(
            abi,
            registry=registry,
            authority_mode="constants",
        ).artifacts

    baseline = collect(tmp_path / "baseline.h5", income=10, weight=1.5)
    weight_changed = collect(tmp_path / "weight.h5", income=10, weight=9.5)
    entity_changed = collect(tmp_path / "entity.h5", income=99, weight=1.5)
    operational_changed = collect(
        tmp_path / "operational.h5",
        income=10,
        weight=1.5,
        publication_run_id="another-nonce",
    )
    period_changed = collect(
        tmp_path / "period.h5",
        income=10,
        weight=1.5,
        period=2025,
    )
    metadata_changed = collect(
        tmp_path / "metadata.h5",
        income=10,
        weight=1.5,
        artifact_kind="different",
    )

    assert baseline["entities"] == weight_changed["entities"]
    assert baseline["weights"] != weight_changed["weights"]
    assert baseline["entities"] != entity_changed["entities"]
    assert baseline["weights"] == entity_changed["weights"]
    assert baseline == operational_changed
    assert baseline["entities"] != period_changed["entities"]
    assert baseline["weights"] != period_changed["weights"]
    assert baseline["entities"] != metadata_changed["entities"]
    assert baseline["weights"] != metadata_changed["weights"]


def test_h5_selectors_refuse_a_different_open_file_than_the_fenced_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = "runtime_output:pool_h5"
    abi = _sealed_abi(
        [
            _artifact(
                "entities",
                locator,
                "selector:h5_all_entity_tables_and_columns_v1",
            ),
            _artifact(
                "weights",
                locator,
                "selector:h5_all_weight_vectors_v1",
            ),
        ]
    )
    original = tmp_path / "original.h5"
    replacement = tmp_path / "replacement.h5"
    _write_pool_h5(original, income=10)
    _write_pool_h5(replacement, income=99)
    registry = ArtifactLocatorRegistry(allowed_roots=[tmp_path])
    registry.bind_file(locator, original)
    registry.fence_file(locator, capture_artifact_file_identity(original))

    real_hdf_store = pd.HDFStore

    def open_replacement(
        _path: object,
        *args: object,
        **kwargs: object,
    ) -> pd.HDFStore:
        return real_hdf_store(replacement, *args, **kwargs)

    monkeypatch.setattr(
        artifact_collection_module.pd,
        "HDFStore",
        open_replacement,
    )
    with pytest.raises(ArtifactCollectionError, match="opened a different H5"):
        collect_artifact_surfaces(
            abi,
            registry=registry,
            authority_mode="constants",
        )


def test_file_identity_fence_refuses_replacement_before_collection(
    tmp_path: Path,
) -> None:
    locator = "runtime_output:pool_h5"
    abi = _sealed_abi(
        [
            _artifact(
                "entities",
                locator,
                "selector:h5_all_entity_tables_and_columns_v1",
            )
        ]
    )
    pool_h5 = tmp_path / "pool.h5"
    replacement = tmp_path / "replacement.h5"
    _write_pool_h5(pool_h5)
    _write_pool_h5(replacement, income=99)
    registry = ArtifactLocatorRegistry(allowed_roots=[tmp_path])
    registry.bind_file(locator, pool_h5)
    registry.fence_file(locator, capture_artifact_file_identity(pool_h5))
    replacement.replace(pool_h5)

    with pytest.raises(ArtifactCollectionError, match="captured artifact identity"):
        collect_artifact_surfaces(
            abi,
            registry=registry,
            authority_mode="constants",
        )


def test_execution_abi_and_code_abi_seals_are_verified_before_io(
    tmp_path: Path,
) -> None:
    abi = _sealed_abi(
        [_artifact("value", "plan_lock:/value", "selector:canonical_json_bytes_v1")]
    )
    registry = ArtifactLocatorRegistry(allowed_roots=[tmp_path])
    registry.bind_json("plan_lock:/value", {"value": 1})

    tampered = copy.deepcopy(abi)
    tampered["normative_artifact_vector"][0]["id"] = "changed"
    with pytest.raises(ArtifactCollectionError, match="execution_abi seal mismatch"):
        collect_artifact_surfaces(
            tampered,
            registry=registry,
            authority_mode="constants",
        )

    tampered = copy.deepcopy(abi)
    tampered["code_abi"]["domain"] = "stacked-artifact-comparison-vector-v2"
    code_unsigned = {
        key: value
        for key, value in tampered["code_abi"].items()
        if key != "implementation_sha256"
    }
    tampered["code_abi"]["implementation_sha256"] = sha256_json(code_unsigned)
    unsigned = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = sha256_json(unsigned)
    with pytest.raises(ArtifactCollectionError, match="execution ABI domain"):
        collect_artifact_surfaces(
            tampered,
            registry=registry,
            authority_mode="constants",
        )

    tampered = copy.deepcopy(abi)
    tampered["code_abi"]["locator_grammar"] = "unsealed"
    unsigned = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = sha256_json(unsigned)
    with pytest.raises(ArtifactCollectionError, match="unsupported.*locator grammar"):
        collect_artifact_surfaces(
            tampered,
            registry=registry,
            authority_mode="constants",
        )

    tampered = copy.deepcopy(abi)
    tampered["code_abi"]["artifact_selector_contract_sha256"] = "f" * 64
    code_unsigned = {
        key: value
        for key, value in tampered["code_abi"].items()
        if key != "implementation_sha256"
    }
    tampered["code_abi"]["implementation_sha256"] = sha256_json(code_unsigned)
    unsigned = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = sha256_json(unsigned)
    with pytest.raises(ArtifactCollectionError, match="selector contract"):
        collect_artifact_surfaces(
            tampered,
            registry=registry,
            authority_mode="constants",
        )

    tampered = copy.deepcopy(abi)
    tampered["code_abi"]["compiler_ir_abi_sha256"] = "f" * 64
    code_unsigned = {
        key: value
        for key, value in tampered["code_abi"].items()
        if key != "implementation_sha256"
    }
    tampered["code_abi"]["implementation_sha256"] = sha256_json(code_unsigned)
    unsigned = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = sha256_json(unsigned)
    with pytest.raises(ArtifactCollectionError, match="implementation attestation"):
        collect_artifact_surfaces(
            tampered,
            registry=registry,
            authority_mode="constants",
        )
