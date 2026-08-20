"""Closed bundle-mode runtime authority and execution-ABI gates."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from microcosm.build.spec_engine import (
    RuntimeAuthorities,
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.artifact_comparison import (
    checkpoint_receipt_surface,
    compare_artifact_sets,
)
from microcosm.build.spec_engine.canonical import normalize_and_project
from microcosm.build.spec_engine.compiler_ir import (
    CHECKPOINT_RECEIPT_OPERATIONAL_POINTER,
)
from microcosm.build.spec_engine.executor import build_run_provenance_identity
from microcosm.build.spec_engine.legacy_adapter import compile_to_legacy_payload
from microcosm.build.spec_engine.model import ResolvedSpec, ResourceKind, thaw_json
from microcosm.build.spec_engine.schemas import load_schema_registry
from microcosm.build.spec_engine.stacked_authority_semantics import (
    project_stacked_authority_receipt,
    project_stacked_checkpoint_static_components,
)


@pytest.fixture(scope="module")
def resolved_us() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def authorities(resolved_us: ResolvedSpec) -> RuntimeAuthorities:
    return compile_runtime_authorities(compile_spec(resolved_us))


def _mutate_domain(
    spec: ResolvedSpec,
    kind: ResourceKind,
    mutation,
) -> ResolvedSpec:
    resources = list(spec.resources)
    index = next(
        index
        for index, resource in enumerate(resources)
        if resource.descriptor.kind is kind
    )
    resource = resources[index]
    value = copy.deepcopy(resource.domain.to_wire())
    mutation(value)
    normalized, projections = normalize_and_project(
        value,
        schema_id=resource.descriptor.schema_id,
        registry=load_schema_registry(),
    )
    resources[index] = replace(
        resource,
        domain=replace(resource.domain, value=normalized),
        projections=projections,
    )
    return replace(spec, resources=tuple(resources))


def _checkpoint_sidecar(stage: str, *, root: str) -> dict[str, object]:
    return {
        "artifact_kind": "populace_us_stacked_pool_checkpoint_operational_receipts",
        "schema_version": 1,
        "materializer_version": 11,
        "stage": stage,
        "identity_sha256": "a" * 64,
        "checkpoint": {
            "filename": f"pool-{stage}.h5",
            "sha256": "b" * 64,
            "size_bytes": 123,
        },
        "operational_stage_receipts": {"impute": {"path": root}},
    }


def test_runtime_capability_is_immutable_and_does_not_expose_full_ir(
    authorities: RuntimeAuthorities,
) -> None:
    assert authorities.identity_generation == 1
    assert len(authorities.nodes) == 38
    assert not hasattr(authorities, "compiled")
    assert not hasattr(authorities, "normalized_resources")
    assert not hasattr(authorities, "resources_wire")
    assert authorities.declared_sources["schema_version"] == 1
    assert (
        authorities.declared_sources["sha256"]
        == (authorities.to_wire()["declared_sources_sha256"])
    )
    with pytest.raises(TypeError):
        authorities.normative["bundle"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        authorities.resource("bundle")["country"] = "mutated"  # type: ignore[index]


def test_runtime_projections_are_direct_ir_compatibility_fences(
    resolved_us: ResolvedSpec,
    authorities: RuntimeAuthorities,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy aggregate adapter must not run")

    monkeypatch.setattr(
        "microcosm.build.spec_engine.legacy_adapter.compile_to_legacy_payload",
        forbidden,
    )
    rebuilt = compile_runtime_authorities(compile_spec(resolved_us))
    assert thaw_json(rebuilt.projection("stacked_authority")) == (
        project_stacked_authority_receipt(resolved_us)
    )
    assert thaw_json(rebuilt.projection("stacked_checkpoint_static_components")) == (
        project_stacked_checkpoint_static_components(resolved_us)
    )
    # Keep the imported symbol live so this test also guards accidental import
    # replacement rather than merely a misspelled monkeypatch target.
    assert callable(compile_to_legacy_payload)
    assert rebuilt.authority_sha256 == authorities.authority_sha256


def test_bundle_run_provenance_is_generation_one_and_receipt_only(
    authorities: RuntimeAuthorities,
) -> None:
    first = authorities.run_provenance_identity(
        run_request={"sample_fraction": 0.01, "sample_seed": 578},
        execution_receipt={"workers": 1},
    )
    second = authorities.run_provenance_identity(
        run_request={"sample_fraction": 0.01, "sample_seed": 578},
        execution_receipt={"workers": 2},
    )
    first_wire = first.to_wire()
    assert first_wire["identity_generation"] == 1
    assert first_wire["source_grammar_receipt"] == (
        authorities.grammar_receipt.to_wire()
    )
    assert first_wire["spec_binding"]["attestation"] == "bundle-authoritative"
    assert first_wire != second.to_wire()
    assert authorities.authority_sha256 == authorities.to_wire()["authority_sha256"]


def test_execution_abi_is_plan_derived_concrete_and_fail_closed(
    authorities: RuntimeAuthorities,
) -> None:
    execution = thaw_json(authorities.execution_abi)
    stages = execution["logical_stages"]
    operations = [operation for stage in stages for operation in stage["operations"]]
    assert operations == execution["pipeline"]["operator_order"]
    assert [stage["id"] for stage in stages if stage["durable_checkpoint"]] == [
        row["id"] for row in execution["durable_checkpoints"]
    ]
    assert stages[-1]["durable_checkpoint"] is False
    assert [stage["operational_receipts_sidecar"] for stage in stages] == [
        "forbidden",
        "required",
        "required",
        "not_applicable",
    ]
    assert [
        checkpoint["operational_receipts_sidecar"]
        for checkpoint in execution["durable_checkpoints"]
    ] == ["forbidden", "required", "required"]

    artifacts = execution["normative_artifact_vector"]
    assert len(artifacts) == 30
    target_banks = [
        row
        for row in artifacts
        if row["kind"]
        in {
            "early_transfer_target_bank",
            "late_transfer_target_bank",
            "primary_qrf_checkpoint",
        }
    ]
    assert len(target_banks) == 22
    assert (
        sum(row["producer_ref"].startswith("producer_node:") for row in target_banks)
        == 20
    )
    assert len({row["id"] for row in artifacts}) == len(artifacts)
    assert all(row["comparison"] == "raw_byte_exact" for row in artifacts)
    assert all(row["content_selector_ref"] for row in artifacts)

    rules = execution["receipt_comparison_vector"]
    assert len(
        {(row["artifact_role"], row["json_pointer_pattern"]) for row in rules}
    ) == (len(rules))
    assert all("**" not in row["json_pointer_pattern"] for row in rules)
    checkpoint_receipt_rules = [
        row for row in rules if row["category"] == "checkpoint_operational_receipt"
    ]
    assert checkpoint_receipt_rules == [
        {
            "artifact_role": f"checkpoint:{checkpoint['id']}:receipts",
            "json_pointer_pattern": CHECKPOINT_RECEIPT_OPERATIONAL_POINTER,
            "rule": "operational_excluded",
            "category": "checkpoint_operational_receipt",
        }
        for checkpoint in execution["durable_checkpoints"]
    ]
    assert execution["code_abi"]["receipt_difference_match"] == (
        "exactly_one_sealed_rule"
    )


def test_live_execution_abi_accepts_only_the_closed_dual_mode_receipt_shape(
    authorities: RuntimeAuthorities,
) -> None:
    execution = thaw_json(authorities.execution_abi)
    artifacts = {
        row["id"]: f"fixture:{row['id']}".encode()
        for row in execution["normative_artifact_vector"]
    }
    constants_provenance = build_run_provenance_identity(
        identity_generation=0,
        source_grammar_receipt=None,
        spec_binding=None,
        authority_versions={
            "stacked_authority": "1" * 64,
            "checkpoint_materializer": "2" * 64,
            "runtime_authority": None,
            "execution_abi": None,
        },
        code_inventory_digest="3" * 64,
        artifact_protocol_inventory={"pipeline": execution["pipeline"]["id"]},
        run_request={"pipeline": execution["pipeline"]["id"], "sample_seed": 578},
        execution_receipt={
            "authority_mode": "constants",
            "pipeline": execution["pipeline"]["id"],
            "code_pin": "fixture-code-pin",
        },
    )
    binding = authorities.spec_binding.to_wire()
    binding["attestation"] = "bundle-authoritative"
    bundle_provenance = build_run_provenance_identity(
        identity_generation=1,
        source_grammar_receipt=authorities.grammar_receipt.to_wire(),
        spec_binding=binding,
        authority_versions={
            "stacked_authority": "1" * 64,
            "checkpoint_materializer": "2" * 64,
            "runtime_authority": authorities.authority_sha256,
            "execution_abi": execution["sha256"],
        },
        code_inventory_digest="3" * 64,
        artifact_protocol_inventory={"pipeline": execution["pipeline"]["id"]},
        run_request={"pipeline": execution["pipeline"]["id"], "sample_seed": 578},
        execution_receipt={
            "authority_mode": "bundle",
            "pipeline": execution["pipeline"]["id"],
            "code_pin": "fixture-code-pin",
        },
    )
    constants_run_config = {
        "config_authority": "constants",
        "spec_binding_status": "absent",
        "identity_generation": 0,
        "run_provenance_identity": constants_provenance.to_wire(),
    }
    bundle_run_config = {
        "config_authority": "bundle",
        "spec_binding_status": "resolved",
        "identity_generation": 1,
        "run_provenance_identity": bundle_provenance.to_wire(),
    }
    constants_receipts: dict[str, object] = {}
    bundle_receipts: dict[str, object] = {}
    for checkpoint in execution["durable_checkpoints"]:
        manifest_role = f"checkpoint:{checkpoint['id']}:manifest"
        receipts_role = f"checkpoint:{checkpoint['id']}:receipts"
        manifest = {"stage": checkpoint["id"], "identity_sha256": "a" * 64}
        constants_receipts[manifest_role] = {
            **manifest,
            "run_config": constants_run_config,
        }
        bundle_receipts[manifest_role] = {
            **manifest,
            "run_config": bundle_run_config,
        }
        if checkpoint["operational_receipts_sidecar"] == "forbidden":
            constants_sidecar = None
            bundle_sidecar = None
        else:
            constants_sidecar = _checkpoint_sidecar(
                checkpoint["id"],
                root=f"/constants/{checkpoint['id']}",
            )
            bundle_sidecar = _checkpoint_sidecar(
                checkpoint["id"],
                root=f"/bundle/{checkpoint['id']}",
            )
        constants_receipts[receipts_role] = checkpoint_receipt_surface(
            constants_sidecar
        )
        bundle_receipts[receipts_role] = checkpoint_receipt_surface(bundle_sidecar)
    constants_receipts["publication_manifest"] = {
        "run_config": constants_run_config,
        "source_broker_receipt": None,
        "release_id": "populace-us-2024-fixture-20260819T010203Z-a1b2c3d4",
        "publication_run_id": "a" * 32,
        "provenance_pins": [{"path": "/constants/input"}],
        "pool_h5": {
            "path": "/constants/pool.h5",
            "sha256": "4" * 64,
            "size_bytes": 100,
            "publication_run_id": "a" * 32,
        },
        "agreement_diagnostics": {
            "path": "/constants/gates.json",
            "sha256": "5" * 64,
            "size_bytes": 200,
            "publication_run_id": "a" * 32,
        },
        "primary_qrf_checkpoint_dir": "/constants/primary",
        "acs_transfer_checkpoint_dir": "/constants/transfer",
        "stage_checkpoints": {"root": "/constants/checkpoints"},
        "stage_receipts": {"root": "/constants/receipts"},
    }
    bundle_receipts["publication_manifest"] = {
        "run_config": bundle_run_config,
        "source_broker_receipt": {"validated_by": "collector"},
        "release_id": "microcosm-us-2024-fixture-20260820T111213Z-deadbeef",
        "publication_run_id": "b" * 32,
        "provenance_pins": [{"path": "/bundle/input"}],
        "pool_h5": {
            "path": "/bundle/pool.h5",
            "sha256": "6" * 64,
            "size_bytes": 300,
            "publication_run_id": "b" * 32,
        },
        "agreement_diagnostics": {
            "path": "/bundle/gates.json",
            "sha256": "7" * 64,
            "size_bytes": 400,
            "publication_run_id": "b" * 32,
        },
        "primary_qrf_checkpoint_dir": "/bundle/primary",
        "acs_transfer_checkpoint_dir": "/bundle/transfer",
        "stage_checkpoints": {"root": "/bundle/checkpoints"},
        "stage_receipts": {"root": "/bundle/receipts"},
    }
    constants_receipts["terminal_gates"] = {
        "release_id": "populace-us-2024-fixture-20260819T010203Z-a1b2c3d4",
        "publication_run_id": "a" * 32,
    }
    bundle_receipts["terminal_gates"] = {
        "release_id": "microcosm-us-2024-fixture-20260820T111213Z-deadbeef",
        "publication_run_id": "b" * 32,
    }

    node_keys = {
        node_id: f"{index + 1:064x}"
        for index, node_id in enumerate(execution["pipeline"]["producer_order"])
    }

    receipt = compare_artifact_sets(
        authorities.execution_abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts=constants_receipts,
        bundle_receipts=bundle_receipts,
        constants_run_provenance_identity=constants_provenance,
        bundle_run_provenance_identity=bundle_provenance,
        constants_node_reuse_keys=node_keys,
        bundle_node_reuse_keys=node_keys,
    )

    assert receipt.passed is True
    assert len(receipt.artifact_rows) == 30
    assert len(receipt.receipt_rows) == len(execution["receipt_comparison_vector"])


@pytest.mark.parametrize("country", ["be", "uk"])
def test_country_without_physical_pipeline_has_explicit_absent_execution_abi(
    country: str,
) -> None:
    compiled = compile_spec(load_bundle(country))
    assert compiled.execution_abi["present"] is False
    assert compiled.execution_abi["normative_artifact_vector"] == ()
    assert compiled.execution_abi["code_abi"] is None


def test_documentation_edit_cannot_rekey_or_escape_runtime_capability(
    resolved_us: ResolvedSpec,
) -> None:
    original_ir = compile_spec(resolved_us)

    def mutate(value: dict[str, object]) -> None:
        release = value["release"]
        assert isinstance(release, dict)
        line = release["line"]
        assert isinstance(line, dict)
        line["note"] = "different documentation only"

    changed_ir = compile_spec(
        _mutate_domain(resolved_us, ResourceKind.PUBLICATION, mutate)
    )
    assert changed_ir.producer_graph.schedule_sha256 == (
        original_ir.producer_graph.schedule_sha256
    )
    assert [node.node_key for node in changed_ir.nodes] == [
        node.node_key for node in original_ir.nodes
    ]
    assert compile_runtime_authorities(changed_ir).authority_sha256 == (
        compile_runtime_authorities(original_ir).authority_sha256
    )


def test_source_acquisition_is_a_narrow_operational_authority(
    resolved_us: ResolvedSpec,
) -> None:
    original_ir = compile_spec(resolved_us)

    def mutate(value: dict[str, object]) -> None:
        rows = value["sources"]
        assert isinstance(rows, list)
        row = rows[1]
        assert isinstance(row, dict)
        acquisition = row["acquisition"]
        assert isinstance(acquisition, dict)
        acquisition["verified_on"] = "2026-08-19"

    changed_ir = compile_spec(_mutate_domain(resolved_us, ResourceKind.SOURCES, mutate))
    original = compile_runtime_authorities(original_ir)
    changed = compile_runtime_authorities(changed_ir)
    assert changed.spec_binding.spec_sha256 == original.spec_binding.spec_sha256
    assert [node.node_key for node in changed.nodes] == [
        node.node_key for node in original.nodes
    ]
    assert changed.declared_sources != original.declared_sources
    assert changed.authority_sha256 != original.authority_sha256
    assert not hasattr(changed, "operational")
