from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import pytest

from microcosm.build.spec_engine.artifact_comparison import (
    ArtifactComparisonError,
    checkpoint_receipt_surface,
)
from microcosm.build.spec_engine.artifact_comparison import (
    compare_artifact_sets as _compare_artifact_sets,
)
from microcosm.build.spec_engine.artifact_selector_contract import (
    ARTIFACT_LOCATOR_GRAMMAR,
    ARTIFACT_SELECTOR_CONTRACT_SHA256,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    EXECUTION_ABI,
    current_compiler_ir_abi,
)
from microcosm.build.spec_engine.executor import (
    RunProvenanceIdentity,
    build_run_provenance_identity,
)
from microcosm.build.spec_engine.final_h5_inventory import (
    build_final_h5_member_inventory,
)
from microcosm.build.spec_engine.model import freeze_json


def _artifact(
    artifact_id: str,
    stage_ref: str,
    *,
    selector: str = "selector:file_bytes_v1",
) -> dict[str, object]:
    return {
        "id": artifact_id,
        "kind": "fixture_payload",
        "producer_ref": f"stage:{stage_ref}",
        "stage_ref": stage_ref,
        "protocol_ref": "fixture:bytes-v1",
        "locator_ref": f"fixture:{artifact_id}",
        "content_selector_ref": selector,
        "surface": "normative",
        "comparison": "raw_byte_exact",
        "required": True,
    }


def _rule(
    pointer: str,
    rule: str,
    *,
    role: str = "manifest",
    category: str = "fixture",
) -> dict[str, str]:
    return {
        "artifact_role": role,
        "json_pointer_pattern": pointer,
        "rule": rule,
        "category": category,
    }


def _execution_abi(
    *,
    artifacts: list[dict[str, object]] | None = None,
    rules: list[dict[str, str]] | None = None,
    receipt_roles: list[str] | None = None,
    receipt_policy: str = "required",
) -> dict[str, object]:
    selected_artifacts = artifacts or [
        _artifact("frame", "prepared"),
        _artifact("bank", "modeled"),
    ]
    selectors = sorted({str(row["content_selector_ref"]) for row in selected_artifacts})
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
    unsigned = {
        "schema_version": 1,
        "present": True,
        "pipeline": {
            "id": "fixture",
            "artifact_protocol": {"fixture": "bytes-v1"},
            "operator_order": ["fixture-operation"],
            "producer_order": ["node:first", "node:second"],
            "seed_stream_map_sha256": "5" * 64,
            "final_h5_member_inventory": build_final_h5_member_inventory(
                authority={"fixture_source_sha256": "4" * 64},
                tables=["person"],
                columns={"person": ["person_id"]},
                weights=[],
            ),
        },
        "operations": [],
        "logical_stages": [
            {
                "id": "prepared",
                "ordinal": 0,
                "operations": [],
                "producer_graph_operation": None,
                "producer_nodes": [],
                "durable_checkpoint": True,
                "operational_receipts_sidecar": "required",
            },
            {
                "id": "modeled",
                "ordinal": 1,
                "operations": [],
                "producer_graph_operation": None,
                "producer_nodes": [],
                "durable_checkpoint": True,
                "operational_receipts_sidecar": "required",
            },
        ],
        "durable_checkpoints": (
            []
            if receipt_roles is None
            else [
                {
                    "id": "fixture",
                    "ordinal": 0,
                    "after_operation": "fixture-operation",
                    "covers_operations": ["fixture-operation"],
                    "artifact_roles": receipt_roles,
                    "operational_receipts_sidecar": receipt_policy,
                }
            ]
        ),
        "code_abi": code_abi,
        "normative_artifact_vector": selected_artifacts,
        "artifact_bindings": [],
        "source_broker_grant": _source_broker_grant(),
        "receipt_comparison_vector": rules or [],
        "resume_predicate": None,
    }
    return {**unsigned, "sha256": sha256_json(unsigned)}


def _source_broker_grant() -> dict[str, object]:
    sources: list[dict[str, object]] = []
    source_set_sha256 = sha256_json(
        {
            "domain": "microcosm.spec-engine.source-broker-grant.v1",
            "sources": sources,
        }
    )
    unsigned = {
        "domain": "microcosm.spec-engine.source-broker-grant.v1",
        "owner": {"kind": "source_stage", "id": "declared_source_preflight"},
        "effects": ["declared_source_read"],
        "sources": sources,
        "source_set_sha256": source_set_sha256,
    }
    return {**unsigned, "sha256": sha256_json(unsigned)}


def _checkpoint_sidecar(
    *,
    stage: str = "fixture",
    operational: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "artifact_kind": "fixture_checkpoint_operational_receipts",
        "schema_version": 1,
        "materializer_version": 2,
        "stage": stage,
        "identity_sha256": "a" * 64,
        "checkpoint": {
            "filename": f"pool-{stage}.h5",
            "sha256": "b" * 64,
            "size_bytes": 123,
        },
        "operational_stage_receipts": (
            {"impute": {"status": "written"}} if operational is None else operational
        ),
    }


def _reseal(abi: dict[str, object]) -> None:
    unsigned = {key: value for key, value in abi.items() if key != "sha256"}
    abi["sha256"] = sha256_json(unsigned)


def _bundle_run_provenance(
    *,
    execution_abi_sha256: str = "e" * 64,
    spec_sha256: str = "b" * 64,
    attestation: str = "bundle-authoritative",
) -> RunProvenanceIdentity:
    return build_run_provenance_identity(
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
            "spec_sha256": spec_sha256,
            "attestation": attestation,
        },
        authority_versions={
            "stacked_authority": "1" * 64,
            "checkpoint_materializer": "2" * 64,
            "runtime_authority": "c" * 64,
            "execution_abi": execution_abi_sha256,
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


def _constants_run_provenance(
    bundle: RunProvenanceIdentity,
) -> RunProvenanceIdentity:
    wire = bundle.to_wire()
    versions = wire["authority_versions"]
    receipt = wire["execution_receipt"]
    assert isinstance(versions, dict) and isinstance(receipt, dict)
    return build_run_provenance_identity(
        identity_generation=0,
        source_grammar_receipt=None,
        spec_binding=None,
        authority_versions={
            "stacked_authority": versions["stacked_authority"],
            "checkpoint_materializer": versions["checkpoint_materializer"],
            "runtime_authority": None,
            "execution_abi": None,
        },
        code_inventory_digest=wire["code_inventory_digest"],
        artifact_protocol_inventory=wire["artifact_protocol_inventory"],
        run_request=wire["run_request"],
        execution_receipt={
            "authority_mode": "constants",
            "pipeline": receipt["pipeline"],
            "code_pin": receipt["code_pin"],
        },
    )


def _identity_pair(
    abi: Mapping[str, object],
) -> tuple[RunProvenanceIdentity, RunProvenanceIdentity]:
    bundle = _bundle_run_provenance(execution_abi_sha256=str(abi["sha256"]))
    return _constants_run_provenance(bundle), bundle


def compare_artifact_sets(
    execution_abi: Mapping[str, object],
    **kwargs: object,
):
    bundle = kwargs.pop("bundle_run_provenance_identity", None)
    constants = kwargs.pop("constants_run_provenance_identity", None)
    if bundle is None:
        constants_default, bundle = _identity_pair(execution_abi)
        if constants is None:
            constants = constants_default
    elif constants is None:
        assert isinstance(bundle, RunProvenanceIdentity)
        constants = _constants_run_provenance(bundle)
    # Historic caller-authored expectations have no production API surface.
    kwargs.pop("generation_expectations", None)
    pipeline = execution_abi["pipeline"]
    assert isinstance(pipeline, Mapping)
    node_ids = pipeline["producer_order"]
    assert isinstance(node_ids, Sequence)
    default_node_keys = {
        str(node_id): f"{index + 1:x}" * 64
        for index, node_id in enumerate(node_ids)
    }
    return _compare_artifact_sets(
        execution_abi,
        constants_run_provenance_identity=constants,  # type: ignore[arg-type]
        bundle_run_provenance_identity=bundle,  # type: ignore[arg-type]
        constants_node_reuse_keys=kwargs.pop(
            "constants_node_reuse_keys", default_node_keys
        ),  # type: ignore[arg-type]
        bundle_node_reuse_keys=kwargs.pop(
            "bundle_node_reuse_keys", default_node_keys
        ),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _receipts(
    bundle_provenance: RunProvenanceIdentity,
) -> tuple[dict[str, object], dict[str, object]]:
    constants_provenance = _constants_run_provenance(bundle_provenance)
    stable = {
        "node_reuse_key": "a" * 64,
        "rows": [{"id": "one", "verdict": "pass"}],
    }
    return (
        {
            "manifest": {
                "release_id": "populace-zone-2024-fixture-20260819T010203Z-a1b2c3d4",
                "run_config": {
                    "config_authority": "constants",
                    "spec_binding_status": "absent",
                    "identity_generation": 0,
                    "run_provenance_identity": constants_provenance.to_wire(),
                },
                "pins": [{"id": "source", "path": "/old/root/input"}],
                **stable,
            }
        },
        {
            "manifest": {
                "release_id": "microcosm-zone-2024-fixture-20260820T111213Z-deadbeef",
                "run_config": {
                    "config_authority": "bundle",
                    "spec_binding_status": "resolved",
                    "identity_generation": 1,
                    "run_provenance_identity": bundle_provenance.to_wire(),
                },
                "pins": [{"id": "source", "path": "/new/root/input"}],
                **stable,
            }
        },
    )


def _comparison_rules() -> list[dict[str, str]]:
    return [
        _rule("/release_id", "equal_after_normalizing_prefix"),
        _rule(
            "/run_config/config_authority",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/spec_binding_status",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/identity_generation",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/run_provenance_identity/identity_generation",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/run_provenance_identity/source_grammar_receipt",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/run_provenance_identity/spec_binding",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/run_provenance_identity/authority_versions/runtime_authority",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/run_provenance_identity/authority_versions/execution_abi",
            "expected_to_differ_by_generation",
        ),
        _rule(
            "/run_config/run_provenance_identity/execution_receipt/authority_mode",
            "expected_to_differ_by_generation",
        ),
        _rule("/pins/*/path", "operational_excluded"),
    ]


def test_plan_driven_comparison_emits_stable_empty_diff_receipt() -> None:
    abi = _execution_abi(rules=_comparison_rules())
    bundle_provenance = _bundle_run_provenance(execution_abi_sha256=str(abi["sha256"]))
    constants_receipts, bundle_receipts = _receipts(bundle_provenance)
    inputs = {
        "execution_abi": freeze_json(abi),
        "constants_artifacts": {"frame": b"same-frame", "bank": b"same-bank"},
        "bundle_artifacts": {"bank": b"same-bank", "frame": b"same-frame"},
        "constants_receipts": constants_receipts,
        "bundle_receipts": bundle_receipts,
        "bundle_run_provenance_identity": bundle_provenance,
    }

    first = compare_artifact_sets(**inputs)
    second = compare_artifact_sets(**inputs)

    assert first == second
    assert first.passed is True
    assert first.normative_equal is True
    assert first.receipts_equal_under_plan is True
    assert first.differences == ()
    assert first.constants_normative_sha256 == first.bundle_normative_sha256
    assert [row.stage_ref for row in first.stage_rows] == ["prepared", "modeled"]
    assert all(row.equal for row in first.stage_rows)
    assert [row.rule for row in first.receipt_rows[:2]] == [
        "operational_excluded",
        "equal_after_normalizing_prefix",
    ]
    assert all(
        row.rule == "expected_to_differ_by_generation"
        for row in first.receipt_rows[2:]
    )
    assert len(first.receipt_rows[2:]) == 9
    assert first.node_reuse_keys_equal is True
    assert len(first.node_reuse_key_rows) == 2
    operational = first.receipt_rows[0]
    assert operational.constants_value_sha256 is None
    assert operational.bundle_value_sha256 is None
    assert operational.raw_equal is None
    assert first.receipt_sha256 == sha256_json(first.body_wire())


def test_normative_byte_difference_is_never_normalized() -> None:
    abi = _execution_abi(rules=_comparison_rules())
    bundle_provenance = _bundle_run_provenance(execution_abi_sha256=str(abi["sha256"]))
    constants_receipts, bundle_receipts = _receipts(bundle_provenance)

    receipt = compare_artifact_sets(
        abi,
        constants_artifacts={"frame": b"constant", "bank": b"same-bank"},
        bundle_artifacts={"frame": b"bundle", "bank": b"same-bank"},
        constants_receipts=constants_receipts,
        bundle_receipts=bundle_receipts,
        bundle_run_provenance_identity=bundle_provenance,
    )

    assert receipt.passed is False
    assert receipt.normative_equal is False
    assert receipt.receipts_equal_under_plan is True
    assert receipt.stage_rows[0].equal is False
    assert receipt.stage_rows[1].equal is True
    assert [difference.kind for difference in receipt.differences] == [
        "normative_artifact_bytes"
    ]
    assert receipt.differences[0].subject == "frame"


@pytest.mark.parametrize(
    ("mode", "artifacts", "message"),
    [
        ("constants", {"frame": b"value"}, "missing=\\['bank'\\]"),
        (
            "bundle",
            {"frame": b"value", "bank": b"value", "extra": b"value"},
            "extra=\\['extra'\\]",
        ),
    ],
)
def test_missing_and_extra_normative_artifacts_fail_closed(
    mode: str, artifacts: dict[str, bytes], message: str
) -> None:
    inputs = {
        "constants_artifacts": {"frame": b"value", "bank": b"value"},
        "bundle_artifacts": {"frame": b"value", "bank": b"value"},
    }
    inputs[f"{mode}_artifacts"] = artifacts

    with pytest.raises(ArtifactComparisonError, match=message):
        compare_artifact_sets(
            _execution_abi(),
            constants_receipts={},
            bundle_receipts={},
            **inputs,
        )


def test_unmatched_receipt_difference_fails_closed() -> None:
    receipt_role = "checkpoint:fixture:receipts"
    constants = {
        "checkpoint:fixture:manifest": {"stable": {"verdict": "pass"}},
        receipt_role: checkpoint_receipt_surface(_checkpoint_sidecar()),
    }
    bundle = copy.deepcopy(constants)
    bundle["checkpoint:fixture:manifest"]["stable"]["verdict"] = "fail"  # type: ignore[index]

    with pytest.raises(
        ArtifactComparisonError,
        match=(
            r"unmatched receipt differences: "
            r"checkpoint:fixture:manifest:/stable/verdict"
        ),
    ):
        compare_artifact_sets(
            _execution_abi(receipt_roles=["checkpoint:fixture:manifest", receipt_role]),
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts=constants,
            bundle_receipts=bundle,
        )


def test_checkpoint_sidecars_close_the_complete_receipt_role_inventory() -> None:
    abi = _execution_abi(
        receipt_roles=[
            "checkpoint:fixture:manifest",
            "checkpoint:fixture:receipts",
        ]
    )
    artifacts = {"frame": b"same", "bank": b"same"}
    complete = {
        "checkpoint:fixture:manifest": {"stable": True},
        "checkpoint:fixture:receipts": checkpoint_receipt_surface(
            _checkpoint_sidecar()
        ),
    }

    receipt = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts=complete,
        bundle_receipts=complete,
    )
    assert receipt.passed is True

    with pytest.raises(
        ArtifactComparisonError,
        match=r"bundle receipt role inventory mismatch:.*checkpoint:fixture:receipts",
    ):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts=complete,
            bundle_receipts={
                "checkpoint:fixture:manifest": {"stable": True},
            },
        )

    with pytest.raises(
        ArtifactComparisonError,
        match=r"constants receipt role inventory mismatch:.*undeclared",
    ):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts={**complete, "undeclared": {}},
            bundle_receipts=complete,
        )


@pytest.mark.parametrize(
    ("receipt_policy", "surface", "expectation"),
    [
        ("required", checkpoint_receipt_surface(None), "must be present"),
        (
            "forbidden",
            checkpoint_receipt_surface(_checkpoint_sidecar()),
            "must be absent",
        ),
    ],
)
def test_checkpoint_sidecar_presence_is_bound_even_when_modes_agree(
    receipt_policy: str,
    surface: dict[str, object],
    expectation: str,
) -> None:
    receipt_role = "checkpoint:fixture:receipts"
    abi = _execution_abi(
        receipt_roles=["checkpoint:fixture:manifest", receipt_role],
        receipt_policy=receipt_policy,
    )
    receipts = {
        "checkpoint:fixture:manifest": {"stable": True},
        receipt_role: surface,
    }

    with pytest.raises(ArtifactComparisonError, match=expectation):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts=receipts,
            bundle_receipts=receipts,
        )


def test_forbidden_checkpoint_sidecar_accepts_exact_absence_surface() -> None:
    receipt_role = "checkpoint:fixture:receipts"
    abi = _execution_abi(
        receipt_roles=["checkpoint:fixture:manifest", receipt_role],
        receipt_policy="forbidden",
    )
    receipts = {
        "checkpoint:fixture:manifest": {"stable": True},
        receipt_role: checkpoint_receipt_surface(None),
    }

    result = compare_artifact_sets(
        abi,
        constants_artifacts={"frame": b"same", "bank": b"same"},
        bundle_artifacts={"frame": b"same", "bank": b"same"},
        constants_receipts=receipts,
        bundle_receipts=receipts,
    )

    assert result.passed is True


def test_checkpoint_receipt_wrapper_excludes_only_operational_surface() -> None:
    receipt_role = "checkpoint:fixture:receipts"
    abi = _execution_abi(
        receipt_roles=["checkpoint:fixture:manifest", receipt_role],
        rules=[
            _rule(
                "/operational",
                "operational_excluded",
                role=receipt_role,
                category="checkpoint_operational_receipt",
            )
        ],
    )
    artifacts = {"frame": b"same", "bank": b"same"}
    stable_manifest = {"identity_sha256": "a" * 64}
    constants = {
        "checkpoint:fixture:manifest": stable_manifest,
        receipt_role: checkpoint_receipt_surface(
            _checkpoint_sidecar(operational={"impute": {"path": "/constants/root"}})
        ),
    }
    bundle = {
        "checkpoint:fixture:manifest": stable_manifest,
        receipt_role: checkpoint_receipt_surface(
            _checkpoint_sidecar(operational={"impute": {"path": "/bundle/root"}})
        ),
    }

    receipt = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts=constants,
        bundle_receipts=bundle,
    )
    assert receipt.passed is True
    assert receipt.receipt_rows[0].rule == "operational_excluded"

    changed = copy.deepcopy(bundle)
    changed[receipt_role]["canonical"]["identity_sha256"] = "b" * 64  # type: ignore[index]
    with pytest.raises(ArtifactComparisonError, match="unmatched receipt differences"):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts=constants,
            bundle_receipts=changed,
        )


def test_checkpoint_receipt_surface_is_a_closed_trusted_split() -> None:
    sidecar = _checkpoint_sidecar(
        stage="transferred",
        operational={
            "impute": {"checkpoint_manifest_path": "/cold/root/manifest.json"}
        },
    )
    surface = checkpoint_receipt_surface(sidecar)
    assert surface == {
        "present": True,
        "canonical": {
            key: value
            for key, value in sidecar.items()
            if key != "operational_stage_receipts"
        },
        "operational": sidecar["operational_stage_receipts"],
    }
    assert checkpoint_receipt_surface(None) == {
        "present": False,
        "canonical": {},
        "operational": {},
    }

    for mutation in ("extra_outer", "missing_outer", "path_filename", "empty_ops"):
        changed = copy.deepcopy(sidecar)
        if mutation == "extra_outer":
            changed["hidden"] = "not allowed"
        elif mutation == "missing_outer":
            changed.pop("checkpoint")
        elif mutation == "path_filename":
            changed["checkpoint"]["filename"] = "/tmp/pool.h5"  # type: ignore[index]
        else:
            changed["operational_stage_receipts"] = {}
        with pytest.raises(ArtifactComparisonError):
            checkpoint_receipt_surface(changed)


def test_complete_equal_receipt_surfaces_enter_comparison_identity() -> None:
    receipt_role = "checkpoint:fixture:receipts"
    abi = _execution_abi(receipt_roles=["checkpoint:fixture:manifest", receipt_role])
    artifacts = {"frame": b"same", "bank": b"same"}
    receipt_surface = checkpoint_receipt_surface(_checkpoint_sidecar())

    first = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={
            "checkpoint:fixture:manifest": {"stable_unruled_field": "first"},
            receipt_role: receipt_surface,
        },
        bundle_receipts={
            "checkpoint:fixture:manifest": {"stable_unruled_field": "first"},
            receipt_role: receipt_surface,
        },
    )
    second = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={
            "checkpoint:fixture:manifest": {"stable_unruled_field": "second"},
            receipt_role: receipt_surface,
        },
        bundle_receipts={
            "checkpoint:fixture:manifest": {"stable_unruled_field": "second"},
            receipt_role: receipt_surface,
        },
    )

    assert first.receipt_rows == second.receipt_rows == ()
    assert first.constants_receipts_sha256 != second.constants_receipts_sha256
    assert first.bundle_receipts_sha256 != second.bundle_receipts_sha256
    assert first.receipt_sha256 != second.receipt_sha256


@pytest.mark.parametrize(
    "rules",
    [
        [
            _rule("/run", "operational_excluded"),
            _rule("/run", "operational_excluded"),
        ],
        [
            _rule("/run", "operational_excluded"),
            _rule("/run/id", "expected_to_differ_by_generation"),
        ],
        [_rule("/*/path", "operational_excluded")],
        [_rule("/paths/*", "operational_excluded")],
        [_rule("/paths/*/*/value", "operational_excluded")],
        [_rule("/paths/**/value", "operational_excluded")],
    ],
)
def test_duplicate_overlapping_and_broad_receipt_rules_are_rejected(
    rules: list[dict[str, str]],
) -> None:
    with pytest.raises(ArtifactComparisonError):
        compare_artifact_sets(
            _execution_abi(rules=rules),
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={"manifest": {"run": {"id": 0}, "paths": []}},
            bundle_receipts={"manifest": {"run": {"id": 1}, "paths": []}},
        )


def test_generation_expectations_are_derived_from_typed_identities() -> None:
    abi = _execution_abi(rules=_comparison_rules())
    constants_identity, bundle_identity = _identity_pair(abi)
    constants_receipts, bundle_receipts = _receipts(bundle_identity)
    receipt = _compare_artifact_sets(
        abi,
        constants_artifacts={"frame": b"same", "bank": b"same"},
        bundle_artifacts={"frame": b"same", "bank": b"same"},
        constants_receipts=constants_receipts,
        bundle_receipts=bundle_receipts,
        constants_run_provenance_identity=constants_identity,
        bundle_run_provenance_identity=bundle_identity,
        constants_node_reuse_keys={"node:first": "1" * 64, "node:second": "2" * 64},
        bundle_node_reuse_keys={"node:first": "1" * 64, "node:second": "2" * 64},
    )
    assert receipt.passed is True

    with pytest.raises(TypeError, match="generation_expectations"):
        _compare_artifact_sets(  # type: ignore[call-arg]
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts=constants_receipts,
            bundle_receipts=bundle_receipts,
            constants_run_provenance_identity=constants_identity,
            bundle_run_provenance_identity=bundle_identity,
            constants_node_reuse_keys={"node:first": "1" * 64, "node:second": "2" * 64},
            bundle_node_reuse_keys={"node:first": "1" * 64, "node:second": "2" * 64},
            generation_expectations={},
        )


def test_behavior_relevant_provenance_leaf_remains_raw_equal() -> None:
    abi = _execution_abi(rules=_comparison_rules())
    constants_identity, bundle_identity = _identity_pair(abi)
    changed_bundle = build_run_provenance_identity(
        identity_generation=1,
        source_grammar_receipt=bundle_identity.to_wire()["source_grammar_receipt"],
        spec_binding=bundle_identity.to_wire()["spec_binding"],
        authority_versions=bundle_identity.to_wire()["authority_versions"],
        code_inventory_digest=bundle_identity.code_inventory_digest,
        artifact_protocol_inventory=bundle_identity.to_wire()[
            "artifact_protocol_inventory"
        ],
        run_request={**bundle_identity.to_wire()["run_request"], "sample_seed": 999},
        execution_receipt=bundle_identity.to_wire()["execution_receipt"],
    )
    constants_receipts, bundle_receipts = _receipts(changed_bundle)
    constants_receipts["manifest"]["run_config"][  # type: ignore[index]
        "run_provenance_identity"
    ] = constants_identity.to_wire()

    with pytest.raises(ArtifactComparisonError, match="unmatched receipt differences"):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts=constants_receipts,
            bundle_receipts=bundle_receipts,
            constants_run_provenance_identity=constants_identity,
            bundle_run_provenance_identity=changed_bundle,
        )


def test_receipt_provenance_root_must_equal_typed_identity() -> None:
    abi = _execution_abi(rules=_comparison_rules())
    constants_identity, bundle_identity = _identity_pair(abi)
    constants_receipts, bundle_receipts = _receipts(bundle_identity)
    bundle_receipts["manifest"]["run_config"]["run_provenance_identity"][  # type: ignore[index]
        "code_inventory_digest"
    ] = "f" * 64

    with pytest.raises(
        ArtifactComparisonError,
        match="bundle receipt provenance differs from typed identity",
    ):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts=constants_receipts,
            bundle_receipts=bundle_receipts,
            constants_run_provenance_identity=constants_identity,
            bundle_run_provenance_identity=bundle_identity,
        )


@pytest.mark.parametrize(
    ("provenance_kwargs", "message"),
    [
        ({"attestation": "mirror-attested"}, "attestation must be bundle-authoritative"),
        ({"execution_abi_sha256": "f" * 64}, "differs from the compared execution ABI"),
    ],
)
def test_typed_bundle_provenance_binds_generation_and_execution_abi(
    provenance_kwargs: dict[str, str],
    message: str,
) -> None:
    abi = _execution_abi()
    provenance = _bundle_run_provenance(
        **{"execution_abi_sha256": str(abi["sha256"]), **provenance_kwargs}
    )
    with pytest.raises(ArtifactComparisonError, match=message):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={},
            bundle_receipts={},
            bundle_run_provenance_identity=provenance,
        )


def test_generation_rules_require_closed_leaf_semantics() -> None:
    with pytest.raises(
        ArtifactComparisonError,
        match="unknown generation-difference field semantics",
    ):
        compare_artifact_sets(
            _execution_abi(
                rules=[_rule("/run/identity", "expected_to_differ_by_generation")]
            ),
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={"manifest": {"run": {"identity": 0}}},
            bundle_receipts={"manifest": {"run": {"identity": 1}}},
        )


def test_node_reuse_key_maps_are_exact_and_receipted() -> None:
    abi = _execution_abi()
    result = compare_artifact_sets(
        abi,
        constants_artifacts={"frame": b"same", "bank": b"same"},
        bundle_artifacts={"frame": b"same", "bank": b"same"},
        constants_receipts={},
        bundle_receipts={},
        constants_node_reuse_keys={"node:first": "1" * 64, "node:second": "2" * 64},
        bundle_node_reuse_keys={"node:first": "1" * 64, "node:second": "3" * 64},
    )
    assert result.passed is False
    assert result.node_reuse_keys_equal is False
    assert result.differences[0].kind == "node_reuse_key"
    assert result.receipt_sha256 == sha256_json(result.body_wire())

    with pytest.raises(ArtifactComparisonError, match="node reuse key inventory"):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={},
            bundle_receipts={},
            constants_node_reuse_keys={"node:first": "1" * 64},
        )


def test_release_rule_strips_only_brand_and_terminal_timestamp_nonce() -> None:
    abi = _execution_abi(rules=[_rule("/release_id", "equal_after_normalizing_prefix")])
    artifacts = {"frame": b"same", "bank": b"same"}
    equal = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={
            "manifest": {
                "release_id": "populace-zone-2024-f001-s578-20260819T010203Z-a1b2c3d4"
            }
        },
        bundle_receipts={
            "manifest": {
                "release_id": "microcosm-zone-2024-f001-s578-20260820T111213Z-deadbeef"
            }
        },
    )
    assert equal.passed is True

    receipt = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={
            "manifest": {
                "release_id": "populace-zone-2024-a-20260819T010203Z-a1b2c3d4"
            }
        },
        bundle_receipts={
            "manifest": {
                "release_id": "microcosm-zone-2024-b-20260820T111213Z-deadbeef"
            }
        },
    )

    assert receipt.passed is False
    assert receipt.receipt_rows[0].rule_satisfied is False
    assert receipt.differences[0].rule == "equal_after_normalizing_prefix"


@pytest.mark.parametrize(
    ("constants_prefix", "bundle_prefix"),
    [("other", "microcosm"), ("populace", "other")],
)
def test_publication_prefix_rule_refuses_unapproved_prefixes(
    constants_prefix: str,
    bundle_prefix: str,
) -> None:
    abi = _execution_abi(rules=[_rule("/release_id", "equal_after_normalizing_prefix")])
    artifacts = {"frame": b"same", "bank": b"same"}

    receipt = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={
            "manifest": {
                "release_id": (
                    f"{constants_prefix}-zone-2024-a-"
                    "20260819T010203Z-a1b2c3d4"
                )
            }
        },
        bundle_receipts={
            "manifest": {
                "release_id": (
                    f"{bundle_prefix}-zone-2024-a-20260820T111213Z-deadbeef"
                )
            }
        },
    )

    assert receipt.passed is False
    assert receipt.receipt_rows[0].rule_satisfied is False


def test_operational_rule_cannot_hide_missing_structure() -> None:
    abi = _execution_abi(rules=[_rule("/pins/*/path", "operational_excluded")])
    artifacts = {"frame": b"same", "bank": b"same"}

    with pytest.raises(ArtifactComparisonError, match="rule scope differs"):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts={
                "manifest": {"pins": [{"id": "source", "path": "/constant"}]}
            },
            bundle_receipts={"manifest": {"pins": [{"id": "source"}]}},
        )


def test_execution_abi_seal_is_verified_before_comparison() -> None:
    abi = copy.deepcopy(_execution_abi())
    artifacts = abi["normative_artifact_vector"]
    assert isinstance(artifacts, list)
    artifacts[0]["stage_ref"] = "tampered"

    with pytest.raises(ArtifactComparisonError, match="execution_abi seal mismatch"):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={},
            bundle_receipts={},
        )

    stale = copy.deepcopy(_execution_abi())
    stale["code_abi"]["domain"] = "stacked-artifact-comparison-vector-v2"
    code_unsigned = {
        key: value
        for key, value in stale["code_abi"].items()
        if key != "implementation_sha256"
    }
    stale["code_abi"]["implementation_sha256"] = sha256_json(code_unsigned)
    _reseal(stale)
    with pytest.raises(ArtifactComparisonError, match="domain is unsupported"):
        compare_artifact_sets(
            stale,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={},
            bundle_receipts={},
        )

    stale = copy.deepcopy(_execution_abi())
    stale["code_abi"]["compiler_ir_abi_sha256"] = "f" * 64
    code_unsigned = {
        key: value
        for key, value in stale["code_abi"].items()
        if key != "implementation_sha256"
    }
    stale["code_abi"]["implementation_sha256"] = sha256_json(code_unsigned)
    _reseal(stale)
    with pytest.raises(ArtifactComparisonError, match="implementation attestation"):
        compare_artifact_sets(
            stale,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={},
            bundle_receipts={},
        )


def test_resealed_undeclared_selector_and_receipt_role_drift_fail_closed() -> None:
    abi = copy.deepcopy(_execution_abi())
    artifacts = abi["normative_artifact_vector"]
    assert isinstance(artifacts, list)
    artifacts[0]["content_selector_ref"] = "selector:undeclared"
    _reseal(abi)

    with pytest.raises(ArtifactComparisonError, match="undeclared content selector"):
        compare_artifact_sets(
            abi,
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={},
            bundle_receipts={},
        )

    with pytest.raises(ArtifactComparisonError, match="receipt role inventory"):
        receipt_role = "checkpoint:fixture:receipts"
        receipt_surface = checkpoint_receipt_surface(_checkpoint_sidecar())
        compare_artifact_sets(
            _execution_abi(receipt_roles=["checkpoint:fixture:manifest", receipt_role]),
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={
                "checkpoint:fixture:manifest": {},
                receipt_role: receipt_surface,
            },
            bundle_receipts={
                "checkpoint:fixture:manifest": {},
                receipt_role: receipt_surface,
                "unexpected": {},
            },
        )
