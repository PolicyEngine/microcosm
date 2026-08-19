from __future__ import annotations

import copy

import pytest

from microcosm.build.spec_engine.artifact_comparison import (
    ArtifactComparisonError,
    GenerationExpectation,
    checkpoint_receipt_surface,
    compare_artifact_sets,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.executor import (
    RunProvenanceIdentity,
    build_run_provenance_identity,
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
        "domain": "fixture-artifact-comparison-v1",
        "content_selectors": selectors,
        "locator_grammar": "fixture-closed-v1",
        "receipt_difference_match": "exactly_one_sealed_rule",
    }
    code_abi = {
        **code_unsigned,
        "implementation_sha256": sha256_json(code_unsigned),
    }
    unsigned = {
        "schema_version": 1,
        "present": True,
        "pipeline": {"id": "fixture"},
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
        "receipt_comparison_vector": rules or [],
        "resume_predicate": None,
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
            {"impute": {"status": "written"}}
            if operational is None
            else operational
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
            "runtime_authority": "c" * 64,
            "execution_abi": execution_abi_sha256,
        },
        code_inventory_digest="d" * 64,
        artifact_protocol_inventory={"fixture": "bytes-v1"},
        run_request={"config_authority": "bundle", "rung": "fixture"},
        execution_receipt={"backend": "cpu"},
    )


def _generation_zero_provenance() -> dict[str, object]:
    return {"identity_generation": 0}


def _receipts(
    bundle_provenance: RunProvenanceIdentity,
) -> tuple[dict[str, object], dict[str, object]]:
    stable = {
        "node_reuse_key": "a" * 64,
        "rows": [{"id": "one", "verdict": "pass"}],
    }
    return (
        {
            "manifest": {
                "release_id": "populace-zone-2024-fixture",
                "run_config": {
                    "config_authority": "constants",
                    "spec_binding_status": "absent",
                    "identity_generation": 0,
                    "run_provenance_identity": _generation_zero_provenance(),
                },
                "pins": [{"id": "source", "path": "/old/root/input"}],
                **stable,
            }
        },
        {
            "manifest": {
                "release_id": "microcosm-zone-2024-fixture",
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
            "/run_config/run_provenance_identity",
            "expected_to_differ_by_generation",
        ),
        _rule("/pins/*/path", "operational_excluded"),
    ]


def _expectations(
    bundle_provenance: RunProvenanceIdentity,
) -> dict[tuple[str, str], GenerationExpectation]:
    return {
        ("manifest", "/run_config/config_authority"): GenerationExpectation(
            constants_value="constants", bundle_value="bundle"
        ),
        ("manifest", "/run_config/spec_binding_status"): GenerationExpectation(
            constants_value="absent", bundle_value="resolved"
        ),
        ("manifest", "/run_config/identity_generation"): GenerationExpectation(
            constants_value=0, bundle_value=1
        ),
        (
            "manifest",
            "/run_config/run_provenance_identity",
        ): GenerationExpectation(
            constants_value=_generation_zero_provenance(),
            bundle_value=bundle_provenance.to_wire(),
        ),
    }


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
        "generation_expectations": _expectations(bundle_provenance),
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
    assert [row.rule for row in first.receipt_rows] == [
        "operational_excluded",
        "equal_after_normalizing_prefix",
            "expected_to_differ_by_generation",
            "expected_to_differ_by_generation",
            "expected_to_differ_by_generation",
            "expected_to_differ_by_generation",
        ]
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
        generation_expectations=_expectations(bundle_provenance),
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
            _execution_abi(
                receipt_roles=["checkpoint:fixture:manifest", receipt_role]
            ),
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
            _checkpoint_sidecar(
                operational={"impute": {"path": "/constants/root"}}
            )
        ),
    }
    bundle = {
        "checkpoint:fixture:manifest": stable_manifest,
        receipt_role: checkpoint_receipt_surface(
            _checkpoint_sidecar(
                operational={"impute": {"path": "/bundle/root"}}
            )
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
    abi = _execution_abi(
        receipt_roles=["checkpoint:fixture:manifest", receipt_role]
    )
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


def test_generation_rule_requires_and_checks_both_expected_values() -> None:
    abi = _execution_abi(
        rules=[
            _rule(
                "/run_config/config_authority",
                "expected_to_differ_by_generation",
            )
        ]
    )
    constants_receipts = {"manifest": {"run_config": {"config_authority": "constants"}}}
    bundle_receipts = {"manifest": {"run_config": {"config_authority": "bundle"}}}
    artifacts = {"frame": b"same", "bank": b"same"}

    with pytest.raises(
        ArtifactComparisonError, match="generation expectation inventory mismatch"
    ):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts=constants_receipts,
            bundle_receipts=bundle_receipts,
        )

    wrong = {
        ("manifest", "/run_config/config_authority"): GenerationExpectation(
            constants_value="constants",
            bundle_value="constants_adapter",
        )
    }
    with pytest.raises(
        ArtifactComparisonError,
        match="differs from the sealed generic transition",
    ):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts=constants_receipts,
            bundle_receipts=bundle_receipts,
            generation_expectations=wrong,
        )


def test_run_provenance_expectation_is_bound_to_explicit_typed_identity() -> None:
    pointer = "/run_config/run_provenance_identity"
    abi = _execution_abi(rules=[_rule(pointer, "expected_to_differ_by_generation")])
    execution_abi_sha256 = str(abi["sha256"])
    trusted = _bundle_run_provenance(
        execution_abi_sha256=execution_abi_sha256,
        spec_sha256="a" * 64,
    )
    arbitrary = _bundle_run_provenance(
        execution_abi_sha256=execution_abi_sha256,
        spec_sha256="b" * 64,
    )
    artifacts = {"frame": b"same", "bank": b"same"}
    constants_receipts = {
        "manifest": {
            "run_config": {"run_provenance_identity": _generation_zero_provenance()}
        }
    }
    bundle_receipts = {
        "manifest": {"run_config": {"run_provenance_identity": arbitrary.to_wire()}}
    }
    arbitrary_expectation = {
        ("manifest", pointer): GenerationExpectation(
            constants_value=_generation_zero_provenance(),
            bundle_value=arbitrary.to_wire(),
        )
    }

    with pytest.raises(
        ArtifactComparisonError,
        match="differs from the explicit typed identity",
    ):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts=constants_receipts,
            bundle_receipts=bundle_receipts,
            generation_expectations=arbitrary_expectation,
            bundle_run_provenance_identity=trusted,
        )


def test_bundle_receipt_must_equal_typed_run_provenance_identity() -> None:
    pointer = "/run_config/run_provenance_identity"
    abi = _execution_abi(rules=[_rule(pointer, "expected_to_differ_by_generation")])
    execution_abi_sha256 = str(abi["sha256"])
    trusted = _bundle_run_provenance(
        execution_abi_sha256=execution_abi_sha256,
        spec_sha256="a" * 64,
    )
    arbitrary = _bundle_run_provenance(
        execution_abi_sha256=execution_abi_sha256,
        spec_sha256="b" * 64,
    )
    artifacts = {"frame": b"same", "bank": b"same"}

    receipt = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={
            "manifest": {
                "run_config": {"run_provenance_identity": _generation_zero_provenance()}
            }
        },
        bundle_receipts={
            "manifest": {"run_config": {"run_provenance_identity": arbitrary.to_wire()}}
        },
        generation_expectations={
            ("manifest", pointer): GenerationExpectation(
                constants_value=_generation_zero_provenance(),
                bundle_value=trusted.to_wire(),
            )
        },
        bundle_run_provenance_identity=trusted,
    )

    assert receipt.passed is False
    assert receipt.receipts_equal_under_plan is False
    assert receipt.differences[0].kind == "receipt_rule_violation"


@pytest.mark.parametrize(
    ("provenance_kwargs", "message"),
    [
        (
            {"attestation": "mirror-attested"},
            "attestation must be bundle-authoritative",
        ),
        (
            {"execution_abi_sha256": "f" * 64},
            "differs from the compared execution ABI",
        ),
    ],
)
def test_typed_bundle_provenance_must_bind_bundle_and_execution_abi(
    provenance_kwargs: dict[str, str],
    message: str,
) -> None:
    pointer = "/run_config/run_provenance_identity"
    abi = _execution_abi(rules=[_rule(pointer, "expected_to_differ_by_generation")])
    arguments = {"execution_abi_sha256": str(abi["sha256"]), **provenance_kwargs}
    provenance = _bundle_run_provenance(**arguments)
    artifacts = {"frame": b"same", "bank": b"same"}

    with pytest.raises(ArtifactComparisonError, match=message):
        compare_artifact_sets(
            abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts={
                "manifest": {
                    "run_config": {
                        "run_provenance_identity": _generation_zero_provenance()
                    }
                }
            },
            bundle_receipts={
                "manifest": {
                    "run_config": {"run_provenance_identity": provenance.to_wire()}
                }
            },
            generation_expectations={
                ("manifest", pointer): GenerationExpectation(
                    constants_value=_generation_zero_provenance(),
                    bundle_value=provenance.to_wire(),
                )
            },
            bundle_run_provenance_identity=provenance,
        )


def test_generation_rules_require_closed_generic_field_semantics() -> None:
    with pytest.raises(
        ArtifactComparisonError,
        match="unknown generation-difference field semantics 'identity'",
    ):
        compare_artifact_sets(
            _execution_abi(
                rules=[_rule("/run/identity", "expected_to_differ_by_generation")]
            ),
            constants_artifacts={"frame": b"same", "bank": b"same"},
            bundle_artifacts={"frame": b"same", "bank": b"same"},
            constants_receipts={"manifest": {"run": {"identity": 0}}},
            bundle_receipts={"manifest": {"run": {"identity": 1}}},
            generation_expectations={
                ("manifest", "/run/identity"): GenerationExpectation(0, 1)
            },
        )


def test_run_provenance_identity_argument_is_required_and_cannot_be_unused() -> None:
    pointer = "/run_config/run_provenance_identity"
    provenance_abi = _execution_abi(
        rules=[_rule(pointer, "expected_to_differ_by_generation")]
    )
    provenance = _bundle_run_provenance(
        execution_abi_sha256=str(provenance_abi["sha256"])
    )
    artifacts = {"frame": b"same", "bank": b"same"}
    expectation = {
        ("manifest", pointer): GenerationExpectation(
            constants_value=_generation_zero_provenance(),
            bundle_value=provenance.to_wire(),
        )
    }
    receipts = (
        {
            "manifest": {
                "run_config": {"run_provenance_identity": _generation_zero_provenance()}
            }
        },
        {"manifest": {"run_config": {"run_provenance_identity": provenance.to_wire()}}},
    )

    with pytest.raises(
        ArtifactComparisonError,
        match="must be an explicit typed RunProvenanceIdentity",
    ):
        compare_artifact_sets(
            provenance_abi,
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts=receipts[0],
            bundle_receipts=receipts[1],
            generation_expectations=expectation,
        )

    with pytest.raises(ArtifactComparisonError, match="unused by the execution ABI"):
        compare_artifact_sets(
            _execution_abi(),
            constants_artifacts=artifacts,
            bundle_artifacts=artifacts,
            constants_receipts={},
            bundle_receipts={},
            bundle_run_provenance_identity=provenance,
        )


def test_publication_prefix_rule_compares_the_complete_suffix() -> None:
    abi = _execution_abi(rules=[_rule("/release_id", "equal_after_normalizing_prefix")])
    artifacts = {"frame": b"same", "bank": b"same"}
    receipt = compare_artifact_sets(
        abi,
        constants_artifacts=artifacts,
        bundle_artifacts=artifacts,
        constants_receipts={"manifest": {"release_id": "populace-zone-2024-a"}},
        bundle_receipts={"manifest": {"release_id": "microcosm-zone-2024-b"}},
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
            "manifest": {"release_id": f"{constants_prefix}-zone-2024-a"}
        },
        bundle_receipts={"manifest": {"release_id": f"{bundle_prefix}-zone-2024-a"}},
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
            _execution_abi(
                receipt_roles=["checkpoint:fixture:manifest", receipt_role]
            ),
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
