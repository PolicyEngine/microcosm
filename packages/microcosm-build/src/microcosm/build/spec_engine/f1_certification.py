"""Typed, fail-closed evidence for the US F1 four-build certification.

The physical pool build owns locator construction and collection of selected
artifact bytes.  This module is the narrow handoff: production emits one
sealed evidence document, the cold runner adds the resume-forbidden audit,
and the four-receipt comparator applies the compiler-issued execution ABI.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

from .artifact_comparison import (
    ArtifactComparisonError,
    ArtifactDigest,
    compare_artifact_digest_sets,
    receipt_determinism_projection,
)
from .canonical import canonical_json_bytes, sha256_json
from .errors import SpecValidationError
from .executor import (
    ExecutorError,
    RunProvenanceIdentity,
    build_run_provenance_identity,
)
from .plan_lock import PLAN_LOCK_SCHEMA_ID
from .schemas import load_schema_registry

PRODUCTION_EVIDENCE_FILENAME = "us-f1-production-evidence.json"
COLD_BUILD_RECEIPT_FILENAME = "us-f1-build-receipt.json"
CERTIFICATION_JSON_FILENAME = "us-f1-certification.json"
CERTIFICATION_MARKDOWN_FILENAME = "us-f1-certification.md"

_PRODUCTION_EVIDENCE_KIND = "microcosm_us_f1_production_evidence"
_COLD_RECEIPT_KIND = "microcosm_us_f1_cold_build_receipt"
_CERTIFICATION_KIND = "microcosm_us_f1_certification"
_NORMATIVE_VECTOR_DOMAIN = "microcosm.spec-engine.normative-artifact-vector-digest.v1"
_RECEIPT_SURFACE_DOMAIN = "microcosm.spec-engine.receipt-surface-digest.v1"
_NODE_REUSE_KEYS_DOMAIN = "microcosm.spec-engine.node-reuse-key-map.v1"
_PLAN_ARTIFACT_KEYS = frozenset(
    {
        "id",
        "kind",
        "producer_ref",
        "stage_ref",
        "protocol_ref",
        "locator_ref",
        "content_selector_ref",
        "surface",
        "comparison",
        "required",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "identity_generation",
        "source_grammar_receipt",
        "spec_binding",
        "authority_versions",
        "code_inventory_digest",
        "artifact_protocol_inventory",
        "run_request",
        "execution_receipt",
    }
)
_STANDARD_SAMPLE_FRACTIONS = frozenset({0.01, 0.04, 0.10, 0.25, 1.0})
_POOL_COVERAGE_DOMAIN = "microcosm.us-pool-artifact-member-coverage.v1"
_CALIBRATION_COVERAGE_DOMAIN = "microcosm.us-f1-calibration-scope-coverage.v1"
_FIXTURE_SELECTOR_COVERAGE_DOMAIN = "microcosm.f1-test.selector-coverage.v1"
_FIXTURE_CALIBRATION_COVERAGE_DOMAIN = "microcosm.f1-test.calibration-scope-coverage.v1"
_FIXTURE_PIPELINE_ID = "fixture"
_DIRECTORY_SELECTOR = "selector:directory_tree_bytes_v1"
_H5_SELECTORS = frozenset(
    {
        "selector:h5_all_entity_tables_and_columns_v1",
        "selector:h5_all_weight_vectors_v1",
    }
)


class F1CertificationError(ValueError):
    """Certification evidence is malformed, incomplete, or incomparable."""


@dataclass(frozen=True, slots=True)
class F1ArtifactDigestRow:
    """One execution-ABI artifact contract plus its selected-byte digest."""

    artifact_id: str
    kind: str
    producer_ref: str
    stage_ref: str
    protocol_ref: str
    locator_ref: str
    content_selector_ref: str
    surface: str
    comparison: str
    required: bool
    digest: ArtifactDigest

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> F1ArtifactDigestRow:
        row = _object(value, location="artifact")
        _exact_keys(
            row,
            _PLAN_ARTIFACT_KEYS | {"sha256", "byte_size"},
            location="artifact",
        )
        for key in _PLAN_ARTIFACT_KEYS - {"required"}:
            _nonempty_string(row[key], location=f"artifact/{key}")
        if not isinstance(row["required"], bool):
            raise F1CertificationError("artifact/required: boolean required")
        digest = _artifact_digest(row["sha256"], row["byte_size"], "artifact")
        return cls(
            artifact_id=str(row["id"]),
            kind=str(row["kind"]),
            producer_ref=str(row["producer_ref"]),
            stage_ref=str(row["stage_ref"]),
            protocol_ref=str(row["protocol_ref"]),
            locator_ref=str(row["locator_ref"]),
            content_selector_ref=str(row["content_selector_ref"]),
            surface=str(row["surface"]),
            comparison=str(row["comparison"]),
            required=bool(row["required"]),
            digest=digest,
        )

    def contract_wire(self) -> dict[str, object]:
        return {
            "id": self.artifact_id,
            "kind": self.kind,
            "producer_ref": self.producer_ref,
            "stage_ref": self.stage_ref,
            "protocol_ref": self.protocol_ref,
            "locator_ref": self.locator_ref,
            "content_selector_ref": self.content_selector_ref,
            "surface": self.surface,
            "comparison": self.comparison,
            "required": self.required,
        }

    def to_wire(self) -> dict[str, object]:
        return {**self.contract_wire(), **self.digest.to_wire()}


@dataclass(frozen=True, slots=True)
class F1CoverageEvidence:
    """Closed inventory evidence supplied by the production locator seam."""

    plan_artifact_ids: tuple[str, ...]
    bound_locator_refs: tuple[str, ...]
    node_reuse_ids: tuple[str, ...]
    node_reuse_inventory_complete: bool
    selector_coverage_receipt: Mapping[str, object]
    selector_coverage_receipt_sha256: str
    selector_inventory_complete: bool
    calibration_scope_receipt: Mapping[str, object]
    calibration_scope_receipt_sha256: str
    calibration_scope_complete: bool
    complete: bool

    def __post_init__(self) -> None:
        for name in ("plan_artifact_ids", "bound_locator_refs", "node_reuse_ids"):
            values = getattr(self, name)
            if (
                _validated_string_sequence(values, location=f"coverage/{name}")
                != values
            ):
                raise F1CertificationError(f"coverage/{name}: tuple required")
        for name in (
            "node_reuse_inventory_complete",
            "selector_inventory_complete",
            "calibration_scope_complete",
            "complete",
        ):
            if not isinstance(getattr(self, name), bool):
                raise F1CertificationError(f"coverage/{name}: boolean required")
        selector = _object(
            self.selector_coverage_receipt,
            location="coverage/selector_coverage_receipt",
        )
        calibration = _object(
            self.calibration_scope_receipt,
            location="coverage/calibration_scope_receipt",
        )
        if self.selector_coverage_receipt_sha256 != sha256_json(selector):
            raise F1CertificationError(
                "coverage/selector_coverage_receipt_sha256: mismatch"
            )
        if self.calibration_scope_receipt_sha256 != sha256_json(calibration):
            raise F1CertificationError(
                "coverage/calibration_scope_receipt_sha256: mismatch"
            )
        _validate_selector_coverage_receipt(
            selector,
            expected=self.selector_inventory_complete,
        )
        _validate_calibration_coverage_receipt(
            calibration,
            expected=self.calibration_scope_complete,
        )
        if self.complete is not (
            self.node_reuse_inventory_complete
            and self.selector_inventory_complete
            and self.calibration_scope_complete
        ):
            raise F1CertificationError(
                "coverage/complete differs from typed component verdicts"
            )

    @classmethod
    def create(
        cls,
        *,
        plan_artifact_ids: Sequence[str],
        bound_locator_refs: Sequence[str],
        node_reuse_ids: Sequence[str],
        node_reuse_inventory_complete: bool,
        selector_coverage_receipt: Mapping[str, object],
        selector_inventory_complete: bool,
        calibration_scope_receipt: Mapping[str, object],
        calibration_scope_complete: bool,
    ) -> F1CoverageEvidence:
        selector = _object(
            selector_coverage_receipt,
            location="coverage/selector_coverage_receipt",
        )
        calibration = _object(
            calibration_scope_receipt,
            location="coverage/calibration_scope_receipt",
        )
        if not isinstance(selector_inventory_complete, bool):
            raise F1CertificationError(
                "coverage/selector_inventory_complete: boolean required"
            )
        if not isinstance(node_reuse_inventory_complete, bool):
            raise F1CertificationError(
                "coverage/node_reuse_inventory_complete: boolean required"
            )
        if not isinstance(calibration_scope_complete, bool):
            raise F1CertificationError(
                "coverage/calibration_scope_complete: boolean required"
            )
        _validate_selector_coverage_receipt(
            selector,
            expected=selector_inventory_complete,
        )
        _validate_calibration_coverage_receipt(
            calibration,
            expected=calibration_scope_complete,
        )
        return cls(
            plan_artifact_ids=_validated_string_sequence(
                plan_artifact_ids, location="coverage/plan_artifact_ids"
            ),
            bound_locator_refs=_validated_string_sequence(
                bound_locator_refs, location="coverage/bound_locator_refs"
            ),
            node_reuse_ids=_validated_string_sequence(
                node_reuse_ids, location="coverage/node_reuse_ids"
            ),
            node_reuse_inventory_complete=node_reuse_inventory_complete,
            selector_coverage_receipt=selector,
            selector_coverage_receipt_sha256=sha256_json(selector),
            selector_inventory_complete=selector_inventory_complete,
            calibration_scope_receipt=calibration,
            calibration_scope_receipt_sha256=sha256_json(calibration),
            calibration_scope_complete=calibration_scope_complete,
            complete=(
                node_reuse_inventory_complete
                and selector_inventory_complete
                and calibration_scope_complete
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> F1CoverageEvidence:
        row = _object(value, location="coverage")
        _exact_keys(
            row,
            frozenset(
                {
                    "plan_artifact_ids",
                    "bound_locator_refs",
                    "node_reuse_ids",
                    "node_reuse_inventory_complete",
                    "selector_coverage_receipt",
                    "selector_coverage_receipt_sha256",
                    "selector_inventory_complete",
                    "calibration_scope_receipt",
                    "calibration_scope_receipt_sha256",
                    "calibration_scope_complete",
                    "complete",
                }
            ),
            location="coverage",
        )
        for key in (
            "node_reuse_inventory_complete",
            "selector_inventory_complete",
            "calibration_scope_complete",
            "complete",
        ):
            if not isinstance(row[key], bool):
                raise F1CertificationError(f"coverage/{key}: boolean required")
        candidate = cls.create(
            plan_artifact_ids=_string_tuple(
                row["plan_artifact_ids"], location="coverage/plan_artifact_ids"
            ),
            bound_locator_refs=_string_tuple(
                row["bound_locator_refs"], location="coverage/bound_locator_refs"
            ),
            node_reuse_ids=_string_tuple(
                row["node_reuse_ids"], location="coverage/node_reuse_ids"
            ),
            node_reuse_inventory_complete=bool(row["node_reuse_inventory_complete"]),
            selector_coverage_receipt=_object(
                row["selector_coverage_receipt"],
                location="coverage/selector_coverage_receipt",
            ),
            selector_inventory_complete=bool(row["selector_inventory_complete"]),
            calibration_scope_receipt=_object(
                row["calibration_scope_receipt"],
                location="coverage/calibration_scope_receipt",
            ),
            calibration_scope_complete=bool(row["calibration_scope_complete"]),
        )
        expected_digests = {
            "selector_coverage_receipt_sha256": (
                candidate.selector_coverage_receipt_sha256
            ),
            "calibration_scope_receipt_sha256": (
                candidate.calibration_scope_receipt_sha256
            ),
        }
        for key, expected in expected_digests.items():
            if _sha256(row[key], location=f"coverage/{key}") != expected:
                raise F1CertificationError(f"coverage/{key}: mismatch")
        if row["complete"] is not candidate.complete:
            raise F1CertificationError(
                "coverage/complete differs from typed component verdicts"
            )
        return candidate

    def to_wire(self) -> dict[str, object]:
        return {
            "plan_artifact_ids": list(self.plan_artifact_ids),
            "bound_locator_refs": list(self.bound_locator_refs),
            "node_reuse_ids": list(self.node_reuse_ids),
            "node_reuse_inventory_complete": self.node_reuse_inventory_complete,
            "selector_coverage_receipt": dict(self.selector_coverage_receipt),
            "selector_coverage_receipt_sha256": (self.selector_coverage_receipt_sha256),
            "selector_inventory_complete": self.selector_inventory_complete,
            "calibration_scope_receipt": dict(self.calibration_scope_receipt),
            "calibration_scope_receipt_sha256": (self.calibration_scope_receipt_sha256),
            "calibration_scope_complete": self.calibration_scope_complete,
            "complete": self.complete,
        }

    def passes(self, plan_lock: Mapping[str, object]) -> bool:
        try:
            _validate_coverage_plan_links(self, plan_lock)
        except F1CertificationError:
            return False
        execution_abi = _execution_abi(plan_lock)
        artifacts = _plan_artifact_rows(execution_abi)
        pipeline = _object(execution_abi["pipeline"], location="execution_abi/pipeline")
        expected_nodes = _string_tuple(
            pipeline.get("producer_order"),
            location="execution_abi/pipeline/producer_order",
        )
        expected_ids = tuple(str(row["id"]) for row in artifacts)
        expected_locators = tuple(
            sorted({str(row["locator_ref"]) for row in artifacts})
        )
        return (
            self.complete
            and self.node_reuse_inventory_complete
            and self.selector_inventory_complete
            and self.calibration_scope_complete
            and self.plan_artifact_ids == expected_ids
            and self.bound_locator_refs == expected_locators
            and self.node_reuse_ids == expected_nodes
        )


@dataclass(frozen=True, slots=True)
class F1ResumeAudit:
    """Typed zero-resume predicate extracted from the publication receipt."""

    deepest_resumed_stage: str | None
    stages: tuple[Mapping[str, object], ...]
    primary_qrf: Mapping[str, object]
    target_banks: tuple[Mapping[str, object], ...]
    total_resume_count: int
    passed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> F1ResumeAudit:
        row = _object(value, location="resume_audit")
        _exact_keys(
            row,
            frozenset(
                {
                    "deepest_resumed_stage",
                    "stages",
                    "primary_qrf",
                    "target_banks",
                    "total_resume_count",
                    "passed",
                }
            ),
            location="resume_audit",
        )
        deepest = row["deepest_resumed_stage"]
        if deepest is not None:
            _nonempty_string(deepest, location="resume_audit/deepest_resumed_stage")
        stages_value = _array(row["stages"], location="resume_audit/stages")
        stages: list[dict[str, object]] = []
        for index, value_row in enumerate(stages_value):
            location = f"resume_audit/stages/{index}"
            stage = _object(value_row, location=location)
            _exact_keys(
                stage,
                frozenset({"stage_ref", "resumed_count"}),
                location=location,
            )
            _nonempty_string(stage["stage_ref"], location=f"{location}/stage_ref")
            _nonnegative_int(
                stage["resumed_count"], location=f"{location}/resumed_count"
            )
            stages.append(stage)
        primary = _object(row["primary_qrf"], location="resume_audit/primary_qrf")
        _exact_keys(
            primary,
            frozenset({"resume_status", "resumed_count"}),
            location="resume_audit/primary_qrf",
        )
        _nonempty_string(
            primary["resume_status"],
            location="resume_audit/primary_qrf/resume_status",
        )
        _nonnegative_int(
            primary["resumed_count"],
            location="resume_audit/primary_qrf/resumed_count",
        )
        bank_values = _array(row["target_banks"], location="resume_audit/target_banks")
        banks: list[dict[str, object]] = []
        for index, value_row in enumerate(bank_values):
            location = f"resume_audit/target_banks/{index}"
            bank = _object(value_row, location=location)
            _exact_keys(
                bank,
                frozenset(
                    {
                        "bank_ref",
                        "target_count",
                        "targets",
                        "load_status_resumed_count",
                        "source_checkpoint_count",
                    }
                ),
                location=location,
            )
            _nonempty_string(bank["bank_ref"], location=f"{location}/bank_ref")
            target_count = _nonnegative_int(
                bank["target_count"], location=f"{location}/target_count"
            )
            target_values = _array(bank["targets"], location=f"{location}/targets")
            targets: list[dict[str, object]] = []
            for target_index, target_value in enumerate(target_values):
                target_location = f"{location}/targets/{target_index}"
                target = _object(target_value, location=target_location)
                _exact_keys(
                    target,
                    frozenset(
                        {
                            "target_ref",
                            "load_status_resumed_count",
                            "source_checkpoint_count",
                        }
                    ),
                    location=target_location,
                )
                _nonempty_string(
                    target["target_ref"], location=f"{target_location}/target_ref"
                )
                for key in (
                    "load_status_resumed_count",
                    "source_checkpoint_count",
                ):
                    count = _nonnegative_int(
                        target[key], location=f"{target_location}/{key}"
                    )
                    if count not in {0, 1}:
                        raise F1CertificationError(
                            f"{target_location}/{key}: zero or one required"
                        )
                targets.append(target)
            target_refs = [str(target["target_ref"]) for target in targets]
            if len(target_refs) != len(set(target_refs)):
                raise F1CertificationError(
                    f"{location}/targets: unique target_ref values required"
                )
            load_count = _nonnegative_int(
                bank["load_status_resumed_count"],
                location=f"{location}/load_status_resumed_count",
            )
            source_count = _nonnegative_int(
                bank["source_checkpoint_count"],
                location=f"{location}/source_checkpoint_count",
            )
            if (
                target_count != len(targets)
                or load_count
                != sum(int(target["load_status_resumed_count"]) for target in targets)
                or source_count
                != sum(int(target["source_checkpoint_count"]) for target in targets)
            ):
                raise F1CertificationError(
                    f"{location}: bank counts differ from per-target audit rows"
                )
            bank = {**bank, "targets": targets}
            banks.append(bank)
        total = _nonnegative_int(
            row["total_resume_count"], location="resume_audit/total_resume_count"
        )
        if not isinstance(row["passed"], bool):
            raise F1CertificationError("resume_audit/passed: boolean required")
        expected_total = (
            sum(int(stage["resumed_count"]) for stage in stages)
            + int(primary["resumed_count"])
            + sum(
                int(bank["load_status_resumed_count"])
                + int(bank["source_checkpoint_count"])
                for bank in banks
            )
        )
        expected_passed = (
            deepest is None
            and primary["resume_status"] == "initialized"
            and expected_total == 0
        )
        if total != expected_total or row["passed"] is not expected_passed:
            raise F1CertificationError(
                "resume_audit totals or verdict differ from typed resume rows"
            )
        return cls(
            deepest_resumed_stage=deepest,
            stages=tuple(stages),
            primary_qrf=primary,
            target_banks=tuple(banks),
            total_resume_count=total,
            passed=expected_passed,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "deepest_resumed_stage": self.deepest_resumed_stage,
            "stages": [dict(row) for row in self.stages],
            "primary_qrf": dict(self.primary_qrf),
            "target_banks": [dict(row) for row in self.target_banks],
            "total_resume_count": self.total_resume_count,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class F1ProductionEvidence:
    """Post-publication evidence emitted only through the pool's typed seam."""

    mode: str
    plan_lock: Mapping[str, object]
    artifacts: tuple[F1ArtifactDigestRow, ...]
    receipt_surfaces: Mapping[str, object]
    run_provenance_identity: RunProvenanceIdentity
    node_reuse_keys: Mapping[str, str]
    coverage: F1CoverageEvidence
    evidence_sha256: str

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        plan_lock: Mapping[str, object],
        artifacts: Mapping[str, bytes | ArtifactDigest],
        receipt_surfaces: Mapping[str, object],
        run_provenance_identity: RunProvenanceIdentity,
        node_reuse_keys: Mapping[str, str],
        coverage: F1CoverageEvidence,
    ) -> F1ProductionEvidence:
        normalized_plan = _validated_plan_lock(plan_lock)
        artifact_rows = _artifact_rows_from_surfaces(normalized_plan, artifacts)
        candidate = cls(
            mode=_mode(mode),
            plan_lock=normalized_plan,
            artifacts=artifact_rows,
            receipt_surfaces=_object(receipt_surfaces, location="receipt_surfaces"),
            run_provenance_identity=run_provenance_identity,
            node_reuse_keys=_node_reuse_keys(node_reuse_keys),
            coverage=coverage,
            evidence_sha256="0" * 64,
        )
        candidate._validate_components()
        digest = sha256_json(candidate.body_wire())
        return cls(
            mode=candidate.mode,
            plan_lock=candidate.plan_lock,
            artifacts=candidate.artifacts,
            receipt_surfaces=candidate.receipt_surfaces,
            run_provenance_identity=candidate.run_provenance_identity,
            node_reuse_keys=candidate.node_reuse_keys,
            coverage=candidate.coverage,
            evidence_sha256=digest,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> F1ProductionEvidence:
        row = _object(value, location="production_evidence")
        _exact_keys(
            row,
            frozenset(
                {
                    "artifact_kind",
                    "schema_version",
                    "mode",
                    "plan_lock",
                    "artifacts",
                    "receipt_surfaces",
                    "run_provenance_identity",
                    "node_reuse_keys",
                    "coverage",
                    "evidence_sha256",
                }
            ),
            location="production_evidence",
        )
        if row["artifact_kind"] != _PRODUCTION_EVIDENCE_KIND:
            raise F1CertificationError("production_evidence/artifact_kind: mismatch")
        if row["schema_version"] != 1:
            raise F1CertificationError(
                "production_evidence/schema_version: unsupported"
            )
        artifacts = tuple(
            F1ArtifactDigestRow.from_mapping(value_row)
            for value_row in _array(
                row["artifacts"], location="production_evidence/artifacts"
            )
        )
        candidate = cls(
            mode=_mode(row["mode"]),
            plan_lock=_validated_plan_lock(row["plan_lock"]),
            artifacts=artifacts,
            receipt_surfaces=_object(
                row["receipt_surfaces"],
                location="production_evidence/receipt_surfaces",
            ),
            run_provenance_identity=_run_provenance(row["run_provenance_identity"]),
            node_reuse_keys=_node_reuse_keys(row["node_reuse_keys"]),
            coverage=F1CoverageEvidence.from_mapping(row["coverage"]),
            evidence_sha256=_sha256(
                row["evidence_sha256"],
                location="production_evidence/evidence_sha256",
            ),
        )
        candidate._validate_components()
        if candidate.evidence_sha256 != sha256_json(candidate.body_wire()):
            raise F1CertificationError("production evidence seal mismatch")
        return candidate

    def _validate_components(self) -> None:
        if not isinstance(self.coverage, F1CoverageEvidence):
            raise F1CertificationError("typed F1 coverage evidence required")
        execution_abi = _execution_abi(self.plan_lock)
        _validate_coverage_plan_links(self.coverage, self.plan_lock)
        _validate_artifact_rows(self.plan_lock, self.artifacts)
        expected_nodes = _producer_order(execution_abi)
        observed_nodes = set(self.node_reuse_keys)
        coverage_nodes = set(self.coverage.node_reuse_ids)
        if observed_nodes != coverage_nodes:
            raise F1CertificationError(
                "production evidence node reuse keys differ from coverage ids"
            )
        if not observed_nodes.issubset(expected_nodes):
            raise F1CertificationError(
                "production evidence contains node reuse ids outside the plan"
            )
        ordered_observed = tuple(
            node_id for node_id in expected_nodes if node_id in observed_nodes
        )
        if self.coverage.node_reuse_ids != ordered_observed:
            raise F1CertificationError(
                "production evidence node reuse ids differ from plan order"
            )
        if self.coverage.node_reuse_inventory_complete and observed_nodes != set(
            expected_nodes
        ):
            raise F1CertificationError(
                "complete production node reuse inventory differs from plan"
            )
        try:
            receipt_determinism_projection(
                execution_abi,
                authority_mode=self.mode,
                receipts=self.receipt_surfaces,
                run_provenance_identity=self.run_provenance_identity,
            )
        except ArtifactComparisonError as error:
            raise F1CertificationError(
                f"production evidence receipt surface is invalid: {error}"
            ) from error

    def body_wire(self) -> dict[str, object]:
        return {
            "artifact_kind": _PRODUCTION_EVIDENCE_KIND,
            "schema_version": 1,
            "mode": self.mode,
            "plan_lock": dict(self.plan_lock),
            "artifacts": [row.to_wire() for row in self.artifacts],
            "receipt_surfaces": dict(self.receipt_surfaces),
            "run_provenance_identity": self.run_provenance_identity.to_wire(),
            "node_reuse_keys": dict(self.node_reuse_keys),
            "coverage": self.coverage.to_wire(),
        }

    def to_wire(self) -> dict[str, object]:
        return {**self.body_wire(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class F1RunRequest:
    sample_fraction: float
    seed: int
    clone_attachment_fraction: float = 1.0
    clone_attachment_seed: int = 578
    resume_policy: str = "forbid"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_fraction, bool)
            or not isinstance(self.sample_fraction, int | float)
            or float(self.sample_fraction) not in _STANDARD_SAMPLE_FRACTIONS
        ):
            raise F1CertificationError("request/sample_fraction: unsupported rung")
        _nonnegative_int(self.seed, location="request/seed")
        if self.clone_attachment_fraction != 1.0:
            raise F1CertificationError(
                "request/clone_attachment_fraction: certification requires 1.0"
            )
        _nonnegative_int(
            self.clone_attachment_seed,
            location="request/clone_attachment_seed",
        )
        if self.clone_attachment_seed != self.seed:
            raise F1CertificationError(
                "request/clone_attachment_seed must equal request/seed"
            )
        if self.resume_policy != "forbid":
            raise F1CertificationError(
                "request/resume_policy: certification requires forbid"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> F1RunRequest:
        row = _object(value, location="request")
        _exact_keys(
            row,
            frozenset(
                {
                    "sample_fraction",
                    "seed",
                    "clone_attachment_fraction",
                    "clone_attachment_seed",
                    "resume_policy",
                }
            ),
            location="request",
        )
        return cls(
            sample_fraction=row["sample_fraction"],  # type: ignore[arg-type]
            seed=row["seed"],  # type: ignore[arg-type]
            clone_attachment_fraction=row["clone_attachment_fraction"],  # type: ignore[arg-type]
            clone_attachment_seed=row["clone_attachment_seed"],  # type: ignore[arg-type]
            resume_policy=row["resume_policy"],  # type: ignore[arg-type]
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "sample_fraction": float(self.sample_fraction),
            "seed": self.seed,
            "clone_attachment_fraction": float(self.clone_attachment_fraction),
            "clone_attachment_seed": self.clone_attachment_seed,
            "resume_policy": self.resume_policy,
        }


@dataclass(frozen=True, slots=True)
class F1ColdBuildReceipt:
    """One independently cold constants or bundle build."""

    certification_run_id: str
    request: F1RunRequest
    production_evidence: F1ProductionEvidence
    resume_audit: F1ResumeAudit
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        request: F1RunRequest,
        production_evidence: F1ProductionEvidence,
        certification_run_id: str | None = None,
    ) -> F1ColdBuildReceipt:
        if not isinstance(request, F1RunRequest):
            raise F1CertificationError("typed F1 run request required")
        if not isinstance(production_evidence, F1ProductionEvidence):
            raise F1CertificationError("typed production evidence required")
        assert_request_matches_evidence(request, production_evidence)
        audit = resume_audit_from_evidence(production_evidence)
        publication_run_id = _publication_certification_run_id(production_evidence)
        resolved_run_id = certification_run_id or publication_run_id
        if _uuid(resolved_run_id, location="certification_run_id") != publication_run_id:
            raise F1CertificationError(
                "certification_run_id differs from publication_run_id"
            )
        candidate = cls(
            certification_run_id=resolved_run_id,
            request=request,
            production_evidence=production_evidence,
            resume_audit=audit,
            receipt_sha256="0" * 64,
        )
        return cls(
            certification_run_id=candidate.certification_run_id,
            request=request,
            production_evidence=production_evidence,
            resume_audit=audit,
            receipt_sha256=sha256_json(candidate.body_wire()),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> F1ColdBuildReceipt:
        row = _object(value, location="cold_build_receipt")
        _exact_keys(
            row,
            frozenset(
                {
                    "artifact_kind",
                    "schema_version",
                    "certification_run_id",
                    "mode",
                    "request",
                    "cold_build",
                    "plan_lock",
                    "plan_lock_sha256",
                    "artifacts",
                    "normative_vector_sha256",
                    "receipt_surfaces",
                    "receipt_surfaces_sha256",
                    "run_provenance_identity",
                    "node_reuse_keys",
                    "node_reuse_keys_sha256",
                    "coverage",
                    "resume_audit",
                    "production_evidence_sha256",
                    "receipt_sha256",
                }
            ),
            location="cold_build_receipt",
        )
        if row["artifact_kind"] != _COLD_RECEIPT_KIND:
            raise F1CertificationError("cold_build_receipt/artifact_kind: mismatch")
        if row["schema_version"] != 1:
            raise F1CertificationError("cold_build_receipt/schema_version: unsupported")
        cold = _object(row["cold_build"], location="cold_build_receipt/cold_build")
        _exact_keys(
            cold,
            frozenset(
                {
                    "resume_policy",
                    "output_root_created_exclusively",
                    "pool_exit_code",
                }
            ),
            location="cold_build_receipt/cold_build",
        )
        if cold != {
            "resume_policy": "forbid",
            "output_root_created_exclusively": True,
            "pool_exit_code": 0,
        }:
            raise F1CertificationError("cold build execution predicate is not sealed")
        evidence_wire = {
            "artifact_kind": _PRODUCTION_EVIDENCE_KIND,
            "schema_version": 1,
            "mode": row["mode"],
            "plan_lock": row["plan_lock"],
            "artifacts": row["artifacts"],
            "receipt_surfaces": row["receipt_surfaces"],
            "run_provenance_identity": row["run_provenance_identity"],
            "node_reuse_keys": row["node_reuse_keys"],
            "coverage": row["coverage"],
            "evidence_sha256": row["production_evidence_sha256"],
        }
        evidence = F1ProductionEvidence.from_mapping(evidence_wire)
        request = F1RunRequest.from_mapping(row["request"])
        audit = F1ResumeAudit.from_mapping(row["resume_audit"])
        assert_request_matches_evidence(request, evidence)
        expected_audit = resume_audit_from_evidence(evidence)
        if audit.to_wire() != expected_audit.to_wire():
            raise F1CertificationError(
                "resume_audit differs from the publication receipt predicate"
            )
        candidate = cls(
            certification_run_id=_uuid(
                row["certification_run_id"], location="certification_run_id"
            ),
            request=request,
            production_evidence=evidence,
            resume_audit=audit,
            receipt_sha256=_sha256(
                row["receipt_sha256"], location="cold_build_receipt/receipt_sha256"
            ),
        )
        if candidate.certification_run_id != _publication_certification_run_id(evidence):
            raise F1CertificationError(
                "certification_run_id differs from publication_run_id"
            )
        _validate_aggregate_digests(candidate, row)
        if candidate.receipt_sha256 != sha256_json(candidate.body_wire()):
            raise F1CertificationError("cold build receipt seal mismatch")
        return candidate

    @property
    def mode(self) -> str:
        return self.production_evidence.mode

    def body_wire(self) -> dict[str, object]:
        evidence = self.production_evidence
        return {
            "artifact_kind": _COLD_RECEIPT_KIND,
            "schema_version": 1,
            "certification_run_id": self.certification_run_id,
            "mode": self.mode,
            "request": self.request.to_wire(),
            "cold_build": {
                "resume_policy": "forbid",
                "output_root_created_exclusively": True,
                "pool_exit_code": 0,
            },
            "plan_lock": dict(evidence.plan_lock),
            "plan_lock_sha256": _plan_lock_sha256(evidence.plan_lock),
            "artifacts": [row.to_wire() for row in evidence.artifacts],
            "normative_vector_sha256": _normative_vector_sha256(evidence.artifacts),
            "receipt_surfaces": dict(evidence.receipt_surfaces),
            "receipt_surfaces_sha256": _receipt_surfaces_sha256(
                evidence.receipt_surfaces
            ),
            "run_provenance_identity": evidence.run_provenance_identity.to_wire(),
            "node_reuse_keys": dict(evidence.node_reuse_keys),
            "node_reuse_keys_sha256": _node_reuse_keys_sha256(evidence.node_reuse_keys),
            "coverage": evidence.coverage.to_wire(),
            "resume_audit": self.resume_audit.to_wire(),
            "production_evidence_sha256": evidence.evidence_sha256,
        }

    def to_wire(self) -> dict[str, object]:
        return {**self.body_wire(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class F1CertificationVerdict:
    body: Mapping[str, object]
    receipt_sha256: str

    def to_wire(self) -> dict[str, object]:
        return {**dict(self.body), "receipt_sha256": self.receipt_sha256}


def complete_coverage_evidence(
    plan_lock: Mapping[str, object],
    *,
    bound_locator_refs: Sequence[str],
    node_reuse_ids: Sequence[str],
    node_reuse_inventory_complete: bool,
    selector_inventory_complete: bool,
    calibration_scope_complete: bool,
    selector_coverage_receipt: Mapping[str, object] | None = None,
    calibration_scope_receipt: Mapping[str, object] | None = None,
) -> F1CoverageEvidence:
    """Build the production coverage row without inventing plan inventories."""

    normalized_plan = _validated_plan_lock(plan_lock)
    execution_abi = _execution_abi(normalized_plan)
    artifacts = _plan_artifact_rows(execution_abi)
    if selector_coverage_receipt is None:
        raise F1CertificationError(
            "production selector coverage requires a sealed typed receipt"
        )
    calibration_receipt = _normalize_calibration_coverage_receipt(
        normalized_plan,
        complete=calibration_scope_complete,
        receipt=calibration_scope_receipt,
    )
    return F1CoverageEvidence.create(
        plan_artifact_ids=tuple(str(row["id"]) for row in artifacts),
        bound_locator_refs=tuple(sorted(set(bound_locator_refs))),
        node_reuse_ids=tuple(node_reuse_ids),
        node_reuse_inventory_complete=node_reuse_inventory_complete,
        selector_coverage_receipt=selector_coverage_receipt,
        selector_inventory_complete=selector_inventory_complete,
        calibration_scope_receipt=calibration_receipt,
        calibration_scope_complete=calibration_scope_complete,
    )


def assert_f1_selector_coverage_contract_current(
    coverage: F1CoverageEvidence,
    expected_contract: object,
) -> None:
    """Require production selector evidence to use one freshly compiled contract."""

    if not isinstance(coverage, F1CoverageEvidence):
        raise F1CertificationError("typed F1 coverage evidence required")
    receipt = _object(
        coverage.selector_coverage_receipt,
        location="coverage/selector_coverage_receipt",
    )
    if receipt.get("domain") != _POOL_COVERAGE_DOMAIN:
        raise F1CertificationError(
            "current selector contract check requires production pool coverage"
        )
    contract = _object(
        receipt.get("contract"),
        location="coverage/selector_coverage_receipt/contract",
    )
    candidate = expected_contract
    to_wire = getattr(candidate, "to_wire", None)
    if callable(to_wire):
        candidate = to_wire()
    expected = _object(
        candidate,
        location="expected_selector_coverage_contract",
    )
    _validate_pool_coverage_contract(expected)
    if contract != expected:
        raise F1CertificationError(
            "selector coverage contract differs from freshly compiled authority"
        )


def emit_f1_production_evidence(
    path: str | Path,
    *,
    mode: str,
    plan_lock: Mapping[str, object],
    artifacts: Mapping[str, bytes | ArtifactDigest],
    receipt_surfaces: Mapping[str, object],
    run_provenance_identity: RunProvenanceIdentity,
    node_reuse_keys: Mapping[str, str],
    coverage: F1CoverageEvidence,
) -> F1ProductionEvidence:
    """Construct and atomically emit the sole pool-to-runner evidence wire."""

    evidence = F1ProductionEvidence.create(
        mode=mode,
        plan_lock=plan_lock,
        artifacts=artifacts,
        receipt_surfaces=receipt_surfaces,
        run_provenance_identity=run_provenance_identity,
        node_reuse_keys=node_reuse_keys,
        coverage=coverage,
    )
    atomic_write_json(path, evidence.to_wire())
    return evidence


def load_f1_production_evidence(path: str | Path) -> F1ProductionEvidence:
    return F1ProductionEvidence.from_mapping(_load_canonical_json(path))


def emit_f1_cold_build_receipt(
    path: str | Path,
    *,
    request: F1RunRequest,
    production_evidence: F1ProductionEvidence,
) -> F1ColdBuildReceipt:
    receipt = F1ColdBuildReceipt.create(
        request=request,
        production_evidence=production_evidence,
    )
    atomic_write_json(path, receipt.to_wire())
    return receipt


def assert_request_matches_evidence(
    request: F1RunRequest,
    evidence: F1ProductionEvidence,
) -> None:
    """Bind runner arguments to independently collected publication evidence."""

    publication = _object(
        evidence.receipt_surfaces.get("publication_manifest"),
        location="receipt_surfaces/publication_manifest",
    )
    sampling = _object(
        publication.get("sampling"),
        location="publication_manifest/sampling",
    )
    clone = _object(
        publication.get("clone_attachment"),
        location="publication_manifest/clone_attachment",
    )
    expected = {
        "sample_fraction": float(request.sample_fraction),
        "sample_seed": request.seed,
        "clone_attachment_fraction": float(request.clone_attachment_fraction),
        "clone_attachment_seed": request.clone_attachment_seed,
    }
    observed = {
        "sample_fraction": sampling.get("sample_fraction"),
        "sample_seed": sampling.get("sample_seed"),
        "clone_attachment_fraction": clone.get("fraction"),
        "clone_attachment_seed": clone.get("seed"),
    }
    if observed != expected:
        raise F1CertificationError(
            "runner request differs from publication evidence: "
            f"expected={expected}, observed={observed}"
        )
    provenance_wire = evidence.run_provenance_identity.to_wire()
    provenance_request = _object(
        provenance_wire.get("run_request"),
        location="run_provenance_identity/run_request",
    )
    provenance_observed = {
        "sample_fraction": provenance_request.get("sample_fraction"),
        "sample_seed": provenance_request.get("sample_seed"),
        "clone_attachment_fraction": provenance_request.get(
            "clone_attachment_fraction"
        ),
        "clone_attachment_seed": provenance_request.get("clone_attachment_seed"),
    }
    if provenance_observed != expected:
        raise F1CertificationError(
            "runner request differs from typed run provenance: "
            f"expected={expected}, observed={provenance_observed}"
        )


def load_f1_cold_build_receipt(path: str | Path) -> F1ColdBuildReceipt:
    return F1ColdBuildReceipt.from_mapping(_load_canonical_json(path))


def resume_audit_from_evidence(evidence: F1ProductionEvidence) -> F1ResumeAudit:
    publication = _object(
        evidence.receipt_surfaces.get("publication_manifest"),
        location="receipt_surfaces/publication_manifest",
    )
    stage_checkpoints = _object(
        publication.get("stage_checkpoints"),
        location="publication_manifest/stage_checkpoints",
    )
    deepest = stage_checkpoints.get("deepest_resumed_stage")
    if deepest is not None:
        _nonempty_string(
            deepest,
            location="publication_manifest/stage_checkpoints/deepest_resumed_stage",
        )
    stage_values = _object(
        stage_checkpoints.get("stages"),
        location="publication_manifest/stage_checkpoints/stages",
    )
    durable = _array(
        _execution_abi(evidence.plan_lock).get("durable_checkpoints"),
        location="execution_abi/durable_checkpoints",
    )
    stage_order = tuple(
        _nonempty_string(
            _object(row, location="durable_checkpoint").get("id"),
            location="durable_checkpoint/id",
        )
        for row in durable
    )
    if set(stage_values) != set(stage_order):
        raise F1CertificationError(
            "publication stage checkpoint inventory differs from execution ABI"
        )
    stages: list[dict[str, object]] = []
    for stage_ref in stage_order:
        stage = _object(
            stage_values[stage_ref],
            location=f"publication_manifest/stage_checkpoints/stages/{stage_ref}",
        )
        resumed_count = int(
            stage.get("source") == "checkpoint" or stage.get("load_status") == "resumed"
        )
        stages.append({"stage_ref": stage_ref, "resumed_count": resumed_count})

    stage_receipts = _object(
        publication.get("stage_receipts"),
        location="publication_manifest/stage_receipts",
    )
    impute = _object(
        stage_receipts.get("impute"),
        location="publication_manifest/stage_receipts/impute",
    )
    primary = _object(
        impute.get("primary_puf_qrf"),
        location="publication_manifest/stage_receipts/impute/primary_puf_qrf",
    )
    primary_status = _nonempty_string(
        primary.get("resume_status"),
        location=(
            "publication_manifest/stage_receipts/impute/primary_puf_qrf/resume_status"
        ),
    )
    primary_row = {
        "resume_status": primary_status,
        "resumed_count": int(primary_status == "resumed"),
    }
    acs_transfer = _object(
        impute.get("acs_qrf_transfer"),
        location="publication_manifest/stage_receipts/impute/acs_qrf_transfer",
    )
    target_bank = _object(
        acs_transfer.get("target_bank"),
        location=(
            "publication_manifest/stage_receipts/impute/acs_qrf_transfer/target_bank"
        ),
    )
    bank_rows: list[dict[str, object]] = []
    for family in ("directions", "late_producer_groups"):
        family_banks = _object(
            target_bank.get(family),
            location=f"publication_manifest/target_bank/{family}",
        )
        for name in sorted(family_banks):
            bank = _object(
                family_banks[name],
                location=f"publication_manifest/target_bank/{family}/{name}",
            )
            targets = _object(
                bank.get("targets"),
                location=f"publication_manifest/target_bank/{family}/{name}/targets",
            )
            target_rows: list[dict[str, object]] = []
            for target in sorted(targets):
                target_receipt = _object(
                    targets[target],
                    location=(
                        f"publication_manifest/target_bank/{family}/{name}/"
                        f"targets/{target}"
                    ),
                )
                target_rows.append(
                    {
                        "target_ref": target,
                        "load_status_resumed_count": int(
                            target_receipt.get("load_status") == "resumed"
                        ),
                        "source_checkpoint_count": int(
                            target_receipt.get("source") == "checkpoint"
                        ),
                    }
                )
            bank_rows.append(
                {
                    "bank_ref": f"{family}/{name}",
                    "target_count": len(target_rows),
                    "targets": target_rows,
                    "load_status_resumed_count": sum(
                        int(row["load_status_resumed_count"]) for row in target_rows
                    ),
                    "source_checkpoint_count": sum(
                        int(row["source_checkpoint_count"]) for row in target_rows
                    ),
                }
            )
    total = (
        sum(int(row["resumed_count"]) for row in stages)
        + int(primary_row["resumed_count"])
        + sum(
            int(row["load_status_resumed_count"]) + int(row["source_checkpoint_count"])
            for row in bank_rows
        )
    )
    wire = {
        "deepest_resumed_stage": deepest,
        "stages": stages,
        "primary_qrf": primary_row,
        "target_banks": bank_rows,
        "total_resume_count": total,
        "passed": deepest is None and primary_status == "initialized" and total == 0,
    }
    return F1ResumeAudit.from_mapping(wire)


def compare_f1_cold_build_receipts(
    *,
    constants_a: F1ColdBuildReceipt,
    constants_b: F1ColdBuildReceipt,
    bundle_a: F1ColdBuildReceipt,
    bundle_b: F1ColdBuildReceipt,
) -> F1CertificationVerdict:
    labeled = (
        ("constants_a", constants_a, "constants"),
        ("constants_b", constants_b, "constants"),
        ("bundle_a", bundle_a, "bundle"),
        ("bundle_b", bundle_b, "bundle"),
    )
    for label, receipt, expected_mode in labeled:
        if not isinstance(receipt, F1ColdBuildReceipt):
            raise F1CertificationError(f"{label}: typed cold receipt required")
        if receipt.mode != expected_mode:
            raise F1CertificationError(
                f"{label}: expected mode {expected_mode}, got {receipt.mode}"
            )
    run_ids = [receipt.certification_run_id for _, receipt, _ in labeled]
    if len(set(run_ids)) != 4:
        raise F1CertificationError("four distinct certification run ids required")
    plan_digests = {
        _plan_lock_sha256(receipt.production_evidence.plan_lock)
        for _, receipt, _ in labeled
    }
    if len(plan_digests) != 1:
        raise F1CertificationError("four receipts do not share one plan lock")
    requests = {
        canonical_json_bytes(receipt.request.to_wire()) for _, receipt, _ in labeled
    }
    if len(requests) != 1:
        raise F1CertificationError("four receipts do not share one build request")

    coverage_rows: dict[str, dict[str, object]] = {}
    for label, receipt, _ in labeled:
        coverage = receipt.production_evidence.coverage
        coverage_rows[label] = {
            "node_reuse_inventory_complete": (coverage.node_reuse_inventory_complete),
            "selector_inventory_complete": coverage.selector_inventory_complete,
            "selector_coverage_receipt": dict(coverage.selector_coverage_receipt),
            "selector_coverage_receipt_sha256": (
                coverage.selector_coverage_receipt_sha256
            ),
            "calibration_scope_complete": coverage.calibration_scope_complete,
            "calibration_scope_receipt": dict(coverage.calibration_scope_receipt),
            "calibration_scope_receipt_sha256": (
                coverage.calibration_scope_receipt_sha256
            ),
            "complete": coverage.complete,
            "passed": coverage.passes(receipt.production_evidence.plan_lock),
        }
    vector_coverage = {
        "receipts": coverage_rows,
        "passed": all(bool(row["passed"]) for row in coverage_rows.values()),
    }
    cold_rows = {
        label: {
            "passed": receipt.resume_audit.passed,
            "resume_audit": receipt.resume_audit.to_wire(),
        }
        for label, receipt, _ in labeled
    }
    cold_builds = {
        "receipts": cold_rows,
        "passed": all(bool(row["passed"]) for row in cold_rows.values()),
    }
    constants_within = _within_mode_comparison(constants_a, constants_b)
    bundle_within = _within_mode_comparison(bundle_a, bundle_b)
    within_mode = {
        "constants": constants_within,
        "bundle": bundle_within,
        "passed": constants_within["passed"] and bundle_within["passed"],
    }
    cross_ac = _cross_mode_comparison(constants_a, bundle_a, pair="A:C")
    cross_bd = _cross_mode_comparison(constants_b, bundle_b, pair="B:D")
    cross_mode = {
        "pairs": [cross_ac, cross_bd],
        "passed": cross_ac["passed"] and cross_bd["passed"],
    }
    differences: list[dict[str, object]] = []
    if not vector_coverage["passed"]:
        differences.append(
            {
                "kind": "vector_coverage",
                "subject": "production evidence closure",
            }
        )
    if not cold_builds["passed"]:
        differences.append({"kind": "cold_resume_audit", "subject": "four cold builds"})
    for comparison in (constants_within, bundle_within):
        differences.extend(comparison["differences"])
    for comparison in (cross_ac, cross_bd):
        if not comparison["passed"]:
            differences.append(
                {
                    "kind": "cross_mode_equality",
                    "subject": comparison["pair"],
                }
            )
    passed = (
        bool(vector_coverage["passed"])
        and bool(cold_builds["passed"])
        and bool(within_mode["passed"])
        and bool(cross_mode["passed"])
    )
    body = {
        "artifact_kind": _CERTIFICATION_KIND,
        "schema_version": 1,
        "plan_lock_sha256": next(iter(plan_digests)),
        "request": constants_a.request.to_wire(),
        "inputs": [
            {
                "role": label,
                "mode": receipt.mode,
                "certification_run_id": receipt.certification_run_id,
                "receipt_sha256": receipt.receipt_sha256,
            }
            for label, receipt, _ in labeled
        ],
        "vector_coverage": vector_coverage,
        "cold_builds": cold_builds,
        "within_mode_determinism": within_mode,
        "cross_mode_equality": cross_mode,
        "resume_gate": {
            "status": "documented_not_run",
            "part_of_four_cold_build_verdict": False,
        },
        "passed": passed,
        "differences": differences,
    }
    return F1CertificationVerdict(body=body, receipt_sha256=sha256_json(body))


def certification_markdown(verdict: F1CertificationVerdict) -> str:
    wire = verdict.to_wire()
    within = _object(
        wire["within_mode_determinism"], location="within_mode_determinism"
    )
    cross = _object(wire["cross_mode_equality"], location="cross_mode_equality")
    coverage = _object(wire["vector_coverage"], location="vector_coverage")
    cold = _object(wire["cold_builds"], location="cold_builds")
    passed = bool(wire["passed"])
    lines = [
        "# US F1 certification verdict",
        "",
        f"Overall verdict: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Gate | Verdict |",
        "|---|---|",
        f"| Vector coverage | {_pass_fail(bool(coverage['passed']))} |",
        f"| Four cold-build audits | {_pass_fail(bool(cold['passed']))} |",
        (
            "| Constants within-mode determinism | "
            f"{_pass_fail(bool(_object(within['constants'], location='constants')['passed']))} |"
        ),
        (
            "| Bundle within-mode determinism | "
            f"{_pass_fail(bool(_object(within['bundle'], location='bundle')['passed']))} |"
        ),
        f"| Cross-mode equality | {_pass_fail(bool(cross['passed']))} |",
        "| Kill/resume gate | DOCUMENTED / NOT RUN |",
        "",
        f"Receipt SHA-256: `{verdict.receipt_sha256}`",
        "",
    ]
    coverage_receipts = _object(
        coverage["receipts"], location="vector_coverage/receipts"
    )
    lines.extend(
        [
            "## Vector coverage evidence",
            "",
            "| Build | Node reuse inventory | Selector/member inventory | Calibration scope | Complete | Plan-bound verdict |",
            "|---|---|---|---|---|---|",
        ]
    )
    for label in ("constants_a", "constants_b", "bundle_a", "bundle_b"):
        row = _object(
            coverage_receipts[label],
            location=f"vector_coverage/receipts/{label}",
        )
        lines.append(
            f"| `{label}` | {_pass_fail(bool(row['node_reuse_inventory_complete']))} "
            f"| {_pass_fail(bool(row['selector_inventory_complete']))} "
            f"| {_pass_fail(bool(row['calibration_scope_complete']))} "
            f"| {_pass_fail(bool(row['complete']))} "
            f"| {_pass_fail(bool(row['passed']))} |"
        )
    lines.append("")
    differences = _array(wire["differences"], location="differences")
    if differences:
        lines.extend(["## Differences", ""])
        for row in differences:
            value = _object(row, location="difference")
            lines.append(f"- `{value.get('kind')}`: {value.get('subject')}")
        lines.append("")
    return "\n".join(lines)


def atomic_write_json(path: str | Path, value: Mapping[str, object]) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        stream = temporary.open("xb")
    except FileExistsError as error:
        raise F1CertificationError(
            f"atomic temporary path already exists: {temporary}"
        ) from error
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _within_mode_comparison(
    left: F1ColdBuildReceipt,
    right: F1ColdBuildReceipt,
) -> dict[str, object]:
    if left.mode != right.mode:  # pragma: no cover - caller fixes roles
        raise F1CertificationError("within-mode pair differs in authority mode")
    execution_abi = _execution_abi(left.production_evidence.plan_lock)
    left_artifacts = _artifact_digest_map(left.production_evidence.artifacts)
    right_artifacts = _artifact_digest_map(right.production_evidence.artifacts)
    artifact_rows = [
        {
            "artifact_id": artifact_id,
            "left": left_artifacts[artifact_id].to_wire(),
            "right": right_artifacts[artifact_id].to_wire(),
            "equal": left_artifacts[artifact_id] == right_artifacts[artifact_id],
        }
        for artifact_id in left_artifacts
    ]
    left_projection = receipt_determinism_projection(
        execution_abi,
        authority_mode=left.mode,
        receipts=left.production_evidence.receipt_surfaces,
        run_provenance_identity=left.production_evidence.run_provenance_identity,
    )
    right_projection = receipt_determinism_projection(
        execution_abi,
        authority_mode=right.mode,
        receipts=right.production_evidence.receipt_surfaces,
        run_provenance_identity=right.production_evidence.run_provenance_identity,
    )
    receipts_equal = canonical_json_bytes(left_projection) == canonical_json_bytes(
        right_projection
    )
    node_complete = (
        left.production_evidence.coverage.node_reuse_inventory_complete
        and right.production_evidence.coverage.node_reuse_inventory_complete
    )
    node_equal = node_complete and (
        left.production_evidence.node_reuse_keys
        == right.production_evidence.node_reuse_keys
    )
    differences: list[dict[str, object]] = [
        {
            "kind": "within_mode_artifact",
            "subject": f"{left.mode}:{row['artifact_id']}",
        }
        for row in artifact_rows
        if not row["equal"]
    ]
    if not receipts_equal:
        differences.append({"kind": "within_mode_receipts", "subject": left.mode})
    if not node_complete:
        differences.append(
            {"kind": "within_mode_node_reuse_inventory", "subject": left.mode}
        )
    elif not node_equal:
        differences.append(
            {"kind": "within_mode_node_reuse_keys", "subject": left.mode}
        )
    artifacts_equal = all(bool(row["equal"]) for row in artifact_rows)
    return {
        "mode": left.mode,
        "left_run_id": left.certification_run_id,
        "right_run_id": right.certification_run_id,
        "artifact_rows": artifact_rows,
        "normative_equal": artifacts_equal,
        "receipt_projection_left_sha256": sha256_json(left_projection),
        "receipt_projection_right_sha256": sha256_json(right_projection),
        "receipts_equal_under_plan": receipts_equal,
        "node_reuse_inventory_complete": node_complete,
        "node_reuse_keys_equal": node_equal,
        "passed": artifacts_equal and receipts_equal and node_equal,
        "differences": differences,
    }


def _cross_mode_comparison(
    constants: F1ColdBuildReceipt,
    bundle: F1ColdBuildReceipt,
    *,
    pair: str,
) -> dict[str, object]:
    if not (
        constants.production_evidence.coverage.node_reuse_inventory_complete
        and bundle.production_evidence.coverage.node_reuse_inventory_complete
    ):
        return {
            "pair": pair,
            "passed": False,
            "comparison_error": "node reuse inventory is incomplete",
        }
    try:
        comparison = compare_artifact_digest_sets(
            _execution_abi(constants.production_evidence.plan_lock),
            constants_artifacts=_artifact_digest_map(
                constants.production_evidence.artifacts
            ),
            bundle_artifacts=_artifact_digest_map(bundle.production_evidence.artifacts),
            constants_receipts=constants.production_evidence.receipt_surfaces,
            bundle_receipts=bundle.production_evidence.receipt_surfaces,
            constants_run_provenance_identity=(
                constants.production_evidence.run_provenance_identity
            ),
            bundle_run_provenance_identity=(
                bundle.production_evidence.run_provenance_identity
            ),
            constants_node_reuse_keys=constants.production_evidence.node_reuse_keys,
            bundle_node_reuse_keys=bundle.production_evidence.node_reuse_keys,
        )
    except ArtifactComparisonError as error:
        return {"pair": pair, "passed": False, "comparison_error": str(error)}
    return {"pair": pair, "passed": comparison.passed, "receipt": comparison.to_wire()}


def _artifact_rows_from_surfaces(
    plan_lock: Mapping[str, object],
    surfaces: Mapping[str, bytes | ArtifactDigest],
) -> tuple[F1ArtifactDigestRow, ...]:
    if not isinstance(surfaces, Mapping):
        raise F1CertificationError("artifacts: mapping required")
    plan_rows = _plan_artifact_rows(_execution_abi(plan_lock))
    expected_ids = {str(row["id"]) for row in plan_rows}
    if set(surfaces) != expected_ids:
        raise F1CertificationError(
            "artifact surface inventory differs from plan: "
            f"missing={sorted(expected_ids - set(surfaces))}, "
            f"extra={sorted(set(surfaces) - expected_ids)}"
        )
    rows: list[F1ArtifactDigestRow] = []
    for plan_row in plan_rows:
        artifact_id = str(plan_row["id"])
        value = surfaces[artifact_id]
        digest = (
            value
            if isinstance(value, ArtifactDigest)
            else ArtifactDigest.from_bytes(value)
        )
        rows.append(F1ArtifactDigestRow.from_mapping({**plan_row, **digest.to_wire()}))
    return tuple(rows)


def _validate_artifact_rows(
    plan_lock: Mapping[str, object],
    rows: Sequence[F1ArtifactDigestRow],
) -> None:
    expected = _plan_artifact_rows(_execution_abi(plan_lock))
    if len(rows) != len(expected):
        raise F1CertificationError("artifact digest row count differs from plan")
    ids: list[str] = []
    for index, (observed, plan_row) in enumerate(zip(rows, expected, strict=True)):
        if not isinstance(observed, F1ArtifactDigestRow):
            raise F1CertificationError(f"artifacts/{index}: typed row required")
        if observed.contract_wire() != plan_row:
            raise F1CertificationError(
                f"artifacts/{index}: contract differs from plan artifact vector"
            )
        ids.append(observed.artifact_id)
    if len(ids) != len(set(ids)):
        raise F1CertificationError("artifact ids must be unique")


def _artifact_digest_map(
    rows: Sequence[F1ArtifactDigestRow],
) -> dict[str, ArtifactDigest]:
    return {row.artifact_id: row.digest for row in rows}


def _validate_aggregate_digests(
    receipt: F1ColdBuildReceipt,
    observed: Mapping[str, object],
) -> None:
    evidence = receipt.production_evidence
    expected = {
        "plan_lock_sha256": _plan_lock_sha256(evidence.plan_lock),
        "normative_vector_sha256": _normative_vector_sha256(evidence.artifacts),
        "receipt_surfaces_sha256": _receipt_surfaces_sha256(evidence.receipt_surfaces),
        "node_reuse_keys_sha256": _node_reuse_keys_sha256(evidence.node_reuse_keys),
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise F1CertificationError(f"cold_build_receipt/{key}: mismatch")


def _normative_vector_sha256(rows: Sequence[F1ArtifactDigestRow]) -> str:
    return sha256_json(
        {
            "domain": _NORMATIVE_VECTOR_DOMAIN,
            "artifacts": [
                {
                    "id": row.artifact_id,
                    "stage_ref": row.stage_ref,
                    "sha256": row.digest.sha256,
                    "byte_size": row.digest.byte_size,
                }
                for row in rows
            ],
        }
    )


def _receipt_surfaces_sha256(receipts: Mapping[str, object]) -> str:
    return sha256_json({"domain": _RECEIPT_SURFACE_DOMAIN, "receipts": dict(receipts)})


def _node_reuse_keys_sha256(values: Mapping[str, str]) -> str:
    return sha256_json(
        {"domain": _NODE_REUSE_KEYS_DOMAIN, "node_reuse_keys": dict(values)}
    )


def _plan_lock_sha256(plan_lock: Mapping[str, object]) -> str:
    return sha256_json(plan_lock)


def _execution_abi(plan_lock: Mapping[str, object]) -> dict[str, object]:
    plan = _object(plan_lock, location="plan_lock")
    return _object(plan.get("execution_abi"), location="plan_lock/execution_abi")


def _plan_artifact_rows(execution_abi: Mapping[str, object]) -> list[dict[str, object]]:
    rows = _array(
        execution_abi.get("normative_artifact_vector"),
        location="execution_abi/normative_artifact_vector",
    )
    result: list[dict[str, object]] = []
    for index, value in enumerate(rows):
        location = f"execution_abi/normative_artifact_vector/{index}"
        row = _object(value, location=location)
        _exact_keys(row, _PLAN_ARTIFACT_KEYS, location=location)
        result.append(row)
    return result


def _producer_order(execution_abi: Mapping[str, object]) -> tuple[str, ...]:
    pipeline = _object(execution_abi.get("pipeline"), location="execution_abi/pipeline")
    return _string_tuple(
        pipeline.get("producer_order"),
        location="execution_abi/pipeline/producer_order",
    )


def _run_provenance(value: object) -> RunProvenanceIdentity:
    row = _object(value, location="run_provenance_identity")
    _exact_keys(row, _PROVENANCE_KEYS, location="run_provenance_identity")
    try:
        return build_run_provenance_identity(
            identity_generation=row["identity_generation"],  # type: ignore[arg-type]
            source_grammar_receipt=row["source_grammar_receipt"],  # type: ignore[arg-type]
            spec_binding=row["spec_binding"],  # type: ignore[arg-type]
            authority_versions=row["authority_versions"],  # type: ignore[arg-type]
            code_inventory_digest=row["code_inventory_digest"],  # type: ignore[arg-type]
            artifact_protocol_inventory=row["artifact_protocol_inventory"],  # type: ignore[arg-type]
            run_request=row["run_request"],  # type: ignore[arg-type]
            execution_receipt=row["execution_receipt"],  # type: ignore[arg-type]
        )
    except (ExecutorError, TypeError) as error:
        raise F1CertificationError(
            f"run_provenance_identity is invalid: {error}"
        ) from error


def _node_reuse_keys(value: object) -> dict[str, str]:
    row = _object(value, location="node_reuse_keys")
    result: dict[str, str] = {}
    for key, child in row.items():
        _nonempty_string(key, location="node_reuse_keys/id")
        result[key] = _sha256(child, location=f"node_reuse_keys/{key}")
    return result


def _artifact_digest(sha: object, size: object, location: str) -> ArtifactDigest:
    try:
        return ArtifactDigest(
            sha256=_sha256(sha, location=f"{location}/sha256"),
            byte_size=_nonnegative_int(size, location=f"{location}/byte_size"),
        )
    except ArtifactComparisonError as error:  # pragma: no cover - prevalidated
        raise F1CertificationError(str(error)) from error


def _load_canonical_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise F1CertificationError(f"unable to read {source}: {error}") from error

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise F1CertificationError(f"{source}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise F1CertificationError(f"{source}: non-finite JSON number {value!r}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except F1CertificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F1CertificationError(f"{source}: invalid JSON: {error}") from error
    result = _object(value, location=str(source))
    if raw != canonical_json_bytes(result):
        raise F1CertificationError(f"{source}: noncanonical JSON evidence")
    return result


def _object(value: object, *, location: str) -> dict[str, object]:
    normalized = _json_value(value, location=location)
    if not isinstance(normalized, dict):
        raise F1CertificationError(f"{location}: object required")
    return normalized


def _array(value: object, *, location: str) -> list[object]:
    normalized = _json_value(value, location=location)
    if not isinstance(normalized, list):
        raise F1CertificationError(f"{location}: array required")
    return normalized


def _json_value(value: object, *, location: str) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise F1CertificationError(f"{location}: finite JSON number required")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise F1CertificationError(f"{location}: string keys required")
            result[key] = _json_value(child, location=f"{location}/{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _json_value(child, location=f"{location}/{index}")
            for index, child in enumerate(value)
        ]
    raise F1CertificationError(
        f"{location}: JSON value required, got {type(value).__name__}"
    )


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, location: str
) -> None:
    if set(value) != expected:
        raise F1CertificationError(
            f"{location}: keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _string_tuple(value: object, *, location: str) -> tuple[str, ...]:
    rows = _array(value, location=location)
    result = tuple(
        _nonempty_string(child, location=f"{location}/{index}")
        for index, child in enumerate(rows)
    )
    if len(result) != len(set(result)):
        raise F1CertificationError(f"{location}: unique values required")
    return result


def _validated_string_sequence(
    value: Sequence[str],
    *,
    location: str,
) -> tuple[str, ...]:
    return _string_tuple(value, location=location)


def _validated_plan_lock(value: object) -> dict[str, object]:
    plan = _object(value, location="plan_lock")
    payload = canonical_json_bytes(plan)
    try:
        _validate_plan_lock_schema_bytes(payload)
    except SpecValidationError as error:
        raise F1CertificationError(f"plan_lock is invalid: {error}") from error
    return plan


@lru_cache(maxsize=16)
def _validate_plan_lock_schema_bytes(payload: bytes) -> None:
    value = json.loads(payload)
    load_schema_registry().validate(value, PLAN_LOCK_SCHEMA_ID)


def _validate_selector_coverage_receipt(
    receipt: Mapping[str, object],
    *,
    expected: bool,
) -> None:
    location = "coverage/selector_coverage_receipt"
    domain = receipt.get("domain")
    if domain == _POOL_COVERAGE_DOMAIN:
        observed = _validate_pool_coverage_receipt(receipt)
    elif domain == _FIXTURE_SELECTOR_COVERAGE_DOMAIN:
        observed = _validate_fixture_coverage_receipt(
            receipt,
            domain=_FIXTURE_SELECTOR_COVERAGE_DOMAIN,
            verdict_key="container_member_coverage_complete",
            location=location,
        )
    else:
        raise F1CertificationError(f"{location}/domain: unsupported")
    if observed is not expected:
        raise F1CertificationError(
            f"{location}: embedded verdict differs from coverage summary"
        )


def _validate_calibration_coverage_receipt(
    receipt: Mapping[str, object],
    *,
    expected: bool,
) -> None:
    location = "coverage/calibration_scope_receipt"
    domain = receipt.get("domain")
    if domain == _CALIBRATION_COVERAGE_DOMAIN:
        _exact_keys(
            receipt,
            frozenset(
                {
                    "domain",
                    "schema_version",
                    "pipeline_id",
                    "spec_sha256",
                    "execution_abi_sha256",
                    "calibration_scope_complete",
                    "reason",
                    "receipt_sha256",
                }
            ),
            location=location,
        )
        if receipt["schema_version"] != 1:
            raise F1CertificationError(f"{location}/schema_version: unsupported")
        _nonempty_string(receipt["pipeline_id"], location=f"{location}/pipeline_id")
        _sha256(receipt["spec_sha256"], location=f"{location}/spec_sha256")
        _sha256(
            receipt["execution_abi_sha256"],
            location=f"{location}/execution_abi_sha256",
        )
        observed = receipt["calibration_scope_complete"]
        if not isinstance(observed, bool):
            raise F1CertificationError(
                f"{location}/calibration_scope_complete: boolean required"
            )
        if observed or receipt["reason"] != (
            "normative_artifact_vector_omits_calibration_weights"
        ):
            raise F1CertificationError(f"{location}/reason: mismatch")
        _validate_mapping_seal(receipt, seal_key="receipt_sha256", location=location)
    elif domain == _FIXTURE_CALIBRATION_COVERAGE_DOMAIN:
        observed = _validate_fixture_coverage_receipt(
            receipt,
            domain=_FIXTURE_CALIBRATION_COVERAGE_DOMAIN,
            verdict_key="calibration_scope_complete",
            location=location,
        )
    else:
        raise F1CertificationError(f"{location}/domain: unsupported")
    if observed is not expected:
        raise F1CertificationError(
            f"{location}: embedded verdict differs from coverage summary"
        )


def _validate_fixture_coverage_receipt(
    receipt: Mapping[str, object],
    *,
    domain: str,
    verdict_key: str,
    location: str,
) -> bool:
    _exact_keys(
        receipt,
        frozenset(
            {
                "domain",
                "schema_version",
                "pipeline_id",
                "spec_sha256",
                "execution_abi_sha256",
                verdict_key,
                "reason",
                "receipt_sha256",
            }
        ),
        location=location,
    )
    if receipt["domain"] != domain or receipt["schema_version"] != 1:
        raise F1CertificationError(f"{location}: fixture type mismatch")
    if receipt["pipeline_id"] != _FIXTURE_PIPELINE_ID:
        raise F1CertificationError(f"{location}/pipeline_id: fixture mismatch")
    _sha256(receipt["spec_sha256"], location=f"{location}/spec_sha256")
    _sha256(
        receipt["execution_abi_sha256"],
        location=f"{location}/execution_abi_sha256",
    )
    verdict = receipt[verdict_key]
    if not isinstance(verdict, bool):
        raise F1CertificationError(f"{location}/{verdict_key}: boolean required")
    if receipt["reason"] != "synthetic_comparator_fixture":
        raise F1CertificationError(f"{location}/reason: fixture mismatch")
    _validate_mapping_seal(receipt, seal_key="receipt_sha256", location=location)
    return verdict


def _validate_pool_coverage_receipt(receipt: Mapping[str, object]) -> bool:
    location = "coverage/selector_coverage_receipt"
    _exact_keys(
        receipt,
        frozenset(
            {
                "domain",
                "schema_version",
                "contract",
                "target_banks",
                "bank_member_coverage_complete",
                "final_pool_h5",
                "container_member_coverage_complete",
                "status",
                "receipt_sha256",
            }
        ),
        location=location,
    )
    if receipt["domain"] != _POOL_COVERAGE_DOMAIN or receipt["schema_version"] != 1:
        raise F1CertificationError(f"{location}: production type mismatch")
    contract = _object(receipt["contract"], location=f"{location}/contract")
    contracts = _validate_pool_coverage_contract(contract)
    final_h5 = _object(receipt["final_pool_h5"], location=f"{location}/final_pool_h5")
    if final_h5 != contract["final_pool_h5"]:
        raise F1CertificationError(
            f"{location}/final_pool_h5: differs from sealed contract"
        )
    results = _array(receipt["target_banks"], location=f"{location}/target_banks")
    if len(results) != len(contracts):
        raise F1CertificationError(
            f"{location}/target_banks: result count differs from contract"
        )
    complete_rows = tuple(
        _validate_pool_bank_result(
            value,
            contract=contract_row,
            location=f"{location}/target_banks/{index}",
        )
        for index, (value, contract_row) in enumerate(
            zip(results, contracts, strict=True)
        )
    )
    bank_complete = all(complete_rows)
    if receipt["bank_member_coverage_complete"] is not bank_complete:
        raise F1CertificationError(
            f"{location}/bank_member_coverage_complete: summary mismatch"
        )
    final_status = final_h5["status"]
    container_complete = bank_complete and final_status == "complete"
    if receipt["container_member_coverage_complete"] is not container_complete:
        raise F1CertificationError(
            f"{location}/container_member_coverage_complete: summary mismatch"
        )
    expected_status = (
        "unsupported"
        if final_status == "unsupported"
        else ("complete" if container_complete else "incomplete")
    )
    if receipt["status"] != expected_status:
        raise F1CertificationError(f"{location}/status: summary mismatch")
    _validate_mapping_seal(receipt, seal_key="receipt_sha256", location=location)
    return container_complete


def _validate_pool_coverage_contract(
    contract: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    location = "coverage/selector_coverage_receipt/contract"
    _exact_keys(
        contract,
        frozenset(
            {
                "domain",
                "schema_version",
                "authority_sha256",
                "spec_sha256",
                "execution_abi_sha256",
                "target_banks",
                "final_pool_h5",
                "sha256",
            }
        ),
        location=location,
    )
    if contract["domain"] != _POOL_COVERAGE_DOMAIN or contract["schema_version"] != 1:
        raise F1CertificationError(f"{location}: production type mismatch")
    for key in ("authority_sha256", "spec_sha256", "execution_abi_sha256"):
        _sha256(contract[key], location=f"{location}/{key}")
    values = _array(contract["target_banks"], location=f"{location}/target_banks")
    if not values:
        raise F1CertificationError(f"{location}/target_banks: non-empty required")
    banks = tuple(
        _validate_pool_bank_contract(
            value,
            location=f"{location}/target_banks/{index}",
        )
        for index, value in enumerate(values)
    )
    for key in ("artifact_id", "locator_ref"):
        observed = tuple(str(bank[key]) for bank in banks)
        if len(observed) != len(set(observed)):
            raise F1CertificationError(
                f"{location}/target_banks: unique {key} values required"
            )
    _validate_final_h5_coverage_contract(
        contract["final_pool_h5"],
        location=f"{location}/final_pool_h5",
    )
    _validate_mapping_seal(contract, seal_key="sha256", location=location)
    return banks


def _validate_pool_bank_contract(
    value: object,
    *,
    location: str,
) -> dict[str, object]:
    row = _object(value, location=location)
    _exact_keys(
        row,
        frozenset(
            {
                "artifact_id",
                "locator_ref",
                "authority_ref",
                "bank_kind",
                "expected_member_count",
                "expected_members_sha256",
                "expected_members",
            }
        ),
        location=location,
    )
    for key in ("artifact_id", "locator_ref", "authority_ref"):
        _nonempty_string(row[key], location=f"{location}/{key}")
    if row["bank_kind"] not in {"gap_fill", "primary_qrf", "late_transfer"}:
        raise F1CertificationError(f"{location}/bank_kind: unsupported")
    members = _coverage_member_rows(
        row["expected_members"],
        location=f"{location}/expected_members",
        nonempty=True,
    )
    count = _nonnegative_int(
        row["expected_member_count"],
        location=f"{location}/expected_member_count",
    )
    if count != len(members):
        raise F1CertificationError(f"{location}/expected_member_count: mismatch")
    if _sha256(
        row["expected_members_sha256"],
        location=f"{location}/expected_members_sha256",
    ) != sha256_json(members):
        raise F1CertificationError(f"{location}/expected_members_sha256: mismatch")
    return row


def _validate_pool_bank_result(
    value: object,
    *,
    contract: Mapping[str, object],
    location: str,
) -> bool:
    row = _object(value, location=location)
    _exact_keys(
        row,
        frozenset(
            {
                "artifact_id",
                "locator_ref",
                "bank_kind",
                "root_status",
                "expected_member_count",
                "expected_members_sha256",
                "observed_member_count",
                "observed_members_sha256",
                "missing_members",
                "extra_members",
                "status",
                "complete",
            }
        ),
        location=location,
    )
    for key in (
        "artifact_id",
        "locator_ref",
        "bank_kind",
        "expected_member_count",
        "expected_members_sha256",
    ):
        if row[key] != contract[key]:
            raise F1CertificationError(f"{location}/{key}: differs from contract")
    if row["root_status"] not in {"directory", "missing", "not_directory"}:
        raise F1CertificationError(f"{location}/root_status: unsupported")
    observed_count = _nonnegative_int(
        row["observed_member_count"],
        location=f"{location}/observed_member_count",
    )
    observed_sha = _sha256(
        row["observed_members_sha256"],
        location=f"{location}/observed_members_sha256",
    )
    expected_members = _coverage_member_rows(
        contract["expected_members"],
        location=f"{location}/contract_expected_members",
        nonempty=True,
    )
    missing = _coverage_member_rows(
        row["missing_members"],
        location=f"{location}/missing_members",
        nonempty=False,
    )
    extra = _coverage_member_rows(
        row["extra_members"],
        location=f"{location}/extra_members",
        nonempty=False,
    )
    expected_keys = {_coverage_member_key(member) for member in expected_members}
    missing_keys = {_coverage_member_key(member) for member in missing}
    extra_keys = {_coverage_member_key(member) for member in extra}
    if not missing_keys.issubset(expected_keys):
        raise F1CertificationError(f"{location}/missing_members: not in contract")
    if extra_keys & expected_keys or missing_keys & extra_keys:
        raise F1CertificationError(f"{location}/extra_members: contradict contract")
    if observed_count != len(expected_keys) - len(missing_keys) + len(extra_keys):
        raise F1CertificationError(f"{location}/observed_member_count: mismatch")
    complete = (
        row["root_status"] == "directory"
        and not missing
        and not extra
        and observed_count == len(expected_members)
        and observed_sha == contract["expected_members_sha256"]
    )
    expected_status = "complete" if complete else "incomplete"
    if row["status"] != expected_status or row["complete"] is not complete:
        raise F1CertificationError(f"{location}: typed result verdict mismatch")
    return complete


def _validate_final_h5_coverage_contract(value: object, *, location: str) -> None:
    row = _object(value, location=location)
    _exact_keys(
        row,
        frozenset(
            {
                "artifact_ids",
                "locator_ref",
                "selector_refs",
                "status",
                "unsupported_reason",
            }
        ),
        location=location,
    )
    _string_tuple(row["artifact_ids"], location=f"{location}/artifact_ids")
    _nonempty_string(row["locator_ref"], location=f"{location}/locator_ref")
    selectors = _string_tuple(
        row["selector_refs"], location=f"{location}/selector_refs"
    )
    if frozenset(selectors) != _H5_SELECTORS:
        raise F1CertificationError(f"{location}/selector_refs: inventory mismatch")
    if row["status"] != "unsupported" or row["unsupported_reason"] != (
        "compiler_authority_lacks_final_h5_entity_column_weight_inventory"
    ):
        raise F1CertificationError(f"{location}: unsupported verdict mismatch")


def _coverage_member_rows(
    value: object,
    *,
    location: str,
    nonempty: bool,
) -> list[dict[str, object]]:
    values = _array(value, location=location)
    if nonempty and not values:
        raise F1CertificationError(f"{location}: non-empty required")
    rows: list[dict[str, object]] = []
    for index, value_row in enumerate(values):
        child_location = f"{location}/{index}"
        row = _object(value_row, location=child_location)
        _exact_keys(
            row,
            frozenset({"relative_path", "kind"}),
            location=child_location,
        )
        relative = _nonempty_string(
            row["relative_path"], location=f"{child_location}/relative_path"
        )
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise F1CertificationError(
                f"{child_location}/relative_path: normalized confined path required"
            )
        if row["kind"] not in {"directory", "file"}:
            raise F1CertificationError(f"{child_location}/kind: unsupported")
        rows.append(row)
    keys = tuple(_coverage_member_key(row) for row in rows)
    if len(keys) != len(set(keys)):
        raise F1CertificationError(f"{location}: duplicate members forbidden")
    if keys != tuple(sorted(keys, key=lambda item: (item[0].encode("utf-8"), item[1]))):
        raise F1CertificationError(f"{location}: selector byte order required")
    return rows


def _coverage_member_key(value: Mapping[str, object]) -> tuple[str, str]:
    return (str(value["relative_path"]), str(value["kind"]))


def _validate_mapping_seal(
    value: Mapping[str, object],
    *,
    seal_key: str,
    location: str,
) -> None:
    observed = _sha256(value[seal_key], location=f"{location}/{seal_key}")
    body = {key: child for key, child in value.items() if key != seal_key}
    if observed != sha256_json(body):
        raise F1CertificationError(f"{location}/{seal_key}: mismatch")


def _normalize_calibration_coverage_receipt(
    plan_lock: Mapping[str, object],
    *,
    complete: bool,
    receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    if complete:
        raise F1CertificationError(
            "production calibration completion has no sealed inventory contract"
        )
    expected_reason = "normative_artifact_vector_omits_calibration_weights"
    if receipt is not None:
        source = _object(receipt, location="coverage/calibration_scope_receipt")
        if set(source) != {
            "domain",
            "schema_version",
            "calibration_scope_complete",
            "reason",
        }:
            return source
        expected_legacy = {
            "domain": _CALIBRATION_COVERAGE_DOMAIN,
            "schema_version": 1,
            "calibration_scope_complete": complete,
            "reason": expected_reason,
        }
        if source != expected_legacy:
            raise F1CertificationError(
                "coverage/calibration_scope_receipt: legacy producer row mismatch"
            )
    execution_abi = _execution_abi(plan_lock)
    pipeline = _object(execution_abi["pipeline"], location="execution_abi/pipeline")
    body = {
        "domain": _CALIBRATION_COVERAGE_DOMAIN,
        "schema_version": 1,
        "pipeline_id": _nonempty_string(
            pipeline["id"], location="execution_abi/pipeline/id"
        ),
        "spec_sha256": _plan_spec_sha256(plan_lock),
        "execution_abi_sha256": _sha256(
            execution_abi["sha256"], location="execution_abi/sha256"
        ),
        "calibration_scope_complete": complete,
        "reason": expected_reason,
    }
    return {**body, "receipt_sha256": sha256_json(body)}


def _validate_coverage_plan_links(
    coverage: F1CoverageEvidence,
    plan_lock: Mapping[str, object],
) -> None:
    execution_abi = _execution_abi(plan_lock)
    pipeline = _object(execution_abi["pipeline"], location="execution_abi/pipeline")
    pipeline_id = _nonempty_string(pipeline["id"], location="execution_abi/pipeline/id")
    spec_sha256 = _plan_spec_sha256(plan_lock)
    execution_sha256 = _sha256(execution_abi["sha256"], location="execution_abi/sha256")
    selector = _object(
        coverage.selector_coverage_receipt,
        location="coverage/selector_coverage_receipt",
    )
    selector_domain = selector["domain"]
    if selector_domain == _POOL_COVERAGE_DOMAIN:
        contract = _object(
            selector["contract"],
            location="coverage/selector_coverage_receipt/contract",
        )
        if contract["spec_sha256"] != spec_sha256:
            raise F1CertificationError(
                "selector coverage contract spec link differs from plan"
            )
        if contract["execution_abi_sha256"] != execution_sha256:
            raise F1CertificationError(
                "selector coverage contract execution link differs from plan"
            )
        _validate_pool_coverage_plan_inventory(selector, execution_abi)
    else:
        if pipeline_id != _FIXTURE_PIPELINE_ID:
            raise F1CertificationError(
                "synthetic selector coverage is restricted to the fixture pipeline"
            )
        _validate_fixture_plan_links(
            selector,
            spec_sha256=spec_sha256,
            execution_sha256=execution_sha256,
            location="coverage/selector_coverage_receipt",
        )
    calibration = _object(
        coverage.calibration_scope_receipt,
        location="coverage/calibration_scope_receipt",
    )
    if calibration["domain"] == _FIXTURE_CALIBRATION_COVERAGE_DOMAIN:
        if pipeline_id != _FIXTURE_PIPELINE_ID:
            raise F1CertificationError(
                "synthetic calibration coverage is restricted to the fixture pipeline"
            )
    elif calibration["pipeline_id"] != pipeline_id:
        raise F1CertificationError(
            "calibration coverage pipeline link differs from plan"
        )
    _validate_fixture_plan_links(
        calibration,
        spec_sha256=spec_sha256,
        execution_sha256=execution_sha256,
        location="coverage/calibration_scope_receipt",
    )


def _validate_fixture_plan_links(
    receipt: Mapping[str, object],
    *,
    spec_sha256: str,
    execution_sha256: str,
    location: str,
) -> None:
    if receipt["spec_sha256"] != spec_sha256:
        raise F1CertificationError(f"{location}: spec link differs from plan")
    if receipt["execution_abi_sha256"] != execution_sha256:
        raise F1CertificationError(f"{location}: execution link differs from plan")


def _validate_pool_coverage_plan_inventory(
    selector: Mapping[str, object],
    execution_abi: Mapping[str, object],
) -> None:
    contract = _object(
        selector["contract"],
        location="coverage/selector_coverage_receipt/contract",
    )
    contract_banks = _array(
        contract["target_banks"],
        location="coverage/selector_coverage_receipt/contract/target_banks",
    )
    artifacts = _plan_artifact_rows(execution_abi)
    directory_rows = tuple(
        row for row in artifacts if row["content_selector_ref"] == _DIRECTORY_SELECTOR
    )
    expected_banks = tuple((row["id"], row["locator_ref"]) for row in directory_rows)
    observed_banks = tuple(
        (
            _object(row, location="selector contract target bank")["artifact_id"],
            _object(row, location="selector contract target bank")["locator_ref"],
        )
        for row in contract_banks
    )
    if observed_banks != expected_banks:
        raise F1CertificationError(
            "selector coverage target banks differ from plan artifact vector"
        )
    final_h5 = _object(
        contract["final_pool_h5"],
        location="coverage/selector_coverage_receipt/contract/final_pool_h5",
    )
    h5_rows = tuple(
        row for row in artifacts if row["content_selector_ref"] in _H5_SELECTORS
    )
    if tuple(final_h5["artifact_ids"]) != tuple(row["id"] for row in h5_rows):
        raise F1CertificationError(
            "selector final-H5 artifacts differ from plan artifact vector"
        )
    if tuple(final_h5["selector_refs"]) != tuple(
        row["content_selector_ref"] for row in h5_rows
    ):
        raise F1CertificationError(
            "selector final-H5 selectors differ from plan artifact vector"
        )
    h5_locators = {row["locator_ref"] for row in h5_rows}
    if len(h5_locators) != 1 or final_h5["locator_ref"] not in h5_locators:
        raise F1CertificationError(
            "selector final-H5 locator differs from plan artifact vector"
        )


def _plan_spec_sha256(plan_lock: Mapping[str, object]) -> str:
    binding = _object(plan_lock.get("spec_binding"), location="plan_lock/spec_binding")
    return _sha256(
        binding.get("spec_sha256"), location="plan_lock/spec_binding/spec_sha256"
    )


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise F1CertificationError(f"{location}: non-empty string required")
    return value


def _nonnegative_int(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F1CertificationError(f"{location}: non-negative integer required")
    return value


def _sha256(value: object, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise F1CertificationError(f"{location}: lowercase SHA-256 required")
    return value


def _uuid(value: object, *, location: str) -> str:
    if not isinstance(value, str):
        raise F1CertificationError(f"{location}: UUID string required")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise F1CertificationError(f"{location}: UUID string required") from error
    if str(parsed) != value:
        raise F1CertificationError(f"{location}: canonical UUID required")
    return value


def _publication_certification_run_id(evidence: F1ProductionEvidence) -> str:
    publication = _object(
        evidence.receipt_surfaces.get("publication_manifest"),
        location="receipt_surfaces/publication_manifest",
    )
    value = publication.get("publication_run_id")
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise F1CertificationError(
            "receipt_surfaces/publication_manifest/publication_run_id: "
            "32-character lowercase UUID hex required"
        )
    return str(uuid.UUID(hex=value))


def _mode(value: object) -> str:
    if value not in {"constants", "bundle"}:
        raise F1CertificationError("mode must be 'constants' or 'bundle'")
    return str(value)


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CERTIFICATION_JSON_FILENAME",
    "CERTIFICATION_MARKDOWN_FILENAME",
    "COLD_BUILD_RECEIPT_FILENAME",
    "PRODUCTION_EVIDENCE_FILENAME",
    "F1ArtifactDigestRow",
    "F1CertificationError",
    "F1CertificationVerdict",
    "F1ColdBuildReceipt",
    "F1CoverageEvidence",
    "F1ProductionEvidence",
    "F1ResumeAudit",
    "F1RunRequest",
    "assert_request_matches_evidence",
    "assert_f1_selector_coverage_contract_current",
    "atomic_write_bytes",
    "atomic_write_json",
    "certification_markdown",
    "compare_f1_cold_build_receipts",
    "complete_coverage_evidence",
    "emit_f1_cold_build_receipt",
    "emit_f1_production_evidence",
    "load_f1_cold_build_receipt",
    "load_f1_production_evidence",
    "resume_audit_from_evidence",
]
