"""Plan-lock-driven comparison of dual-mode artifacts and receipts.

The compiler-issued execution ABI is the sole inventory and comparison
authority.  Normative artifacts are compared as raw bytes.  Receipt values
are structurally exact except at paths named by one, and only one, sealed
comparison rule.

This module accepts already materialized artifact bytes or typed digests of
those selected bytes.  File and directory discovery belongs to the execution
ABI's locator and selector implementation, not to the comparison kernel.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .artifact_selector_contract import (
    ARTIFACT_LOCATOR_GRAMMAR,
    ARTIFACT_SELECTOR_CONTRACT_SHA256,
    normalize_release_id,
)
from .canonical import canonical_json_bytes, sha256_json
from .compiler_ir import current_compiler_ir_abi
from .executor import RunProvenanceIdentity

_COMPARISON_DOMAIN = "microcosm.spec-engine.artifact-comparison.v1"
_NORMATIVE_VECTOR_DOMAIN = "microcosm.spec-engine.normative-artifact-vector-digest.v1"
_STAGE_VECTOR_DOMAIN = "microcosm.spec-engine.stage-artifact-vector-digest.v1"
_RECEIPT_SURFACE_DOMAIN = "microcosm.spec-engine.receipt-surface-digest.v1"
_ALLOWED_RECEIPT_RULES = frozenset(
    {
        "equal_after_normalizing_prefix",
        "expected_to_differ_by_generation",
        "operational_excluded",
    }
)
_TOP_GENERATION_POINTERS = frozenset(
    {
        ("run_config", "config_authority"),
        ("run_config", "spec_binding_status"),
        ("run_config", "identity_generation"),
    }
)
_PROVENANCE_ROOT_TOKENS = ("run_config", "run_provenance_identity")
_PROVENANCE_GENERATION_SUFFIXES = frozenset(
    {
        ("identity_generation",),
        ("source_grammar_receipt",),
        ("spec_binding",),
        ("authority_versions", "runtime_authority"),
        ("authority_versions", "execution_abi"),
        ("execution_receipt", "authority_mode"),
    }
)
_GENERATION_POINTERS = _TOP_GENERATION_POINTERS | frozenset(
    (*_PROVENANCE_ROOT_TOKENS, *suffix) for suffix in _PROVENANCE_GENERATION_SUFFIXES
)
_EXECUTION_ABI_KEYS = frozenset(
    {
        "schema_version",
        "present",
        "pipeline",
        "operations",
        "logical_stages",
        "durable_checkpoints",
        "code_abi",
        "normative_artifact_vector",
        "artifact_bindings",
        "source_broker_grant",
        "receipt_comparison_vector",
        "resume_predicate",
        "sha256",
    }
)
_ARTIFACT_ROW_KEYS = frozenset(
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
_RULE_ROW_KEYS = frozenset(
    {"artifact_role", "json_pointer_pattern", "rule", "category"}
)
_CODE_ABI_KEYS = frozenset(
    {
        "domain",
        "content_selectors",
        "locator_grammar",
        "artifact_selector_contract_sha256",
        "compiler_ir_abi_sha256",
        "receipt_difference_match",
        "implementation_sha256",
    }
)
_DURABLE_CHECKPOINT_KEYS = frozenset(
    {
        "id",
        "ordinal",
        "after_operation",
        "covers_operations",
        "artifact_roles",
        "operational_receipts_sidecar",
    }
)
_CHECKPOINT_RECEIPT_SURFACE_KEYS = frozenset({"present", "canonical", "operational"})
_CHECKPOINT_RECEIPT_SIDECAR_KEYS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "materializer_version",
        "stage",
        "identity_sha256",
        "checkpoint",
        "operational_stage_receipts",
    }
)
_CHECKPOINT_BINDING_KEYS = frozenset({"filename", "sha256", "size_bytes"})
_PIPELINE_KEYS = frozenset(
    {
        "id",
        "artifact_protocol",
        "operator_order",
        "producer_order",
        "seed_stream_map_sha256",
    }
)
_ARTIFACT_BINDING_KEYS = frozenset(
    {
        "id",
        "receipt_role",
        "envelope_pointer",
        "locator_ref",
        "raw_identity",
        "manifest_publication_run_id_pointer",
        "manifest_release_id_pointer",
        "embedded_identity_protocol",
        "embedded_publication_run_id_pointer",
        "embedded_release_id_pointer",
    }
)
_NODE_REUSE_KEYS_DOMAIN = "microcosm.spec-engine.node-reuse-key-map.v1"
_SOURCE_BROKER_GRANT_KEYS = frozenset(
    {"domain", "owner", "effects", "sources", "source_set_sha256", "sha256"}
)
_SOURCE_BROKER_GRANT_DOMAIN = "microcosm.spec-engine.source-broker-grant.v1"
_MISSING = object()


class ArtifactComparisonError(ValueError):
    """The plan or supplied comparison surfaces fail closed."""


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """One selected artifact surface represented without retaining its bytes."""

    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        _validate_sha256(self.sha256, location="artifact_digest/sha256")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise ArtifactComparisonError(
                "artifact_digest/byte_size: non-negative integer required"
            )

    @classmethod
    def from_bytes(cls, value: bytes) -> ArtifactDigest:
        """Digest one already-selected normative byte surface."""

        if not isinstance(value, bytes):
            raise ArtifactComparisonError("artifact digest input must be raw bytes")
        return cls(sha256=hashlib.sha256(value).hexdigest(), byte_size=len(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ArtifactDigest:
        """Strictly reconstruct a digest from its JSON wire representation."""

        row = _json_object(value, location="artifact_digest")
        _require_exact_keys(
            row,
            frozenset({"sha256", "byte_size"}),
            location="artifact_digest",
        )
        return cls(sha256=row["sha256"], byte_size=row["byte_size"])  # type: ignore[arg-type]

    def to_wire(self) -> dict[str, object]:
        return {"sha256": self.sha256, "byte_size": self.byte_size}


@dataclass(frozen=True, slots=True)
class GenerationExpectation:
    """Internally derived generation-specific values on both authority sides."""

    constants_value: object
    bundle_value: object


@dataclass(frozen=True, slots=True)
class ArtifactComparisonRow:
    artifact_id: str
    stage_ref: str
    constants_sha256: str
    bundle_sha256: str
    constants_byte_size: int
    bundle_byte_size: int
    equal: bool

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "stage_ref": self.stage_ref,
            "constants_sha256": self.constants_sha256,
            "bundle_sha256": self.bundle_sha256,
            "constants_byte_size": self.constants_byte_size,
            "bundle_byte_size": self.bundle_byte_size,
            "equal": self.equal,
        }


@dataclass(frozen=True, slots=True)
class StageComparisonRow:
    stage_ref: str
    artifact_ids: tuple[str, ...]
    constants_sha256: str
    bundle_sha256: str
    equal: bool

    def to_wire(self) -> dict[str, object]:
        return {
            "stage_ref": self.stage_ref,
            "artifact_ids": list(self.artifact_ids),
            "constants_sha256": self.constants_sha256,
            "bundle_sha256": self.bundle_sha256,
            "equal": self.equal,
        }


@dataclass(frozen=True, slots=True)
class ReceiptComparisonRow:
    artifact_role: str
    json_pointer: str
    rule: str
    category: str
    constants_value_sha256: str | None
    bundle_value_sha256: str | None
    normalized_value_sha256: str | None
    raw_equal: bool | None
    rule_satisfied: bool

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_role": self.artifact_role,
            "json_pointer": self.json_pointer,
            "rule": self.rule,
            "category": self.category,
            "constants_value_sha256": self.constants_value_sha256,
            "bundle_value_sha256": self.bundle_value_sha256,
            "normalized_value_sha256": self.normalized_value_sha256,
            "raw_equal": self.raw_equal,
            "rule_satisfied": self.rule_satisfied,
        }


@dataclass(frozen=True, slots=True)
class NodeReuseKeyComparisonRow:
    node_id: str
    constants_node_reuse_key: str
    bundle_node_reuse_key: str
    equal: bool

    def to_wire(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "constants_node_reuse_key": self.constants_node_reuse_key,
            "bundle_node_reuse_key": self.bundle_node_reuse_key,
            "equal": self.equal,
        }


@dataclass(frozen=True, slots=True)
class ComparisonDifference:
    kind: str
    subject: str
    stage_ref: str | None
    constants_sha256: str | None
    bundle_sha256: str | None
    rule: str | None

    def to_wire(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "stage_ref": self.stage_ref,
            "constants_sha256": self.constants_sha256,
            "bundle_sha256": self.bundle_sha256,
            "rule": self.rule,
        }


@dataclass(frozen=True, slots=True)
class ArtifactComparisonReceipt:
    execution_abi_sha256: str
    artifact_rows: tuple[ArtifactComparisonRow, ...]
    stage_rows: tuple[StageComparisonRow, ...]
    receipt_rows: tuple[ReceiptComparisonRow, ...]
    node_reuse_key_rows: tuple[NodeReuseKeyComparisonRow, ...]
    constants_normative_sha256: str
    bundle_normative_sha256: str
    constants_receipts_sha256: str
    bundle_receipts_sha256: str
    constants_node_reuse_keys_sha256: str
    bundle_node_reuse_keys_sha256: str
    normative_equal: bool
    receipts_equal_under_plan: bool
    node_reuse_keys_equal: bool
    passed: bool
    differences: tuple[ComparisonDifference, ...]
    receipt_sha256: str

    def body_wire(self) -> dict[str, object]:
        return {
            "domain": _COMPARISON_DOMAIN,
            "schema_version": 1,
            "execution_abi_sha256": self.execution_abi_sha256,
            "artifact_rows": [row.to_wire() for row in self.artifact_rows],
            "stage_rows": [row.to_wire() for row in self.stage_rows],
            "receipt_rows": [row.to_wire() for row in self.receipt_rows],
            "node_reuse_key_rows": [row.to_wire() for row in self.node_reuse_key_rows],
            "constants_normative_sha256": self.constants_normative_sha256,
            "bundle_normative_sha256": self.bundle_normative_sha256,
            "constants_receipts_sha256": self.constants_receipts_sha256,
            "bundle_receipts_sha256": self.bundle_receipts_sha256,
            "constants_node_reuse_keys_sha256": (self.constants_node_reuse_keys_sha256),
            "bundle_node_reuse_keys_sha256": self.bundle_node_reuse_keys_sha256,
            "normative_equal": self.normative_equal,
            "receipts_equal_under_plan": self.receipts_equal_under_plan,
            "node_reuse_keys_equal": self.node_reuse_keys_equal,
            "passed": self.passed,
            "differences": [row.to_wire() for row in self.differences],
        }

    def to_wire(self) -> dict[str, object]:
        return {**self.body_wire(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class _ArtifactContract:
    artifact_id: str
    stage_ref: str


@dataclass(frozen=True, slots=True)
class _ReceiptRule:
    artifact_role: str
    pointer_pattern: str
    tokens: tuple[str, ...]
    rule: str
    category: str


@dataclass(frozen=True, slots=True)
class _CheckpointReceiptContract:
    artifact_role: str
    checkpoint_id: str
    expected_present: bool


def checkpoint_receipt_surface(
    sidecar: Mapping[str, object] | None,
) -> dict[str, object]:
    """Split one optional checkpoint receipt into sealed comparison surfaces.

    The physical sidecar is explicitly non-identity-bearing operational
    observability.  Its outer checkpoint binding remains exact; only the
    compiler-declared ``/operational`` wrapper field may differ between cold
    roots.  Accepting raw sidecar mappings here, rather than a pre-split caller
    shape, prevents arbitrary fields from being hidden under that exclusion.
    """

    if sidecar is None:
        return {"present": False, "canonical": {}, "operational": {}}
    value = _json_object(sidecar, location="checkpoint_receipt_sidecar")
    _require_exact_keys(
        value,
        _CHECKPOINT_RECEIPT_SIDECAR_KEYS,
        location="checkpoint_receipt_sidecar",
    )
    for field in ("artifact_kind", "stage"):
        _nonempty_string(
            value[field],
            location=f"checkpoint_receipt_sidecar/{field}",
        )
    for field in ("schema_version", "materializer_version"):
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ArtifactComparisonError(
                f"checkpoint_receipt_sidecar/{field}: positive integer required"
            )
    _validate_sha256(
        value["identity_sha256"],
        location="checkpoint_receipt_sidecar/identity_sha256",
    )
    checkpoint = _json_object(
        value["checkpoint"],
        location="checkpoint_receipt_sidecar/checkpoint",
    )
    _require_exact_keys(
        checkpoint,
        _CHECKPOINT_BINDING_KEYS,
        location="checkpoint_receipt_sidecar/checkpoint",
    )
    filename = _nonempty_string(
        checkpoint["filename"],
        location="checkpoint_receipt_sidecar/checkpoint/filename",
    )
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ArtifactComparisonError(
            "checkpoint_receipt_sidecar/checkpoint/filename: basename required"
        )
    _validate_sha256(
        checkpoint["sha256"],
        location="checkpoint_receipt_sidecar/checkpoint/sha256",
    )
    size = checkpoint["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ArtifactComparisonError(
            "checkpoint_receipt_sidecar/checkpoint/size_bytes: "
            "non-negative integer required"
        )
    operational = _json_object(
        value["operational_stage_receipts"],
        location="checkpoint_receipt_sidecar/operational_stage_receipts",
    )
    if not operational:
        raise ArtifactComparisonError(
            "checkpoint_receipt_sidecar/operational_stage_receipts: "
            "non-empty object required"
        )
    canonical = {
        key: child
        for key, child in value.items()
        if key != "operational_stage_receipts"
    }
    return {
        "present": True,
        "canonical": canonical,
        "operational": operational,
    }


def compare_artifact_sets(
    execution_abi: Mapping[str, object],
    *,
    constants_artifacts: Mapping[str, bytes],
    bundle_artifacts: Mapping[str, bytes],
    constants_receipts: Mapping[str, object],
    bundle_receipts: Mapping[str, object],
    constants_run_provenance_identity: RunProvenanceIdentity,
    bundle_run_provenance_identity: RunProvenanceIdentity,
    constants_node_reuse_keys: Mapping[str, str],
    bundle_node_reuse_keys: Mapping[str, str],
) -> ArtifactComparisonReceipt:
    """Compare materialized artifact bytes through the digest-level kernel."""

    return compare_artifact_digest_sets(
        execution_abi,
        constants_artifacts=_artifact_digests_from_bytes(
            constants_artifacts,
            mode="constants",
        ),
        bundle_artifacts=_artifact_digests_from_bytes(
            bundle_artifacts,
            mode="bundle",
        ),
        constants_receipts=constants_receipts,
        bundle_receipts=bundle_receipts,
        constants_run_provenance_identity=constants_run_provenance_identity,
        bundle_run_provenance_identity=bundle_run_provenance_identity,
        constants_node_reuse_keys=constants_node_reuse_keys,
        bundle_node_reuse_keys=bundle_node_reuse_keys,
    )


def compare_artifact_digest_sets(
    execution_abi: Mapping[str, object],
    *,
    constants_artifacts: Mapping[str, ArtifactDigest],
    bundle_artifacts: Mapping[str, ArtifactDigest],
    constants_receipts: Mapping[str, object],
    bundle_receipts: Mapping[str, object],
    constants_run_provenance_identity: RunProvenanceIdentity,
    bundle_run_provenance_identity: RunProvenanceIdentity,
    constants_node_reuse_keys: Mapping[str, str],
    bundle_node_reuse_keys: Mapping[str, str],
) -> ArtifactComparisonReceipt:
    """Compare two cold-build output sets under one sealed execution ABI.

    Artifact mappings contain digests computed from the collector's selected
    bytes and must contain exactly the ids in ``normative_artifact_vector``.
    This public entry point lets certification receipts compare a large vector
    without re-materializing or fabricating artifact bytes.  Receipt mappings
    carry JSON-shaped values keyed by artifact role.  The generation-zero and
    generation-one provenance identities are typed inputs and are bound in full
    at every provenance receipt root.  Expected leaf transitions are derived
    only from those typed values.  Node reuse maps are bound to the compiler's
    producer inventory and compared exactly; neither surface has a
    caller-authored escape hatch.

    Invalid or incomplete inventories, ambiguous rules, and uncovered receipt
    differences raise :class:`ArtifactComparisonError`.  Covered comparison
    failures return a sealed receipt with ``passed == False`` and explicit
    deterministic differences.
    """

    (
        abi,
        artifact_contracts,
        rules,
        required_receipt_roles,
        checkpoint_receipt_contracts,
        producer_order,
    ) = _validate_execution_abi(execution_abi)
    _validate_run_provenance_pair(
        constants_run_provenance_identity,
        bundle_run_provenance_identity,
        execution_abi_sha256=str(abi["sha256"]),
    )
    constants_node_keys = _validate_node_reuse_keys(
        constants_node_reuse_keys,
        expected_node_ids=producer_order,
        mode="constants",
    )
    bundle_node_keys = _validate_node_reuse_keys(
        bundle_node_reuse_keys,
        expected_node_ids=producer_order,
        mode="bundle",
    )
    node_reuse_key_rows = tuple(
        NodeReuseKeyComparisonRow(
            node_id=node_id,
            constants_node_reuse_key=constants_node_keys[node_id],
            bundle_node_reuse_key=bundle_node_keys[node_id],
            equal=constants_node_keys[node_id] == bundle_node_keys[node_id],
        )
        for node_id in producer_order
    )
    constants_node_reuse_keys_sha256 = _node_reuse_keys_digest(constants_node_keys)
    bundle_node_reuse_keys_sha256 = _node_reuse_keys_digest(bundle_node_keys)
    node_reuse_keys_equal = all(row.equal for row in node_reuse_key_rows)
    constants_digests = _validate_artifact_digests(
        constants_artifacts,
        expected=artifact_contracts,
        mode="constants",
    )
    bundle_digests = _validate_artifact_digests(
        bundle_artifacts,
        expected=artifact_contracts,
        mode="bundle",
    )

    artifact_rows = tuple(
        _artifact_digest_row(contract, constants_digests, bundle_digests)
        for contract in artifact_contracts
    )
    stage_rows = _stage_rows(artifact_rows)
    constants_normative_sha256 = _normative_digest(artifact_rows, "constants")
    bundle_normative_sha256 = _normative_digest(artifact_rows, "bundle")
    constants_receipts_sha256 = _receipt_surface_digest(
        constants_receipts,
        mode="constants",
    )
    bundle_receipts_sha256 = _receipt_surface_digest(
        bundle_receipts,
        mode="bundle",
    )
    normative_equal = all(row.equal for row in artifact_rows)

    receipt_rows = _compare_receipts(
        rules,
        constants_receipts=constants_receipts,
        bundle_receipts=bundle_receipts,
        constants_run_provenance_identity=constants_run_provenance_identity,
        bundle_run_provenance_identity=bundle_run_provenance_identity,
        required_receipt_roles=required_receipt_roles,
        checkpoint_receipt_contracts=checkpoint_receipt_contracts,
    )
    receipts_equal_under_plan = all(row.rule_satisfied for row in receipt_rows)

    differences: list[ComparisonDifference] = []
    for row in artifact_rows:
        if not row.equal:
            differences.append(
                ComparisonDifference(
                    kind="normative_artifact_bytes",
                    subject=row.artifact_id,
                    stage_ref=row.stage_ref,
                    constants_sha256=row.constants_sha256,
                    bundle_sha256=row.bundle_sha256,
                    rule=None,
                )
            )
    for row in receipt_rows:
        if not row.rule_satisfied:
            differences.append(
                ComparisonDifference(
                    kind="receipt_rule_violation",
                    subject=f"{row.artifact_role}:{row.json_pointer}",
                    stage_ref=None,
                    constants_sha256=row.constants_value_sha256,
                    bundle_sha256=row.bundle_value_sha256,
                    rule=row.rule,
                )
            )
    for row in node_reuse_key_rows:
        if not row.equal:
            differences.append(
                ComparisonDifference(
                    kind="node_reuse_key",
                    subject=row.node_id,
                    stage_ref=None,
                    constants_sha256=row.constants_node_reuse_key,
                    bundle_sha256=row.bundle_node_reuse_key,
                    rule="raw_exact",
                )
            )
    difference_rows = tuple(differences)
    passed = normative_equal and receipts_equal_under_plan and node_reuse_keys_equal
    body = {
        "domain": _COMPARISON_DOMAIN,
        "schema_version": 1,
        "execution_abi_sha256": abi["sha256"],
        "artifact_rows": [row.to_wire() for row in artifact_rows],
        "stage_rows": [row.to_wire() for row in stage_rows],
        "receipt_rows": [row.to_wire() for row in receipt_rows],
        "node_reuse_key_rows": [row.to_wire() for row in node_reuse_key_rows],
        "constants_normative_sha256": constants_normative_sha256,
        "bundle_normative_sha256": bundle_normative_sha256,
        "constants_receipts_sha256": constants_receipts_sha256,
        "bundle_receipts_sha256": bundle_receipts_sha256,
        "constants_node_reuse_keys_sha256": constants_node_reuse_keys_sha256,
        "bundle_node_reuse_keys_sha256": bundle_node_reuse_keys_sha256,
        "normative_equal": normative_equal,
        "receipts_equal_under_plan": receipts_equal_under_plan,
        "node_reuse_keys_equal": node_reuse_keys_equal,
        "passed": passed,
        "differences": [row.to_wire() for row in difference_rows],
    }
    return ArtifactComparisonReceipt(
        execution_abi_sha256=str(abi["sha256"]),
        artifact_rows=artifact_rows,
        stage_rows=stage_rows,
        receipt_rows=receipt_rows,
        node_reuse_key_rows=node_reuse_key_rows,
        constants_normative_sha256=constants_normative_sha256,
        bundle_normative_sha256=bundle_normative_sha256,
        constants_receipts_sha256=constants_receipts_sha256,
        bundle_receipts_sha256=bundle_receipts_sha256,
        constants_node_reuse_keys_sha256=constants_node_reuse_keys_sha256,
        bundle_node_reuse_keys_sha256=bundle_node_reuse_keys_sha256,
        normative_equal=normative_equal,
        receipts_equal_under_plan=receipts_equal_under_plan,
        node_reuse_keys_equal=node_reuse_keys_equal,
        passed=passed,
        differences=difference_rows,
        receipt_sha256=sha256_json(body),
    )


def receipt_determinism_projection(
    execution_abi: Mapping[str, object],
    *,
    authority_mode: str,
    receipts: Mapping[str, object],
    run_provenance_identity: RunProvenanceIdentity,
) -> dict[str, object]:
    """Return a plan-validated same-mode receipt comparison projection.

    Undeclared fields remain byte-exact canonical JSON.  Operational leaves
    are replaced only when a sealed rule names them, publication identifiers
    are normalized through the selector contract, and generation-specific
    leaves are retained after validation against the typed run provenance.
    Two cold builds in the same authority mode are deterministic exactly when
    these projections are equal alongside their normative artifact and node
    reuse-key vectors.
    """

    (
        abi,
        _artifact_contracts,
        rules,
        required_receipt_roles,
        checkpoint_receipt_contracts,
        _producer_order,
    ) = _validate_execution_abi(execution_abi)
    identity_wire = _validate_single_run_provenance_identity(
        run_provenance_identity,
        authority_mode=authority_mode,
        execution_abi_sha256=str(abi["sha256"]),
    )
    surface = _json_object(receipts, location=f"{authority_mode}_receipts")
    expected_roles = set(required_receipt_roles) | {row.artifact_role for row in rules}
    if set(surface) != expected_roles:
        raise ArtifactComparisonError(
            f"{authority_mode} receipt role inventory mismatch: "
            f"missing={sorted(expected_roles - set(surface))}, "
            f"extra={sorted(set(surface) - expected_roles)}"
        )
    _validate_checkpoint_receipt_surfaces(
        surface,
        contracts=checkpoint_receipt_contracts,
        mode=authority_mode,
    )

    concrete_rules: dict[tuple[str, tuple[str, ...]], _ReceiptRule] = {}
    for rule in rules:
        paths = _expand_pattern(surface[rule.artifact_role], rule.tokens)
        if not paths:
            raise ArtifactComparisonError(
                "receipt rule matched no field: "
                f"{rule.artifact_role}:{rule.pointer_pattern}"
            )
        for path in paths:
            key = (rule.artifact_role, path)
            if key in concrete_rules:  # pragma: no cover - ABI validator closes this
                raise ArtifactComparisonError(
                    "receipt field matched more than one sealed rule: "
                    f"{rule.artifact_role}:{_encode_pointer(path)}"
                )
            concrete_rules[key] = rule

    generation_roles = {
        role
        for (role, path), rule in concrete_rules.items()
        if rule.rule == "expected_to_differ_by_generation"
        and path[: len(_PROVENANCE_ROOT_TOKENS)] == _PROVENANCE_ROOT_TOKENS
    }
    for role in sorted(generation_roles):
        observed = _value_at(surface[role], _PROVENANCE_ROOT_TOKENS)
        if _json_bytes(
            observed,
            location=f"{authority_mode}_receipts/{role}/run_provenance_identity",
        ) != _json_bytes(
            identity_wire,
            location=f"{authority_mode}_run_provenance_identity",
        ):
            raise ArtifactComparisonError(
                f"{authority_mode} receipt provenance differs from typed identity "
                f"at {role}"
            )

    projection = _json_object(surface, location="receipt_determinism_projection")
    for (role, path), rule in sorted(
        concrete_rules.items(),
        key=lambda item: (item[0][0], _encode_pointer(item[0][1])),
    ):
        observed = _value_at(surface[role], path)
        if rule.rule == "operational_excluded":
            replacement: object = {
                "comparison_rule": "operational_excluded",
            }
        elif rule.rule == "equal_after_normalizing_prefix":
            normalized = _normalize_prefix(observed, authority_mode=authority_mode)
            if normalized is None:
                raise ArtifactComparisonError(
                    f"{authority_mode}_receipts/{role}{_encode_pointer(path)}: "
                    "invalid publication identifier for prefix normalization"
                )
            replacement = normalized
        else:
            expected = _generation_value_for_mode(
                path,
                authority_mode=authority_mode,
                identity_wire=identity_wire,
            )
            if _json_bytes(
                observed,
                location=f"{authority_mode}_receipts/{role}{_encode_pointer(path)}",
            ) != _json_bytes(
                expected,
                location=(
                    f"{authority_mode}_generation_expectation/"
                    f"{role}{_encode_pointer(path)}"
                ),
            ):
                raise ArtifactComparisonError(
                    f"{authority_mode}_receipts/{role}{_encode_pointer(path)}: "
                    "generation value differs from typed provenance"
                )
            replacement = observed
        _replace_value_at(projection[role], path, replacement)
    return projection


def _validate_execution_abi(
    execution_abi: Mapping[str, object],
) -> tuple[
    dict[str, object],
    tuple[_ArtifactContract, ...],
    tuple[_ReceiptRule, ...],
    frozenset[str],
    tuple[_CheckpointReceiptContract, ...],
    tuple[str, ...],
]:
    abi = _json_object(execution_abi, location="execution_abi")
    _require_exact_keys(abi, _EXECUTION_ABI_KEYS, location="execution_abi")
    if abi["schema_version"] != 1:
        raise ArtifactComparisonError("execution_abi/schema_version: unsupported")
    if abi["present"] is not True:
        raise ArtifactComparisonError("execution_abi is absent for this plan")
    _validate_sha256(abi["sha256"], location="execution_abi/sha256")
    unsigned = {key: value for key, value in abi.items() if key != "sha256"}
    if sha256_json(unsigned) != abi["sha256"]:
        raise ArtifactComparisonError("execution_abi seal mismatch")

    code_abi = _json_object(abi["code_abi"], location="execution_abi/code_abi")
    _require_exact_keys(code_abi, _CODE_ABI_KEYS, location="execution_abi/code_abi")
    if code_abi["receipt_difference_match"] != "exactly_one_sealed_rule":
        raise ArtifactComparisonError(
            "execution_abi/code_abi does not require exactly one sealed rule"
        )
    if code_abi["locator_grammar"] != ARTIFACT_LOCATOR_GRAMMAR:
        raise ArtifactComparisonError(
            "execution_abi/code_abi locator grammar is unsupported"
        )
    if (
        code_abi["artifact_selector_contract_sha256"]
        != ARTIFACT_SELECTOR_CONTRACT_SHA256
    ):
        raise ArtifactComparisonError(
            "execution_abi/code_abi artifact selector contract is unsupported"
        )
    if code_abi["compiler_ir_abi_sha256"] != current_compiler_ir_abi().sha256:
        raise ArtifactComparisonError(
            "execution_abi/code_abi compiler implementation attestation is stale"
        )
    _validate_sha256(
        code_abi["implementation_sha256"],
        location="execution_abi/code_abi/implementation_sha256",
    )
    code_unsigned = {
        key: value for key, value in code_abi.items() if key != "implementation_sha256"
    }
    if sha256_json(code_unsigned) != code_abi["implementation_sha256"]:
        raise ArtifactComparisonError("execution_abi/code_abi seal mismatch")
    selector_refs = _string_array(
        code_abi["content_selectors"],
        location="execution_abi/code_abi/content_selectors",
    )
    if len(selector_refs) != len(set(selector_refs)):
        raise ArtifactComparisonError(
            "execution_abi/code_abi/content_selectors contains duplicates"
        )

    pipeline = _json_object(abi["pipeline"], location="execution_abi/pipeline")
    _require_exact_keys(pipeline, _PIPELINE_KEYS, location="execution_abi/pipeline")
    _nonempty_string(pipeline["id"], location="execution_abi/pipeline/id")
    _json_object(
        pipeline["artifact_protocol"],
        location="execution_abi/pipeline/artifact_protocol",
    )
    operator_order = _string_array(
        pipeline["operator_order"],
        location="execution_abi/pipeline/operator_order",
    )
    producer_order = _string_array(
        pipeline["producer_order"],
        location="execution_abi/pipeline/producer_order",
    )
    if len(operator_order) != len(set(operator_order)):
        raise ArtifactComparisonError(
            "execution_abi/pipeline/operator_order contains duplicates"
        )
    if len(producer_order) != len(set(producer_order)):
        raise ArtifactComparisonError(
            "execution_abi/pipeline/producer_order contains duplicates"
        )
    _validate_sha256(
        pipeline["seed_stream_map_sha256"],
        location="execution_abi/pipeline/seed_stream_map_sha256",
    )
    _validate_source_broker_grant_abi(abi["source_broker_grant"])

    artifact_values = _json_array(
        abi["normative_artifact_vector"],
        location="execution_abi/normative_artifact_vector",
    )
    if not artifact_values:
        raise ArtifactComparisonError("normative artifact vector must not be empty")
    artifact_contracts: list[_ArtifactContract] = []
    artifact_ids: set[str] = set()
    selector_set = set(selector_refs)
    for index, value in enumerate(artifact_values):
        location = f"execution_abi/normative_artifact_vector/{index}"
        row = _json_object(value, location=location)
        _require_exact_keys(row, _ARTIFACT_ROW_KEYS, location=location)
        artifact_id = _nonempty_string(row["id"], location=f"{location}/id")
        if artifact_id in artifact_ids:
            raise ArtifactComparisonError(
                f"duplicate normative artifact id {artifact_id!r}"
            )
        artifact_ids.add(artifact_id)
        stage_ref = _nonempty_string(row["stage_ref"], location=f"{location}/stage_ref")
        if (
            row["surface"] != "normative"
            or row["comparison"] != "raw_byte_exact"
            or row["required"] is not True
        ):
            raise ArtifactComparisonError(
                f"{location}: artifacts must be required normative raw-byte exact"
            )
        selector = _nonempty_string(
            row["content_selector_ref"],
            location=f"{location}/content_selector_ref",
        )
        if selector not in selector_set:
            raise ArtifactComparisonError(
                f"{location}: undeclared content selector {selector!r}"
            )
        for key in ("kind", "producer_ref", "protocol_ref", "locator_ref"):
            _nonempty_string(row[key], location=f"{location}/{key}")
        artifact_contracts.append(
            _ArtifactContract(artifact_id=artifact_id, stage_ref=stage_ref)
        )

    artifact_locator_selectors: dict[str, set[str]] = {}
    for value in artifact_values:
        assert isinstance(value, dict)
        artifact_locator_selectors.setdefault(str(value["locator_ref"]), set()).add(
            str(value["content_selector_ref"])
        )
    binding_ids: set[str] = set()
    binding_locators: set[str] = set()
    binding_scopes: set[tuple[str, tuple[str, ...]]] = set()
    binding_values = _json_array(
        abi["artifact_bindings"],
        location="execution_abi/artifact_bindings",
    )
    for index, value in enumerate(binding_values):
        location = f"execution_abi/artifact_bindings/{index}"
        row = _json_object(value, location=location)
        _require_exact_keys(row, _ARTIFACT_BINDING_KEYS, location=location)
        binding_id = _nonempty_string(row["id"], location=f"{location}/id")
        if binding_id in binding_ids:
            raise ArtifactComparisonError(
                f"duplicate artifact binding id {binding_id!r}"
            )
        binding_ids.add(binding_id)
        role = _nonempty_string(
            row["receipt_role"], location=f"{location}/receipt_role"
        )
        envelope = _parse_concrete_pointer(
            _nonempty_string(
                row["envelope_pointer"], location=f"{location}/envelope_pointer"
            )
        )
        if (role, envelope) in binding_scopes:
            raise ArtifactComparisonError(f"{location}: duplicate binding envelope")
        binding_scopes.add((role, envelope))
        locator = _nonempty_string(
            row["locator_ref"], location=f"{location}/locator_ref"
        )
        if locator not in artifact_locator_selectors:
            raise ArtifactComparisonError(
                f"{location}/locator_ref: absent from normative artifact vector"
            )
        if locator in binding_locators:
            raise ArtifactComparisonError(f"{location}: duplicate bound locator")
        binding_locators.add(locator)
        if row["raw_identity"] != "resolved_path_sha256_size_bytes_v1":
            raise ArtifactComparisonError(f"{location}: unsupported raw identity")
        protocol = row["embedded_identity_protocol"]
        if protocol not in {
            "h5_artifact_metadata_v1",
            "json_root_publication_identity_v1",
        }:
            raise ArtifactComparisonError(
                f"{location}: unsupported embedded identity protocol"
            )
        for field in (
            "manifest_publication_run_id_pointer",
            "manifest_release_id_pointer",
            "embedded_publication_run_id_pointer",
        ):
            _parse_concrete_pointer(
                _nonempty_string(row[field], location=f"{location}/{field}")
            )
        embedded_release = row["embedded_release_id_pointer"]
        if protocol == "h5_artifact_metadata_v1":
            if embedded_release is not None:
                raise ArtifactComparisonError(
                    f"{location}/embedded_release_id_pointer: null required"
                )
        else:
            _parse_concrete_pointer(
                _nonempty_string(
                    embedded_release,
                    location=f"{location}/embedded_release_id_pointer",
                )
            )
    all_selectors = {
        selector
        for selectors in artifact_locator_selectors.values()
        for selector in selectors
    }
    expected_binding_locators = (
        {
            locator
            for locator, selectors in artifact_locator_selectors.items()
            if selectors
            in (
                {
                    "selector:h5_all_entity_tables_and_columns_v1",
                    "selector:h5_all_weight_vectors_v1",
                },
                {"selector:terminal_gate_normative_rows_v1"},
            )
        }
        if "selector:publication_normative_vector_v1" in all_selectors
        else set()
    )
    if binding_locators != expected_binding_locators:
        raise ArtifactComparisonError("artifact binding inventory mismatch")

    checkpoint_values = _json_array(
        abi["durable_checkpoints"],
        location="execution_abi/durable_checkpoints",
    )
    required_receipt_roles: set[str] = set()
    checkpoint_receipt_contracts: list[_CheckpointReceiptContract] = []
    checkpoint_ids: set[str] = set()
    checkpoint_artifact_roles: set[str] = set()
    for index, value in enumerate(checkpoint_values):
        location = f"execution_abi/durable_checkpoints/{index}"
        checkpoint = _json_object(value, location=location)
        _require_exact_keys(checkpoint, _DURABLE_CHECKPOINT_KEYS, location=location)
        checkpoint_id = _nonempty_string(checkpoint["id"], location=f"{location}/id")
        if checkpoint_id in checkpoint_ids:
            raise ArtifactComparisonError(
                f"duplicate durable checkpoint id {checkpoint_id!r}"
            )
        checkpoint_ids.add(checkpoint_id)
        receipts_policy = _nonempty_string(
            checkpoint["operational_receipts_sidecar"],
            location=f"{location}/operational_receipts_sidecar",
        )
        if receipts_policy not in {"forbidden", "required"}:
            raise ArtifactComparisonError(
                f"{location}/operational_receipts_sidecar: "
                "forbidden or required expected"
            )
        roles = _string_array(
            checkpoint["artifact_roles"],
            location=f"{location}/artifact_roles",
        )
        if len(roles) != len(set(roles)):
            raise ArtifactComparisonError(
                f"{location}/artifact_roles contains duplicates"
            )
        for role in roles:
            if role in checkpoint_artifact_roles:
                raise ArtifactComparisonError(
                    f"durable checkpoint artifact role is repeated: {role!r}"
                )
            checkpoint_artifact_roles.add(role)
            if role in artifact_ids:
                continue
            role_kind = role.rsplit(":", 1)[-1]
            if role_kind not in {"manifest", "receipts"}:
                raise ArtifactComparisonError(
                    f"{location}/artifact_roles: non-normative role {role!r} "
                    "must be a manifest or receipts sidecar"
                )
            required_receipt_roles.add(role)
        receipt_role = f"checkpoint:{checkpoint_id}:receipts"
        if receipt_role not in roles:
            raise ArtifactComparisonError(
                f"{location}/artifact_roles: missing {receipt_role!r}"
            )
        checkpoint_receipt_contracts.append(
            _CheckpointReceiptContract(
                artifact_role=receipt_role,
                checkpoint_id=checkpoint_id,
                expected_present=receipts_policy == "required",
            )
        )

    rule_values = _json_array(
        abi["receipt_comparison_vector"],
        location="execution_abi/receipt_comparison_vector",
    )
    rules: list[_ReceiptRule] = []
    for index, value in enumerate(rule_values):
        location = f"execution_abi/receipt_comparison_vector/{index}"
        row = _json_object(value, location=location)
        _require_exact_keys(row, _RULE_ROW_KEYS, location=location)
        role = _nonempty_string(
            row["artifact_role"], location=f"{location}/artifact_role"
        )
        pattern = _nonempty_string(
            row["json_pointer_pattern"],
            location=f"{location}/json_pointer_pattern",
        )
        tokens = _parse_pointer_pattern(pattern, location=location)
        rule = _nonempty_string(row["rule"], location=f"{location}/rule")
        if rule not in _ALLOWED_RECEIPT_RULES:
            raise ArtifactComparisonError(f"{location}: unknown receipt rule {rule!r}")
        if (
            rule == "expected_to_differ_by_generation"
            and tokens not in _GENERATION_POINTERS
        ):
            raise ArtifactComparisonError(
                f"{location}: unknown generation-difference field semantics "
                f"{_encode_pointer(tokens)!r}"
            )
        category = _nonempty_string(row["category"], location=f"{location}/category")
        candidate = _ReceiptRule(
            artifact_role=role,
            pointer_pattern=pattern,
            tokens=tokens,
            rule=rule,
            category=category,
        )
        for previous in rules:
            if previous.artifact_role == role and _rule_scopes_overlap(
                previous.tokens, tokens
            ):
                raise ArtifactComparisonError(
                    "overlapping or duplicate receipt rules for "
                    f"{role!r}: {previous.pointer_pattern!r} and {pattern!r}"
                )
        rules.append(candidate)
    return (
        abi,
        tuple(artifact_contracts),
        tuple(rules),
        frozenset(required_receipt_roles),
        tuple(checkpoint_receipt_contracts),
        producer_order,
    )


def _validate_source_broker_grant_abi(value: object) -> None:
    location = "execution_abi/source_broker_grant"
    grant = _json_object(value, location=location)
    _require_exact_keys(grant, _SOURCE_BROKER_GRANT_KEYS, location=location)
    if (
        grant["domain"] != _SOURCE_BROKER_GRANT_DOMAIN
        or grant["owner"] != {"kind": "source_stage", "id": "declared_source_preflight"}
        or grant["effects"] != ["declared_source_read"]
    ):
        raise ArtifactComparisonError(f"{location}: source grant contract changed")
    _validate_sha256(grant["sha256"], location=f"{location}/sha256")
    unsigned = {key: child for key, child in grant.items() if key != "sha256"}
    if sha256_json(unsigned) != grant["sha256"]:
        raise ArtifactComparisonError(f"{location}: source grant seal mismatch")
    rows = _json_array(grant["sources"], location=f"{location}/sources")
    normalized: list[dict[str, object]] = []
    ids: list[str] = []
    for index, value in enumerate(rows):
        row_location = f"{location}/sources/{index}"
        row = _json_object(value, location=row_location)
        _require_exact_keys(
            row,
            frozenset({"id", "sha256", "byte_size"}),
            location=row_location,
        )
        source_id = _nonempty_string(row["id"], location=f"{row_location}/id")
        _validate_sha256(row["sha256"], location=f"{row_location}/sha256")
        size = row["byte_size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ArtifactComparisonError(
                f"{row_location}/byte_size: positive integer required"
            )
        ids.append(source_id)
        normalized.append({"id": source_id, "sha256": row["sha256"], "byte_size": size})
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ArtifactComparisonError(f"{location}/sources: sorted unique ids required")
    expected_set_sha256 = sha256_json(
        {"domain": _SOURCE_BROKER_GRANT_DOMAIN, "sources": normalized}
    )
    if grant["source_set_sha256"] != expected_set_sha256:
        raise ArtifactComparisonError(f"{location}/source_set_sha256: mismatch")


def _artifact_digests_from_bytes(
    values: Mapping[str, bytes],
    *,
    mode: str,
) -> dict[str, ArtifactDigest]:
    if not isinstance(values, Mapping):
        raise ArtifactComparisonError(f"{mode}_artifacts: object required")
    observed: dict[str, ArtifactDigest] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ArtifactComparisonError(f"{mode}_artifacts: string keys required")
        if not isinstance(value, bytes):
            raise ArtifactComparisonError(f"{mode}_artifacts/{key}: raw bytes required")
        observed[key] = ArtifactDigest.from_bytes(value)
    return observed


def _validate_artifact_digests(
    values: Mapping[str, ArtifactDigest],
    *,
    expected: Sequence[_ArtifactContract],
    mode: str,
) -> dict[str, ArtifactDigest]:
    if not isinstance(values, Mapping):
        raise ArtifactComparisonError(f"{mode}_artifacts: object required")
    observed: dict[str, ArtifactDigest] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ArtifactComparisonError(f"{mode}_artifacts: string keys required")
        if not isinstance(value, ArtifactDigest):
            raise ArtifactComparisonError(
                f"{mode}_artifacts/{key}: typed ArtifactDigest required"
            )
        observed[key] = value
    expected_ids = {row.artifact_id for row in expected}
    observed_ids = set(observed)
    if expected_ids != observed_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ArtifactComparisonError(
            f"{mode} artifact inventory mismatch: missing={missing}, extra={extra}"
        )
    return observed


def _artifact_digest_row(
    contract: _ArtifactContract,
    constants_artifacts: Mapping[str, ArtifactDigest],
    bundle_artifacts: Mapping[str, ArtifactDigest],
) -> ArtifactComparisonRow:
    constants_value = constants_artifacts[contract.artifact_id]
    bundle_value = bundle_artifacts[contract.artifact_id]
    return ArtifactComparisonRow(
        artifact_id=contract.artifact_id,
        stage_ref=contract.stage_ref,
        constants_sha256=constants_value.sha256,
        bundle_sha256=bundle_value.sha256,
        constants_byte_size=constants_value.byte_size,
        bundle_byte_size=bundle_value.byte_size,
        equal=constants_value == bundle_value,
    )


def _stage_rows(
    artifact_rows: Sequence[ArtifactComparisonRow],
) -> tuple[StageComparisonRow, ...]:
    stage_order = tuple(dict.fromkeys(row.stage_ref for row in artifact_rows))
    rows: list[StageComparisonRow] = []
    for stage_ref in stage_order:
        selected = tuple(row for row in artifact_rows if row.stage_ref == stage_ref)
        constants_sha256 = _stage_digest(stage_ref, selected, "constants")
        bundle_sha256 = _stage_digest(stage_ref, selected, "bundle")
        rows.append(
            StageComparisonRow(
                stage_ref=stage_ref,
                artifact_ids=tuple(row.artifact_id for row in selected),
                constants_sha256=constants_sha256,
                bundle_sha256=bundle_sha256,
                equal=all(row.equal for row in selected),
            )
        )
    return tuple(rows)


def _stage_digest(
    stage_ref: str,
    rows: Sequence[ArtifactComparisonRow],
    mode: str,
) -> str:
    return sha256_json(
        {
            "domain": _STAGE_VECTOR_DOMAIN,
            "stage_ref": stage_ref,
            "artifacts": [
                {
                    "id": row.artifact_id,
                    "sha256": getattr(row, f"{mode}_sha256"),
                    "byte_size": getattr(row, f"{mode}_byte_size"),
                }
                for row in rows
            ],
        }
    )


def _normative_digest(
    rows: Sequence[ArtifactComparisonRow],
    mode: str,
) -> str:
    return sha256_json(
        {
            "domain": _NORMATIVE_VECTOR_DOMAIN,
            "artifacts": [
                {
                    "id": row.artifact_id,
                    "stage_ref": row.stage_ref,
                    "sha256": getattr(row, f"{mode}_sha256"),
                    "byte_size": getattr(row, f"{mode}_byte_size"),
                }
                for row in rows
            ],
        }
    )


def _receipt_surface_digest(
    receipts: Mapping[str, object],
    *,
    mode: str,
) -> str:
    surface = _json_object(receipts, location=f"{mode}_receipts")
    return sha256_json(
        {
            "domain": _RECEIPT_SURFACE_DOMAIN,
            "receipts": surface,
        }
    )


def _compare_receipts(
    rules: Sequence[_ReceiptRule],
    *,
    constants_receipts: Mapping[str, object],
    bundle_receipts: Mapping[str, object],
    constants_run_provenance_identity: RunProvenanceIdentity,
    bundle_run_provenance_identity: RunProvenanceIdentity,
    required_receipt_roles: frozenset[str],
    checkpoint_receipt_contracts: Sequence[_CheckpointReceiptContract],
) -> tuple[ReceiptComparisonRow, ...]:
    constants = _json_object(constants_receipts, location="constants_receipts")
    bundle = _json_object(bundle_receipts, location="bundle_receipts")
    rule_roles = {row.artifact_role for row in rules}
    expected_roles = set(required_receipt_roles) | rule_roles
    if set(constants) != expected_roles:
        missing = sorted(expected_roles - set(constants))
        extra = sorted(set(constants) - expected_roles)
        raise ArtifactComparisonError(
            "constants receipt role inventory mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if set(bundle) != expected_roles:
        missing = sorted(expected_roles - set(bundle))
        extra = sorted(set(bundle) - expected_roles)
        raise ArtifactComparisonError(
            f"bundle receipt role inventory mismatch: missing={missing}, extra={extra}"
        )
    _validate_checkpoint_receipt_surfaces(
        constants,
        contracts=checkpoint_receipt_contracts,
        mode="constants",
    )
    _validate_checkpoint_receipt_surfaces(
        bundle,
        contracts=checkpoint_receipt_contracts,
        mode="bundle",
    )
    concrete_rules: dict[tuple[str, tuple[str, ...]], _ReceiptRule] = {}
    for rule in rules:
        constants_paths = set(
            _expand_pattern(constants[rule.artifact_role], rule.tokens)
        )
        bundle_paths = set(_expand_pattern(bundle[rule.artifact_role], rule.tokens))
        if not constants_paths and not bundle_paths:
            raise ArtifactComparisonError(
                "receipt rule matched no field: "
                f"{rule.artifact_role}:{rule.pointer_pattern}"
            )
        if constants_paths != bundle_paths:
            raise ArtifactComparisonError(
                "receipt rule scope differs between modes: "
                f"{rule.artifact_role}:{rule.pointer_pattern}"
            )
        for path in sorted(constants_paths):
            key = (rule.artifact_role, path)
            if key in concrete_rules:
                raise ArtifactComparisonError(
                    "receipt field matched more than one sealed rule: "
                    f"{rule.artifact_role}:{_encode_pointer(path)}"
                )
            concrete_rules[key] = rule

    generation_keys = {
        (role, _encode_pointer(path))
        for (role, path), rule in concrete_rules.items()
        if rule.rule == "expected_to_differ_by_generation"
    }
    expectations = _derive_generation_expectations(
        generation_keys=generation_keys,
        constants_run_provenance_identity=constants_run_provenance_identity,
        bundle_run_provenance_identity=bundle_run_provenance_identity,
    )
    _bind_receipt_provenance_roots(
        constants,
        bundle,
        generation_keys=generation_keys,
        constants_run_provenance_identity=constants_run_provenance_identity,
        bundle_run_provenance_identity=bundle_run_provenance_identity,
    )

    unmatched: list[tuple[str, str]] = []
    ignored_by_role: dict[str, set[tuple[str, ...]]] = {}
    for (role, path), _rule in concrete_rules.items():
        ignored_by_role.setdefault(role, set()).add(path)
    for role in sorted(constants):
        for path in _unmatched_differences(
            constants[role],
            bundle[role],
            ignored_paths=ignored_by_role.get(role, set()),
        ):
            unmatched.append((role, _encode_pointer(path)))
    if unmatched:
        rendered = ", ".join(f"{role}:{path}" for role, path in unmatched)
        raise ArtifactComparisonError(f"unmatched receipt differences: {rendered}")

    rows: list[ReceiptComparisonRow] = []
    for (role, path), rule in sorted(
        concrete_rules.items(),
        key=lambda item: (item[0][0], _encode_pointer(item[0][1])),
    ):
        pointer = _encode_pointer(path)
        constants_value = _value_at(constants[role], path)
        bundle_value = _value_at(bundle[role], path)
        rows.append(
            _receipt_rule_row(
                role,
                pointer,
                rule,
                constants_value,
                bundle_value,
                expectation=expectations.get((role, pointer)),
            )
        )
    return tuple(rows)


def _validate_checkpoint_receipt_surfaces(
    receipts: Mapping[str, object],
    *,
    contracts: Sequence[_CheckpointReceiptContract],
    mode: str,
) -> None:
    for contract in contracts:
        location = f"{mode}_receipts/{contract.artifact_role}"
        surface = _json_object(receipts[contract.artifact_role], location=location)
        _require_exact_keys(
            surface,
            _CHECKPOINT_RECEIPT_SURFACE_KEYS,
            location=location,
        )
        present = surface["present"]
        if not isinstance(present, bool):
            raise ArtifactComparisonError(f"{location}/present: boolean required")
        if present is not contract.expected_present:
            expectation = "present" if contract.expected_present else "absent"
            raise ArtifactComparisonError(
                f"{location}: sidecar must be {expectation} under the sealed plan"
            )
        if not present:
            if surface != checkpoint_receipt_surface(None):
                raise ArtifactComparisonError(
                    f"{location}: absent sidecar surface must be exactly empty"
                )
            continue
        canonical = _json_object(
            surface["canonical"],
            location=f"{location}/canonical",
        )
        operational = _json_object(
            surface["operational"],
            location=f"{location}/operational",
        )
        reconstructed = checkpoint_receipt_surface(
            {**canonical, "operational_stage_receipts": operational}
        )
        if reconstructed != surface:
            raise ArtifactComparisonError(
                f"{location}: sidecar surface differs from the trusted split"
            )
        if canonical.get("stage") != contract.checkpoint_id:
            raise ArtifactComparisonError(
                f"{location}/canonical/stage: differs from checkpoint id"
            )


def _receipt_rule_row(
    role: str,
    pointer: str,
    rule: _ReceiptRule,
    constants_value: object,
    bundle_value: object,
    *,
    expectation: GenerationExpectation | None,
) -> ReceiptComparisonRow:
    constants_bytes = _json_bytes(
        constants_value, location=f"constants_receipts/{role}{pointer}"
    )
    bundle_bytes = _json_bytes(
        bundle_value, location=f"bundle_receipts/{role}{pointer}"
    )
    raw_equal = constants_bytes == bundle_bytes
    constants_sha256: str | None = hashlib.sha256(constants_bytes).hexdigest()
    bundle_sha256: str | None = hashlib.sha256(bundle_bytes).hexdigest()
    normalized_sha256: str | None = None
    if rule.rule == "operational_excluded":
        rule_satisfied = True
        raw_equal_surface: bool | None = None
        constants_sha256 = None
        bundle_sha256 = None
    elif rule.rule == "equal_after_normalizing_prefix":
        constants_normalized = _normalize_prefix(
            constants_value,
            authority_mode="constants",
        )
        bundle_normalized = _normalize_prefix(
            bundle_value,
            authority_mode="bundle",
        )
        rule_satisfied = (
            constants_normalized is not None
            and constants_normalized == bundle_normalized
        )
        raw_equal_surface = raw_equal
        if rule_satisfied:
            normalized_sha256 = sha256_json(constants_normalized)
    else:
        if expectation is None:
            raise AssertionError("generation expectation inventory was not closed")
        expected_constants = _json_bytes(
            expectation.constants_value,
            location=f"generation_expectations/{role}{pointer}/constants",
        )
        expected_bundle = _json_bytes(
            expectation.bundle_value,
            location=f"generation_expectations/{role}{pointer}/bundle",
        )
        rule_satisfied = (
            expected_constants != expected_bundle
            and constants_bytes == expected_constants
            and bundle_bytes == expected_bundle
            and not raw_equal
        )
        raw_equal_surface = raw_equal
    return ReceiptComparisonRow(
        artifact_role=role,
        json_pointer=pointer,
        rule=rule.rule,
        category=rule.category,
        constants_value_sha256=constants_sha256,
        bundle_value_sha256=bundle_sha256,
        normalized_value_sha256=normalized_sha256,
        raw_equal=raw_equal_surface,
        rule_satisfied=rule_satisfied,
    )


def _validate_run_provenance_pair(
    constants_identity: RunProvenanceIdentity,
    bundle_identity: RunProvenanceIdentity,
    *,
    execution_abi_sha256: str,
) -> None:
    if not isinstance(constants_identity, RunProvenanceIdentity):
        raise ArtifactComparisonError(
            "constants_run_provenance_identity must be a typed RunProvenanceIdentity"
        )
    if not isinstance(bundle_identity, RunProvenanceIdentity):
        raise ArtifactComparisonError(
            "bundle_run_provenance_identity must be a typed RunProvenanceIdentity"
        )
    if constants_identity.identity_generation != 0:
        raise ArtifactComparisonError(
            "constants_run_provenance_identity must have identity_generation 0"
        )
    if bundle_identity.identity_generation != 1:
        raise ArtifactComparisonError(
            "bundle_run_provenance_identity must have identity_generation 1"
        )
    constants_wire = constants_identity.to_wire()
    bundle_wire = bundle_identity.to_wire()
    binding = bundle_wire["spec_binding"]
    if (
        not isinstance(binding, dict)
        or binding.get("attestation") != "bundle-authoritative"
    ):
        raise ArtifactComparisonError(
            "bundle_run_provenance_identity spec_binding attestation must be "
            "bundle-authoritative"
        )
    expected_authority_fields = {
        "stacked_authority",
        "checkpoint_materializer",
        "runtime_authority",
        "execution_abi",
    }
    for mode, wire in (("constants", constants_wire), ("bundle", bundle_wire)):
        authority_versions = wire["authority_versions"]
        if not isinstance(authority_versions, dict) or set(authority_versions) != (
            expected_authority_fields
        ):
            raise ArtifactComparisonError(
                f"{mode}_run_provenance_identity authority_versions keys changed"
            )
        execution_receipt = wire["execution_receipt"]
        if not isinstance(execution_receipt, dict) or set(execution_receipt) != {
            "authority_mode",
            "pipeline",
            "code_pin",
        }:
            raise ArtifactComparisonError(
                f"{mode}_run_provenance_identity execution_receipt keys changed"
            )
        if execution_receipt["authority_mode"] != mode:
            raise ArtifactComparisonError(
                f"{mode}_run_provenance_identity execution authority_mode differs"
            )
    constants_versions = constants_wire["authority_versions"]
    bundle_versions = bundle_wire["authority_versions"]
    assert isinstance(constants_versions, dict) and isinstance(bundle_versions, dict)
    if (
        constants_versions["runtime_authority"] is not None
        or constants_versions["execution_abi"] is not None
    ):
        raise ArtifactComparisonError(
            "constants_run_provenance_identity bundle authority versions must be null"
        )
    _validate_sha256(
        bundle_versions["runtime_authority"],
        location="bundle_run_provenance_identity/authority_versions/runtime_authority",
    )
    if bundle_versions["execution_abi"] != execution_abi_sha256:
        raise ArtifactComparisonError(
            "bundle_run_provenance_identity authority_versions/execution_abi "
            "differs from the compared execution ABI"
        )


def _validate_single_run_provenance_identity(
    identity: RunProvenanceIdentity,
    *,
    authority_mode: str,
    execution_abi_sha256: str,
) -> dict[str, object]:
    if authority_mode not in {"constants", "bundle"}:
        raise ArtifactComparisonError("authority_mode must be 'constants' or 'bundle'")
    if not isinstance(identity, RunProvenanceIdentity):
        raise ArtifactComparisonError(
            f"{authority_mode}_run_provenance_identity must be a typed "
            "RunProvenanceIdentity"
        )
    expected_generation = 0 if authority_mode == "constants" else 1
    if identity.identity_generation != expected_generation:
        raise ArtifactComparisonError(
            f"{authority_mode}_run_provenance_identity must have "
            f"identity_generation {expected_generation}"
        )
    wire = identity.to_wire()
    authority_versions = wire["authority_versions"]
    expected_authority_fields = {
        "stacked_authority",
        "checkpoint_materializer",
        "runtime_authority",
        "execution_abi",
    }
    if not isinstance(authority_versions, dict) or set(authority_versions) != (
        expected_authority_fields
    ):
        raise ArtifactComparisonError(
            f"{authority_mode}_run_provenance_identity authority_versions keys changed"
        )
    execution_receipt = wire["execution_receipt"]
    if not isinstance(execution_receipt, dict) or set(execution_receipt) != {
        "authority_mode",
        "pipeline",
        "code_pin",
    }:
        raise ArtifactComparisonError(
            f"{authority_mode}_run_provenance_identity execution_receipt keys changed"
        )
    if execution_receipt["authority_mode"] != authority_mode:
        raise ArtifactComparisonError(
            f"{authority_mode}_run_provenance_identity execution authority_mode differs"
        )
    if authority_mode == "constants":
        if (
            authority_versions["runtime_authority"] is not None
            or authority_versions["execution_abi"] is not None
        ):
            raise ArtifactComparisonError(
                "constants_run_provenance_identity bundle authority versions "
                "must be null"
            )
    else:
        binding = wire["spec_binding"]
        if (
            not isinstance(binding, dict)
            or binding.get("attestation") != "bundle-authoritative"
        ):
            raise ArtifactComparisonError(
                "bundle_run_provenance_identity spec_binding attestation must be "
                "bundle-authoritative"
            )
        _validate_sha256(
            authority_versions["runtime_authority"],
            location=(
                "bundle_run_provenance_identity/authority_versions/runtime_authority"
            ),
        )
        if authority_versions["execution_abi"] != execution_abi_sha256:
            raise ArtifactComparisonError(
                "bundle_run_provenance_identity authority_versions/execution_abi "
                "differs from the compared execution ABI"
            )
    return wire


def _generation_value_for_mode(
    path: tuple[str, ...],
    *,
    authority_mode: str,
    identity_wire: Mapping[str, object],
) -> object:
    scalars: dict[tuple[str, ...], tuple[object, object]] = {
        ("run_config", "config_authority"): ("constants", "bundle"),
        ("run_config", "spec_binding_status"): ("absent", "resolved"),
        ("run_config", "identity_generation"): (0, 1),
    }
    if path in scalars:
        return scalars[path][0 if authority_mode == "constants" else 1]
    if path[: len(_PROVENANCE_ROOT_TOKENS)] != _PROVENANCE_ROOT_TOKENS:
        raise AssertionError("unknown generation semantics passed plan validation")
    return _value_at(identity_wire, path[len(_PROVENANCE_ROOT_TOKENS) :])


def _derive_generation_expectations(
    *,
    generation_keys: set[tuple[str, str]],
    constants_run_provenance_identity: RunProvenanceIdentity,
    bundle_run_provenance_identity: RunProvenanceIdentity,
) -> dict[tuple[str, str], GenerationExpectation]:
    scalar_pairs = {
        ("run_config", "config_authority"): GenerationExpectation(
            "constants", "bundle"
        ),
        ("run_config", "spec_binding_status"): GenerationExpectation(
            "absent", "resolved"
        ),
        ("run_config", "identity_generation"): GenerationExpectation(0, 1),
    }
    constants_wire = constants_run_provenance_identity.to_wire()
    bundle_wire = bundle_run_provenance_identity.to_wire()
    result: dict[tuple[str, str], GenerationExpectation] = {}
    for role, pointer in sorted(generation_keys):
        tokens = _parse_concrete_pointer(pointer)
        if tokens in scalar_pairs:
            result[(role, pointer)] = scalar_pairs[tokens]
            continue
        if tokens[: len(_PROVENANCE_ROOT_TOKENS)] != _PROVENANCE_ROOT_TOKENS:
            raise AssertionError("unknown generation semantics passed plan validation")
        suffix = tokens[len(_PROVENANCE_ROOT_TOKENS) :]
        result[(role, pointer)] = GenerationExpectation(
            constants_value=_value_at(constants_wire, suffix),
            bundle_value=_value_at(bundle_wire, suffix),
        )
    return result


def _bind_receipt_provenance_roots(
    constants_receipts: Mapping[str, object],
    bundle_receipts: Mapping[str, object],
    *,
    generation_keys: set[tuple[str, str]],
    constants_run_provenance_identity: RunProvenanceIdentity,
    bundle_run_provenance_identity: RunProvenanceIdentity,
) -> None:
    roles = {
        role
        for role, pointer in generation_keys
        if _parse_concrete_pointer(pointer)[: len(_PROVENANCE_ROOT_TOKENS)]
        == _PROVENANCE_ROOT_TOKENS
    }
    constants_wire = constants_run_provenance_identity.to_wire()
    bundle_wire = bundle_run_provenance_identity.to_wire()
    for role in sorted(roles):
        constants_value = _value_at(constants_receipts[role], _PROVENANCE_ROOT_TOKENS)
        bundle_value = _value_at(bundle_receipts[role], _PROVENANCE_ROOT_TOKENS)
        if _json_bytes(
            constants_value,
            location=f"constants_receipts/{role}/run_provenance_identity",
        ) != _json_bytes(
            constants_wire,
            location="constants_run_provenance_identity",
        ):
            raise ArtifactComparisonError(
                f"constants receipt provenance differs from typed identity at {role}"
            )
        if _json_bytes(
            bundle_value,
            location=f"bundle_receipts/{role}/run_provenance_identity",
        ) != _json_bytes(bundle_wire, location="bundle_run_provenance_identity"):
            raise ArtifactComparisonError(
                f"bundle receipt provenance differs from typed identity at {role}"
            )


def _validate_node_reuse_keys(
    values: Mapping[str, str],
    *,
    expected_node_ids: Sequence[str],
    mode: str,
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ArtifactComparisonError(f"{mode}_node_reuse_keys: object required")
    observed: dict[str, str] = {}
    for node_id, value in values.items():
        if not isinstance(node_id, str) or not node_id:
            raise ArtifactComparisonError(
                f"{mode}_node_reuse_keys: non-empty string node ids required"
            )
        _validate_sha256(value, location=f"{mode}_node_reuse_keys/{node_id}")
        observed[node_id] = value
    expected = set(expected_node_ids)
    if set(observed) != expected:
        raise ArtifactComparisonError(
            f"{mode} node reuse key inventory mismatch: "
            f"missing={sorted(expected - set(observed))}, "
            f"extra={sorted(set(observed) - expected)}"
        )
    return {node_id: observed[node_id] for node_id in expected_node_ids}


def _node_reuse_keys_digest(values: Mapping[str, str]) -> str:
    return sha256_json(
        {
            "domain": _NODE_REUSE_KEYS_DOMAIN,
            "node_reuse_keys": dict(values),
        }
    )


def _normalize_prefix(value: object, *, authority_mode: str) -> str | None:
    try:
        return normalize_release_id(value, authority_mode=authority_mode)
    except ValueError:
        return None


def _unmatched_differences(
    constants: object,
    bundle: object,
    *,
    ignored_paths: set[tuple[str, ...]],
    path: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    if path in ignored_paths:
        return ()
    if isinstance(constants, Mapping) and isinstance(bundle, Mapping):
        differences: list[tuple[str, ...]] = []
        for key in sorted(set(constants) | set(bundle)):
            child_path = (*path, key)
            if key not in constants or key not in bundle:
                if child_path not in ignored_paths:
                    differences.append(child_path)
                continue
            differences.extend(
                _unmatched_differences(
                    constants[key],
                    bundle[key],
                    ignored_paths=ignored_paths,
                    path=child_path,
                )
            )
        return tuple(differences)
    if _is_array(constants) and _is_array(bundle):
        if len(constants) != len(bundle):
            return (path,)
        differences = []
        for index, (constants_item, bundle_item) in enumerate(
            zip(constants, bundle, strict=True)
        ):
            differences.extend(
                _unmatched_differences(
                    constants_item,
                    bundle_item,
                    ignored_paths=ignored_paths,
                    path=(*path, str(index)),
                )
            )
        return tuple(differences)
    if _json_bytes(constants, location="constants_receipt_value") != _json_bytes(
        bundle, location="bundle_receipt_value"
    ):
        return (path,)
    return ()


def _expand_pattern(
    value: object,
    tokens: tuple[str, ...],
    *,
    path: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    if not tokens:
        return (path,)
    token, *remaining = tokens
    rest = tuple(remaining)
    if token == "*":
        if isinstance(value, Mapping):
            paths: list[tuple[str, ...]] = []
            for key in sorted(value):
                paths.extend(_expand_pattern(value[key], rest, path=(*path, key)))
            return tuple(paths)
        if _is_array(value):
            paths = []
            for index, child in enumerate(value):
                paths.extend(_expand_pattern(child, rest, path=(*path, str(index))))
            return tuple(paths)
        return ()
    child = _child(value, token)
    if child is _MISSING:
        return ()
    return _expand_pattern(child, rest, path=(*path, token))


def _value_at(value: object, path: tuple[str, ...]) -> object:
    current = value
    for token in path:
        current = _child(current, token)
        if current is _MISSING:
            raise AssertionError("validated concrete receipt path disappeared")
    return current


def _replace_value_at(
    value: object, path: tuple[str, ...], replacement: object
) -> None:
    if not path:  # pragma: no cover - receipt rules cannot name the root
        raise AssertionError("cannot replace the receipt root")
    parent = _value_at(value, path[:-1]) if len(path) > 1 else value
    token = path[-1]
    if isinstance(parent, dict):
        if token not in parent:  # pragma: no cover - expanded paths are concrete
            raise AssertionError("validated concrete receipt path disappeared")
        parent[token] = replacement
        return
    if isinstance(parent, list) and token.isascii() and token.isdigit():
        index = int(token)
        if index < len(parent):
            parent[index] = replacement
            return
    raise AssertionError("validated concrete receipt path is not replaceable")


def _child(value: object, token: str) -> object:
    if isinstance(value, Mapping):
        return value.get(token, _MISSING)
    if _is_array(value) and token.isascii() and token.isdigit():
        if token != "0" and token.startswith("0"):
            return _MISSING
        index = int(token)
        if index < len(value):
            return value[index]
    return _MISSING


def _parse_pointer_pattern(pattern: str, *, location: str) -> tuple[str, ...]:
    tokens = _parse_pointer(pattern, location=location)
    wildcard_count = sum(token == "*" for token in tokens)
    if any("*" in token and token != "*" for token in tokens):
        raise ArtifactComparisonError(f"{location}: wildcard must fill one segment")
    if "**" in pattern:
        raise ArtifactComparisonError(f"{location}: recursive wildcard is forbidden")
    if wildcard_count > 1:
        raise ArtifactComparisonError(f"{location}: multiple wildcards are too broad")
    if wildcard_count and (tokens[0] == "*" or tokens[-1] == "*"):
        raise ArtifactComparisonError(
            f"{location}: root or terminal wildcard is too broad"
        )
    return tokens


def _parse_concrete_pointer(pointer: str) -> tuple[str, ...]:
    tokens = _parse_pointer(pointer, location="generation expectation pointer")
    if any("*" in token for token in tokens):
        raise ArtifactComparisonError(
            "generation expectation pointers must be concrete"
        )
    return tokens


def _parse_pointer(pointer: str, *, location: str) -> tuple[str, ...]:
    if not pointer.startswith("/") or pointer == "/":
        raise ArtifactComparisonError(f"{location}: non-root JSON pointer required")
    encoded_tokens = pointer[1:].split("/")
    if any(not token for token in encoded_tokens):
        raise ArtifactComparisonError(f"{location}: empty JSON pointer segment")
    tokens = tuple(
        _decode_pointer_token(token, location=location) for token in encoded_tokens
    )
    if _encode_pointer(tokens) != pointer:
        raise ArtifactComparisonError(f"{location}: non-canonical JSON pointer")
    return tokens


def _decode_pointer_token(token: str, *, location: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 == len(token) or token[index + 1] not in {"0", "1"}:
            raise ArtifactComparisonError(f"{location}: invalid JSON pointer escape")
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _encode_pointer(tokens: Sequence[str]) -> str:
    return "/" + "/".join(
        token.replace("~", "~0").replace("/", "~1") for token in tokens
    )


def _rule_scopes_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return all(
        left_token == right_token or "*" in {left_token, right_token}
        for left_token, right_token in zip(shorter, longer, strict=False)
    )


def _json_object(value: object, *, location: str) -> dict[str, object]:
    wire = _wire_json(value, location=location)
    if not isinstance(wire, dict):
        raise ArtifactComparisonError(f"{location}: object required")
    return wire


def _json_array(value: object, *, location: str) -> list[object]:
    wire = _wire_json(value, location=location)
    if not isinstance(wire, list):
        raise ArtifactComparisonError(f"{location}: array required")
    return wire


def _string_array(value: object, *, location: str) -> tuple[str, ...]:
    values = _json_array(value, location=location)
    return tuple(
        _nonempty_string(child, location=f"{location}/{index}")
        for index, child in enumerate(values)
    )


def _json_bytes(value: object, *, location: str) -> bytes:
    wire = _wire_json(value, location=location)
    try:
        return canonical_json_bytes(wire)
    except (TypeError, ValueError) as error:
        raise ArtifactComparisonError(f"{location}: canonical JSON required") from error


def _wire_json(value: object, *, location: str) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArtifactComparisonError(f"{location}: string keys required")
            result[key] = _wire_json(child, location=f"{location}/{key}")
        return result
    if _is_array(value):
        return [
            _wire_json(child, location=f"{location}/{index}")
            for index, child in enumerate(value)
        ]
    raise ArtifactComparisonError(
        f"{location}: JSON value required, got {type(value).__name__}"
    )


def _is_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactComparisonError(f"{location}: non-empty string required")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, location: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ArtifactComparisonError(
            f"{location}: keys differ: missing={missing}, extra={extra}"
        )


def _validate_sha256(value: object, *, location: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactComparisonError(f"{location}: lowercase sha256 required")
