"""Plan-lock-driven comparison of dual-mode artifacts and receipts.

The compiler-issued execution ABI is the sole inventory and comparison
authority.  Normative artifacts are compared as raw bytes.  Receipt values
are structurally exact except at paths named by one, and only one, sealed
comparison rule.

This module deliberately accepts already materialized artifact bytes.  File
and directory discovery belongs to the execution ABI's locator and selector
implementation, not to the comparison kernel.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .canonical import canonical_json_bytes, sha256_json
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
_GENERATION_RULE_FIELDS = frozenset(
    {
        "config_authority",
        "identity_generation",
        "run_provenance_identity",
        "spec_binding_status",
    }
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
_CHECKPOINT_RECEIPT_SURFACE_KEYS = frozenset(
    {"present", "canonical", "operational"}
)
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
_MISSING = object()


class ArtifactComparisonError(ValueError):
    """The plan or supplied comparison surfaces fail closed."""


@dataclass(frozen=True, slots=True)
class GenerationExpectation:
    """Expected generation-specific values on both authority sides.

    The comparator binds the current closed generation semantics independently
    of these caller-supplied values.  In particular, bundle provenance must
    equal an explicit typed generation-one identity.  Requiring both values
    prevents an ``expected_to_differ_by_generation`` rule from becoming an
    ignore rule while retaining an exact constants-era oracle value.
    """

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
    constants_normative_sha256: str
    bundle_normative_sha256: str
    constants_receipts_sha256: str
    bundle_receipts_sha256: str
    normative_equal: bool
    receipts_equal_under_plan: bool
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
            "constants_normative_sha256": self.constants_normative_sha256,
            "bundle_normative_sha256": self.bundle_normative_sha256,
            "constants_receipts_sha256": self.constants_receipts_sha256,
            "bundle_receipts_sha256": self.bundle_receipts_sha256,
            "normative_equal": self.normative_equal,
            "receipts_equal_under_plan": self.receipts_equal_under_plan,
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
    generation_expectations: Mapping[tuple[str, str], GenerationExpectation]
    | None = None,
    bundle_run_provenance_identity: RunProvenanceIdentity | None = None,
) -> ArtifactComparisonReceipt:
    """Compare two cold-build output sets under one sealed execution ABI.

    Artifact mappings must contain exactly the ids in
    ``normative_artifact_vector``.  Receipt mappings carry JSON-shaped values
    keyed by artifact role.  Generation expectations are keyed by
    ``(artifact_role, concrete_json_pointer)`` and are mandatory for every
    concrete ``expected_to_differ_by_generation`` rule match.  A sealed
    ``run_provenance_identity`` difference additionally requires the explicit
    typed generation-one bundle identity; caller-authored expectation data is
    never its authority.

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
    ) = _validate_execution_abi(execution_abi)
    constants_bytes = _validate_artifacts(
        constants_artifacts,
        expected=artifact_contracts,
        mode="constants",
    )
    bundle_bytes = _validate_artifacts(
        bundle_artifacts,
        expected=artifact_contracts,
        mode="bundle",
    )

    artifact_rows = tuple(
        _artifact_row(contract, constants_bytes, bundle_bytes)
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
        generation_expectations=(
            {} if generation_expectations is None else generation_expectations
        ),
        bundle_run_provenance_identity=bundle_run_provenance_identity,
        required_receipt_roles=required_receipt_roles,
        checkpoint_receipt_contracts=checkpoint_receipt_contracts,
        execution_abi_sha256=str(abi["sha256"]),
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
    difference_rows = tuple(differences)
    passed = normative_equal and receipts_equal_under_plan
    body = {
        "domain": _COMPARISON_DOMAIN,
        "schema_version": 1,
        "execution_abi_sha256": abi["sha256"],
        "artifact_rows": [row.to_wire() for row in artifact_rows],
        "stage_rows": [row.to_wire() for row in stage_rows],
        "receipt_rows": [row.to_wire() for row in receipt_rows],
        "constants_normative_sha256": constants_normative_sha256,
        "bundle_normative_sha256": bundle_normative_sha256,
        "constants_receipts_sha256": constants_receipts_sha256,
        "bundle_receipts_sha256": bundle_receipts_sha256,
        "normative_equal": normative_equal,
        "receipts_equal_under_plan": receipts_equal_under_plan,
        "passed": passed,
        "differences": [row.to_wire() for row in difference_rows],
    }
    return ArtifactComparisonReceipt(
        execution_abi_sha256=str(abi["sha256"]),
        artifact_rows=artifact_rows,
        stage_rows=stage_rows,
        receipt_rows=receipt_rows,
        constants_normative_sha256=constants_normative_sha256,
        bundle_normative_sha256=bundle_normative_sha256,
        constants_receipts_sha256=constants_receipts_sha256,
        bundle_receipts_sha256=bundle_receipts_sha256,
        normative_equal=normative_equal,
        receipts_equal_under_plan=receipts_equal_under_plan,
        passed=passed,
        differences=difference_rows,
        receipt_sha256=sha256_json(body),
    )


def _validate_execution_abi(
    execution_abi: Mapping[str, object],
) -> tuple[
    dict[str, object],
    tuple[_ArtifactContract, ...],
    tuple[_ReceiptRule, ...],
    frozenset[str],
    tuple[_CheckpointReceiptContract, ...],
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
            and tokens[-1] not in _GENERATION_RULE_FIELDS
        ):
            raise ArtifactComparisonError(
                f"{location}: unknown generation-difference field semantics "
                f"{tokens[-1]!r}"
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
    )


def _validate_artifacts(
    values: Mapping[str, bytes],
    *,
    expected: Sequence[_ArtifactContract],
    mode: str,
) -> dict[str, bytes]:
    if not isinstance(values, Mapping):
        raise ArtifactComparisonError(f"{mode}_artifacts: object required")
    observed: dict[str, bytes] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ArtifactComparisonError(f"{mode}_artifacts: string keys required")
        if not isinstance(value, bytes):
            raise ArtifactComparisonError(f"{mode}_artifacts/{key}: raw bytes required")
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


def _artifact_row(
    contract: _ArtifactContract,
    constants_artifacts: Mapping[str, bytes],
    bundle_artifacts: Mapping[str, bytes],
) -> ArtifactComparisonRow:
    constants_value = constants_artifacts[contract.artifact_id]
    bundle_value = bundle_artifacts[contract.artifact_id]
    return ArtifactComparisonRow(
        artifact_id=contract.artifact_id,
        stage_ref=contract.stage_ref,
        constants_sha256=hashlib.sha256(constants_value).hexdigest(),
        bundle_sha256=hashlib.sha256(bundle_value).hexdigest(),
        constants_byte_size=len(constants_value),
        bundle_byte_size=len(bundle_value),
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
    generation_expectations: Mapping[tuple[str, str], GenerationExpectation],
    bundle_run_provenance_identity: RunProvenanceIdentity | None,
    required_receipt_roles: frozenset[str],
    checkpoint_receipt_contracts: Sequence[_CheckpointReceiptContract],
    execution_abi_sha256: str,
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
    expectations = _validate_generation_expectations(generation_expectations)

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
    if set(expectations) != generation_keys:
        missing = sorted(generation_keys - set(expectations))
        extra = sorted(set(expectations) - generation_keys)
        raise ArtifactComparisonError(
            "generation expectation inventory mismatch: "
            f"missing={missing}, extra={extra}"
        )
    _bind_generation_expectations(
        expectations,
        generation_keys=generation_keys,
        bundle_run_provenance_identity=bundle_run_provenance_identity,
        execution_abi_sha256=execution_abi_sha256,
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
            expected_prefix="populace",
        )
        bundle_normalized = _normalize_prefix(
            bundle_value,
            expected_prefix="microcosm",
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


def _validate_generation_expectations(
    values: Mapping[tuple[str, str], GenerationExpectation],
) -> dict[tuple[str, str], GenerationExpectation]:
    if not isinstance(values, Mapping):
        raise ArtifactComparisonError("generation_expectations: object required")
    validated: dict[tuple[str, str], GenerationExpectation] = {}
    for key, value in values.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(part, str) and part for part in key)
        ):
            raise ArtifactComparisonError(
                "generation expectation keys must be (artifact_role, json_pointer)"
            )
        role, pointer = key
        tokens = _parse_concrete_pointer(pointer)
        canonical_pointer = _encode_pointer(tokens)
        if pointer != canonical_pointer:
            raise ArtifactComparisonError(
                f"generation expectation pointer is not canonical: {pointer!r}"
            )
        if not isinstance(value, GenerationExpectation):
            raise ArtifactComparisonError(
                f"generation expectation {role}:{pointer} has invalid value"
            )
        _json_bytes(
            value.constants_value,
            location=f"generation_expectations/{role}{pointer}/constants",
        )
        _json_bytes(
            value.bundle_value,
            location=f"generation_expectations/{role}{pointer}/bundle",
        )
        validated[(role, pointer)] = value
    return validated


def _bind_generation_expectations(
    expectations: Mapping[tuple[str, str], GenerationExpectation],
    *,
    generation_keys: set[tuple[str, str]],
    bundle_run_provenance_identity: RunProvenanceIdentity | None,
    execution_abi_sha256: str,
) -> None:
    """Bind caller expectations to the closed generation semantics.

    The execution ABI chooses fields, but it cannot turn a caller-authored pair
    into authority.  Scalar transitions are fixed by the generic runtime
    contract.  The generation-one provenance value is issued through the
    typed executor identity and compared in full.
    """

    provenance_keys = {
        key
        for key in generation_keys
        if _parse_concrete_pointer(key[1])[-1] == "run_provenance_identity"
    }
    if provenance_keys:
        if not isinstance(bundle_run_provenance_identity, RunProvenanceIdentity):
            raise ArtifactComparisonError(
                "bundle_run_provenance_identity must be an explicit typed "
                "RunProvenanceIdentity when required by the execution ABI"
            )
        if bundle_run_provenance_identity.identity_generation != 1:
            raise ArtifactComparisonError(
                "bundle_run_provenance_identity must have identity_generation 1"
            )
        provenance_wire = bundle_run_provenance_identity.to_wire()
        binding = provenance_wire["spec_binding"]
        if (
            not isinstance(binding, dict)
            or binding.get("attestation") != "bundle-authoritative"
        ):
            raise ArtifactComparisonError(
                "bundle_run_provenance_identity spec_binding attestation must be "
                "bundle-authoritative"
            )
        authority_versions = provenance_wire["authority_versions"]
        if (
            not isinstance(authority_versions, dict)
            or authority_versions.get("execution_abi") != execution_abi_sha256
        ):
            raise ArtifactComparisonError(
                "bundle_run_provenance_identity authority_versions/execution_abi "
                "differs from the compared execution ABI"
            )
    elif bundle_run_provenance_identity is not None:
        raise ArtifactComparisonError(
            "bundle_run_provenance_identity is unused by the execution ABI"
        )

    scalar_pairs = {
        "config_authority": GenerationExpectation(
            constants_value="constants",
            bundle_value="bundle",
        ),
        "spec_binding_status": GenerationExpectation(
            constants_value="absent",
            bundle_value="resolved",
        ),
        "identity_generation": GenerationExpectation(
            constants_value=0,
            bundle_value=1,
        ),
    }
    for role, pointer in sorted(generation_keys):
        field_name = _parse_concrete_pointer(pointer)[-1]
        expectation = expectations[(role, pointer)]
        required_pair = scalar_pairs.get(field_name)
        if required_pair is not None:
            if not _expectation_equal(expectation, required_pair):
                raise ArtifactComparisonError(
                    "generation expectation differs from the sealed generic "
                    f"transition at {role}:{pointer}"
                )
            continue
        if field_name != "run_provenance_identity":  # pragma: no cover - ABI gate
            raise AssertionError("unknown generation semantics passed plan validation")
        assert bundle_run_provenance_identity is not None
        constants_wire = _wire_json(
            expectation.constants_value,
            location=f"generation_expectations/{role}{pointer}/constants",
        )
        constants_generation = (
            constants_wire.get("identity_generation")
            if isinstance(constants_wire, dict)
            else None
        )
        if (
            not isinstance(constants_wire, dict)
            or isinstance(constants_generation, bool)
            or not isinstance(constants_generation, int)
            or constants_generation != 0
        ):
            raise ArtifactComparisonError(
                "constants run_provenance_identity expectation must have "
                f"identity_generation 0 at {role}:{pointer}"
            )
        required_bundle = bundle_run_provenance_identity.to_wire()
        if _json_bytes(
            expectation.bundle_value,
            location=f"generation_expectations/{role}{pointer}/bundle",
        ) != _json_bytes(
            required_bundle,
            location="bundle_run_provenance_identity",
        ):
            raise ArtifactComparisonError(
                "bundle run_provenance_identity expectation differs from the "
                f"explicit typed identity at {role}:{pointer}"
            )


def _expectation_equal(
    observed: GenerationExpectation,
    required: GenerationExpectation,
) -> bool:
    return _json_bytes(
        observed.constants_value,
        location="generation_expectation/constants",
    ) == _json_bytes(
        required.constants_value,
        location="required_generation_expectation/constants",
    ) and _json_bytes(
        observed.bundle_value,
        location="generation_expectation/bundle",
    ) == _json_bytes(
        required.bundle_value,
        location="required_generation_expectation/bundle",
    )


def _normalize_prefix(value: object, *, expected_prefix: str) -> str | None:
    if not isinstance(value, str):
        return None
    prefix, separator, suffix = value.partition("-")
    if not separator or prefix != expected_prefix or not suffix:
        return None
    return suffix


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
