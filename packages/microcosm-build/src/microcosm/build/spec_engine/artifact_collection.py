"""Fail-closed materialization of a compiler-issued artifact vector.

The comparison kernel deliberately accepts bytes instead of paths.  This
module is the corresponding I/O boundary: production code binds the opaque
``locator_ref`` values issued by the compiler to concrete outputs as those
outputs are constructed, then this collector applies only the selectors
sealed by the execution ABI.

No country, program, producer, or logical-stage name is interpreted here.
Checkpoint receipt sidecars are the one optional filesystem surface.  They
are always read as raw JSON and split by :func:`checkpoint_receipt_surface`;
callers cannot supply an already-normalized receipt wrapper.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .artifact_comparison import checkpoint_receipt_surface
from .canonical import canonical_json_bytes, sha256_json
from .model import FrozenMap, thaw_json

_LOCATOR_GRAMMAR = "closed-runtime-output-plan-and-checkpoint-receipt-v2"
_SUPPORTED_SELECTORS = frozenset(
    {
        "selector:canonical_json_bytes_v1",
        "selector:directory_tree_bytes_v1",
        "selector:file_bytes_v1",
        "selector:h5_all_entity_tables_and_columns_v1",
        "selector:h5_all_weight_vectors_v1",
        "selector:publication_normative_vector_v1",
        "selector:terminal_gate_normative_rows_v1",
    }
)
_JSON_SELECTOR_RECEIPT_ROLES = {
    "selector:publication_normative_vector_v1": "publication_manifest",
    "selector:terminal_gate_normative_rows_v1": "terminal_gates",
}
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
_CODE_ABI_KEYS = frozenset(
    {
        "domain",
        "content_selectors",
        "locator_grammar",
        "receipt_difference_match",
        "implementation_sha256",
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
_CHECKPOINT_KEYS = frozenset(
    {
        "id",
        "ordinal",
        "after_operation",
        "covers_operations",
        "operational_receipts_sidecar",
        "artifact_roles",
    }
)
_RULE_KEYS = frozenset({"artifact_role", "json_pointer_pattern", "rule", "category"})
_RECEIPT_RULES = frozenset(
    {
        "equal_after_normalizing_prefix",
        "expected_to_differ_by_generation",
        "operational_excluded",
    }
)
_MISSING = object()


class ArtifactCollectionError(ValueError):
    """The execution ABI, locator registry, or selected output is invalid."""


class LocatorSourceKind(StrEnum):
    """Physical shape supplied for one opaque locator reference."""

    FILE = "file"
    DIRECTORY = "directory"
    OPTIONAL_FILE = "optional_file"
    JSON_VALUE = "json_value"


@dataclass(frozen=True, slots=True)
class LocatorBinding:
    """One immutable construction-time binding for an opaque locator."""

    locator_ref: str
    kind: LocatorSourceKind
    path: Path | None = None
    canonical_json: bytes | None = None


class ArtifactLocatorRegistry:
    """Duplicate-refusing registry bounded by explicit filesystem roots."""

    def __init__(self, *, allowed_roots: Sequence[str | Path]) -> None:
        roots: list[Path] = []
        for value in allowed_roots:
            root = _absolute_path(value)
            if root == Path(root.anchor):
                raise ArtifactCollectionError(
                    "allowed locator root cannot be a filesystem root"
                )
            if root in roots:
                raise ArtifactCollectionError(f"duplicate allowed root: {root}")
            roots.append(root)
        if not roots:
            raise ArtifactCollectionError("at least one allowed locator root required")
        self._roots = tuple(roots)
        self._bindings: dict[str, LocatorBinding] = {}

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return self._roots

    def bind_file(self, locator_ref: str, path: str | Path) -> None:
        self._bind_path(locator_ref, path, LocatorSourceKind.FILE)

    def bind_directory(self, locator_ref: str, path: str | Path) -> None:
        self._bind_path(locator_ref, path, LocatorSourceKind.DIRECTORY)

    def bind_optional_file(self, locator_ref: str, path: str | Path) -> None:
        self._bind_path(locator_ref, path, LocatorSourceKind.OPTIONAL_FILE)

    def bind_json(self, locator_ref: str, value: object) -> None:
        ref = _nonempty_string(locator_ref, location="locator_ref")
        self._refuse_duplicate(ref)
        try:
            payload = canonical_json_bytes(_json_clone(value, location=ref))
        except (TypeError, ValueError) as error:
            raise ArtifactCollectionError(
                f"locator {ref!r}: canonical JSON value required"
            ) from error
        self._bindings[ref] = LocatorBinding(
            locator_ref=ref,
            kind=LocatorSourceKind.JSON_VALUE,
            canonical_json=payload,
        )

    def _bind_path(
        self,
        locator_ref: str,
        path: str | Path,
        kind: LocatorSourceKind,
    ) -> None:
        ref = _nonempty_string(locator_ref, location="locator_ref")
        self._refuse_duplicate(ref)
        self._bindings[ref] = LocatorBinding(
            locator_ref=ref,
            kind=kind,
            path=_absolute_path(path),
        )

    def _refuse_duplicate(self, locator_ref: str) -> None:
        if locator_ref in self._bindings:
            raise ArtifactCollectionError(f"duplicate locator binding: {locator_ref!r}")

    def snapshot(self) -> Mapping[str, LocatorBinding]:
        """Return an immutable copy; later binds cannot change a collection."""

        return MappingProxyType(dict(self._bindings))


@dataclass(frozen=True, slots=True)
class CollectedArtifactSurfaces:
    """Normative artifact bytes and complete structural receipt surfaces."""

    artifacts: Mapping[str, bytes]
    receipts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ArtifactRow:
    artifact_id: str
    locator_ref: str
    selector_ref: str
    protocol_ref: str


@dataclass(frozen=True, slots=True)
class _CheckpointRow:
    checkpoint_id: str
    receipts_policy: str
    payload_role: str
    manifest_role: str
    receipts_role: str


@dataclass(frozen=True, slots=True)
class _ReceiptRule:
    artifact_role: str
    tokens: tuple[str, ...]
    rule: str


@dataclass(frozen=True, slots=True)
class _CollectionPlan:
    artifacts: tuple[_ArtifactRow, ...]
    checkpoints: tuple[_CheckpointRow, ...]
    rules: tuple[_ReceiptRule, ...]
    required_locators: frozenset[str]


def collect_artifact_surfaces(
    execution_abi: Mapping[str, object],
    *,
    registry: ArtifactLocatorRegistry,
    authority_mode: str,
) -> CollectedArtifactSurfaces:
    """Materialize every sealed artifact and receipt from exact locator binds.

    ``authority_mode`` is used only to validate the approved publication-line
    prefix while producing normalized JSON selector bytes.  The unmodified
    JSON documents remain in ``receipts`` for structural comparison.
    """

    if not isinstance(registry, ArtifactLocatorRegistry):
        raise TypeError("registry must be ArtifactLocatorRegistry")
    if authority_mode not in {"constants", "bundle"}:
        raise ArtifactCollectionError("authority_mode must be 'constants' or 'bundle'")
    abi = _json_object(execution_abi, location="execution_abi")
    plan = _compile_collection_plan(abi)
    bindings = registry.snapshot()
    observed = frozenset(bindings)
    if observed != plan.required_locators:
        raise ArtifactCollectionError(
            "locator inventory mismatch: "
            f"missing={sorted(plan.required_locators - observed)}, "
            f"extra={sorted(observed - plan.required_locators)}"
        )

    for binding in bindings.values():
        _validate_binding(binding, allowed_roots=registry.allowed_roots)

    documents: dict[str, dict[str, object]] = {}
    artifacts: dict[str, bytes] = {}
    receipt_role_sources: dict[str, str] = {}
    for row in plan.artifacts:
        binding = bindings[row.locator_ref]
        if row.selector_ref in _JSON_SELECTOR_RECEIPT_ROLES:
            role = _JSON_SELECTOR_RECEIPT_ROLES[row.selector_ref]
            previous = receipt_role_sources.setdefault(role, row.locator_ref)
            if previous != row.locator_ref:
                raise ArtifactCollectionError(
                    f"receipt role {role!r} resolves through multiple locators"
                )
            document = documents.setdefault(
                role,
                _read_json_binding(binding, location=row.locator_ref),
            )
            artifacts[row.artifact_id] = _normalized_document_bytes(
                document,
                role=role,
                rules=plan.rules,
                authority_mode=authority_mode,
            )
        else:
            artifacts[row.artifact_id] = _apply_selector(
                row.selector_ref,
                binding,
            )
        _authenticate_plan_value(row, artifacts[row.artifact_id], abi=abi)

    receipts: dict[str, object] = dict(documents)
    for checkpoint in plan.checkpoints:
        payload_binding = bindings[checkpoint.payload_role]
        manifest_binding = bindings[checkpoint.manifest_role]
        sidecar_binding = bindings[checkpoint.receipts_role]
        payload = artifacts[checkpoint.payload_role]
        manifest = _read_json_binding(
            manifest_binding,
            location=checkpoint.manifest_role,
        )
        receipts[checkpoint.manifest_role] = manifest
        sidecar_present = _optional_file_present(sidecar_binding)
        expected_present = checkpoint.receipts_policy == "required"
        if sidecar_present is not expected_present:
            expectation = "present" if expected_present else "absent"
            raise ArtifactCollectionError(
                f"{checkpoint.receipts_role}: sidecar must be {expectation} "
                "under the sealed plan"
            )
        raw_sidecar = (
            _read_json_binding(
                sidecar_binding,
                location=checkpoint.receipts_role,
            )
            if sidecar_present
            else None
        )
        try:
            surface = checkpoint_receipt_surface(raw_sidecar)
        except ValueError as error:
            raise ArtifactCollectionError(
                f"{checkpoint.receipts_role}: raw checkpoint receipt sidecar "
                f"is invalid: {error}"
            ) from error
        _cross_authenticate_checkpoint(
            checkpoint,
            payload_binding=payload_binding,
            payload=payload,
            manifest=manifest,
            receipt_surface=surface,
        )
        receipts[checkpoint.receipts_role] = surface

    expected_receipt_roles = {rule.artifact_role for rule in plan.rules} | {
        role
        for checkpoint in plan.checkpoints
        for role in (checkpoint.manifest_role, checkpoint.receipts_role)
    }
    if set(receipts) != expected_receipt_roles:
        raise ArtifactCollectionError(
            "receipt role inventory cannot be resolved from sealed selectors: "
            f"missing={sorted(expected_receipt_roles - set(receipts))}, "
            f"extra={sorted(set(receipts) - expected_receipt_roles)}"
        )
    return CollectedArtifactSurfaces(
        artifacts=MappingProxyType(artifacts),
        receipts=MappingProxyType(receipts),
    )


def _compile_collection_plan(abi: dict[str, object]) -> _CollectionPlan:
    if set(abi) != _EXECUTION_ABI_KEYS:
        raise ArtifactCollectionError("execution_abi keys changed")
    if abi.get("schema_version") != 1 or abi.get("present") is not True:
        raise ArtifactCollectionError("execution_abi must be present schema version 1")
    seal = abi.get("sha256")
    if not _is_sha256(seal):
        raise ArtifactCollectionError("execution_abi/sha256: lowercase sha256 required")
    unsigned = {key: value for key, value in abi.items() if key != "sha256"}
    if sha256_json(unsigned) != seal:
        raise ArtifactCollectionError("execution_abi seal mismatch")

    code_abi = _mapping(abi.get("code_abi"), location="execution_abi/code_abi")
    if set(code_abi) != _CODE_ABI_KEYS:
        raise ArtifactCollectionError("execution_abi/code_abi keys changed")
    if code_abi.get("locator_grammar") != _LOCATOR_GRAMMAR:
        raise ArtifactCollectionError("unsupported execution ABI locator grammar")
    if code_abi.get("receipt_difference_match") != "exactly_one_sealed_rule":
        raise ArtifactCollectionError(
            "execution_abi/code_abi must require exactly one sealed receipt rule"
        )
    implementation = code_abi.get("implementation_sha256")
    if not _is_sha256(implementation):
        raise ArtifactCollectionError("execution_abi/code_abi seal is invalid")
    code_unsigned = {
        key: value for key, value in code_abi.items() if key != "implementation_sha256"
    }
    if sha256_json(code_unsigned) != implementation:
        raise ArtifactCollectionError("execution_abi/code_abi seal mismatch")
    declared_selectors = _string_sequence(
        code_abi.get("content_selectors"),
        location="execution_abi/code_abi/content_selectors",
    )
    if len(declared_selectors) != len(set(declared_selectors)):
        raise ArtifactCollectionError("content selector inventory contains duplicates")

    artifact_rows: list[_ArtifactRow] = []
    artifact_ids: set[str] = set()
    used_selectors: set[str] = set()
    locator_selectors: dict[str, set[str]] = {}
    for index, raw in enumerate(
        _sequence(
            abi.get("normative_artifact_vector"),
            location="execution_abi/normative_artifact_vector",
        )
    ):
        location = f"execution_abi/normative_artifact_vector/{index}"
        row = _mapping(raw, location=location)
        if set(row) != _ARTIFACT_ROW_KEYS:
            raise ArtifactCollectionError(f"{location}: artifact row keys changed")
        artifact_id = _nonempty_string(row.get("id"), location=f"{location}/id")
        if artifact_id in artifact_ids:
            raise ArtifactCollectionError(f"duplicate artifact id {artifact_id!r}")
        artifact_ids.add(artifact_id)
        if (
            row.get("surface") != "normative"
            or row.get("comparison") != "raw_byte_exact"
            or row.get("required") is not True
        ):
            raise ArtifactCollectionError(
                f"{location}: required normative raw-byte artifact expected"
            )
        locator = _nonempty_string(
            row.get("locator_ref"), location=f"{location}/locator_ref"
        )
        for field in ("kind", "producer_ref", "stage_ref", "protocol_ref"):
            _nonempty_string(row.get(field), location=f"{location}/{field}")
        selector = _nonempty_string(
            row.get("content_selector_ref"),
            location=f"{location}/content_selector_ref",
        )
        if selector not in _SUPPORTED_SELECTORS:
            raise ArtifactCollectionError(f"unsupported selector {selector!r}")
        used_selectors.add(selector)
        locator_selectors.setdefault(locator, set()).add(selector)
        artifact_rows.append(
            _ArtifactRow(
                artifact_id,
                locator,
                selector,
                str(row["protocol_ref"]),
            )
        )
    if not artifact_rows:
        raise ArtifactCollectionError("normative artifact vector must not be empty")
    if set(declared_selectors) != used_selectors:
        raise ArtifactCollectionError(
            "declared selector inventory differs from artifact selector inventory"
        )
    for locator, selectors in locator_selectors.items():
        if len(selectors) > 1 and selectors != {
            "selector:h5_all_entity_tables_and_columns_v1",
            "selector:h5_all_weight_vectors_v1",
        }:
            raise ArtifactCollectionError(
                f"locator {locator!r} is reused by incompatible selectors"
            )

    checkpoints: list[_CheckpointRow] = []
    required_locators = set(locator_selectors)
    checkpoint_ids: set[str] = set()
    for index, raw in enumerate(
        _sequence(
            abi.get("durable_checkpoints"),
            location="execution_abi/durable_checkpoints",
        )
    ):
        location = f"execution_abi/durable_checkpoints/{index}"
        row = _mapping(raw, location=location)
        if set(row) != _CHECKPOINT_KEYS:
            raise ArtifactCollectionError(f"{location}: checkpoint row keys changed")
        checkpoint_id = _nonempty_string(row.get("id"), location=f"{location}/id")
        if checkpoint_id in checkpoint_ids:
            raise ArtifactCollectionError(
                f"duplicate durable checkpoint id {checkpoint_id!r}"
            )
        checkpoint_ids.add(checkpoint_id)
        policy = _nonempty_string(
            row.get("operational_receipts_sidecar"),
            location=f"{location}/operational_receipts_sidecar",
        )
        if policy not in {"forbidden", "required"}:
            raise ArtifactCollectionError(
                f"{location}: forbidden or required receipt policy expected"
            )
        roles = _string_sequence(
            row.get("artifact_roles"), location=f"{location}/artifact_roles"
        )
        expected_roles = (
            f"checkpoint:{checkpoint_id}:payload",
            f"checkpoint:{checkpoint_id}:manifest",
            f"checkpoint:{checkpoint_id}:receipts",
        )
        if tuple(roles) != expected_roles:
            raise ArtifactCollectionError(
                f"{location}: checkpoint artifact role grammar changed"
            )
        if expected_roles[0] not in artifact_ids:
            raise ArtifactCollectionError(
                f"{location}: checkpoint payload is absent from normative vector"
            )
        required_locators.update(expected_roles[1:])
        checkpoints.append(_CheckpointRow(checkpoint_id, policy, *expected_roles))

    rules: list[_ReceiptRule] = []
    seen_rule_scopes: set[tuple[str, tuple[str, ...]]] = set()
    for index, raw in enumerate(
        _sequence(
            abi.get("receipt_comparison_vector"),
            location="execution_abi/receipt_comparison_vector",
        )
    ):
        location = f"execution_abi/receipt_comparison_vector/{index}"
        row = _mapping(raw, location=location)
        if set(row) != _RULE_KEYS:
            raise ArtifactCollectionError(f"{location}: receipt rule keys changed")
        role = _nonempty_string(
            row.get("artifact_role"), location=f"{location}/artifact_role"
        )
        tokens = _parse_pointer(
            _nonempty_string(
                row.get("json_pointer_pattern"),
                location=f"{location}/json_pointer_pattern",
            ),
            location=location,
        )
        rule = _nonempty_string(row.get("rule"), location=f"{location}/rule")
        if rule not in _RECEIPT_RULES:
            raise ArtifactCollectionError(f"{location}: unknown receipt rule")
        _nonempty_string(row.get("category"), location=f"{location}/category")
        marker = (role, tokens)
        if marker in seen_rule_scopes:
            raise ArtifactCollectionError(f"{location}: duplicate receipt rule")
        if any(
            previous.artifact_role == role
            and _rule_scopes_overlap(previous.tokens, tokens)
            for previous in rules
        ):
            raise ArtifactCollectionError(f"{location}: overlapping receipt rule")
        seen_rule_scopes.add(marker)
        rules.append(_ReceiptRule(role, tokens, rule))

    return _CollectionPlan(
        artifacts=tuple(artifact_rows),
        checkpoints=tuple(checkpoints),
        rules=tuple(rules),
        required_locators=frozenset(required_locators),
    )


def _apply_selector(selector: str, binding: LocatorBinding) -> bytes:
    if selector == "selector:canonical_json_bytes_v1":
        if binding.kind is not LocatorSourceKind.JSON_VALUE:
            raise ArtifactCollectionError(
                f"{binding.locator_ref}: JSON-value binding required"
            )
        assert binding.canonical_json is not None
        return binding.canonical_json
    if selector == "selector:file_bytes_v1":
        _require_kind(binding, LocatorSourceKind.FILE)
        assert binding.path is not None
        return _read_regular_file(binding.path)
    if selector == "selector:directory_tree_bytes_v1":
        _require_kind(binding, LocatorSourceKind.DIRECTORY)
        assert binding.path is not None
        return _directory_tree_bytes(binding.path)
    if selector in {
        "selector:h5_all_entity_tables_and_columns_v1",
        "selector:h5_all_weight_vectors_v1",
    }:
        _require_kind(binding, LocatorSourceKind.FILE)
        assert binding.path is not None
        return _h5_logical_bytes(
            binding.path,
            weights=selector == "selector:h5_all_weight_vectors_v1",
        )
    raise ArtifactCollectionError(f"selector {selector!r} requires a JSON document")


def _authenticate_plan_value(
    row: _ArtifactRow,
    payload: bytes,
    *,
    abi: Mapping[str, object],
) -> None:
    """Bind in-memory plan components to a digest carried by execution ABI.

    The current grammar exposes one plan-lock component.  Future components
    must add an equally explicit sealed digest; accepting an unbound JSON value
    would let a construction-time caller silently replace compiler output.
    """

    if not row.locator_ref.startswith("plan_lock:/"):
        return
    if row.protocol_ref != "plan_lock:seed_stream_map":
        raise ArtifactCollectionError(
            f"{row.locator_ref}: plan-lock protocol has no sealed collector binding"
        )
    pipeline = _mapping(abi.get("pipeline"), location="execution_abi/pipeline")
    expected = pipeline.get("seed_stream_map_sha256")
    if not _is_sha256(expected):
        raise ArtifactCollectionError(
            "execution_abi/pipeline/seed_stream_map_sha256 is invalid"
        )
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ArtifactCollectionError(
            f"{row.locator_ref}: bound plan component digest differs from execution ABI"
        )


def _normalized_document_bytes(
    document: Mapping[str, object],
    *,
    role: str,
    rules: Sequence[_ReceiptRule],
    authority_mode: str,
) -> bytes:
    value = copy.deepcopy(dict(document))
    selected = [rule for rule in rules if rule.artifact_role == role]
    if not selected:
        raise ArtifactCollectionError(
            f"normalized JSON selector role {role!r} has no sealed rules"
        )
    mutations: list[tuple[tuple[str, ...], str]] = []
    for rule in selected:
        paths = _expand_pointer(value, rule.tokens)
        if not paths:
            raise ArtifactCollectionError(
                f"receipt rule {role}:{_encode_pointer(rule.tokens)} matched no field"
            )
        mutations.extend((path, rule.rule) for path in paths)
    concrete = [path for path, _rule in mutations]
    if len(concrete) != len(set(concrete)):
        raise ArtifactCollectionError(
            f"receipt rules for {role!r} overlap after wildcard expansion"
        )
    for path, rule in sorted(mutations, key=lambda row: row[0], reverse=True):
        parent, token = _pointer_parent(value, path)
        if not isinstance(parent, dict):
            raise ArtifactCollectionError(
                "normalized JSON selectors only exclude or replace object fields"
            )
        observed = parent[token]
        if rule == "operational_excluded":
            del parent[token]
        elif rule == "expected_to_differ_by_generation":
            parent[token] = {"comparison_rule": rule}
        else:
            expected = "populace" if authority_mode == "constants" else "microcosm"
            if not isinstance(observed, str):
                raise ArtifactCollectionError(
                    f"{role}:{_encode_pointer(path)}: prefixed string required"
                )
            prefix, separator, suffix = observed.partition("-")
            if not separator or prefix != expected or not suffix:
                raise ArtifactCollectionError(
                    f"{role}:{_encode_pointer(path)}: expected {expected!r} prefix"
                )
            parent[token] = suffix
    return canonical_json_bytes(value)


def _cross_authenticate_checkpoint(
    checkpoint: _CheckpointRow,
    *,
    payload_binding: LocatorBinding,
    payload: bytes,
    manifest: Mapping[str, object],
    receipt_surface: Mapping[str, object],
) -> None:
    if payload_binding.path is None:  # pragma: no cover - selector gate
        raise AssertionError("checkpoint payload has no path")
    if manifest.get("stage") != checkpoint.checkpoint_id:
        raise ArtifactCollectionError(
            f"{checkpoint.manifest_role}: stage differs from sealed checkpoint id"
        )
    binding = _mapping(
        manifest.get("checkpoint"),
        location=f"{checkpoint.manifest_role}/checkpoint",
    )
    expected = {
        "filename": payload_binding.path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if binding != expected:
        raise ArtifactCollectionError(
            f"{checkpoint.manifest_role}: payload binding differs from actual file"
        )
    if receipt_surface.get("present") is not True:
        return
    canonical = _mapping(
        receipt_surface.get("canonical"),
        location=f"{checkpoint.receipts_role}/canonical",
    )
    for field in ("schema_version", "materializer_version", "stage", "identity_sha256"):
        if canonical.get(field) != manifest.get(field):
            raise ArtifactCollectionError(
                f"{checkpoint.receipts_role}: {field} differs from manifest"
            )
    if canonical.get("checkpoint") != expected:
        raise ArtifactCollectionError(
            f"{checkpoint.receipts_role}: payload binding differs from actual file"
        )


def _validate_binding(
    binding: LocatorBinding,
    *,
    allowed_roots: Sequence[Path],
) -> None:
    if binding.kind is LocatorSourceKind.JSON_VALUE:
        if binding.path is not None or binding.canonical_json is None:
            raise ArtifactCollectionError(
                f"{binding.locator_ref}: malformed JSON-value binding"
            )
        return
    if binding.path is None or binding.canonical_json is not None:
        raise ArtifactCollectionError(f"{binding.locator_ref}: malformed path binding")
    containing = [root for root in allowed_roots if binding.path.is_relative_to(root)]
    if not containing:
        raise ArtifactCollectionError(
            f"{binding.locator_ref}: path escapes allowed roots: {binding.path}"
        )
    root = max(containing, key=lambda item: len(item.parts))
    _validate_path_chain(
        root, binding.path, optional=binding.kind is LocatorSourceKind.OPTIONAL_FILE
    )
    if binding.kind is LocatorSourceKind.DIRECTORY and not binding.path.is_dir():
        raise ArtifactCollectionError(f"{binding.locator_ref}: directory required")
    if binding.kind is LocatorSourceKind.FILE and not binding.path.is_file():
        raise ArtifactCollectionError(f"{binding.locator_ref}: regular file required")


def _validate_path_chain(root: Path, target: Path, *, optional: bool) -> None:
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        _require_nonsymlink_directory(current)
    relative = target.relative_to(root)
    for index, part in enumerate(relative.parts):
        current /= part
        is_target = index == len(relative.parts) - 1
        try:
            status = os.lstat(current)
        except FileNotFoundError as error:
            if optional and is_target:
                return
            raise ArtifactCollectionError(
                f"required locator path is missing: {current}"
            ) from error
        if stat.S_ISLNK(status.st_mode):
            raise ArtifactCollectionError(
                f"symlink locator path is forbidden: {current}"
            )
        if not is_target and not stat.S_ISDIR(status.st_mode):
            raise ArtifactCollectionError(
                f"locator parent is not a directory: {current}"
            )
        if is_target and not (
            stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)
        ):
            raise ArtifactCollectionError(
                f"special locator file is forbidden: {current}"
            )


def _require_nonsymlink_directory(path: Path) -> None:
    try:
        status = os.lstat(path)
    except FileNotFoundError as error:
        raise ArtifactCollectionError(f"allowed root is missing: {path}") from error
    if stat.S_ISLNK(status.st_mode):
        raise ArtifactCollectionError(f"symlink path is forbidden: {path}")
    if not stat.S_ISDIR(status.st_mode):
        raise ArtifactCollectionError(f"allowed root parent is not a directory: {path}")


def _optional_file_present(binding: LocatorBinding) -> bool:
    _require_kind(binding, LocatorSourceKind.OPTIONAL_FILE)
    assert binding.path is not None
    return binding.path.exists()


def _read_json_binding(
    binding: LocatorBinding,
    *,
    location: str,
) -> dict[str, object]:
    if binding.kind not in {LocatorSourceKind.FILE, LocatorSourceKind.OPTIONAL_FILE}:
        raise ArtifactCollectionError(f"{location}: JSON file binding required")
    assert binding.path is not None
    raw = _read_regular_file(binding.path)
    try:
        return _strict_json_object(raw)
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise ArtifactCollectionError(
            f"{location}: strict JSON object required"
        ) from error


def _strict_json_object(raw: bytes) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _read_regular_file(path: Path) -> bytes:
    descriptor = _open_regular_file(path)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, mode="rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require_unchanged_file(path, before=before, after=after)
    return payload


def _regular_file_sha256(path: Path) -> bytes:
    descriptor = _open_regular_file(path)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require_unchanged_file(path, before=before, after=after)
    return digest.digest()


def _open_regular_file(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactCollectionError(
            f"unable to open regular file {path}: {error}"
        ) from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ArtifactCollectionError(f"regular file required: {path}")
    return descriptor


def _require_unchanged_file(
    path: Path,
    *,
    before: os.stat_result,
    after: os.stat_result,
) -> None:
    observed = os.lstat(path)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_path = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_path:
        raise ArtifactCollectionError(f"file changed while collecting: {path}")


def _directory_tree_bytes(root: Path) -> bytes:
    entries: list[tuple[bytes, bool, Path]] = []

    def visit(directory: Path, prefix: str) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda row: row.name)
        except OSError as error:
            raise ArtifactCollectionError(
                f"unable to scan artifact directory {directory}: {error}"
            ) from error
        for child in children:
            relative = f"{prefix}/{child.name}" if prefix else child.name
            try:
                encoded = relative.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ArtifactCollectionError(
                    f"artifact path is not UTF-8 encodable: {relative!r}"
                ) from error
            if child.is_symlink():
                raise ArtifactCollectionError(
                    f"symlink in artifact directory is forbidden: {child.path}"
                )
            if child.is_dir(follow_symlinks=False):
                entries.append((encoded, True, Path(child.path)))
                visit(Path(child.path), relative)
            elif child.is_file(follow_symlinks=False):
                entries.append((encoded, False, Path(child.path)))
            else:
                raise ArtifactCollectionError(
                    f"special file in artifact directory is forbidden: {child.path}"
                )

    visit(root, "")
    entries.sort(key=lambda row: row[0])
    output = bytearray(b"microcosm.selector.directory-tree-bytes.v1\0")
    for relative, is_directory, path in entries:
        output.extend(b"D" if is_directory else b"F")
        _append_frame(output, relative)
        if not is_directory:
            _append_frame(output, _read_regular_file(path))
    return bytes(output)


def _h5_logical_bytes(path: Path, *, weights: bool) -> bytes:
    initial_sha256 = _regular_file_sha256(path)
    domain = (
        b"microcosm.selector.h5-all-weight-vectors.v1\0"
        if weights
        else b"microcosm.selector.h5-all-entity-tables-and-columns.v1\0"
    )
    output = bytearray(domain)
    selected = 0
    try:
        with pd.HDFStore(path, mode="r") as store:
            for key in sorted(store.keys()):
                value = store[key]
                if not isinstance(value, pd.DataFrame):
                    continue
                entity = key.rsplit("/", 1)[-1]
                weight_name = f"{entity}_weight"
                if weights:
                    columns = [weight_name] if weight_name in value.columns else []
                else:
                    columns = [
                        column for column in value.columns if column != weight_name
                    ]
                if not columns and weights:
                    continue
                _encode_table(output, key, value, columns=columns)
                selected += 1
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactCollectionError):
            raise
        raise ArtifactCollectionError(
            f"unable to select logical H5 content: {path}"
        ) from error
    if selected == 0:
        surface = "weight vector" if weights else "entity table"
        raise ArtifactCollectionError(f"H5 contains no {surface} surface: {path}")
    if _regular_file_sha256(path) != initial_sha256:
        raise ArtifactCollectionError(f"H5 changed while collecting: {path}")
    return bytes(output)


def _encode_table(
    output: bytearray,
    key: str,
    table: pd.DataFrame,
    *,
    columns: Sequence[object],
) -> None:
    if len(set(table.columns)) != len(table.columns):
        raise ArtifactCollectionError(f"H5 table {key!r} has duplicate columns")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ArtifactCollectionError(f"H5 table {key!r} requires string columns")
    output.extend(b"T")
    _append_frame(output, key.encode("utf-8"))
    _encode_index(output, table.index)
    output.extend(struct.pack(">Q", len(columns)))
    for column in columns:
        assert isinstance(column, str)
        series = table[column]
        output.extend(b"C")
        _append_frame(output, column.encode("utf-8"))
        _append_frame(output, canonical_json_bytes(_dtype_descriptor(series.dtype)))
        output.extend(struct.pack(">Q", len(series)))
        for value in series.array:
            _encode_scalar(output, value)


def _encode_index(output: bytearray, index: pd.Index) -> None:
    output.extend(b"I")
    if isinstance(index, pd.MultiIndex):
        descriptor: object = {
            "kind": "multi_index",
            "dtypes": [str(dtype) for dtype in index.dtypes],
            "names": list(index.names),
        }
    else:
        descriptor = {
            "kind": type(index).__name__,
            "dtype": str(index.dtype),
            "name": index.name,
        }
    _append_frame(output, canonical_json_bytes(descriptor))
    output.extend(struct.pack(">Q", len(index)))
    for value in index.tolist():
        _encode_scalar(output, value)


def _dtype_descriptor(dtype: object) -> object:
    if isinstance(dtype, pd.CategoricalDtype):
        categories = bytearray()
        for value in dtype.categories.tolist():
            _encode_scalar(categories, value)
        return {
            "kind": "category",
            "ordered": dtype.ordered,
            "categories_sha256": hashlib.sha256(categories).hexdigest(),
            "categories_count": len(dtype.categories),
        }
    return {"kind": type(dtype).__name__, "name": str(dtype)}


def _encode_scalar(output: bytearray, value: object) -> None:
    if value is None or value is pd.NA or value is pd.NaT:
        output.extend(b"N")
        return
    if isinstance(value, np.generic):
        if np.issubdtype(type(value), np.datetime64) and np.isnat(value):
            output.extend(b"N")
            return
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        output.extend(b"N")
    elif isinstance(value, (bool, np.bool_)):
        output.extend(b"B1" if bool(value) else b"B0")
    elif isinstance(value, int) and not isinstance(value, bool):
        output.extend(b"Z")
        _append_frame(output, str(value).encode("ascii"))
    elif isinstance(value, float):
        output.extend(b"R")
        output.extend(struct.pack(">d", value))
    elif isinstance(value, str):
        output.extend(b"S")
        _append_frame(output, value.encode("utf-8"))
    elif isinstance(value, bytes):
        output.extend(b"Y")
        _append_frame(output, value)
    elif isinstance(value, pd.Timestamp):
        output.extend(b"P")
        _append_frame(output, str(value.tz).encode("utf-8") if value.tz else b"")
        output.extend(struct.pack(">q", value.value))
    elif isinstance(value, pd.Timedelta):
        output.extend(b"L")
        output.extend(struct.pack(">q", value.value))
    elif isinstance(value, tuple):
        output.extend(b"Q")
        output.extend(struct.pack(">Q", len(value)))
        for child in value:
            _encode_scalar(output, child)
    else:
        raise ArtifactCollectionError(
            f"unsupported logical H5 scalar type: {type(value).__name__}"
        )


def _append_frame(output: bytearray, payload: bytes) -> None:
    output.extend(struct.pack(">Q", len(payload)))
    output.extend(payload)


def _require_kind(binding: LocatorBinding, expected: LocatorSourceKind) -> None:
    if binding.kind is not expected:
        raise ArtifactCollectionError(
            f"{binding.locator_ref}: {expected.value} binding required, "
            f"got {binding.kind.value}"
        )


def _parse_pointer(pointer: str, *, location: str) -> tuple[str, ...]:
    if not pointer.startswith("/"):
        raise ArtifactCollectionError(f"{location}: non-root JSON pointer required")
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        if not raw:
            raise ArtifactCollectionError(f"{location}: empty JSON pointer segment")
        token = ""
        index = 0
        while index < len(raw):
            if raw[index] != "~":
                token += raw[index]
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise ArtifactCollectionError(
                    f"{location}: invalid JSON pointer escape"
                )
            token += "~" if raw[index + 1] == "0" else "/"
            index += 2
        if token == "**":
            raise ArtifactCollectionError(f"{location}: recursive wildcard forbidden")
        tokens.append(token)
    if tokens.count("*") > 1:
        raise ArtifactCollectionError(f"{location}: multiple wildcards forbidden")
    return tuple(tokens)


def _expand_pointer(
    value: object,
    tokens: tuple[str, ...],
    *,
    path: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    if not tokens:
        return (path,)
    token, *tail = tokens
    remaining = tuple(tail)
    if token == "*":
        if isinstance(value, Mapping):
            return tuple(
                concrete
                for key in sorted(value)
                for concrete in _expand_pointer(
                    value[key], remaining, path=(*path, key)
                )
            )
        if isinstance(value, list):
            return tuple(
                concrete
                for index, child in enumerate(value)
                for concrete in _expand_pointer(
                    child, remaining, path=(*path, str(index))
                )
            )
        return ()
    child = _pointer_child(value, token)
    if child is _MISSING:
        return ()
    return _expand_pointer(child, remaining, path=(*path, token))


def _pointer_parent(value: object, path: tuple[str, ...]) -> tuple[object, str]:
    current = value
    for token in path[:-1]:
        current = _pointer_child(current, token)
        if current is _MISSING:  # pragma: no cover - expansion established path
            raise AssertionError("expanded JSON pointer disappeared")
    return current, path[-1]


def _pointer_child(value: object, token: str) -> object:
    if isinstance(value, Mapping):
        return value.get(token, _MISSING)
    if isinstance(value, list) and token.isascii() and token.isdigit():
        index = int(token)
        if str(index) == token and index < len(value):
            return value[index]
    return _MISSING


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


def _absolute_path(value: str | Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as error:
        raise ArtifactCollectionError(f"invalid locator path: {value!r}") from error


def _json_clone(value: object, *, location: str) -> object:
    if isinstance(value, FrozenMap):
        value = thaw_json(value)
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactCollectionError(
            f"{location}: canonical JSON value required"
        ) from error


def _json_object(value: object, *, location: str) -> dict[str, object]:
    cloned = _json_clone(value, location=location)
    if not isinstance(cloned, dict):
        raise ArtifactCollectionError(f"{location}: object required")
    return cloned


def _mapping(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactCollectionError(f"{location}: object required")
    if any(not isinstance(key, str) for key in value):
        raise ArtifactCollectionError(f"{location}: string keys required")
    return dict(value)


def _sequence(value: object, *, location: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ArtifactCollectionError(f"{location}: array required")
    return tuple(value)


def _string_sequence(value: object, *, location: str) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(child, location=f"{location}/{index}")
        for index, child in enumerate(_sequence(value, location=location))
    )


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactCollectionError(f"{location}: non-empty string required")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ArtifactCollectionError",
    "ArtifactLocatorRegistry",
    "CollectedArtifactSurfaces",
    "LocatorBinding",
    "LocatorSourceKind",
    "collect_artifact_surfaces",
]
