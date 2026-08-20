"""Fail-closed member coverage for US pool container artifacts.

The execution ABI already names every normative artifact, but a directory
selector can only report the members that happened to be present.  This
module adds the independent side of that check for the QRF target banks: the
expected directory and file members are derived from the sealed physical
authorities and matched back to the exact artifact-vector rows.

The final pool H5 is deliberately different.  The runtime plan seals
selectors that enumerate all *observed* tables, columns, and weight vectors,
but it does not contain an exact final materialized inventory against which
that observation can be checked.  The typed contract therefore records that
surface as unsupported.  It must not be promoted to a coverage pass by a
caller merely because the H5 selectors returned bytes.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.model import FrozenMap
from microcosm.build.us_runtime.acs_transfer import _model_target_names
from microcosm.build.us_runtime.acs_transfer_bank import (
    AcsTransferTargetBankStore,
)
from microcosm.build.us_runtime.pool_kernel_authority import (
    USPoolKernelAuthorities,
)
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_DONOR_FILENAME,
    PRIMARY_QRF_RECIPIENT_FILENAME,
    PRIMARY_QRF_TARGETS_DIRNAME,
)
from microcosm.build.us_runtime.puf_qrf_chain import (
    _target_path as _primary_qrf_target_path,
)
from microcosm.build.us_runtime.stacked_spine import (
    _LATE_PRIMARY_QRF_INPUT_BINDING_FILENAME,
)

_COVERAGE_DOMAIN = "microcosm.us-pool-artifact-member-coverage.v1"
_COVERAGE_SCHEMA_VERSION = 1
_DIRECTORY_SELECTOR = "selector:directory_tree_bytes_v1"
_H5_ENTITY_SELECTOR = "selector:h5_all_entity_tables_and_columns_v1"
_H5_WEIGHT_SELECTOR = "selector:h5_all_weight_vectors_v1"
_H5_SELECTORS = frozenset({_H5_ENTITY_SELECTOR, _H5_WEIGHT_SELECTOR})
_RAW_BYTE_COMPARISON = "raw_byte_exact"
_NORMATIVE_SURFACE = "normative"
_VIRTUAL_BANK_ROOT = Path("__microcosm_artifact_coverage_bank_root__")


class PoolArtifactCoverageError(ValueError):
    """The plan, authorities, bindings, or observed member tree is not closed."""


class ArtifactMemberKind(StrEnum):
    """Filesystem shape of one expected directory-tree member."""

    DIRECTORY = "directory"
    FILE = "file"


class TargetBankKind(StrEnum):
    """Physical target-bank protocols covered from compiler authority."""

    GAP_FILL = "gap_fill"
    PRIMARY_QRF = "primary_qrf"
    LATE_TRANSFER = "late_transfer"


class CoverageStatus(StrEnum):
    """Typed outcome for one member-inventory coverage surface."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


class CoverageRootStatus(StrEnum):
    """Observed filesystem shape at one bound bank root."""

    DIRECTORY = "directory"
    MISSING = "missing"
    NOT_DIRECTORY = "not_directory"


class CoverageUnsupportedReason(StrEnum):
    """Stable reason codes that prohibit a self-derived coverage pass."""

    FINAL_H5_INVENTORY_NOT_COMPILED = (
        "compiler_authority_lacks_final_h5_entity_column_weight_inventory"
    )


@dataclass(frozen=True, slots=True)
class ArtifactMemberDescriptor:
    """One exact POSIX-relative directory or file member."""

    relative_path: str
    kind: ArtifactMemberKind

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise PoolArtifactCoverageError(
                "artifact member relative_path must be a non-empty string"
            )
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or path.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise PoolArtifactCoverageError(
                "artifact member relative_path must be normalized and confined: "
                f"{self.relative_path!r}"
            )
        if not isinstance(self.kind, ArtifactMemberKind):
            raise TypeError("artifact member kind must be ArtifactMemberKind")

    def to_wire(self) -> dict[str, object]:
        return {"relative_path": self.relative_path, "kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class TargetBankCoverageContract:
    """Exact plan-bound member inventory for one target-bank artifact."""

    artifact_id: str
    locator_ref: str
    authority_ref: str
    bank_kind: TargetBankKind
    expected_members: tuple[ArtifactMemberDescriptor, ...]

    def __post_init__(self) -> None:
        for name in ("artifact_id", "locator_ref", "authority_ref"):
            _nonempty_string(getattr(self, name), location=f"target bank/{name}")
        if not isinstance(self.bank_kind, TargetBankKind):
            raise TypeError("target bank kind must be TargetBankKind")
        _validate_member_inventory(
            self.expected_members,
            location=f"target bank {self.locator_ref!r}",
        )

    @property
    def expected_members_sha256(self) -> str:
        return sha256_json([member.to_wire() for member in self.expected_members])

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "locator_ref": self.locator_ref,
            "authority_ref": self.authority_ref,
            "bank_kind": self.bank_kind.value,
            "expected_member_count": len(self.expected_members),
            "expected_members_sha256": self.expected_members_sha256,
            "expected_members": [member.to_wire() for member in self.expected_members],
        }


@dataclass(frozen=True, slots=True)
class FinalH5CoverageContract:
    """Typed statement that final-H5 member closure is not compiler-issued."""

    artifact_ids: tuple[str, ...]
    locator_ref: str
    selector_refs: tuple[str, ...]
    status: CoverageStatus
    unsupported_reason: CoverageUnsupportedReason

    def __post_init__(self) -> None:
        if self.status is not CoverageStatus.UNSUPPORTED:
            raise PoolArtifactCoverageError(
                "final H5 coverage cannot be supported without a compiled inventory"
            )
        if not isinstance(self.unsupported_reason, CoverageUnsupportedReason):
            raise TypeError(
                "final H5 unsupported_reason must be CoverageUnsupportedReason"
            )
        _unique_nonempty_strings(self.artifact_ids, location="final H5 artifact_ids")
        _nonempty_string(self.locator_ref, location="final H5 locator_ref")
        _unique_nonempty_strings(self.selector_refs, location="final H5 selectors")
        if frozenset(self.selector_refs) != _H5_SELECTORS:
            raise PoolArtifactCoverageError(
                "final H5 coverage must bind both sealed logical selectors"
            )

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "locator_ref": self.locator_ref,
            "selector_refs": list(self.selector_refs),
            "status": self.status.value,
            "unsupported_reason": self.unsupported_reason.value,
        }


@dataclass(frozen=True, slots=True)
class PoolArtifactCoverageContract:
    """Compiler/physical-authority-bound container member contract."""

    authority_sha256: str
    spec_sha256: str
    execution_abi_sha256: str
    target_banks: tuple[TargetBankCoverageContract, ...]
    final_pool_h5: FinalH5CoverageContract

    def __post_init__(self) -> None:
        for name in ("authority_sha256", "spec_sha256", "execution_abi_sha256"):
            _sha256(getattr(self, name), location=name)
        if not self.target_banks:
            raise PoolArtifactCoverageError(
                "artifact coverage requires at least one target-bank contract"
            )
        locators = tuple(bank.locator_ref for bank in self.target_banks)
        _unique_nonempty_strings(locators, location="target-bank locators")
        artifact_ids = tuple(bank.artifact_id for bank in self.target_banks)
        _unique_nonempty_strings(artifact_ids, location="target-bank artifact ids")
        if not isinstance(self.final_pool_h5, FinalH5CoverageContract):
            raise TypeError("final_pool_h5 must be FinalH5CoverageContract")

    def _wire_body(self) -> dict[str, object]:
        return {
            "domain": _COVERAGE_DOMAIN,
            "schema_version": _COVERAGE_SCHEMA_VERSION,
            "authority_sha256": self.authority_sha256,
            "spec_sha256": self.spec_sha256,
            "execution_abi_sha256": self.execution_abi_sha256,
            "target_banks": [bank.to_wire() for bank in self.target_banks],
            "final_pool_h5": self.final_pool_h5.to_wire(),
        }

    @property
    def contract_sha256(self) -> str:
        return sha256_json(self._wire_body())

    def to_wire(self) -> dict[str, object]:
        wire = self._wire_body()
        wire["sha256"] = self.contract_sha256
        return wire


@dataclass(frozen=True, slots=True)
class TargetBankCoverageResult:
    """Observed result for one exact target-bank member inventory."""

    artifact_id: str
    locator_ref: str
    bank_kind: TargetBankKind
    root_status: CoverageRootStatus
    expected_member_count: int
    expected_members_sha256: str
    observed_members: tuple[ArtifactMemberDescriptor, ...]
    missing_members: tuple[ArtifactMemberDescriptor, ...]
    extra_members: tuple[ArtifactMemberDescriptor, ...]
    status: CoverageStatus

    @property
    def observed_members_sha256(self) -> str:
        return sha256_json([member.to_wire() for member in self.observed_members])

    @property
    def complete(self) -> bool:
        return self.status is CoverageStatus.COMPLETE

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "locator_ref": self.locator_ref,
            "bank_kind": self.bank_kind.value,
            "root_status": self.root_status.value,
            "expected_member_count": self.expected_member_count,
            "expected_members_sha256": self.expected_members_sha256,
            "observed_member_count": len(self.observed_members),
            "observed_members_sha256": self.observed_members_sha256,
            "missing_members": [member.to_wire() for member in self.missing_members],
            "extra_members": [member.to_wire() for member in self.extra_members],
            "status": self.status.value,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class PoolArtifactCoverageReceipt:
    """Typed bank result plus an explicit non-pass for unsupported H5 closure."""

    contract: PoolArtifactCoverageContract
    target_banks: tuple[TargetBankCoverageResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, PoolArtifactCoverageContract):
            raise TypeError("contract must be PoolArtifactCoverageContract")
        expected = tuple(bank.locator_ref for bank in self.contract.target_banks)
        observed = tuple(bank.locator_ref for bank in self.target_banks)
        if observed != expected:
            raise PoolArtifactCoverageError(
                "coverage result order differs from its target-bank contract"
            )

    @property
    def bank_member_coverage_complete(self) -> bool:
        return all(bank.complete for bank in self.target_banks)

    @property
    def container_member_coverage_complete(self) -> bool:
        return (
            self.bank_member_coverage_complete
            and self.contract.final_pool_h5.status is CoverageStatus.COMPLETE
        )

    @property
    def status(self) -> CoverageStatus:
        if self.contract.final_pool_h5.status is CoverageStatus.UNSUPPORTED:
            return CoverageStatus.UNSUPPORTED
        if self.container_member_coverage_complete:
            return CoverageStatus.COMPLETE
        return CoverageStatus.INCOMPLETE

    def _wire_body(self) -> dict[str, object]:
        return {
            "domain": _COVERAGE_DOMAIN,
            "schema_version": _COVERAGE_SCHEMA_VERSION,
            "contract": self.contract.to_wire(),
            "target_banks": [bank.to_wire() for bank in self.target_banks],
            "bank_member_coverage_complete": self.bank_member_coverage_complete,
            "final_pool_h5": self.contract.final_pool_h5.to_wire(),
            "container_member_coverage_complete": (
                self.container_member_coverage_complete
            ),
            "status": self.status.value,
        }

    @property
    def receipt_sha256(self) -> str:
        return sha256_json(self._wire_body())

    def to_wire(self) -> dict[str, object]:
        wire = self._wire_body()
        wire["receipt_sha256"] = self.receipt_sha256
        return wire


@dataclass(frozen=True, slots=True)
class _ExpectedBank:
    artifact_row: FrozenMap
    authority_ref: str
    bank_kind: TargetBankKind
    expected_members: tuple[ArtifactMemberDescriptor, ...]


def compile_pool_artifact_coverage(
    plan: USPoolRuntimePlan,
    authorities: USPoolKernelAuthorities,
) -> PoolArtifactCoverageContract:
    """Compile exact bank members and an explicit unsupported final-H5 record."""

    if not isinstance(plan, USPoolRuntimePlan):
        raise TypeError("plan must be USPoolRuntimePlan")
    if not isinstance(authorities, USPoolKernelAuthorities):
        raise TypeError("authorities must be USPoolKernelAuthorities")
    for name in ("authority_sha256", "spec_sha256"):
        if getattr(plan, name) != getattr(authorities, name):
            raise PoolArtifactCoverageError(
                f"runtime plan and kernel authorities have different {name}"
            )

    artifact_rows = plan.execution.artifact_vector
    directory_rows = tuple(
        row
        for row in artifact_rows
        if row.get("content_selector_ref") == _DIRECTORY_SELECTOR
    )
    expected: list[_ExpectedBank] = []

    gap_schedule = _gap_fill_schedule_rows(authorities)
    for direction in authorities.physical.gap_fill.directions:
        artifact_id = f"gap_fill_direction_bank:{direction.name}"
        row = _one_artifact_row(
            directory_rows,
            location=f"gap-fill target bank {direction.name!r}",
            artifact_id=artifact_id,
        )
        _validate_bank_artifact_row(
            row,
            artifact_kind="early_transfer_target_bank",
            protocol_ref="code_abi:transfer_target_bank",
        )
        schedule_row = _one_frozen_row(
            gap_schedule,
            location=f"gap-fill schedule {direction.name!r}",
            field="name",
            value=direction.name,
        )
        model_targets = _gap_fill_model_targets(schedule_row, direction.name)
        expected.append(
            _ExpectedBank(
                artifact_row=row,
                authority_ref=f"gap_fill_direction:{direction.name}",
                bank_kind=TargetBankKind.GAP_FILL,
                expected_members=_acs_target_bank_members(model_targets),
            )
        )

    primary = authorities.physical.primary_qrf
    row = _one_artifact_row(
        directory_rows,
        location="primary QRF target bank",
        artifact_kind="primary_qrf_checkpoint",
        producer_ref=f"producer_node:{primary.node.id}",
    )
    _validate_bank_artifact_row(
        row,
        artifact_kind="primary_qrf_checkpoint",
        protocol_ref="code_abi:primary_qrf_checkpoint",
    )
    expected.append(
        _ExpectedBank(
            artifact_row=row,
            authority_ref=f"producer_node:{primary.node.id}",
            bank_kind=TargetBankKind.PRIMARY_QRF,
            expected_members=_primary_qrf_members(
                target_order=primary.target_order,
                manifest_filename=primary.manifest_filename,
            ),
        )
    )

    for group in authorities.physical.late_producers.transfer_groups:
        row = _one_artifact_row(
            directory_rows,
            location=f"late-transfer target bank {group.name!r}",
            artifact_kind="late_transfer_target_bank",
            producer_ref=f"producer_node:{group.name}",
        )
        _validate_bank_artifact_row(
            row,
            artifact_kind="late_transfer_target_bank",
            protocol_ref="code_abi:late_transfer_target_bank",
        )
        expected.append(
            _ExpectedBank(
                artifact_row=row,
                authority_ref=f"producer_node:{group.name}",
                bank_kind=TargetBankKind.LATE_TRANSFER,
                expected_members=_acs_target_bank_members(
                    _model_target_names(group.targets)
                ),
            )
        )

    expected_by_locator = {
        _row_string(item.artifact_row, "locator_ref"): item for item in expected
    }
    if len(expected_by_locator) != len(expected):
        raise PoolArtifactCoverageError(
            "physical authorities resolved duplicate target-bank locators"
        )
    directory_locators = tuple(
        _row_string(row, "locator_ref") for row in directory_rows
    )
    if frozenset(expected_by_locator) != frozenset(directory_locators):
        raise PoolArtifactCoverageError(
            "physical target-bank authorities do not exactly cover the sealed "
            "directory artifacts: "
            f"missing={sorted(set(directory_locators) - set(expected_by_locator))}, "
            f"extra={sorted(set(expected_by_locator) - set(directory_locators))}"
        )

    banks = tuple(
        TargetBankCoverageContract(
            artifact_id=_row_string(row, "id"),
            locator_ref=locator_ref,
            authority_ref=expected_by_locator[locator_ref].authority_ref,
            bank_kind=expected_by_locator[locator_ref].bank_kind,
            expected_members=expected_by_locator[locator_ref].expected_members,
        )
        for row in directory_rows
        for locator_ref in (_row_string(row, "locator_ref"),)
    )
    h5 = _compile_final_h5_coverage(artifact_rows)
    return PoolArtifactCoverageContract(
        authority_sha256=plan.authority_sha256,
        spec_sha256=plan.spec_sha256,
        execution_abi_sha256=plan.execution.abi_sha256,
        target_banks=banks,
        final_pool_h5=h5,
    )


def validate_pool_artifact_coverage(
    contract: PoolArtifactCoverageContract,
    *,
    bank_roots: Mapping[str, str | Path],
) -> PoolArtifactCoverageReceipt:
    """Validate exact observed bank members for every contract locator.

    Missing and extra ordinary members are returned as typed incomplete rows.
    Symlinks, special files, duplicate roots, and locator-binding mismatches are
    malformed evidence and raise :class:`PoolArtifactCoverageError`.
    """

    if not isinstance(contract, PoolArtifactCoverageContract):
        raise TypeError("contract must be PoolArtifactCoverageContract")
    if not isinstance(bank_roots, Mapping):
        raise TypeError("bank_roots must be a locator-to-path mapping")
    if any(not isinstance(key, str) for key in bank_roots):
        raise PoolArtifactCoverageError("bank_roots locator keys must be strings")
    expected_locators = frozenset(bank.locator_ref for bank in contract.target_banks)
    observed_locators = frozenset(bank_roots)
    if observed_locators != expected_locators:
        raise PoolArtifactCoverageError(
            "target-bank locator binding inventory mismatch: "
            f"missing={sorted(expected_locators - observed_locators)}, "
            f"extra={sorted(observed_locators - expected_locators)}"
        )

    normalized_roots: dict[str, Path] = {}
    root_owners: dict[Path, str] = {}
    for locator_ref, value in bank_roots.items():
        if not isinstance(value, str | Path):
            raise TypeError(f"bank root for {locator_ref!r} must be str or Path")
        root = Path(value).absolute()
        previous = root_owners.get(root)
        if previous is not None:
            raise PoolArtifactCoverageError(
                "target-bank locators cannot share one physical root: "
                f"{previous!r}, {locator_ref!r} -> {root}"
            )
        root_owners[root] = locator_ref
        normalized_roots[locator_ref] = root

    results: list[TargetBankCoverageResult] = []
    for bank in contract.target_banks:
        root_status, observed_members = _scan_directory_members(
            normalized_roots[bank.locator_ref]
        )
        expected_by_key = {
            (member.relative_path, member.kind): member
            for member in bank.expected_members
        }
        observed_by_key = {
            (member.relative_path, member.kind): member for member in observed_members
        }
        missing = _sort_members(
            tuple(
                expected_by_key[key]
                for key in expected_by_key.keys() - observed_by_key.keys()
            )
        )
        extra = _sort_members(
            tuple(
                observed_by_key[key]
                for key in observed_by_key.keys() - expected_by_key.keys()
            )
        )
        complete = (
            root_status is CoverageRootStatus.DIRECTORY and not missing and not extra
        )
        results.append(
            TargetBankCoverageResult(
                artifact_id=bank.artifact_id,
                locator_ref=bank.locator_ref,
                bank_kind=bank.bank_kind,
                root_status=root_status,
                expected_member_count=len(bank.expected_members),
                expected_members_sha256=bank.expected_members_sha256,
                observed_members=observed_members,
                missing_members=missing,
                extra_members=extra,
                status=(
                    CoverageStatus.COMPLETE if complete else CoverageStatus.INCOMPLETE
                ),
            )
        )
    return PoolArtifactCoverageReceipt(contract=contract, target_banks=tuple(results))


def _compile_final_h5_coverage(
    artifact_rows: tuple[FrozenMap, ...],
) -> FinalH5CoverageContract:
    rows = tuple(
        row for row in artifact_rows if row.get("content_selector_ref") in _H5_SELECTORS
    )
    if len(rows) != len(_H5_SELECTORS):
        raise PoolArtifactCoverageError(
            "sealed artifact vector must contain both final-pool H5 selectors"
        )
    selectors = tuple(_row_string(row, "content_selector_ref") for row in rows)
    if frozenset(selectors) != _H5_SELECTORS:
        raise PoolArtifactCoverageError(
            "sealed final-pool H5 selector inventory is incomplete or duplicated"
        )
    locators = {_row_string(row, "locator_ref") for row in rows}
    if len(locators) != 1:
        raise PoolArtifactCoverageError(
            "final-pool H5 selectors must share exactly one locator"
        )
    for row in rows:
        if (
            row.get("kind") != "logical_h5_content"
            or row.get("surface") != _NORMATIVE_SURFACE
            or row.get("comparison") != _RAW_BYTE_COMPARISON
            or row.get("required") is not True
        ):
            raise PoolArtifactCoverageError(
                "final-pool H5 artifact row differs from the required normative "
                "raw-byte contract"
            )
    return FinalH5CoverageContract(
        artifact_ids=tuple(_row_string(row, "id") for row in rows),
        locator_ref=locators.pop(),
        selector_refs=selectors,
        status=CoverageStatus.UNSUPPORTED,
        unsupported_reason=(CoverageUnsupportedReason.FINAL_H5_INVENTORY_NOT_COMPILED),
    )


def _gap_fill_schedule_rows(
    authorities: USPoolKernelAuthorities,
) -> tuple[FrozenMap, ...]:
    value = authorities.physical.gap_fill.schedule_receipt.get("directions")
    if (
        not isinstance(value, tuple)
        or not value
        or not all(isinstance(row, FrozenMap) for row in value)
    ):
        raise PoolArtifactCoverageError(
            "gap-fill authority requires a non-empty sealed direction schedule"
        )
    return value


def _gap_fill_model_targets(schedule_row: FrozenMap, name: str) -> tuple[str, ...]:
    raw_targets = schedule_row.get("targets")
    if (
        not isinstance(raw_targets, tuple)
        or not raw_targets
        or not all(isinstance(row, FrozenMap) for row in raw_targets)
    ):
        raise PoolArtifactCoverageError(
            f"gap-fill schedule {name!r} requires non-empty sealed target rows"
        )

    groups: list[tuple[tuple[str, str], list[str]]] = []
    seen_groups: set[tuple[str, str]] = set()
    for row in raw_targets:
        key = (_row_string(row, "entity"), _row_string(row, "family"))
        target = _row_string(row, "column")
        if not groups or groups[-1][0] != key:
            if key in seen_groups:
                raise PoolArtifactCoverageError(
                    f"gap-fill schedule {name!r} repeats non-contiguous family {key}"
                )
            seen_groups.add(key)
            groups.append((key, []))
        groups[-1][1].append(target)

    model_targets = tuple(
        model_target
        for _key, targets in groups
        for model_target in _model_target_names(tuple(targets))
    )
    _unique_nonempty_strings(
        model_targets,
        location=f"gap-fill schedule {name!r} model targets",
    )
    return model_targets


def _acs_target_bank_members(
    model_targets: Sequence[str],
) -> tuple[ArtifactMemberDescriptor, ...]:
    resolved_targets = tuple(model_targets)
    _unique_nonempty_strings(resolved_targets, location="ACS bank model targets")
    store = AcsTransferTargetBankStore(_VIRTUAL_BANK_ROOT, identity={})
    members = [ArtifactMemberDescriptor("targets", ArtifactMemberKind.DIRECTORY)]
    members.extend(
        ArtifactMemberDescriptor(
            store.target_path(index, target).relative_to(_VIRTUAL_BANK_ROOT).as_posix(),
            ArtifactMemberKind.FILE,
        )
        for index, target in enumerate(resolved_targets)
    )
    return _sort_members(tuple(members))


def _primary_qrf_members(
    *,
    target_order: tuple[str, ...],
    manifest_filename: str,
) -> tuple[ArtifactMemberDescriptor, ...]:
    _unique_nonempty_strings(target_order, location="primary QRF target order")
    manifest = {"target_order": list(target_order)}
    members = [
        ArtifactMemberDescriptor(
            PRIMARY_QRF_TARGETS_DIRNAME,
            ArtifactMemberKind.DIRECTORY,
        ),
        ArtifactMemberDescriptor(manifest_filename, ArtifactMemberKind.FILE),
        ArtifactMemberDescriptor(
            PRIMARY_QRF_DONOR_FILENAME,
            ArtifactMemberKind.FILE,
        ),
        ArtifactMemberDescriptor(
            PRIMARY_QRF_RECIPIENT_FILENAME,
            ArtifactMemberKind.FILE,
        ),
        ArtifactMemberDescriptor(
            _LATE_PRIMARY_QRF_INPUT_BINDING_FILENAME,
            ArtifactMemberKind.FILE,
        ),
    ]
    members.extend(
        ArtifactMemberDescriptor(
            _primary_qrf_target_path(_VIRTUAL_BANK_ROOT, manifest, index)
            .relative_to(_VIRTUAL_BANK_ROOT)
            .as_posix(),
            ArtifactMemberKind.FILE,
        )
        for index in range(len(target_order))
    )
    return _sort_members(tuple(members))


def _validate_bank_artifact_row(
    row: FrozenMap,
    *,
    artifact_kind: str,
    protocol_ref: str,
) -> None:
    expected = {
        "kind": artifact_kind,
        "protocol_ref": protocol_ref,
        "content_selector_ref": _DIRECTORY_SELECTOR,
        "surface": _NORMATIVE_SURFACE,
        "comparison": _RAW_BYTE_COMPARISON,
        "required": True,
    }
    changed = {
        key: (value, row.get(key))
        for key, value in expected.items()
        if row.get(key) != value
    }
    if changed:
        raise PoolArtifactCoverageError(
            f"target-bank artifact {_row_string(row, 'id')!r} differs from its "
            f"required contract: {changed}"
        )


def _one_artifact_row(
    rows: tuple[FrozenMap, ...],
    *,
    location: str,
    artifact_id: str | None = None,
    artifact_kind: str | None = None,
    producer_ref: str | None = None,
) -> FrozenMap:
    matches = tuple(
        row
        for row in rows
        if (artifact_id is None or row.get("id") == artifact_id)
        and (artifact_kind is None or row.get("kind") == artifact_kind)
        and (producer_ref is None or row.get("producer_ref") == producer_ref)
    )
    if len(matches) != 1:
        raise PoolArtifactCoverageError(
            f"{location}: exactly one sealed artifact row required; "
            f"matched {len(matches)}"
        )
    return matches[0]


def _one_frozen_row(
    rows: tuple[FrozenMap, ...],
    *,
    location: str,
    field: str,
    value: str,
) -> FrozenMap:
    matches = tuple(row for row in rows if row.get(field) == value)
    if len(matches) != 1:
        raise PoolArtifactCoverageError(
            f"{location}: exactly one sealed row required; matched {len(matches)}"
        )
    return matches[0]


def _scan_directory_members(
    root: Path,
) -> tuple[CoverageRootStatus, tuple[ArtifactMemberDescriptor, ...]]:
    if root.is_symlink():
        raise PoolArtifactCoverageError(
            f"target-bank root cannot be a symbolic link: {root}"
        )
    if not root.exists():
        return CoverageRootStatus.MISSING, ()
    if not root.is_dir():
        return CoverageRootStatus.NOT_DIRECTORY, ()

    members: list[ArtifactMemberDescriptor] = []

    def visit(directory: Path, relative: PurePosixPath | None = None) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = tuple(iterator)
        except OSError as error:
            raise PoolArtifactCoverageError(
                f"cannot enumerate target-bank directory {directory}: {error}"
            ) from error
        for entry in sorted(entries, key=lambda item: item.name.encode("utf-8")):
            child_relative = (
                PurePosixPath(entry.name) if relative is None else relative / entry.name
            )
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise PoolArtifactCoverageError(
                    f"cannot inspect target-bank member {entry.path}: {error}"
                ) from error
            if stat.S_ISLNK(mode):
                raise PoolArtifactCoverageError(
                    f"target-bank member cannot be a symbolic link: {entry.path}"
                )
            if stat.S_ISDIR(mode):
                members.append(
                    ArtifactMemberDescriptor(
                        child_relative.as_posix(),
                        ArtifactMemberKind.DIRECTORY,
                    )
                )
                visit(Path(entry.path), child_relative)
                continue
            if stat.S_ISREG(mode):
                members.append(
                    ArtifactMemberDescriptor(
                        child_relative.as_posix(),
                        ArtifactMemberKind.FILE,
                    )
                )
                continue
            raise PoolArtifactCoverageError(
                f"target-bank member must be a directory or regular file: {entry.path}"
            )

    visit(root)
    observed = _sort_members(tuple(members))
    _validate_member_inventory(observed, location=f"observed target bank {root}")
    return CoverageRootStatus.DIRECTORY, observed


def _validate_member_inventory(
    members: tuple[ArtifactMemberDescriptor, ...],
    *,
    location: str,
) -> None:
    if not members or not all(
        isinstance(member, ArtifactMemberDescriptor) for member in members
    ):
        raise PoolArtifactCoverageError(
            f"{location}: non-empty typed member inventory required"
        )
    if members != _sort_members(members):
        raise PoolArtifactCoverageError(
            f"{location}: members must use directory-selector byte ordering"
        )
    keys = tuple((member.relative_path, member.kind) for member in members)
    if len(keys) != len(set(keys)):
        raise PoolArtifactCoverageError(f"{location}: duplicate members forbidden")
    shapes_by_path: dict[str, ArtifactMemberKind] = {}
    for member in members:
        previous = shapes_by_path.setdefault(member.relative_path, member.kind)
        if previous is not member.kind:
            raise PoolArtifactCoverageError(
                f"{location}: member path has conflicting shapes: "
                f"{member.relative_path!r}"
            )


def _sort_members(
    members: tuple[ArtifactMemberDescriptor, ...],
) -> tuple[ArtifactMemberDescriptor, ...]:
    return tuple(
        sorted(
            members,
            key=lambda member: (
                member.relative_path.encode("utf-8"),
                member.kind.value,
            ),
        )
    )


def _row_string(row: FrozenMap, field: str) -> str:
    return _nonempty_string(row.get(field), location=f"artifact row/{field}")


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise PoolArtifactCoverageError(f"{location}: non-empty string required")
    return value


def _unique_nonempty_strings(values: tuple[str, ...], *, location: str) -> None:
    if not values or not all(isinstance(value, str) and value for value in values):
        raise PoolArtifactCoverageError(f"{location}: non-empty string array required")
    if len(values) != len(set(values)):
        raise PoolArtifactCoverageError(f"{location}: duplicate values forbidden")


def _sha256(value: object, *, location: str) -> str:
    text = _nonempty_string(value, location=location)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise PoolArtifactCoverageError(
            f"{location}: lowercase hexadecimal SHA-256 required"
        )
    return text


__all__ = [
    "ArtifactMemberDescriptor",
    "ArtifactMemberKind",
    "CoverageRootStatus",
    "CoverageStatus",
    "CoverageUnsupportedReason",
    "FinalH5CoverageContract",
    "PoolArtifactCoverageContract",
    "PoolArtifactCoverageError",
    "PoolArtifactCoverageReceipt",
    "TargetBankCoverageContract",
    "TargetBankCoverageResult",
    "TargetBankKind",
    "compile_pool_artifact_coverage",
    "validate_pool_artifact_coverage",
]
