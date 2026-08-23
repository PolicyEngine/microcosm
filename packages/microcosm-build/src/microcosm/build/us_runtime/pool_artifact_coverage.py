"""Fail-closed member coverage for US pool container artifacts.

The execution ABI already names every normative artifact, but a directory
selector can only report the members that happened to be present.  This
module adds the independent side of that check for the QRF target banks: the
expected directory and file members are derived from the sealed physical
authorities and matched back to the exact artifact-vector rows.

The final pool H5 is closed independently from the logical selectors.  Its
expected table, column, and weight members come only from the compiler-sealed
runtime inventory.  The validator reads only fixed-storer schema axes, rejects
unmodelled HDF keys, and fences the same physical file across artifact
collection and coverage inspection.
"""

from __future__ import annotations

import json
import os
import stat
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import tables

from microcosm.build.spec_engine.artifact_collection import (
    ArtifactCollectionError,
    capture_artifact_file_identity,
)
from microcosm.build.spec_engine.artifact_collection import (
    ArtifactFileIdentity as PoolH5FileIdentity,
)
from microcosm.build.spec_engine.artifact_selector_contract import (
    H5_ARTIFACT_METADATA_KEY,
    H5_TIME_PERIOD_KEY,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.final_h5_inventory import (
    FinalH5InventoryError,
    canonical_final_h5_member_descriptors,
    validate_final_h5_member_inventory,
)
from microcosm.build.spec_engine.model import FrozenMap, freeze_json, thaw_json
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
_COVERAGE_SCHEMA_VERSION = 2
_DIRECTORY_SELECTOR = "selector:directory_tree_bytes_v1"
_H5_ENTITY_SELECTOR = "selector:h5_all_entity_tables_and_columns_v1"
_H5_WEIGHT_SELECTOR = "selector:h5_all_weight_vectors_v1"
_H5_SELECTORS = frozenset({_H5_ENTITY_SELECTOR, _H5_WEIGHT_SELECTOR})
_RAW_BYTE_COMPARISON = "raw_byte_exact"
_NORMATIVE_SURFACE = "normative"
_VIRTUAL_BANK_ROOT = Path("__microcosm_artifact_coverage_bank_root__")
_FINAL_H5_HEADER_MAX_BYTES = 1_048_576


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


class CoverageRootStatus(StrEnum):
    """Observed filesystem shape at one bound bank root."""

    DIRECTORY = "directory"
    MISSING = "missing"
    NOT_DIRECTORY = "not_directory"


class FinalH5MemberKind(StrEnum):
    """Logical member kinds sealed by the compact final-H5 inventory."""

    ENTITY_TABLE = "entity_table"
    ENTITY_COLUMN = "entity_column"
    WEIGHT_VECTOR = "weight_vector"


_FINAL_H5_KIND_ORDER = {
    FinalH5MemberKind.ENTITY_TABLE: 0,
    FinalH5MemberKind.ENTITY_COLUMN: 1,
    FinalH5MemberKind.WEIGHT_VECTOR: 2,
}


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
class FinalH5MemberDescriptor:
    """One table, ordinary column, or weight member in canonical set order."""

    kind: FinalH5MemberKind
    entity: str
    column: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FinalH5MemberKind):
            raise TypeError("final H5 member kind must be FinalH5MemberKind")
        _nonempty_string(self.entity, location="final H5 member/entity")
        if self.kind is FinalH5MemberKind.ENTITY_TABLE:
            if self.column is not None:
                raise PoolArtifactCoverageError(
                    "final H5 table member cannot carry a column"
                )
        else:
            _nonempty_string(self.column, location="final H5 member/column")

    def to_wire(self) -> dict[str, str]:
        wire = {"kind": self.kind.value, "entity": self.entity}
        if self.column is not None:
            wire["column"] = self.column
        return wire


@dataclass(frozen=True, slots=True)
class FinalH5CoverageContract:
    """Exact compiler-issued logical-member inventory for the final pool H5."""

    artifact_ids: tuple[str, ...]
    locator_ref: str
    selector_refs: tuple[str, ...]
    member_inventory: FrozenMap
    expected_members: tuple[FinalH5MemberDescriptor, ...]

    def __post_init__(self) -> None:
        _unique_nonempty_strings(self.artifact_ids, location="final H5 artifact_ids")
        _nonempty_string(self.locator_ref, location="final H5 locator_ref")
        _unique_nonempty_strings(self.selector_refs, location="final H5 selectors")
        if frozenset(self.selector_refs) != _H5_SELECTORS:
            raise PoolArtifactCoverageError(
                "final H5 coverage must bind both sealed logical selectors"
            )
        if not isinstance(self.member_inventory, FrozenMap):
            raise TypeError("final H5 member_inventory must be FrozenMap")
        try:
            inventory = validate_final_h5_member_inventory(
                thaw_json(self.member_inventory)
            )
        except FinalH5InventoryError as error:
            raise PoolArtifactCoverageError(
                "final H5 member inventory is invalid"
            ) from error
        expected = _final_h5_members_from_inventory(inventory)
        if self.expected_members != expected:
            raise PoolArtifactCoverageError(
                "final H5 expected members differ from the compact inventory"
            )

    @property
    def inventory_sha256(self) -> str:
        return _inventory_string(self.member_inventory, "inventory_sha256")

    @property
    def expected_members_sha256(self) -> str:
        return sha256_json([member.to_wire() for member in self.expected_members])

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "locator_ref": self.locator_ref,
            "selector_refs": list(self.selector_refs),
            "member_inventory": thaw_json(self.member_inventory),
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
class FinalH5CoverageResult:
    """Observed logical-member closure for one identity-fenced final pool H5."""

    artifact_ids: tuple[str, ...]
    locator_ref: str
    selector_refs: tuple[str, ...]
    inventory_sha256: str
    expected_member_count: int
    expected_members_sha256: str
    observed_members: tuple[FinalH5MemberDescriptor, ...]
    missing_members: tuple[FinalH5MemberDescriptor, ...]
    extra_members: tuple[FinalH5MemberDescriptor, ...]
    status: CoverageStatus

    def __post_init__(self) -> None:
        _unique_nonempty_strings(
            self.artifact_ids, location="final H5 result artifacts"
        )
        _nonempty_string(self.locator_ref, location="final H5 result locator")
        _unique_nonempty_strings(
            self.selector_refs, location="final H5 result selectors"
        )
        if frozenset(self.selector_refs) != _H5_SELECTORS:
            raise PoolArtifactCoverageError(
                "final H5 result must bind both sealed logical selectors"
            )
        _sha256(self.inventory_sha256, location="final H5 result inventory sha256")
        _sha256(
            self.expected_members_sha256,
            location="final H5 result expected-members sha256",
        )
        if (
            isinstance(self.expected_member_count, bool)
            or not isinstance(self.expected_member_count, int)
            or self.expected_member_count <= 0
        ):
            raise PoolArtifactCoverageError(
                "final H5 expected member count must be a positive integer"
            )
        _validate_final_h5_members(
            self.observed_members,
            location="observed final H5 members",
            nonempty=False,
        )
        _validate_final_h5_members(
            self.missing_members,
            location="missing final H5 members",
            nonempty=False,
        )
        _validate_final_h5_members(
            self.extra_members,
            location="extra final H5 members",
            nonempty=False,
        )
        if self.status not in {CoverageStatus.COMPLETE, CoverageStatus.INCOMPLETE}:
            raise PoolArtifactCoverageError(
                "final H5 result status must be complete or incomplete"
            )
        if self.complete is not (not self.missing_members and not self.extra_members):
            raise PoolArtifactCoverageError(
                "final H5 result status contradicts its missing/extra members"
            )

    @property
    def observed_members_sha256(self) -> str:
        return sha256_json([member.to_wire() for member in self.observed_members])

    @property
    def complete(self) -> bool:
        return self.status is CoverageStatus.COMPLETE

    def to_wire(self) -> dict[str, object]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "locator_ref": self.locator_ref,
            "selector_refs": list(self.selector_refs),
            "inventory_sha256": self.inventory_sha256,
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
    """Typed bank and final-H5 results linked to one compiled contract."""

    contract: PoolArtifactCoverageContract
    target_banks: tuple[TargetBankCoverageResult, ...]
    final_pool_h5: FinalH5CoverageResult

    def __post_init__(self) -> None:
        if not isinstance(self.contract, PoolArtifactCoverageContract):
            raise TypeError("contract must be PoolArtifactCoverageContract")
        expected = tuple(bank.locator_ref for bank in self.contract.target_banks)
        observed = tuple(bank.locator_ref for bank in self.target_banks)
        if observed != expected:
            raise PoolArtifactCoverageError(
                "coverage result order differs from its target-bank contract"
            )
        if not isinstance(self.final_pool_h5, FinalH5CoverageResult):
            raise TypeError("final_pool_h5 must be FinalH5CoverageResult")
        final_contract = self.contract.final_pool_h5
        links = (
            (
                "artifact_ids",
                self.final_pool_h5.artifact_ids,
                final_contract.artifact_ids,
            ),
            ("locator_ref", self.final_pool_h5.locator_ref, final_contract.locator_ref),
            (
                "selector_refs",
                self.final_pool_h5.selector_refs,
                final_contract.selector_refs,
            ),
            (
                "inventory_sha256",
                self.final_pool_h5.inventory_sha256,
                final_contract.inventory_sha256,
            ),
            (
                "expected_member_count",
                self.final_pool_h5.expected_member_count,
                len(final_contract.expected_members),
            ),
            (
                "expected_members_sha256",
                self.final_pool_h5.expected_members_sha256,
                final_contract.expected_members_sha256,
            ),
        )
        changed = {
            name: (observed_value, expected_value)
            for name, observed_value, expected_value in links
            if observed_value != expected_value
        }
        if changed:
            raise PoolArtifactCoverageError(
                f"final H5 result differs from its compiled contract: {changed}"
            )
        expected_by_key = {
            _final_h5_member_key(member): member
            for member in final_contract.expected_members
        }
        observed_by_key = {
            _final_h5_member_key(member): member
            for member in self.final_pool_h5.observed_members
        }
        exact_missing = _sort_final_h5_members(
            tuple(
                expected_by_key[key]
                for key in expected_by_key.keys() - observed_by_key.keys()
            )
        )
        exact_extra = _sort_final_h5_members(
            tuple(
                observed_by_key[key]
                for key in observed_by_key.keys() - expected_by_key.keys()
            )
        )
        if (
            self.final_pool_h5.missing_members != exact_missing
            or self.final_pool_h5.extra_members != exact_extra
        ):
            raise PoolArtifactCoverageError(
                "final H5 result missing/extra members contradict its observations"
            )

    @property
    def bank_member_coverage_complete(self) -> bool:
        return all(bank.complete for bank in self.target_banks)

    @property
    def container_member_coverage_complete(self) -> bool:
        return self.bank_member_coverage_complete and self.final_pool_h5.complete

    @property
    def status(self) -> CoverageStatus:
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
            "final_pool_h5": self.final_pool_h5.to_wire(),
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
    """Compile exact bank and final-H5 logical-member inventories."""

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
    h5 = _compile_final_h5_coverage(
        artifact_rows,
        member_inventory=plan.execution.pipeline.get("final_h5_member_inventory"),
    )
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
    pool_h5: str | Path,
    pool_h5_identity: PoolH5FileIdentity,
) -> PoolArtifactCoverageReceipt:
    """Validate exact bank and final-H5 members for every contract locator.

    Missing and extra ordinary members are returned as typed incomplete rows.
    Symlinks, special files, duplicate roots, locator mismatches, unmodelled HDF
    keys, and file-identity changes are malformed evidence and raise
    :class:`PoolArtifactCoverageError`.
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
    final_h5_result = _validate_final_h5_coverage(
        contract.final_pool_h5,
        pool_h5=pool_h5,
        pool_h5_identity=pool_h5_identity,
    )
    return PoolArtifactCoverageReceipt(
        contract=contract,
        target_banks=tuple(results),
        final_pool_h5=final_h5_result,
    )


def _compile_final_h5_coverage(
    artifact_rows: tuple[FrozenMap, ...],
    *,
    member_inventory: object,
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
    try:
        inventory = validate_final_h5_member_inventory(member_inventory)
    except FinalH5InventoryError as error:
        raise PoolArtifactCoverageError(
            "execution pipeline requires a valid final_h5_member_inventory"
        ) from error
    frozen_inventory = freeze_json(inventory)
    if not isinstance(frozen_inventory, FrozenMap):  # pragma: no cover - validator owns
        raise AssertionError("final H5 inventory did not freeze to a mapping")
    expected_members = _final_h5_members_from_inventory(inventory)
    for member in expected_members:
        weight_protocol_match = member.column == f"{member.entity}_weight"
        if member.kind is FinalH5MemberKind.ENTITY_TABLE:
            continue
        if (member.kind is FinalH5MemberKind.WEIGHT_VECTOR) is not (
            weight_protocol_match
        ):
            raise PoolArtifactCoverageError(
                "final H5 member kind differs from the sealed selector naming "
                f"protocol: {member.to_wire()}"
            )
    return FinalH5CoverageContract(
        artifact_ids=tuple(_row_string(row, "id") for row in rows),
        locator_ref=locators.pop(),
        selector_refs=selectors,
        member_inventory=frozen_inventory,
        expected_members=expected_members,
    )


def capture_pool_h5_file_identity(path: str | Path) -> PoolH5FileIdentity:
    """Capture one symlink-free regular-file identity without serializing it."""

    absolute = _absolute_pool_h5_path(path)
    _require_nonsymlink_regular_path(absolute)
    try:
        return capture_artifact_file_identity(absolute)
    except ArtifactCollectionError as error:
        raise PoolArtifactCoverageError(
            f"cannot capture final pool H5 identity: {absolute}: {error}"
        ) from error


def _validate_final_h5_coverage(
    contract: FinalH5CoverageContract,
    *,
    pool_h5: str | Path,
    pool_h5_identity: PoolH5FileIdentity,
) -> FinalH5CoverageResult:
    if not isinstance(pool_h5_identity, PoolH5FileIdentity):
        raise TypeError("pool_h5_identity must be PoolH5FileIdentity")
    path = _absolute_pool_h5_path(pool_h5)
    if path != pool_h5_identity.path:
        raise PoolArtifactCoverageError(
            "final pool H5 path differs from its pre-collection identity: "
            f"expected={pool_h5_identity.path}, observed={path}"
        )
    with _fenced_pool_h5(path, pool_h5_identity):
        observed_members = _scan_final_h5_members(
            path,
            contract=contract,
            pool_h5_identity=pool_h5_identity,
        )

    expected_by_key = {
        _final_h5_member_key(row): row for row in contract.expected_members
    }
    observed_by_key = {_final_h5_member_key(row): row for row in observed_members}
    missing = _sort_final_h5_members(
        tuple(
            expected_by_key[key]
            for key in expected_by_key.keys() - observed_by_key.keys()
        )
    )
    extra = _sort_final_h5_members(
        tuple(
            observed_by_key[key]
            for key in observed_by_key.keys() - expected_by_key.keys()
        )
    )
    complete = not missing and not extra
    return FinalH5CoverageResult(
        artifact_ids=contract.artifact_ids,
        locator_ref=contract.locator_ref,
        selector_refs=contract.selector_refs,
        inventory_sha256=contract.inventory_sha256,
        expected_member_count=len(contract.expected_members),
        expected_members_sha256=contract.expected_members_sha256,
        observed_members=observed_members,
        missing_members=missing,
        extra_members=extra,
        status=CoverageStatus.COMPLETE if complete else CoverageStatus.INCOMPLETE,
    )


@contextmanager
def _fenced_pool_h5(
    path: Path,
    expected: PoolH5FileIdentity,
) -> Iterator[Path]:
    _require_current_pool_h5_identity(path, expected)
    try:
        yield path
    finally:
        _require_current_pool_h5_identity(path, expected)


def _require_current_pool_h5_identity(
    path: Path,
    expected: PoolH5FileIdentity,
) -> None:
    observed = capture_pool_h5_file_identity(path)
    if observed != expected:
        raise PoolArtifactCoverageError(
            "final pool H5 changed since its pre-collection identity was captured: "
            f"{path}"
        )


def _scan_final_h5_members(
    path: Path,
    *,
    contract: FinalH5CoverageContract,
    pool_h5_identity: PoolH5FileIdentity,
) -> tuple[FinalH5MemberDescriptor, ...]:
    members: list[FinalH5MemberDescriptor] = []
    expected_tables = frozenset(
        member.entity
        for member in contract.expected_members
        if member.kind is FinalH5MemberKind.ENTITY_TABLE
    )
    maximum_name_bytes = max(
        len(value.encode("utf-8"))
        for member in contract.expected_members
        for value in (member.entity, member.column)
        if value is not None
    )
    maximum_axis_bytes = len(contract.expected_members) * maximum_name_bytes
    try:
        with pd.HDFStore(path, mode="r") as store:
            _require_open_pool_h5_identity(store, pool_h5_identity)
            root_names = _validated_h5_root_names(
                store,
                expected_tables=expected_tables,
            )
            _validate_final_h5_headers(
                store,
                root_names=root_names,
            )
            for entity in sorted(root_names & expected_tables):
                key = f"/{entity}"
                entity = _entity_from_h5_key(key)
                columns = _fixed_frame_columns(
                    store,
                    key=key,
                    maximum_columns=len(contract.expected_members),
                    maximum_axis_bytes=maximum_axis_bytes,
                )
                members.append(
                    FinalH5MemberDescriptor(
                        kind=FinalH5MemberKind.ENTITY_TABLE,
                        entity=entity,
                    )
                )
                for column in columns:
                    kind = (
                        FinalH5MemberKind.WEIGHT_VECTOR
                        if column == f"{entity}_weight"
                        else FinalH5MemberKind.ENTITY_COLUMN
                    )
                    members.append(
                        FinalH5MemberDescriptor(
                            kind=kind,
                            entity=entity,
                            column=column,
                        )
                    )
            _require_open_pool_h5_identity(store, pool_h5_identity)
    except PoolArtifactCoverageError:
        raise
    except Exception as error:
        raise PoolArtifactCoverageError(
            f"cannot inspect final pool H5 logical-member closure: {path}"
        ) from error
    observed = _sort_final_h5_members(tuple(members))
    _validate_final_h5_members(
        observed,
        location=f"observed final H5 {path}",
        nonempty=False,
    )
    return observed


def _validate_final_h5_headers(
    store: pd.HDFStore,
    *,
    root_names: frozenset[str],
) -> None:
    required = {
        _entity_from_h5_key(H5_TIME_PERIOD_KEY),
        _entity_from_h5_key(H5_ARTIFACT_METADATA_KEY),
    }
    missing = required - root_names
    if missing:
        raise PoolArtifactCoverageError(
            f"final pool H5 is missing required header keys: {sorted(missing)}"
        )
    period_rows = _bounded_series_header(
        store,
        key=H5_TIME_PERIOD_KEY,
    )
    if not isinstance(period_rows, pd.Series) or len(period_rows) != 1:
        raise PoolArtifactCoverageError(
            "final pool H5 time-period header must be a one-row Series"
        )
    period = period_rows.iloc[0]
    if isinstance(period, np.integer):
        period = int(period)
    if not isinstance(period, int) or isinstance(period, bool):
        raise PoolArtifactCoverageError(
            "final pool H5 time-period header must contain one integer"
        )

    metadata_rows = _bounded_series_header(
        store,
        key=H5_ARTIFACT_METADATA_KEY,
    )
    if not isinstance(metadata_rows, pd.Series) or len(metadata_rows) != 1:
        raise PoolArtifactCoverageError(
            "final pool H5 artifact-metadata header must be a one-row Series"
        )
    metadata_text = metadata_rows.iloc[0]
    if not isinstance(metadata_text, str):
        raise PoolArtifactCoverageError(
            "final pool H5 artifact-metadata header must contain JSON text"
        )
    try:
        _strict_json_object(metadata_text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise PoolArtifactCoverageError(
            "final pool H5 artifact-metadata header must contain a strict JSON object"
        ) from error


def _fixed_frame_columns(
    store: pd.HDFStore,
    *,
    key: str,
    maximum_columns: int,
    maximum_axis_bytes: int,
) -> tuple[str, ...]:
    storer = store.get_storer(key)
    if (
        storer is None
        or storer.is_table
        or getattr(storer.attrs, "pandas_type", None) != "frame"
    ):
        raise PoolArtifactCoverageError(
            f"unexpected non-fixed-DataFrame key in final pool H5: {key!r}"
        )
    nblocks = getattr(storer.attrs, "nblocks", None)
    if (
        isinstance(nblocks, bool)
        or not isinstance(nblocks, int | np.integer)
        or nblocks < 1
        or nblocks > maximum_columns
    ):
        raise PoolArtifactCoverageError(
            f"final pool H5 key {key!r} has invalid fixed-frame block cardinality"
        )
    expected_nodes = {
        "axis0",
        "axis1",
        *(
            node_name
            for block in range(int(nblocks))
            for node_name in (f"block{block}_items", f"block{block}_values")
        ),
    }
    _require_exact_group_leaves(
        storer.group,
        expected_names=expected_nodes,
        location=f"final pool H5 key {key!r}",
    )
    axis = storer.group.axis0
    shape = tuple(axis.shape)
    size_in_memory = axis.size_in_memory
    if (
        len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int | np.integer)
        or shape[0] < 0
        or shape[0] > maximum_columns
        or isinstance(size_in_memory, bool)
        or not isinstance(size_in_memory, int | np.integer)
        or size_in_memory < 0
        or size_in_memory > maximum_axis_bytes
    ):
        raise PoolArtifactCoverageError(
            f"final pool H5 key {key!r} column axis exceeds its sealed memory bound"
        )
    try:
        index = storer.read_index("axis0")
    except Exception as error:
        raise PoolArtifactCoverageError(
            f"cannot read fixed-DataFrame column axis for final pool H5 key {key!r}"
        ) from error
    columns = tuple(index.tolist())
    if not all(isinstance(column, str) and column for column in columns):
        raise PoolArtifactCoverageError(
            f"final pool H5 key {key!r} has a non-string or empty column name"
        )
    if len(columns) != len(set(columns)):
        raise PoolArtifactCoverageError(
            f"final pool H5 key {key!r} has duplicate column names"
        )
    for column in columns:
        if unicodedata.normalize("NFC", column) != column:
            raise PoolArtifactCoverageError(
                f"final pool H5 key {key!r} has a non-NFC column name"
            )
    return columns


def _validated_h5_root_names(
    store: pd.HDFStore,
    *,
    expected_tables: frozenset[str],
) -> frozenset[str]:
    header_names = {
        _entity_from_h5_key(H5_TIME_PERIOD_KEY),
        _entity_from_h5_key(H5_ARTIFACT_METADATA_KEY),
    }
    allowed_names = expected_tables | header_names
    observed: set[str] = set()
    for node in store._handle.root._f_iter_nodes():
        name = node._v_name
        if not isinstance(name, str) or not name:
            raise PoolArtifactCoverageError(
                "final pool H5 exposes a malformed physical root-node name"
            )
        if name not in allowed_names:
            raise PoolArtifactCoverageError(
                f"unmodelled physical root node in final pool H5: {name!r}"
            )
        if name in observed:
            raise PoolArtifactCoverageError(
                f"duplicate physical root node in final pool H5: {name!r}"
            )
        if not isinstance(node, tables.Group):
            raise PoolArtifactCoverageError(
                f"final pool H5 root node must be a pandas group: {name!r}"
            )
        observed.add(name)
    return frozenset(observed)


def _bounded_series_header(
    store: pd.HDFStore,
    *,
    key: str,
) -> pd.Series:
    storer = store.get_storer(key)
    if (
        storer is None
        or not storer.is_table
        or getattr(storer.attrs, "pandas_type", None) != "series_table"
        or storer.nrows != 1
    ):
        raise PoolArtifactCoverageError(
            f"final pool H5 header {key!r} must be a one-row table Series"
        )
    _require_exact_group_leaves(
        storer.group,
        expected_names={"table"},
        location=f"final pool H5 header {key!r}",
    )
    size_in_memory = storer.table.size_in_memory
    if (
        isinstance(size_in_memory, bool)
        or not isinstance(size_in_memory, int | np.integer)
        or size_in_memory < 0
        or size_in_memory > _FINAL_H5_HEADER_MAX_BYTES
    ):
        raise PoolArtifactCoverageError(
            f"final pool H5 header {key!r} exceeds its sealed memory bound"
        )
    rows = store.select(key, start=0, stop=1)
    if not isinstance(rows, pd.Series) or len(rows) != 1:
        raise PoolArtifactCoverageError(
            f"final pool H5 header {key!r} did not decode as one Series row"
        )
    return rows


def _require_exact_group_leaves(
    group: tables.Group,
    *,
    expected_names: set[str],
    location: str,
) -> None:
    observed: set[str] = set()
    for node in group._f_iter_nodes():
        name = node._v_name
        if name not in expected_names:
            raise PoolArtifactCoverageError(
                f"{location} contains an unmodelled physical child node: {name!r}"
            )
        if name in observed or not isinstance(node, tables.Leaf):
            raise PoolArtifactCoverageError(
                f"{location} contains a duplicate or nested physical node: {name!r}"
            )
        observed.add(name)
    if observed != expected_names:
        raise PoolArtifactCoverageError(
            f"{location} physical child inventory differs: "
            f"missing={sorted(expected_names - observed)}, "
            f"extra={sorted(observed - expected_names)}"
        )


def _require_open_pool_h5_identity(
    store: pd.HDFStore,
    expected: PoolH5FileIdentity,
) -> None:
    try:
        descriptor = store._handle.fileno()
        status = os.fstat(descriptor)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise PoolArtifactCoverageError(
            "cannot bind the open pandas H5 handle to its captured file identity"
        ) from error
    if not stat.S_ISREG(status.st_mode) or _stat_identity_key(status) != (
        expected.device,
        expected.inode,
        expected.size_bytes,
        expected.mtime_ns,
        expected.ctime_ns,
    ):
        raise PoolArtifactCoverageError(
            "pandas opened a different final pool H5 than the pre-collection file"
        )


def _entity_from_h5_key(key: object) -> str:
    if (
        not isinstance(key, str)
        or not key.startswith("/")
        or key.count("/") != 1
        or len(key) == 1
    ):
        raise PoolArtifactCoverageError(
            f"unexpected nested or malformed final pool H5 key: {key!r}"
        )
    entity = key[1:]
    if unicodedata.normalize("NFC", entity) != entity:
        raise PoolArtifactCoverageError(f"final pool H5 entity key is not NFC: {key!r}")
    return entity


def _strict_json_object(raw: str) -> dict[str, object]:
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
        raw,
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _absolute_pool_h5_path(value: str | Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as error:
        raise PoolArtifactCoverageError(
            f"invalid final pool H5 path: {value!r}"
        ) from error


def _require_nonsymlink_regular_path(path: Path) -> None:
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current /= part
        is_target = index == len(path.parts[1:]) - 1
        try:
            status = os.lstat(current)
        except OSError as error:
            raise PoolArtifactCoverageError(
                f"required final pool H5 path is unavailable: {current}: {error}"
            ) from error
        if stat.S_ISLNK(status.st_mode):
            raise PoolArtifactCoverageError(
                f"symbolic link in final pool H5 path is forbidden: {current}"
            )
        if is_target and not stat.S_ISREG(status.st_mode):
            raise PoolArtifactCoverageError(
                f"final pool H5 must be a regular file: {current}"
            )
        if not is_target and not stat.S_ISDIR(status.st_mode):
            raise PoolArtifactCoverageError(
                f"final pool H5 parent must be a directory: {current}"
            )


def _stat_identity_key(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _final_h5_members_from_inventory(
    inventory: object,
) -> tuple[FinalH5MemberDescriptor, ...]:
    try:
        rows = canonical_final_h5_member_descriptors(inventory)
    except FinalH5InventoryError as error:
        raise PoolArtifactCoverageError(
            "cannot expand final H5 member inventory"
        ) from error
    members = tuple(
        FinalH5MemberDescriptor(
            kind=FinalH5MemberKind(row["kind"]),
            entity=row["entity"],
            column=row.get("column"),
        )
        for row in rows
    )
    _validate_final_h5_members(
        members,
        location="compiled final H5 members",
        nonempty=True,
    )
    return members


def _validate_final_h5_members(
    members: tuple[FinalH5MemberDescriptor, ...],
    *,
    location: str,
    nonempty: bool,
) -> None:
    if not isinstance(members, tuple) or not all(
        isinstance(member, FinalH5MemberDescriptor) for member in members
    ):
        raise PoolArtifactCoverageError(f"{location}: typed member tuple required")
    if nonempty and not members:
        raise PoolArtifactCoverageError(f"{location}: must not be empty")
    if members != _sort_final_h5_members(members):
        raise PoolArtifactCoverageError(f"{location}: canonical member order required")
    keys = tuple(_final_h5_member_key(member) for member in members)
    if len(keys) != len(set(keys)):
        raise PoolArtifactCoverageError(f"{location}: duplicate members forbidden")


def _sort_final_h5_members(
    members: tuple[FinalH5MemberDescriptor, ...],
) -> tuple[FinalH5MemberDescriptor, ...]:
    return tuple(
        sorted(
            members,
            key=lambda member: (
                _FINAL_H5_KIND_ORDER[member.kind],
                member.entity,
                member.column or "",
            ),
        )
    )


def _final_h5_member_key(
    member: FinalH5MemberDescriptor,
) -> tuple[FinalH5MemberKind, str, str | None]:
    return member.kind, member.entity, member.column


def _inventory_string(inventory: FrozenMap, field: str) -> str:
    return _nonempty_string(
        inventory.get(field),
        location=f"final H5 member inventory/{field}",
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
    "FinalH5CoverageContract",
    "FinalH5CoverageResult",
    "FinalH5MemberDescriptor",
    "FinalH5MemberKind",
    "PoolArtifactCoverageContract",
    "PoolArtifactCoverageError",
    "PoolArtifactCoverageReceipt",
    "PoolH5FileIdentity",
    "TargetBankCoverageContract",
    "TargetBankCoverageResult",
    "TargetBankKind",
    "capture_pool_h5_file_identity",
    "compile_pool_artifact_coverage",
    "validate_pool_artifact_coverage",
]
