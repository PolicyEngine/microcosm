"""Plan-derived member closure for US pool target-bank artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from microcosm.build.spec_engine import (
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.us_runtime.pool_artifact_coverage import (
    ArtifactMemberDescriptor,
    ArtifactMemberKind,
    CoverageStatus,
    CoverageUnsupportedReason,
    PoolArtifactCoverageContract,
    PoolArtifactCoverageError,
    TargetBankKind,
    compile_pool_artifact_coverage,
    validate_pool_artifact_coverage,
)
from microcosm.build.us_runtime.pool_kernel_authority import (
    USPoolKernelAuthorities,
)
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.spec_authority import compile_us_spec_authority


@pytest.fixture(scope="module")
def plan() -> USPoolRuntimePlan:
    return USPoolRuntimePlan.from_spec_authority(
        compile_us_spec_authority(
            compile_runtime_authorities(compile_spec(load_bundle("us")))
        )
    )


@pytest.fixture(scope="module")
def authorities(plan: USPoolRuntimePlan) -> USPoolKernelAuthorities:
    return USPoolKernelAuthorities.from_runtime_plan(plan)


@pytest.fixture(scope="module")
def coverage_contract(
    plan: USPoolRuntimePlan,
    authorities: USPoolKernelAuthorities,
) -> PoolArtifactCoverageContract:
    return compile_pool_artifact_coverage(plan, authorities)


def _materialize_target_banks(
    root: Path,
    contract: PoolArtifactCoverageContract,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for bank_index, bank in enumerate(contract.target_banks):
        bank_root = root / f"bank-{bank_index:02d}"
        bank_root.mkdir(parents=True)
        for member in bank.expected_members:
            path = bank_root / member.relative_path
            if member.kind is ArtifactMemberKind.DIRECTORY:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
        roots[bank.locator_ref] = bank_root
    return roots


def _result_by_locator(receipt, locator_ref: str):
    return next(row for row in receipt.target_banks if row.locator_ref == locator_ref)


def test_compiles_exact_bank_inventory_and_refuses_to_invent_h5_closure(
    plan: USPoolRuntimePlan,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    directory_rows = tuple(
        row
        for row in plan.execution.artifact_vector
        if row.get("content_selector_ref") == "selector:directory_tree_bytes_v1"
    )
    assert tuple(bank.locator_ref for bank in coverage_contract.target_banks) == tuple(
        row.get("locator_ref") for row in directory_rows
    )
    assert Counter(bank.bank_kind for bank in coverage_contract.target_banks) == {
        TargetBankKind.GAP_FILL: 2,
        TargetBankKind.PRIMARY_QRF: 1,
        TargetBankKind.LATE_TRANSFER: 19,
    }

    primary = next(
        bank
        for bank in coverage_contract.target_banks
        if bank.bank_kind is TargetBankKind.PRIMARY_QRF
    )
    primary_paths = {member.relative_path for member in primary.expected_members}
    assert {
        "targets",
        "manifest.json",
        "donor.frame.h5",
        "recipient.frame.h5",
        "late-producer-input-binding.json",
    }.issubset(primary_paths)
    assert len(primary.expected_members) == 70

    immigration = next(
        bank
        for bank in coverage_contract.target_banks
        if bank.authority_ref
        == "producer_node:transfer:person/source_operator_immigration"
    )
    immigration_files = tuple(
        member
        for member in immigration.expected_members
        if member.kind is ArtifactMemberKind.FILE
    )
    assert len(immigration_files) == 1
    assert immigration_files[0].relative_path.endswith(
        "__acs_transfer_immigration_status_pair.h5"
    )

    assert coverage_contract.final_pool_h5.status is CoverageStatus.UNSUPPORTED
    assert coverage_contract.final_pool_h5.unsupported_reason is (
        CoverageUnsupportedReason.FINAL_H5_INVENTORY_NOT_COMPILED
    )
    assert coverage_contract.final_pool_h5.artifact_ids == (
        "final_pool_entity_tables",
        "final_pool_weight_vectors",
    )


def test_matching_bank_members_complete_only_the_supported_bank_surface(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path, coverage_contract)

    receipt = validate_pool_artifact_coverage(
        coverage_contract,
        bank_roots=roots,
    )

    assert receipt.bank_member_coverage_complete is True
    assert receipt.container_member_coverage_complete is False
    assert receipt.status is CoverageStatus.UNSUPPORTED
    assert all(row.complete for row in receipt.target_banks)
    wire = receipt.to_wire()
    assert wire["receipt_sha256"] == receipt.receipt_sha256
    assert json.loads(json.dumps(wire, allow_nan=False)) == wire


def test_missing_bank_member_is_reported_exactly(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path, coverage_contract)
    bank = coverage_contract.target_banks[0]
    missing = next(
        member
        for member in bank.expected_members
        if member.kind is ArtifactMemberKind.FILE
    )
    (roots[bank.locator_ref] / missing.relative_path).unlink()

    receipt = validate_pool_artifact_coverage(
        coverage_contract,
        bank_roots=roots,
    )

    result = _result_by_locator(receipt, bank.locator_ref)
    assert result.status is CoverageStatus.INCOMPLETE
    assert result.missing_members == (missing,)
    assert result.extra_members == ()
    assert receipt.bank_member_coverage_complete is False


def test_extra_bank_member_is_reported_exactly(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path, coverage_contract)
    bank = coverage_contract.target_banks[-1]
    (roots[bank.locator_ref] / "unexpected.bin").write_bytes(b"unexpected")

    receipt = validate_pool_artifact_coverage(
        coverage_contract,
        bank_roots=roots,
    )

    result = _result_by_locator(receipt, bank.locator_ref)
    assert result.status is CoverageStatus.INCOMPLETE
    assert result.missing_members == ()
    assert result.extra_members == (
        ArtifactMemberDescriptor("unexpected.bin", ArtifactMemberKind.FILE),
    )
    assert receipt.bank_member_coverage_complete is False


def test_refuses_incomplete_locator_bindings(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path, coverage_contract)
    roots.pop(next(iter(roots)))

    with pytest.raises(
        PoolArtifactCoverageError,
        match="locator binding inventory mismatch",
    ):
        validate_pool_artifact_coverage(coverage_contract, bank_roots=roots)


def test_refuses_symlink_member_as_malformed_evidence(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path, coverage_contract)
    bank = coverage_contract.target_banks[0]
    member = next(
        item for item in bank.expected_members if item.kind is ArtifactMemberKind.FILE
    )
    member_path = roots[bank.locator_ref] / member.relative_path
    member_path.unlink()
    member_path.symlink_to(roots[bank.locator_ref] / "targets")

    with pytest.raises(PoolArtifactCoverageError, match="symbolic link"):
        validate_pool_artifact_coverage(coverage_contract, bank_roots=roots)
