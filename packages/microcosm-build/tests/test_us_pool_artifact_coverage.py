"""Plan-derived member closure for US pool target-bank artifacts."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import tables

import microcosm.build.us_runtime.pool_artifact_coverage as coverage_module
from microcosm.build.spec_engine import (
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.us_runtime.h5_io import (
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.pool_artifact_coverage import (
    ArtifactMemberDescriptor,
    ArtifactMemberKind,
    CoverageStatus,
    FinalH5MemberDescriptor,
    FinalH5MemberKind,
    PoolArtifactCoverageContract,
    PoolArtifactCoverageError,
    TargetBankKind,
    capture_pool_h5_file_identity,
    compile_pool_artifact_coverage,
    validate_pool_artifact_coverage,
)
from microcosm.build.us_runtime.pool_kernel_authority import (
    USPoolKernelAuthorities,
)
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.spec_authority import compile_us_spec_authority
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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


def _materialize_final_h5(
    path: Path,
    contract: PoolArtifactCoverageContract,
    *,
    omitted_member: FinalH5MemberDescriptor | None = None,
    extra_column: tuple[str, str] | None = None,
    extra_series_key: str | None = None,
    include_period_header: bool = True,
    period_value: object = 2024,
    metadata_text: str = '{"artifact_kind":"fixture"}',
) -> Path:
    final = contract.final_pool_h5
    tables = {
        member.entity: []
        for member in final.expected_members
        if member.kind is FinalH5MemberKind.ENTITY_TABLE and member != omitted_member
    }
    for member in final.expected_members:
        if member == omitted_member or member.kind is FinalH5MemberKind.ENTITY_TABLE:
            continue
        if member.entity not in tables:
            continue
        assert member.column is not None
        tables[member.entity].append(member.column)
    if extra_column is not None:
        entity, column = extra_column
        tables[entity].append(column)

    with pd.HDFStore(path, mode="w") as store:
        for entity, columns in tables.items():
            table = pd.DataFrame([[0] * len(columns)], columns=columns)
            store.put(entity, table, format="fixed")
        if include_period_header:
            store.put("_time_period", pd.Series([period_value]), format="table")
        store.put(
            "_populace_staging_metadata",
            pd.Series([metadata_text]),
            format="table",
            index=False,
        )
        if extra_series_key is not None:
            store.put(extra_series_key, pd.Series([1]), format="table")
    return path


def _materialize_complete_inputs(
    tmp_path: Path,
    contract: PoolArtifactCoverageContract,
) -> tuple[dict[str, Path], Path]:
    roots = _materialize_target_banks(tmp_path / "banks", contract)
    pool_h5 = _materialize_final_h5(tmp_path / "pool.h5", contract)
    return roots, pool_h5


def _validate_complete_path(
    contract: PoolArtifactCoverageContract,
    *,
    roots: dict[str, Path],
    pool_h5: Path,
):
    return validate_pool_artifact_coverage(
        contract,
        bank_roots=roots,
        pool_h5=pool_h5,
        pool_h5_identity=capture_pool_h5_file_identity(pool_h5),
    )


def test_compiles_exact_bank_and_final_h5_inventory(
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

    final_h5 = coverage_contract.final_pool_h5
    assert final_h5.artifact_ids == (
        "final_pool_entity_tables",
        "final_pool_weight_vectors",
    )
    assert len(final_h5.expected_members) == 398
    assert Counter(member.kind for member in final_h5.expected_members) == {
        FinalH5MemberKind.ENTITY_TABLE: 6,
        FinalH5MemberKind.ENTITY_COLUMN: 391,
        FinalH5MemberKind.WEIGHT_VECTOR: 1,
    }
    assert final_h5.member_inventory.get("member_count") == 398
    assert (
        final_h5.member_inventory.get("members_sha256")
        == final_h5.expected_members_sha256
    )


def test_matching_bank_and_final_h5_members_complete_container_coverage(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots, pool_h5 = _materialize_complete_inputs(tmp_path, coverage_contract)
    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    assert receipt.bank_member_coverage_complete is True
    assert receipt.final_pool_h5.complete is True
    assert receipt.container_member_coverage_complete is True
    assert receipt.status is CoverageStatus.COMPLETE
    assert all(row.complete for row in receipt.target_banks)
    wire = receipt.to_wire()
    assert wire["receipt_sha256"] == receipt.receipt_sha256
    assert json.loads(json.dumps(wire, allow_nan=False)) == wire


def test_production_h5_writer_matches_compiler_issued_member_inventory(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    table_columns: dict[str, list[str]] = {}
    for member in coverage_contract.final_pool_h5.expected_members:
        if member.kind is FinalH5MemberKind.ENTITY_TABLE:
            table_columns[member.entity] = []
        elif member.kind is FinalH5MemberKind.ENTITY_COLUMN:
            assert member.column is not None
            table_columns[member.entity].append(member.column)
    tables = {
        entity: pd.DataFrame(
            {
                column: pd.Series([0], dtype="int64")
                for column in entity_columns
            }
        )
        for entity, entity_columns in table_columns.items()
    }
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=pd.Series([1.0], dtype="float64").to_numpy(),
                kind=WeightKind.IMPORTANCE,
            )
        },
    )
    pool_h5 = tmp_path / "production-writer.pool.h5"
    write_nullable_us_h5(
        frame,
        pool_h5,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        publication_run_id="final-h5-closure-fixture",
        materializer_version=US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
    )
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    assert receipt.final_pool_h5.complete is True
    assert receipt.final_pool_h5.observed_members == (
        coverage_contract.final_pool_h5.expected_members
    )


def test_missing_bank_member_is_reported_exactly(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots, pool_h5 = _materialize_complete_inputs(tmp_path, coverage_contract)
    bank = coverage_contract.target_banks[0]
    missing = next(
        member
        for member in bank.expected_members
        if member.kind is ArtifactMemberKind.FILE
    )
    (roots[bank.locator_ref] / missing.relative_path).unlink()

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
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
    roots, pool_h5 = _materialize_complete_inputs(tmp_path, coverage_contract)
    bank = coverage_contract.target_banks[-1]
    (roots[bank.locator_ref] / "unexpected.bin").write_bytes(b"unexpected")

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
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
    roots, pool_h5 = _materialize_complete_inputs(tmp_path, coverage_contract)
    roots.pop(next(iter(roots)))

    with pytest.raises(
        PoolArtifactCoverageError,
        match="locator binding inventory mismatch",
    ):
        validate_pool_artifact_coverage(
            coverage_contract,
            bank_roots=roots,
            pool_h5=pool_h5,
            pool_h5_identity=capture_pool_h5_file_identity(pool_h5),
        )


def test_refuses_symlink_member_as_malformed_evidence(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots, pool_h5 = _materialize_complete_inputs(tmp_path, coverage_contract)
    bank = coverage_contract.target_banks[0]
    member = next(
        item for item in bank.expected_members if item.kind is ArtifactMemberKind.FILE
    )
    member_path = roots[bank.locator_ref] / member.relative_path
    member_path.unlink()
    member_path.symlink_to(roots[bank.locator_ref] / "targets")

    with pytest.raises(PoolArtifactCoverageError, match="symbolic link"):
        validate_pool_artifact_coverage(
            coverage_contract,
            bank_roots=roots,
            pool_h5=pool_h5,
            pool_h5_identity=capture_pool_h5_file_identity(pool_h5),
        )


def test_missing_final_h5_column_is_reported_exactly(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    missing = next(
        member
        for member in coverage_contract.final_pool_h5.expected_members
        if member.kind is FinalH5MemberKind.ENTITY_COLUMN
    )
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        omitted_member=missing,
    )

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    assert receipt.final_pool_h5.status is CoverageStatus.INCOMPLETE
    assert receipt.final_pool_h5.missing_members == (missing,)
    assert receipt.final_pool_h5.extra_members == ()
    assert receipt.container_member_coverage_complete is False


def test_missing_final_h5_weight_is_reported_exactly(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    missing = next(
        member
        for member in coverage_contract.final_pool_h5.expected_members
        if member.kind is FinalH5MemberKind.WEIGHT_VECTOR
    )
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        omitted_member=missing,
    )

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    assert receipt.final_pool_h5.status is CoverageStatus.INCOMPLETE
    assert receipt.final_pool_h5.missing_members == (missing,)
    assert receipt.final_pool_h5.extra_members == ()


def test_extra_final_h5_column_is_reported_exactly(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    entity = coverage_contract.final_pool_h5.expected_members[0].entity
    extra = FinalH5MemberDescriptor(
        kind=FinalH5MemberKind.ENTITY_COLUMN,
        entity=entity,
        column="unexpected_compiler_undeclared_column",
    )
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        extra_column=(entity, extra.column),
    )

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    assert receipt.final_pool_h5.status is CoverageStatus.INCOMPLETE
    assert receipt.final_pool_h5.missing_members == ()
    assert receipt.final_pool_h5.extra_members == (extra,)


def test_missing_final_h5_table_reports_table_and_all_declared_children(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    missing_table = next(
        member
        for member in coverage_contract.final_pool_h5.expected_members
        if member.kind is FinalH5MemberKind.ENTITY_TABLE
    )
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        omitted_member=missing_table,
    )

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    missing = receipt.final_pool_h5.missing_members
    assert missing_table in missing
    assert {member.entity for member in missing} == {missing_table.entity}
    assert len(missing) > 1


def test_refuses_unmodelled_non_dataframe_hdf_root(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        extra_series_key="unexpected_series",
    )

    with pytest.raises(PoolArtifactCoverageError, match="unmodelled physical root"):
        _validate_complete_path(
            coverage_contract,
            roots=roots,
            pool_h5=pool_h5,
        )


@pytest.mark.parametrize(
    ("include_period_header", "period_value", "metadata_text", "message"),
    [
        (False, 2024, '{"artifact_kind":"fixture"}', "missing required header"),
        (True, "2024", '{"artifact_kind":"fixture"}', "one integer"),
        (True, 2024, "[]", "strict JSON object"),
        (True, 2024, '{"duplicate":1,"duplicate":2}', "strict JSON object"),
        (True, 2024, '{"nonfinite":NaN}', "strict JSON object"),
    ],
)
def test_refuses_missing_or_malformed_final_h5_headers(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
    include_period_header: bool,
    period_value: object,
    metadata_text: str,
    message: str,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        include_period_header=include_period_header,
        period_value=period_value,
        metadata_text=metadata_text,
    )

    with pytest.raises(PoolArtifactCoverageError, match=message):
        _validate_complete_path(
            coverage_contract,
            roots=roots,
            pool_h5=pool_h5,
        )


def test_pool_h5_identity_capture_refuses_symlink_path(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    real = _materialize_final_h5(tmp_path / "real.h5", coverage_contract)
    link = tmp_path / "link.h5"
    link.symlink_to(real)

    with pytest.raises(PoolArtifactCoverageError, match="symbolic link"):
        capture_pool_h5_file_identity(link)


def test_pool_h5_identity_rejects_replacement_after_capture(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(tmp_path / "pool.h5", coverage_contract)
    identity = capture_pool_h5_file_identity(pool_h5)
    replacement = _materialize_final_h5(
        tmp_path / "replacement.h5",
        coverage_contract,
    )
    os.replace(replacement, pool_h5)

    with pytest.raises(PoolArtifactCoverageError, match="changed since"):
        validate_pool_artifact_coverage(
            coverage_contract,
            bank_roots=roots,
            pool_h5=pool_h5,
            pool_h5_identity=identity,
        )


def test_pool_h5_identity_rejects_mutation_during_schema_scan(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(tmp_path / "pool.h5", coverage_contract)
    identity = capture_pool_h5_file_identity(pool_h5)
    real_hdf_store = pd.HDFStore

    class MutatingHDFStore:
        def __init__(self, *args, **kwargs) -> None:
            self._store = real_hdf_store(*args, **kwargs)

        def __enter__(self):
            return self._store.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            result = self._store.__exit__(exc_type, exc_value, traceback)
            status = pool_h5.stat()
            os.utime(
                pool_h5,
                ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000),
            )
            return result

    monkeypatch.setattr(coverage_module.pd, "HDFStore", MutatingHDFStore)

    with pytest.raises(PoolArtifactCoverageError, match="changed since"):
        validate_pool_artifact_coverage(
            coverage_contract,
            bank_roots=roots,
            pool_h5=pool_h5,
            pool_h5_identity=identity,
        )


def test_pool_h5_open_descriptor_must_match_precollection_identity(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(tmp_path / "pool.h5", coverage_contract)
    replacement = _materialize_final_h5(
        tmp_path / "replacement.h5",
        coverage_contract,
    )
    identity = capture_pool_h5_file_identity(pool_h5)
    real_hdf_store = pd.HDFStore
    monkeypatch.setattr(
        coverage_module.pd,
        "HDFStore",
        lambda *_args, **_kwargs: real_hdf_store(replacement, mode="r"),
    )

    with pytest.raises(PoolArtifactCoverageError, match="pandas opened a different"):
        validate_pool_artifact_coverage(
            coverage_contract,
            bank_roots=roots,
            pool_h5=pool_h5,
            pool_h5_identity=identity,
        )


@pytest.mark.parametrize("nested", [False, True])
def test_refuses_raw_pytables_nodes_hidden_from_pandas_keys(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
    nested: bool,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    pool_h5 = _materialize_final_h5(tmp_path / "pool.h5", coverage_contract)
    with tables.open_file(pool_h5, mode="a") as handle:
        if nested:
            handle.create_array("/household", "rogue_payload", [1])
        else:
            group = handle.create_group("/", "rogue_raw_group")
            handle.create_array(group, "payload", [1])

    with pytest.raises(
        PoolArtifactCoverageError,
        match="unmodelled physical (root|child) node",
    ):
        _validate_complete_path(
            coverage_contract,
            roots=roots,
            pool_h5=pool_h5,
        )


def test_refuses_column_axis_before_loading_unsealed_oversized_metadata(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    entity = coverage_contract.final_pool_h5.expected_members[0].entity
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        extra_column=(entity, "x" * 30_000),
    )

    with pytest.raises(PoolArtifactCoverageError, match="sealed memory bound"):
        _validate_complete_path(
            coverage_contract,
            roots=roots,
            pool_h5=pool_h5,
        )


def test_refuses_oversized_header_before_materializing_json(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
) -> None:
    roots = _materialize_target_banks(tmp_path / "banks", coverage_contract)
    metadata = '{"payload":"' + (
        "x" * coverage_module._FINAL_H5_HEADER_MAX_BYTES
    ) + '"}'
    pool_h5 = _materialize_final_h5(
        tmp_path / "pool.h5",
        coverage_contract,
        metadata_text=metadata,
    )

    with pytest.raises(PoolArtifactCoverageError, match="sealed memory bound"):
        _validate_complete_path(
            coverage_contract,
            roots=roots,
            pool_h5=pool_h5,
        )


def test_final_h5_scan_never_materializes_entity_values(
    tmp_path: Path,
    coverage_contract: PoolArtifactCoverageContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots, pool_h5 = _materialize_complete_inputs(tmp_path, coverage_contract)
    real_hdf_store = pd.HDFStore
    selected: list[str] = []

    class MetadataOnlyHDFStore:
        def __init__(self, *args, **kwargs) -> None:
            self._store = real_hdf_store(*args, **kwargs)

        def __enter__(self):
            self._store.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._store.__exit__(exc_type, exc_value, traceback)

        def __getattr__(self, name: str):
            return getattr(self._store, name)

        def __getitem__(self, key: str):
            raise AssertionError(f"entity/header value materialization forbidden: {key}")

        def select(self, key: str, *args, **kwargs):
            assert key in {"/_time_period", "/_populace_staging_metadata"}
            selected.append(key)
            return self._store.select(key, *args, **kwargs)

    monkeypatch.setattr(coverage_module.pd, "HDFStore", MetadataOnlyHDFStore)

    receipt = _validate_complete_path(
        coverage_contract,
        roots=roots,
        pool_h5=pool_h5,
    )

    assert receipt.final_pool_h5.complete is True
    assert selected == ["/_time_period", "/_populace_staging_metadata"]
