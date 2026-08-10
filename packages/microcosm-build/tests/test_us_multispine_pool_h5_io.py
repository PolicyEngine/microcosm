from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.h5_io as h5_io
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.serialization_dtypes import (
    CANONICAL_STRING_DTYPE,
    canonicalize_frame_string_dtypes,
    canonicalize_table_string_dtypes,
)
from microcosm.build.us_runtime.h5_io import (
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    AuthenticatedPoolH5MismatchError,
    load_simulation_ready_us_multispine_pool,
    write_nullable_us_h5,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_frame() -> Frame:
    ids = np.asarray([10, 20, 30], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                US_SCHEMA.membership_column(entity): ids
                for entity in US_SCHEMA.group_entities
            },
            "nullable_input": np.asarray([True, None, False], dtype=object),
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 3.0, 5.0]),
                WeightKind.IMPORTANCE,
            )
        },
    )


def _pool_frame_with_object_strings_on_every_entity() -> Frame:
    """Match the assembled pool's object-backed source-string shape."""

    frame = _pool_frame()
    tables = {}
    for entity in frame.entities:
        table = frame.table(entity).copy()
        column = "PERIDNUM" if entity == "person" else f"{entity}_source_label"
        table.insert(
            0,
            column,
            pd.Series(
                [f"{entity}-0", None, f"{entity}-2"],
                index=table.index,
                dtype=object,
            ),
        )
        tables[entity] = table
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _semantic_string_columns(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in table.columns
        if isinstance(table[column].dtype, pd.StringDtype)
        or (
            pd.api.types.is_object_dtype(table[column].dtype)
            and pd.api.types.infer_dtype(table[column], skipna=True) == "string"
        )
    )


def test_string_canonicalization_reuses_unchanged_numeric_storage() -> None:
    source = pd.DataFrame(
        {
            "PERIDNUM": pd.Series(["1", None, "3"], dtype=object),
            "employment_income": np.asarray([10.0, 20.0, 30.0]),
        }
    )

    canonical = canonicalize_table_string_dtypes(
        source,
        boundary="fixture checkpoint load",
        table_name="person",
    )

    assert canonical is not source
    assert source["PERIDNUM"].dtype == np.dtype(object)
    assert canonical["PERIDNUM"].dtype == CANONICAL_STRING_DTYPE
    assert np.shares_memory(
        canonical["employment_income"].to_numpy(),
        source["employment_income"].to_numpy(),
    )


def test_in_place_frame_canonicalization_reuses_all_numeric_storage() -> None:
    frame = _pool_frame_with_object_strings_on_every_entity()
    numeric_storage = {
        entity: frame.table(entity)[US_SCHEMA.entity_id_column(entity)].to_numpy()
        for entity in US_SCHEMA.entities
    }

    canonical = canonicalize_frame_string_dtypes(
        frame,
        boundary="fixture checkpoint load",
        in_place=True,
    )

    assert canonical is frame
    for entity in US_SCHEMA.entities:
        table = canonical.table(entity)
        assert np.shares_memory(
            table[US_SCHEMA.entity_id_column(entity)].to_numpy(),
            numeric_storage[entity],
        )
        assert all(
            table[column].dtype == CANONICAL_STRING_DTYPE
            for column in _semantic_string_columns(table)
        )


def test_in_place_frame_canonicalization_is_atomic_on_ambiguity() -> None:
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("household")["household_source_label"] = pd.Series(
        ["household-0", 2, None],
        dtype=object,
    )

    with pytest.raises(TypeError, match="household.household_source_label"):
        canonicalize_frame_string_dtypes(
            frame,
            boundary="fixture checkpoint load",
            in_place=True,
        )

    assert frame.table("person")["PERIDNUM"].dtype == np.dtype(object)


def test_object_string_simulated_checkpoint_resume_exports_canonical_strings(
    tmp_path: Path,
) -> None:
    """Regress the exact production shape while proving loader symmetry."""

    pytest.importorskip("h5py")
    pytest.importorskip("tables")
    fresh = _pool_frame_with_object_strings_on_every_entity()
    assert fresh.table("person").columns[0] == "PERIDNUM"
    checkpoint_path = tmp_path / "simulated.checkpoint.h5"
    write_frame_checkpoint(
        checkpoint_path,
        fresh,
        metadata={"stage": "simulated"},
    )
    resumed = load_frame_checkpoint(checkpoint_path).frame

    for entity in US_SCHEMA.entities:
        string_columns = _semantic_string_columns(fresh.table(entity))
        assert string_columns
        for column in string_columns:
            assert fresh.table(entity)[column].dtype == np.dtype(object)
            assert resumed.table(entity)[column].dtype == np.dtype(object)

    for label, frame in (("fresh", fresh), ("resumed", resumed)):
        output = tmp_path / f"{label}.pool.h5"
        write_nullable_us_h5(
            frame,
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
            publication_run_id=f"{label}-fixture-publication",
        )
        # Fixed-format HDF persists the logical ``str`` dtype but not the
        # pandas storage backend; a raw read resolves storage from the
        # environment (python without pyarrow installed, pyarrow with).  Pin
        # the persisted logical dtype, then prove the load boundary restores
        # the exact canonical dtype in either environment.
        with pd.HDFStore(output, mode="r") as store:
            for entity in US_SCHEMA.entities:
                stored = store[entity]
                string_columns = _semantic_string_columns(stored)
                assert string_columns
                for column in string_columns:
                    dtype = stored[column].dtype
                    assert isinstance(dtype, pd.StringDtype)
                    assert dtype.na_value is np.nan
                canonical = canonicalize_table_string_dtypes(
                    stored,
                    boundary="raw pool store load",
                    table_name=entity,
                )
                assert all(
                    canonical[column].dtype == CANONICAL_STRING_DTYPE
                    for column in string_columns
                )
        with pd.option_context("mode.string_storage", "python"):
            with pd.HDFStore(output, mode="r") as store:
                for entity in US_SCHEMA.entities:
                    stored = store[entity]
                    assert all(
                        stored[column].dtype == CANONICAL_STRING_DTYPE
                        for column in _semantic_string_columns(stored)
                    )
        for entity in US_SCHEMA.entities:
            string_columns = _semantic_string_columns(frame.table(entity))
            assert all(
                frame.table(entity)[column].dtype == np.dtype(object)
                for column in string_columns
            )


def test_pool_export_rejects_ambiguous_object_strings_before_replacement(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("person")["PERIDNUM"] = pd.Series(
        ["person-0", 2, None],
        dtype=object,
    )
    output = tmp_path / "existing.pool.h5"
    output.write_bytes(b"previous-good-pool")

    with pytest.raises(
        TypeError,
        match=(
            "nullable US H5 export.*person.PERIDNUM.*"
            "offending value types.*builtins.int"
        ),
    ):
        write_nullable_us_h5(
            frame,
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        )

    assert output.read_bytes() == b"previous-good-pool"


def test_pool_export_rejects_untyped_all_missing_objects_before_replacement(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("household")["household_source_label"] = pd.Series(
        [None] * len(frame.table("household")),
        dtype=object,
    )
    output = tmp_path / "existing.pool.h5"
    output.write_bytes(b"previous-good-pool")

    with pytest.raises(
        TypeError,
        match=(
            "nullable US H5 export.*household.household_source_label.*"
            "no observed values.*declare an explicit dtype"
        ),
    ):
        write_nullable_us_h5(
            frame,
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        )

    assert output.read_bytes() == b"previous-good-pool"


def test_pool_export_canonicalizes_explicit_all_missing_strings(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("household")["household_source_label"] = pd.Series(
        pd.NA,
        index=frame.table("household").index,
        dtype=pd.StringDtype(storage="python", na_value=pd.NA),
    )
    output = tmp_path / "typed-all-missing.pool.h5"

    write_nullable_us_h5(
        frame,
        output,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    )

    with pd.HDFStore(output, mode="r") as store:
        stored = store["household"]["household_source_label"]
    assert isinstance(stored.dtype, pd.StringDtype)
    assert stored.dtype.na_value is np.nan
    assert stored.isna().all()
    with pd.option_context("mode.string_storage", "python"):
        with pd.HDFStore(output, mode="r") as store:
            repinned = store["household"]["household_source_label"]
    assert repinned.dtype == CANONICAL_STRING_DTYPE
    assert repinned.isna().all()


def test_pool_h5_load_boundary_canonicalizes_under_pyarrow_default(
    tmp_path: Path,
) -> None:
    """Raw read-back storage is environment-resolved; the load boundary owns
    the exact canonical dtype even under a pyarrow string-storage default."""

    pytest.importorskip("tables")
    pytest.importorskip("pyarrow")
    frame = _pool_frame_with_object_strings_on_every_entity()
    output = tmp_path / "pyarrow-default.pool.h5"
    write_nullable_us_h5(
        frame,
        output,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    )
    with pd.option_context("mode.string_storage", "pyarrow"):
        with pd.HDFStore(output, mode="r") as store:
            stored = store["person"]
        assert stored["PERIDNUM"].dtype != CANONICAL_STRING_DTYPE
        canonical = canonicalize_table_string_dtypes(
            stored,
            boundary="pyarrow-default pool load",
            table_name="person",
        )
    assert canonical["PERIDNUM"].dtype == CANONICAL_STRING_DTYPE
    assert canonical["PERIDNUM"].iloc[0] == "person-0"
    assert pd.isna(canonical["PERIDNUM"].iloc[1])


def _write_ready_pool(tmp_path: Path, *, stacked: bool = False) -> Path:
    run_id = "fixture-publication"
    pool_path = tmp_path / "pool.h5"
    diagnostics_path = tmp_path / "pool.agreement.json"
    manifest_path = tmp_path / "pool.manifest.json"
    agreement_gate = {
        "passed": True,
        "gates": {
            "us_spine_agreement": {
                "passed": True,
                "failures": [],
                "details": {"fixture": True},
            }
        },
    }
    schema_version = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION if stacked else 4
    write_nullable_us_h5(
        _pool_frame_with_object_strings_on_every_entity(),
        pool_path,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        publication_run_id=run_id,
    )
    diagnostics = {
        "artifact_kind": (US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND),
        "schema_version": schema_version,
        "simulation_ready": True,
        "publication_run_id": run_id,
        "agreement_gate": agreement_gate,
    }
    if stacked:
        diagnostics.update(
            {
                "pipeline": "us-stacked-pool",
                "semantic_kind": "stacked_terminal_gates",
                "terminal_gates": agreement_gate,
            }
        )
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest = {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": schema_version,
        "status": "simulation_ready",
        "simulation_ready": True,
        "publication_run_id": run_id,
        "period": 2024,
        "operator_order": [
            "assemble",
            "clone",
            "impute",
            "derive",
            "seed",
            "simulate",
            "agreement",
        ],
        "stage_receipts": {
            stage: {"operator": stage}
            for stage in ("impute", "derive", "seed", "simulate")
        },
        "stage_checkpoints": {
            "artifact_kind": "populace_us_multispine_pool_checkpoint_provenance",
            "schema_version": 1,
            "materializer_version": 3 if not stacked else 9,
            "enabled": False,
            "agreement": {
                "source": "always_fresh",
                "cached": False,
                "terminal_verdict_persisted": False,
            },
        },
        "agreement_gate": agreement_gate,
        "provenance_counts": {"household": {"rows": 3}},
        "pool_h5": {
            "path": str(pool_path.resolve()),
            "sha256": _sha256(pool_path),
            "size_bytes": pool_path.stat().st_size,
            "artifact_kind": US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
            "publication_run_id": run_id,
        },
        "agreement_diagnostics": {
            "path": str(diagnostics_path.resolve()),
            "sha256": _sha256(diagnostics_path),
            "publication_run_id": run_id,
        },
    }
    if stacked:
        dag = _canonical_stacked_late_dag_receipt()
        transition_authority = (
            stacked_spine_module._late_producer_transition_authority_receipt(dag)
        )
        manifest.update(
            {
                "pipeline": "us-stacked-pool",
                "late_producer_transition_authority_sha256": (
                    transition_authority["sha256"]
                ),
                "terminal_gates": agreement_gate,
                "operator_order": [
                    "assemble_stacked_spine",
                    "prepare_multispine_source_inputs_for_clone",
                    "gap_fill_stacked_spine",
                    "run_stacked_puf_pass",
                    "run_stacked_late_producer_dag",
                    "prepare_stacked_tail_derivation",
                    "derive_multispine_pool_inputs",
                    "seed_multispine_pool_inputs",
                    "materialize_multispine_agreement_outputs",
                    "stacked_completeness_gate",
                    "by_origin_battery",
                ],
                "stage_receipts": {
                    "impute": {
                        "source_operator_chain": {
                            "late_dag_completion": dag["source_completion"],
                        },
                        "stacked_late_producer_dag": dag,
                        "stacked_post_puf_transfer": dag["post_puf_transfer"],
                    }
                },
            }
        )
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest_path


def _canonical_stacked_late_dag_receipt() -> dict[str, object]:
    """Build a signed fixture receipt over the live canonical contracts."""

    schedule = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE
    schedule_receipt = stacked_spine_module._json_ready(
        stacked_spine_module.us_late_producer_schedule_receipt()
    )
    source_order = [
        producer.removeprefix("source:")
        for producer in schedule.order
        if producer.startswith("source:")
    ]
    source_receipts = {
        operator: {
            "phase": "post_clone",
            "operator_order": [operator],
            "cps_source_evidence": None,
            "suboperators": [{"operator": operator}],
        }
        for operator in source_order
    }
    source_completion = {
        "phase": "post_clone",
        "operator_order": source_order,
        "cps_source_evidence": None,
        "suboperators": [
            {"operator": operator, "order_index": index}
            for index, operator in enumerate(source_order)
        ],
        "deferred_transfer_inputs": {
            "inputs": {
                column: {}
                for column in (
                    "bank_account_assets",
                    "bond_assets",
                    "stock_assets",
                )
            }
        },
    }
    group_receipts = {
        group.name: {
            "producer": group.name,
            "entity": group.entity,
            "family": group.family,
            "ordered_targets": list(group.targets),
            "targets": {
                f"{group.entity}/{group.family}/{target}": {
                    "residual_null_rows": 0,
                }
                for target in group.targets
            },
        }
        for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
    }
    group_by_name = {
        group.name: group
        for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
    }
    canonical_family = {
        (entity, target): family
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    aggregate_targets = {
        f"{group.entity}/{canonical_family[(group.entity, target)]}/{target}": (
            group_receipts[group.name]["targets"][
                f"{group.entity}/{group.family}/{target}"
            ]
        )
        for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
        for target in group.targets
    }
    transfer = {
        "authority": dict(stacked_spine_module.stacked_spine_authority_receipt()),
        "producer_schedule": schedule_receipt,
        "producer_execution_order": [
            producer
            for producer in schedule.order
            if producer != stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
        ],
        "groups": group_receipts,
        "targets": aggregate_targets,
        "completion": {
            "status": "complete",
            "group_count": 19,
            "target_count": 70,
            "residual_null_rows": 0,
        },
    }
    input_frame_sha256 = "1" * 64
    previous_sha256 = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=schedule_receipt["payload_sha256"],
        input_frame_sha256=input_frame_sha256,
    )
    execution = []
    for index, producer_name in enumerate(schedule.order):
        contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
            producer_name
        ]
        if contract.kind == "acs_earnings_universe":
            available = (
                stacked_spine_module._late_acs_earnings_universe_resource_receipts()
            )
        elif contract.kind == "primary_puf":
            available = stacked_spine_module.stacked_late_primary_resource_receipts(
                pd.DataFrame({"fixture_donor": [1.0]}),
                primary_qrf_checkpoint_identity_sha256="5" * 64,
                clone_attachment_fraction=1.0,
                clone_attachment_seed=578,
                seed=0,
                n_estimators=100,
                fit_records_enabled=True,
                tail_bound_diagnostics_enabled=True,
            )
        elif contract.kind == "post_clone_source":
            available = stacked_spine_module._late_source_resource_receipts(
                producer_name=producer_name,
            )
        elif contract.kind == "source_finalizer":
            available = {
                f"person.@source_receipt:{operator}": (
                    stacked_spine_module._late_available_input_receipt(
                        producer=producer_name,
                        entity="person",
                        column=f"@source_receipt:{operator}",
                        rows=1,
                        binding={
                            "resource_kind": "source_operator_receipt",
                            "schema_version": 1,
                            "source_operator": operator,
                            "source_receipt_sha256": (
                                stacked_spine_module._canonical_sha256(source_receipt)
                            ),
                        },
                    )
                )
                for operator, source_receipt in source_receipts.items()
            }
        elif contract.kind == "late_transfer":
            group = group_by_name[producer_name]
            available = stacked_spine_module._late_transfer_resource_receipts(
                group_name=group.name,
                entity=group.entity,
                family=group.family,
                targets=group.targets,
                seed=0,
                n_estimators=100,
                max_targets_per_fit=(
                    stacked_spine_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                ),
                target_bank=None,
            )
        else:
            available = {}
        declared_inputs = []
        for requirement in contract.inputs:
            alternatives = []
            for alternative in requirement.alternatives:
                physical_evidence = []
                for column in alternative:
                    is_virtual = (
                        column.column.startswith("@")
                        and column.column != "@resolved_weight"
                        and column.entity != "frame"
                    )
                    key = f"{column.entity}.{column.column}"
                    resource_receipt = available.get(key) if is_virtual else None
                    present = not is_virtual or resource_receipt is not None
                    physical_evidence.append(
                        {
                            "entity": column.entity,
                            "column": column.column,
                            "value_kind": column.value_kind,
                            "required_scope": requirement.required_scope,
                            "scope_rows": 1,
                            "missing_rows": 0 if present else 1,
                            "invalid_rows": 0,
                            "status": "present" if present else "absent",
                            "content_sha256": (
                                stacked_spine_module._canonical_sha256(resource_receipt)
                                if resource_receipt is not None
                                else "2" * 64
                            ),
                            **(
                                {"weight_kind": "household_weight"}
                                if column.column == "@resolved_weight"
                                else {}
                            ),
                        }
                    )
                alternatives.append(physical_evidence)
            evidence = {"alternatives": alternatives}
            evidence["sha256"] = stacked_spine_module._canonical_sha256(evidence)
            declared_inputs.append(
                {
                    "entity": requirement.entity,
                    "column": requirement.column,
                    "required_scope": requirement.required_scope,
                    "producing_stage": requirement.producing_stage,
                    "unfilled_rows": 0,
                    "invalid_rows": 0,
                    "evidence": evidence,
                }
            )
        output_surface = [
            {
                "entity": output.entity,
                "column": output.column,
                "coverage_scope": output.coverage_scope,
                "status": "present",
                "content_sha256": "3" * 64,
                **({} if output.entity == "frame" else {"scope_rows": 1}),
                **(
                    {"weight_kind": "household_weight"}
                    if output.column == "@resolved_weight"
                    else {}
                ),
            }
            for output in contract.outputs
        ]
        if contract.kind == "acs_earnings_universe":
            producer_receipt = {"fixture": "acs_earnings_universe"}
        elif contract.kind == "primary_puf":
            producer_receipt = {
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(available)
                )
            }
        elif contract.kind == "post_clone_source":
            producer_receipt = source_receipts[producer_name.removeprefix("source:")]
        elif contract.kind == "source_finalizer":
            producer_receipt = source_completion
        elif contract.kind == "late_transfer":
            producer_receipt = group_receipts[group_by_name[producer_name].name]
        else:
            producer_receipt = {}
        for output in output_surface:
            if output["column"].startswith("@source_receipt:"):
                output["content_sha256"] = stacked_spine_module._canonical_sha256(
                    producer_receipt
                )
        row = {
            "execution_index": index,
            "producer": producer_name,
            "kind": contract.kind,
            "declared_inputs": declared_inputs,
            "declared_absence_receipts": {},
            "available_input_receipts": available,
            "input_surface_sha256": stacked_spine_module._canonical_sha256(
                declared_inputs
            ),
            "output_surface": output_surface,
            "output_surface_sha256": stacked_spine_module._canonical_sha256(
                output_surface
            ),
            "producer_receipt": producer_receipt,
            "producer_receipt_sha256": stacked_spine_module._canonical_sha256(
                producer_receipt
            ),
            "previous_execution_sha256": previous_sha256,
            "status": "complete",
        }
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous_sha256 = row["sha256"]
        execution.append(row)
    receipt = {
        "version": stacked_spine_module.US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "producer_schedule": schedule_receipt,
        "input_frame_sha256": input_frame_sha256,
        "output_frame_sha256": "4" * 64,
        "execution_chain_sha256": previous_sha256,
        "execution": execution,
        "source_completion": source_completion,
        "post_puf_transfer": transfer,
    }
    receipt["sha256"] = stacked_spine_module._canonical_sha256(receipt)
    stacked_spine_module.validate_stacked_late_producer_receipt(
        receipt,
        boundary="canonical stacked H5 fixture",
    )
    return receipt


def test_ready_pool_loader_preserves_importance_weights_and_nullable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    expected_manifest_sha256 = _sha256(manifest_path)
    original_read_bytes = Path.read_bytes
    manifest_reads = 0

    def replace_after_pinned_read(path: Path) -> bytes:
        nonlocal manifest_reads
        raw = original_read_bytes(path)
        if path == manifest_path:
            manifest_reads += 1
            replacement = json.loads(raw)
            replacement["publication_run_id"] = "replacement-publication"
            path.write_text(json.dumps(replacement), encoding="utf-8")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_pinned_read)

    frame, manifest, authenticated_pool_h5 = load_simulation_ready_us_multispine_pool(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    weights = frame.weights_for("household")
    assert weights.kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(weights.values, [2.0, 3.0, 5.0])
    assert frame.table("person")["nullable_input"].tolist() == [True, None, False]
    assert frame.n("household") == 3
    for entity in US_SCHEMA.entities:
        string_columns = _semantic_string_columns(frame.table(entity))
        assert string_columns
        assert all(
            frame.table(entity)[column].dtype == CANONICAL_STRING_DTYPE
            for column in string_columns
        )
    assert manifest["publication_run_id"] == "fixture-publication"
    assert authenticated_pool_h5.path == Path(manifest["pool_h5"]["path"])
    assert authenticated_pool_h5.sha256 == manifest["pool_h5"]["sha256"]
    assert authenticated_pool_h5.size_bytes == manifest["pool_h5"]["size_bytes"]
    assert authenticated_pool_h5.publication_run_id == "fixture-publication"
    assert authenticated_pool_h5.manifest_sha256 == expected_manifest_sha256
    assert "terminal_gates" not in manifest
    assert manifest_reads == 1
    assert json.loads(manifest_path.read_text())["publication_run_id"] == (
        "replacement-publication"
    )


def test_ready_legacy_pool_loader_accepts_pre_653_schema_four_envelope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(written_manifest["agreement_diagnostics"]["path"])
    written_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    frame, loaded_manifest, _authenticated_h5 = (
        load_simulation_ready_us_multispine_pool(manifest_path)
    )

    assert written_manifest["schema_version"] == 4
    assert written_diagnostics["schema_version"] == 4
    assert loaded_manifest["schema_version"] == 4
    assert frame.n("household") == 3


def test_ready_legacy_pool_loader_rejects_schema_six_envelope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    diagnostics["schema_version"] = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous stacked envelope"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_rejects_a_false_h5_size_receipt(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pool_h5"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pool_h5 size_bytes .* does not match"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_authenticated_pool_h5_copy_rejects_a_raced_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    _, _, authenticated_pool_h5 = load_simulation_ready_us_multispine_pool(
        manifest_path
    )
    replacement = bytearray(authenticated_pool_h5.path.read_bytes())
    replacement[0] ^= 1
    original_copy = h5_io._copy_file_bytes

    def replace_then_copy(source: Path, destination: Path) -> None:
        source.write_bytes(replacement)
        original_copy(source, destination)

    monkeypatch.setattr(h5_io, "_copy_file_bytes", replace_then_copy)
    destination = tmp_path / "audit" / "base_pool.h5"

    with pytest.raises(
        AuthenticatedPoolH5MismatchError,
        match="builder final local-audit copy.*copied bytes",
    ):
        authenticated_pool_h5.copy_verified_to(
            destination,
            consumer="builder final local-audit copy",
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))


def test_ready_pool_loader_reconciles_manifest_and_h5_household_counts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance_counts"]["household"]["rows"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="household row count 4.*H5 count 3"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_requires_explicitly_green_agreement_receipt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agreement_gate"]["passed"] = False
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["agreement_gate"]["passed"] = False
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no passing agreement-gate verdict"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_binds_diagnostics_agreement_verdict(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["agreement_gate"]["gates"]["us_spine_agreement"]["details"] = {
        "fixture": False
    }
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="verdict does not match"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_binds_terminal_gate_aliases(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)

    frame, manifest, _ = load_simulation_ready_us_multispine_pool(manifest_path)

    assert manifest["terminal_gates"] == manifest["agreement_gate"]
    transition_authority = frame.metadata[
        stacked_spine_module.US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY
    ]
    assert (
        transition_authority["sha256"]
        == manifest["late_producer_transition_authority_sha256"]
    )


def test_ready_stacked_pool_loader_requires_schema_six_late_dag_proof(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["stage_receipts"]["impute"]["stacked_late_producer_dag"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="has no late-producer DAG receipt"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_rejects_schema_four_envelope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 4
    diagnostics["schema_version"] = 4
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy envelope carries stacked-only"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_cannot_be_downgraded_to_legacy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 4
    manifest.pop("pipeline")
    manifest.pop("terminal_gates")
    diagnostics["schema_version"] = 4
    diagnostics.pop("pipeline")
    diagnostics.pop("semantic_kind")
    diagnostics.pop("terminal_gates")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy envelope carries stacked-only"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_cannot_be_stripped_into_legacy_shape(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 4
    for field in (
        "pipeline",
        "release_id",
        "sampling",
        "clone_attachment",
        "input_pins_digest",
        "late_producer_transition_authority_sha256",
        "stack_manifest",
        "terminal_gates",
        "operator_order",
        "stage_receipts",
    ):
        manifest.pop(field, None)
    diagnostics["schema_version"] = 4
    for field in ("pipeline", "semantic_kind", "release_id", "terminal_gates"):
        diagnostics.pop(field, None)
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical legacy envelope"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("authority", [None, "0" * 64])
def test_ready_stacked_pool_loader_rejects_late_authority_mismatch(
    tmp_path: Path,
    authority: str | None,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if authority is None:
        del manifest["late_producer_transition_authority_sha256"]
    else:
        manifest["late_producer_transition_authority_sha256"] = authority
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="independently carried late-producer transition authority",
    ):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("document", ["manifest", "diagnostics"])
def test_ready_stacked_pool_loader_rejects_divergent_terminal_gate_alias(
    tmp_path: Path,
    document: str,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document == "manifest":
        manifest["terminal_gates"]["passed"] = False
    else:
        diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["terminal_gates"]["passed"] = False
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal_gates do not match agreement_gate"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("document", ["manifest", "diagnostics"])
def test_ready_stacked_pool_loader_requires_both_terminal_gate_aliases(
    tmp_path: Path,
    document: str,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document == "manifest":
        del manifest["terminal_gates"]
    else:
        diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        del diagnostics["terminal_gates"]
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal_gates must be an object"):
        load_simulation_ready_us_multispine_pool(manifest_path)
