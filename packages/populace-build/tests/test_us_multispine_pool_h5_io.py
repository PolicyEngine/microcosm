from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.h5_io as h5_io
from populace.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from populace.build.serialization_dtypes import (
    CANONICAL_STRING_DTYPE,
    canonicalize_frame_string_dtypes,
    canonicalize_table_string_dtypes,
)
from populace.build.us_runtime.h5_io import (
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    AuthenticatedPoolH5MismatchError,
    load_simulation_ready_us_multispine_pool,
    write_nullable_us_h5,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


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
        with pd.HDFStore(output, mode="r") as store:
            for entity in US_SCHEMA.entities:
                stored = store[entity]
                string_columns = _semantic_string_columns(stored)
                assert string_columns
                assert all(
                    stored[column].dtype == CANONICAL_STRING_DTYPE
                    for column in string_columns
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
    assert stored.dtype == CANONICAL_STRING_DTYPE
    assert stored.isna().all()


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
    write_nullable_us_h5(
        _pool_frame_with_object_strings_on_every_entity(),
        pool_path,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        publication_run_id=run_id,
    )
    diagnostics = {
        "artifact_kind": (US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND),
        "schema_version": US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
        "simulation_ready": True,
        "publication_run_id": run_id,
        "agreement_gate": agreement_gate,
    }
    if stacked:
        diagnostics.update(
            {
                "pipeline": "us-stacked-pool",
                "terminal_gates": agreement_gate,
            }
        )
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest = {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
        "status": "simulation_ready",
        "simulation_ready": True,
        "publication_run_id": run_id,
        "period": 2024,
        "stage_checkpoints": {
            "agreement": {
                "source": "always_fresh",
                "cached": False,
                "terminal_verdict_persisted": False,
            }
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
        manifest.update(
            {
                "pipeline": "us-stacked-pool",
                "terminal_gates": agreement_gate,
            }
        )
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest_path


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

    _, manifest, _ = load_simulation_ready_us_multispine_pool(manifest_path)

    assert manifest["terminal_gates"] == manifest["agreement_gate"]


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
