from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.asec_checkpoint as checkpoint_module
from microcosm.build.frame_checkpoint import write_frame_checkpoint
from microcosm.build.outer_stage_runtime import (
    OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
    frame_identity,
)
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime import (
    ASEC_RAW_STAGE_ARTIFACT_KIND,
    ASEC_RAW_STAGE_OPERATOR_STATUS,
    ASEC_RAW_STAGE_SCHEMA_VERSION,
    ASEC_RAW_STAGE_STAGE,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
    assert_operator_free_source_frame,
    load_asec_pre_clone_checkpoint,
    load_asec_raw_stage_checkpoint,
    load_take_up_contract,
)
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights


def _v4_frame() -> Frame:
    frame = _raw_us_frame()
    for column in checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        frame.table("person")[column] = np.asarray([1, 2], dtype=np.int64)
    return frame


def _v4_binding(frame: Frame) -> dict:
    metadata = _raw_binding(frame)
    metadata["schema_version"] = 4
    metadata["source_receipt"]["sources"].append(
        dict(metadata["source_receipt"]["sources"][0], year=2023)
    )
    pins = checkpoint_module.ASEC_EDUCATION_ASSISTANCE_ARCHIVES
    for column in checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        metadata["raw_source_mappings"][column] = {
            "column": column,
            "entity": "person",
            "operation": "exact_source_join",
            "join_keys": ["source_year", "PERIDNUM"],
            "source_pins": [
                {
                    "income_year": year,
                    "locator": pins[year].zip_url,
                    "member": pins[year].member,
                    "sha256": pins[year].zip_sha256,
                    "member_sha256": pins[year].member_sha256,
                }
                for year in (2022, 2023)
            ],
            # Audit describes the full official member, not this two-row subset.
            "audit": {
                str(year): {
                    "rows": pins[year].rows,
                    "yes_rows": 1,
                    "no_rows": pins[year].rows - 1,
                    "weighted_yes_share": 0.25,
                }
                for year in (2022, 2023)
            },
        }
    return metadata


def test_v4_roundtrip_requires_explicit_codec(tmp_path: Path) -> None:
    frame = _v4_frame()
    path = tmp_path / "v4.h5"
    _write_checkpoint(path, frame, metadata=_v4_binding(frame))
    loaded, binding = checkpoint_module.load_asec_raw_stage_checkpoint_v4(path)
    assert binding["schema_version"] == 4
    assert loaded.table("person")["NOW_MCAID"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="unsupported raw-stage schema version"):
        load_asec_raw_stage_checkpoint(path)
    path3 = tmp_path / "v3.h5"
    legacy = _raw_us_frame()
    _write_checkpoint(path3, legacy, metadata=_raw_binding(legacy))
    with pytest.raises(ValueError, match="unsupported raw-stage schema version"):
        checkpoint_module.load_asec_raw_stage_checkpoint_v4(path3)


def test_restored_raw_observations_carry_to_input_leaves_without_engine() -> None:
    from microcosm.build.us_runtime.cps_carried import derive_us_cps_carried_inputs

    frame = _raw_us_frame()
    person = frame.table("person")
    person["NOW_GRP"] = [1, 2]
    person["NOW_MRK"] = [2, 1]
    person["OI_VAL"] = [0.0, 0.0]
    person["OI_OFF"] = [0, 0]
    sidecar = person[["source_year", "PERIDNUM"]].copy()
    sidecar["PH_SEQ"] = [101, 102]
    sidecar["P_SEQ"] = [1, 1]
    sidecar["A_LINENO"] = [1, 1]
    for column in checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        sidecar[column] = [1, 2]
    result = derive_us_cps_carried_inputs(frame, reported_coverage_source=sidecar)
    assert "NOW_MCAID" not in frame.table("person")
    assert result.table("person")[
        "has_medicaid_health_coverage_at_interview"
    ].tolist() == [True, False]
    assert result.table("person")["has_esi"].tolist() == [True, False]
    assert result.table("person")[
        "has_marketplace_health_coverage_at_interview"
    ].tolist() == [False, True]
    assert result.weights_for("household").kind is WeightKind.DESIGN
    pd.testing.assert_frame_equal(
        result.table("person"), derive_us_cps_carried_inputs(result).table("person")
    )


@pytest.mark.parametrize(
    "with_mass_history,forward_metadata",
    [(False, False), (True, False), (True, True)],
    ids=["empty_log", "declared_mass_change", "metadata_forwarding"],
)
def test_v4_restoration_producer_authenticates_source_and_writes_new_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_mass_history: bool,
    forward_metadata: bool,
) -> None:
    import hashlib
    from dataclasses import replace

    from microcosm.build.us_runtime import asec_raw_stage_v4 as restoration
    from microcosm.build.us_runtime import reported_coverage_source
    from microcosm.build.us_runtime.asec_raw_stage_v4 import restore_asec_raw_stage_v4
    from microcosm.frame import MassChange

    legacy = _raw_us_frame()
    if with_mass_history:
        original = legacy
        weights = original.weights_for("household")
        legacy = original.with_weights(
            "household",
            Weights(weights.values * 2.0, weights.kind),
            mass=MassChange(
                factor=2.0, reason="Invented raw-source design-weight rescaling"
            ),
        )
        assert original.mass_log == ()
        assert len(legacy.mass_log) == 1
        record = legacy.mass_log[0]
        assert record.entity == "household"
        assert record.old_total == float(weights.values.sum())
        assert record.new_total == 2.0 * record.old_total
        assert record.declared_factor == 2.0
        assert record.reason == "Invented raw-source design-weight rescaling"
    expected_mass_log = legacy.mass_log
    assert bool(expected_mass_log) is with_mass_history
    binding = _raw_binding(legacy)
    binding["source_receipt"]["sources"].append(
        dict(binding["source_receipt"]["sources"][0], year=2023)
    )
    input_path = tmp_path / "v3.h5"
    _write_checkpoint(input_path, legacy, metadata=binding)
    loaded_input, _ = checkpoint_module.load_asec_raw_stage_checkpoint(input_path)
    assert loaded_input.mass_log == expected_mass_log
    np.testing.assert_array_equal(
        loaded_input.weights_for("household").values,
        legacy.weights_for("household").values,
    )
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    pins = {}
    paths = {}
    for year, key in (
        (2022, "0000000000000000000001"),
        (2023, "0000000000000000000002"),
    ):
        path = tmp_path / f"person-{year}.csv"
        row = {
            "PERIDNUM": key,
            "PH_SEQ": 1,
            "P_SEQ": 1,
            "A_LINENO": 1,
            "A_FNLWGT": 100.0,
        }
        row.update(
            {
                column: 1 if year == 2022 else 2
                for column in checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS
            }
        )
        pd.DataFrame([row]).to_csv(path, index=False)
        pins[year] = replace(
            checkpoint_module.ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year],
            rows=1,
            member=path.name,
            member_size_bytes=path.stat().st_size,
            member_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        paths[year] = path
    monkeypatch.setattr(checkpoint_module, "ASEC_EDUCATION_ASSISTANCE_ARCHIVES", pins)
    monkeypatch.setattr(
        reported_coverage_source, "ASEC_EDUCATION_ASSISTANCE_ARCHIVES", pins
    )
    forwarded = []
    if forward_metadata:
        # The checkpoint codec does not persist Frame.metadata. Supply a
        # descriptive value only after the real v3 loader has validated its
        # input, then observe the actual converter at the real writer boundary.
        supplied_metadata = {"forwarding_probe": {"purpose": "invented observation"}}
        real_load = checkpoint_module.load_asec_raw_stage_checkpoint
        real_write = restoration.write_frame_checkpoint

        def load_with_descriptive_metadata(path):
            loaded, binding = real_load(path)
            assert not loaded.metadata
            return (
                Frame(
                    {entity: loaded.table(entity) for entity in loaded.entities},
                    loaded.schema,
                    {
                        entity: loaded.weights_for(entity)
                        for entity in loaded.weighted_entities
                    },
                    loaded.strata,
                    mass_log=loaded.mass_log,
                    metadata=supplied_metadata,
                ),
                binding,
            )

        def observe_real_write(path, frame, **kwargs):
            assert set(frame.metadata) == {"forwarding_probe"}
            assert (
                dict(frame.metadata["forwarding_probe"])
                == supplied_metadata["forwarding_probe"]
            )
            assert frame.mass_log == expected_mass_log
            forwarded.append(True)
            return real_write(path, frame, **kwargs)

        monkeypatch.setattr(
            checkpoint_module,
            "load_asec_raw_stage_checkpoint",
            load_with_descriptive_metadata,
        )
        monkeypatch.setattr(restoration, "write_frame_checkpoint", observe_real_write)
    output_dir = tmp_path / "restored"
    receipt = restore_asec_raw_stage_v4(
        input_path,
        expected_sha256=input_sha,
        coverage_paths=paths,
        output_dir=output_dir,
    )
    assert forwarded == ([True] if forward_metadata else [])
    assert receipt["input_sha256"] == input_sha
    output = output_dir / "asec_raw_stage.checkpoint.h5"
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_sha
    restored, metadata = checkpoint_module.load_asec_raw_stage_checkpoint_v4(output)
    assert restored.mass_log == expected_mass_log
    # The codec does not reconstruct the separately supplied Frame metadata.
    assert not restored.metadata
    assert restored.weights_for("household").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        restored.weights_for("household").values,
        loaded_input.weights_for("household").values,
    )
    assert restored.table("person")["NOW_MCAID"].tolist() == [1, 2]
    assert metadata["source_receipt"] == binding["source_receipt"]
    assert json.loads((output_dir / "restoration.receipt.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        restore_asec_raw_stage_v4(
            input_path,
            expected_sha256=input_sha,
            coverage_paths=paths,
            output_dir=output_dir,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        restore_asec_raw_stage_v4(
            input_path,
            expected_sha256="f" * 64,
            coverage_paths=paths,
            output_dir=tmp_path / "bad",
        )
    with pytest.raises(ValueError, match="local coverage paths"):
        restore_asec_raw_stage_v4(
            input_path,
            expected_sha256=input_sha,
            coverage_paths={2022: paths[2022]},
            output_dir=tmp_path / "missing",
        )
    paths[2022].write_text("tampered")
    with pytest.raises(ValueError, match="byte length mismatch"):
        restore_asec_raw_stage_v4(
            input_path,
            expected_sha256=input_sha,
            coverage_paths=paths,
            output_dir=tmp_path / "tampered",
        )
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "missing").exists()
    assert not (tmp_path / "tampered").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "pin_sha",
        "pin_member",
        "pin_year_missing",
        "pin_year_duplicate",
        "audit_year",
        "audit_rows",
        "audit_counts",
        "audit_nan",
        "audit_bool",
        "receipt_year",
        "missing_mapping",
        "float_version",
    ),
)
def test_v4_refuses_inconsistent_attestation(tmp_path: Path, mutation: str) -> None:
    frame = _v4_frame()
    metadata = _v4_binding(frame)
    mapping = metadata["raw_source_mappings"]["NOW_MCAID"]
    if mutation == "pin_sha":
        mapping["source_pins"][0]["sha256"] = "f" * 64
    elif mutation == "pin_member":
        mapping["source_pins"][0]["member"] = "other.csv"
    elif mutation == "pin_year_missing":
        mapping["source_pins"].pop()
    elif mutation == "pin_year_duplicate":
        mapping["source_pins"].append(dict(mapping["source_pins"][0]))
    elif mutation == "audit_year":
        del mapping["audit"]["2023"]
    elif mutation == "audit_rows":
        mapping["audit"]["2022"]["rows"] = 2
    elif mutation == "audit_counts":
        mapping["audit"]["2022"]["no_rows"] = 0
    elif mutation == "audit_nan":
        mapping["audit"]["2022"]["weighted_yes_share"] = float("nan")
    elif mutation == "audit_bool":
        mapping["audit"]["2022"]["yes_rows"] = True
    elif mutation == "receipt_year":
        metadata["source_receipt"]["sources"].pop()
    elif mutation == "missing_mapping":
        del metadata["raw_source_mappings"]["NOW_MCAID"]
    elif mutation == "float_version":
        metadata["schema_version"] = 4.0
    path = tmp_path / "invalid.h5"
    # Exercise the same metadata validators directly for NaN because canonical
    # checkpoint serialization correctly refuses it before the loader does.
    if mutation == "audit_nan":
        with pytest.raises(ValueError, match="audit counts/share"):
            checkpoint_module._validate_coverage_v4_binding(frame, metadata, path=path)
    else:
        _write_checkpoint(path, frame, metadata=metadata)
        with pytest.raises(ValueError):
            checkpoint_module.load_asec_raw_stage_checkpoint_v4(path)


@pytest.mark.parametrize("value", (0, 3, 1.5, float("inf"), float("nan"), True))
def test_v4_refuses_bad_measured_coverage(tmp_path: Path, value: object) -> None:
    frame = _v4_frame()
    frame.table("person")["NOW_MCAID"] = [value, value]
    path = tmp_path / "bad-code.h5"
    _write_checkpoint(path, frame, metadata=_v4_binding(frame))
    with pytest.raises(ValueError, match="NOW_MCAID must be complete integer recodes"):
        checkpoint_module.load_asec_raw_stage_checkpoint_v4(path)


@pytest.mark.parametrize(
    "mutation", ("missing_column", "peridnum", "duplicate", "weights")
)
def test_v4_refuses_wrong_source_boundary(tmp_path: Path, mutation: str) -> None:
    frame = _v4_frame()
    person = frame.table("person")
    if mutation == "missing_column":
        person.drop(columns="NOW_IHSFLG", inplace=True)
    elif mutation == "peridnum":
        person["PERIDNUM"] = ["1", "2"]
    elif mutation == "duplicate":
        person["source_year"] = [2022, 2022]
        person["PERIDNUM"] = ["0" * 22, "0" * 22]
    elif mutation == "weights":
        frame = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": Weights(np.asarray([2.0, 3.0]), WeightKind.CALIBRATED)},
            frame.strata,
        )
    metadata = _v4_binding(frame)
    if mutation == "duplicate":
        metadata["source_receipt"]["sources"].pop()
    path = tmp_path / "boundary.h5"
    _write_checkpoint(path, frame, metadata=metadata)
    with pytest.raises(ValueError):
        checkpoint_module.load_asec_raw_stage_checkpoint_v4(path)


def test_v4_duplicate_guard_compares_normalized_identity() -> None:
    frame = _v4_frame()
    person = frame.table("person")
    person["source_year"] = pd.Series([2022, "2022"], dtype=object)
    person["PERIDNUM"] = pd.Series(["0" * 22, b"0" * 22], dtype=object)
    metadata = _v4_binding(frame)
    metadata["source_receipt"]["sources"].pop()
    for column in checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        metadata["raw_source_mappings"][column]["source_pins"].pop()
        del metadata["raw_source_mappings"][column]["audit"]["2023"]
    with pytest.raises(ValueError, match="repeats a source-year/PERIDNUM"):
        checkpoint_module._validate_coverage_v4_binding(
            frame, metadata, path=Path("fixture.h5")
        )


_OUTER_STAGE_ARTIFACT_KIND = "populace_outer_stage_frame"


def _us_frame(
    *,
    id_offset: int = 0,
    household_weights: tuple[float, float] = (2.0, 3.0),
    include_age: bool = True,
    person_weights: bool = False,
) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64) + id_offset
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids + 10,
            "person_spm_unit_id": ids + 20,
            "person_family_id": ids + 30,
            "person_marital_unit_id": ids + 40,
        }
    )
    if include_age:
        person["age"] = np.asarray([30, 50], dtype=np.int16)
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids + 10}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 20}),
        "family": pd.DataFrame({"family_id": ids + 30}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 40}),
    }
    weights = {
        "household": Weights(
            np.asarray(household_weights, dtype=np.float64),
            WeightKind.DESIGN,
        )
    }
    if person_weights:
        weights["person"] = Weights(
            np.asarray([2.0, 3.0], dtype=np.float64),
            WeightKind.DESIGN,
        )
    return Frame(
        tables,
        US_SCHEMA,
        weights,
        pd.Series(["asec_2023", "asec_2024"], dtype=object),
    )


def _non_us_frame() -> Frame:
    schema = EntitySchema(group_entities=("household",))
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1], dtype=np.int64),
                    "person_household_id": np.asarray([1], dtype=np.int64),
                }
            ),
            "household": pd.DataFrame(
                {"household_id": np.asarray([1], dtype=np.int64)}
            ),
        },
        schema,
        {
            "household": Weights(
                np.asarray([1.0], dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["fixture"], dtype=object),
    )


def _binding(frame: Frame) -> dict[str, object]:
    return {
        "artifact_kind": _OUTER_STAGE_ARTIFACT_KIND,
        "identity": frame_identity(frame).to_payload(),
        "pipeline_sha256": "a" * 64,
        "schema_version": OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
        "stage": "pre_clone_enrichment",
        "stage_index": 1,
    }


def _raw_binding(frame: Frame) -> dict[str, object]:
    pin = {
        "income_year": 2022,
        "locator": "https://example.test/asec.zip",
        "member": "pppub.csv",
        "member_sha256": "b" * 64,
        "sha256": "a" * 64,
    }
    return {
        "artifact_kind": ASEC_RAW_STAGE_ARTIFACT_KIND,
        "identity": frame_identity(frame).to_payload(),
        "operator_status": ASEC_RAW_STAGE_OPERATOR_STATUS,
        "pipeline_sha256": "c" * 64,
        "raw_source_mappings": {
            column: {
                "audit": {"rows": 2},
                "column": column,
                "entity": "person",
                "join_keys": ["source_year", "PERIDNUM"],
                "operation": "exact_source_join",
                "source_pins": [pin],
            }
            for column in ("ED_VAL", "LKWEEKS", "PAW_TYP")
        },
        "schema_version": ASEC_RAW_STAGE_SCHEMA_VERSION,
        "source_construction_identity": frame_identity(frame).to_payload(),
        "source_receipt": {
            "kind": "pooled_asec",
            "sources": [
                {
                    "max_households": None,
                    "path": "/raw/asec_2022.h5",
                    "sha256": "d" * 64,
                    "share": 1.0,
                    "year": 2022,
                }
            ],
            "target_year": 2022,
        },
        "stage": ASEC_RAW_STAGE_STAGE,
    }


def _raw_us_frame(*, id_offset: int = 0) -> Frame:
    source = _us_frame(id_offset=id_offset, include_age=False)
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    tables["person"]["source_year"] = [2022, 2023]
    tables["person"]["PERIDNUM"] = [
        "0000000000000000000001",
        "0000000000000000000002",
    ]
    tables["person"]["ED_VAL"] = [0.0, 500.0]
    tables["person"]["LKWEEKS"] = [-1, 12]
    tables["person"]["PAW_TYP"] = np.asarray([0, 1], dtype=np.int64)
    return Frame(
        tables,
        source.schema,
        {entity: source.weights_for(entity) for entity in source.weighted_entities},
        source.strata,
    )


def _write_checkpoint(
    path: Path,
    frame: Frame,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    write_frame_checkpoint(path, frame, metadata=metadata or _binding(frame))


def test_loads_bound_input_complete_asec_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "001_pre_clone_enrichment.frame.h5"
    source = _us_frame()
    expected_metadata = _binding(source)
    _write_checkpoint(path, source, metadata=expected_metadata)

    frame, metadata = load_asec_pre_clone_checkpoint(path)

    assert frame_identity(frame) == frame_identity(source)
    assert frame.schema == US_SCHEMA
    assert frame.weighted_entities == ("household",)
    assert metadata == expected_metadata
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_kind", "other", "not an outer-stage Frame artifact"),
        ("schema_version", 999, "unsupported outer-stage schema version"),
        ("stage", "source_construction", "must be bound to stage"),
        ("stage_index", 0, "must be bound to stage_index 1"),
        ("pipeline_sha256", "not-a-digest", "lowercase SHA-256 digest"),
    ),
)
def test_rejects_wrong_outer_stage_binding(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / f"wrong-{field}.frame.h5"
    frame = _us_frame()
    metadata = _binding(frame)
    metadata[field] = value
    _write_checkpoint(path, frame, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        load_asec_pre_clone_checkpoint(path)


def test_loads_operator_untouched_raw_stage_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "asec_raw_stage.checkpoint.h5"
    source = _raw_us_frame()
    metadata = _raw_binding(source)
    _write_checkpoint(path, source, metadata=metadata)

    frame, loaded_metadata = load_asec_raw_stage_checkpoint(path)

    assert frame_identity(frame) == frame_identity(source)
    assert loaded_metadata == metadata
    assert loaded_metadata["artifact_kind"] == ASEC_RAW_STAGE_ARTIFACT_KIND
    assert loaded_metadata["stage"] == ASEC_RAW_STAGE_STAGE
    assert loaded_metadata["operator_status"] == ASEC_RAW_STAGE_OPERATOR_STATUS
    # The load is a declared string-storage canonicalization boundary: no
    # entity may leak a non-canonical pandas string dtype downstream,
    # whatever storage the checkpoint was written under.
    for entity in frame.entities:
        for column, dtype in frame.table(entity).dtypes.items():
            if isinstance(dtype, pd.StringDtype):
                assert dtype == CANONICAL_STRING_DTYPE, (entity, column)


@pytest.mark.parametrize(
    "column",
    ("ED_VAL", "LKWEEKS", "PAW_TYP", "PERIDNUM", "source_year"),
)
def test_raw_loader_rejects_missing_input_complete_source_column(
    tmp_path: Path,
    column: str,
) -> None:
    source = _raw_us_frame()
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    tables["person"] = tables["person"].drop(columns=[column])
    incomplete = Frame(
        tables,
        source.schema,
        {entity: source.weights_for(entity) for entity in source.weighted_entities},
        source.strata,
    )
    path = tmp_path / f"raw-missing-{column}.checkpoint.h5"
    _write_checkpoint(path, incomplete, metadata=_raw_binding(incomplete))

    with pytest.raises(ValueError, match=rf"input-complete.*{column}"):
        load_asec_raw_stage_checkpoint(path)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    (
        ("source_year", [2022, np.nan], "source_year must be complete"),
        ("PERIDNUM", ["0000000000000000000001", ""], "PERIDNUM must be complete"),
        ("ED_VAL", [0.0, np.nan], "ED_VAL must be complete"),
        ("LKWEEKS", [-1, 53], "LKWEEKS must be complete"),
        ("PAW_TYP", [0, 4], "PAW_TYP must be complete integers"),
    ),
)
def test_raw_loader_rejects_invalid_input_complete_source_values(
    tmp_path: Path,
    column: str,
    values: list[object],
    message: str,
) -> None:
    source = _raw_us_frame()
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    tables["person"][column] = values
    invalid = Frame(
        tables,
        source.schema,
        {entity: source.weights_for(entity) for entity in source.weighted_entities},
        source.strata,
    )
    path = tmp_path / f"raw-invalid-{column}.checkpoint.h5"
    _write_checkpoint(path, invalid, metadata=_raw_binding(invalid))

    with pytest.raises(ValueError, match=message):
        load_asec_raw_stage_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_kind", _OUTER_STAGE_ARTIFACT_KIND, "not a dedicated raw-stage"),
        ("schema_version", 999, "unsupported raw-stage schema version"),
        ("stage", "pre_clone_enrichment", "must be bound to stage"),
        ("operator_status", "operator_enriched", "must declare operator_status"),
        ("pipeline_sha256", "not-a-digest", "lowercase SHA-256 digest"),
    ),
)
def test_raw_loader_rejects_wrong_artifact_binding(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / f"raw-wrong-{field}.checkpoint.h5"
    frame = _raw_us_frame()
    metadata = _raw_binding(frame)
    metadata[field] = value
    _write_checkpoint(path, frame, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        load_asec_raw_stage_checkpoint(path)


def test_raw_loader_rejects_legacy_enriched_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "001_pre_clone_enrichment.frame.h5"
    frame = _us_frame()
    _write_checkpoint(path, frame, metadata=_binding(frame))

    with pytest.raises(ValueError, match="incomplete raw-stage artifact binding"):
        load_asec_raw_stage_checkpoint(path)


def test_raw_loader_rejects_identity_not_bound_to_frame(tmp_path: Path) -> None:
    path = tmp_path / "raw-wrong-identity.checkpoint.h5"
    frame = _raw_us_frame()
    metadata = _raw_binding(frame)
    metadata["identity"] = frame_identity(_raw_us_frame(id_offset=100)).to_payload()
    _write_checkpoint(path, frame, metadata=metadata)

    with pytest.raises(ValueError, match="Frame identity changed"):
        load_asec_raw_stage_checkpoint(path)


def test_raw_loader_rejects_wrong_source_construction_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw-wrong-source-identity.checkpoint.h5"
    frame = _raw_us_frame()
    metadata = _raw_binding(frame)
    metadata["source_construction_identity"] = frame_identity(
        _raw_us_frame(id_offset=100)
    ).to_payload()
    _write_checkpoint(path, frame, metadata=metadata)

    with pytest.raises(ValueError, match="source-construction structural identity"):
        load_asec_raw_stage_checkpoint(path)


_OPERATOR_OUTPUT_CASES = tuple(
    (family, entity, column)
    for family, by_entity in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.items()
    for entity, columns in by_entity.items()
    for column in sorted(columns)
)


def test_operator_boundary_enumerates_full_take_up_contract() -> None:
    expected: dict[str, set[str]] = {}
    for program in load_take_up_contract().programs:
        expected.setdefault(program.entity, set()).add(program.variable)

    assert PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["take_up"] == {
        entity: frozenset(columns) for entity, columns in expected.items()
    }


def test_operator_boundary_accepts_only_receipted_acs_native_exception() -> None:
    source = _us_frame(include_age=False)
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    tables["person"]["AGEP"] = [30, 50]
    tables["person"]["age"] = [30, 50]
    acs = Frame(
        tables,
        source.schema,
        {entity: source.weights_for(entity) for entity in source.weighted_entities},
        source.strata,
    )
    receipt = {
        "age": {
            "entity": "person",
            "source_columns": ["AGEP"],
            "transformation": "identity",
            "provenance": "acs_2024_1yr_native",
            "observed_rows": 2,
            "missing_rows": 0,
        }
    }

    assert_operator_free_source_frame(
        acs,
        label="ACS fixture",
        native_inputs=receipt,
    )
    with pytest.raises(ValueError, match="canonical operator output"):
        assert_operator_free_source_frame(acs, label="unreceipted ACS fixture")

    malformed = {"age": {**receipt["age"], "provenance": "fixture_allowlist"}}
    with pytest.raises(ValueError, match="provenance"):
        assert_operator_free_source_frame(
            acs,
            label="malformed ACS fixture",
            native_inputs=malformed,
        )


def test_operator_boundary_rejects_forged_native_receipt_for_operator_output() -> None:
    source = _us_frame(include_age=False)
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    tables["person"]["AGEP"] = [30, 50]
    tables["person"]["takes_up_wic_if_eligible"] = [True, False]
    forged = Frame(
        tables,
        source.schema,
        {entity: source.weights_for(entity) for entity in source.weighted_entities},
        source.strata,
    )
    forged_receipt = {
        "takes_up_wic_if_eligible": {
            "entity": "person",
            "source_columns": ["AGEP"],
            "transformation": "identity",
            "provenance": "acs_2024_1yr_native",
            "observed_rows": 2,
            "missing_rows": 0,
        }
    }

    with pytest.raises(ValueError, match="not a declared ACS native mapping"):
        assert_operator_free_source_frame(
            forged,
            label="forged ACS fixture",
            native_inputs=forged_receipt,
        )


@pytest.mark.parametrize(
    ("family", "entity", "column"),
    _OPERATOR_OUTPUT_CASES,
)
def test_raw_loader_rejects_every_registered_operator_output(
    tmp_path: Path,
    family: str,
    entity: str,
    column: str,
) -> None:
    source = _raw_us_frame()
    tables = {
        table_entity: source.table(table_entity).copy()
        for table_entity in source.entities
    }
    tables[entity][column] = 0
    contaminated = Frame(
        tables,
        source.schema,
        {
            weighted_entity: source.weights_for(weighted_entity)
            for weighted_entity in source.weighted_entities
        },
        source.strata,
    )
    path = tmp_path / f"{family}-{entity}-{column}.checkpoint.h5"
    _write_checkpoint(path, contaminated, metadata=_raw_binding(contaminated))

    with pytest.raises(ValueError, match=rf"{family}:{entity}"):
        load_asec_raw_stage_checkpoint(path)


def test_rejects_incomplete_outer_stage_binding(tmp_path: Path) -> None:
    path = tmp_path / "missing-stage.frame.h5"
    frame = _us_frame()
    metadata = _binding(frame)
    del metadata["stage"]
    _write_checkpoint(path, frame, metadata=metadata)

    with pytest.raises(ValueError, match="incomplete outer-stage artifact binding"):
        load_asec_pre_clone_checkpoint(path)


def test_rejects_identity_not_bound_to_loaded_frame(tmp_path: Path) -> None:
    path = tmp_path / "wrong-identity.frame.h5"
    frame = _us_frame()
    metadata = _binding(frame)
    metadata["identity"] = frame_identity(_us_frame(id_offset=100)).to_payload()
    _write_checkpoint(path, frame, metadata=metadata)

    with pytest.raises(ValueError, match="Frame identity changed"):
        load_asec_pre_clone_checkpoint(path)


@pytest.mark.parametrize(
    ("frame", "message"),
    (
        (_non_us_frame(), "must use the US entity schema"),
        (_us_frame(person_weights=True), "must carry household weights only"),
        (
            _us_frame(household_weights=(2.0, 0.0)),
            "household weights must be strictly positive and finite",
        ),
    ),
)
def test_rejects_invalid_asec_frame_boundary(
    tmp_path: Path,
    frame: Frame,
    message: str,
) -> None:
    path = tmp_path / f"invalid-{len(list(tmp_path.iterdir()))}.frame.h5"
    _write_checkpoint(path, frame)

    with pytest.raises(ValueError, match=message):
        load_asec_pre_clone_checkpoint(path)
