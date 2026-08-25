from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
