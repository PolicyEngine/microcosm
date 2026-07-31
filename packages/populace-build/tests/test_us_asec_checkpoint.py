from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.frame_checkpoint import write_frame_checkpoint
from populace.build.outer_stage_runtime import (
    OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
    frame_identity,
)
from populace.build.us_runtime import load_asec_pre_clone_checkpoint
from populace.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights

_OUTER_STAGE_ARTIFACT_KIND = "populace_outer_stage_frame"


def _us_frame(
    *,
    id_offset: int = 0,
    household_weights: tuple[float, float] = (2.0, 3.0),
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
            "age": np.asarray([30, 50], dtype=np.int16),
        }
    )
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
