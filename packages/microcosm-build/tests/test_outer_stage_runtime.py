from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.outer_stage_runtime import (
    Stage,
    StagePipeline,
    StageRuntime,
    assert_clone_expansion,
    assert_unchanged_identity,
    frame_identity,
)
from microcosm.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)


def _pipeline() -> StagePipeline:
    return StagePipeline(
        (
            Stage("source", "Construct the pooled source frame."),
            Stage("enrich", "Add pre-clone source enrichments."),
            Stage("clone", "Clone the source into ordered support channels."),
        )
    )


def _source_frame() -> Frame:
    schema = EntitySchema(group_entities=("household",))
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3], dtype=np.int64),
            "person_household_id": np.array([10, 10, 20], dtype=np.int64),
            "source_year": np.array([2023, 2023, 2024], dtype=np.int16),
            "source_household_id": np.array([7, 7, 9], dtype=np.int64),
            "source_person_id": pd.Series(["a", "b", "c"], dtype=object),
            "source_row_id": np.array([0, 1, 0], dtype=np.int32),
            "income": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array([10, 20], dtype=np.int64),
            "tenure": np.array([1, 2], dtype=np.int8),
        }
    )
    return Frame(
        {"person": person, "household": household},
        schema,
        {
            "household": Weights(
                np.array([4.0, 6.0], dtype=np.float64),
                WeightKind.IMPORTANCE,
            )
        },
        pd.Series(["asec_2023", "asec_2023", "asec_2024"], dtype=object),
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=5.0,
                new_total=10.0,
                declared_factor=2.0,
                reason="test control",
            ),
        ),
    )


def _with_income(frame: Frame, values: list[float]) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["income"] = np.asarray(values, dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _clone_frame(frame: Frame) -> Frame:
    channels = ("asec", "puf")
    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        source = frame.table(entity)
        primary_id = frame.schema.entity_id_column(entity)
        pieces = []
        for clone_index, channel in enumerate(channels):
            piece = source.copy()
            piece[f"{entity}_source_id"] = source[primary_id].to_numpy()
            piece[f"{entity}_support_channel"] = channel
            piece[f"{entity}_support_clone_index"] = clone_index
            piece[primary_id] = source[primary_id].to_numpy() + clone_index * 100
            if entity == frame.schema.person_entity:
                piece["person_household_id"] = (
                    source["person_household_id"].to_numpy() + clone_index * 100
                )
            pieces.append(piece)
        tables[entity] = pd.concat(pieces, ignore_index=True)
    weights = {
        entity: Weights(
            np.tile(frame.weights_for(entity).values / 2, 2),
            frame.weights_for(entity).kind,
        )
        for entity in frame.weighted_entities
    }
    return Frame(
        tables,
        frame.schema,
        weights,
        pd.concat([frame.strata, frame.strata], ignore_index=True),
        mass_log=frame.mass_log,
    )


def test_pipeline_hash_binds_order_names_and_descriptions() -> None:
    pipeline = _pipeline()

    assert pipeline.names == ("source", "enrich", "clone")
    assert len(pipeline.sha256) == 64
    assert pipeline.sha256 == _pipeline().sha256
    assert pipeline.sha256 != StagePipeline(tuple(reversed(pipeline.stages))).sha256
    assert (
        pipeline.sha256
        != StagePipeline(
            (
                Stage("source", "A different source construction."),
                *pipeline.stages[1:],
            )
        ).sha256
    )


def test_runtime_requires_the_exact_predecessor_prefix(tmp_path: Path) -> None:
    run_config = {"seed": 42, "sources": ["2023.h5", "2024.h5"]}
    runtime = StageRuntime(tmp_path, _pipeline(), run_config=run_config)

    runtime.require_ready("source")
    with pytest.raises(ValueError, match="requires completed prefix"):
        runtime.require_ready("enrich")

    runtime.complete("source", _source_frame(), metadata={"seed": 42})
    runtime.require_ready("enrich")
    loaded = runtime.load_predecessor("enrich")

    assert loaded is not None
    assert loaded.stage == "source"
    assert loaded.metadata == {"seed": 42}
    assert loaded.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    assert loaded.frame.strata.equals(_source_frame().strata)
    assert loaded.frame.mass_log == _source_frame().mass_log
    assert runtime.context.completed == ("source",)
    assert runtime.require_run_config(run_config) == run_config
    assert not (tmp_path / ".stage_run_context.json.tmp").exists()

    with pytest.raises(ValueError, match="already complete"):
        runtime.complete("source", _source_frame())

    with pytest.raises(ValueError, match="locked run context"):
        StageRuntime(tmp_path, _pipeline(), run_config={"seed": 43})


def test_runtime_carries_a_checkpoint_without_rewriting_it(tmp_path: Path) -> None:
    runtime = StageRuntime(tmp_path, _pipeline())
    source = runtime.complete("source", _source_frame(), metadata={"source": "pool"})
    original_bytes = source.path.read_bytes()

    carried_path = runtime.complete_without_frame(
        "enrich", metadata={"qrf_targets": 64, "donor_rows": 100}
    )
    carried = runtime.load("enrich")

    assert carried_path == source.path
    assert carried.path == source.path
    assert carried.checkpoint_stage == "source"
    assert carried.metadata == {"qrf_targets": 64, "donor_rows": 100}
    assert carried.path.read_bytes() == original_bytes
    assert not (tmp_path / "001_enrich.frame.h5").exists()
    context_payload = json.loads(
        (tmp_path / "stage_run_context.json").read_text(encoding="utf-8")
    )
    assert (
        context_payload["stage_records"][1]["checkpoint_sha256"]
        == context_payload["stage_records"][0]["checkpoint_sha256"]
    )
    assert runtime.metadata == {
        "source": {"source": "pool"},
        "enrich": {"qrf_targets": 64, "donor_rows": 100},
    }


def test_runtime_rejects_a_nonprefix_or_changed_pipeline_context(
    tmp_path: Path,
) -> None:
    runtime = StageRuntime(tmp_path, _pipeline())
    runtime.complete("source", _source_frame())
    context_path = tmp_path / "stage_run_context.json"
    payload = json.loads(context_path.read_text())
    payload["completed"] = ["source", "clone"]
    context_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exact prefix"):
        StageRuntime(tmp_path, _pipeline())

    payload["completed"] = ["source"]
    context_path.write_text(json.dumps(payload))
    changed = StagePipeline(
        (
            Stage("source", "Construct a changed source."),
            *_pipeline().stages[1:],
        )
    )
    with pytest.raises(ValueError, match="pipeline"):
        StageRuntime(tmp_path, changed)


def test_loaded_checkpoint_revalidates_identity_metadata(tmp_path: Path) -> None:
    runtime = StageRuntime(tmp_path, _pipeline())
    runtime.complete("source", _source_frame())
    checkpoint_path = tmp_path / "000_source.frame.h5"
    checkpoint_path.write_bytes(b"not an HDF5 checkpoint")

    with pytest.raises((OSError, ValueError)):
        runtime.load("source")


def test_loaded_checkpoint_rejects_nonidentity_weight_bit_tampering(
    tmp_path: Path,
) -> None:
    runtime = StageRuntime(tmp_path, _pipeline())
    completed = runtime.complete("source", _source_frame())
    with h5py.File(completed.path, mode="r+") as h5:
        weights = h5["_populace_frame_checkpoint/weights/w00000"]
        weights[0] = np.nextafter(weights[0], np.inf)

    with pytest.raises(ValueError, match="SHA-256 changed"):
        runtime.load("source")


def test_identity_ignores_values_but_binds_rows_and_source_provenance() -> None:
    original = _source_frame()
    changed_values = _with_income(original, [100.0, 200.0, 300.0])

    assert frame_identity(original) == frame_identity(changed_values)
    assert_unchanged_identity(original, changed_values, stage="value_only")

    tables = {entity: original.table(entity).copy() for entity in original.entities}
    order = [1, 0, 2]
    tables["person"] = tables["person"].iloc[order].reset_index(drop=True)
    reordered = Frame(
        tables,
        original.schema,
        {entity: original.weights_for(entity) for entity in original.weighted_entities},
        original.strata.iloc[order].reset_index(drop=True),
        mass_log=original.mass_log,
    )
    with pytest.raises(AssertionError, match="value_only"):
        assert_unchanged_identity(original, reordered, stage="value_only")

    tables = {entity: original.table(entity).copy() for entity in original.entities}
    tables["person"].loc[0, "source_year"] = 2022
    changed_source = Frame(
        tables,
        original.schema,
        {entity: original.weights_for(entity) for entity in original.weighted_entities},
        original.strata,
        mass_log=original.mass_log,
    )
    assert frame_identity(original) != frame_identity(changed_source)


def test_clone_expansion_asserts_doubling_order_channel_and_source_ids() -> None:
    source = _source_frame()
    clone = _clone_frame(source)

    assert_clone_expansion(source, clone, channels=("asec", "puf"))

    tables = {entity: clone.table(entity).copy() for entity in clone.entities}
    tables["person"].loc[0, "person_support_channel"] = "puf"
    wrong_channel = Frame(
        tables,
        clone.schema,
        {entity: clone.weights_for(entity) for entity in clone.weighted_entities},
        clone.strata,
        mass_log=clone.mass_log,
    )
    with pytest.raises(AssertionError, match="channel order"):
        assert_clone_expansion(source, wrong_channel, channels=("asec", "puf"))

    tables = {entity: clone.table(entity).copy() for entity in clone.entities}
    tables["household"].loc[2, "household_source_id"] = 20
    wrong_source_id = Frame(
        tables,
        clone.schema,
        {entity: clone.weights_for(entity) for entity in clone.weighted_entities},
        clone.strata,
        mass_log=clone.mass_log,
    )
    with pytest.raises(AssertionError, match="source-ID order"):
        assert_clone_expansion(
            source,
            wrong_source_id,
            channels=("asec", "puf"),
        )
