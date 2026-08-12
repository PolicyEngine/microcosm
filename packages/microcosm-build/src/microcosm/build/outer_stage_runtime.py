"""Checkpoint and invariant runtime for ordered fresh-process build stages.

The runtime deliberately knows nothing about a country rules-engine H5.  Every
stage boundary is a lossless :class:`microcosm.frame.Frame` checkpoint written
through :mod:`microcosm.build.frame_checkpoint`, and a small atomic JSON context
records which exact prefix of an immutable pipeline has completed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.frame import Frame

__all__ = [
    "OUTER_STAGE_CONTEXT_FILENAME",
    "FrameIdentity",
    "LoadedStageCheckpoint",
    "Stage",
    "StagePipeline",
    "StageRunContext",
    "StageRuntime",
    "assert_clone_expansion",
    "assert_unchanged_identity",
    "frame_identity",
]

OUTER_STAGE_CONTEXT_FILENAME = "stage_run_context.json"
OUTER_STAGE_CONTEXT_SCHEMA_VERSION = 2

_ARTIFACT_KIND = "populace_outer_stage_frame"
_CONTEXT_ARTIFACT_KIND = "populace_outer_stage_run_context"
_STAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_IDENTITY_HASH_CHUNK_ROWS = 100_000
_POOLED_SOURCE_PROVENANCE_COLUMNS = (
    "source_year",
    "source_household_id",
    "source_person_id",
    "source_row_id",
)


@dataclass(frozen=True)
class Stage:
    """One stable, descriptive outer-pipeline stage."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _STAGE_NAME.fullmatch(self.name):
            raise ValueError(
                f"stage name must match [a-z0-9][a-z0-9_-]*, got {self.name!r}."
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("stage description must be a non-empty string.")


@dataclass(frozen=True)
class StagePipeline:
    """An immutable ordered tuple whose digest binds names and descriptions."""

    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        object.__setattr__(self, "stages", stages)
        if not stages:
            raise ValueError("stage pipeline must contain at least one stage.")
        if any(not isinstance(stage, Stage) for stage in stages):
            raise TypeError("stage pipeline entries must be Stage instances.")
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"stage names must be unique; duplicated: {duplicates}.")

    @property
    def names(self) -> tuple[str, ...]:
        """Return stage names in their load-bearing execution order."""

        return tuple(stage.name for stage in self.stages)

    @property
    def sha256(self) -> str:
        """Return the canonical digest of the ordered descriptive pipeline."""

        return _mapping_sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible pipeline definition."""

        return {
            "stages": [
                {"description": stage.description, "name": stage.name}
                for stage in self.stages
            ]
        }

    def index(self, stage_name: str) -> int:
        """Return the index of ``stage_name`` or raise a descriptive error."""

        try:
            return self.names.index(stage_name)
        except ValueError as error:
            raise ValueError(
                f"unknown stage {stage_name!r}; pipeline stages are {self.names}."
            ) from error


@dataclass(frozen=True)
class FrameIdentity:
    """Stable ordered structural identity for a :class:`Frame`."""

    sha256: str
    row_counts: tuple[tuple[str, int], ...]
    columns: tuple[tuple[str, tuple[str, ...]], ...]

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible identity record."""

        return {
            "columns": [
                {"columns": list(columns), "table": table}
                for table, columns in self.columns
            ],
            "row_counts": [
                {"rows": count, "table": table} for table, count in self.row_counts
            ],
            "sha256": self.sha256,
        }

    @classmethod
    def from_payload(cls, payload: object, *, label: str) -> FrameIdentity:
        """Validate and restore a serialized identity record."""

        value = _require_mapping(payload, label)
        sha256 = value.get("sha256")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest.")
        raw_counts = _require_list(value.get("row_counts"), f"{label}.row_counts")
        row_counts: list[tuple[str, int]] = []
        for index, raw_entry in enumerate(raw_counts):
            entry = _require_mapping(raw_entry, f"{label}.row_counts[{index}]")
            table = entry.get("table")
            count = entry.get("rows")
            if not isinstance(table, str) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{label}.row_counts is malformed.")
            row_counts.append((table, count))
        raw_columns = _require_list(value.get("columns"), f"{label}.columns")
        columns: list[tuple[str, tuple[str, ...]]] = []
        for index, raw_entry in enumerate(raw_columns):
            entry = _require_mapping(raw_entry, f"{label}.columns[{index}]")
            table = entry.get("table")
            raw_names = entry.get("columns")
            if not isinstance(table, str) or not isinstance(raw_names, list):
                raise ValueError(f"{label}.columns[{index}] is malformed.")
            names = tuple(raw_names)
            if any(not isinstance(name, str) for name in names):
                raise ValueError(f"{label}.columns[{index}] is malformed.")
            columns.append((table, names))
        if tuple(table for table, _columns in columns) != tuple(
            table for table, _count in row_counts
        ):
            raise ValueError(f"{label} table order is inconsistent.")
        return cls(sha256, tuple(row_counts), tuple(columns))


@dataclass(frozen=True)
class _CompletedStage:
    stage: str
    checkpoint_stage: str
    checkpoint_filename: str
    checkpoint_sha256: str
    identity: FrameIdentity
    metadata: dict[str, object]
    wrote_frame: bool


@dataclass(frozen=True)
class StageRunContext:
    """Validated immutable view of an outer-stage run context."""

    pipeline_sha256: str
    completed: tuple[str, ...]
    run_config: dict[str, object]
    _records: tuple[_CompletedStage, ...]

    @property
    def metadata(self) -> dict[str, dict[str, object]]:
        """Return a copy of all completed per-stage JSON metadata."""

        return {
            record.stage: _normalize_json_mapping(record.metadata, label="metadata")
            for record in self._records
        }


@dataclass(frozen=True)
class LoadedStageCheckpoint:
    """A stage-bound Frame plus that stage's accumulated JSON metadata."""

    stage: str
    checkpoint_stage: str
    path: Path
    frame: Frame
    identity: FrameIdentity
    metadata: dict[str, object]


class StageRuntime:
    """Advance and resume one exact prefix of an immutable stage pipeline."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        pipeline: StagePipeline,
        *,
        run_config: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(pipeline, StagePipeline):
            raise TypeError("pipeline must be a StagePipeline.")
        self._root = Path(checkpoint_dir)
        self._pipeline = pipeline
        self._context_path = self._root / OUTER_STAGE_CONTEXT_FILENAME
        requested_config = (
            None
            if run_config is None
            else _normalize_json_mapping(run_config, label="run_config")
        )
        self._root.mkdir(parents=True, exist_ok=True)
        if self._context_path.exists():
            self._context = self._load_context()
            if requested_config is not None:
                self.require_run_config(requested_config)
        else:
            self._context = StageRunContext(
                pipeline_sha256=pipeline.sha256,
                completed=(),
                run_config=requested_config or {},
                _records=(),
            )
            self._write_context(self._context)

    @property
    def context(self) -> StageRunContext:
        """Return the latest validated run context."""

        self._context = self._load_context()
        return self._context

    @property
    def metadata(self) -> dict[str, dict[str, object]]:
        """Return completed stage metadata keyed in pipeline order."""

        return self.context.metadata

    def require_run_config(self, expected: Mapping[str, object]) -> dict[str, object]:
        """Require the resume arguments to equal the locked first-run config."""

        normalized = _normalize_json_mapping(expected, label="run_config")
        context = self._load_context()
        if normalized != context.run_config:
            raise ValueError(
                "outer-stage run_config differs from the locked run context; "
                "start a new checkpoint directory for changed build inputs."
            )
        self._context = context
        return _normalize_json_mapping(context.run_config, label="run_config")

    def require_ready(self, stage_name: str) -> None:
        """Require that exactly the named stage's predecessor prefix completed."""

        context = self._load_context()
        stage_index = self._pipeline.index(stage_name)
        if stage_index < len(context.completed):
            raise ValueError(f"stage {stage_name!r} is already complete.")
        expected = self._pipeline.names[:stage_index]
        if context.completed != expected:
            raise ValueError(
                f"stage {stage_name!r} requires completed prefix {expected}, "
                f"but the run context has {context.completed}."
            )
        self._context = context

    def complete(
        self,
        stage_name: str,
        frame: Frame,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> LoadedStageCheckpoint:
        """Write a lossless Frame checkpoint, then atomically complete a stage."""

        self.require_ready(stage_name)
        if not isinstance(frame, Frame):
            raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
        stage_index = self._pipeline.index(stage_name)
        identity = frame_identity(frame)
        stage_metadata = _normalize_json_mapping(metadata or {}, label="metadata")
        checkpoint_path = self._checkpoint_path(stage_index, stage_name)
        write_frame_checkpoint(
            checkpoint_path,
            frame,
            metadata={
                "artifact_kind": _ARTIFACT_KIND,
                "identity": identity.to_payload(),
                "pipeline_sha256": self._pipeline.sha256,
                "schema_version": OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
                "stage": stage_name,
                "stage_index": stage_index,
            },
        )
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        record = _CompletedStage(
            stage=stage_name,
            checkpoint_stage=stage_name,
            checkpoint_filename=checkpoint_path.name,
            checkpoint_sha256=checkpoint_sha256,
            identity=identity,
            metadata=stage_metadata,
            wrote_frame=True,
        )
        self._append_record(record)
        return LoadedStageCheckpoint(
            stage=stage_name,
            checkpoint_stage=stage_name,
            path=checkpoint_path,
            frame=frame,
            identity=identity,
            metadata=_normalize_json_mapping(stage_metadata, label="metadata"),
        )

    def complete_without_frame(
        self,
        stage_name: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Complete a non-mutating stage by carrying its predecessor checkpoint."""

        self.require_ready(stage_name)
        context = self._context
        if not context._records:
            raise ValueError(
                f"stage {stage_name!r} has no predecessor Frame checkpoint to carry."
            )
        predecessor = context._records[-1]
        record = _CompletedStage(
            stage=stage_name,
            checkpoint_stage=predecessor.checkpoint_stage,
            checkpoint_filename=predecessor.checkpoint_filename,
            checkpoint_sha256=predecessor.checkpoint_sha256,
            identity=predecessor.identity,
            metadata=_normalize_json_mapping(metadata or {}, label="metadata"),
            wrote_frame=False,
        )
        self._append_record(record)
        return self._root / record.checkpoint_filename

    def load(
        self,
        stage_name: str,
        *,
        frame_metadata_key: str | None = None,
    ) -> LoadedStageCheckpoint:
        """Load and validate the Frame checkpoint bound to a completed stage.

        ``frame_metadata_key`` names a stage-metadata entry whose mapping
        value is restored as :class:`~microcosm.frame.Frame` metadata during
        the loader's one Frame construction (checkpoints do not serialize
        frame metadata). The entry travels in the run context and was
        normalized at :meth:`complete`, so restoring from it keeps the
        metadata bound to the recorded stage rather than to a side channel.
        Naming a key that is absent or not a mapping fails closed.
        """

        context = self._load_context()
        stage_index = self._pipeline.index(stage_name)
        if stage_index >= len(context._records):
            raise ValueError(f"stage {stage_name!r} is not complete.")
        record = context._records[stage_index]
        if record.stage != stage_name:
            raise ValueError(
                f"run context record {stage_index} is {record.stage!r}, not "
                f"{stage_name!r}."
            )
        frame_metadata: Mapping[str, object] | None = None
        if frame_metadata_key is not None:
            value = record.metadata.get(frame_metadata_key)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"stage {record.stage!r} metadata carries no mapping under "
                    f"{frame_metadata_key!r} to restore as frame metadata."
                )
            frame_metadata = value
        checkpoint_path = self._root / record.checkpoint_filename
        actual_checkpoint_sha256 = _file_sha256(checkpoint_path)
        if actual_checkpoint_sha256 != record.checkpoint_sha256:
            raise ValueError(
                f"outer-stage checkpoint {checkpoint_path} SHA-256 changed: "
                f"expected {record.checkpoint_sha256}, got "
                f"{actual_checkpoint_sha256}."
            )
        loaded = load_frame_checkpoint(checkpoint_path, frame_metadata=frame_metadata)
        expected_checkpoint_index = self._pipeline.index(record.checkpoint_stage)
        expected_metadata = {
            "artifact_kind": _ARTIFACT_KIND,
            "identity": record.identity.to_payload(),
            "pipeline_sha256": self._pipeline.sha256,
            "schema_version": OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
            "stage": record.checkpoint_stage,
            "stage_index": expected_checkpoint_index,
        }
        if loaded.metadata != expected_metadata:
            raise ValueError(
                f"outer-stage checkpoint {checkpoint_path} metadata does not "
                "match its run-context binding."
            )
        actual_identity = frame_identity(loaded.frame)
        if actual_identity != record.identity:
            raise ValueError(
                f"outer-stage checkpoint {checkpoint_path} Frame identity changed."
            )
        self._context = context
        return LoadedStageCheckpoint(
            stage=stage_name,
            checkpoint_stage=record.checkpoint_stage,
            path=checkpoint_path,
            frame=loaded.frame,
            identity=actual_identity,
            metadata=_normalize_json_mapping(record.metadata, label="metadata"),
        )

    def load_predecessor(
        self,
        stage_name: str,
        *,
        frame_metadata_key: str | None = None,
    ) -> LoadedStageCheckpoint | None:
        """Validate readiness and load the immediate predecessor, if one exists."""

        self.require_ready(stage_name)
        stage_index = self._pipeline.index(stage_name)
        if stage_index == 0:
            return None
        return self.load(
            self._pipeline.names[stage_index - 1],
            frame_metadata_key=frame_metadata_key,
        )

    def _checkpoint_path(self, stage_index: int, stage_name: str) -> Path:
        return self._root / f"{stage_index:03d}_{stage_name}.frame.h5"

    def _append_record(self, record: _CompletedStage) -> None:
        expected_stage = self._pipeline.names[len(self._context._records)]
        if record.stage != expected_stage:
            raise AssertionError(
                f"cannot append stage {record.stage!r}; expected {expected_stage!r}."
            )
        updated = StageRunContext(
            pipeline_sha256=self._context.pipeline_sha256,
            completed=(*self._context.completed, record.stage),
            run_config=self._context.run_config,
            _records=(*self._context._records, record),
        )
        self._write_context(updated)
        self._context = updated

    def _load_context(self) -> StageRunContext:
        try:
            raw = json.loads(self._context_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"outer-stage run context is not valid JSON: {self._context_path}."
            ) from error
        payload = _require_mapping(raw, "run context")
        if payload.get("artifact_kind") != _CONTEXT_ARTIFACT_KIND:
            raise ValueError("outer-stage run context has the wrong artifact kind.")
        if payload.get("schema_version") != OUTER_STAGE_CONTEXT_SCHEMA_VERSION:
            raise ValueError("outer-stage run context has an unknown schema version.")
        stored_pipeline = _require_mapping(payload.get("pipeline"), "pipeline")
        stored_digest = payload.get("pipeline_sha256")
        if stored_digest != _mapping_sha256(stored_pipeline):
            raise ValueError("outer-stage run context pipeline digest is invalid.")
        if stored_pipeline != self._pipeline.to_payload() or (
            stored_digest != self._pipeline.sha256
        ):
            raise ValueError(
                "outer-stage run context pipeline differs from the requested pipeline."
            )
        raw_completed = _require_list(payload.get("completed"), "completed")
        completed = tuple(raw_completed)
        if any(not isinstance(name, str) for name in completed):
            raise ValueError("outer-stage run context completed must contain names.")
        expected_prefix = self._pipeline.names[: len(completed)]
        if completed != expected_prefix:
            raise ValueError(
                "outer-stage run context completed stages are not an exact prefix "
                f"of the pipeline: got {completed}, expected {expected_prefix}."
            )
        run_config = _normalize_json_mapping(
            _require_mapping(payload.get("run_config"), "run_config"),
            label="run_config",
        )
        raw_records = _require_list(payload.get("stage_records"), "stage_records")
        if len(raw_records) != len(completed):
            raise ValueError(
                "outer-stage run context must have one record per completed stage."
            )
        records = tuple(
            self._record_from_payload(entry, index=index)
            for index, entry in enumerate(raw_records)
        )
        if tuple(record.stage for record in records) != completed:
            raise ValueError(
                "outer-stage run-context records do not match the completed prefix."
            )
        return StageRunContext(
            pipeline_sha256=str(stored_digest),
            completed=completed,
            run_config=run_config,
            _records=records,
        )

    def _record_from_payload(self, payload: object, *, index: int) -> _CompletedStage:
        value = _require_mapping(payload, f"stage_records[{index}]")
        stage = value.get("stage")
        checkpoint_stage = value.get("checkpoint_stage")
        checkpoint_filename = value.get("checkpoint_filename")
        checkpoint_sha256 = value.get("checkpoint_sha256")
        wrote_frame = value.get("wrote_frame")
        if not isinstance(stage, str) or not isinstance(checkpoint_stage, str):
            raise ValueError(f"stage_records[{index}] stage binding is malformed.")
        if checkpoint_stage not in self._pipeline.names[: index + 1]:
            raise ValueError(
                f"stage_records[{index}] checkpoint stage is not a predecessor."
            )
        expected_filename = self._checkpoint_path(
            self._pipeline.index(checkpoint_stage), checkpoint_stage
        ).name
        if checkpoint_filename != expected_filename:
            raise ValueError(
                f"stage_records[{index}] checkpoint filename is not canonical."
            )
        if not isinstance(checkpoint_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", checkpoint_sha256
        ):
            raise ValueError(
                f"stage_records[{index}].checkpoint_sha256 must be a lowercase "
                "SHA-256 digest."
            )
        if not isinstance(wrote_frame, bool):
            raise ValueError(f"stage_records[{index}].wrote_frame must be boolean.")
        if wrote_frame != (stage == checkpoint_stage):
            raise ValueError(f"stage_records[{index}] Frame ownership is malformed.")
        identity = FrameIdentity.from_payload(
            value.get("identity"), label=f"stage_records[{index}].identity"
        )
        metadata = _normalize_json_mapping(
            _require_mapping(value.get("metadata"), f"stage_records[{index}].metadata"),
            label=f"stage_records[{index}].metadata",
        )
        return _CompletedStage(
            stage=stage,
            checkpoint_stage=checkpoint_stage,
            checkpoint_filename=str(checkpoint_filename),
            checkpoint_sha256=checkpoint_sha256,
            identity=identity,
            metadata=metadata,
            wrote_frame=wrote_frame,
        )

    def _write_context(self, context: StageRunContext) -> None:
        payload = {
            "artifact_kind": _CONTEXT_ARTIFACT_KIND,
            "completed": list(context.completed),
            "pipeline": self._pipeline.to_payload(),
            "pipeline_sha256": self._pipeline.sha256,
            "run_config": context.run_config,
            "schema_version": OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
            "stage_records": [
                {
                    "checkpoint_filename": record.checkpoint_filename,
                    "checkpoint_sha256": record.checkpoint_sha256,
                    "checkpoint_stage": record.checkpoint_stage,
                    "identity": record.identity.to_payload(),
                    "metadata": record.metadata,
                    "stage": record.stage,
                    "wrote_frame": record.wrote_frame,
                }
                for record in context._records
            ],
        }
        _atomic_write_json(self._context_path, payload)


def frame_identity(frame: Frame) -> FrameIdentity:
    """Digest ordered entity IDs, memberships, clone metadata and provenance."""

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    selected = _identity_tables_and_columns(frame)
    row_counts = tuple((name, len(table)) for name, table, _columns in selected)
    columns = tuple((name, names) for name, _table, names in selected)
    digest = hashlib.sha256()
    schema_payload = {
        "links": [
            {
                "left_entity": link.left_entity,
                "name": link.name,
                "right_entity": link.right_entity,
            }
            for link in frame.schema.links
        ],
        "person_entity": frame.schema.person_entity,
        "group_entities": list(frame.schema.group_entities),
    }
    _update_length_prefixed(digest, _canonical_json(schema_payload).encode())
    for table_name, table, identity_columns in selected:
        header = {
            "columns": [
                {"dtype": str(table[column].dtype), "name": column}
                for column in identity_columns
            ],
            "rows": len(table),
            "table": table_name,
        }
        _update_length_prefixed(digest, _canonical_json(header).encode())
        for column in identity_columns:
            series = table[column]
            for start in range(0, len(series), _IDENTITY_HASH_CHUNK_ROWS):
                chunk = series.iloc[start : start + _IDENTITY_HASH_CHUNK_ROWS]
                hashes = pd.util.hash_pandas_object(
                    chunk, index=False, categorize=False
                ).to_numpy(dtype="<u8", copy=False)
                _update_length_prefixed(digest, hashes.tobytes())
    return FrameIdentity(digest.hexdigest(), row_counts, columns)


def assert_unchanged_identity(
    before: Frame,
    after: Frame,
    *,
    stage: str,
) -> None:
    """Assert a non-clone transform preserved structural identity and row order."""

    before_identity = frame_identity(before)
    after_identity = frame_identity(after)
    if after_identity != before_identity:
        raise AssertionError(
            f"stage {stage!r} changed Frame identity or row order: "
            f"{before_identity.sha256} -> {after_identity.sha256}; "
            f"row counts {before_identity.row_counts} -> {after_identity.row_counts}."
        )


def assert_clone_expansion(
    before: Frame,
    after: Frame,
    *,
    channels: Sequence[str],
) -> None:
    """Assert exact two-channel, channel-major structural clone expansion."""

    ordered_channels = tuple(channels)
    if (
        len(ordered_channels) != 2
        or any(not isinstance(channel, str) or not channel for channel in channels)
        or len(set(ordered_channels)) != 2
    ):
        raise ValueError("clone invariant requires exactly two unique channel names.")
    if before.schema != after.schema:
        raise AssertionError("clone expansion changed the Frame entity schema.")
    if before.links != after.links:
        raise AssertionError("clone expansion changed the set of link tables.")

    for entity in before.entities:
        source = before.table(entity)
        expanded = after.table(entity)
        n_rows = len(source)
        if len(expanded) != 2 * n_rows:
            raise AssertionError(
                f"clone expansion must double {entity!r} rows: expected "
                f"{2 * n_rows}, got {len(expanded)}."
            )
        primary_id = before.schema.entity_id_column(entity)
        source_id = f"{entity}_source_id"
        channel_column = f"{entity}_support_channel"
        clone_index_column = f"{entity}_support_clone_index"
        metadata_columns = (source_id, channel_column, clone_index_column)
        collisions = [column for column in metadata_columns if column in source]
        if collisions:
            raise AssertionError(
                f"clone source {entity!r} already carries clone metadata {collisions}."
            )
        missing = [column for column in metadata_columns if column not in expanded]
        if missing:
            raise AssertionError(
                f"cloned {entity!r} table is missing metadata columns {missing}."
            )
        expected_source_ids = np.tile(source[primary_id].to_numpy(), 2)
        if not _ordered_values_equal(expanded[source_id], expected_source_ids):
            raise AssertionError(
                f"cloned {entity!r} source-ID order does not reproduce the source "
                "once per channel."
            )
        expected_channels = np.repeat(ordered_channels, n_rows)
        if not _ordered_values_equal(expanded[channel_column], expected_channels):
            raise AssertionError(
                f"cloned {entity!r} channel order is not channel-major "
                f"{ordered_channels}."
            )
        expected_clone_indexes = np.repeat(np.arange(2), n_rows)
        if not _ordered_values_equal(
            expanded[clone_index_column], expected_clone_indexes
        ):
            raise AssertionError(
                f"cloned {entity!r} clone-index order must be 0 then 1."
            )
        if not _ordered_values_equal(
            expanded[primary_id].iloc[:n_rows], source[primary_id]
        ):
            raise AssertionError(
                f"cloned {entity!r} first channel did not retain source IDs."
            )
        for provenance in _POOLED_SOURCE_PROVENANCE_COLUMNS:
            if provenance not in source:
                continue
            if provenance not in expanded or not _ordered_values_equal(
                expanded[provenance], np.tile(source[provenance].to_numpy(), 2)
            ):
                raise AssertionError(
                    f"cloned {entity!r} did not preserve pooled provenance "
                    f"{provenance!r} in channel-major order."
                )

    person_entity = before.schema.person_entity
    before_person = before.table(person_entity)
    after_person = after.table(person_entity)
    for group in before.schema.group_entities:
        membership = before.schema.membership_column(group)
        group_id = before.schema.entity_id_column(group)
        group_source_id = f"{group}_source_id"
        after_group = after.table(group)
        group_rows = len(before.table(group))
        for clone_index in range(2):
            group_start = clone_index * group_rows
            group_stop = group_start + group_rows
            mapping = dict(
                zip(
                    after_group[group_source_id].iloc[group_start:group_stop],
                    after_group[group_id].iloc[group_start:group_stop],
                    strict=True,
                )
            )
            expected_membership = before_person[membership].map(mapping)
            person_start = clone_index * len(before_person)
            person_stop = person_start + len(before_person)
            if expected_membership.isna().any() or not _ordered_values_equal(
                after_person[membership].iloc[person_start:person_stop],
                expected_membership,
            ):
                raise AssertionError(
                    f"cloned person membership {membership!r} is inconsistent with "
                    f"the {group!r} clone IDs for channel index {clone_index}."
                )

    expected_strata = pd.concat([before.strata, before.strata], ignore_index=True)
    if not _ordered_values_equal(after.strata, expected_strata):
        raise AssertionError("clone expansion did not preserve channel-major strata.")

    for link_name in before.links:
        source_link = before.link(link_name)
        expanded_link = after.link(link_name)
        if len(expanded_link) != 2 * len(source_link):
            raise AssertionError(f"clone expansion did not double link {link_name!r}.")
        link = next(link for link in before.schema.links if link.name == link_name)
        for entity in (link.left_entity, link.right_entity):
            id_column = before.schema.entity_id_column(entity)
            before_entity = before.table(entity)
            after_entity = after.table(entity)
            entity_source_id = f"{entity}_source_id"
            for clone_index in range(2):
                entity_rows = len(before_entity)
                entity_start = clone_index * entity_rows
                entity_stop = entity_start + entity_rows
                mapping = dict(
                    zip(
                        after_entity[entity_source_id].iloc[entity_start:entity_stop],
                        after_entity[id_column].iloc[entity_start:entity_stop],
                        strict=True,
                    )
                )
                expected = source_link[id_column].map(mapping)
                link_start = clone_index * len(source_link)
                link_stop = link_start + len(source_link)
                if expected.isna().any() or not _ordered_values_equal(
                    expanded_link[id_column].iloc[link_start:link_stop], expected
                ):
                    raise AssertionError(
                        f"cloned link {link_name!r} IDs are inconsistent for "
                        f"{entity!r} channel index {clone_index}."
                    )


def _identity_tables_and_columns(
    frame: Frame,
) -> tuple[tuple[str, pd.DataFrame, tuple[str, ...]], ...]:
    selected: list[tuple[str, pd.DataFrame, tuple[str, ...]]] = []
    for entity in frame.entities:
        table = frame.table(entity)
        candidates = [frame.schema.entity_id_column(entity)]
        if entity == frame.schema.person_entity:
            candidates.extend(
                frame.schema.membership_column(group)
                for group in frame.schema.group_entities
            )
        candidates.extend(
            (
                f"{entity}_source_id",
                f"{entity}_support_channel",
                f"{entity}_support_clone_index",
            )
        )
        candidates.extend(_POOLED_SOURCE_PROVENANCE_COLUMNS)
        names = tuple(dict.fromkeys(name for name in candidates if name in table))
        selected.append((entity, table, names))
    links = {link.name: link for link in frame.schema.links}
    for link_name in frame.links:
        table = frame.link(link_name)
        link = links[link_name]
        names = (
            frame.schema.entity_id_column(link.left_entity),
            frame.schema.entity_id_column(link.right_entity),
        )
        selected.append((link_name, table, names))
    return tuple(selected)


def _ordered_values_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    left_series = pd.Series(left).reset_index(drop=True)
    right_series = pd.Series(right).reset_index(drop=True)
    if left_series.equals(right_series):
        return True
    if len(left_series) != len(right_series):
        return False
    for left_value, right_value in zip(left_series, right_series, strict=True):
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if left_value != right_value:
            return False
    return True


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        data = (_canonical_json(payload) + "\n").encode("utf-8")
        with temporary.open("wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_json_mapping(
    value: Mapping[str, object], *, label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    try:
        canonical = _canonical_json(dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must contain finite JSON-compatible values."
        ) from error
    normalized = json.loads(canonical)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must encode a JSON object.")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="little", signed=False))
    digest.update(value)


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")
    return value
