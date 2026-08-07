"""UK frame-metadata round-trip for outer-stage checkpoints (#612 inc 3).

Frame checkpoints deliberately do not serialize ``Frame.metadata`` — the
caller owns restoring it (:func:`microcosm.build.frame_checkpoint.load_frame_checkpoint`'s
``frame_metadata`` hatch). The UK carrier keeps ``time_period`` in frame
metadata and ``validate_uk_national_frame`` refuses a frame without it, so a
naive checkpoint round-trip would strand every stage boundary.

These helpers close the loop through the stage runtime's own records: the
frame's metadata is embedded in the stage metadata at ``complete`` under one
reserved key, and restored from the (already validated) run-context record
at load. No side channel, no shared checkpoint-schema change.
"""

from __future__ import annotations

from collections.abc import Mapping

from microcosm.build.outer_stage_runtime import LoadedStageCheckpoint, StageRuntime
from microcosm.build.uk_runtime.national_frame import validate_uk_national_frame
from microcosm.frame import Frame

__all__ = [
    "UK_FRAME_METADATA_KEY",
    "load_uk_stage_checkpoint",
    "load_uk_stage_predecessor",
    "uk_stage_metadata",
]

#: Reserved stage-metadata key carrying the frame's own metadata mapping.
UK_FRAME_METADATA_KEY = "uk_frame_metadata"


def uk_stage_metadata(
    frame: Frame,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Stage metadata for ``StageRuntime.complete`` on a UK national frame.

    Validates the frame, then embeds its metadata under
    :data:`UK_FRAME_METADATA_KEY` beside any caller-supplied entries. The
    reserved key is refused in ``extra`` so a caller cannot smuggle a
    different metadata mapping past the frame it claims to describe.
    """

    validate_uk_national_frame(frame)
    metadata: dict[str, object] = {UK_FRAME_METADATA_KEY: dict(frame.metadata)}
    if extra is not None:
        if UK_FRAME_METADATA_KEY in extra:
            raise ValueError(
                f"stage metadata key {UK_FRAME_METADATA_KEY!r} is reserved for "
                "the frame's own metadata."
            )
        metadata.update(extra)
    return metadata


def load_uk_stage_checkpoint(
    runtime: StageRuntime,
    stage_name: str,
) -> LoadedStageCheckpoint:
    """Load a completed UK stage checkpoint with frame metadata restored.

    The frame metadata comes from the run-context record the runtime already
    validated against the checkpoint bytes; the reconstructed frame is then
    revalidated as a UK national frame, so a checkpoint that lost its
    metadata (or was written without :func:`uk_stage_metadata`) fails closed
    rather than surfacing later as a missing ``time_period``.
    """

    loaded = runtime.load(stage_name, frame_metadata_key=UK_FRAME_METADATA_KEY)
    validate_uk_national_frame(loaded.frame)
    return loaded


def load_uk_stage_predecessor(
    runtime: StageRuntime,
    stage_name: str,
) -> LoadedStageCheckpoint | None:
    """Load the immediate predecessor's UK checkpoint, if one exists."""

    loaded = runtime.load_predecessor(
        stage_name,
        frame_metadata_key=UK_FRAME_METADATA_KEY,
    )
    if loaded is not None:
        validate_uk_national_frame(loaded.frame)
    return loaded
