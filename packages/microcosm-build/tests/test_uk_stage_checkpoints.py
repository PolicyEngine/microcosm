"""UK frame-metadata round-trip for outer-stage checkpoints (#612 inc 3).

Frame checkpoints do not serialize frame metadata, and the UK carrier keeps
``time_period`` there — so the round trip must restore it from the stage
runtime's own records, fail closed when it cannot, and hand back a frame
whose content identity matches what was checkpointed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.outer_stage_runtime import Stage, StagePipeline, StageRuntime
from microcosm.build.uk_runtime import uk_frame_content_identity
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.stage_checkpoints import (
    UK_FRAME_METADATA_KEY,
    load_uk_stage_checkpoint,
    load_uk_stage_predecessor,
    uk_stage_metadata,
)

_PIPELINE = StagePipeline(
    (
        Stage("retain", "retained-leaves toy stage"),
        Stage("restore", "spi-restoration toy stage"),
    )
)


def _frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [1, 1],
                "person_benunit_id": [1, 1],
                "pay": [20_000.0, 0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [10.0],
            }
        ),
        time_period="2023",
    )


def _runtime(tmp_path: Path) -> StageRuntime:
    return StageRuntime(
        tmp_path / "checkpoints",
        _PIPELINE,
        run_config={"input_sha256": "a" * 64},
    )


def test_round_trip_restores_metadata_and_content(tmp_path: Path) -> None:
    frame = _frame()
    runtime = _runtime(tmp_path)
    runtime.complete("retain", frame, metadata=uk_stage_metadata(frame))

    # A fresh runtime models the next stage's worker process: nothing
    # survives but the checkpoint directory.
    resumed = _runtime(tmp_path)
    loaded = load_uk_stage_checkpoint(resumed, "retain")

    assert loaded.frame is not frame
    validate_uk_national_frame(loaded.frame)
    assert uk_time_period(loaded.frame) == "2023"
    assert uk_frame_content_identity(loaded.frame) == uk_frame_content_identity(frame)

    predecessor = load_uk_stage_predecessor(resumed, "restore")
    assert predecessor is not None
    assert uk_time_period(predecessor.frame) == "2023"


def test_uk_no_extension_checkpoint_keeps_its_schema_2_byte_golden(
    tmp_path: Path,
) -> None:
    frame = _frame()
    completed = _runtime(tmp_path).complete(
        "retain",
        frame,
        metadata=uk_stage_metadata(frame),
    )

    assert completed.path.name == "000_retain.frame.h5"
    assert hashlib.sha256(completed.path.read_bytes()).hexdigest() == (
        "65952080dbebd4149051659c0b367a5384ff5ad9206a6f8d90b1dba89ff5b631"
    )


def test_checkpoint_without_frame_metadata_fails_closed(tmp_path: Path) -> None:
    frame = _frame()
    runtime = _runtime(tmp_path)
    runtime.complete("retain", frame, metadata={"note": "no reserved key"})

    resumed = _runtime(tmp_path)
    with pytest.raises(ValueError, match=UK_FRAME_METADATA_KEY):
        load_uk_stage_checkpoint(resumed, "retain")


def test_stage_metadata_refuses_reserved_key_collisions() -> None:
    frame = _frame()

    metadata = uk_stage_metadata(frame, extra={"seed": 42})
    assert metadata[UK_FRAME_METADATA_KEY] == {"time_period": "2023"}
    assert metadata["seed"] == 42

    with pytest.raises(ValueError, match="reserved"):
        uk_stage_metadata(frame, extra={UK_FRAME_METADATA_KEY: {}})


def test_plain_runtime_load_still_strands_uk_metadata(tmp_path: Path) -> None:
    # Documents why the helpers exist: the default load path restores no
    # frame metadata, so the UK validator refuses the reconstruction.
    frame = _frame()
    runtime = _runtime(tmp_path)
    runtime.complete("retain", frame, metadata=uk_stage_metadata(frame))

    resumed = _runtime(tmp_path)
    bare = resumed.load("retain")
    with pytest.raises(ValueError, match="time_period"):
        validate_uk_national_frame(bare.frame)


def test_nested_frame_metadata_round_trips_through_the_stage_record(
    tmp_path: Path,
) -> None:
    """Frozen nested metadata is thawed for the record, not refused."""

    import numpy as np

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "person_household_id": [101],
                    "person_benunit_id": [11],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [11]}),
            "household": pd.DataFrame({"household_id": [101]}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0], dtype=np.float64), WeightKind.DESIGN)},
        metadata={
            "time_period": "2023",
            "provenance": {"sources": ["frs", "spi"], "tier": "frs"},
        },
    )
    runtime = _runtime(tmp_path)
    runtime.complete("retain", frame, metadata=uk_stage_metadata(frame))

    loaded = load_uk_stage_checkpoint(_runtime(tmp_path), "retain")
    restored = loaded.frame.metadata
    assert restored["provenance"]["tier"] == "frs"
    assert list(restored["provenance"]["sources"]) == ["frs", "spi"]


def test_set_metadata_round_trips_with_an_unchanged_content_identity(
    tmp_path: Path,
) -> None:
    """A set in frame metadata must not change the frame's content identity.

    JSON has no set type, so a set rides the record as a sequence and
    re-freezes as a tuple. The thaw sorts members with the same canonical
    key the identity digest uses; without that shared order the round trip
    would produce a spurious "drifted record" refusal downstream, since
    set iteration order varies across processes under hash randomization.
    """

    import numpy as np

    from microcosm.build.uk_runtime.content_identity import (
        uk_frame_content_identity,
    )
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "person_household_id": [101],
                    "person_benunit_id": [11],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [11]}),
            "household": pd.DataFrame({"household_id": [101]}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0], dtype=np.float64), WeightKind.DESIGN)},
        metadata={
            "time_period": "2023",
            "provenance": {"sources": {"spi", "frs", "ukds"}},
        },
    )
    runtime = _runtime(tmp_path)
    runtime.complete("retain", frame, metadata=uk_stage_metadata(frame))

    loaded = load_uk_stage_checkpoint(_runtime(tmp_path), "retain")
    restored = loaded.frame.metadata["provenance"]["sources"]
    assert list(restored) == ["frs", "spi", "ukds"]
    assert uk_frame_content_identity(loaded.frame) == uk_frame_content_identity(frame)
