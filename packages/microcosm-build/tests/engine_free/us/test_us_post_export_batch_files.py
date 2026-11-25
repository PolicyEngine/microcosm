"""The post-export scorer keeps its household batches in files (1.4 · 4).

``_HouseholdBatchedPostExportScorer`` loads the written H5 once, bound to its
sha256, and splits it into household batches. It writes each batch to its own
file with the file's sha256 recorded, frees the frames, and reads a batch back
from its file each time it scores it. So the process that scores holds the
H5's path and the batch files' paths and digests, and no frame of the written
H5 between batches. These tests run the scorer against the fake engine on a
small nested frame; ``engine_workflow/us/test_us_post_export_scoring.py``
checks the batches of a fixture H5 written by the engine's own writer.
"""

from __future__ import annotations

# Test functions take the imported ``builder`` fixture as a parameter.
# ruff: noqa: F811
import gc
import pickle
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.units import US_SCHEMA
from test_support.microcosm_build.us_post_export_batch_files import assert_same_batch
from test_support.microcosm_build.us_post_export_scoring import (
    _EngineLog,
    _fake_engine,
    _nested_frame,
    _written_h5,
    builder,  # noqa: F401 - the module-scoped fixture
)

_AGE_KEY = ("age", 2024, None)


def _frame_with_edge_values() -> Frame:
    """The nested fixture frame, plus columns whose values a lossy copy would
    change: a negative zero and a NaN, 32-bit integers, booleans, strings."""

    nested = _nested_frame()
    tables = {entity: nested.table(entity).copy() for entity in nested.entities}
    person = tables["person"]
    n_persons = len(person)
    person["signed_amount"] = np.resize(
        np.asarray([-0.0, 0.0, np.nan, 1.5, -2.25]), n_persons
    )
    person["count_32"] = np.arange(n_persons, dtype=np.int32) - 3
    person["is_flagged"] = np.arange(n_persons) % 2 == 0
    person["label"] = pd.Series(
        [f"person {index}" for index in range(n_persons)], dtype=object
    )
    household = tables["household"]
    household["region_name"] = pd.Series(
        [f"region {index % 3}" for index in range(len(household))], dtype=object
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                nested.weights_for("household").values, WeightKind.CALIBRATED
            )
        },
    )


def _open_scorer(
    builder,
    frame: Frame,
    tmp_path: Path,
    *,
    batch_size: int | None = 3,
    scored_frames: list | None = None,
    **options,
):
    """A scorer on ``frame`` (served as the written H5) and the fake engine.

    Every frame the scorer hands an engine is appended to ``scored_frames``.
    """

    def dataset_from_frame(batch_frame):
        if scored_frames is not None:
            scored_frames.append(batch_frame)
        return batch_frame

    return builder._HouseholdBatchedPostExportScorer(
        _written_h5(tmp_path),
        maximum_microsim_batch_size=batch_size,
        microsimulation_cls=_fake_engine(_EngineLog()),
        dataset_from_frame=dataset_from_frame,
        load_frame=lambda load_path, *, expected_sha256: frame,
        **options,
    )


def _frames_held_by(root: object) -> list[str]:
    """Where a Frame or DataFrame is reachable from ``root``'s attributes.

    Follows attributes, containers and the scorer's own helper objects; stops
    at classes, functions and modules, which reach the whole interpreter.
    """

    found: list[str] = []
    visited: set[int] = set()

    def walk(value: object, where: str) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, (Frame, pd.DataFrame)):
            found.append(where)
            return
        if value is None or isinstance(
            value,
            (
                str,
                bytes,
                int,
                float,
                Path,
                type,
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodType,
                types.ModuleType,
            ),
        ):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{where}[{key!r}]")
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                walk(item, f"{where}.{name}")

    walk(root, "scorer")
    return found


def _expected_batches(builder, frame: Frame, batch_size: int) -> list[Frame]:
    """The batches the scorer selected in memory before 1.4 · 4."""

    return [
        frame
        if len(positions) == frame.n("household")
        else builder._select_households_by_position(frame, positions)
        for positions in builder._household_position_batches(
            frame.n("household"), batch_size
        )
    ]


def test_the_open_scorer_holds_no_frame_of_the_written_h5(builder, tmp_path) -> None:
    scorer = _open_scorer(builder, _frame_with_edge_values(), tmp_path)
    try:
        assert scorer.n_batches == 3
        assert _frames_held_by(scorer) == []
        scorer._score([_AGE_KEY], label="age")
        # A pass reads its batches and keeps none of them.
        assert _frames_held_by(scorer) == []
    finally:
        scorer.close()


def test_each_batch_is_read_from_its_own_file_each_time_it_is_scored(
    builder, tmp_path
) -> None:
    frame = _frame_with_edge_values()
    scored_frames: list[Frame] = []
    scorer = _open_scorer(builder, frame, tmp_path, scored_frames=scored_frames)
    try:
        assert len(sorted(scorer.batch_directory.iterdir())) == scorer.n_batches
        scorer._score([_AGE_KEY], label="first pass")
        scorer._score([_AGE_KEY], label="second pass")
    finally:
        scorer.close()

    expected = _expected_batches(builder, frame, 3)
    assert len(scored_frames) == 2 * len(expected)
    # Each pass reads every batch from its file again: no frame object is
    # served twice (every one is still referenced here, so ids are distinct).
    assert len({id(scored) for scored in scored_frames}) == len(scored_frames)
    for scored, selected in zip(scored_frames, expected * 2, strict=True):
        assert_same_batch(scored, selected)


def test_batches_score_the_same_values_as_the_selected_batches(
    builder, tmp_path
) -> None:
    frame = _frame_with_edge_values()
    scorer = _open_scorer(builder, frame, tmp_path / "files")
    try:
        from_files = scorer._score([_AGE_KEY], label="age")[_AGE_KEY]
    finally:
        scorer.close()
    # The fake engine's ages, over the batches the scorer selected in memory.
    engine_cls = _fake_engine(_EngineLog())
    expected_values, expected_weights = [], []
    for selected in _expected_batches(builder, frame, 3):
        values, weights = builder._post_export_values_and_weights(
            engine_cls(dataset=selected).calculate(*_AGE_KEY), _AGE_KEY
        )
        expected_values.append(values)
        expected_weights.append(weights)
    assert from_files._values.tobytes() == np.concatenate(expected_values).tobytes()
    assert from_files._weights.tobytes() == np.concatenate(expected_weights).tobytes()


def test_a_batch_file_whose_bytes_changed_is_refused(builder, tmp_path) -> None:
    scorer = _open_scorer(builder, _nested_frame(), tmp_path)
    try:
        first_file = sorted(scorer.batch_directory.iterdir())[0]
        first_file.write_bytes(first_file.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="SHA-256"):
            scorer._score([_AGE_KEY], label="age")
    finally:
        scorer.close()


def test_a_missing_batch_file_is_refused(builder, tmp_path) -> None:
    scorer = _open_scorer(builder, _nested_frame(), tmp_path)
    try:
        sorted(scorer.batch_directory.iterdir())[-1].unlink()
        with pytest.raises(FileNotFoundError):
            scorer._score([_AGE_KEY], label="age")
    finally:
        scorer.close()


def test_closing_the_scorer_removes_its_batch_files(builder, tmp_path) -> None:
    scorer = _open_scorer(builder, _nested_frame(), tmp_path)
    directory = scorer.batch_directory
    assert directory.is_dir()
    scorer.close()
    assert not directory.exists()
    with pytest.raises(RuntimeError, match="closed"):
        scorer._batches()


def test_a_scorer_dropped_without_closing_removes_its_batch_files(
    builder, tmp_path
) -> None:
    scorer = _open_scorer(builder, _nested_frame(), tmp_path)
    directory = scorer.batch_directory
    assert directory.is_dir()
    del scorer
    gc.collect()
    assert not directory.exists()


def test_a_given_batch_directory_holds_the_files_and_is_removed_at_close(
    builder, tmp_path
) -> None:
    directory = tmp_path / "post_export_batches"
    scorer = _open_scorer(
        builder, _nested_frame(), tmp_path / "h5", batch_directory=directory
    )
    assert scorer.batch_directory == directory
    assert len(sorted(directory.iterdir())) == scorer.n_batches
    scorer.close()
    assert not directory.exists()


def test_a_given_batch_directory_that_exists_is_refused_and_kept(
    builder, tmp_path
) -> None:
    directory = tmp_path / "post_export_batches"
    directory.mkdir()
    kept = directory / "another_scorers_batch"
    kept.write_bytes(b"not this scorer's")
    with pytest.raises(FileExistsError):
        _open_scorer(
            builder, _nested_frame(), tmp_path / "h5", batch_directory=directory
        )
    assert kept.read_bytes() == b"not this scorer's"


def test_a_scorer_that_cannot_write_its_batches_leaves_no_files(
    builder, tmp_path
) -> None:
    nested = _nested_frame()
    tables = {entity: nested.table(entity).copy() for entity in nested.entities}
    # A value pickle cannot write: the first batch's file fails part-written.
    tables["household"]["unwritable"] = pd.Series(
        [lambda: None for _ in range(len(tables["household"]))], dtype=object
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                nested.weights_for("household").values, WeightKind.CALIBRATED
            )
        },
    )
    directory = tmp_path / "post_export_batches"
    # Python 3.13 raises AttributeError for a local function, 3.14
    # PicklingError.
    with pytest.raises((pickle.PicklingError, AttributeError)):
        _open_scorer(builder, frame, tmp_path / "h5", batch_directory=directory)
    gc.collect()
    assert not directory.exists()


def test_certification_removes_an_earlier_attempts_batch_files(
    builder, tmp_path
) -> None:
    """Certification's batch directory is named for it in the build's
    ``--out``; placing the build removes one an earlier, killed attempt left."""

    release_root = tmp_path / "out"
    build_dir = release_root / "build"
    build_dir.mkdir(parents=True)
    stale = release_root / builder.US_POST_EXPORT_BATCH_DIRECTORY
    stale.mkdir()
    (stale / "batch-00000.pickle").write_bytes(b"an earlier attempt's batch")
    record = SimpleNamespace(
        build="fixture-build",
        release_root=release_root,
        build_dir=build_dir,
        payload={"files": {}},
    )
    builder._place_build_for_certification(record, release_root / "releases" / "r")
    assert not stale.exists()
