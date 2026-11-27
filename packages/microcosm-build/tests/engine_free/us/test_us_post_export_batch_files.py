"""The post-export scorer keeps its household batches in files.

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
import shutil
import types
import weakref
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
    """The batches the scorer selected, and held in memory, before it kept
    them in files."""

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
        with pytest.raises(builder.PostExportBatchFileError, match="SHA-256"):
            scorer._score([_AGE_KEY], label="age")
    finally:
        scorer.close()


def test_a_missing_batch_file_is_refused(builder, tmp_path) -> None:
    scorer = _open_scorer(builder, _nested_frame(), tmp_path)
    try:
        sorted(scorer.batch_directory.iterdir())[-1].unlink()
        with pytest.raises(builder.PostExportBatchFileError, match="cannot be read"):
            scorer._score([_AGE_KEY], label="age")
    finally:
        scorer.close()


@pytest.mark.parametrize(
    ("writer", "consumer", "filename"),
    [
        (
            "_write_ssi_social_security_holdout",
            "_ssi_social_security_holdout_consumer",
            "us_ssi_social_security_holdout.json",
        ),
        (
            "_write_medicare_irmaa_holdout",
            "_medicare_irmaa_holdout_consumer",
            "us_medicare_irmaa_holdout.json",
        ),
    ],
)
def test_a_holdout_raises_a_refused_batch_file_instead_of_reporting_it(
    builder, tmp_path, writer, consumer, filename
) -> None:
    """A held-out report records a measurement it cannot make as unavailable;
    a batch file that changed or went missing is not such a measurement, and
    stops certification as it does in every other post-export stage."""

    scorer = _open_scorer(builder, _nested_frame(), tmp_path / "h5")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    try:
        first_file = sorted(scorer.batch_directory.iterdir())[0]
        first_file.write_bytes(first_file.read_bytes() + b"changed")
        with pytest.raises(builder.PostExportBatchFileError):
            getattr(builder, writer)(
                release_dir=release_dir,
                dataset_path=scorer.dataset_path,
                release_id="fixture-release",
                post_export_scorer=scorer,
                baseline_plan=builder._record_post_export_baseline_plan(
                    getattr(builder, consumer)
                ),
            )
    finally:
        scorer.close()
    assert not (release_dir / filename).exists()


_MERGED_PLANS = {"first_consumer": (_AGE_KEY,), "second_consumer": (_AGE_KEY,)}


def test_the_merged_baseline_pass_raises_a_changed_batch_file_without_falling_back(
    builder, tmp_path, capsys
) -> None:
    """A batch file that changed stops the merged baseline pass at once: the
    scorer neither logs a fallback nor scores each consumer's plan again."""

    scorer = _open_scorer(
        builder, _nested_frame(), tmp_path, baseline_plans=_MERGED_PLANS
    )
    try:
        first_file = sorted(scorer.batch_directory.iterdir())[0]
        first_file.write_bytes(first_file.read_bytes() + b"changed")
        capsys.readouterr()
        with pytest.raises(builder.PostExportBatchFileError, match="SHA-256"):
            scorer.open_consumer("first_consumer", _MERGED_PLANS["first_consumer"])
        out = capsys.readouterr().out
        assert "merged baseline plan failed" not in out
        assert "first_consumer baseline plan" not in out
        assert not scorer._merged_baseline_failed
    finally:
        scorer.close()


def test_the_merged_baseline_pass_raises_a_batch_file_it_could_not_read_once(
    builder, tmp_path, monkeypatch
) -> None:
    """A batch file the merged pass could not read stops certification, even
    when a later read of the same file would succeed: the per-consumer plans
    do not score it again in a fallback pass."""

    scorer = _open_scorer(
        builder, _nested_frame(), tmp_path, baseline_plans=_MERGED_PLANS
    )
    try:
        first_file = sorted(scorer.batch_directory.iterdir())[0]
        real_read_bytes = Path.read_bytes
        failed_reads = []

        def read_bytes_failing_once(path):
            if path == first_file and not failed_reads:
                failed_reads.append(path)
                raise OSError("fixture read error")
            return real_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", read_bytes_failing_once)
        with pytest.raises(builder.PostExportBatchFileError, match="cannot be read"):
            scorer.open_consumer("first_consumer", _MERGED_PLANS["first_consumer"])
        assert failed_reads == [first_file]
    finally:
        scorer.close()


def test_a_pass_frees_each_batch_before_it_reads_the_next(
    builder, tmp_path, monkeypatch
) -> None:
    """Between batches the scorer holds no batch frame: the frame read for a
    batch is freed (the batch engine holds its own copy of the data) before
    the next batch's file is read."""

    batch_files = builder._PostExportBatchFiles
    read_batch = batch_files.__getitem__
    served: list[weakref.ref] = []
    alive_at_each_read: list[int] = []

    def tracked_read(self, position):
        alive_at_each_read.append(sum(ref() is not None for ref in served))
        batch_frame = read_batch(self, position)
        # A Frame takes no weak reference; its person table, the same object
        # on every call, lives exactly as long as it.
        served.append(weakref.ref(batch_frame.table("person")))
        return batch_frame

    monkeypatch.setattr(batch_files, "__getitem__", tracked_read)
    scorer = _open_scorer(builder, _nested_frame(), tmp_path)
    try:
        scorer._score([_AGE_KEY], label="age")
    finally:
        scorer.close()
    assert alive_at_each_read == [0] * scorer.n_batches


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


def test_a_given_parent_holds_the_scorers_own_directory_until_close(
    builder, tmp_path
) -> None:
    parent = tmp_path / "post_export_batches"
    scorer = _open_scorer(
        builder, _nested_frame(), tmp_path / "h5", batch_parent=parent
    )
    assert scorer.batch_directory.parent == parent
    assert len(sorted(scorer.batch_directory.iterdir())) == scorer.n_batches
    scorer.close()
    # The scorer made the parent, and removes it once its own directory is gone.
    assert not parent.exists()


def test_a_given_parent_that_exists_is_kept_with_what_it_holds(
    builder, tmp_path
) -> None:
    parent = tmp_path / "post_export_batches"
    parent.mkdir()
    kept = parent / "another_scorers_batch"
    kept.write_bytes(b"not this scorer's")
    scorer = _open_scorer(
        builder, _nested_frame(), tmp_path / "h5", batch_parent=parent
    )
    directory = scorer.batch_directory
    scorer.close()
    assert not directory.exists()
    assert kept.read_bytes() == b"not this scorer's"


def test_a_scorer_freed_late_leaves_a_newer_scorers_files_in_the_same_parent(
    builder, tmp_path
) -> None:
    """A scorer that was neither closed nor freed (a failed certification whose
    exception is still held) whose parent is then emptied, as placing a build
    for certification does, must not remove a newer scorer's files there when
    it is finally freed."""

    parent = tmp_path / "out" / "post_export_batches"
    earlier = _open_scorer(
        builder, _nested_frame(), tmp_path / "earlier", batch_parent=parent
    )
    shutil.rmtree(parent)
    newer = _open_scorer(
        builder, _nested_frame(), tmp_path / "newer", batch_parent=parent
    )
    try:
        del earlier
        gc.collect()
        assert len(sorted(newer.batch_directory.iterdir())) == newer.n_batches
        newer._score([_AGE_KEY], label="age")
    finally:
        newer.close()


def test_a_scorer_that_cannot_write_its_batches_leaves_no_files(
    builder, tmp_path
) -> None:
    nested = _nested_frame()
    tables = {entity: nested.table(entity).copy() for entity in nested.entities}
    # A value pickle cannot write: pickling the first batch raises before its
    # file is written.
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
    parent = tmp_path / "post_export_batches"
    # Python 3.13 raises AttributeError for a local function, 3.14
    # PicklingError.
    with pytest.raises((pickle.PicklingError, AttributeError)) as raised:
        _open_scorer(builder, frame, tmp_path / "h5", batch_parent=parent)
    # The traceback still holds the half-made scorer, so its finalizer has not
    # run: the scorer removed its files itself.
    assert raised.traceback
    assert not parent.exists()


def test_a_scorer_whose_disk_fills_leaves_no_part_written_files(
    builder, tmp_path, monkeypatch
) -> None:
    """A write that fails partway (a full disk) leaves no batch file behind,
    neither the part-written one nor those written before it."""

    write_bytes = Path.write_bytes
    batch_writes: list[Path] = []

    def disk_fills_on_the_second_batch(self, data):
        if not self.name.startswith("batch-"):
            return write_bytes(self, data)
        batch_writes.append(self)
        if len(batch_writes) == 2:
            write_bytes(self, data[: len(data) // 2])
            raise OSError(28, "No space left on device")
        return write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", disk_fills_on_the_second_batch)
    parent = tmp_path / "post_export_batches"
    with pytest.raises(OSError, match="No space left") as raised:
        _open_scorer(builder, _nested_frame(), tmp_path / "h5", batch_parent=parent)
    assert raised.traceback
    assert len(batch_writes) == 2
    assert not parent.exists()


def test_certification_removes_an_earlier_attempts_batch_files(
    builder, tmp_path
) -> None:
    """Certification's scorer keeps its batch directory in a parent named for
    it in the build's ``--out``; placing the build removes what an earlier,
    killed attempt left there."""

    release_root = tmp_path / "out"
    build_dir = release_root / "build"
    build_dir.mkdir(parents=True)
    stale = release_root / builder.US_POST_EXPORT_BATCH_DIRECTORY
    (stale / "post-export-batches-earlier").mkdir(parents=True)
    (stale / "post-export-batches-earlier" / "batch-00000.pickle").write_bytes(
        b"an earlier attempt's batch"
    )
    # Not a scorer's directory: someone else's, which placement leaves.
    kept = stale / "notes.txt"
    kept.write_text("not a scorer's")
    record = SimpleNamespace(
        build="fixture-build",
        release_root=release_root,
        build_dir=build_dir,
        payload={"files": {}},
    )
    builder._place_build_for_certification(record, release_root / "releases" / "r")
    assert sorted(path.name for path in stale.iterdir()) == ["notes.txt"]
    assert kept.read_text() == "not a scorer's"
