"""Source-codec registry modes and shipped codec contracts."""

from __future__ import annotations

import contextlib
import json
import os
import signal
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import codecs
from microcosm.graph.codecs import (
    RAW_BYTES_MAX_BYTES,
    SOURCE_CODECS,
    SourceCodecRegistry,
    load_csv_tables,
    load_source,
    load_source_bytes,
)
from microcosm.graph.store import ContentStore, StoreUnavailable


@contextlib.contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    """Fail on a blocked read instead of hanging.

    Nothing bounds a test's runtime here — pytest-timeout is not installed and
    ``addopts`` sets no deadline — so a codec that blocked on a FIFO would take
    the lane out for the runner's whole budget with no assertion and no
    traceback. This turns that into an ordinary failure.
    """

    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - POSIX only
        yield
        return

    def expire(signum: int, frame: object) -> None:
        raise TimeoutError(f"raw-bytes-v1 blocked for more than {seconds}s.")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _frame() -> Frame:
    schema = EntitySchema(group_entities=("household",))
    person = pd.DataFrame(
        {
            "person_id": pd.Series([1, 2, 3], dtype="int64"),
            "person_household_id": pd.Series([10, 10, 20], dtype="int64"),
            "flag": pd.Series([True, pd.NA, False], dtype="boolean"),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": pd.Series([10, 20], dtype="int64"),
            "size": pd.Series([2, 1], dtype="int64"),
        }
    )
    return Frame(
        {"person": person, "household": household},
        schema,
        {"household": Weights(np.asarray([2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["urban", "urban", "rural"], dtype="string"),
    )


def test_csv_tables_loads_assignment_weights_json_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    frame = _frame()
    frame.table("person").assign(stratum=frame.strata).to_csv(
        source / "person.csv", index=False
    )
    frame.table("household").to_csv(source / "household.csv", index=False)
    (source / "schema.json").write_text(
        json.dumps(
            {
                "person_entity": "person",
                "group_entities": ["household"],
                "strata_column": "stratum",
                "dtypes": {
                    "person": {
                        "person_id": "int64",
                        "person_household_id": "int64",
                        "flag": "boolean",
                        "stratum": "string",
                    },
                    "household": {"household_id": "int64", "size": "int64"},
                },
            }
        )
    )
    (source / "weights.json").write_text(
        json.dumps({"household": {"kind": "design", "values": [2.0, 3.0]}})
    )

    loaded = load_source("csv-tables", source)

    assert loaded.schema == frame.schema
    for entity in frame.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), frame.table(entity))
    pd.testing.assert_series_equal(loaded.strata, frame.strata, check_names=False)
    assert loaded.weights_for("household").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        loaded.weights_for("household").values,
        frame.weights_for("household").values,
    )


def test_csv_tables_loads_tables_mapping_and_weights_csv_layout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    frame = _frame()
    frame.table("person").assign(stratum=frame.strata).to_csv(
        source / "people.csv", index=False
    )
    frame.table("household").to_csv(source / "homes.csv", index=False)
    pd.DataFrame({"household_id": [10, 20], "design_weight": [2.0, 3.0]}).to_csv(
        source / "weights.csv", index=False
    )
    (source / "schema.json").write_text(
        json.dumps(
            {
                "person_entity": "person",
                "group_entities": ["household"],
                "tables": {"person": "people.csv", "household": "homes.csv"},
                "dtypes": {
                    "person_id": "int64",
                    "person_household_id": "int64",
                    "household_id": "int64",
                    "size": "int64",
                    "flag": "boolean",
                    "stratum": "string",
                    "design_weight": "float64",
                },
                "strata_column": "stratum",
                "weights": {
                    "entity": "household",
                    "column": "design_weight",
                    "kind": "design",
                },
                "weights_table": "weights.csv",
            }
        )
    )

    loaded = SOURCE_CODECS.load("csv-tables", source)

    for entity in frame.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), frame.table(entity))
    pd.testing.assert_series_equal(loaded.strata, frame.strata, check_names=False)
    np.testing.assert_array_equal(
        loaded.weights_for("household").values,
        np.asarray([2.0, 3.0]),
    )


def test_frame_store_codec_loads_verified_object_directory(tmp_path: Path) -> None:
    frame = _frame()
    store = ContentStore(tmp_path / "store")
    path = store.put_frame("a" * 64, frame)

    loaded = SOURCE_CODECS.load("frame-store", path, store=store)

    assert loaded.schema == frame.schema
    for entity in frame.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), frame.table(entity))


def test_missing_codec_and_codec_dependency_are_store_unavailable(
    tmp_path: Path,
) -> None:
    registry = SourceCodecRegistry()
    with pytest.raises(StoreUnavailable, match="not installed"):
        registry.load("missing", tmp_path)

    def missing_dependency(_path: Path, *, store: ContentStore | None = None) -> Frame:
        del store
        raise ImportError("optional engine")

    registry.register("engine", missing_dependency)
    with pytest.raises(StoreUnavailable, match="dependency"):
        registry.load("engine", tmp_path)


def test_raw_bytes_reads_one_regular_file_and_refuses_anything_else(
    tmp_path: Path,
) -> None:
    payload = np.random.default_rng(0).integers(0, 256, size=1024, dtype=np.uint8)
    lookup = tmp_path / "lookup.npz"
    np.savez(lookup, values=payload)
    expected = lookup.read_bytes()

    loaded = load_source_bytes("raw-bytes-v1", lookup)

    assert isinstance(loaded, bytes)
    assert loaded == expected
    # A symlink to a regular file is the same source: the executor hands the
    # codec a path it already resolved.
    link = tmp_path / "link.npz"
    link.symlink_to(lookup)
    assert load_source_bytes("raw-bytes-v1", link) == expected

    with pytest.raises(ValueError, match="is a directory"):
        load_source_bytes("raw-bytes-v1", tmp_path)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    # Opened non-blocking and refused from the descriptor's own mode. The
    # deadline is the regression guard: a blocking open of a FIFO with no
    # writer would hang here rather than fail.
    with _deadline(10), pytest.raises(ValueError, match="not a regular file"):
        load_source_bytes("raw-bytes-v1", fifo)
    with pytest.raises(ValueError, match="not readable"):
        load_source_bytes("raw-bytes-v1", tmp_path / "absent.npz")


def test_raw_bytes_reads_up_to_the_limit_and_refuses_one_byte_more(
    tmp_path: Path,
) -> None:
    at_limit = tmp_path / "at_limit.bin"
    with at_limit.open("wb") as handle:
        handle.truncate(RAW_BYTES_MAX_BYTES)
    assert len(load_source_bytes("raw-bytes-v1", at_limit)) == RAW_BYTES_MAX_BYTES

    over_limit = tmp_path / "over_limit.bin"
    with over_limit.open("wb") as handle:
        handle.truncate(RAW_BYTES_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="larger than"):
        load_source_bytes("raw-bytes-v1", over_limit)


def test_the_oversize_refusal_reads_no_more_than_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is not the bound; pin what the codec pulls off disk.

    Asserting only that an oversized file raises leaves the bound untested: an
    unbounded ``handle.read()`` reads the whole file and still refuses it on
    size afterwards, with every other assertion here still green.
    """

    over_limit = tmp_path / "over_limit.bin"
    with over_limit.open("wb") as handle:
        handle.truncate(RAW_BYTES_MAX_BYTES + 1)

    requested: list[int] = []
    real_open = open

    class Recording:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return self._handle.read(size)

        def fileno(self) -> int:
            return self._handle.fileno()

        def __enter__(self) -> Recording:
            self._handle.__enter__()
            return self

        def __exit__(self, *exception: object) -> object:
            return self._handle.__exit__(*exception)

    monkeypatch.setattr(
        codecs,
        "open",
        lambda *args, **kwargs: Recording(real_open(*args, **kwargs)),
        raising=False,
    )

    with pytest.raises(ValueError, match="larger than"):
        load_source_bytes("raw-bytes-v1", over_limit)

    assert requested == [RAW_BYTES_MAX_BYTES + 1]


def test_registry_snapshots_are_read_only_and_do_not_track_later_registrations(
    tmp_path: Path,
) -> None:
    registry = SourceCodecRegistry()

    def byte_loader(_path: Path, *, store: ContentStore | None = None) -> bytes:
        del store
        return b""

    registry.register_bytes("blob", byte_loader)
    frame_snapshot = registry.as_mapping()
    byte_snapshot = registry.as_bytes_mapping()

    with pytest.raises(TypeError):
        byte_snapshot["other"] = byte_loader
    with pytest.raises(TypeError):
        frame_snapshot["other"] = load_csv_tables

    registry.register("tables", load_csv_tables)
    assert dict(frame_snapshot) == {}
    assert dict(byte_snapshot) == {"blob": byte_loader}
    assert set(registry.as_mapping()) == {"tables"}


def test_codec_names_are_unique_across_both_modes() -> None:
    registry = SourceCodecRegistry()

    def frame_loader(_path: Path, *, store: ContentStore | None = None) -> Frame:
        del store
        return _frame()

    def byte_loader(_path: Path, *, store: ContentStore | None = None) -> bytes:
        del store
        return b""

    registry.register("tables", frame_loader)
    registry.register_bytes("blob", byte_loader)
    # Re-registering the same loader under its own name stays idempotent.
    assert registry.register("tables", frame_loader) is frame_loader
    assert registry.register_bytes("blob", byte_loader) is byte_loader

    with pytest.raises(ValueError, match="already registered as a Frame codec"):
        registry.register_bytes("tables", byte_loader)
    with pytest.raises(ValueError, match="already registered as a raw-bytes codec"):
        registry.register("blob", frame_loader)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("tables", lambda _path, *, store=None: _frame())

    # Each mode is enumerated by its own pair, so a snapshot taken through
    # either one stays resolvable through it.
    assert registry.names() == ("tables",)
    assert registry.bytes_names() == ("blob",)
    assert dict(registry.as_mapping()) == {"tables": frame_loader}
    assert dict(registry.as_bytes_mapping()) == {"blob": byte_loader}
    assert registry.names() == tuple(sorted(registry.as_mapping()))
    assert registry.bytes_names() == tuple(sorted(registry.as_bytes_mapping()))
    # Availability is answered by name, in either mode: this is the question
    # the executor asks before any kernel runs.
    assert registry.get("tables") is frame_loader
    assert registry.get("blob") is byte_loader
    with pytest.raises(StoreUnavailable, match="not installed"):
        registry.get("absent")


def test_reading_a_codec_in_the_wrong_mode_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "lookup.bin"
    source.write_bytes(b"payload")

    with pytest.raises(TypeError, match="is a raw-bytes codec"):
        SOURCE_CODECS.load("raw-bytes-v1", source)
    with pytest.raises(TypeError, match="is a Frame codec"):
        SOURCE_CODECS.load_bytes("csv-tables", tmp_path)
    with pytest.raises(StoreUnavailable, match="not installed"):
        SOURCE_CODECS.load_bytes("raw-bytes-v2", source)

    assert SOURCE_CODECS.names() == ("csv-tables", "frame-store")
    assert SOURCE_CODECS.bytes_names() == ("raw-bytes-v1",)
    assert set(SOURCE_CODECS.as_mapping()) == {"csv-tables", "frame-store"}
    assert set(SOURCE_CODECS.as_bytes_mapping()) == {"raw-bytes-v1"}


def test_byte_codec_dependency_and_return_type_are_held_to_the_contract(
    tmp_path: Path,
) -> None:
    registry = SourceCodecRegistry()

    def missing_dependency(_path: Path, *, store: ContentStore | None = None) -> bytes:
        del store
        raise ImportError("optional reader")

    def mutable_payload(_path: Path, *, store: ContentStore | None = None) -> bytes:
        del store
        return bytearray(b"mutable")

    registry.register_bytes("engine", missing_dependency)
    registry.register_bytes("mutable", mutable_payload)

    with pytest.raises(StoreUnavailable, match="dependency"):
        registry.load_bytes("engine", tmp_path)
    with pytest.raises(TypeError, match="returned bytearray, not bytes"):
        registry.load_bytes("mutable", tmp_path)
