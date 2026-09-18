"""Source-codec registry and shipped codec contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.codecs as codecs_module
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import (
    SOURCE_CODECS,
    SourceCodecRegistry,
    load_raw_bytes,
    load_source,
    load_source_bytes,
)
from microcosm.graph.keys import source_content_key
from microcosm.graph.store import StoreUnavailable


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


# --- raw-byte codecs ---------------------------------------------------------


def test_raw_bytes_codec_reads_one_regular_file_verbatim(tmp_path: Path) -> None:
    """``raw-bytes-v1`` ships registered in raw-byte mode and interprets nothing."""

    payload = b"\x00lookup\xff" * 3
    source = tmp_path / "table.npz"
    source.write_bytes(payload)
    assert SOURCE_CODECS.bytes_names() == ("raw-bytes-v1",)
    assert "raw-bytes-v1" not in SOURCE_CODECS.names()
    assert SOURCE_CODECS.get("raw-bytes-v1") is load_raw_bytes
    assert load_source_bytes("raw-bytes-v1", source) == payload
    assert SOURCE_CODECS.load_bytes("raw-bytes-v1", source) == payload
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert load_source_bytes("raw-bytes-v1", empty) == b""
    # A symlink is followed, as the executor's source resolution already does.
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    assert load_source_bytes("raw-bytes-v1", link) == payload


def test_raw_bytes_codec_refuses_anything_but_one_bounded_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="is a directory"):
        load_raw_bytes(tmp_path)
    with pytest.raises(ValueError, match="not readable"):
        load_raw_bytes(tmp_path / "absent.bin")
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    # Opened non-blocking, so a FIFO with no writer is refused from its mode
    # instead of hanging the run.
    with pytest.raises(ValueError, match="not a regular file"):
        load_raw_bytes(fifo)
    monkeypatch.setattr(codecs_module, "RAW_BYTES_MAX_BYTES", 8)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 9)
    with pytest.raises(ValueError, match="larger than the 8-byte"):
        load_raw_bytes(big)
    bounded = tmp_path / "bounded.bin"
    bounded.write_bytes(b"x" * 8)
    assert load_raw_bytes(bounded) == b"x" * 8


def test_registry_modes_are_exclusive_and_names_reserved_across_modes(
    tmp_path: Path,
) -> None:
    registry = SourceCodecRegistry()
    registry.register("frame", lambda path, *, store=None: _frame())
    registry.register_bytes("bytes", lambda path, *, store=None: b"raw")
    assert registry.names() == ("frame",)
    assert registry.bytes_names() == ("bytes",)
    assert set(registry.as_mapping()) == {"frame"}
    assert set(registry.as_bytes_mapping()) == {"bytes"}
    # Availability resolves in either mode; decoding holds each to its own.
    registry.get("frame")
    registry.get("bytes")
    source = tmp_path / "source"
    source.write_bytes(b"")
    assert registry.load_bytes("bytes", source) == b"raw"
    assert isinstance(registry.load("frame", source), Frame)
    with pytest.raises(TypeError, match="is a raw-bytes codec"):
        registry.load("bytes", source)
    with pytest.raises(TypeError, match="is a Frame codec"):
        registry.load_bytes("frame", source)
    with pytest.raises(ValueError, match="already registered as a raw-bytes codec"):
        registry.register("bytes", lambda path, *, store=None: _frame())
    with pytest.raises(ValueError, match="already registered as a Frame codec"):
        registry.register_bytes("frame", lambda path, *, store=None: b"")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_bytes("bytes", lambda path, *, store=None: b"other")
    with pytest.raises(ValueError, match="non-empty strings"):
        registry.register_bytes("", lambda path, *, store=None: b"")
    with pytest.raises(TypeError, match="must be callable"):
        registry.register_bytes("thing", object())  # type: ignore[arg-type]
    with pytest.raises(StoreUnavailable, match="not installed"):
        registry.load_bytes("absent", source)
    with pytest.raises(StoreUnavailable, match="not installed"):
        registry.get("absent")

    registry.register_bytes("text", lambda path, *, store=None: "text")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match="returned str, not bytes"):
        registry.load_bytes("text", source)

    def needs_dependency(path: Path, *, store: ContentStore | None = None) -> bytes:
        raise ImportError("no dependency")

    registry.register_bytes("needs", needs_dependency)
    with pytest.raises(StoreUnavailable, match="unavailable dependency"):
        registry.load_bytes("needs", source)


class _BytesCreate:
    """A CREATE kernel whose only source is a raw-byte lookup table."""

    ref = "bytes.create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self) -> str:
        return "0" * 64

    def run(self, context: KernelContext) -> KernelResult:
        payload = load_source_bytes("raw-bytes-v1", context.sources["table"])
        frame = _frame()
        household = frame.table("household").assign(
            size=pd.Series([len(payload)] * 2, dtype="int64")
        )
        return KernelResult(
            frame=Frame(
                {"person": frame.table("person"), "household": household},
                frame.schema,
                {"household": frame.weights_for("household")},
                frame.strata,
            ),
            receipt={"payload_bytes": len(payload)},
        )


def test_a_graph_source_may_be_raw_bytes_and_keys_on_its_content(
    tmp_path: Path,
) -> None:
    """The executor admits a raw-byte SourceRef and hands the kernel its path.

    Identity, the pre-run content key and the post-run mutation check stay in
    the executor, so the codec adds no second identity scheme.
    """

    table = tmp_path / "table.bin"
    table.write_bytes(b"0123456789")
    graph = Graph(
        "toy",
        (SourceRef("table", "raw-bytes-v1"),),
        (
            Node(
                "survey",
                _BytesCreate.ref,
                sources=("table",),
                structural=StructuralDelta.CREATE,
                outputs=(
                    Owned("person", "flag", "boolean"),
                    Owned("household", "size", "int64"),
                ),
            ),
        ),
    )
    registry = KernelRegistry()
    registry.register(_BytesCreate())
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compile_graph(graph), sources={"table": table}, store=store, kernels=registry
    )
    assert manifest.nodes["survey"].receipt["payload_bytes"] == 10
    assert set(manifest.populations["survey"].household["size"]) == {10}
    cold_key = manifest.nodes["survey"].key
    cold_source = source_content_key("table", table)
    table.write_bytes(b"0123456789!")
    assert source_content_key("table", table) != cold_source
    changed = run_graph(
        compile_graph(graph), sources={"table": table}, store=store, kernels=registry
    )
    assert changed.nodes["survey"].key != cold_key
    assert changed.nodes["survey"].receipt["payload_bytes"] == 11
