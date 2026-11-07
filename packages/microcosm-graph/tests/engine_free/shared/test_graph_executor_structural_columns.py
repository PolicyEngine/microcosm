"""A structural version's columns pass through the store one at a time.

A structural node stores every column of its population version as its own
id-indexed column object, and then the whole frame. ``_write_node`` used to
copy every column, each with its own copied id index, before it wrote the
first one, so a write held a second frame and one id array per column. It now
copies each column only when it writes it, copies each entity's ids once, and
lets each copy go before the next. A cache hit on a structural node used to
load every stored column and hold them all while it loaded the frame, though
it returns only the frame. It now checks each column and lets it go. These
tests pin that nothing written or returned moves:

- every store object, key and bytes, and every node's artifacts, in order,
  equal those written from ``_reference_version_series`` below, the old
  structural branch kept verbatim but for its names, for CREATE, FILTER,
  REWEIGHT and EXPAND nodes and the ordinary nodes between them, cold, warm
  and rewritten under ``resume="forbid"``, and a warm run reports what the
  cold run did;
- a direct write of a frame with every graph dtype, nulls, and entities and
  columns out of sorted order returns the reference's artifacts and record
  and stores the reference's objects;
- an unsupported dtype is still refused before anything is written;
- each column copy is gone before the next is made, each entity's columns
  share one id index, and the write's traced peak is a small part of the
  reference's;
- a structural hit still loads every column, in the record's order, and
  refuses a damaged or repeated one, but it holds none of them, so its traced
  peak is its frame load's.

Synthetic frames only: no data file, no country package, no engine.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import tracemalloc
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.executor as graph_executor
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import (
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)
from microcosm.graph.executor import (
    NodeRejected,
    _dtype_token,
    _structural_columns,
    run_graph,
)
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
)
from microcosm.graph.manifest import NodeReceipt, RunManifest
from microcosm.graph.population import Population
from microcosm.graph.store import ContentStore, StoreCorrupt

SOURCE = SourceRef("survey", "csv-tables")


# --------------------------------------------------------------------------
# The pre-change column copies, kept as the reference.
# --------------------------------------------------------------------------


def _reference_series_for_column(frame: Frame, entity: str, column: str) -> pd.Series:
    """The pre-change column copy, kept verbatim: a new id index every call."""

    table = frame.table(entity)
    id_column = frame.schema.entity_id_column(entity)
    return pd.Series(
        table[column].array.copy(),
        index=pd.Index(table[id_column].to_numpy(copy=True), name=id_column),
        name=column,
        dtype=table[column].dtype,
    )


def _reference_version_series(
    frame: Frame,
) -> list[tuple[tuple[str, str], tuple[pd.Series, str]]]:
    """The pre-change structural branch of ``_write_node``, verbatim but for names.

    It copies every column, each with its own id index, and takes each copy's
    dtype token before the first is written. The write then walks the copies
    in sorted order. Only this step changed, so the reference stands in for
    ``_version_series`` alone and the rest of the write is shared. It returns
    the sorted list itself, so the write holds every copy until it returns,
    as the old dict did.
    """

    columns: dict[tuple[str, str], tuple[pd.Series, str]] = {}
    for entity in frame.entities:
        for column in frame.table(entity).columns:
            series = _reference_series_for_column(frame, entity, column)
            columns[(entity, column)] = (series, _dtype_token(series))
    return sorted(columns.items())


@contextmanager
def _column_copies(*, reference: bool) -> Iterator[None]:
    """Have ``_write_node`` take the reference's column copies when asked."""

    with pytest.MonkeyPatch.context() as patch:
        if reference:
            patch.setattr(graph_executor, "_version_series", _reference_version_series)
        yield


@contextmanager
def _without_collection() -> Iterator[None]:
    """No garbage collection runs, so an object is dead once nothing holds it."""

    collecting = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if collecting:
            gc.enable()


def _object_bytes(store: ContentStore) -> dict[str, bytes]:
    return {
        path.relative_to(store.objects).as_posix(): path.read_bytes()
        for path in sorted(store.objects.rglob("*"))
        if path.is_file()
    }


def _all_coordinates(frame: Frame) -> list[tuple[str, str]]:
    """Every (entity, column) a frame carries, in its own order."""

    return [
        (entity, column)
        for entity in frame.entities
        for column in frame.table(entity).columns
    ]


def _column_entries(record: Mapping[str, object]) -> list[dict[str, str]]:
    """A node record's column entries, in its order."""

    entries = record["columns"]
    assert isinstance(entries, list)
    return entries


# --------------------------------------------------------------------------
# A toy graph with every structural delta.
# --------------------------------------------------------------------------


class _Kernel:
    def __init__(self, ref: str, capabilities: Capabilities, compute) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self.compute = compute

    def implementation_hash(self) -> str:
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        return self.compute(context)


def _source_path(root: Path) -> Path:
    root.mkdir()
    (root / "value.txt").write_text("0", encoding="utf-8")
    return root


def _survey_frame() -> Frame:
    """Columns out of sorted order, one of every graph dtype, nulls included."""

    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "zeta": np.asarray([-0.0, np.nan, np.inf, 5e-324], dtype=np.float64),
            "person_household_id": np.asarray([10, 10, 20, 20], dtype=np.int64),
            "selected": np.asarray([True, True, True, False]),
            "age": np.asarray([40, 12, 70, 35], dtype=np.int64),
            "maybe": pd.array([True, None, False, True], dtype="boolean"),
            "count": np.asarray([1, -2, 3, 0], dtype=np.int32),
            "nullable": pd.array([None, 2, -3, 4], dtype="Int64"),
            "ratio": np.asarray([0.5, 1.5, -0.25, 3.0], dtype=np.float32),
            "label": pd.array(["a", "", None, "é\U0001f600"], dtype="string"),
        }
    )
    household = pd.DataFrame(
        {
            "size": np.asarray([2, 2], dtype=np.int64),
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "region": pd.array(["north", None], dtype="string"),
        }
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b", "b"], name="stratum"),
    )


def _source(context: KernelContext) -> KernelResult:
    return KernelResult(frame=_survey_frame())


def _double(context: KernelContext) -> KernelResult:
    table = context.tables["person"]
    return KernelResult(
        columns={
            ("person", "a"): pd.Series(
                table["age"].to_numpy(dtype=np.float64) * 2.0,
                index=pd.Index(table["person_id"], name="person_id"),
                dtype="float64",
            )
        }
    )


def _keep(context: KernelContext) -> KernelResult:
    table = context.tables["person"]
    return KernelResult(
        keep=pd.Series(
            table["selected"].to_numpy(dtype=np.bool_),
            index=table["person_id"],
            dtype="bool",
        )
    )


def _reweight(context: KernelContext) -> KernelResult:
    before = context.weights["household"].values
    return KernelResult(weights=Weights(before * 2.0, WeightKind.IMPORTANCE))


def _clone(context: KernelContext) -> KernelResult:
    """Household 10 and its two people are cloned as household 30.

    The clone and its source share the source's doubled weight.
    """

    return KernelResult(
        expand={
            "person": pd.Series(
                [1, 2], index=pd.Index([5, 6], name="person_id"), dtype="int64"
            ),
            "household": pd.Series(
                [10], index=pd.Index([30], name="household_id"), dtype="int64"
            ),
        },
        columns={
            ("household", "is_clone"): pd.Series(
                [False, False, True],
                index=pd.Index([10, 20, 30], name="household_id"),
                dtype="bool",
            )
        },
        weights=Weights(np.asarray([1.0, 4.0, 1.0]), WeightKind.IMPORTANCE),
    )


def _claim(context: KernelContext) -> KernelResult:
    household = context.tables["household"]
    return KernelResult(
        columns={
            ("household", "is_clone"): pd.Series(
                household["is_clone"].array.copy(),
                index=pd.Index(household["household_id"], name="household_id"),
                dtype="bool",
            )
        }
    )


def _registry() -> KernelRegistry:
    deterministic = Capabilities(Determinism.DETERMINISTIC)
    registry = KernelRegistry()
    for ref, structural, compute in (
        ("source@1", StructuralDelta.CREATE, _source),
        ("double@1", StructuralDelta.NONE, _double),
        ("keep@1", StructuralDelta.FILTER, _keep),
        ("reweight@1", StructuralDelta.REWEIGHT, _reweight),
        ("clone@1", StructuralDelta.EXPAND, _clone),
        ("claim@1", StructuralDelta.NONE, _claim),
    ):
        capabilities = (
            deterministic
            if structural is StructuralDelta.NONE
            else Capabilities(Determinism.DETERMINISTIC, structural=structural)
        )
        registry.register(_Kernel(ref, capabilities, compute))
    return registry


def _graph() -> Graph:
    """survey -> kept -> pool -> clone, with an ordinary node on two versions.

    ``kept`` drops person 4, ``pool`` doubles the household weights and
    ``clone`` adds household 30 with two people and a new column.
    """

    return Graph(
        "toy",
        (SOURCE,),
        (
            Node(
                "survey",
                "source@1",
                sources=("survey",),
                structural=StructuralDelta.CREATE,
                outputs=(
                    Owned("person", "zeta", "float64"),
                    Owned("person", "selected", "bool"),
                    Owned("person", "age", "int64"),
                    Owned("person", "maybe", "boolean"),
                    Owned("person", "count", "int32"),
                    Owned("person", "nullable", "Int64"),
                    Owned("person", "ratio", "float32"),
                    Owned("person", "label", "string"),
                    Owned("household", "size", "int64"),
                    Owned("household", "region", "string"),
                ),
            ),
            Node(
                "a",
                "double@1",
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "a", "float64"),),
                population="survey",
            ),
            Node(
                "kept",
                "keep@1",
                structural=StructuralDelta.FILTER,
                base="survey",
                inputs=(Slice("person", ("selected",)),),
                mass="free",
            ),
            Node(
                "pool",
                "reweight@1",
                structural=StructuralDelta.REWEIGHT,
                base="kept",
                inputs=(Slice("household", ("size",)),),
                weights=WeightTransition("household", "importance", mass="free"),
                mass="free",
            ),
            Node(
                "clone",
                "clone@1",
                structural=StructuralDelta.EXPAND,
                base="pool",
                params={
                    "expand_cells": (("household", "is_clone", "bool"),),
                    "expand_weight_entity": "household",
                    "expand_weight_kind": "importance",
                },
                mass="free",
                entrants=True,
            ),
            Node(
                "claim_clone",
                "claim@1",
                outputs=(Owned("household", "is_clone", "bool"),),
                params={"materialized_expand_outputs": ("household.is_clone",)},
                population="clone",
            ),
        ),
    )


STRUCTURAL = ("survey", "kept", "pool", "clone")


def _run(
    source: Path,
    store: ContentStore,
    *,
    resume: str = "auto",
    reference: bool = False,
) -> RunManifest:
    with _column_copies(reference=reference):
        return run_graph(
            compile_graph(_graph()),
            sources={"survey": source},
            store=store,
            kernels=_registry(),
            resume=resume,  # type: ignore[arg-type]
        )


def _portable(manifest: RunManifest) -> dict[str, object]:
    """The manifest's JSON less the fields that move between two runs.

    Those are the two timestamps, the host, and the run-level fields of each
    receipt (``NodeReceipt.RUN_LEVEL_FIELDS``). None of them reaches the key.
    """

    payload = json.loads(manifest.to_json())
    for name in ("started_at", "finished_at", "host"):
        del payload[name]
    for receipt in payload["nodes"].values():
        for name in sorted(NodeReceipt.RUN_LEVEL_FIELDS):
            del receipt[name]
    return payload


@pytest.mark.parametrize(
    "resume", ("cold", "warm", "forbid"), ids=("cold", "warm", "forbid-over-warm")
)
def test_every_structural_node_stores_the_reference_objects(
    tmp_path: Path, resume: str
) -> None:
    """The live and the reference column copies, each into its own store.

    A warm run writes nothing and every node hits, so it shows the streamed
    objects load back as the reference's do, and that a hit reports what the
    cold run computed. ``resume="forbid"`` rewrites every object over the warm
    store, through the store's write-only replacement.
    """

    source = _source_path(tmp_path / "source")
    live_store = ContentStore(tmp_path / "live")
    reference_store = ContentStore(tmp_path / "reference")
    cold: RunManifest | None = None
    if resume != "cold":
        cold = _run(source, live_store)
        _run(source, reference_store, reference=True)
    policy = "forbid" if resume == "forbid" else "auto"
    live = _run(source, live_store, resume=policy)
    reference = _run(source, reference_store, resume=policy, reference=True)

    assert _object_bytes(live_store) == _object_bytes(reference_store)
    assert all(receipt.hit is (resume == "warm") for receipt in live.nodes.values())
    assert live.key == reference.key
    assert _portable(live) == _portable(reference)
    if cold is not None:
        assert _portable(live) == _portable(cold)
    for node_id, receipt in live.nodes.items():
        expected = reference.nodes[node_id]
        assert list(receipt.artifacts.items()) == list(expected.artifacts.items())
        assert receipt.frame_key == expected.frame_key
        assert receipt.weight_key == expected.weight_key
    for node_id in STRUCTURAL:
        receipt = live.nodes[node_id]
        assert receipt.frame_key is not None
        written = live_store.load_frame(receipt.frame_key)
        artifacts = list(receipt.artifacts)
        assert artifacts == sorted(_all_coordinates(written))
        # The frame lists the person first, and the record the household.
        assert written.entities[0] == "person"
        assert artifacts[0][0] == "household"


# --------------------------------------------------------------------------
# One direct write, against the reference.
# --------------------------------------------------------------------------


KEY = hashlib.sha256(b"a structural write").hexdigest()


def _every_dtype_frame(*, benunit_membership: str = "int64") -> Frame:
    """Three entities, which sort as benunit, household, person.

    The person ids are not in order, every graph dtype is present with its
    nulls, and each table lists its columns out of sorted order.
    """

    person = pd.DataFrame(
        {
            "person_id": np.asarray([30, 10, 20, 40], dtype=np.int64),
            "zeta": np.asarray([-0.0, np.nan, np.inf, 5e-324], dtype=np.float64),
            "person_household_id": np.asarray([100, 100, 200, 200], dtype=np.int64),
            "person_benunit_id": np.asarray([7, 7, 8, 9], dtype=benunit_membership),
            "flag": np.asarray([True, False, True, False]),
            "maybe": pd.array([True, None, False, None], dtype="boolean"),
            "count": np.asarray([1, -2, 3, 0], dtype=np.int32),
            "nullable": pd.array([None, 2, None, -4], dtype="Int64"),
            "ratio": np.asarray([0.5, 1.5, -0.25, 3.0], dtype=np.float32),
            "label": pd.array(
                ["a", "", None, "é\U0001f600"], dtype=pd.StringDtype("python")
            ),
            "age": np.asarray([40, 12, 70, 35], dtype=np.int64),
            "code": pd.array(["x", None, "y\0z", ""], dtype="string"),
        }
    )
    household = pd.DataFrame(
        {
            "size": np.asarray([2, 2], dtype=np.int64),
            "household_id": np.asarray([100, 200], dtype=np.int64),
            "region": pd.array(["north", None], dtype=pd.StringDtype("python")),
        }
    )
    benunit = pd.DataFrame(
        {
            "zz": np.asarray([1.0, -1.0, 0.0], dtype=np.float64),
            "benunit_id": np.asarray([7, 8, 9], dtype=np.int64),
            "aa": pd.array([None, 1, 2], dtype="Int64"),
        }
    )
    return Frame(
        {"person": person, "household": household, "benunit": benunit},
        EntitySchema(group_entities=("household", "benunit")),
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b", "b"], name="stratum"),
    )


def _create_node(frame: Frame) -> Node:
    return Node(
        "survey",
        "source@1",
        sources=("survey",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, column, _dtype_token(frame.table(entity)[column]))
            for entity in frame.entities
            for column in frame.table(entity).columns
            if column not in _structural_columns(frame, entity)
        ),
    )


def _write(
    store: ContentStore, frame: Frame, *, reference: bool = False
) -> tuple[dict[tuple[str, str], str], dict[str, object]]:
    """Write ``frame`` as a CREATE node's version, from either column copies."""

    node = _create_node(frame)
    with _column_copies(reference=reference):
        return graph_executor._write_node(
            store,
            node,
            key=KEY,
            kernel_impl_hash=hashlib.sha256(b"source@1").hexdigest(),
            capabilities=Capabilities(
                Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
            ),
            result=KernelResult(frame=frame),
            population=Population.from_frame(frame, node.id),
            receipt={"rows": frame.n("person")},
            opaque_artifacts={"diagnostic": b"created"},
            verify_existing=True,
        )


def test_a_direct_write_returns_and_stores_what_the_reference_does(
    tmp_path: Path,
) -> None:
    frame = _every_dtype_frame()
    live_store = ContentStore(tmp_path / "live")
    reference_store = ContentStore(tmp_path / "reference")

    live_artifacts, live_record = _write(live_store, frame)
    reference_artifacts, reference_record = _write(
        reference_store, frame, reference=True
    )

    assert list(live_artifacts.items()) == list(reference_artifacts.items())
    assert live_record == reference_record
    assert _object_bytes(live_store) == _object_bytes(reference_store)
    coordinates = [
        (entry["entity"], entry["column"]) for entry in _column_entries(live_record)
    ]
    assert coordinates == list(live_artifacts) == sorted(_all_coordinates(frame))
    assert frame.entities == ("person", "household", "benunit")
    assert coordinates[0] == ("benunit", "aa")
    # Each column reads back with its entity's ids, in table order.
    for (entity, column), output_key in live_artifacts.items():
        table = frame.table(entity)
        id_column = frame.schema.entity_id_column(entity)
        loaded = live_store.load_column(output_key)
        assert loaded.index.name == id_column
        assert loaded.index.tolist() == table[id_column].tolist()
        pd.testing.assert_series_equal(
            loaded, table[column], check_index=False, check_names=False
        )


def test_an_unsupported_dtype_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    """``person_benunit_id`` sorts after every benunit and household column.

    A write that took each column's dtype token only as it reached it would
    store those first.
    """

    frame = _every_dtype_frame(benunit_membership="int8")
    for name, reference in (("live", False), ("reference", True)):
        store = ContentStore(tmp_path / name)
        with pytest.raises(NodeRejected, match="unsupported dtype int8"):
            _write(store, frame, reference=reference)
        assert _object_bytes(store) == {}


# --------------------------------------------------------------------------
# What the write holds.
# --------------------------------------------------------------------------


def test_each_copy_goes_before_the_next_and_each_entity_shares_one_id_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weak references to every column copy the write makes.

    The write holds no copy it has stored once it makes the next one. Each
    entity's columns are indexed by the first id index the write made for
    that entity. That index is held here, so an index rebuilt for a later
    column could not reuse its memory and pass for it.
    """

    frame = _every_dtype_frame()
    copies: list[weakref.ref[pd.Series]] = []
    alive: list[int] = []
    first_ids: dict[str, pd.Index] = {}
    copy = graph_executor._series_for_column

    def watched(table: pd.DataFrame, column: str, ids: pd.Index) -> pd.Series:
        alive.append(sum(earlier() is not None for earlier in copies))
        series = copy(table, column, ids)
        copies.append(weakref.ref(series))
        assert ids is first_ids.setdefault(str(ids.name), ids)
        assert series.index is ids
        return series

    monkeypatch.setattr(graph_executor, "_series_for_column", watched)
    with _without_collection():
        _write(ContentStore(tmp_path / "store"), frame)
        dead = [earlier() is None for earlier in copies]

    assert len(copies) == len(_all_coordinates(frame))
    assert alive == [0] * len(copies)
    assert all(dead)
    assert set(first_ids) == {"benunit_id", "household_id", "person_id"}


def _wide_frame(*, households: int, people_each: int, columns: int) -> Frame:
    """A frame of float columns, wide enough for the write's copies to show."""

    generator = np.random.default_rng(20261107)
    people = households * people_each
    person = pd.DataFrame(
        {
            "person_id": np.arange(people, dtype=np.int64),
            "person_household_id": np.repeat(
                np.arange(households, dtype=np.int64), people_each
            ),
            **{
                f"x{index:02d}": generator.normal(size=people)
                for index in range(columns)
            },
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.arange(households, dtype=np.int64),
            **{
                f"h{index:02d}": generator.normal(size=households)
                for index in range(columns)
            },
        }
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(households), WeightKind.DESIGN)},
    )


def _column_bytes(frame: Frame) -> int:
    return sum(
        frame.table(entity)[column].to_numpy().nbytes
        for entity, column in _all_coordinates(frame)
    )


def _traced_peak(action: Callable[[], object]) -> int:
    """Bytes traced at the action's peak, above what was held when it began."""

    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        action()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        if not tracing:
            tracemalloc.stop()


def test_a_structural_write_holds_one_column_at_a_time_not_a_second_frame(
    tmp_path: Path,
) -> None:
    """Small sizes: 20,000 people and 5,000 households, 48 float columns each.

    The reference holds a copy of every column and one id array per column,
    about twice the frame's column bytes. The streamed write holds one column
    copy and one id array at a time, beside what the frame write itself
    needs.
    """

    frame = _wide_frame(households=5_000, people_each=4, columns=48)
    column_bytes = _column_bytes(frame)

    reference = _traced_peak(
        lambda: _write(ContentStore(tmp_path / "reference"), frame, reference=True)
    )
    live = _traced_peak(lambda: _write(ContentStore(tmp_path / "live"), frame))

    assert reference > 2 * column_bytes
    assert reference - live > column_bytes
    assert live < column_bytes / 2


# --------------------------------------------------------------------------
# What a cache hit holds.
# --------------------------------------------------------------------------


def test_a_structural_hit_loads_every_column_in_order_and_keeps_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weak references to every column a structural hit loads.

    The hit loads each column its record lists, in the record's order, and
    then the frame. It holds no column once it loads the next one, or the
    frame, and it returns only the frame and the record's artifacts.
    """

    frame = _every_dtype_frame()
    store = ContentStore(tmp_path / "store")
    artifacts, record = _write(store, frame)
    loads: list[str] = []
    loaded: list[weakref.ref[pd.Series]] = []
    alive: list[int] = []
    load_column = store.load_column
    load_frame = store.load_frame

    def watched_column(key: str) -> pd.Series:
        alive.append(sum(earlier() is not None for earlier in loaded))
        loads.append(key)
        series = load_column(key)
        loaded.append(weakref.ref(series))
        return series

    def watched_frame(key: str) -> Frame:
        alive.append(sum(earlier() is not None for earlier in loaded))
        loads.append(key)
        return load_frame(key)

    monkeypatch.setattr(store, "load_column", watched_column)
    monkeypatch.setattr(store, "load_frame", watched_frame)
    with _without_collection():
        result, hit_artifacts = graph_executor._load_cached_result(
            store, _create_node(frame), None, record
        )
        dead = [earlier() is None for earlier in loaded]

    entries = _column_entries(record)
    assert loads == [*(entry["key"] for entry in entries), record["frame_key"]]
    assert len(loaded) == len(_all_coordinates(frame))
    assert alive == [0] * len(loads)
    assert all(dead)
    assert dict(result.columns) == {}
    assert list(hit_artifacts.items()) == list(artifacts.items())
    assert result.frame is not None
    for entity in frame.entities:
        pd.testing.assert_frame_equal(result.frame.table(entity), frame.table(entity))


def test_a_structural_hit_still_refuses_a_damaged_or_repeated_column(
    tmp_path: Path,
) -> None:
    """The hit keeps no structural column, but it still checks every one.

    The frame is stored whole and stays intact, so only the column's own
    check can refuse the hit.
    """

    frame = _every_dtype_frame()
    node = _create_node(frame)
    store = ContentStore(tmp_path / "store")
    artifacts, record = _write(store, frame)
    entries = _column_entries(record)

    repeated = {**record, "columns": [*entries, entries[0]]}
    with pytest.raises(StoreCorrupt, match=r"repeats column benunit\.aa"):
        graph_executor._load_cached_result(store, node, None, repeated)

    payload = store.object_path(artifacts[("person", "zeta")]) / "values.npy"
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    failed = f"{payload.parent.name}: payload 'values.npy' failed"
    with pytest.raises(StoreCorrupt, match=re.escape(failed)):
        graph_executor._load_cached_result(store, node, None, record)
    # The frame still loads, so the refusal is the damaged column's.
    store.load_frame(str(record["frame_key"]))


def test_a_structural_hit_holds_no_column_beside_its_frame(tmp_path: Path) -> None:
    """Small sizes: the write test's frame, 10 MB of float columns.

    A hit that kept each column it loaded, each with its own ids, would hold
    about twice the frame's column bytes while it loads the frame. This one's
    peak is its frame load's, give or take one column.
    """

    frame = _wide_frame(households=5_000, people_each=4, columns=48)
    column_bytes = _column_bytes(frame)
    store = ContentStore(tmp_path / "store")
    _, record = _write(store, frame)
    node = _create_node(frame)

    frame_load = _traced_peak(lambda: store.load_frame(str(record["frame_key"])))
    hit = _traced_peak(
        lambda: graph_executor._load_cached_result(store, node, None, record)
    )

    assert hit - frame_load < column_bytes / 2
