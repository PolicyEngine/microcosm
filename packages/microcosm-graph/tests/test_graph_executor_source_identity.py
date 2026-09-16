"""The executor's source-identity contract, and the per-run cache under it.

Until this file the per-node source re-derivation at the heart of
``run_graph`` had no regression test at all: nothing asserted that a source
mutated while a node ran makes the run refuse, and nothing asserted that the
refusal lands before the node's artifacts reach the store. Both are asserted
here, first, so the cache added underneath them is measured against a pinned
contract rather than against silence.

The cache reuses a source's content key only while the path's stat signature
is unchanged, so every mutation that moves any stat field -- content, size,
inode, timestamps, or a directory's roster -- still refuses at the same node
it refuses at today. The two cases a stat signature cannot decide are closed
by an unconditional full re-derivation before the manifest is built, which
also gives the executor something it never had: a source changed during a node
that declares no source is now caught.

Synthetic toy graph only: no data file, no country package, no engine.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
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
    compile_graph,
)
from microcosm.graph.executor import NodeRejected, run_graph
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
)
from microcosm.graph.store import ContentStore

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    assert _SPEC is not None and _SPEC.loader is not None
    _TOY = importlib.util.module_from_spec(_SPEC)
    sys.modules["_toy"] = _TOY
    _SPEC.loader.exec_module(_TOY)

SOURCE = SourceRef("survey", "csv-tables")


class _Kernel:
    def __init__(self, ref, capabilities, run):
        self._ref = ref
        self._capabilities = capabilities
        self._run = run

    @property
    def ref(self):
        return self._ref

    @property
    def capabilities(self):
        return self._capabilities

    def implementation_hash(self):
        return f"impl-{self._ref}"

    def run(self, context):
        return self._run(context)


def _source_path(root: Path, value: int = 0) -> Path:
    root.mkdir()
    (root / "value.txt").write_text(str(value), encoding="utf-8")
    return root


def _source_frame(path: Path) -> Frame:
    offset = int((path / "value.txt").read_text(encoding="utf-8"))
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "age": np.asarray([10, 20, 30], dtype=np.int64) + offset,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "size": np.asarray([2, 1], dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b"], name="stratum"),
    )


CREATE = Node(
    "survey",
    "source@1",
    structural=StructuralDelta.CREATE,
    sources=("survey",),
    outputs=(Owned("person", "age", "int64"), Owned("household", "size", "int64")),
)
FOLLOWER = Node(
    "follower",
    "follower@1",
    inputs=(Slice("person", ("age",)),),
    outputs=(Owned("person", "doubled", "float64"),),
    population="survey",
)


def _graph() -> Graph:
    return Graph("toy", (SOURCE,), (CREATE, FOLLOWER))


def _registry(
    *,
    on_source: Callable[[Path], None] | None = None,
    on_follower: Callable[[], None] | None = None,
) -> KernelRegistry:
    def source(context: KernelContext) -> KernelResult:
        frame = _source_frame(context.sources["survey"])
        if on_source is not None:
            on_source(context.sources["survey"])
        return KernelResult(frame=frame, receipt={"rows": frame.n("person")})

    def follower(context: KernelContext) -> KernelResult:
        if on_follower is not None:
            on_follower()
        table = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "doubled"): pd.Series(
                    table["age"].to_numpy(dtype=np.float64) * 2.0,
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            },
            receipt={"rows": len(table)},
        )

    registry = KernelRegistry()
    registry.register(
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            source,
        )
    )
    registry.register(
        _Kernel("follower@1", Capabilities(Determinism.DETERMINISTIC), follower)
    )
    return registry


def _run(source: Path, store: ContentStore, registry: KernelRegistry):
    return run_graph(
        compile_graph(_graph()),
        sources={"survey": source},
        store=store,
        kernels=registry,
    )


def _stored_objects(store: ContentStore) -> set[str]:
    return {
        path.relative_to(store.objects).as_posix()
        for path in store.objects.rglob("*")
        if path.is_file()
    }


# --------------------------------------------------------------------------
# The contract: a source that moves while the run executes makes it refuse.
# --------------------------------------------------------------------------


def test_a_source_rewritten_by_its_own_node_refuses_that_node(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    before = _stored_objects(store)

    def rewrite(path: Path) -> None:
        (path / "value.txt").write_text("7", encoding="utf-8")

    with pytest.raises(NodeRejected, match="changed source"):
        _run(source, store, _registry(on_source=rewrite))
    # The refusal precedes the node's own persistence: nothing new was written.
    assert _stored_objects(store) == before


def test_a_file_added_to_a_directory_source_refuses(tmp_path: Path) -> None:
    """The roster is part of the signature, so an addition is never cached over."""

    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")

    def add(path: Path) -> None:
        (path / "extra.txt").write_text("invented", encoding="utf-8")

    with pytest.raises(NodeRejected, match="changed source"):
        _run(source, store, _registry(on_source=add))


def test_a_file_removed_from_a_directory_source_refuses(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")
    (source / "extra.txt").write_text("invented", encoding="utf-8")
    store = ContentStore(tmp_path / "store")

    def remove(path: Path) -> None:
        (path / "extra.txt").unlink()

    with pytest.raises(NodeRejected, match="changed source"):
        _run(source, store, _registry(on_source=remove))


def test_a_source_changed_during_a_source_free_node_refuses_at_run_end(
    tmp_path: Path,
) -> None:
    """The guarantee the per-node check never had: no node declares it here."""

    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")

    def rewrite() -> None:
        (source / "value.txt").write_text("9", encoding="utf-8")

    with pytest.raises(NodeRejected, match="Run changed source 'survey'"):
        _run(source, store, _registry(on_follower=rewrite))


def test_a_rewritten_source_still_refuses_when_only_its_bytes_moved(
    tmp_path: Path,
) -> None:
    """Same length and modification time: st_ctime_ns is what still moves.

    This is the strongest rewrite an unprivileged process can perform here, and
    it is *not* stat-preserving: st_ctime_ns moves, so the signature moves and
    the node refuses. The genuinely stat-preserving case -- all five fields
    identical -- is the residual the run-end re-derivation exists for, and
    `test_a_source_changed_during_a_source_free_node_refuses_at_run_end`
    reaches that re-derivation by the other route.
    """

    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    target = source / "value.txt"
    target.write_text("1", encoding="utf-8")
    original = target.stat()

    def rewrite(path: Path) -> None:
        (path / "value.txt").write_text("2", encoding="utf-8")
        os.utime(path / "value.txt", ns=(original.st_atime_ns, original.st_mtime_ns))

    with pytest.raises(NodeRejected, match="source"):
        _run(source, store, _registry(on_source=rewrite))


# --------------------------------------------------------------------------
# The cache: fewer reads, identical keys.
# --------------------------------------------------------------------------


def _counting(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    original = graph_executor.source_content_key

    def counted(name, path):
        seen.append((name, str(path)))
        return original(name, path)

    monkeypatch.setattr(graph_executor, "source_content_key", counted)
    return seen


def test_each_source_is_read_twice_per_run_not_once_per_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    seen = _counting(monkeypatch)
    manifest = _run(source, store, _registry())
    # Once when the run's identities are established, once when they are
    # re-proved at the end. The node in between reuses the first read.
    assert [name for name, _ in seen] == ["survey", "survey"]
    assert manifest.nodes["survey"].hit is False


def test_an_unchanged_source_is_never_re_read_by_a_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wider graph adds nodes, not reads."""

    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    wide = Graph(
        "toy",
        (SOURCE,),
        (
            CREATE,
            FOLLOWER,
            Node(
                "second",
                "source@1",
                structural=StructuralDelta.CREATE,
                sources=("survey",),
                outputs=(
                    Owned("person", "age", "int64"),
                    Owned("household", "size", "int64"),
                ),
            ),
        ),
    )
    seen = _counting(monkeypatch)
    run_graph(
        compile_graph(wide),
        sources={"survey": source},
        store=store,
        kernels=_registry(),
    )
    assert len(seen) == 2


def test_a_touched_but_unchanged_source_re_reads_and_still_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signature miss is a re-read, not a refusal."""

    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    seen = _counting(monkeypatch)

    def touch(path: Path) -> None:
        os.utime(path / "value.txt", None)

    manifest = _run(source, store, _registry(on_source=touch))
    assert len(seen) == 3  # run start, the node's miss, run end
    assert manifest.nodes["survey"].hit is False


# --------------------------------------------------------------------------
# The record, and what it must not move.
# --------------------------------------------------------------------------


def test_the_manifest_records_the_run_end_identities_without_moving_anything(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    manifest = _run(source, store, _registry())
    identity = graph_executor.source_content_key("survey", source)
    assert dict(manifest.source_identities) == {"survey": identity}

    bare = type(manifest)(
        country=manifest.country,
        nodes=manifest.nodes,
        decisions=manifest.decisions,
        started_at=manifest.started_at,
        finished_at=manifest.finished_at,
        host=manifest.host,
    )
    assert bare.key == manifest.key
    assert bare.to_json() == manifest.to_json()
    assert "source_identities" not in manifest.to_json()
    assert dict(manifest.content_addressed) == dict(bare.content_addressed)


def test_the_recorded_identities_are_rejected_when_malformed() -> None:
    from microcosm.graph.manifest import RunManifest

    with pytest.raises(TypeError, match="source_identities keys"):
        RunManifest(country="toy", nodes={}, source_identities={1: "a"})
    with pytest.raises(TypeError, match="source_identities values"):
        RunManifest(country="toy", nodes={}, source_identities={"a": 1})


# --------------------------------------------------------------------------
# The cache object itself.
# --------------------------------------------------------------------------


def test_the_cache_reuses_only_an_unchanged_signature(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")
    identities = graph_executor._SourceIdentities()
    first = identities.key("survey", source)
    assert identities.key("survey", source) == first
    (source / "value.txt").write_text("5", encoding="utf-8")
    second = identities.key("survey", source)
    assert second != first
    assert identities.key("survey", source) == second


def test_the_cache_is_bypassed_by_derive(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")
    identities = graph_executor._SourceIdentities()
    identities.key("survey", source)
    calls: list[str] = []
    original = graph_executor.source_content_key

    def counted(name, path):
        calls.append(name)
        return original(name, path)

    graph_executor.source_content_key = counted
    try:
        identities.key("survey", source)
        assert calls == []
        identities.derive("survey", source)
        assert calls == ["survey"]
    finally:
        graph_executor.source_content_key = original


def test_a_regular_file_source_signature_moves_with_its_bytes(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_bytes(b"id\n1\n")
    before = graph_executor._source_stat_signature(path)
    assert before[0] == "file"
    path.write_bytes(b"id\n2\n")
    assert graph_executor._source_stat_signature(path) != before


def test_a_directory_signature_covers_names_as_well_as_contents(
    tmp_path: Path,
) -> None:
    root = _source_path(tmp_path / "source")
    before = graph_executor._source_stat_signature(root)
    assert before[0] == "dir"
    (root / "value.txt").rename(root / "renamed.txt")
    renamed = graph_executor._source_stat_signature(root)
    assert renamed != before
    nested = root / "inner"
    nested.mkdir()
    (nested / "deep.txt").write_text("x", encoding="utf-8")
    assert graph_executor._source_stat_signature(root) != renamed


def test_a_source_that_moves_during_its_own_read_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn read leaves no cache entry, so the next consumer re-derives."""

    path = tmp_path / "table.csv"
    path.write_bytes(b"id\n1\n")
    identities = graph_executor._SourceIdentities()
    original = graph_executor.source_content_key

    def tearing(name, source_path):
        result = original(name, source_path)
        Path(source_path).write_bytes(b"id\n22\n")
        return result

    monkeypatch.setattr(graph_executor, "source_content_key", tearing)
    identities.key("table", path)
    monkeypatch.setattr(graph_executor, "source_content_key", original)
    calls: list[str] = []

    def counted(name, source_path):
        calls.append(name)
        return original(name, source_path)

    monkeypatch.setattr(graph_executor, "source_content_key", counted)
    identities.key("table", path)
    assert calls == ["table"]


# --------------------------------------------------------------------------
# A member symlink: the bytes are the target's, so the signature follows it.
# --------------------------------------------------------------------------


def test_a_symlinked_member_refuses_at_the_node_that_changed_its_target(
    tmp_path: Path,
) -> None:
    """The link's own five stat fields do not move when its target's bytes do.

    ``keys._directory_identity`` selects members with ``is_file()`` and reads
    them with ``read_bytes()``, both of which follow the link, so a member
    link's target is inside the source's content key. If the signature stopped
    at the link, a target rewritten mid-run would be answered from cache at the
    per-node check and refused only by the run-end re-derivation -- after
    intervening nodes had written store records. It refuses at the node.
    """

    source = _source_path(tmp_path / "source")
    target = tmp_path / "outside" / "linked.txt"
    target.parent.mkdir()
    target.write_text("one", encoding="utf-8")
    (source / "linked.txt").symlink_to(target)
    store = ContentStore(tmp_path / "store")
    before = _stored_objects(store)

    def rewrite(_path: Path) -> None:
        target.write_text("another", encoding="utf-8")

    with pytest.raises(NodeRejected, match="Node 'survey' changed source"):
        _run(source, store, _registry(on_source=rewrite))
    # The refusal still precedes the node's own persistence.
    assert _stored_objects(store) == before


def test_a_directory_signature_follows_a_member_symlink(tmp_path: Path) -> None:
    root = _source_path(tmp_path / "source")
    target = tmp_path / "outside.txt"
    target.write_text("one", encoding="utf-8")
    (root / "linked.txt").symlink_to(target)
    before = graph_executor._source_stat_signature(root)
    target.write_text("another", encoding="utf-8")
    assert graph_executor._source_stat_signature(root) != before


def test_a_broken_member_symlink_signs_as_absent_without_raising(
    tmp_path: Path,
) -> None:
    """``_directory_identity`` skips it; the signature records that it is gone."""

    root = _source_path(tmp_path / "source")
    (root / "dangling.txt").symlink_to(tmp_path / "never-written.txt")
    signature = graph_executor._source_stat_signature(root)
    assert any(member[-1][0] == "absent" for member in signature[2] if len(member) == 4)
    (tmp_path / "never-written.txt").write_text("now here", encoding="utf-8")
    assert graph_executor._source_stat_signature(root) != signature
