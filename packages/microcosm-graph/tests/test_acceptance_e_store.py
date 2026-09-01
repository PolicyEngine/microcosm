"""Charter group E: store and resume.

Finding 3 of the architecture review: ``except Exception`` at checkpoint
discovery, and again in the durable loader, collapses ``ImportError``, a
missing ``h5py``, and a genuinely stale identity into one outcome — rebuild.
On 8/31 a worktree resync without the engine extras produced exactly that: the
adapter raised ``ImportError``, the builder printed "Ignored stacked checkpoint
discovery manifest", and a cold rebuild began. Nothing was written down.

These five properties make each of those a distinct, named, testable outcome:
``StoreCorruptError`` for altered bytes, ``StoreUnavailableError`` for a verifier that is
not installed, ``StoreMissError`` under ``resume="require"``, no partial artifact
after an interruption, and a manifest that lists every node either way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from microcosm.graph import ContentStore, Node, Owned, Slice

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]


def _files_under(root: Path) -> dict[str, int]:
    """Every regular file below ``root``, by relative path and size."""
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _file_holding(root: Path, key: str) -> Path:
    """The stored file whose name carries ``key``.

    A content-addressed store names an artifact after its key; this locates
    the bytes E1 then alters. If the runtime lane names artifacts some other
    way, this assertion is where that shows up.
    """
    matches = [path for path in root.rglob("*") if path.is_file() and key in path.name]
    assert matches, f"no stored file carries the artifact key {key}"
    return matches[0]


@pytest.mark.xfail(strict=True, reason="charter E1: pending")
def test_e1_content_validation_on_load(tmp_path: Path) -> None:
    """Altered artifact bytes are refused on load and never used."""
    first = toy.run_toy(toy.small_graph(), tmp_path / "run")
    root = tmp_path / "run" / "store"
    key = first.manifest.nodes["target_a"].artifacts[("person", "target_a")]

    path = _file_holding(root, key)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))

    from microcosm.graph import StoreCorruptError

    with pytest.raises(StoreCorruptError, match=key):
        toy.run_toy(
            toy.small_graph(),
            tmp_path / "run",
            sources=first.sources,
            store=ContentStore(root),
        )


@pytest.mark.xfail(strict=True, reason="charter E2: pending")
def test_e2_verifier_unavailability_is_fatal(tmp_path: Path) -> None:
    """A missing codec stops the run before any kernel executes.

    The 8/31 replay: the thing that could read the artifact was not installed.
    That is not a stale checkpoint and must not become a cold rebuild.
    """
    registry = toy.toy_registry()
    from microcosm.graph import StoreUnavailableError

    with pytest.raises(StoreUnavailableError, match="csv-tables"):
        toy.run_toy(
            toy.small_graph(),
            tmp_path / "run",
            registry=registry,
            store=ContentStore(tmp_path / "run" / "store", codecs={}),
        )
    assert toy.total_calls(registry) == 0


@pytest.mark.xfail(strict=True, reason="charter E3: pending")
def test_e3_resume_policy_is_real(tmp_path: Path) -> None:
    """``require``, ``forbid``, and ``auto`` behave differently on one graph."""
    from microcosm.graph import StoreMissError

    cold = toy.toy_registry()
    with pytest.raises(StoreMissError):
        toy.run_toy(
            toy.small_graph(),
            tmp_path / "run",
            registry=cold,
            resume="require",
        )
    assert toy.total_calls(cold) == 0

    first = toy.run_toy(toy.small_graph(), tmp_path / "run")
    warm = toy.toy_registry()
    required = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources=first.sources,
        registry=warm,
        store=ContentStore(tmp_path / "run" / "store"),
        resume="require",
    )
    assert required.misses() == set()
    assert toy.total_calls(warm) == 0

    forbidding = toy.toy_registry()
    forbidden = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources=first.sources,
        registry=forbidding,
        store=ContentStore(tmp_path / "run" / "store"),
        resume="forbid",
    )
    assert forbidden.misses() == set(forbidden.compiled.order)
    assert toy.total_calls(forbidding) == len(forbidden.compiled.order)
    assert forbidden.all_bytes() == first.all_bytes()

    automatic = toy.toy_registry()
    memoized = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources=first.sources,
        registry=automatic,
        store=ContentStore(tmp_path / "run" / "store"),
        resume="auto",
    )
    assert memoized.misses() == set()
    assert toy.total_calls(automatic) == 0


@pytest.mark.xfail(strict=True, reason="charter E4: pending")
def test_e4_atomic_writes(tmp_path: Path) -> None:
    """An interrupted run leaves no partial artifact, and the node stays a miss.

    A run that dies part-way is reproduced by a kernel that raises after its
    predecessors are already in the store. The store's files are then compared
    against the snapshot taken before that run: a partial artifact or a
    leftover temporary shows up as an extra file, a truncated one as a
    different size, and neither is permitted.
    """
    prefix = toy.small_graph(
        nodes=(toy.CREATE, toy.derive("resources", ("age",), "resources"))
    )
    clean = toy.run_toy(prefix, tmp_path / "run")
    root = tmp_path / "run" / "store"
    before = _files_under(root)

    explodes = Node(
        "explodes",
        "bad.raise@1",
        inputs=(Slice("person", ("resources",)),),
        outputs=(Owned("person", "never_written", "float64"),),
        params={"target": "never_written"},
        population="survey",
    )
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="explodes"):
        toy.run_toy(
            toy.small_graph(nodes=(*prefix.nodes, explodes)),
            tmp_path / "run",
            sources=clean.sources,
            store=ContentStore(root),
        )
    assert _files_under(root) == before

    registry = toy.toy_registry()
    resumed = toy.run_toy(
        toy.small_graph(
            nodes=(
                *prefix.nodes,
                toy.count_node("explodes", ("resources",), "never_written"),
            )
        ),
        tmp_path / "run",
        sources=clean.sources,
        registry=registry,
        store=ContentStore(root),
    )
    assert resumed.misses() == {"explodes"}
    assert toy.calls_by_ref(registry)["count.calls@1"] == 1


@pytest.mark.xfail(strict=True, reason="charter E5: pending")
def test_e5_manifest_completeness(tmp_path: Path) -> None:
    """Every node, hit or miss, with its key, seed, receipt, and artifacts.

    Two runs of the same graph agree on everything except the run-level facts:
    which nodes were read from the store rather than executed.
    """
    first = toy.run_toy(toy.full_graph(), tmp_path / "run")
    assert set(first.manifest.nodes) == set(first.compiled.order)
    for node_id in first.compiled.order:
        record = first.manifest.nodes[node_id]
        assert isinstance(record.key, str) and record.key
        assert isinstance(record.seed, int)
        assert isinstance(record.hit, bool)
        assert record.receipt is not None
        assert record.receipt["capabilities"]["determinism"]
        assert record.artifacts is not None

    second = toy.run_toy(
        toy.full_graph(),
        tmp_path / "run",
        sources=first.sources,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert first.misses() == set(first.compiled.order)
    assert second.misses() == set()
    assert second.keys() == first.keys()
    assert second.seeds() == first.seeds()
    assert {
        node_id: dict(second.manifest.nodes[node_id].artifacts)
        for node_id in second.compiled.order
    } == {
        node_id: dict(first.manifest.nodes[node_id].artifacts)
        for node_id in first.compiled.order
    }
