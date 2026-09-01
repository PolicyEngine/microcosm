"""The four incident replays.

Candidate-26 is a complete trace of the pattern the node graph exists to
break, so the charter makes it the acceptance fixture for the properties that
matter most. Each test below reconstructs one incident's *shape* on the toy
country — not its data, not its scale, and not its code paths, which are gone
in this design; the shape is what has to become impossible.

1. **WIC dtype breach (B3).** On 8/30 a reconciliation callback wrote a dense
   ``bool`` column over a nullable ``boolean`` ASEC producer column. The guard
   fired, correctly, because ``.equals`` is dtype-sensitive and donor cells are
   meant to stay byte-identical. Here the executor rejects the node and no
   downstream node runs at all.
2. **``0347a009`` repack (C1 + C2).** Five person targets were removed from the
   late PUF-transfer surface; the max-eight packer then moved 31 of the 32
   survivors to new batch positions, so they conditioned on different
   predecessor prefixes and drew from different RNG positions, and eight
   regressed in the by-origin battery. Here five leaf nodes are removed and
   every surviving key, seed, and artifact is unchanged.
3. **Engine-less environment (E2).** On 8/31 a worktree resync ran ``uv sync``
   without the engine extras; checkpoint discovery asked the adapter for a
   manifest, caught the ``ImportError`` as ``except Exception``, and began a
   cold rebuild that the operator killed after four minutes. Here the verifier
   is unavailable and the run stops before any kernel executes.
4. **Evidence flip (F3).** Changing one field in a serialized release manifest
   makes an evidence release pass the certified validator. Here the manifest is
   keyed by its content and the loader refuses it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from microcosm.graph import (
    ContentStore,
    Node,
    Owned,
    RunManifest,
    Slice,
    StructuralDelta,
    compile_graph,
)

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: The five person targets ``0347a009`` removed, as toy leaf names.
REMOVED = ("a", "b", "c", "d", "e")


@pytest.mark.xfail(strict=True, reason="charter B3: pending")
def test_replay_wic_dtype_breach_stops_at_the_node_boundary(tmp_path: Path) -> None:
    """A dense ``bool`` over a nullable ``boolean`` incumbent is rejected here.

    The breach and its blast radius are both asserted: the node fails, and the
    node that reads what it would have written never executes. In the incident
    the guard fired after the write, and the repair cost a ten-hour rebuild
    including 110 target refits that had nothing to do with WIC.
    """
    breach = toy.patch_node(
        "wic_recode",
        "receives_x",
        "boolean",
        True,
        population="pool",
        kernel="bad.dense_bool@1",
    )
    downstream = toy.count_node(
        "consumes_the_recode", ("receives_x",), "downstream", population="pool"
    )
    graph = toy.small_graph(nodes=(toy.CREATE, toy.POOL, breach, downstream))
    assert "wic_recode" in toy.descendants(compile_graph(graph), "pool"), (
        "the replay only means something if the breach is upstream of a reader"
    )

    registry = toy.toy_registry()
    from microcosm.graph import NodeRejected

    with pytest.raises(NodeRejected, match="receives_x"):
        toy.run_toy(graph, tmp_path / "breach", registry=registry)

    assert toy.calls_by_ref(registry)["count.calls@1"] == 0
    assert toy.calls_by_ref(registry)["bad.dense_bool@1"] == 1


@pytest.mark.xfail(strict=True, reason="charter C2: pending")
def test_replay_0347a009_repack_moves_no_survivor(tmp_path: Path) -> None:
    """Five targets removed, zero survivors re-modelled.

    The incident's counterfactual, asserted exactly: of the survivors, not one
    key, not one seed, and not one artifact byte moves, and the second run
    executes no kernel at all. In the incident 31 of 32 survivors moved; only
    ``tax_exempt_interest_income``, at batch 1 position 0, kept its slot.
    """
    before = toy.run_toy(toy.chained_graph(REMOVED), tmp_path / "run")
    removed = {f"leaf_{name}" for name in REMOVED}
    survivors = set(before.compiled.order) - removed
    assert len(removed) == 5

    registry = toy.toy_registry()
    after = toy.run_toy(
        toy.drop_nodes(toy.chained_graph(REMOVED), *sorted(removed)),
        tmp_path / "run",
        sources=before.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert set(after.compiled.order) == survivors
    assert after.keys() == {k: v for k, v in before.keys().items() if k in survivors}
    assert after.seeds() == {k: v for k, v in before.seeds().items() if k in survivors}
    assert after.all_bytes() == {
        k: v for k, v in before.all_bytes().items() if k in survivors
    }
    assert after.misses() == set()
    assert toy.total_calls(registry) == 0


@pytest.mark.xfail(strict=True, reason="charter E2: pending")
def test_replay_engine_less_environment_stops_before_any_kernel(
    tmp_path: Path,
) -> None:
    """An unavailable verifier is not a stale checkpoint and not a rebuild.

    The counting kernels prove the distinction is real: the run raises
    ``StoreUnavailable`` and executes nothing. The incident's four wasted
    minutes came from ``except Exception`` collapsing ``ImportError`` into
    "rebuild"; the executor cannot make that mistake if it never starts.
    """
    registry = toy.toy_registry()
    from microcosm.graph import StoreUnavailable

    with pytest.raises(StoreUnavailable, match="csv-tables"):
        toy.run_toy(
            toy.full_graph(),
            tmp_path / "run",
            registry=registry,
            store=ContentStore(tmp_path / "run" / "store", codecs={}),
        )
    assert toy.total_calls(registry) == 0
    assert not any((tmp_path / "run" / "store").rglob("*.*"))


@pytest.mark.xfail(strict=True, reason="charter F3: pending")
def test_replay_evidence_flip_is_refused_by_the_loader(tmp_path: Path) -> None:
    """The one-field edit that made an evidence release load as certified.

    The review's reproduction: leave the failed gate, the evidence tier, and
    the non-empty known failures in place, change one field, and the certified
    validator accepts it. Here the manifest is keyed by its content, so the
    edit changes the key and the loader refuses the file.
    """
    red = toy.run_toy(
        toy.full_graph(gate_low=1e11, gate_high=1e12), tmp_path / "release"
    )
    path = tmp_path / "release" / "manifest.json"
    red.manifest.save(path)

    document = json.loads(path.read_text())
    assert document["tier"] == "evidence"
    assert document["known_failures"]
    document["tier"] = "certified"
    document["schema_version"] = 1
    path.write_text(json.dumps(document))

    from microcosm.graph import NodeRejected, StoreCorrupt

    with pytest.raises(StoreCorrupt, match=red.manifest.key):
        RunManifest.load(path, store=red.store)
    with pytest.raises((StoreCorrupt, NodeRejected)):
        RunManifest.load_certified(path, store=red.store)


def test_every_replay_the_charter_names_is_here() -> None:
    """Green from the first commit: four replays, named after four incidents.

    ``docs/graph-acceptance.md`` lists them under "The four incident replays".
    If one is deleted or renamed away, this fails rather than the burndown
    quietly improving.
    """
    text = Path(__file__).read_text()
    replays = [
        line for line in text.splitlines() if line.startswith("def test_replay_")
    ]
    assert len(replays) == 4
    charter = (Path(__file__).parents[3] / "docs" / "graph-acceptance.md").read_text()
    assert "## The four incident replays" in charter
    for property_id in ("B3", "C1 + C2", "E2", "F3"):
        assert property_id in charter.split("## The four incident replays", 1)[1]


def test_the_replay_kernels_really_do_misbehave() -> None:
    """Green from the first commit: the toy country's bad kernels are bad.

    A replay that used a well-behaved kernel by accident would go green for
    the wrong reason once the executor lands, so the misbehaviour is asserted
    here directly rather than inferred from a rejection.
    """
    registry = toy.toy_registry()
    dense = registry.get("bad.dense_bool@1")
    assert dense.capabilities.structural is StructuralDelta.NONE
    assert type(dense).__name__ == "ReturnsDenseBool"

    for ref, expected in (
        ("bad.outside@1", "WritesOutsideOwnership"),
        ("bad.mutate@1", "MutatesItsInput"),
        ("bad.absent@1", "WritesIntoAbsentCell"),
        ("bad.raise@1", "Raises"),
    ):
        assert type(registry.get(ref)).__name__ == expected

    assert Node("probe", "k@1", inputs=(Slice("person", ("age",)),)).kernel == "k@1"
    assert Owned("person", "x", "boolean").dtype == "boolean"
