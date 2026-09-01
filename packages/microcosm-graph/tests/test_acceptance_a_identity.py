"""Charter group A: identity and reuse.

One executable test per property of ``docs/graph-acceptance.md`` section A,
asserting that property's exact statement against the public API of
``microcosm.graph`` and the toy country in ``_toy.py``. Every test is
committed red under ``xfail(strict=True)``; the pull request that implements
the property deletes its marker.

The compile-time halves of these properties are already covered by
``test_graph_decl.py`` and are not repeated here. What is asserted here is
what only a running executor can show: node keys, store hits, seeds, and
artifact bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.graph import ContentStore, Graph, SourceRef

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: A whole run reduced to what A1 compares across processes: node keys, and a
#: digest of every artifact's value bytes and null mask.
_SUBPROCESS_RUNNER = """
import hashlib, importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("_toy", sys.argv[1])
toy = importlib.util.module_from_spec(spec)
sys.modules["_toy"] = toy
spec.loader.exec_module(toy)

run = toy.run_toy(toy.small_graph(), Path(sys.argv[2]))
digest = {}
for node_id, cells in run.all_bytes().items():
    accumulator = hashlib.sha256()
    for cell, (values, mask) in sorted(cells.items()):
        accumulator.update(repr(cell).encode())
        accumulator.update(values)
        accumulator.update(mask)
    digest[node_id] = accumulator.hexdigest()
print(json.dumps({"keys": run.keys(), "digest": digest}))
"""


def _in_a_fresh_process(tmp_path: Path, root: Path) -> dict[str, object]:
    """Run the small graph in a new interpreter and return keys and digests."""
    script = tmp_path / f"runner_{root.name}.py"
    script.write_text(_SUBPROCESS_RUNNER)
    completed = subprocess.run(
        [sys.executable, str(script), str(toy.__file__), str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _digest(run) -> dict[str, str]:
    digest = {}
    for node_id, cells in run.all_bytes().items():
        accumulator = hashlib.sha256()
        for cell, (values, mask) in sorted(cells.items()):
            accumulator.update(repr(cell).encode())
            accumulator.update(values)
            accumulator.update(mask)
        digest[node_id] = accumulator.hexdigest()
    return digest


@pytest.mark.xfail(strict=True, reason="charter A1: pending")
def test_a1_determinism_across_processes_and_a_reloaded_store(tmp_path: Path) -> None:
    """Same graph, same source bytes, same kernels: same keys, same bytes.

    Across two roots (so no key can depend on a path), across two fresh
    interpreters (so none can depend on process state), and through a store
    object reconstructed over an existing root.
    """
    first = toy.run_toy(toy.small_graph(), tmp_path / "first")
    second = toy.run_toy(toy.small_graph(), tmp_path / "second")
    assert first.keys() == second.keys()
    assert first.all_bytes() == second.all_bytes()

    restarted = _in_a_fresh_process(tmp_path, tmp_path / "third")
    again = _in_a_fresh_process(tmp_path, tmp_path / "fourth")
    assert restarted == again
    assert restarted["keys"] == first.keys()
    assert restarted["digest"] == _digest(first)

    reloaded = toy.run_toy(
        toy.small_graph(),
        tmp_path / "first",
        sources=first.sources,
        store=ContentStore(tmp_path / "first" / "store"),
    )
    assert reloaded.keys() == first.keys()
    assert reloaded.all_bytes() == first.all_bytes()


@pytest.mark.xfail(strict=True, reason="charter A2: pending")
def test_a2_memoization_executes_zero_kernels(tmp_path: Path) -> None:
    """The second run of an unchanged graph runs nothing and hits everywhere."""
    first = toy.run_toy(toy.small_graph(), tmp_path / "run")
    assert first.misses() == set(first.compiled.order)

    registry = toy.toy_registry()
    second = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources=first.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert toy.total_calls(registry) == 0
    assert second.misses() == set()
    assert all(second.manifest.nodes[node].hit for node in second.compiled.order)


@pytest.mark.xfail(strict=True, reason="charter A3: pending")
def test_a3_descendant_exact_invalidation(tmp_path: Path) -> None:
    """One normative parameter moves exactly that node and its descendants.

    Every ancestor and every sibling is a store hit. The miss set is asserted
    exactly: this is the property the review measured as 21 stores and 117
    files invalidated by an edit to a ``reason`` string.
    """
    graph = toy.full_graph()
    first = toy.run_toy(graph, tmp_path / "run")

    edited = toy.replace_node(
        graph, toy.impute("target_a", ("age", "income"), "target_a", noise=0.75)
    )
    registry = toy.toy_registry()
    second = toy.run_toy(
        edited,
        tmp_path / "run",
        sources=first.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    expected = {"target_a"} | toy.descendants(second.compiled, "target_a")
    assert second.misses() == expected
    assert second.hits() == set(second.compiled.order) - expected
    assert toy.calls_by_ref(registry)["derive.add@1"] == 0
    assert toy.total_calls(registry) == len(expected)


@pytest.mark.xfail(strict=True, reason="charter A4: pending")
def test_a4_inert_field_invariance(tmp_path: Path) -> None:
    """Descriptive fields change no node key: not on a node, not on a source,
    not on the graph's country label."""
    graph = toy.small_graph()
    first = toy.run_toy(graph, tmp_path / "run")

    relabelled = Graph(
        "a different label for the same country",
        (SourceRef("survey", "csv-tables", description="reworded"),),
        toy.replace_node(
            graph,
            toy.derive(
                "resources",
                ("age", "income"),
                "resources",
                scale=1.5,
                description="the sum of age and income, times a half again",
                citation="toy country handbook, table 1",
            ),
        ).nodes,
    )
    registry = toy.toy_registry()
    second = toy.run_toy(
        relabelled,
        tmp_path / "run",
        sources=first.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert second.keys() == first.keys()
    assert second.misses() == set()
    assert toy.total_calls(registry) == 0


@pytest.mark.xfail(strict=True, reason="charter A5: pending")
def test_a5_code_identity_invalidates_only_that_kernels_nodes(
    tmp_path: Path,
) -> None:
    """A kernel's implementation hash moves exactly its nodes and descendants."""
    graph = toy.full_graph()
    first = toy.run_toy(graph, tmp_path / "run")

    registry = toy.toy_registry(variants={"impute.chain@1": "recompiled"})
    second = toy.run_toy(
        graph,
        tmp_path / "run",
        sources=first.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    bound = {node.id for node in graph.nodes if node.kernel == "impute.chain@1"}
    assert bound == {"target_a", "target_b"}
    expected = bound | toy.descendants(second.compiled, *sorted(bound))
    assert second.misses() == expected
    assert second.hits() == set(second.compiled.order) - expected


@pytest.mark.xfail(strict=True, reason="charter A6: pending")
def test_a6_input_content_identity(tmp_path: Path) -> None:
    """Source bytes decide reuse; the source's path and file name do not."""
    original = toy.copy_source(tmp_path / "original")
    first = toy.run_toy(
        toy.small_graph(), tmp_path / "run", sources={"survey": original}
    )

    moved = toy.copy_source(tmp_path / "moved_somewhere_else")
    registry = toy.toy_registry()
    unchanged = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources={"survey": moved},
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert unchanged.keys() == first.keys()
    assert unchanged.misses() == set()
    assert toy.total_calls(registry) == 0

    edited = toy.copy_source(
        tmp_path / "edited", edit={"person.csv": ("9167.26", "9167.27")}
    )
    third = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources={"survey": edited},
        store=ContentStore(tmp_path / "run" / "store"),
    )
    consumers = {"survey"} | toy.descendants(third.compiled, "survey")
    assert third.misses() == consumers == set(third.compiled.order)
    assert third.keys()["survey"] != first.keys()["survey"]


@pytest.mark.xfail(strict=True, reason="charter A7: pending")
def test_a7_provenance_is_separate_from_reuse(tmp_path: Path) -> None:
    """A signed human decision lands in the manifest and in no node key."""
    first = toy.run_toy(toy.small_graph(), tmp_path / "run")

    registry = toy.toy_registry()
    second = toy.run_toy(
        toy.small_graph(),
        tmp_path / "run",
        sources=first.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
        decisions=(toy.PUBLISH_DECISION,),
    )
    assert second.keys() == first.keys()
    assert second.misses() == set()
    assert toy.total_calls(registry) == 0
    assert [dict(record) for record in second.manifest.decisions] == [
        toy.PUBLISH_DECISION
    ]


def test_the_acceptance_suite_only_touches_the_public_api() -> None:
    """The suite is black-box: no acceptance file imports a private module.

    Charter process rule 3 — the lane that writes a property's test is not the
    lane that makes it pass — only holds if the tests cannot see the
    implementation. This guard is green from the first commit and stays green.
    """
    forbidden = (
        "microcosm.graph.executor",
        "microcosm.graph.store",
        "microcosm.graph.keys",
        "microcosm.graph.canonical",
        "microcosm.graph.manifest",
        "microcosm.graph.population",
        "microcosm.graph.view",
    )
    here = Path(__file__).parent
    files = sorted(here.glob("test_acceptance_*.py")) + [here / "_toy.py"]
    assert len(files) >= 10
    for path in files:
        text = path.read_text()
        for module in forbidden:
            assert module not in text, f"{path.name} reaches into {module}"
