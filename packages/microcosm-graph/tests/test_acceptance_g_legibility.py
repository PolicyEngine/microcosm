"""Charter group G: legibility and country neutrality.

Finding 14 of the architecture review: answering "what predicts
``charitable_non_cash_donations``" takes three files, six regions, and 424
displayed lines today, and verifying the runtime chaining takes eight files and
about 800. G1 is the target — one screen, from the graph alone.

G2 is the other half of the same idea: the executor is the country-neutral
part, so a UK graph and a US graph are the same object with different
declarations, and the shard itself may not know either country's name. G3 is
issue #378 step 2: the whole shape running in CI on synthetic data.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from microcosm.graph import ContentStore, Graph, compile_graph, describe

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: Country-bearing tokens the shard may not carry. Every one of them appears in
#: the country runtimes and the data shards; none belongs in an executor whose
#: whole claim is that it does not know what country it is running.
COUNTRY_TOKENS = (
    "policyengine",
    "united states",
    "united kingdom",
    "asec",
    "hbai",
    "hmrc",
    "acs ",
    "cps ",
    "frs ",
    "puf ",
    "irs ",
)

#: Top-level modules that are country packages: importing one ends G2.
COUNTRY_PACKAGES = frozenset(
    {"policyengine_us", "policyengine_uk", "policyengine_canada", "policyengine_core"}
)


def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.xfail(strict=True, reason="charter G1: pending")
def test_g1_one_screen_view() -> None:
    """``describe`` answers the whole question from the graph alone.

    Predecessors, parameters, seed derivation, owned cells, and kernel
    identity, under 40 lines, without a store, a registry, or a source. The
    signature is the argument for "no other file consulted": there is nothing
    else to consult.
    """
    compiled = compile_graph(toy.full_graph())
    text = describe(compiled, "target_b")
    lines = text.splitlines()
    assert len(lines) < 40, f"{len(lines)} lines is more than one screen"

    for expected in (
        "target_b",
        "impute.chain@1",
        "survey",
        "target_a",
        "noise",
        "predictors",
        "person",
        "seed",
    ):
        assert expected in text, f"describe() never mentions {expected!r}"

    chained = describe(compiled, "sim")
    assert "calibrated" in chained
    assert "tax" in chained
    assert len(chained.splitlines()) < 40


@pytest.mark.xfail(strict=True, reason="charter G2: pending")
def test_g2_the_executor_knows_no_country(tmp_path: Path) -> None:
    """The shard names no country, and one executor runs two of them."""
    for path in toy.graph_source_files():
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        assert not (_imported_roots(tree) & COUNTRY_PACKAGES), (
            f"{path.name} imports a country package"
        )
        literals = " ".join(
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        for token in COUNTRY_TOKENS:
            assert token not in literals, f"{path.name} names {token!r}"

    nodes = toy.full_graph().nodes
    kingdom = Graph("uk", (toy.SOURCE,), nodes)
    states = Graph("us", (toy.SOURCE,), nodes)

    registry = toy.toy_registry()
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    first = toy.run_toy(
        kingdom, tmp_path, sources=sources, registry=registry, store=store
    )
    second = toy.run_toy(
        states, tmp_path, sources=sources, registry=registry, store=store
    )
    assert second.keys() == first.keys()
    assert second.misses() == set()
    assert toy.total_calls(registry) == len(first.compiled.order)


@pytest.mark.xfail(strict=True, reason="charter G3: pending")
def test_g3_toy_country_in_ci(tmp_path: Path) -> None:
    """Source, two chained targets, calibrate, simulate, gate, release.

    End to end, on synthetic data, with zero restricted inputs, in under a
    minute on the fast lane. Issue #378 step 2.
    """
    started = time.monotonic()
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    elapsed = time.monotonic() - started

    assert run.compiled.order == (
        "survey",
        "resources",
        "target_a",
        "target_b",
        "pool",
        "calibrated",
        "sim",
        "gate_tax",
        "release",
    )
    assert run.misses() == set(run.compiled.order)
    frame = run.manifest.population("calibrated")
    assert frame.n("person") == 502
    assert set(frame.person.columns) >= {"target_a", "target_b", "tax"}
    tier_key = run.manifest.nodes["release"].artifacts[("release", "tier")]
    assert str(run.store.load_column(tier_key).iloc[0]) == "certified"
    assert elapsed < 60, f"the toy country took {elapsed:.1f}s"


def test_the_toy_country_reads_only_files_it_ships() -> None:
    """Green from the first commit: zero restricted data, zero network.

    G3's "with zero restricted data" is only true if the fixture the toy
    country loads is the one committed beside it, so this guards the fixture
    set rather than restating the property.
    """
    fixtures = sorted(path.name for path in toy.FIXTURES.iterdir())
    assert fixtures == [
        "household.csv",
        "person.csv",
        "release.csv",
        "schema.json",
        "weights.csv",
    ]
    text = Path(toy.__file__).read_text()
    for forbidden in ("http://", "https://", "urlopen", "requests", "huggingface"):
        assert forbidden not in text
