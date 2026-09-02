"""Charter group H: parity (migration acceptance).

These three properties are the only ones in the charter whose subject is not
the toy country. Each compares a graph node's output against a pinned artifact
produced by the lane that wraps the legacy kernel or migrates the country
spine, so each waits on a fixture this lane cannot manufacture: inventing one
would prove that the suite agrees with itself, which is exactly what parity
must not mean.

Every test names the fixture path it expects and reads it and nothing else, so
the producing lane can drop its artifact in and delete the marker. Until then
the fixture is absent, the test fails, and the ``xfail`` reason says whose
fixture it is waiting for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: Where the parity lanes drop their pinned fixtures.
PARITY = Path(__file__).parent / "fixtures" / "parity"

#: H1: one directory per wrapped legacy kernel, each holding ``graph.json``
#: (the node declaration), ``inputs.csv``, ``direct.csv`` (the direct call's
#: output at the pinned seed), and ``pins.json`` (seed, kernel ref, kernel
#: implementation hash, and the dependency versions the pin was taken under).
KERNEL_PARITY = PARITY / "kernels"

#: H2: ``uk_spine.json`` — the 26-stage FRS spine expressed as a graph — plus
#: ``uk_frame_content_identity.txt``, the identity the current spine produces
#: on the same fixture.
UK_SPINE_PARITY = PARITY / "uk_spine"

#: H3: ``us_post_transfer.json`` — the derive/seed/simulate subgraph of the
#: stacked spine — plus ``expected.csv``, its pinned output.
US_POST_TRANSFER_PARITY = PARITY / "us_post_transfer"

#: The wrapped kernels H1 covers, in the order the charter names them.
#: The three wrapped kernels the kernel lane shipped: ``fit.qrf@1`` fits on
#: donors and draws on recipients in one node, so there is no separate draw.
WRAPPED_KERNELS = ("fit.qrf", "calibrate", "simulate")

#: What each wrapper honestly claims about its numbers. The forest stack does
#: not promise cross-platform bit stability, so ``fit.qrf@1`` says so; parity
#: in the locked environment is still asserted byte for byte below.
NUMERIC_CLAIMS = {
    "fit.qrf": "tolerance_bound",
    "calibrate": "bitwise",
    "simulate": "bitwise",
}


def _assert_same_bytes(actual, expected) -> None:
    assert actual.dtype == expected.dtype
    assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert np.array_equal(actual.isna().to_numpy(), expected.isna().to_numpy())


def _require(path: Path, produced_by: str) -> Path:
    """Fail with the fixture's path and its owner, never with a bare error."""
    assert path.exists(), (
        f"missing parity fixture {path}; it is produced by {produced_by}, not "
        "by the acceptance lane — inventing it here would make the test agree "
        "with itself instead of with the legacy kernel."
    )
    return path


def test_h1_kernel_parity(tmp_path: Path) -> None:
    """A wrapped legacy kernel is byte-identical to the direct call.

    Expects ``packages/microcosm-graph/tests/fixtures/parity/kernels/<name>/``
    for each of ``fit.qrf``, ``draw.qrf``, ``calibrate``, and ``simulate``,
    each holding ``graph.json``, ``inputs.csv``, ``direct.csv``, and
    ``pins.json``. The wrappers live in ``microcosm-fit``,
    ``microcosm-calibrate``, and the ``RulesEngine`` adapter; the lane that
    writes them produces these fixtures at the same pinned seed.
    """
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph
    from tools.graph_parity_fixtures import parity_registry

    _require(KERNEL_PARITY, "the kernel-wrapper lane (#378 step 3)")
    registry = parity_registry()
    for name in WRAPPED_KERNELS:
        case = _require(KERNEL_PARITY / name, "the kernel-wrapper lane")
        pins = json.loads((case / "pins.json").read_text())
        assert set(pins) >= {"seed", "kernel", "implementation_hash", "dependencies"}
        kernel = registry.get(pins["kernel"])
        assert kernel.implementation_hash() == pins["implementation_hash"]
        assert set(pins["dependencies"]) == set(kernel.capabilities.dependencies)

        store = ContentStore(tmp_path / name)
        manifest = run_graph(
            compile_graph(graph_from_json((case / "graph.json").read_text())),
            sources={"fixture": case},
            store=store,
            kernels=registry,
            resume="forbid",
            decisions=(),
        )
        node = manifest.nodes[pins["node"]]
        assert node.receipt["capabilities"]["numeric"] == NUMERIC_CLAIMS[name]
        # A structural node re-keys every carried column as an artifact of its
        # own; the direct call produced only what direct.csv holds, so those
        # are the cells compared. A weight transition is compared through the
        # weight artifact under the ``<entity>.weights`` column.
        direct = _direct_table(case)
        compared = 0
        for cell, key in node.artifacts.items():
            label = f"{cell[0]}.{cell[1]}"
            if label in direct.columns:
                _assert_same_bytes(store.load_column(key), direct[label])
                compared += 1
        if node.weight_key is not None:
            entity = (
                graph_from_json((case / "graph.json").read_text())
                .node(pins["node"])
                .weights.entity
            )
            _assert_same_bytes(
                store.load_column(node.weight_key), direct[f"{entity}.weights"]
            )
            compared += 1
        assert compared, f"{name}: the fixture exposed nothing to compare"


def _direct_table(case: Path):
    import pandas as pd

    return pd.read_csv(case / "direct.csv", float_precision="round_trip")


@pytest.mark.xfail(strict=True, reason="charter H2: awaiting fixture from UK lane")
def test_h2_uk_spine_parity() -> None:
    """The UK spine as a graph reproduces ``uk_frame_content_identity``.

    Expects ``packages/microcosm-graph/tests/fixtures/parity/uk_spine/`` with
    ``uk_spine.json`` and ``uk_frame_content_identity.txt``. Stage order comes
    from declared ``consumes``: the assertion below is that the compiled
    topological order is derived, so the hand-maintained ``_STAGE_NAMES`` tuple
    in ``tools/build_uk_frs_spine.py`` — the 26 names intersected with a
    28-stage packaged manifest, kept in step by hand — can be deleted.
    """
    _require(UK_SPINE_PARITY, "the UK migration lane (charter H2, María reviews)")
    expected = (UK_SPINE_PARITY / "uk_frame_content_identity.txt").read_text().strip()

    from microcosm.graph import ContentStore, Graph, compile_graph, run_graph

    declaration = json.loads((UK_SPINE_PARITY / "uk_spine.json").read_text())
    compiled = compile_graph(Graph(**declaration))
    assert len(compiled.order) >= 26
    assert all(
        set(compiled.predecessors[node_id]) <= set(compiled.order[:index])
        for index, node_id in enumerate(compiled.order)
    )

    manifest = run_graph(
        compiled,
        sources={"frs": UK_SPINE_PARITY},
        store=ContentStore(UK_SPINE_PARITY / "_store"),
        kernels=toy.toy_registry(),
        resume="forbid",
        decisions=(),
    )
    assert manifest.nodes[compiled.order[-1]].receipt["content_identity"] == expected


@pytest.mark.xfail(strict=True, reason="charter H3: awaiting fixture from US lane")
def test_h3_us_post_transfer_parity() -> None:
    """The derive/seed/simulate subgraph reproduces its pinned fixture output.

    Expects
    ``packages/microcosm-graph/tests/fixtures/parity/us_post_transfer/`` with
    ``us_post_transfer.json`` and ``expected.csv``. This is the subgraph the
    stacked spine runs after the ACS transfer, so it is the first real US
    surface the executor owns.
    """
    _require(US_POST_TRANSFER_PARITY, "the US migration lane (#378 step 3)")

    import pandas as pd

    from microcosm.graph import ContentStore, Graph, compile_graph, run_graph

    declaration = json.loads(
        (US_POST_TRANSFER_PARITY / "us_post_transfer.json").read_text()
    )
    store = ContentStore(US_POST_TRANSFER_PARITY / "_store")
    manifest = run_graph(
        compile_graph(Graph(**declaration)),
        sources={"stacked": US_POST_TRANSFER_PARITY},
        store=store,
        kernels=toy.toy_registry(),
        resume="forbid",
        decisions=(),
    )
    expected = pd.read_csv(US_POST_TRANSFER_PARITY / "expected.csv")
    for column in expected.columns:
        entity, name = column.split(".", 1)
        key = manifest.nodes[declaration["owners"][column]].artifacts[(entity, name)]
        assert store.load_column(key).to_numpy().tobytes() == (
            expected[column].to_numpy().tobytes()
        )


def test_the_parity_fixtures_are_declared_but_not_faked() -> None:
    """Green from the first commit: no parity fixture is invented here.

    If a directory ever appears under ``fixtures/parity/`` in a commit that
    also touches this file, that is the acceptance lane manufacturing its own
    evidence. The suite says so out loud instead.
    """
    if not PARITY.exists():
        return
    for case in sorted(PARITY.iterdir()):
        assert case.is_dir()
        assert (case / "PRODUCED_BY.txt").exists(), (
            f"{case} carries no note saying which lane produced it"
        )
