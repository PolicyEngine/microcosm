"""Tests split from packages/microcosm-graph/tests/test_acceptance_h_parity.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_graph.acceptance_h_parity import *


def test_h3_us_post_transfer_parity(tmp_path: Path) -> None:
    """The derive/seed/simulate subgraph reproduces its pinned fixture output.

    Expects
    ``packages/microcosm-graph/tests/fixtures/parity/us_post_transfer/`` with
    ``us_post_transfer.json`` (the graph, pinned), ``sources/`` (a synthetic
    stacked pool the CREATE node loads), and ``expected.csv`` (the cells the
    current ``prepare_stacked_tail_derivation`` → ``derive`` → ``seed`` →
    ``materialize`` functions produce on those sources, as ``entity.column``
    columns). This is the subgraph the stacked spine runs after the ACS
    transfer, so it is the first real US surface the executor owns. It needs
    the US engine, so it runs in the engine tier.
    """
    _require(US_POST_TRANSFER_PARITY, "the US migration lane (#378 step 3)")

    import pandas as pd

    from microcosm.build.us_runtime.graph import us_post_transfer_graph, us_registry
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph

    # The graph the US lane ships is pinned as JSON beside the fixture, so a
    # silent change to the declaration shows up as a fixture diff.
    graph = us_post_transfer_graph()
    assert (
        graph_from_json((US_POST_TRANSFER_PARITY / "us_post_transfer.json").read_text())
        == graph
    )
    compiled = compile_graph(graph)
    manifest = run_graph(
        compiled,
        sources={"stacked": US_POST_TRANSFER_PARITY / "sources"},
        store=ContentStore(tmp_path / "store"),
        kernels=us_registry(),
        resume="forbid",
        decisions=(),
    )
    final = manifest.population(compiled.versions[compiled.order[-1]])
    expected = pd.read_csv(
        US_POST_TRANSFER_PARITY / "expected.csv", float_precision="round_trip"
    )
    assert len(expected.columns) >= 3, (
        "the fixture pins the derived, seeded, and simulated cells"
    )
    for column in expected.columns:
        entity, name = column.split(".", 1)
        actual = final.table(entity)[name]
        assert actual.dtype == expected[column].dtype, column
        assert actual.to_numpy().tobytes() == expected[column].to_numpy().tobytes(), (
            column
        )
