"""Tests split from packages/microcosm-graph/tests/test_acceptance_h_parity.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_graph.acceptance_h_parity import *
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-graph")


def test_h2_uk_spine_parity(tmp_path: Path) -> None:
    """The UK spine as a graph reproduces ``uk_frame_content_identity``.

    The spine's stages read the UK engine's parameters, so this runs in the
    engine tier (``requires_uk``) and skips in the engine-free fast lane.

    Expects ``packages/microcosm-graph/tests/fixtures/parity/uk_spine/`` with
    ``uk_spine.json`` and ``sources/``; both sides run the 32 transforms from
    those sources in this process, root included, because the root weights
    differ at the last bit between machines. Stage order comes
    from declared ``consumes``: the assertion below is that the compiled
    topological order is derived, so the hand-maintained ``_STAGE_NAMES`` tuple
    in ``tools/build_uk_frs_spine.py`` — the 30 names intersected with a
    33-stage packaged manifest, kept in step by hand — can be deleted.
    """
    _require(UK_SPINE_PARITY, "the UK migration lane (charter H2, María reviews)")

    from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
    from microcosm.build.uk_runtime.graph import uk_registry, uk_spine_graph
    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph
    from tools.graph_uk_spine_fixture import legacy_oracle_frame

    # The identity is a byte-exact fingerprint of every cell, and the FRS root
    # transform's weights differ at the last bit between machines, so both
    # sides derive the root from the same raw tables here, in this process:
    # the oracle through the legacy StagePlan, the graph through a CREATE
    # kernel bound to the same root transform class (its production path).
    oracle = legacy_oracle_frame(UK_SPINE_PARITY)
    expected = uk_frame_content_identity(oracle)
    _, implementations = fixture_stage_plan_inputs(UK_SPINE_PARITY / "sources")

    # The graph the UK lane ships is also pinned as JSON beside the fixture, so
    # a silent change to the declaration shows up as a fixture diff.
    graph = uk_spine_graph()
    assert graph_from_json((UK_SPINE_PARITY / "uk_spine.json").read_text()) == graph
    compiled = compile_graph(graph)
    assert len(compiled.order) >= 32, "a CREATE node plus the 32 spine stages"
    assert all(
        set(compiled.predecessors[node_id]) <= set(compiled.order[:index])
        for index, node_id in enumerate(compiled.order)
    )

    manifest = run_graph(
        compiled,
        sources={"frs": UK_SPINE_PARITY / "sources"},
        store=ContentStore(tmp_path / "store"),
        kernels=uk_registry(dict(implementations)),
        resume="forbid",
        decisions=(),
    )
    final_version = compiled.versions[compiled.order[-1]]
    final = manifest.population(final_version)
    actual = uk_frame_content_identity(final)
    if actual != expected:
        # Say which cells disagree, and whether the legacy path itself is
        # process-deterministic on this machine, before failing.
        differences = _frame_differences(final, oracle)
        repeated = uk_frame_content_identity(legacy_oracle_frame(UK_SPINE_PARITY))
        stability = (
            "legacy oracle reproduces its own identity in this process"
            if repeated == expected
            else f"legacy oracle is NOT process-deterministic here: {repeated}"
        )
        print(f"H2 graph {actual} != legacy {expected}\n{stability}\n{differences}")
        pytest.fail(f"graph != legacy oracle; {stability}\n{differences}")

    # Charter F12: stage order is derived from declared inputs, so the
    # hand-maintained tuple in the UK driver is gone.
    driver = (_TEST_PATHS.repository / "tools" / "build_uk_frs_spine.py").read_text()
    assert "_STAGE_NAMES" not in driver
