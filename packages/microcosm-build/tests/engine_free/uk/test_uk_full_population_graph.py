"""Full-build population graph preserves the maintained geography operation."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
)
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.uk_full_population_graph import (
    graph_and_registry,
    source_frame,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_population_graph_preserves_rows_weights_geography_and_replays(
    k, toy_ladder, tmp_path
):
    ladder, path = toy_ladder
    expected = clone_uk_dataset_with_ladder_geography(
        source_frame(),
        ladder,
        n_clones=k,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
    ).frame
    graph, registry = graph_and_registry(k)
    store = ContentStore(tmp_path / "store")
    compiled = compile_graph(graph)
    assert "uk.full.locations" in compiled.predecessors["uk.full.geography_mapping"]
    first = run_graph(
        compiled,
        sources={"uk_ladder": path, "fixture": path},
        store=store,
        kernels=registry,
    )
    actual = first.population("uk.full.expand")
    for entity in expected.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity)[expected.table(entity).columns],
            expected.table(entity),
            check_dtype=False,
        )
    np.testing.assert_array_equal(
        actual.weights_for("household").values, expected.weights_for("household").values
    )
    assert actual.mass_log == expected.mass_log
    # A new registry cannot obtain results from mutable objects of the cold run.
    _, fresh_registry = graph_and_registry(k)
    replay = run_graph(
        compiled,
        sources={"uk_ladder": path, "fixture": path},
        store=store,
        kernels=fresh_registry,
        resume="require",
    )
    assert replay.population("uk.full.expand").mass_log == actual.mass_log


def test_k_changes_expansion_but_never_sampling():
    first, _ = graph_and_registry(1)
    second, _ = graph_and_registry(3)
    assert first.node("uk.full.sample") == second.node("uk.full.sample")
    assert first.node("uk.full.normalize") == second.node("uk.full.normalize")
    assert first.node("uk.full.expand").params["n_clones"] == 1
    assert second.node("uk.full.expand").params["n_clones"] == 3
