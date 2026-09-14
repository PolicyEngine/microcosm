"""Full-build population graph preserves the maintained geography operation."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from test_uk_ladder_rowwise_clone import _seam_frame
from test_uk_ladder_rowwise_clone import toy_ladder as toy_ladder

from microcosm.build.uk_runtime.graph_kernels import UKClaimKernel
from microcosm.build.uk_runtime.graph_population import (
    append_uk_population_nodes,
    register_uk_population_kernels,
)
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
)
from microcosm.frame import Frame
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)


def source_frame():
    original = _seam_frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    tables["person"]["age"] = 40
    tables["benunit"]["would_claim_uc"] = True
    tables["household"]["region"] = tables["household"]["region"].astype("string")
    return Frame(
        tables,
        original.schema,
        {"household": original.weights_for("household")},
        original.strata,
        mass_log=original.mass_log,
        metadata=original.metadata,
    )


class Source(KernelBase):
    ref = "uk.test.full-source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context):
        return KernelResult(frame=source_frame())


def graph_and_registry(k):
    frame = source_frame()
    source = Node(
        "source",
        Source.ref,
        structural=StructuralDelta.CREATE,
        sources=("fixture",),
        outputs=tuple(
            Owned(
                entity,
                col,
                "string" if table[col].dtype.kind in "OUS" else str(table[col].dtype),
            )
            for entity in frame.entities
            for table in [frame.table(entity)]
            for col in table.columns
            if col
            not in {
                "person_id",
                "person_household_id",
                "person_benunit_id",
                "household_id",
                "benunit_id",
            }
        ),
    )
    graph = append_uk_population_nodes(
        Graph("uk", (SourceRef("fixture", "raw-bytes-v1"),), (source,)),
        population="source",
        time_period="2023",
        weight_kind="importance",
        n_clones=k,
        seed=7,
        source_year=2023,
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(UKClaimKernel())
    register_uk_population_kernels(registry)
    return graph, registry


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
