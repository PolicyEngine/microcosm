"""Synthetic full-build population graph: the seam-frame source kernel and
the graph/registry pair the UK full-graph tests build on."""

# ruff: noqa: F401

import hashlib

import numpy as np
import pandas as pd
import pytest

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
from test_support.microcosm_build.uk_ladder_rowwise_clone import _seam_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
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


__all__ = [name for name in globals() if not name.startswith("__")]
