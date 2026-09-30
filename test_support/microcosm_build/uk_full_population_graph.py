"""Synthetic full-build population graph: the seam-frame source kernel and
the graph/registry pair (legacy ladder draw or identity-keyed atomic-area
assignment) the UK full-graph tests build on."""

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

SOURCE_VINTAGE = "toy_2023_24"


def source_frame():
    original = _seam_frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    tables["person"]["age"] = 40
    tables["benunit"]["would_claim_uc"] = True
    household = tables["household"]
    household["region"] = household["region"].astype("string")
    # Explicit spine lineage: every toy household is its own FRS original.
    household["source_household_id"] = household["household_id"].astype("int64")
    household["household_support_channel"] = pd.array(
        ["frs"] * len(household), dtype="string"
    )
    household["household_support_clone_index"] = np.zeros(len(household), dtype="int64")
    household["household_is_spi_synthetic"] = False
    household["household_is_capital_gains_clone"] = False
    household["household_is_cgt_support_copy"] = False
    household["cgt_support_copy_index"] = np.zeros(len(household), dtype="int64")
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


def source_node(frame):
    return Node(
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


def graph_and_registry(k, *, geography_assignment="legacy", definition=None, seed=7):
    graph = append_uk_population_nodes(
        Graph(
            "uk",
            (SourceRef("fixture", "raw-bytes-v1"),),
            (source_node(source_frame()),),
        ),
        population="source",
        time_period="2023",
        weight_kind="importance",
        n_clones=k,
        seed=seed,
        source_year=2023,
        geography_assignment=geography_assignment,
        atomic_geography_definition=definition,
        source_vintage=SOURCE_VINTAGE if geography_assignment == "atomic" else None,
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(UKClaimKernel())
    register_uk_population_kernels(registry)
    return graph, registry


__all__ = [name for name in globals() if not name.startswith("__")]
