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
    household["household_is_cgt_residential_clone"] = False
    household["cgt_residential_clone_index"] = np.zeros(len(household), dtype="int64")
    # microcosm#1240: the housing columns the post-geography property nodes
    # read and rewrite; one owner per region with a positive value, the
    # Scottish household a renter, the Northern Irish one an owner at zero.
    household["tenure_type"] = pd.array(
        ["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE", "RENT_PRIVATELY", "OWNED_OUTRIGHT"][
            : len(household)
        ],
        dtype="string",
    )
    household["main_residence_value"] = np.asarray(
        [400_000.0, 250_000.0, 0.0, 0.0][: len(household)], dtype="float64"
    )
    household["property_wealth"] = np.asarray(
        [450_000.0, 250_000.0, 0.0, 10_000.0][: len(household)], dtype="float64"
    )
    household["council_tax_band"] = pd.array(
        ["D", "C", "B", "D"][: len(household)], dtype="string"
    )
    # microcosm#1262: the spine's regional BRMA draw, which the area-BRMA node
    # replaces outside Northern Ireland.
    household["brma"] = pd.array(
        ["MAIDSTONE", "MAIDSTONE", "MAIDSTONE", "BELFAST"][: len(household)],
        dtype="string",
    )
    # The Scottish household is the toy's one private renter; the private-rent
    # node re-draws its rent from its area's toy list.
    household["rent"] = np.asarray(
        [0.0, 0.0, 9_000.0, 0.0][: len(household)], dtype="float64"
    )
    household["num_bedrooms"] = np.asarray(
        [3, 3, 2, 3][: len(household)], dtype="int64"
    )
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


def graph_and_registry(
    k,
    *,
    geography_assignment="legacy",
    definition=None,
    seed=7,
    property_localisation_mode="none",
    brma_localisation_mode="none",
    private_rent_localisation_mode="none",
):
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
        property_localisation_mode=property_localisation_mode,
        brma_localisation_mode=brma_localisation_mode,
        private_rent_localisation_mode=private_rent_localisation_mode,
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(UKClaimKernel())
    register_uk_population_kernels(registry)
    return graph, registry


__all__ = [name for name in globals() if not name.startswith("__")]
