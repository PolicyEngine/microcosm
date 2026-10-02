"""Synthetic full-build population graph: the seam-frame source kernel and
the graph/registry pair (legacy ladder draw or identity-keyed atomic-area
assignment) the UK full-graph tests build on."""

# ruff: noqa: F401

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.graph_kernels import UKClaimKernel
from microcosm.build.uk_runtime.graph_population import (
    UKBRMATableKernel,
    append_uk_population_nodes,
    register_uk_population_kernels,
)
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
    ladder_clone_index_column,
)
from microcosm.frame import Frame
from microcosm.frame.schema import VariableMetadata
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
BRMA_REGIONS = ("LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND")


def toy_brma_count_resource():
    """Invented cells: alphabetical order differs from the engine's rate order."""
    return {
        "cells": {
            region: {
                category: {
                    f"{region}_A": 1,
                    f"{region}_B": 2,
                    f"{region}_C": 1,
                }
                for category in "ABCDE"
            }
            for region in BRMA_REGIONS
        }
    }


class FakeBRMAEngine:
    """A UK protocol double calculating per-unit category and BRMA rates."""

    country = "uk"

    def __init__(self, *, uncapped=True):
        self.uncapped = uncapped
        self.calls = []

    def variable_metadata(self, name):
        if name == "uncapped_BRMA_LHA_rate" and not self.uncapped:
            raise ValueError("Unknown variable 'uncapped_BRMA_LHA_rate'.")
        if name not in {
            "LHA_category",
            "uncapped_BRMA_LHA_rate",
            "BRMA_LHA_rate",
        }:
            raise ValueError(f"Unknown variable {name!r}.")
        return VariableMetadata(
            name, "benunit", "str" if name == "LHA_category" else "float", "year"
        )

    def materialize(self, frame, variables, period):
        assert str(period) == "2023"
        self.calls.append(tuple(variables))
        benunit = frame.table("benunit")
        placements = frame.table("person").drop_duplicates("person_benunit_id")
        household_ids = benunit["benunit_id"].map(
            placements.set_index("person_benunit_id")["person_household_id"]
        )
        values = {}
        for variable in variables:
            self.variable_metadata(variable)
            if variable == "LHA_category":
                values[variable] = np.full(len(benunit), "B", dtype=object)
            else:
                names = household_ids.map(
                    frame.table("household").set_index("household_id")["brma"]
                )
                values[variable] = np.asarray(
                    [{"A": 300.0, "B": 100.0, "C": 200.0}[n[-1]] for n in names]
                )
        return values


class FakeBRMATableKernel(UKBRMATableKernel):
    """Exercise the real calculation without importing a country distribution."""

    capabilities = replace(UKBRMATableKernel.capabilities, dependencies=())


def register_toy_population_kernels(registry, *, engine=None):
    engine = FakeBRMAEngine() if engine is None else engine
    resource = toy_brma_count_resource()
    production = KernelRegistry()
    register_uk_population_kernels(
        production, engine=engine, brma_count_resource=resource
    )
    for kernel in production.as_mapping().values():
        registry.register(
            FakeBRMATableKernel(engine=engine, count_resource=resource)
            if kernel.ref == UKBRMATableKernel.ref
            else kernel
        )


def pregeographic_key(row, *, vintage=SOURCE_VINTAGE):
    """Independent lineage oracle, deliberately excluding local clone identity."""
    path = []
    if row.household_support_channel == "spi":
        path.append(("spi_support_channel", int(row.household_support_clone_index)))
    if row.cgt_support_copy_index:
        path.append(("cgt_support_split", int(row.cgt_support_copy_index)))
    if row.household_is_capital_gains_clone:
        path.append(("cgt_incidence_clone", 1))
    return household_draw_key(
        source="frs",
        source_vintage=vintage,
        source_household_id=int(row.source_household_id),
        clone_path=tuple(path),
    )


def expected_clone_brmas(household, k, *, vintage=SOURCE_VINTAGE):
    """Inverse the invented B,C,A walk directly, without BRMA spread helpers."""
    rows = list(household.itertuples(index=False))
    uniforms = stable_identity_uniforms(
        [pregeographic_key(row, vintage=vintage) for row in rows],
        seed=0,
        salt="brma:clone_spread",
    )
    clone_column = ladder_clone_index_column("household")
    quantiles = (uniforms + household[clone_column].to_numpy() / k) % 1.0
    suffixes = np.where(quantiles < 0.5, "B", np.where(quantiles < 0.75, "C", "A"))
    return np.asarray(
        [f"{row.region}_{suffix}" for row, suffix in zip(rows, suffixes, strict=True)]
    )


def source_frame():
    original = _seam_frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    tables["person"]["age"] = 40
    tables["benunit"]["would_claim_uc"] = True
    household = tables["household"]
    household["region"] = household["region"].astype("string")
    household["brma"] = pd.array(
        [f"{region}_A" for region in household["region"]], dtype="string"
    )
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
        source_vintage=SOURCE_VINTAGE,
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(UKClaimKernel())
    register_toy_population_kernels(registry)
    return graph, registry


__all__ = [name for name in globals() if not name.startswith("__")]
