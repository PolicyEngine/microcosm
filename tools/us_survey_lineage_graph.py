"""Integrated survey declarations and a scoped cache-canonical demographic adapter.

The caller supplies the complete initial composition inventory and the reviewed
demographic projection identity. Full materialized origin verification remains
mandatory after execution. This factory itself grants no genuine execution.
"""

import hashlib
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import graph_combined_clone as clone
from microcosm.build.us_runtime import graph_composed_asec_demographics as demographic
from microcosm.build.us_runtime import graph_native_household_origin as origin
from microcosm.build.us_runtime.graph_geography import GEOGRAPHY_PHASE
from microcosm.frame import Weights
from microcosm.graph import KernelBase, Owned, compile_graph
from microcosm.graph.canonical import canonical_json
from microcosm.graph.kernel import source_hash

NATIVE_POPULATION = f"{GEOGRAPHY_PHASE}.boundary"
NATIVE_BINDING = "survey_origins.native"
CLONE_BINDING = "survey_origins.clones"
CLONE_POPULATION = clone.COMBINED_CLONE_PREFIX


def canonical_demographic_outputs():
    """Own all rows only for the three newly appended nullable state columns."""
    return tuple(
        replace(cell, rows="all")
        if cell.column not in demographic.CANONICAL_COLUMNS
        else cell
        for cell in demographic.demographic_owned_declarations()
    )


def canonical_demographic_inputs():
    return tuple(
        replace(part, rows="all") for part in demographic.demographic_bind_inputs()
    )


class CacheCanonicalDemographicBindKernel(KernelBase):
    """Delegate all source/roster semantics; explicitly emit canonical new nulls.

    Generic masked Int64 creation uses pandas' hidden value 1 beneath nulls;
    ContentStore canonicalizes that value to 0. A later cached EXPAND correctly
    compares physical storage and refuses the mismatch. This adapter owns the
    complete three NEW state columns and emits zero null backing values on ACS.
    The two incumbent canonical booleans retain their accepted ASEC-only masks.
    No existing source column, null mask, row, ID or weight changes.
    """

    ref = "us.composed_asec_demographic_bind.cache_canonical@1"
    capabilities = demographic.USComposedAsecDemographicBindKernel.capabilities

    def implementation_hash(self):
        return hashlib.sha256(
            canonical_json(
                {
                    "schema": "microcosm.survey-demographic-cache-adapter.v1",
                    "accepted_binding": demographic.USComposedAsecDemographicBindKernel().implementation_hash(),
                    "adapter_module": source_hash(
                        sys.modules[__name__],
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        ).hexdigest()

    def run(self, context):
        if (
            context.node.outputs != canonical_demographic_outputs()
            or context.node.inputs != canonical_demographic_inputs()
        ):
            raise ValueError("CANONICAL_DEMOGRAPHIC_DECLARATION")
        original = demographic.USComposedAsecDemographicBindKernel()
        table = context.tables["person"]
        row_column = demographic.demographic_bind_inputs()[0].rows
        mask = table[row_column].to_numpy(dtype=bool)
        # Recreate precisely the accepted masked context; its implementation
        # authenticates every typed producer edge and every native source row.
        masked = replace(
            context,
            node=replace(
                context.node,
                kernel=original.ref,
                inputs=demographic.demographic_bind_inputs(),
                outputs=demographic.demographic_owned_declarations(),
            ),
            tables={"person": table.loc[mask]},
            weights={
                e: Weights(w.values[mask], w.kind) for e, w in context.weights.items()
            },
            strata=context.strata.loc[mask],
        )
        result = original.run(masked)
        columns = dict(result.columns)
        ids = pd.Index(table["person_id"].to_numpy(copy=True), name="person_id")
        for column, _ in demographic._RETAINED_STATES:
            cell = ("person", column)
            full = pd.Series(
                pd.arrays.IntegerArray(
                    np.zeros(len(ids), dtype=np.int64), np.ones(len(ids), dtype=bool)
                ),
                index=ids,
            )
            full.loc[columns[cell].index] = columns[cell].array
            columns[cell] = full
        return replace(
            result,
            columns=columns,
            receipt={
                **result.receipt,
                "cache_storage_adapter": {
                    "implementation_hash": self.implementation_hash(),
                    "new_state_null_backing_value": 0,
                    "canonical_incumbent_masks_preserved": True,
                },
            },
        )


def integrated_graph(
    columns, *, sample_fraction, sample_seed, projection_sha256, rows, cohort_rows
):
    """Compose, append demographics, bind native origins, clone, bind descendants."""
    graph = demographic.composed_asec_demographic_population_graph(
        columns,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        projection_sha256=projection_sha256,
        rows=rows,
        cohort_rows=cohort_rows,
        geography=True,
    )
    graph = replace(
        graph,
        nodes=tuple(
            replace(
                node,
                kernel=CacheCanonicalDemographicBindKernel.ref,
                inputs=canonical_demographic_inputs(),
                outputs=canonical_demographic_outputs(),
            )
            if node.id == demographic.DEMOGRAPHIC_BIND_NODE
            else node
            for node in graph.nodes
        ),
    )
    # Follow actual compiled ownership, including geography, measured columns,
    # nullable canonical rewrites and all carried source columns. The initial
    # inventory alone cannot describe the final native population.
    compiled = compile_graph(graph)
    inventory = {}
    for node_id in compiled.order:
        for cell in graph.node(node_id).outputs:
            inventory[cell.entity, cell.column] = Owned(
                cell.entity, cell.column, cell.dtype
            )
    final_columns = tuple(inventory[key] for key in sorted(inventory))
    nodes = (
        *graph.nodes,
        *origin.native_origin_source_nodes(population=NATIVE_POPULATION),
        origin.native_origin_binding_node(
            final_columns,
            population=NATIVE_POPULATION,
            node_id=NATIVE_BINDING,
        ),
        *clone.us_combined_survey_clone_nodes(
            final_columns,
            base=NATIVE_POPULATION,
            source_channels=("acs", "asec"),
        ),
        origin.native_origin_binding_node(
            final_columns,
            population=CLONE_POPULATION,
            node_id=CLONE_BINDING,
            parent_binding_node=NATIVE_BINDING,
        ),
    )
    integrated = replace(graph, sources=(*graph.sources, *origin.SOURCES), nodes=nodes)
    result = compile_graph(integrated)
    assert NATIVE_BINDING in result.predecessors[CLONE_POPULATION]
    assert demographic.DEMOGRAPHIC_BIND_NODE in result.predecessors[NATIVE_BINDING]
    return integrated


def integrated_registry():
    registry = demographic.composed_asec_demographic_registry(geography=True)
    registry.register(CacheCanonicalDemographicBindKernel())
    clone.register_us_combined_survey_clone_kernels(registry)
    origin.register_native_origin_kernels(registry)
    return registry
