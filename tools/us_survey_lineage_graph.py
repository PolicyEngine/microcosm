"""Integrated survey declarations; no source I/O and no new kernel identity.

The caller supplies the complete initial composition inventory and the reviewed
demographic projection identity. Full materialized origin verification remains
mandatory after execution. This factory itself grants no genuine execution.
"""

from dataclasses import replace

from microcosm.build.us_runtime import graph_combined_clone as clone
from microcosm.build.us_runtime import graph_composed_asec_demographics as demographic
from microcosm.build.us_runtime import graph_native_household_origin as origin
from microcosm.build.us_runtime.graph_geography import GEOGRAPHY_PHASE
from microcosm.graph import Owned, compile_graph

NATIVE_POPULATION = f"{GEOGRAPHY_PHASE}.boundary"
NATIVE_BINDING = "survey_origins.native"
CLONE_BINDING = "survey_origins.clones"
CLONE_POPULATION = clone.COMBINED_CLONE_PREFIX


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
    clone.register_us_combined_survey_clone_kernels(registry)
    origin.register_native_origin_kernels(registry)
    return registry
