"""The UK full build's pool checkpoint and the population helpers its nodes share.

The dense line builds only on a geography-first spine
(:mod:`~microcosm.build.uk_runtime.spine_posture`), as that spine stands: its
``geographic_expand`` and ``geographic_identity`` stages give every household
its geographic clones and keys, its geography stages assign each clone its
area, and its root samples at the spine's rung, so the full build samples,
expands, keys and places nothing of its own (microcosm#931). Its targets and
measures read the spine's version, and :class:`UKPoolCheckpointKernel` keeps
every row of it before the solve. Selection, row ancestry, typed weights,
content storage and replay are enforced by the shared graph runtime.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from microcosm.frame import Frame
from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    Slice,
    StructuralDelta,
    compile_graph,
    source_hash,
)

from . import national_frame
from .national_frame import UK_NATIONAL_SCHEMA


def context_frame(context: KernelContext) -> Frame:
    """Rebuild only declared slices, with the shared immutable weight context."""

    return Frame(
        {
            entity: context.tables[entity]
            .loc[
                :,
                list(
                    context.frame_column_order.get(
                        entity, tuple(context.tables[entity].columns)
                    )
                ),
            ]
            .copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        mass_log=getattr(context, "frame_mass_log", ()),
        metadata=getattr(context, "frame_metadata", {})
        or {"time_period": str(context.params["time_period"])},
    )


def population_columns(graph: Graph, population: str) -> dict[tuple[str, str], str]:
    """Resolve the carried and newly owned columns in a compiled version."""

    compiled = compile_graph(graph)
    holder = graph.node(population)
    cells = {} if holder.base is None else population_columns(graph, holder.base)
    for node in graph.nodes:
        if compiled.versions[node.id] == population:
            cells.update({(o.entity, o.column): o.dtype for o in node.outputs})
    return cells


def population_slices(cells: Mapping[tuple[str, str], str]) -> tuple[Slice, ...]:
    return tuple(
        Slice(entity, tuple(sorted(c for e, c in cells if e == entity)))
        for entity in sorted({e for e, _ in cells})
    )


class UKPoolCheckpointKernel(KernelBase):
    """Checkpoint the bound spine before the solve, keeping every row.

    A keep-all FILTER on the spine's version: the dense pool is the bound
    spine, its geographic clones and areas with it (microcosm#931). The node's
    base is that version, so it follows every operation on it, the target
    compilation, selection, measures and problem among them, and it binds the
    complete incoming ledger before the calibration nodes reconstruct any
    Frame slices. It reads and emits no typed artifact.
    """

    ref = "uk.full.pool@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        structural=StructuralDelta.FILTER,
    )

    def implementation_hash(self) -> str:
        # The checkpoint reads the person axis national_frame's schema names,
        # and nothing else (microcosm#931).
        return source_hash(type(self), national_frame)

    def run(self, context: KernelContext) -> KernelResult:
        person = context.tables[UK_NATIONAL_SCHEMA.person_entity]
        id_column = UK_NATIONAL_SCHEMA.person_id_column
        ids = pd.Index(person[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(
            keep=pd.Series(True, index=ids, dtype="bool"),
            receipt={"scope": "complete_pool_checkpoint"},
        )


def register_uk_population_kernels(registry: KernelRegistry) -> None:
    """Register the pool checkpoint. The shared atomic-geography kernels are
    not the UK's to register: a geography-first spine assigns its areas in its
    own stages, and NZ's transport registers the shared kernels it runs."""

    registry.register(UKPoolCheckpointKernel())
