"""A separate receiving version for optional common-survey completion.

The country host binds the complete actual Frame and verifies the real FILTER
result, including its ledger and original parent custody. This operation keeps
every row and supplies neither completed values nor source/population authority.
"""

from __future__ import annotations

import sys

import pandas as pd

from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.decl import ROWS_ALL
from microcosm.graph.keys import opaque_artifact_key

from . import graph_property_tax_leaves as tax

PROTOCOL = "microcosm.us.survey-completion-receiving.v1"
NODE = "survey_completion.receiving"
KERNEL = "us.survey_completion_receiving@1"


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_COMPLETION_" + reason)


def _ordering(ordering):
    _require(
        type(ordering) is tuple
        and all(type(edge) is ArtifactInput for edge in ordering)
        and len({edge.name for edge in ordering}) == len(ordering),
        "ORDERING_EDGES",
    )
    return ordering


def _node(population, slices, ordering):
    _require(
        type(population) is str and bool(population) and population != NODE,
        "POPULATION_VERSION",
    )
    _require(
        type(slices) is tuple
        and all(type(s) is Slice and s.rows == ROWS_ALL for s in slices)
        and len({s.entity for s in slices}) == len(slices)
        and "person" in {s.entity for s in slices},
        "FULL_SLICES",
    )
    return Node(
        NODE,
        KERNEL,
        base=population,
        structural=StructuralDelta.FILTER,
        inputs=slices,
        params={"protocol": PROTOCOL, "mode": "all_rows_supported_shape"},
        artifact_inputs=_ordering(ordering),
        description="Open a separate all-row completion version after the actual property parent; retain unknowns, geography and all household members.",
    )


def completion_receiving_node(
    frame: Frame,
    *,
    population: str,
    ordering: tuple[ArtifactInput, ...] = (),
) -> Node:
    """Declare FILTER-all from the actual complete property receiving Frame.

    Reuse the tax receiving profile's component/leaf and shape refusals. Read
    every nonstructural column, including source identifiers ending in ``_id``;
    entity IDs and memberships are supplied by the executor's structural view.
    The host must rederive this declaration from its retained actual parent.
    """
    tax._slices(frame)
    slices = []
    for entity in frame.entities:
        structural = {frame.schema.entity_id_column(entity)}
        if entity == frame.schema.person_entity:
            structural.update(
                frame.schema.membership_column(group)
                for group in frame.schema.group_entities
            )
        slices.append(
            Slice(entity, tuple(c for c in frame.table(entity) if c not in structural))
        )
    return _node(population, tuple(slices), ordering)


class CompletionReceivingKernel(KernelBase):
    """Keep all admitted rows; the host independently verifies the full result."""

    ref = KERNEL
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.FILTER,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            tax,
            tax.cps_carried,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        expected = _node(node.base, node.inputs, node.artifact_inputs)
        _require(
            node.normative() == expected.normative()
            and dict(context.params) == dict(expected.params)
            and not context.sources,
            "DECLARATION",
        )
        _require(
            set(context.artifacts) == {e.name for e in expected.artifact_inputs},
            "ARTIFACT_ROSTER",
        )
        for edge in expected.artifact_inputs:
            value = context.artifacts[edge.name]
            _require(
                type(value) is ArtifactValue
                and value.type == edge.type
                and value.key == opaque_artifact_key(value.producer_key, edge.artifact),
                "ARTIFACT_BINDING",
            )
        _require(
            set(context.tables) == {s.entity for s in expected.inputs},
            "CONTEXT_ENTITIES",
        )
        person = context.tables["person"]
        ids = tax._person(person)
        _require(len(ids) > 0, "EMPTY_RECEIVING_FRAME")
        for entity, table in context.tables.items():
            if entity == "person":
                continue
            _require(
                type(table.index) is pd.RangeIndex
                and table.index.equals(pd.RangeIndex(len(table)))
                and table.index.name is None,
                "UNSUPPORTED_GROUP_INDEX:" + entity,
            )
            _require(
                set(table[entity + "_id"]) == set(person["person_" + entity + "_id"]),
                "ORPHAN_GROUP:" + entity,
            )
        return KernelResult(
            keep=pd.Series(True, index=ids),
            receipt={
                "protocol": PROTOCOL,
                "persons": len(ids),
                "all_rows_retained": True,
                "population_ledger_transition": "filter_conserve",
                "source_authority": False,
            },
        )
