"""CPS measured facts use declared person/SPM views and the real source boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib import import_module
from importlib.resources import files

import pandas as pd

from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.kernel import source_hash

from . import cps_carried
from .graph_context import US_FRAME_CONTEXT_TYPE, _decode, _mass_records, _row_identity
from .multispine_pool import (
    multispine_cps_carried_boundary_columns,
    multispine_cps_carried_implementation_contract,
    observe_multispine_cps_carried_tables,
    validate_multispine_cps_carried_person_boundary,
)

CPS_CARRIED_NODE = "early_preparation.cps_carried"
CPS_CARRIED_OUTER_PHASE = "prepare_multispine_source_inputs_for_clone"
CPS_CARRIED_DEPENDENCIES = ("numpy", "pandas", "pyarrow")
_CPS_IMPLEMENTATION_MODULES = (
    "microcosm.build.us_runtime.graph_cps_carried",
    "microcosm.build.us_runtime.cps_carried",
    "microcosm.build.us_runtime.alimony",
    "microcosm.build.us_runtime.public_assistance_type_source",
    "microcosm.build.us_runtime.reported_coverage_source",
    "microcosm.build.us_runtime.graph_context",
    "microcosm.build.us_runtime.multispine_pool",
    "microcosm.build.us_runtime.operator_boundary",
    "microcosm.build.us_runtime.support_provenance",
    "microcosm.build.source_manifest",
    "microcosm.build.source_runtime",
    "microcosm.build.gates",
    "microcosm.frame",
    "microcosm.frame.accounting",
    "microcosm.frame.bundle",
    "microcosm.frame.kernels",
    "microcosm.frame.materialize",
    "microcosm.frame.rules",
    "microcosm.frame.schema",
    "microcosm.frame.units",
    "microcosm.frame.weights",
    "microcosm.graph",
    "microcosm.graph.artifact_edges",
    "microcosm.graph.attachments",
    "microcosm.graph.availability",
    "microcosm.graph.canonical",
    "microcosm.graph.codecs",
    "microcosm.graph.decl",
    "microcosm.graph.errors",
    "microcosm.graph.executor",
    "microcosm.graph.explain",
    "microcosm.graph.kernel",
    "microcosm.graph.keys",
    "microcosm.graph.manifest",
    "microcosm.graph.population",
    "microcosm.graph.randomness",
    "microcosm.graph.serialize",
    "microcosm.graph.store",
    "microcosm.graph.view",
)


def _reads() -> dict[str, frozenset[str]]:
    return {
        "person": cps_carried.CPS_CARRIED_RAW_PERSON_COLUMNS
        | cps_carried.CPS_CARRIED_PERSON_INPUTS
        | cps_carried.CPS_CARRIED_FORMULA_OWNED_COLUMNS,
        "spm_unit": cps_carried.CPS_CARRIED_SPM_UNIT_INPUTS,
    }


def _outputs() -> dict[str, frozenset[str]]:
    return {
        "person": cps_carried.CPS_CARRIED_PERSON_INPUTS,
        "spm_unit": cps_carried.CPS_CARRIED_SPM_UNIT_INPUTS,
    }


class _CpsKernel(KernelBase):
    def implementation_hash(self) -> str:
        modules = source_hash(
            *(import_module(name) for name in _CPS_IMPLEMENTATION_MODULES),
            dependencies=CPS_CARRIED_DEPENDENCIES,
        )
        resource = files("microcosm.build.us").joinpath("source_stages.json")
        return hashlib.sha256(
            canonical_json(
                {
                    "schema": "microcosm.us.cps-carried-implementation.v1",
                    "modules": modules,
                    "operator_contract": multispine_cps_carried_implementation_contract(),
                    "reads": {
                        entity: sorted(columns) for entity, columns in _reads().items()
                    },
                    "outputs": {
                        entity: sorted(columns)
                        for entity, columns in _outputs().items()
                    },
                    "boolean_inputs": sorted(cps_carried.CPS_CARRIED_BOOLEAN_INPUTS),
                    "resources": {
                        "microcosm.build.us/source_stages.json": hashlib.sha256(
                            resource.read_bytes()
                        ).hexdigest()
                    },
                }
            )
        ).hexdigest()


def _context(
    context: KernelContext, *, entities: set[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if context.params != {"phase": CPS_CARRIED_OUTER_PHASE}:
        raise ValueError("CPS-carried graph requires its authored outer phase.")
    artifact = context.artifacts["frame_context"]
    if artifact.type != US_FRAME_CONTEXT_TYPE:
        raise ValueError("CPS-carried requires the full typed frame context.")
    document = _decode(artifact.payload)
    _mass_records(document["mass_log"])
    if set(context.tables) != entities:
        raise ValueError("CPS-carried requires its actual declared entity views.")
    tables = {}
    for entity in entities:
        table = context.tables[entity]
        declared = document["entities"][entity]
        if any(
            declared[key] != value
            for key, value in _row_identity(table, entity).items()
        ):
            raise ValueError(f"CPS-carried {entity} identity differs from its context.")
        if set(table.columns) - set(declared["columns"]):
            raise ValueError("CPS-carried view includes unknown columns.")
        tables[entity] = table.loc[
            :, [name for name in declared["columns"] if name in table]
        ]
    person = tables["person"]
    for group in US_SCHEMA.group_entities:
        if document["entities"][group]["rows"] != int(
            person[US_SCHEMA.membership_column(group)].nunique()
        ):
            raise ValueError("CPS-carried context counts differ from memberships.")
    sources = document["weight_sources"]
    if len(sources) != 1:
        raise ValueError("CPS-carried graph requires one explicit weight source.")
    if context.weights["person"].kind.value != next(iter(sources.values())):
        raise ValueError("CPS-carried canonical weight kind differs from context.")
    return tables, document


class USCpsCarriedBoundaryKernel(_CpsKernel):
    ref = "us.production.cps_carried_boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=CPS_CARRIED_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        tables, document = _context(context, entities={"person"})
        person = tables["person"]
        receipt = validate_multispine_cps_carried_person_boundary(
            person, metadata=document["metadata"]
        )
        return KernelResult(
            keep=pd.Series(
                True, index=pd.Index(person.person_id, name="person_id"), dtype=bool
            ),
            artifacts={"frame_context": context.artifacts["frame_context"].payload},
            receipt={"source_boundary": receipt},
        )


class USCpsCarriedKernel(_CpsKernel):
    ref = "us.production.cps_carried@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=CPS_CARRIED_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        tables, document = _context(context, entities={"person", "spm_unit"})
        for entity, candidates in _reads().items():
            required = candidates & set(document["entities"][entity]["columns"])
            if required - set(tables[entity].columns):
                raise ValueError(
                    f"CPS-carried declaration omits available {entity} raw or incumbent inputs."
                )
        outputs = _outputs()
        if {(value.entity, value.column) for value in context.node.outputs} != {
            (entity, name) for entity, names in outputs.items() for name in names
        }:
            raise ValueError(
                "CPS-carried must own exactly its actual person and SPM outputs."
            )
        produced, receipt = observe_multispine_cps_carried_tables(
            tables["person"],
            tables["spm_unit"],
            weights=context.weights["person"],
            metadata=document["metadata"],
            pool_rows={
                entity: value["rows"] for entity, value in document["entities"].items()
            },
        )
        result_tables = {"person": produced.person, "spm_unit": produced.spm_unit}
        columns = {}
        for owned in context.node.outputs:
            table = result_tables[owned.entity]
            values = table[owned.column]
            if str(values.dtype) != owned.dtype:
                raise ValueError(
                    f"CPS-carried output dtype differs from declaration: {owned.entity}.{owned.column}."
                )
            columns[(owned.entity, owned.column)] = pd.Series(
                values.array,
                index=pd.Index(
                    table[US_SCHEMA.entity_id_column(owned.entity)],
                    name=US_SCHEMA.entity_id_column(owned.entity),
                ),
            )
        for entity, table in result_tables.items():
            declared = document["entities"][entity]["columns"]
            for name in table:
                if name in outputs[entity] and name not in declared:
                    declared.append(name)
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": canonical_json(document)},
            receipt={"source_preparation": receipt},
        )


def us_cps_carried_nodes(
    columns: Sequence[Owned], *, base: str, context_producer: str
) -> tuple[Node, ...]:
    """Declare the measured source operation; downstream consumes its context."""
    inventory = {(item.entity, item.column): item for item in columns}
    if len(inventory) != len(columns):
        raise ValueError("CPS-carried declarations repeat coordinates.")
    boundary_columns = multispine_cps_carried_boundary_columns()
    if any(("person", name) not in inventory for name in boundary_columns):
        raise ValueError("CPS-carried source-boundary evidence is absent.")
    outputs = []
    for entity, names in _outputs().items():
        for name in sorted(names):
            incumbent = inventory.get((entity, name))
            dtype = incumbent.dtype if incumbent is not None else "float64"
            if name in cps_carried.CPS_CARRIED_BOOLEAN_INPUTS:
                if incumbent is not None and incumbent.dtype != "boolean":
                    raise ValueError(
                        f"CPS-carried boolean incumbent requires nullable boolean: {entity}.{name}."
                    )
                dtype = "boolean"
            outputs.append(Owned(entity, name, dtype, rewrite=incumbent is not None))
    inputs = []
    for entity, candidates in _reads().items():
        actual = {name for name in candidates if (entity, name) in inventory}
        if entity == "person":
            actual.update(boundary_columns)
        if actual:
            inputs.append(Slice(entity, tuple(sorted(actual))))
    context_output = (ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),)
    boundary = f"{CPS_CARRIED_NODE}.boundary"
    return (
        Node(
            boundary,
            USCpsCarriedBoundaryKernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(Slice("person", boundary_columns),),
            params={"phase": CPS_CARRIED_OUTER_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context",
                    context_producer,
                    "frame_context",
                    US_FRAME_CONTEXT_TYPE,
                ),
            ),
            artifact_outputs=context_output,
        ),
        Node(
            CPS_CARRIED_NODE,
            USCpsCarriedKernel.ref,
            population=boundary,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            params={"phase": CPS_CARRIED_OUTER_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context", boundary, "frame_context", US_FRAME_CONTEXT_TYPE
                ),
            ),
            artifact_outputs=context_output,
        ),
    )


def register_us_cps_carried_kernels(registry: KernelRegistry) -> None:
    registry.register(USCpsCarriedBoundaryKernel())
    registry.register(USCpsCarriedKernel())
