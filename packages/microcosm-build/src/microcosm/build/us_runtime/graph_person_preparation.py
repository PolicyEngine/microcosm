"""Pre-clone relationship and eligibility preparation on declared person cells.

Both families are deterministic ASEC mappings that the pool runs before the
clone stage: relationship after the prior-year join, eligibility after housing.
Each declares a structural source boundary and a derivation that owns exactly
its person input leaves. Neither wrapper imposes a post-derivation scientific
signal requirement, so neither adapter declares a gate node; the boundary node
attests the source boundary and nothing more. Role stripping, CPS selection,
canonical weight selection and the full-pool ID merge stay inside the
registered ``multispine_pool`` owner.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

from . import eligibility_inputs, relationship_inputs
from .graph_context import US_FRAME_CONTEXT_TYPE, _decode, _mass_records, _row_identity
from .multispine_pool import (
    multispine_eligibility_boundary_columns,
    multispine_eligibility_implementation_contract,
    multispine_relationship_boundary_columns,
    multispine_relationship_implementation_contract,
    observe_multispine_eligibility_person,
    observe_multispine_relationship_person,
    validate_multispine_eligibility_person_boundary,
    validate_multispine_relationship_person_boundary,
)

RELATIONSHIP_NODE = "early_preparation.relationship"
ELIGIBILITY_NODE = "early_preparation.eligibility"
PERSON_PREPARATION_OUTER_PHASE = "prepare_multispine_source_inputs_for_clone"
PERSON_PREPARATION_DEPENDENCIES = ("numpy", "pandas", "pyarrow")

#: Whole-module closure of the route both families actually execute, plus the
#: shared summary-policy validator both input modules import. Both kernels live
#: in this module, so an edit to either family already invalidates the other;
#: binding both input modules to both scopes states that conservatism instead
#: of implying a precision the shared file does not have.
_PERSON_PREPARATION_IMPLEMENTATION_MODULES = (
    "microcosm.build.us_runtime.graph_person_preparation",
    "microcosm.build.us_runtime.relationship_inputs",
    "microcosm.build.us_runtime.eligibility_inputs",
    "microcosm.build.us_runtime._person_signal_summary",
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

_BOOLEAN_DTYPE = "boolean"
_NUMERIC_DTYPE = "float64"


@dataclass(frozen=True)
class _Family:
    """One deterministic person-preparation family's declared surface."""

    name: str
    label: str
    node: str
    required_source_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    boolean_outputs: frozenset[str]
    boundary_columns: Callable[[], tuple[str, ...]]
    validate_boundary: Callable[..., dict[str, object]]
    observe: Callable[..., tuple[pd.DataFrame, dict[str, object]]]
    contract: Callable[[], dict[str, object]]

    def reads(self) -> frozenset[str]:
        return frozenset(self.required_source_columns) | frozenset(self.output_columns)

    def outputs(self) -> dict[str, frozenset[str]]:
        return {"person": frozenset(self.output_columns)}

    def dtype_for(self, column: str) -> str | None:
        """The declared dtype a nullable boolean output must carry, if any."""
        return _BOOLEAN_DTYPE if column in self.boolean_outputs else None


RELATIONSHIP = _Family(
    name="relationship",
    label="Relationship",
    node=RELATIONSHIP_NODE,
    required_source_columns=(
        relationship_inputs.US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS
    ),
    output_columns=relationship_inputs.US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS,
    boolean_outputs=frozenset(
        relationship_inputs.US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
    ),
    boundary_columns=multispine_relationship_boundary_columns,
    validate_boundary=validate_multispine_relationship_person_boundary,
    observe=observe_multispine_relationship_person,
    contract=multispine_relationship_implementation_contract,
)

ELIGIBILITY = _Family(
    name="eligibility",
    label="Eligibility",
    node=ELIGIBILITY_NODE,
    required_source_columns=(
        eligibility_inputs.US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS
    ),
    output_columns=eligibility_inputs.US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS,
    # ``own_children_in_household`` and ``veterans_benefits`` are the measured
    # float64 leaves the helper writes; the other three are physical booleans
    # the full-pool merge materializes as nullable booleans.
    boolean_outputs=frozenset(
        {"is_disabled", "is_blind", "is_full_time_college_student"}
    ),
    boundary_columns=multispine_eligibility_boundary_columns,
    validate_boundary=validate_multispine_eligibility_person_boundary,
    observe=observe_multispine_eligibility_person,
    contract=multispine_eligibility_implementation_contract,
)


class _PersonPreparationKernel(KernelBase):
    family: _Family

    def implementation_hash(self) -> str:
        modules = source_hash(
            *(
                import_module(name)
                for name in _PERSON_PREPARATION_IMPLEMENTATION_MODULES
            ),
            dependencies=PERSON_PREPARATION_DEPENDENCIES,
        )
        resource = files("microcosm.build.us").joinpath("source_stages.json")
        return hashlib.sha256(
            canonical_json(
                {
                    "schema": (
                        f"microcosm.us.{self.family.name}-preparation-implementation.v1"
                    ),
                    "modules": modules,
                    "operator_contract": self.family.contract(),
                    "reads": {"person": sorted(self.family.reads())},
                    "outputs": {
                        entity: sorted(columns)
                        for entity, columns in self.family.outputs().items()
                    },
                    "boolean_outputs": sorted(self.family.boolean_outputs),
                    "resources": {
                        "microcosm.build.us/source_stages.json": hashlib.sha256(
                            resource.read_bytes()
                        ).hexdigest()
                    },
                }
            )
        ).hexdigest()


def _context(
    context: KernelContext, family: _Family
) -> tuple[pd.DataFrame, dict[str, object]]:
    if context.params != {"phase": PERSON_PREPARATION_OUTER_PHASE}:
        raise ValueError(
            f"{family.label} preparation requires its authored outer phase."
        )
    artifact = context.artifacts["frame_context"]
    if artifact.type != US_FRAME_CONTEXT_TYPE:
        raise ValueError(
            f"{family.label} preparation requires the full typed frame context."
        )
    document = _decode(artifact.payload)
    _mass_records(document["mass_log"])
    if set(context.tables) != {"person"}:
        raise ValueError(
            f"{family.label} preparation requires only its declared person view."
        )
    person = context.tables["person"]
    declared = document["entities"]["person"]
    if any(
        declared[key] != value for key, value in _row_identity(person, "person").items()
    ):
        raise ValueError(f"{family.label} person identity differs from its context.")
    if set(person.columns) - set(declared["columns"]):
        raise ValueError(f"{family.label} person view includes unknown columns.")
    for group in US_SCHEMA.group_entities:
        if document["entities"][group]["rows"] != int(
            person[US_SCHEMA.membership_column(group)].nunique()
        ):
            raise ValueError(f"{family.label} context counts differ from memberships.")
    sources = document["weight_sources"]
    if len(sources) != 1:
        raise ValueError(
            f"{family.label} preparation requires one explicit weight source."
        )
    if context.weights["person"].kind.value != next(iter(sources.values())):
        raise ValueError(f"{family.label} canonical weight kind differs from context.")
    ordered = [column for column in declared["columns"] if column in person]
    return person.loc[:, ordered], document


def _boundary_result(context: KernelContext, family: _Family) -> KernelResult:
    person, document = _context(context, family)
    receipt = family.validate_boundary(person, metadata=document["metadata"])
    return KernelResult(
        keep=pd.Series(
            True, index=pd.Index(person.person_id, name="person_id"), dtype=bool
        ),
        artifacts={"frame_context": context.artifacts["frame_context"].payload},
        receipt={"source_boundary": receipt},
    )


def _derive_result(context: KernelContext, family: _Family) -> KernelResult:
    person, document = _context(context, family)
    required = family.reads() & set(document["entities"]["person"]["columns"])
    if required - set(person.columns):
        raise ValueError(
            f"{family.label} declaration omits available person raw or "
            "incumbent inputs."
        )
    outputs = family.outputs()
    if {(value.entity, value.column) for value in context.node.outputs} != {
        ("person", name) for name in family.output_columns
    }:
        raise ValueError(
            f"{family.label} preparation must own exactly its "
            f"{len(family.output_columns)} pre-clone person outputs."
        )
    produced, receipt = family.observe(
        person,
        weights=context.weights["person"],
        metadata=document["metadata"],
        pool_rows={
            entity: value["rows"] for entity, value in document["entities"].items()
        },
    )
    columns = {}
    for owned in context.node.outputs:
        values = produced[owned.column]
        if str(values.dtype) != owned.dtype:
            raise ValueError(
                f"{family.label} output dtype differs from declaration: "
                f"{owned.entity}.{owned.column}."
            )
        columns[("person", owned.column)] = pd.Series(
            values.array, index=pd.Index(produced.person_id, name="person_id")
        )
    # Person identity was verified against the full context and the registered
    # source boundary verified the selected structure and ID merge. Extend only
    # the person column inventory; every other descriptor stays untouched.
    declared = document["entities"]["person"]["columns"]
    for name in produced.columns:
        if name in outputs["person"] and name not in declared:
            declared.append(name)
    return KernelResult(
        columns=columns,
        artifacts={"frame_context": canonical_json(document)},
        receipt={"source_preparation": receipt},
    )


class USRelationshipBoundaryKernel(_PersonPreparationKernel):
    ref = "us.production.relationship_boundary@1"
    family = RELATIONSHIP
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=PERSON_PREPARATION_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        return _boundary_result(context, self.family)


class USRelationshipSourceKernel(_PersonPreparationKernel):
    ref = "us.production.relationship@1"
    family = RELATIONSHIP
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=PERSON_PREPARATION_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        return _derive_result(context, self.family)


class USEligibilityBoundaryKernel(_PersonPreparationKernel):
    ref = "us.production.eligibility_boundary@1"
    family = ELIGIBILITY
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=PERSON_PREPARATION_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        return _boundary_result(context, self.family)


class USEligibilitySourceKernel(_PersonPreparationKernel):
    ref = "us.production.eligibility@1"
    family = ELIGIBILITY
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=PERSON_PREPARATION_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        return _derive_result(context, self.family)


def _person_preparation_nodes(
    family: _Family,
    boundary_kernel: type[_PersonPreparationKernel],
    derive_kernel: type[_PersonPreparationKernel],
    columns: Sequence[Owned],
    *,
    base: str,
    context_producer: str,
) -> tuple[Node, ...]:
    inventory = {(item.entity, item.column): item for item in columns}
    if len(inventory) != len(columns):
        raise ValueError(f"{family.label} declarations repeat coordinates.")
    boundary_columns = family.boundary_columns()
    if any(("person", name) not in inventory for name in boundary_columns):
        raise ValueError(f"{family.label} source-boundary evidence is absent.")
    outputs = []
    for name in sorted(family.output_columns):
        incumbent = inventory.get(("person", name))
        nullable = family.dtype_for(name)
        if nullable is not None:
            if incumbent is not None and incumbent.dtype != nullable:
                raise ValueError(
                    f"{family.label} boolean incumbent requires nullable "
                    f"boolean: person.{name}."
                )
            dtype = nullable
        else:
            dtype = incumbent.dtype if incumbent is not None else _NUMERIC_DTYPE
        outputs.append(Owned("person", name, dtype, rewrite=incumbent is not None))
    actual = {name for name in family.reads() if ("person", name) in inventory}
    actual.update(boundary_columns)
    context_output = (ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),)
    boundary = f"{family.node}.boundary"
    return (
        Node(
            boundary,
            boundary_kernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(Slice("person", boundary_columns),),
            params={"phase": PERSON_PREPARATION_OUTER_PHASE},
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
            family.node,
            derive_kernel.ref,
            population=boundary,
            inputs=(Slice("person", tuple(sorted(actual))),),
            outputs=tuple(outputs),
            params={"phase": PERSON_PREPARATION_OUTER_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context", boundary, "frame_context", US_FRAME_CONTEXT_TYPE
                ),
            ),
            artifact_outputs=context_output,
        ),
    )


def us_relationship_source_nodes(
    columns: Sequence[Owned], *, base: str, context_producer: str
) -> tuple[Node, ...]:
    """Declare relationship preparation after the accepted prior-year context.

    ``base``/``context_producer`` stay explicit so the caller stitches the
    reviewed pre-clone order — CPS-carried, accepted hours, prior year, this
    node, housing, eligibility — without this module naming its neighbours.
    """
    return _person_preparation_nodes(
        RELATIONSHIP,
        USRelationshipBoundaryKernel,
        USRelationshipSourceKernel,
        columns,
        base=base,
        context_producer=context_producer,
    )


def us_eligibility_source_nodes(
    columns: Sequence[Owned], *, base: str, context_producer: str
) -> tuple[Node, ...]:
    """Declare eligibility preparation after the housing producer's context.

    The housing adapter is authored separately; binding it is the caller's
    step. Nothing here asserts that a housing node ran.
    """
    return _person_preparation_nodes(
        ELIGIBILITY,
        USEligibilityBoundaryKernel,
        USEligibilitySourceKernel,
        columns,
        base=base,
        context_producer=context_producer,
    )


def register_us_person_preparation_kernels(registry: KernelRegistry) -> None:
    for kernel in (
        USRelationshipBoundaryKernel(),
        USRelationshipSourceKernel(),
        USEligibilityBoundaryKernel(),
        USEligibilitySourceKernel(),
    ):
        registry.register(kernel)
